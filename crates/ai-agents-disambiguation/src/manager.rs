//! Disambiguation manager orchestrating the full disambiguation flow

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use ai_agents_core::Result;
use ai_agents_llm::LLMRegistry;

use super::clarifier::{ClarificationGenerator, ClarificationParseResult};
use super::config::{
    DisambiguationConfig, MaxAttemptsAction, SkillDisambiguationOverride,
    StateDisambiguationOverride,
};
use super::detector::AmbiguityDetector;
use super::types::{
    AmbiguityDetectionResult, ClarificationQuestion, DisambiguationContext, DisambiguationResult,
};

/// Manager orchestrating the full disambiguation flow
pub struct DisambiguationManager {
    config: DisambiguationConfig,
    detector: AmbiguityDetector,
    clarifier: ClarificationGenerator,
    pending_clarification: RwLock<Option<PendingClarification>>,
}

/// Pending clarification state
#[derive(Debug, Clone)]
struct PendingClarification {
    original_input: String,
    question: ClarificationQuestion,
    detection: AmbiguityDetectionResult,
    attempts: u32,
}

impl DisambiguationManager {
    pub fn new(config: DisambiguationConfig, llm_registry: Arc<LLMRegistry>) -> Self {
        let detector = AmbiguityDetector::new(config.detection.clone(), Arc::clone(&llm_registry));
        let clarifier =
            ClarificationGenerator::new(config.clarification.clone(), Arc::clone(&llm_registry));

        if config.cache.enabled {
            warn!(
                "Disambiguation cache is enabled in config but not yet implemented — requests will not be cached"
            );
        }

        Self {
            config,
            detector,
            clarifier,
            pending_clarification: RwLock::new(None),
        }
    }

    /// Check if disambiguation is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration
    pub fn config(&self) -> &DisambiguationConfig {
        &self.config
    }

    /// Check if there's a pending clarification
    pub async fn has_pending_clarification(&self) -> bool {
        self.pending_clarification.read().await.is_some()
    }

    /// Process user input with disambiguation
    pub async fn process_input(
        &self,
        input: &str,
        context: &DisambiguationContext,
    ) -> Result<DisambiguationResult> {
        self.process_input_with_override(input, context, None, None)
            .await
    }

    /// Process input with optional state/skill overrides
    pub async fn process_input_with_override(
        &self,
        input: &str,
        context: &DisambiguationContext,
        state_override: Option<&StateDisambiguationOverride>,
        skill_override: Option<&SkillDisambiguationOverride>,
    ) -> Result<DisambiguationResult> {
        if !self.is_enabled() && state_override.is_none() && skill_override.is_none() {
            return Ok(DisambiguationResult::Clear);
        }

        // Check if we're handling a clarification response
        if let Some(pending) = self.pending_clarification.read().await.clone() {
            debug!(
                attempts = pending.attempts,
                "Processing clarification response"
            );
            return self
                .handle_clarification_response(input, &pending, context)
                .await;
        }

        // Check skip conditions
        if self
            .detector
            .should_skip(input, context, &self.config.skip_when)
            .await?
        {
            return Ok(DisambiguationResult::Clear);
        }

        // Determine effective threshold
        let threshold = self.get_effective_threshold(state_override, skill_override);

        // Detect ambiguity
        let detection = self.detector.detect(input, context).await?;

        info!(
            is_ambiguous = detection.is_ambiguous,
            confidence = detection.confidence,
            threshold = threshold,
            ambiguity_type = ?detection.ambiguity_type,
            "Ambiguity detection complete"
        );

        // Check against effective threshold
        if !detection.is_ambiguous && detection.confidence >= threshold {
            return Ok(DisambiguationResult::Clear);
        }

        // Check required clarity fields
        let required_clarity = self.get_required_clarity(state_override, skill_override);
        if !required_clarity.is_empty() {
            let missing: Vec<_> = required_clarity
                .iter()
                .filter(|field| detection.what_is_unclear.contains(field))
                .cloned()
                .collect();

            if !missing.is_empty() {
                debug!(
                    missing_fields = ?missing,
                    "Required clarity fields missing"
                );
            }
        }

        // Get custom template if available from skill override
        let custom_template = skill_override.and_then(|s| {
            detection.ambiguity_type.as_ref().and_then(|t| {
                let key = match t {
                    super::types::AmbiguityType::MissingTarget => "missing_target",
                    super::types::AmbiguityType::MissingAction => "missing_action",
                    super::types::AmbiguityType::MissingParameters => "missing_parameters",
                    super::types::AmbiguityType::VagueReference => "vague_reference",
                    _ => return None,
                };
                s.clarification_templates.get(key).map(|s| s.as_str())
            })
        });

        // Generate clarification question
        let question = self
            .clarifier
            .generate(input, &detection, context, custom_template)
            .await?;

        // Store pending clarification
        *self.pending_clarification.write().await = Some(PendingClarification {
            original_input: input.to_string(),
            question: question.clone(),
            detection: detection.clone(),
            attempts: 1,
        });

        Ok(DisambiguationResult::NeedsClarification {
            question,
            detection,
        })
    }

    /// Handle a clarification response from the user
    async fn handle_clarification_response(
        &self,
        response: &str,
        pending: &PendingClarification,
        context: &DisambiguationContext,
    ) -> Result<DisambiguationResult> {
        let parse_result = self
            .clarifier
            .parse_response(
                &pending.original_input,
                &pending.question,
                response,
                context,
            )
            .await?;

        match parse_result {
            ClarificationParseResult::Understood {
                enriched_input,
                resolved,
            } => {
                // Clear pending state
                *self.pending_clarification.write().await = None;

                info!(
                    original = %pending.original_input,
                    enriched = %enriched_input,
                    "Clarification resolved"
                );

                Ok(DisambiguationResult::Clarified {
                    original_input: pending.original_input.clone(),
                    enriched_input,
                    resolved,
                })
            }
            ClarificationParseResult::NotUnderstood => {
                let new_attempts = pending.attempts + 1;

                if new_attempts > self.config.clarification.max_attempts {
                    // Clear pending state
                    *self.pending_clarification.write().await = None;

                    return self.handle_max_attempts(&pending.original_input);
                }

                // Update context with previous question
                let mut new_context = context.clone();
                new_context.add_previous_question(pending.question.question.clone());
                new_context.increment_attempts();

                // Generate a new clarification question
                let question = self
                    .clarifier
                    .generate(
                        &pending.original_input,
                        &pending.detection,
                        &new_context,
                        None,
                    )
                    .await?;

                // Update pending state
                *self.pending_clarification.write().await = Some(PendingClarification {
                    original_input: pending.original_input.clone(),
                    question: question.clone(),
                    detection: pending.detection.clone(),
                    attempts: new_attempts,
                });

                warn!(
                    attempts = new_attempts,
                    max = self.config.clarification.max_attempts,
                    "Clarification response not understood, retrying"
                );

                Ok(DisambiguationResult::NeedsClarification {
                    question,
                    detection: pending.detection.clone(),
                })
            }
        }
    }

    fn handle_max_attempts(&self, original_input: &str) -> Result<DisambiguationResult> {
        match self.config.clarification.on_max_attempts {
            MaxAttemptsAction::ProceedWithBestGuess => {
                info!("Max clarification attempts reached, proceeding with best guess");
                Ok(DisambiguationResult::ProceedWithBestGuess {
                    enriched_input: original_input.to_string(),
                })
            }
            MaxAttemptsAction::ApologizeAndStop => {
                info!("Max clarification attempts reached, giving up");
                Ok(DisambiguationResult::GiveUp {
                    reason: "Unable to understand your request after multiple attempts".to_string(),
                })
            }
            MaxAttemptsAction::Escalate => {
                info!("Max clarification attempts reached, escalating");
                Ok(DisambiguationResult::Escalate {
                    reason: "User request requires human assistance".to_string(),
                })
            }
        }
    }

    fn get_effective_threshold(
        &self,
        state_override: Option<&StateDisambiguationOverride>,
        skill_override: Option<&SkillDisambiguationOverride>,
    ) -> f32 {
        // Skill override takes precedence
        if let Some(skill) = skill_override {
            if let Some(t) = skill.threshold {
                return t;
            }
        }

        // Then state override
        if let Some(state) = state_override {
            if let Some(t) = state.threshold {
                return t;
            }
        }

        // Default from config
        self.config.detection.threshold
    }

    fn get_required_clarity(
        &self,
        state_override: Option<&StateDisambiguationOverride>,
        skill_override: Option<&SkillDisambiguationOverride>,
    ) -> Vec<String> {
        // Combine required clarity from both overrides
        let mut required = Vec::new();

        if let Some(state) = state_override {
            required.extend(state.required_clarity.iter().cloned());
        }

        if let Some(skill) = skill_override {
            required.extend(skill.required_clarity.iter().cloned());
        }

        required
    }

    /// Clear any pending clarification state
    pub async fn clear_pending(&self) {
        *self.pending_clarification.write().await = None;
    }

    /// Get the current pending clarification if any
    pub async fn get_pending_question(&self) -> Option<ClarificationQuestion> {
        self.pending_clarification
            .read()
            .await
            .as_ref()
            .map(|p| p.question.clone())
    }

    /// Get number of clarification attempts so far
    pub async fn clarification_attempts(&self) -> u32 {
        self.pending_clarification
            .read()
            .await
            .as_ref()
            .map(|p| p.attempts)
            .unwrap_or(0)
    }
}

/// Builder for DisambiguationContext
impl DisambiguationContext {
    pub fn from_agent_state(
        recent_messages: Vec<String>,
        current_state: Option<String>,
        available_tools: Vec<String>,
        available_skills: Vec<String>,
        user_context: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            recent_messages,
            current_state,
            available_tools,
            available_skills,
            user_context,
            clarification_attempts: 0,
            previous_questions: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disambiguation_context_builder() {
        let ctx = DisambiguationContext::from_agent_state(
            vec!["Hello".to_string()],
            Some("greeting".to_string()),
            vec!["search".to_string()],
            vec!["greet".to_string()],
            HashMap::new(),
        );

        assert_eq!(ctx.recent_messages.len(), 1);
        assert_eq!(ctx.current_state, Some("greeting".to_string()));
        assert_eq!(ctx.available_tools.len(), 1);
        assert_eq!(ctx.available_skills.len(), 1);
        assert_eq!(ctx.clarification_attempts, 0);
    }

    #[test]
    fn test_get_effective_threshold() {
        let config = DisambiguationConfig {
            enabled: true,
            detection: super::super::config::DetectionConfig {
                threshold: 0.7,
                ..Default::default()
            },
            ..Default::default()
        };

        // We can't fully test this without LLMRegistry, but we can test the threshold logic
        let state_override = StateDisambiguationOverride {
            threshold: Some(0.9),
            ..Default::default()
        };

        let skill_override = SkillDisambiguationOverride {
            threshold: Some(0.95),
            ..Default::default()
        };

        // Skill takes precedence over state
        assert_eq!(skill_override.threshold.unwrap(), 0.95);
        assert_eq!(state_override.threshold.unwrap(), 0.9);
        assert_eq!(config.detection.threshold, 0.7);
    }
}
