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
    required_clarity: Vec<String>,
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

        // Check if we're handling a clarification response.
        // ISSUE: Solved
        // Clone and drop the read guard before entering the async block to avoid deadlocking with the write lock inside handle_clarification_response.
        let pending = self.pending_clarification.read().await.clone();
        if let Some(pending) = pending {
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

        // Populate required_clarity on context so the detector prompt knows
        // which domain-required fields to check for in the user's message.
        let mut context = context.clone();
        let rc = self.get_required_clarity(state_override, skill_override);
        if !rc.is_empty() {
            context.required_clarity = rc;
        }

        // Detect ambiguity
        let detection = self.detector.detect(input, &context).await?;

        info!(
            is_ambiguous = detection.is_ambiguous,
            confidence = detection.confidence,
            threshold = threshold,
            ambiguity_type = ?detection.ambiguity_type,
            "Ambiguity detection complete"
        );

        // Check required clarity fields BEFORE the threshold check.
        // required_clarity is a hard gate: if any required field appears in what_is_unclear, force clarification regardless of confidence score.
        // This handles domain ambiguity ("transfer money" is linguistically clear but missing recipient/amount for the operation).
        let required_clarity = self.get_required_clarity(state_override, skill_override);
        if !required_clarity.is_empty() {
            let missing: Vec<_> = required_clarity
                .iter()
                .filter(|field| detection.what_is_unclear.contains(field))
                .cloned()
                .collect();

            if !missing.is_empty() {
                info!(
                    missing_fields = ?missing,
                    confidence = detection.confidence,
                    "Required clarity fields missing — forcing clarification"
                );
                // Override the detection to ensure we proceed to clarification
                // even though confidence might be above threshold.
                let mut forced_detection = detection.clone();
                forced_detection.is_ambiguous = true;
                if forced_detection.ambiguity_type.is_none() {
                    forced_detection.ambiguity_type =
                        Some(super::types::AmbiguityType::MissingParameters);
                }
                // Merge missing required fields into what_is_unclear
                for field in &missing {
                    if !forced_detection.what_is_unclear.contains(field) {
                        forced_detection.what_is_unclear.push(field.clone());
                    }
                }

                // Get custom template if available
                let custom_template = skill_override.and_then(|s| {
                    if s.clarification_templates.is_empty() {
                        return None;
                    }
                    forced_detection.what_is_unclear.iter().find_map(|field| {
                        let prefixed = format!("missing_{}", field);
                        s.clarification_templates
                            .get(&prefixed)
                            .or_else(|| s.clarification_templates.get(field.as_str()))
                            .map(|v| v.as_str())
                    })
                });

                let question = self
                    .clarifier
                    .generate(
                        input,
                        &forced_detection,
                        &context,
                        custom_template,
                        &required_clarity,
                    )
                    .await?;

                *self.pending_clarification.write().await = Some(PendingClarification {
                    original_input: input.to_string(),
                    question: question.clone(),
                    detection: forced_detection.clone(),
                    attempts: 1,
                    required_clarity: required_clarity.clone(),
                });

                return Ok(DisambiguationResult::NeedsClarification {
                    question,
                    detection: forced_detection,
                });
            }
        }

        // Check against effective threshold
        if !detection.is_ambiguous && detection.confidence >= threshold {
            return Ok(DisambiguationResult::Clear);
        }

        // Get custom template if available from skill override.
        // Lookup order:
        //   1. Match by ambiguity_type (missing_target, missing_action, etc.)
        //   2. Match by what_is_unclear fields (e.g. "recipient" -> "missing_recipient" or "recipient")
        //   3. No match -> None -> fall through to LLM generation
        let custom_template = skill_override.and_then(|s| {
            if s.clarification_templates.is_empty() {
                return None;
            }

            // Step 1: try ambiguity type match
            let by_type = detection.ambiguity_type.as_ref().and_then(|t| {
                let key = match t {
                    super::types::AmbiguityType::MissingTarget => "missing_target",
                    super::types::AmbiguityType::MissingAction => "missing_action",
                    super::types::AmbiguityType::MissingParameters => "missing_parameters",
                    super::types::AmbiguityType::VagueReference => "vague_reference",
                    _ => return None,
                };
                s.clarification_templates.get(key).map(|v| v.as_str())
            });

            if by_type.is_some() {
                return by_type;
            }

            // Step 2: try what_is_unclear field match (supports custom keys)
            detection.what_is_unclear.iter().find_map(|field| {
                let prefixed = format!("missing_{}", field);
                s.clarification_templates
                    .get(&prefixed)
                    .or_else(|| s.clarification_templates.get(field.as_str()))
                    .map(|v| v.as_str())
            })
        });

        // Generate clarification question
        let required_clarity = self.get_required_clarity(state_override, skill_override);
        let question = self
            .clarifier
            .generate(
                input,
                &detection,
                &context,
                custom_template,
                &required_clarity,
            )
            .await?;

        // Store pending clarification
        *self.pending_clarification.write().await = Some(PendingClarification {
            original_input: input.to_string(),
            question: question.clone(),
            detection: detection.clone(),
            attempts: 1,
            required_clarity: required_clarity.clone(),
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

                if new_attempts >= self.config.clarification.max_attempts {
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
                        &pending.required_clarity,
                    )
                    .await?;

                // Update pending state
                *self.pending_clarification.write().await = Some(PendingClarification {
                    original_input: pending.original_input.clone(),
                    question: question.clone(),
                    detection: pending.detection.clone(),
                    attempts: new_attempts,
                    required_clarity: pending.required_clarity.clone(),
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
            ClarificationParseResult::Abandoned => {
                *self.pending_clarification.write().await = None;
                info!("User abandoned clarification");
                Ok(DisambiguationResult::Abandoned { new_input: None })
            }
            ClarificationParseResult::TopicSwitch => {
                *self.pending_clarification.write().await = None;
                info!("User switched to a different topic during clarification");
                Ok(DisambiguationResult::Abandoned {
                    new_input: Some(response.to_string()),
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
        state_prompt: Option<String>,
        available_tools: Vec<String>,
        available_skills: Vec<String>,
        available_intents: Vec<String>,
        user_context: HashMap<String, serde_json::Value>,
    ) -> Self {
        // Populate previous_questions from recent assistant messages that end with '?'.
        // This lets the answering_agent_question skip condition work: if the last assistant message was a question, the user's next input is likely an answer, not a new ambiguous request.
        // Only check the most recent assistant message for a trailing '?'.
        // Multiple historical questions are noise - only the last one matters for the answering_agent_question skip decision.
        let previous_questions: Vec<String> = recent_messages
            .iter()
            .rev()
            .find(|m| m.starts_with("Assistant:"))
            .filter(|m| m.trim_end().ends_with('?'))
            .cloned()
            .into_iter()
            .collect();

        Self {
            recent_messages,
            current_state,
            state_prompt,
            available_tools,
            available_skills,
            available_intents,
            user_context,
            clarification_attempts: 0,
            previous_questions,
            required_clarity: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disambiguation_context_builder() {
        let ctx = DisambiguationContext::from_agent_state(
            vec![
                "Assistant: What is your order number?".to_string(),
                "User: Hello".to_string(),
            ],
            Some("greeting".to_string()),
            Some("Welcome the user".to_string()),
            vec!["search".to_string()],
            vec!["greet".to_string()],
            vec!["cancel_order".to_string()],
            HashMap::new(),
        );

        assert_eq!(ctx.recent_messages.len(), 2);
        assert_eq!(ctx.current_state, Some("greeting".to_string()));
        assert_eq!(ctx.state_prompt, Some("Welcome the user".to_string()));
        assert_eq!(ctx.available_tools.len(), 1);
        assert_eq!(ctx.available_skills.len(), 1);
        assert_eq!(ctx.clarification_attempts, 0);
        // Assistant message ending with '?' is detected
        assert_eq!(ctx.previous_questions.len(), 1);
    }

    #[test]
    fn test_previous_questions_not_populated_without_question_mark() {
        let ctx = DisambiguationContext::from_agent_state(
            vec![
                "Assistant: Here is your result.".to_string(),
                "User: Thanks".to_string(),
            ],
            None,
            None,
            vec![],
            vec![],
            vec![],
            HashMap::new(),
        );

        assert!(ctx.previous_questions.is_empty());
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
//
