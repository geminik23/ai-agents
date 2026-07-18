//! Disambiguation manager orchestrating the full disambiguation flow

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use ai_agents_core::Result;
use ai_agents_llm::LLMRegistry;

use super::clarifier::{ClarificationGenerator, ClarificationObserver, ClarificationParseResult};
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

    pub fn with_clarification_observer(mut self, observer: Arc<dyn ClarificationObserver>) -> Self {
        self.clarifier = self.clarifier.with_observer(observer);
        self
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
        // Handle a pending clarification before applying new-turn enablement so an
        // existing exchange remains resolvable if the active configuration layer changes.
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

        if !self.get_effective_enabled(state_override, skill_override) {
            debug!("Disambiguation disabled by effective layered configuration");
            return Ok(DisambiguationResult::Clear);
        }

        // Check skip conditions
        if self
            .detector
            .should_skip(input, context, &self.config.skip_when)
            .await?
        {
            return Ok(DisambiguationResult::Clear);
        }

        // Resolve layered controls before detection so both prompt guidance and the
        // final decision use the same authoritative threshold and required fields.
        let threshold = self.get_effective_threshold(state_override, skill_override);
        let required_clarity =
            self.get_required_clarity(&context.required_clarity, state_override, skill_override);
        let mut context = context.clone();
        context.required_clarity = required_clarity.clone();

        // Preserve the detector's raw payload, then normalize only the effective
        // boolean. Confidence and all structured evidence remain unchanged.
        let mut detection = self
            .detector
            .detect_with_threshold(input, &context, threshold)
            .await?;
        let detector_is_ambiguous = detection.is_ambiguous;
        detection.is_ambiguous = detection.confidence < threshold;

        info!(
            detector_is_ambiguous,
            is_ambiguous = detection.is_ambiguous,
            confidence = detection.confidence,
            threshold,
            ambiguity_type = ?detection.ambiguity_type,
            "Ambiguity detection complete"
        );

        // Check required clarity fields BEFORE the threshold check.
        // required_clarity is a hard gate: if any required field appears in what_is_unclear, force clarification regardless of confidence score.
        // This handles domain ambiguity ("transfer money" is linguistically clear but missing recipient/amount for the operation).
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

        // The effective confidence threshold is authoritative when no required-field
        // hard gate fired. The model's raw boolean cannot override this decision.
        if detection.confidence >= threshold {
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

    fn get_effective_enabled(
        &self,
        state_override: Option<&StateDisambiguationOverride>,
        skill_override: Option<&SkillDisambiguationOverride>,
    ) -> bool {
        skill_override
            .and_then(|skill| skill.enabled)
            .or_else(|| state_override.and_then(|state| state.enabled))
            .unwrap_or(self.config.enabled)
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
        context_required: &[String],
        state_override: Option<&StateDisambiguationOverride>,
        skill_override: Option<&SkillDisambiguationOverride>,
    ) -> Vec<String> {
        // Preserve every source in layered order while avoiding duplicate prompt/evidence entries.
        let mut required = Vec::new();
        for field in context_required
            .iter()
            .chain(
                state_override
                    .into_iter()
                    .flat_map(|state| state.required_clarity.iter()),
            )
            .chain(
                skill_override
                    .into_iter()
                    .flat_map(|skill| skill.required_clarity.iter()),
            )
        {
            if !required.contains(field) {
                required.push(field.clone());
            }
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
    use ai_agents_llm::mock::MockLLMProvider;

    const CLARIFICATION_RESPONSE: &str = r#"{"question":"Please clarify.","options":null}"#;

    fn manager_with_responses(
        threshold: f32,
        responses: Vec<&str>,
    ) -> (DisambiguationManager, MockLLMProvider) {
        manager_with_enabled_and_responses(true, threshold, responses)
    }

    fn manager_with_enabled_and_responses(
        enabled: bool,
        threshold: f32,
        responses: Vec<&str>,
    ) -> (DisambiguationManager, MockLLMProvider) {
        let mut mock = MockLLMProvider::new("disambiguation-test");
        mock.set_responses(responses.into_iter().map(String::from).collect(), false);
        let observer = mock.clone();

        let mut registry = LLMRegistry::new();
        registry.register("router", Arc::new(mock));

        let config = DisambiguationConfig {
            enabled,
            detection: super::super::config::DetectionConfig {
                threshold,
                ..Default::default()
            },
            ..Default::default()
        };

        (
            DisambiguationManager::new(config, Arc::new(registry)),
            observer,
        )
    }

    fn detection_response(is_ambiguous: bool, confidence: f32) -> String {
        format!(
            r#"{{"is_ambiguous":{is_ambiguous},"confidence":{confidence},"ambiguity_type":"missing_target","reasoning":"raw detector reasoning","what_is_unclear":["target"],"detected_language":"en"}}"#
        )
    }

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

    #[tokio::test]
    async fn state_disable_overrides_enabled_base() {
        let detection = detection_response(true, 0.1);
        let (manager, mock) = manager_with_responses(0.7, vec![detection.as_str()]);
        let state_override = StateDisambiguationOverride {
            enabled: Some(false),
            ..Default::default()
        };

        let result = manager
            .process_input_with_override(
                "Send it",
                &DisambiguationContext::new(),
                Some(&state_override),
                None,
            )
            .await
            .unwrap();

        assert!(result.is_clear());
        assert_eq!(mock.call_count(), 0);
    }

    #[tokio::test]
    async fn skill_disable_overrides_enabled_state_and_base() {
        let detection = detection_response(true, 0.1);
        let (manager, mock) = manager_with_responses(0.7, vec![detection.as_str()]);
        let state_override = StateDisambiguationOverride {
            enabled: Some(true),
            ..Default::default()
        };
        let skill_override = SkillDisambiguationOverride {
            enabled: Some(false),
            ..Default::default()
        };

        let result = manager
            .process_input_with_override(
                "Send it",
                &DisambiguationContext::new(),
                Some(&state_override),
                Some(&skill_override),
            )
            .await
            .unwrap();

        assert!(result.is_clear());
        assert_eq!(mock.call_count(), 0);
    }

    #[tokio::test]
    async fn state_enable_overrides_disabled_base() {
        let detection = detection_response(false, 0.6);
        let (manager, mock) = manager_with_enabled_and_responses(
            false,
            0.7,
            vec![detection.as_str(), CLARIFICATION_RESPONSE],
        );
        let state_override = StateDisambiguationOverride {
            enabled: Some(true),
            ..Default::default()
        };

        let result = manager
            .process_input_with_override(
                "Send it",
                &DisambiguationContext::new(),
                Some(&state_override),
                None,
            )
            .await
            .unwrap();

        assert!(result.needs_clarification());
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn skill_enable_overrides_disabled_state_and_base() {
        let detection = detection_response(false, 0.6);
        let (manager, mock) = manager_with_enabled_and_responses(
            false,
            0.7,
            vec![detection.as_str(), CLARIFICATION_RESPONSE],
        );
        let state_override = StateDisambiguationOverride {
            enabled: Some(false),
            ..Default::default()
        };
        let skill_override = SkillDisambiguationOverride {
            enabled: Some(true),
            ..Default::default()
        };

        let result = manager
            .process_input_with_override(
                "Send it",
                &DisambiguationContext::new(),
                Some(&state_override),
                Some(&skill_override),
            )
            .await
            .unwrap();

        assert!(result.needs_clarification());
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn base_threshold_is_authoritative_and_preserves_raw_detection_fields() {
        let detection = detection_response(false, 0.69);
        let (manager, _) =
            manager_with_responses(0.7, vec![detection.as_str(), CLARIFICATION_RESPONSE]);

        let result = manager
            .process_input("Send it", &DisambiguationContext::new())
            .await
            .unwrap();

        let DisambiguationResult::NeedsClarification { detection, .. } = result else {
            panic!("confidence below the base threshold must trigger clarification");
        };
        assert!(detection.is_ambiguous);
        assert!((detection.confidence - 0.69).abs() < f32::EPSILON);
        assert_eq!(
            detection.ambiguity_type,
            Some(super::super::types::AmbiguityType::MissingTarget)
        );
        assert_eq!(detection.reasoning, "raw detector reasoning");
        assert_eq!(detection.what_is_unclear, vec!["target"]);
        assert_eq!(detection.detected_language.as_deref(), Some("en"));
    }

    #[tokio::test]
    async fn stricter_state_threshold_overrides_base_threshold() {
        let detection = detection_response(false, 0.8);
        let (manager, mock) =
            manager_with_responses(0.7, vec![detection.as_str(), CLARIFICATION_RESPONSE]);
        let state_override = StateDisambiguationOverride {
            threshold: Some(0.9),
            ..Default::default()
        };

        let result = manager
            .process_input_with_override(
                "Send it",
                &DisambiguationContext::new(),
                Some(&state_override),
                None,
            )
            .await
            .unwrap();

        assert!(result.needs_clarification());
        let prompt = &mock.call_history()[0].messages[1].content;
        assert!(prompt.contains("confidence MUST be below 0.9"));
    }

    #[tokio::test]
    async fn stricter_skill_threshold_overrides_state_and_base_thresholds() {
        let detection = detection_response(false, 0.92);
        let (manager, mock) =
            manager_with_responses(0.7, vec![detection.as_str(), CLARIFICATION_RESPONSE]);
        let state_override = StateDisambiguationOverride {
            threshold: Some(0.9),
            ..Default::default()
        };
        let skill_override = SkillDisambiguationOverride {
            threshold: Some(0.95),
            ..Default::default()
        };

        let result = manager
            .process_input_with_override(
                "Send it",
                &DisambiguationContext::new(),
                Some(&state_override),
                Some(&skill_override),
            )
            .await
            .unwrap();

        assert!(result.needs_clarification());
        let prompt = &mock.call_history()[0].messages[1].content;
        assert!(prompt.contains("confidence MUST be below 0.95"));
    }

    #[tokio::test]
    async fn required_clarity_merges_all_sources_and_forces_clarification_without_losing_evidence()
    {
        let response = r#"{"is_ambiguous":false,"confidence":0.99,"ambiguity_type":null,"reasoning":"operation is clear","what_is_unclear":["account"],"detected_language":"en"}"#;
        let (manager, mock) = manager_with_responses(0.7, vec![response, CLARIFICATION_RESPONSE]);
        let context = DisambiguationContext {
            required_clarity: vec!["account".to_string()],
            ..Default::default()
        };
        let state_override = StateDisambiguationOverride {
            required_clarity: vec!["recipient".to_string()],
            ..Default::default()
        };
        let skill_override = SkillDisambiguationOverride {
            required_clarity: vec!["amount".to_string(), "recipient".to_string()],
            ..Default::default()
        };

        assert_eq!(
            manager.get_required_clarity(
                &context.required_clarity,
                Some(&state_override),
                Some(&skill_override),
            ),
            vec!["account", "recipient", "amount"]
        );

        let result = manager
            .process_input_with_override(
                "Make the payment",
                &context,
                Some(&state_override),
                Some(&skill_override),
            )
            .await
            .unwrap();

        let DisambiguationResult::NeedsClarification {
            question,
            detection,
        } = result
        else {
            panic!("a missing required field must override high confidence");
        };
        assert!(detection.is_ambiguous);
        assert!((detection.confidence - 0.99).abs() < f32::EPSILON);
        assert_eq!(
            detection.ambiguity_type,
            Some(super::super::types::AmbiguityType::MissingParameters)
        );
        assert_eq!(detection.reasoning, "operation is clear");
        assert_eq!(detection.what_is_unclear, vec!["account"]);
        assert_eq!(question.clarifying, vec!["account"]);

        let prompt = &mock.call_history()[0].messages[1].content;
        assert!(prompt.contains("account, recipient, amount"));
    }

    #[tokio::test]
    async fn confidence_wins_when_detector_boolean_is_inconsistent() {
        let detection = detection_response(true, 0.8);
        let (manager, mock) = manager_with_responses(0.7, vec![detection.as_str()]);

        let result = manager
            .process_input("Send it", &DisambiguationContext::new())
            .await
            .unwrap();

        assert!(result.is_clear());
        assert_eq!(mock.call_count(), 1);
    }
}
//
