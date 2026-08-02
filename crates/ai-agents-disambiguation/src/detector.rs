//! LLM-based ambiguity detector

use std::sync::Arc;
use tracing::{debug, warn};

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::{ChatMessage, LLMRegistry};

use super::config::{DetectionConfig, SkipCondition};
use super::types::{AmbiguityDetectionResult, AmbiguityType, DisambiguationContext};
use super::util::extract_json;

/// LLM-based detector for ambiguous user inputs
pub struct AmbiguityDetector {
    config: DetectionConfig,
    llm_registry: Arc<LLMRegistry>,
}

impl AmbiguityDetector {
    pub fn new(config: DetectionConfig, llm_registry: Arc<LLMRegistry>) -> Self {
        Self {
            config,
            llm_registry,
        }
    }

    /// Check if input should be skipped based on skip conditions
    pub async fn should_skip(
        &self,
        input: &str,
        context: &DisambiguationContext,
        skip_conditions: &[SkipCondition],
    ) -> Result<bool> {
        for condition in skip_conditions {
            match condition {
                SkipCondition::ShortInput { max_chars } => {
                    if input.chars().count() <= *max_chars {
                        debug!(input_len = input.len(), max_chars, "Skipping: short input");
                        return Ok(true);
                    }
                }
                SkipCondition::InState { states } => {
                    if let Some(ref current) = context.current_state
                        && states.contains(current)
                    {
                        debug!(state = %current, "Skipping: in excluded state");
                        return Ok(true);
                    }
                }
                SkipCondition::Social { .. } => {
                    if self.is_social_message(input).await? {
                        debug!("Skipping: social message");
                        return Ok(true);
                    }
                }
                SkipCondition::AnsweringAgentQuestion => {
                    if !context.previous_questions.is_empty() {
                        // Syntactic pre-filter passed (last assistant message ends with '?').
                        // Now do a semantic check: is the user actually answering that question?
                        if self.is_answering_previous_question(input, context).await? {
                            debug!("Skipping: answering agent question (semantic match)");
                            return Ok(true);
                        }
                    }
                }
                SkipCondition::CompleteToolCall => {
                    // Skip if input looks like a complete tool call response
                    if self.is_complete_tool_response(input).await? {
                        debug!("Skipping: complete tool call response");
                        return Ok(true);
                    }
                }
                SkipCondition::Custom { condition } => {
                    if self
                        .evaluate_custom_condition(input, context, condition)
                        .await?
                    {
                        debug!(condition = %condition, "Skipping: custom condition");
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Detect if input is ambiguous using the configured base threshold.
    pub async fn detect(
        &self,
        input: &str,
        context: &DisambiguationContext,
    ) -> Result<AmbiguityDetectionResult> {
        let mut detection = self
            .detect_with_threshold(input, context, self.config.threshold)
            .await?;
        detection.is_ambiguous = detection.confidence < self.config.threshold;
        Ok(detection)
    }

    /// Detect ambiguity while describing the manager-resolved threshold to the model.
    ///
    /// The returned payload preserves the model's raw boolean, confidence, and structured fields.
    /// The manager remains responsible for applying the effective threshold.
    pub(crate) async fn detect_with_threshold(
        &self,
        input: &str,
        context: &DisambiguationContext,
        effective_threshold: f32,
    ) -> Result<AmbiguityDetectionResult> {
        let llm = self.llm_registry.get(&self.config.llm).map_err(|_| {
            AgentError::Config(format!(
                "LLM '{}' not found for disambiguation",
                self.config.llm
            ))
        })?;

        let prompt = self.build_detection_prompt(input, context, effective_threshold);
        let messages = vec![
            ChatMessage::system(
                "You are an expert at analyzing user intent clarity. Respond only with valid JSON.",
            ),
            ChatMessage::user(&prompt),
        ];

        let response = llm
            .complete(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(format!("Disambiguation detection failed: {}", e)))?;

        self.parse_detection_response(&response.content)
    }

    fn build_detection_prompt(
        &self,
        input: &str,
        context: &DisambiguationContext,
        effective_threshold: f32,
    ) -> String {
        if let Some(ref custom_prompt) = self.config.prompt {
            return self.render_custom_prompt(custom_prompt, input, context, effective_threshold);
        }

        let aspects_desc: Vec<&str> = self
            .config
            .aspects
            .iter()
            .map(|a| a.description())
            .collect();

        let mut prompt = format!(
            r#"Analyze if the following user message is ambiguous or unclear.

User message: "{}"

Check for these aspects of ambiguity:
{}

"#,
            input,
            aspects_desc
                .iter()
                .enumerate()
                .map(|(i, a)| format!("{}. {}", i + 1, a))
                .collect::<Vec<_>>()
                .join("\n")
        );

        if !context.recent_messages.is_empty() {
            prompt.push_str(&format!(
                "Recent conversation context:\n{}\n\n",
                context
                    .recent_messages
                    .iter()
                    .map(|m| format!("- {}", m))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }

        if !context.available_tools.is_empty() {
            prompt.push_str(&format!(
                "Available actions/tools: {}\n\n",
                context.available_tools.join(", ")
            ));
        }

        if let Some(ref state) = context.current_state {
            prompt.push_str(&format!("Current state: {}\n", state));
            if let Some(ref state_prompt) = context.state_prompt {
                prompt.push_str(&format!(
                    "State instructions: {}\nThe user's message is likely a response to what this state is asking for. Consider this before flagging ambiguity.\n\n",
                    state_prompt.trim()
                ));
            } else {
                prompt.push('\n');
            }
        }

        if !context.available_intents.is_empty() {
            prompt.push_str(&format!(
                "Available intents in current state: {}\nIf the user's message could match multiple intents, use ambiguity_type \"multiple_intents\".\n\n",
                context.available_intents.join(", ")
            ));
        }

        if !context.required_clarity.is_empty() {
            let fields = context.required_clarity.join(", ");
            prompt.push_str(&format!(
                r#"REQUIRED FIELDS CHECK -- this overrides all other analysis.
The following fields MUST each have a specific, concrete value in the user's message: {fields}
For EACH field, decide: does the message contain an explicit value for it?
A field is present only if the user stated a specific value (a name, a number, a date, etc.).
A field is missing if the message refers to the action but does not supply that value.

Rules:
- Add every missing field name to what_is_unclear.
- Set is_ambiguous to true if ANY required field is missing.
- Set ambiguity_type to "missing_parameters" if ANY required field is missing.
- Set confidence below {effective_threshold} if ANY required field is missing.
- NEVER return an empty what_is_unclear when required fields are missing.

"#
            ));
        }

        prompt.push_str(&format!(
            r#"Respond in JSON format:
{{
  "is_ambiguous": true/false,
  "confidence": 0.0-1.0 (confidence that the user's intent is clear and actionable),
  "ambiguity_type": "missing_target|missing_action|missing_parameters|multiple_intents|vague_reference|implicit_context|null",
  "reasoning": "brief explanation",
  "what_is_unclear": ["list", "of", "unclear", "parts"],
  "detected_language": "language code (e.g., en, ko, ja, zh)"
}}

CONFIDENCE CONSISTENCY RULES:
- If is_ambiguous is true, confidence MUST be below {effective_threshold}.
- If is_ambiguous is false, confidence SHOULD be {effective_threshold} or higher.
- Confidence measures clarity, not confidence in the ambiguity diagnosis.
- The runtime uses the confidence cutoff above as the authoritative ambiguity decision.

IMPORTANT: Output ONLY valid JSON, no other text."#,
        ));

        prompt
    }

    fn render_custom_prompt(
        &self,
        template: &str,
        input: &str,
        context: &DisambiguationContext,
        effective_threshold: f32,
    ) -> String {
        // Preserve custom-prompt compatibility: do not append the default schema or guidance.
        // Existing templates render exactly as before; threshold/required_clarity are opt-in placeholders.
        let mut result = template.to_string();
        result = result.replace("{{ user_input }}", input);
        result = result.replace("{{ recent_messages }}", &context.recent_messages.join("\n"));
        result = result.replace(
            "{{ available_actions }}",
            &context.available_tools.join(", "),
        );
        result = result.replace(
            "{{ current_state }}",
            context.current_state.as_deref().unwrap_or("none"),
        );
        result = result.replace("{{ threshold }}", &effective_threshold.to_string());
        result = result.replace(
            "{{ effective_threshold }}",
            &effective_threshold.to_string(),
        );
        result = result.replace(
            "{{ required_clarity }}",
            &context.required_clarity.join(", "),
        );
        result
    }

    fn parse_detection_response(&self, content: &str) -> Result<AmbiguityDetectionResult> {
        let json_str = extract_json(content);

        let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            AgentError::Other(format!("Failed to parse disambiguation response: {}", e))
        })?;

        let is_ambiguous = parsed["is_ambiguous"].as_bool().unwrap_or(false);
        let confidence = parsed["confidence"].as_f64().unwrap_or(1.0) as f32;

        let ambiguity_type = parsed["ambiguity_type"].as_str().map(|s| match s {
            "missing_target" => AmbiguityType::MissingTarget,
            "missing_action" => AmbiguityType::MissingAction,
            "missing_parameters" => AmbiguityType::MissingParameters,
            "multiple_intents" => AmbiguityType::MultipleIntents,
            "vague_reference" | "vague_references" => AmbiguityType::VagueReference,
            "implicit_context" => AmbiguityType::ImplicitContext,
            _ => AmbiguityType::Unknown,
        });

        let reasoning = parsed["reasoning"]
            .as_str()
            .unwrap_or("Ambiguity detected")
            .to_string();

        let what_is_unclear: Vec<String> = parsed["what_is_unclear"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let detected_language = parsed["detected_language"].as_str().map(String::from);

        Ok(AmbiguityDetectionResult {
            is_ambiguous,
            confidence,
            ambiguity_type,
            reasoning,
            what_is_unclear,
            detected_language,
        })
    }

    /// LLM-based check for whether the user's input answers the assistant's last question.
    async fn is_answering_previous_question(
        &self,
        input: &str,
        context: &DisambiguationContext,
    ) -> Result<bool> {
        let llm = match self.llm_registry.get(&self.config.llm) {
            Ok(l) => l,
            Err(_) => return Ok(false),
        };

        let last_question = context
            .previous_questions
            .last()
            .cloned()
            .unwrap_or_default();

        let prompt = format!(
            r#"The assistant previously asked: {}

The user responded: "{}"

Is the user's message a direct answer or response to the assistant's question?

Answer "yes" ONLY if the user is clearly trying to answer or respond to that specific question.
Answer "no" if the user is:
- Making a new, unrelated request
- Repeating a previous request
- Asking their own question
- Ignoring the assistant's question and saying something else

Answer only "yes" or "no"."#,
            last_question, input
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm.complete(&messages, None).await.map_err(|e| {
            warn!(error = %e, "Answering-agent-question semantic check failed");
            AgentError::LLM(e.to_string())
        })?;

        Ok(response.content.trim().to_lowercase().starts_with("yes"))
    }

    async fn is_social_message(&self, input: &str) -> Result<bool> {
        let llm = match self.llm_registry.get(&self.config.llm) {
            Ok(l) => l,
            Err(_) => return Ok(false),
        };

        let prompt = format!(
            r#"Is this message ONLY a social/greeting message (hello, thanks, bye, etc.) that doesn't require any action?

IMPORTANT — the following are NOT social messages (answer "no" for these):
- Affirmative/negative responses: "yes", "no", "ok", "응", "네", "はい", "y", "n"
- Responses that include a request or confirmation: "응 해줘", "응 취소해줘", "yes please", "ok do it", "해달라고"
- Any message that contains a verb or action request

Only pure greetings/thanks/farewells with no action component are social messages.

Message: "{}"
Answer only "yes" or "no"."#,
            input
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm.complete(&messages, None).await.map_err(|e| {
            warn!(error = %e, "Social message check failed");
            AgentError::LLM(e.to_string())
        })?;

        Ok(response.content.trim().to_lowercase().starts_with("yes"))
    }

    async fn is_complete_tool_response(&self, input: &str) -> Result<bool> {
        let llm = match self.llm_registry.get(&self.config.llm) {
            Ok(l) => l,
            Err(_) => return Ok(false),
        };

        let prompt = format!(
            r#"Is this message a direct, complete answer to a question (e.g. providing a specific value, ID, name, or structured data) rather than a new request?
Message: "{}"
Answer only "yes" or "no"."#,
            input
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm.complete(&messages, None).await.map_err(|e| {
            warn!(error = %e, "Complete tool response check failed");
            AgentError::LLM(e.to_string())
        })?;

        Ok(response.content.trim().to_lowercase().starts_with("yes"))
    }

    async fn evaluate_custom_condition(
        &self,
        input: &str,
        context: &DisambiguationContext,
        condition: &str,
    ) -> Result<bool> {
        let llm = match self.llm_registry.get(&self.config.llm) {
            Ok(l) => l,
            Err(_) => return Ok(false),
        };

        let prompt = format!(
            r#"Evaluate if this condition is true for the given input:
Condition: {}
User input: "{}"
Context state: {}
Answer only "yes" or "no"."#,
            condition,
            input,
            context.current_state.as_deref().unwrap_or("none")
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm.complete(&messages, None).await.map_err(|e| {
            warn!(error = %e, "Custom condition check failed");
            AgentError::LLM(e.to_string())
        })?;

        Ok(response.content.trim().to_lowercase().starts_with("yes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector(config: DetectionConfig) -> AmbiguityDetector {
        AmbiguityDetector::new(config, Arc::new(LLMRegistry::new()))
    }

    #[test]
    fn test_extract_json_via_util() {
        assert_eq!(extract_json(r#"{"ok": true}"#), r#"{"ok": true}"#);
        let md = "```json\n{\"ok\": true}\n```";
        assert_eq!(extract_json(md), r#"{"ok": true}"#);
    }

    #[test]
    fn parse_preserves_raw_confidence_and_structured_fields_for_clear_boolean() {
        let detector = make_detector(DetectionConfig::default());
        let result = detector
            .parse_detection_response(
                r#"{
                    "is_ambiguous": false,
                    "confidence": 0.82,
                    "ambiguity_type": "missing_parameters",
                    "reasoning": "Intent is known but details are incomplete",
                    "what_is_unclear": ["recipient", "amount"],
                    "detected_language": "en"
                }"#,
            )
            .unwrap();

        assert!(!result.is_ambiguous);
        assert!((result.confidence - 0.82).abs() < f32::EPSILON);
        assert_eq!(
            result.ambiguity_type,
            Some(AmbiguityType::MissingParameters)
        );
        assert_eq!(
            result.reasoning,
            "Intent is known but details are incomplete"
        );
        assert_eq!(result.what_is_unclear, vec!["recipient", "amount"]);
        assert_eq!(result.detected_language.as_deref(), Some("en"));
    }

    #[test]
    fn default_prompt_uses_effective_threshold_for_confidence_guidance() {
        let detector = make_detector(DetectionConfig::default());
        let context = DisambiguationContext {
            required_clarity: vec!["recipient".to_string()],
            ..Default::default()
        };

        let prompt = detector.build_detection_prompt("Send money", &context, 0.93);

        assert!(prompt.contains("confidence below 0.93"));
        assert!(prompt.contains("confidence MUST be below 0.93"));
        assert!(prompt.contains("confidence SHOULD be 0.93 or higher"));
        assert!(!prompt.contains("confidence below 0.5"));
        assert!(!prompt.contains("confidence SHOULD be 0.7 or higher"));
    }

    #[test]
    fn custom_prompt_preserves_legacy_rendering_and_supports_opt_in_control_placeholders() {
        let config = DetectionConfig {
            prompt: Some(
                "Input={{ user_input }}; state={{ current_state }}; recent={{ recent_messages }}"
                    .to_string(),
            ),
            ..Default::default()
        };
        let detector = make_detector(config);
        let context = DisambiguationContext::new()
            .with_state("payment")
            .with_recent_messages(vec!["Earlier".to_string()]);

        assert_eq!(
            detector.build_detection_prompt("Pay Alice", &context, 0.95),
            "Input=Pay Alice; state=payment; recent=Earlier"
        );

        let config = DetectionConfig {
            prompt: Some(
                "threshold={{ threshold }}; effective={{ effective_threshold }}; required={{ required_clarity }}"
                    .to_string(),
            ),
            ..Default::default()
        };
        let detector = make_detector(config);
        let context = DisambiguationContext {
            required_clarity: vec!["recipient".to_string(), "amount".to_string()],
            ..Default::default()
        };

        assert_eq!(
            detector.build_detection_prompt("Pay Alice", &context, 0.95),
            "threshold=0.95; effective=0.95; required=recipient, amount"
        );
    }
}
