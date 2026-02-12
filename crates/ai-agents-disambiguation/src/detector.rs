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
                    if let Some(ref current) = context.current_state {
                        if states.contains(current) {
                            debug!(state = %current, "Skipping: in excluded state");
                            return Ok(true);
                        }
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
                        debug!("Skipping: answering agent question");
                        return Ok(true);
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

    /// Detect if input is ambiguous
    pub async fn detect(
        &self,
        input: &str,
        context: &DisambiguationContext,
    ) -> Result<AmbiguityDetectionResult> {
        let llm = self.llm_registry.get(&self.config.llm).map_err(|_| {
            AgentError::Config(format!(
                "LLM '{}' not found for disambiguation",
                self.config.llm
            ))
        })?;

        let prompt = self.build_detection_prompt(input, context);
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

    fn build_detection_prompt(&self, input: &str, context: &DisambiguationContext) -> String {
        if let Some(ref custom_prompt) = self.config.prompt {
            return self.render_custom_prompt(custom_prompt, input, context);
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
            prompt.push_str(&format!("Current state: {}\n\n", state));
        }

        prompt.push_str(
            r#"Respond in JSON format:
{
  "is_ambiguous": true/false,
  "confidence": 0.0-1.0 (how confident the user's intent is clear),
  "ambiguity_type": "missing_target|missing_action|missing_parameters|multiple_intents|vague_reference|implicit_context|null",
  "reasoning": "brief explanation",
  "what_is_unclear": ["list", "of", "unclear", "parts"],
  "detected_language": "language code (e.g., en, ko, ja, zh)"
}

IMPORTANT: Output ONLY valid JSON, no other text."#,
        );

        prompt
    }

    fn render_custom_prompt(
        &self,
        template: &str,
        input: &str,
        context: &DisambiguationContext,
    ) -> String {
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
        result
    }

    fn parse_detection_response(&self, content: &str) -> Result<AmbiguityDetectionResult> {
        let json_str = extract_json(content);

        let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            AgentError::Other(format!("Failed to parse disambiguation response: {}", e))
        })?;

        let is_ambiguous = parsed["is_ambiguous"].as_bool().unwrap_or(false);
        let confidence = parsed["confidence"].as_f64().unwrap_or(1.0) as f32;

        // If confidence is below threshold, mark as ambiguous
        let is_ambiguous = is_ambiguous || confidence < self.config.threshold;

        if !is_ambiguous {
            return Ok(AmbiguityDetectionResult::clear());
        }

        let ambiguity_type = parsed["ambiguity_type"]
            .as_str()
            .and_then(|s| match s {
                "missing_target" => Some(AmbiguityType::MissingTarget),
                "missing_action" => Some(AmbiguityType::MissingAction),
                "missing_parameters" => Some(AmbiguityType::MissingParameters),
                "multiple_intents" => Some(AmbiguityType::MultipleIntents),
                "vague_reference" => Some(AmbiguityType::VagueReference),
                "implicit_context" => Some(AmbiguityType::ImplicitContext),
                _ => None,
            })
            .unwrap_or(AmbiguityType::Unknown);

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

        let mut result = AmbiguityDetectionResult::ambiguous(
            confidence,
            ambiguity_type,
            reasoning,
            what_is_unclear,
        );
        if let Some(lang) = detected_language {
            result = result.with_language(lang);
        }

        Ok(result)
    }

    async fn is_social_message(&self, input: &str) -> Result<bool> {
        let llm = match self.llm_registry.get(&self.config.llm) {
            Ok(l) => l,
            Err(_) => return Ok(false),
        };

        let prompt = format!(
            r#"Is this message a social/greeting message (hello, thanks, bye, etc.) that doesn't require any action?
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

    #[test]
    fn test_extract_json_via_util() {
        assert_eq!(extract_json(r#"{"ok": true}"#), r#"{"ok": true}"#);
        let md = "```json\n{\"ok\": true}\n```";
        assert_eq!(extract_json(md), r#"{"ok": true}"#);
    }
}
