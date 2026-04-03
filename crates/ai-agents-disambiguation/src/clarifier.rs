//! LLM-based clarification question generator

use std::sync::Arc;
use tracing::debug;

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::{ChatMessage, LLMRegistry};

use super::config::{ClarificationConfig, ClarificationStyle};
use super::types::{
    AmbiguityDetectionResult, ClarificationOption, ClarificationQuestion, DisambiguationContext,
};
use super::util::extract_json;

/// LLM-based generator for clarification questions
pub struct ClarificationGenerator {
    config: ClarificationConfig,
    llm_registry: Arc<LLMRegistry>,
}

impl ClarificationGenerator {
    pub fn new(config: ClarificationConfig, llm_registry: Arc<LLMRegistry>) -> Self {
        Self {
            config,
            llm_registry,
        }
    }

    /// Generate a clarification question based on detection result
    pub async fn generate(
        &self,
        input: &str,
        detection: &AmbiguityDetectionResult,
        context: &DisambiguationContext,
        custom_template: Option<&str>,
        required_clarity: &[String],
    ) -> Result<ClarificationQuestion> {
        // If custom template provided (from skill override), use it
        if let Some(template) = custom_template {
            return Ok(ClarificationQuestion::open(template));
        }

        let llm_alias = self.config.llm.as_deref().unwrap_or("router");

        let llm = self.llm_registry.get(llm_alias).map_err(|_| {
            AgentError::Config(format!("LLM '{}' not found for clarification", llm_alias))
        })?;

        let style = self.determine_style(detection);
        let prompt =
            self.build_generation_prompt(input, detection, context, &style, required_clarity);

        let messages = vec![
            ChatMessage::system(
                "You are a helpful assistant that asks clarifying questions. \
                 Be concise and friendly. Match the user's language. \
                 Respond only with valid JSON.",
            ),
            ChatMessage::user(&prompt),
        ];

        let response = llm
            .complete(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(format!("Clarification generation failed: {}", e)))?;

        self.parse_generation_response(&response.content, &style, &detection.what_is_unclear)
    }

    /// Parse user's response to clarification question
    pub async fn parse_response(
        &self,
        original_input: &str,
        question: &ClarificationQuestion,
        user_response: &str,
        context: &DisambiguationContext,
    ) -> Result<ClarificationParseResult> {
        let llm_alias = self.config.llm.as_deref().unwrap_or("router");

        let llm = self.llm_registry.get(llm_alias).map_err(|_| {
            AgentError::Config(format!("LLM '{}' not found for parsing", llm_alias))
        })?;

        let prompt = self.build_parse_prompt(original_input, question, user_response, context);

        let messages = vec![
            ChatMessage::system(
                "You are an expert at understanding user intent. Respond only with valid JSON.",
            ),
            ChatMessage::user(&prompt),
        ];

        let response = llm
            .complete(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(format!("Clarification parsing failed: {}", e)))?;

        self.parse_response_result(&response.content, original_input)
    }

    fn determine_style(&self, detection: &AmbiguityDetectionResult) -> ClarificationStyle {
        if self.config.style != ClarificationStyle::Auto {
            return self.config.style.clone();
        }

        // Auto-determine based on ambiguity type
        match &detection.ambiguity_type {
            Some(t) => match t {
                super::types::AmbiguityType::MultipleIntents => ClarificationStyle::Options,
                super::types::AmbiguityType::VagueReference => ClarificationStyle::Options,
                super::types::AmbiguityType::MissingTarget => ClarificationStyle::Hybrid,
                super::types::AmbiguityType::MissingParameters => ClarificationStyle::Open,
                _ => ClarificationStyle::Open,
            },
            None => ClarificationStyle::Open,
        }
    }

    fn build_generation_prompt(
        &self,
        input: &str,
        detection: &AmbiguityDetectionResult,
        context: &DisambiguationContext,
        style: &ClarificationStyle,
        required_clarity: &[String],
    ) -> String {
        let language_hint = detection
            .detected_language
            .as_deref()
            .map(|l| format!("Respond in {} language.", language_name(l)))
            .unwrap_or_else(|| "Match the user's language.".to_string());

        let style_instruction = match style {
            ClarificationStyle::Options => format!(
                "Provide exactly {} clear options for the user to choose from. Label them A), B), C), etc.",
                self.config.max_options
            ),
            ClarificationStyle::YesNo => {
                "You MUST ask exactly ONE yes/no question. Do NOT list options or multiple choices. \
                 Pick the single most likely interpretation and ask the user to confirm or deny it. \
                 Example: \"Did you mean to cancel your subscription? (yes/no)\"".to_string()
            }
            ClarificationStyle::Hybrid => format!(
                "Provide up to {} labeled options (A, B, C...) but also allow free-form input by ending with something like \"or describe what you need\".",
                self.config.max_options
            ),
            ClarificationStyle::Open | ClarificationStyle::Auto => {
                "Ask a single open-ended clarifying question. Do NOT list options.".to_string()
            }
        };

        let other_option = if self.config.include_other_option
            && matches!(
                style,
                ClarificationStyle::Options | ClarificationStyle::Hybrid
            ) {
            "Include an 'Other' option for free-form input."
        } else {
            ""
        };

        // If the state machine declares canonical intents, constrain options to them
        let intent_constraint = if !context.available_intents.is_empty() {
            format!(
                "IMPORTANT: The options MUST correspond to these available workflows: {}. Do NOT invent other categories.",
                context.available_intents.join(", ")
            )
        } else {
            String::new()
        };

        let mut prompt = format!(
            r#"The user said: "{}"

This message is ambiguous because: {}
What is unclear: {}

{}
{}
{}
{}
"#,
            input,
            detection.reasoning,
            detection.what_is_unclear.join(", "),
            language_hint,
            style_instruction,
            other_option,
            intent_constraint
        );

        if !required_clarity.is_empty() {
            prompt.push_str(&format!(
                "IMPORTANT: Only ask about these specific fields: {}. Do NOT ask about anything else (e.g. method, purpose, reason). If the user has already provided some of these, only ask about the missing ones.\n\n",
                required_clarity.join(", ")
            ));
        }

        if !context.recent_messages.is_empty() {
            prompt.push_str(&format!(
                "\nRecent conversation:\n{}\n",
                context.recent_messages.join("\n")
            ));
        }

        if !context.previous_questions.is_empty() {
            prompt.push_str(&format!(
                "\nPrevious clarification questions asked:\n{}\n",
                context.previous_questions.join("\n")
            ));
            prompt.push_str("Ask something different from the previous questions.\n");
        }

        let json_format = match style {
            ClarificationStyle::YesNo | ClarificationStyle::Open => {
                r#"
Respond in JSON format:
{
  "question": "Your single clarifying question",
  "options": null
}

IMPORTANT:
- The "question" field is a single plain question with NO option lists.
- Set "options" to null.
- Output ONLY valid JSON, no other text."#
            }
            _ => {
                r#"
Respond in JSON format:
{
  "question": "The full clarifying question including labeled options in the text itself",
  "options": [
    {"id": "1", "label": "First option"},
    {"id": "2", "label": "Second option"}
  ]
}

IMPORTANT:
- The "question" field MUST be self-contained: if there are options, include them naturally in the question text (e.g., "What would you like to cancel?\nA) Order\nB) Reservation\nC) Subscription").
- The "options" array is a structured copy of the same options for programmatic use.
- Output ONLY valid JSON, no other text."#
            }
        };

        prompt.push_str(json_format);

        prompt
    }

    fn build_parse_prompt(
        &self,
        original_input: &str,
        question: &ClarificationQuestion,
        user_response: &str,
        _context: &DisambiguationContext,
    ) -> String {
        let mut prompt = format!(
            r#"Original user request: "{}"

We asked for clarification: "{}"
"#,
            original_input, question.question
        );

        if let Some(ref options) = question.options {
            prompt.push_str("Available options:\n");
            for opt in options {
                prompt.push_str(&format!("- {}: {}\n", opt.id, opt.label));
            }
        }

        prompt.push_str(&format!(
            r#"
User responded: "{}"

Determine the user's intent regarding the clarification question.

There are four possibilities:
1. "answered" - The user clearly selected one of the available options or provided the requested information.
2. "abandoned" - The user wants to cancel, drop, or forget this topic. Examples: "forget it", "never mind", "cancel", "skip this", "I don't want that anymore".
3. "switched" - The user's message is about a DIFFERENT topic entirely and is NOT an attempt to answer the clarification question. They have moved on to something else.
4. "unclear" - The user's response does not clearly answer the question, but they have not abandoned or switched topics either.

Rules:
- "answered" requires a clear selection or direct answer.
- "abandoned" requires an explicit signal that the user no longer wants to pursue the original request.
- "switched" means the response has no relationship to the original request or the clarification question.
- "unclear" is the default when none of the above clearly apply. It triggers a retry.
- Do NOT guess. If the response could be an attempt to answer (even a bad one), use "unclear" not "switched".
- Repeating the same words as the original request (e.g. original was "send it" and response is "just send it") is NOT a valid selection - use "unclear".

Respond in JSON format:
{{
  "status": "answered|abandoned|switched|unclear",
  "selected_option": "option_id if answered, null otherwise",
  "enriched_input": "The original request rewritten to be unambiguous (only if status is answered, otherwise null)",
  "resolved": {{
    "intent": "a short snake_case label derived from the selected option (only if status is answered, otherwise null)"
  }}
}}

IMPORTANT:
- When in doubt, use "unclear". It is better to ask again than to guess wrong.
- Output ONLY valid JSON, no other text."#,
            user_response
        ));

        prompt
    }

    fn parse_generation_response(
        &self,
        content: &str,
        style: &ClarificationStyle,
        clarifying: &[String],
    ) -> Result<ClarificationQuestion> {
        let json_str = extract_json(content);

        let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            AgentError::Other(format!("Failed to parse clarification response: {}", e))
        })?;

        let question = parsed["question"]
            .as_str()
            .ok_or_else(|| AgentError::Other("Missing question in response".to_string()))?
            .to_string();

        let options: Option<Vec<ClarificationOption>> = parsed["options"].as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let id = v["id"].as_str()?.to_string();
                    let label = v["label"].as_str()?.to_string();
                    let description = v["description"].as_str().map(String::from);
                    Some(ClarificationOption {
                        id,
                        label,
                        description,
                    })
                })
                .collect()
        });

        debug!(
            question = %question,
            options_count = options.as_ref().map(|o| o.len()).unwrap_or(0),
            "Generated clarification question"
        );

        Ok(ClarificationQuestion {
            question,
            options,
            style: style.clone(),
            clarifying: clarifying.to_vec(),
        })
    }

    fn parse_response_result(
        &self,
        content: &str,
        original_input: &str,
    ) -> Result<ClarificationParseResult> {
        let json_str = extract_json(content);

        let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            AgentError::Other(format!("Failed to parse clarification result: {}", e))
        })?;

        // New 4-way status check
        if let Some(status) = parsed["status"].as_str() {
            return match status {
                "answered" => {
                    let enriched_input = parsed["enriched_input"]
                        .as_str()
                        .unwrap_or(original_input)
                        .to_string();
                    let resolved = parsed["resolved"]
                        .as_object()
                        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();
                    Ok(ClarificationParseResult::Understood {
                        enriched_input,
                        resolved,
                    })
                }
                "abandoned" => Ok(ClarificationParseResult::Abandoned),
                "switched" => Ok(ClarificationParseResult::TopicSwitch),
                _ => Ok(ClarificationParseResult::NotUnderstood),
            };
        }

        // Backward-compatible: old "understood" boolean format
        let understood = parsed["understood"].as_bool().unwrap_or(false);

        if !understood {
            return Ok(ClarificationParseResult::NotUnderstood);
        }

        let enriched_input = parsed["enriched_input"]
            .as_str()
            .unwrap_or(original_input)
            .to_string();

        let resolved = parsed["resolved"]
            .as_object()
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Ok(ClarificationParseResult::Understood {
            enriched_input,
            resolved,
        })
    }
}

/// Result of parsing user's clarification response
#[derive(Debug, Clone)]
pub enum ClarificationParseResult {
    /// User's response was understood
    Understood {
        enriched_input: String,
        resolved: std::collections::HashMap<String, serde_json::Value>,
    },
    /// User's response was not clear
    NotUnderstood,
    /// User explicitly abandoned the clarification (e.g. "forget it", "never mind")
    Abandoned,
    /// User switched to a completely different topic
    TopicSwitch,
}

fn language_name(code: &str) -> &str {
    match code {
        "en" => "English",
        "ko" => "Korean",
        "ja" => "Japanese",
        "zh" => "Chinese",
        "es" => "Spanish",
        "fr" => "French",
        "de" => "German",
        "pt" => "Portuguese",
        "ru" => "Russian",
        "ar" => "Arabic",
        "hi" => "Hindi",
        "vi" => "Vietnamese",
        "th" => "Thai",
        _ => "the same",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_name() {
        assert_eq!(language_name("ko"), "Korean");
        assert_eq!(language_name("ja"), "Japanese");
        assert_eq!(language_name("unknown"), "the same");
    }

    #[test]
    fn test_clarification_parse_result() {
        let result = ClarificationParseResult::Understood {
            enriched_input: "Send $100 to John".to_string(),
            resolved: std::collections::HashMap::new(),
        };

        match result {
            ClarificationParseResult::Understood { enriched_input, .. } => {
                assert_eq!(enriched_input, "Send $100 to John");
            }
            _ => panic!("Expected Understood variant"),
        }
    }

    #[test]
    fn test_parse_result_abandoned() {
        let generator = ClarificationGenerator::new(
            super::super::config::ClarificationConfig::default(),
            Arc::new(LLMRegistry::new()),
        );
        let content = r#"{"status": "abandoned"}"#;
        let result = generator.parse_response_result(content, "send it").unwrap();
        assert!(matches!(result, ClarificationParseResult::Abandoned));
    }

    #[test]
    fn test_parse_result_switched() {
        let generator = ClarificationGenerator::new(
            super::super::config::ClarificationConfig::default(),
            Arc::new(LLMRegistry::new()),
        );
        let content = r#"{"status": "switched"}"#;
        let result = generator.parse_response_result(content, "send it").unwrap();
        assert!(matches!(result, ClarificationParseResult::TopicSwitch));
    }

    #[test]
    fn test_parse_result_backward_compat() {
        let generator = ClarificationGenerator::new(
            super::super::config::ClarificationConfig::default(),
            Arc::new(LLMRegistry::new()),
        );
        // Old format with "understood" boolean should still work
        let content = r#"{"understood": true, "enriched_input": "Send $100", "resolved": {"intent": "send_money"}}"#;
        let result = generator.parse_response_result(content, "send it").unwrap();
        match result {
            ClarificationParseResult::Understood { enriched_input, .. } => {
                assert_eq!(enriched_input, "Send $100");
            }
            _ => panic!("Expected Understood"),
        }

        let content = r#"{"understood": false}"#;
        let result = generator.parse_response_result(content, "send it").unwrap();
        assert!(matches!(result, ClarificationParseResult::NotUnderstood));
    }

    #[test]
    fn test_parse_result_answered_status() {
        let generator = ClarificationGenerator::new(
            super::super::config::ClarificationConfig::default(),
            Arc::new(LLMRegistry::new()),
        );
        let content = r#"{"status": "answered", "enriched_input": "Cancel my order #123", "resolved": {"intent": "cancel_order"}}"#;
        let result = generator
            .parse_response_result(content, "cancel it")
            .unwrap();
        match result {
            ClarificationParseResult::Understood {
                enriched_input,
                resolved,
            } => {
                assert_eq!(enriched_input, "Cancel my order #123");
                assert_eq!(
                    resolved.get("intent").and_then(|v| v.as_str()),
                    Some("cancel_order")
                );
            }
            _ => panic!("Expected Understood"),
        }
    }

    #[test]
    fn test_parse_result_unclear_status() {
        let generator = ClarificationGenerator::new(
            super::super::config::ClarificationConfig::default(),
            Arc::new(LLMRegistry::new()),
        );
        let content = r#"{"status": "unclear"}"#;
        let result = generator.parse_response_result(content, "send it").unwrap();
        assert!(matches!(result, ClarificationParseResult::NotUnderstood));
    }
}
