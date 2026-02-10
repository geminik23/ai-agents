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
        let prompt = self.build_generation_prompt(input, detection, context, &style);

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
    ) -> String {
        let language_hint = detection
            .detected_language
            .as_deref()
            .map(|l| format!("Respond in {} language.", language_name(l)))
            .unwrap_or_else(|| "Match the user's language.".to_string());

        let style_instruction = match style {
            ClarificationStyle::Options => format!(
                "Provide {} clear options for the user to choose from.",
                self.config.max_options
            ),
            ClarificationStyle::YesNo => "Ask a yes/no question.".to_string(),
            ClarificationStyle::Hybrid => format!(
                "Provide up to {} options but also allow free-form input.",
                self.config.max_options
            ),
            ClarificationStyle::Open | ClarificationStyle::Auto => {
                "Ask an open-ended clarifying question.".to_string()
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

        let mut prompt = format!(
            r#"The user said: "{}"

This message is ambiguous because: {}
What is unclear: {}

{}
{}
{}
"#,
            input,
            detection.reasoning,
            detection.what_is_unclear.join(", "),
            language_hint,
            style_instruction,
            other_option
        );

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

        prompt.push_str(
            r#"
Respond in JSON format:
{
  "question": "The clarifying question to ask",
  "options": [
    {"id": "1", "label": "First option"},
    {"id": "2", "label": "Second option"}
  ] // Only include if style requires options, otherwise null
}

IMPORTANT: Output ONLY valid JSON, no other text."#,
        );

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

Parse the user's response and provide:
1. Whether they made a clear choice
2. The enriched/clarified version of their original request

Respond in JSON format:
{{
  "understood": true/false,
  "selected_option": "option_id if applicable, null otherwise",
  "enriched_input": "The original request with clarifications incorporated",
  "resolved": {{}} // Key-value pairs of what was clarified
}}

IMPORTANT: Output ONLY valid JSON, no other text."#,
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
}
