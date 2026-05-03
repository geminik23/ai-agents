use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use ai_agents_core::{AgentError, ChatMessage, LLMConfig, LLMProvider, Result, Role};

use crate::types::{
    Relationship, RelationshipDimensionDefinition, RelationshipEvaluation, RelationshipModel,
};

/// Evaluates how a relationship should change after a turn.
#[async_trait]
pub trait RelationshipEvaluatorTrait: Send + Sync {
    /// Evaluate recent messages and propose relationship changes for the current actor.
    async fn evaluate_turn(
        &self,
        current: &Relationship,
        messages: &[ChatMessage],
        dimensions: &HashMap<String, RelationshipDimensionDefinition>,
    ) -> Result<RelationshipEvaluation>;
}

/// LLM-backed relationship evaluator used by automatic relationship updates.
pub struct RelationshipEvaluator {
    llm: Arc<dyn LLMProvider>,
}

impl RelationshipEvaluator {
    /// Create an LLM-backed relationship evaluator.
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self { llm }
    }

    fn build_prompt(
        &self,
        current: &Relationship,
        messages: &[ChatMessage],
        dimensions: &HashMap<String, RelationshipDimensionDefinition>,
    ) -> String {
        let mut prompt = if matches!(current.model, RelationshipModel::TwoSided) {
            String::from(
                "You are a relationship memory evaluator.\n\
                 Evaluate how stored relationship perspectives should change after the recent conversation.\n\
                 Writable perspectives:\n\
                 - agent_to_actor: the agent's stance toward the actor.\n\
                 - perceived_actor_to_agent: the agent's inferred view of the actor's stance toward the agent.\n\
                 The mutual view is derived from those two stored perspectives and is read-only. Do not output mutual.\n\n\
                 Dimensions:\n",
            )
        } else {
            String::from(
                "You are a relationship memory evaluator.\n\
                 Evaluate how the agent's relationship toward the current actor should change after the recent conversation.\n\
                 The relationship describes the agent's stance toward the actor, not the actor's stance toward the agent.\n\n\
                 Dimensions:\n",
            )
        };

        let mut defs: Vec<_> = dimensions.iter().collect();
        defs.sort_by(|a, b| a.0.cmp(b.0));
        for (name, def) in defs {
            let current_value = current.dimensions.get(name).copied().unwrap_or(def.default);
            if matches!(current.model, RelationshipModel::TwoSided) {
                let perceived = current
                    .perceived_actor_to_agent
                    .get(name)
                    .copied()
                    .unwrap_or(def.default);
                prompt.push_str(&format!(
                    "- {}: {}. range [{:.2}, {:.2}], agent_to_actor {:.2}, perceived_actor_to_agent {:.2}\n",
                    name, def.description, def.min, def.max, current_value, perceived
                ));
            } else {
                prompt.push_str(&format!(
                    "- {}: {}. range [{:.2}, {:.2}], current {:.2}\n",
                    name, def.description, def.min, def.max, current_value
                ));
            }
        }

        if matches!(current.model, RelationshipModel::TwoSided) {
            prompt.push_str(
                "\nOutput JSON only in this shape:\n\
                 {\n\
                   \"changes\": [\n\
                     {\"perspective\": \"agent_to_actor\", \"dimension\": \"trust\", \"delta\": 0.1, \"confidence\": 0.8, \"reason\": \"agent received reliable information from the actor\"},\n\
                     {\"perspective\": \"perceived_actor_to_agent\", \"dimension\": \"trust\", \"delta\": 0.1, \"confidence\": 0.8, \"reason\": \"actor explicitly said they trust the agent more\"}\n\
                   ],\n\
                   \"notable_event\": {\"description\": \"short event\", \"significance\": 0.7}\n\
                 }\n\n\
                 Rules:\n\
                 - Every change must include perspective, dimension, delta, confidence, and reason.\n\
                 - Use small deltas for small signals. Use 0 changes if nothing meaningful changed.\n\
                 - Do not output dimensions that are not listed.\n\
                 - For two-sided relationships, use only agent_to_actor or perceived_actor_to_agent.\n\
                 - Do not output mutual. Mutual is derived by the runtime from the two stored perspectives.\n\
                 - If both stored perspectives changed, output two separate changes.\n\
                 - Update perceived_actor_to_agent only when the actor explicitly or strongly implies their stance toward the agent changed.\n\
                 - Confidence is 0.0 to 1.0.\n\
                 - Delta may be positive or negative. Runtime will clamp values.\n\
                 - notable_event may be null if nothing notable happened.\n\
                 - Keep reasons and event descriptions short and concrete.\n\n\
                 Recent conversation:\n",
            );
        } else {
            prompt.push_str(
                "\nOutput JSON only in this shape:\n\
                 {\n\
                   \"changes\": [\n\
                     {\"perspective\": \"agent_to_actor\", \"dimension\": \"trust\", \"delta\": 0.1, \"confidence\": 0.8, \"reason\": \"actor gave reliable information\"}\n\
                   ],\n\
                   \"notable_event\": {\"description\": \"short event\", \"significance\": 0.7}\n\
                 }\n\n\
                 Rules:\n\
                 - Every change must include perspective, dimension, delta, confidence, and reason.\n\
                 - Use small deltas for small signals. Use 0 changes if nothing meaningful changed.\n\
                 - Do not output dimensions that are not listed.\n\
                 - For one-sided relationships, use perspective agent_to_actor.\n\
                 - Confidence is 0.0 to 1.0.\n\
                 - Delta may be positive or negative. Runtime will clamp values.\n\
                 - notable_event may be null if nothing notable happened.\n\
                 - Keep reasons and event descriptions short and concrete.\n\n\
                 Recent conversation:\n",
            );
        }

        for msg in messages {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
                Role::Function => "Function",
                Role::Tool => "Tool",
            };
            prompt.push_str(&format!("{}: {}\n", role, msg.content));
        }

        prompt.push_str("\nReturn JSON only.\n");
        prompt
    }

    fn parse_response(&self, content: &str) -> Result<RelationshipEvaluation> {
        let object = extract_json_object(content).ok_or_else(|| {
            AgentError::LLM("relationship evaluator response did not contain a JSON object".into())
        })?;
        serde_json::from_str(&object).map_err(|e| {
            AgentError::LLM(format!(
                "relationship evaluator response could not be parsed: {}",
                e
            ))
        })
    }
}

#[async_trait]
impl RelationshipEvaluatorTrait for RelationshipEvaluator {
    async fn evaluate_turn(
        &self,
        current: &Relationship,
        messages: &[ChatMessage],
        dimensions: &HashMap<String, RelationshipDimensionDefinition>,
    ) -> Result<RelationshipEvaluation> {
        if messages.is_empty() {
            return Ok(RelationshipEvaluation::default());
        }

        let prompt = self.build_prompt(current, messages, dimensions);
        let llm_messages = vec![ChatMessage::user(prompt)];
        let config = LLMConfig {
            temperature: Some(0.1),
            max_tokens: Some(1024),
            ..Default::default()
        };

        let response = self
            .llm
            .complete(&llm_messages, Some(&config))
            .await
            .map_err(|e| AgentError::LLM(format!("relationship evaluator failed: {}", e)))?;

        match self.parse_response(&response.content) {
            Ok(eval) => {
                debug!(
                    changes = eval.changes.len(),
                    "relationship evaluation parsed"
                );
                Ok(eval)
            }
            Err(e) => {
                warn!(error = %e, "relationship evaluator parse failed");
                Err(e)
            }
        }
    }
}

fn extract_json_object(text: &str) -> Option<String> {
    if serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|v| v.as_object().map(|_| ()))
        .is_some()
    {
        return Some(text.to_string());
    }

    if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            let inner = after[..end].trim();
            if serde_json::from_str::<serde_json::Value>(inner)
                .ok()
                .and_then(|v| v.as_object().map(|_| ()))
                .is_some()
            {
                return Some(inner.to_string());
            }
        }
    }

    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        if let Some(end) = after.find("```") {
            let inner = after[..end].trim();
            if serde_json::from_str::<serde_json::Value>(inner)
                .ok()
                .and_then(|v| v.as_object().map(|_| ()))
                .is_some()
            {
                return Some(inner.to_string());
            }
        }
    }

    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    let candidate = &text[start..=end];
    if serde_json::from_str::<serde_json::Value>(candidate)
        .ok()
        .and_then(|v| v.as_object().map(|_| ()))
        .is_some()
    {
        Some(candidate.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;

    use ai_agents_core::{FinishReason, LLMChunk, LLMError, LLMFeature, LLMResponse};

    use super::*;

    struct TestLLM;

    #[async_trait]
    impl LLMProvider for TestLLM {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<LLMResponse, LLMError> {
            Ok(LLMResponse::new("{}", FinishReason::Stop))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<
            Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            Ok(Box::new(stream::empty()))
        }

        fn provider_name(&self) -> &str {
            "test"
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    #[test]
    fn test_extract_json_object_direct() {
        let text = r#"{"changes":[],"notable_event":null}"#;
        assert!(extract_json_object(text).is_some());
    }

    #[test]
    fn test_extract_json_object_fenced() {
        let text = "```json\n{\"changes\":[],\"notable_event\":null}\n```";
        assert!(extract_json_object(text).is_some());
    }

    #[test]
    fn test_parse_response() {
        let evaluator = RelationshipEvaluator::new(Arc::new(TestLLM));
        let parsed = evaluator
            .parse_response(r#"{"changes":[],"notable_event":null}"#)
            .unwrap();
        assert!(parsed.changes.is_empty());
    }

    #[test]
    fn test_parse_response_requires_perspective_and_confidence() {
        let evaluator = RelationshipEvaluator::new(Arc::new(TestLLM));
        assert!(
            evaluator
                .parse_response(
                    r#"{"changes":[{"dimension":"trust","delta":0.1,"confidence":0.8}],"notable_event":null}"#,
                )
                .is_err()
        );
        assert!(
            evaluator
                .parse_response(
                    r#"{"changes":[{"perspective":"agent_to_actor","dimension":"trust","delta":0.1}],"notable_event":null}"#,
                )
                .is_err()
        );
    }

    #[test]
    fn test_two_sided_prompt_treats_mutual_as_read_only() {
        let evaluator = RelationshipEvaluator::new(Arc::new(TestLLM));
        let relationship = Relationship::new(
            "actor_1",
            None,
            &crate::defaults::builtin_dimensions(),
            RelationshipModel::TwoSided,
        );
        let prompt = evaluator.build_prompt(
            &relationship,
            &[ChatMessage::user("That answer helped me trust you more.")],
            &crate::defaults::builtin_dimensions(),
        );
        assert!(prompt.contains("Do not output mutual"));
        assert!(!prompt.contains("mutual 0.00"));
        assert!(prompt.contains("perceived_actor_to_agent"));
    }
}
