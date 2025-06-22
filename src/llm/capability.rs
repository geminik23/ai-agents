use async_trait::async_trait;
use std::sync::Arc;

use crate::llm::{
    ChatMessage, LLMCapability, LLMError, LLMProvider, LLMResponse, Role, TaskContext,
    ToolSelection, prompts::*,
};

pub struct DefaultLLMCapability {
    provider: Arc<dyn LLMProvider>,
}

impl DefaultLLMCapability {
    pub fn new(provider: Arc<dyn LLMProvider>) -> Self {
        Self { provider }
    }

    fn extract_json(content: &str) -> Result<serde_json::Value, LLMError> {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
            return Ok(value);
        }

        if let Some(json_str) = extract_json_from_markdown(content) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_str) {
                return Ok(value);
            }
        }

        if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                let json_str = &content[start..=end];
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    return Ok(value);
                }
            }
        }

        Err(LLMError::Serialization(format!(
            "Could not extract JSON from response: {}",
            content
        )))
    }
}

#[async_trait]
impl LLMCapability for DefaultLLMCapability {
    async fn select_tool(
        &self,
        context: &TaskContext,
        user_input: &str,
    ) -> Result<ToolSelection, LLMError> {
        let prompt = ToolSelectionPromptBuilder::new(user_input)
            .with_tools(context.available_tools.clone())
            .with_context(context.clone())
            .build();

        let messages = vec![ChatMessage {
            timestamp: None,
            role: Role::User,
            content: prompt,
            name: None,
        }];

        let response = self.provider.complete(&messages, None).await?;
        let json = Self::extract_json(&response.content)?;

        let tool_id = json
            .get("tool_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LLMError::Serialization("Missing tool_id in response".to_string()))?
            .to_string();

        let confidence = json
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        let reasoning = json
            .get("reasoning")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ToolSelection {
            tool_id,
            confidence,
            reasoning,
        })
    }

    async fn generate_tool_args(
        &self,
        tool_id: &str,
        user_input: &str,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value, LLMError> {
        let prompt = ToolArgsPromptBuilder::new(tool_id, schema.clone(), user_input).build();

        let messages = vec![ChatMessage {
            timestamp: None,
            role: Role::User,
            content: prompt,
            name: None,
        }];

        let response = self.provider.complete(&messages, None).await?;
        Self::extract_json(&response.content)
    }

    async fn evaluate_yesno(
        &self,
        question: &str,
        context: &TaskContext,
    ) -> Result<(bool, String), LLMError> {
        let prompt = YesNoPromptBuilder::new(question)
            .with_context(context.clone())
            .build();

        let messages = vec![ChatMessage {
            timestamp: None,
            role: Role::User,
            content: prompt,
            name: None,
        }];

        let response = self.provider.complete(&messages, None).await?;
        let json = Self::extract_json(&response.content)?;

        let answer = json
            .get("answer")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| LLMError::Serialization("Missing answer in response".to_string()))?;

        let reasoning = json
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("No reasoning provided")
            .to_string();

        Ok((answer, reasoning))
    }

    async fn classify(
        &self,
        input: &str,
        categories: &[String],
    ) -> Result<(String, f32), LLMError> {
        let prompt = ClassificationPromptBuilder::new(input, categories.to_vec()).build();

        let messages = vec![ChatMessage {
            timestamp: None,
            role: Role::User,
            content: prompt,
            name: None,
        }];

        let response = self.provider.complete(&messages, None).await?;
        let json = Self::extract_json(&response.content)?;

        let category = json
            .get("category")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LLMError::Serialization("Missing category in response".to_string()))?
            .to_string();

        let confidence = json
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        Ok((category, confidence))
    }

    async fn process_task(
        &self,
        context: &TaskContext,
        system_prompt: &str,
    ) -> Result<LLMResponse, LLMError> {
        let prompt = TaskProcessingPromptBuilder::new(system_prompt, context.clone()).build();

        let mut messages = vec![ChatMessage {
            timestamp: None,
            role: Role::System,
            content: prompt,
            name: None,
        }];
        messages.extend(context.recent_messages.clone());

        self.provider.complete(&messages, None).await
    }
}

fn extract_json_from_markdown(content: &str) -> Option<String> {
    if let Some(start_pos) = content.find("```json") {
        let start_idx = start_pos + "```json".len();
        if let Some(end_pos) = content[start_idx..].find("```") {
            let json_str = &content[start_idx..start_idx + end_pos];
            return Some(json_str.trim().to_string());
        }
    }

    if let Some(start_pos) = content.find("```") {
        let start_idx = start_pos + "```".len();
        if let Some(end_pos) = content[start_idx..].find("```") {
            let json_str = &content[start_idx..start_idx + end_pos];
            return Some(json_str.trim().to_string());
        }
    }

    None
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::FinishReason;
    use crate::llm::mock::MockLLMProvider;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_select_tool() {
        let mut mock = MockLLMProvider::new("test");
        mock.add_response(LLMResponse::new(
            r#"{"tool_id": "calculator", "confidence": 0.95, "reasoning": "Math calculation needed"}"#,
            FinishReason::Stop,
        ));

        let capability = DefaultLLMCapability::new(Arc::new(mock));

        let context = TaskContext {
            current_state: None,
            available_tools: vec!["calculator".to_string(), "echo".to_string()],
            memory_slots: HashMap::new(),
            recent_messages: vec![],
        };

        let result = capability
            .select_tool(&context, "Calculate 2 + 2")
            .await
            .unwrap();

        assert_eq!(result.tool_id, "calculator");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.reasoning.unwrap(), "Math calculation needed");
    }

    #[tokio::test]
    async fn test_select_tool_with_markdown() {
        let mut mock = MockLLMProvider::new("test");
        mock.add_response(LLMResponse::new(
            r#"Here's my selection:
\`\`\`json
{"tool_id": "calculator", "confidence": 0.9}
\`\`\`"#,
            FinishReason::Stop,
        ));

        let capability = DefaultLLMCapability::new(Arc::new(mock));

        let context = TaskContext {
            current_state: None,
            available_tools: vec!["calculator".to_string()],
            memory_slots: HashMap::new(),
            recent_messages: vec![],
        };

        let result = capability.select_tool(&context, "Do math").await.unwrap();

        assert_eq!(result.tool_id, "calculator");
        assert_eq!(result.confidence, 0.9);
    }

    #[tokio::test]
    async fn test_generate_tool_args() {
        let mut mock = MockLLMProvider::new("test");
        mock.add_response(LLMResponse::new(
            r#"{"expression": "2 + 2"}"#,
            FinishReason::Stop,
        ));

        let capability = DefaultLLMCapability::new(Arc::new(mock));

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            }
        });

        let result = capability
            .generate_tool_args("calculator", "Calculate 2 + 2", &schema)
            .await
            .unwrap();

        assert_eq!(result["expression"], "2 + 2");
    }

    #[tokio::test]
    async fn test_evaluate_yesno() {
        let mut mock = MockLLMProvider::new("test");
        mock.add_response(LLMResponse::new(
            r#"{"answer": true, "reasoning": "User is authenticated in context"}"#,
            FinishReason::Stop,
        ));

        let capability = DefaultLLMCapability::new(Arc::new(mock));

        let context = TaskContext {
            current_state: Some("authenticated".to_string()),
            available_tools: vec![],
            memory_slots: HashMap::new(),
            recent_messages: vec![],
        };

        let (answer, reasoning) = capability
            .evaluate_yesno("Is the user authenticated?", &context)
            .await
            .unwrap();

        assert!(answer);
        assert_eq!(reasoning, "User is authenticated in context");
    }

    #[tokio::test]
    async fn test_classify() {
        let mut mock = MockLLMProvider::new("test");
        mock.add_response(LLMResponse::new(
            r#"{"category": "greeting", "confidence": 0.98}"#,
            FinishReason::Stop,
        ));

        let capability = DefaultLLMCapability::new(Arc::new(mock));

        let categories = vec![
            "greeting".to_string(),
            "question".to_string(),
            "command".to_string(),
        ];

        let (category, confidence) = capability
            .classify("Hello there!", &categories)
            .await
            .unwrap();

        assert_eq!(category, "greeting");
        assert_eq!(confidence, 0.98);
    }

    #[tokio::test]
    async fn test_process_task() {
        let mut mock = MockLLMProvider::new("test");
        mock.add_response(LLMResponse::new(
            "I'll help you with that calculation.",
            FinishReason::Stop,
        ));

        let capability = DefaultLLMCapability::new(Arc::new(mock));

        let context = TaskContext {
            current_state: None,
            available_tools: vec!["calculator".to_string()],
            memory_slots: HashMap::new(),
            recent_messages: vec![ChatMessage {
                timestamp: None,
                role: Role::User,
                content: "What's 2 + 2?".to_string(),
                name: None,
            }],
        };

        let result = capability
            .process_task(&context, "You are a helpful assistant")
            .await
            .unwrap();

        assert_eq!(result.content, "I'll help you with that calculation.");
    }

    #[test]
    fn test_extract_json() {
        let json = r#"{"key": "value"}"#;
        let result = DefaultLLMCapability::extract_json(json).unwrap();
        assert_eq!(result["key"], "value");

        let text = r#"Here is the result: {"key": "value"} done"#;
        let result = DefaultLLMCapability::extract_json(text).unwrap();
        assert_eq!(result["key"], "value");
    }
}
