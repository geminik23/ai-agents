use serde_json::Value;
use std::collections::HashMap;

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::{ChatMessage, LLMRegistry};

use super::config::{
    ApprovalMessage, LlmGenerateConfig, MessageLanguageConfig, MessageLanguageStrategy,
    ToolApprovalConfig,
};
use super::handler::ApprovalHandler;

pub struct MessageResolver<'a> {
    global_config: &'a MessageLanguageConfig,
    llm_registry: Option<&'a LLMRegistry>,
}

impl<'a> MessageResolver<'a> {
    pub fn new(global_config: &'a MessageLanguageConfig) -> Self {
        Self {
            global_config,
            llm_registry: None,
        }
    }

    pub fn with_llm_registry(mut self, registry: &'a LLMRegistry) -> Self {
        self.llm_registry = Some(registry);
        self
    }

    pub async fn resolve(
        &self,
        approval_message: &ApprovalMessage,
        local_config: Option<&MessageLanguageConfig>,
        context: &HashMap<String, Value>,
        handler: &dyn ApprovalHandler,
    ) -> Result<String> {
        let effective_config = local_config.unwrap_or(self.global_config);
        let strategies = std::iter::once(effective_config.strategy.clone())
            .chain(effective_config.fallback.iter().cloned());

        for strategy in strategies {
            if let Some(message) = self
                .try_strategy(
                    strategy,
                    approval_message,
                    effective_config,
                    context,
                    handler,
                )
                .await?
            {
                return Ok(render_template(&message, context));
            }
        }

        Ok(approval_message
            .get_any()
            .or_else(|| approval_message.description().map(String::from))
            .unwrap_or_else(|| "Approval required".to_string()))
    }

    async fn try_strategy(
        &self,
        strategy: MessageLanguageStrategy,
        approval_message: &ApprovalMessage,
        config: &MessageLanguageConfig,
        context: &HashMap<String, Value>,
        handler: &dyn ApprovalHandler,
    ) -> Result<Option<String>> {
        match strategy {
            MessageLanguageStrategy::Auto => Ok(None),

            MessageLanguageStrategy::Approver => {
                if let Some(lang) = handler.preferred_language() {
                    return Ok(approval_message.get(&lang));
                }
                Ok(None)
            }

            MessageLanguageStrategy::User => {
                let lang = get_user_language(context);
                if let Some(lang) = lang {
                    return Ok(approval_message.get(&lang));
                }
                Ok(None)
            }

            MessageLanguageStrategy::Explicit => {
                if let Some(ref lang) = config.explicit {
                    return Ok(approval_message.get(lang));
                }
                Ok(None)
            }

            MessageLanguageStrategy::LlmGenerate => {
                if let Some(registry) = self.llm_registry {
                    let message = self
                        .generate_message_with_llm(
                            approval_message,
                            config.llm_generate.as_ref(),
                            context,
                            handler,
                            registry,
                        )
                        .await?;
                    return Ok(Some(message));
                }
                Ok(None)
            }
        }
    }

    async fn generate_message_with_llm(
        &self,
        approval_message: &ApprovalMessage,
        llm_config: Option<&LlmGenerateConfig>,
        context: &HashMap<String, Value>,
        handler: &dyn ApprovalHandler,
        registry: &LLMRegistry,
    ) -> Result<String> {
        let config = llm_config.cloned().unwrap_or_default();

        let target_lang = handler
            .preferred_language()
            .or_else(|| get_user_language(context))
            .unwrap_or_else(|| "English".to_string());

        let description = approval_message
            .description()
            .map(String::from)
            .or_else(|| approval_message.get_any())
            .unwrap_or_else(|| "Approval required for this action".to_string());

        let context_str = if config.include_context && !context.is_empty() {
            format!(
                "\nContext: {}",
                serde_json::to_string_pretty(context).unwrap_or_default()
            )
        } else {
            String::new()
        };

        let prompt = format!(
            "Generate a clear, concise approval request message in {}.\n\
             Action: {}{}\n\
             Requirements:\n\
             - Keep it under 100 words\n\
             - Be direct and professional\n\
             - Include any relevant context values using {{ key }} placeholders if applicable\n\
             - Output only the message text, no explanations",
            target_lang, description, context_str
        );

        let llm = registry.get(&config.llm).map_err(|e| {
            AgentError::Other(format!("Failed to get LLM for message generation: {}", e))
        })?;

        let response = llm
            .complete(&[ChatMessage::user(&prompt)], None)
            .await
            .map_err(|e| AgentError::Other(format!("LLM generation failed: {}", e)))?;

        Ok(response.content.trim().to_string())
    }
}

fn get_user_language(context: &HashMap<String, Value>) -> Option<String> {
    context
        .get("user.language")
        .and_then(|v| v.as_str())
        .or_else(|| {
            context
                .get("input.detected.language")
                .and_then(|v| v.as_str())
        })
        .or_else(|| context.get("language").and_then(|v| v.as_str()))
        .map(String::from)
}

fn render_template(template: &str, context: &HashMap<String, Value>) -> String {
    let mut result = template.to_string();

    for (key, value) in context {
        let placeholder = format!("{{{{ {} }}}}", key);
        let replacement = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".to_string(),
            _ => serde_json::to_string(value).unwrap_or_default(),
        };
        result = result.replace(&placeholder, &replacement);
    }

    result
}

pub fn resolve_best_language(
    approval_message: &ApprovalMessage,
    handler: &dyn ApprovalHandler,
    context: &HashMap<String, Value>,
) -> Option<String> {
    if let Some(lang) = handler.preferred_language() {
        if approval_message.get(&lang).is_some() {
            return Some(lang);
        }
    }

    if let Some(lang) = get_user_language(context) {
        if approval_message.get(&lang).is_some() {
            return Some(lang);
        }
    }

    if approval_message.get("en").is_some() {
        return Some("en".to_string());
    }

    approval_message.available_languages().into_iter().next()
}

pub async fn resolve_tool_message(
    tool_config: &ToolApprovalConfig,
    tool_name: &str,
    global_config: &MessageLanguageConfig,
    context: &HashMap<String, Value>,
    handler: &dyn ApprovalHandler,
    llm_registry: Option<&LLMRegistry>,
) -> Result<String> {
    if tool_config.approval_message.is_empty() {
        return Ok(format!("Approve execution of tool '{}'?", tool_name));
    }

    let mut resolver = MessageResolver::new(global_config);
    if let Some(registry) = llm_registry {
        resolver = resolver.with_llm_registry(registry);
    }

    resolver
        .resolve(
            &tool_config.approval_message,
            tool_config.message_language.as_ref(),
            context,
            handler,
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ApprovalResult;

    struct TestHandler {
        language: Option<String>,
    }

    impl TestHandler {
        fn new() -> Self {
            Self { language: None }
        }

        fn with_language(language: impl Into<String>) -> Self {
            Self {
                language: Some(language.into()),
            }
        }
    }

    #[async_trait::async_trait]
    impl ApprovalHandler for TestHandler {
        async fn request_approval(
            &self,
            _request: crate::types::ApprovalRequest,
        ) -> ApprovalResult {
            ApprovalResult::Approved
        }

        fn preferred_language(&self) -> Option<String> {
            self.language.clone()
        }
    }

    #[test]
    fn test_render_template_basic() {
        let mut context = HashMap::new();
        context.insert("amount".to_string(), Value::Number(1000.into()));
        context.insert("currency".to_string(), Value::String("USD".to_string()));

        let template = "Approve {{ amount }} {{ currency }}?";
        let result = render_template(template, &context);
        assert_eq!(result, "Approve 1000 USD?");
    }

    #[test]
    fn test_render_template_no_placeholders() {
        let context = HashMap::new();
        let template = "Simple message";
        let result = render_template(template, &context);
        assert_eq!(result, "Simple message");
    }

    #[test]
    fn test_get_user_language_from_user_language() {
        let mut context = HashMap::new();
        context.insert("user.language".to_string(), Value::String("ko".to_string()));

        assert_eq!(get_user_language(&context), Some("ko".to_string()));
    }

    #[test]
    fn test_get_user_language_from_detected() {
        let mut context = HashMap::new();
        context.insert(
            "input.detected.language".to_string(),
            Value::String("ja".to_string()),
        );

        assert_eq!(get_user_language(&context), Some("ja".to_string()));
    }

    #[test]
    fn test_get_user_language_priority() {
        let mut context = HashMap::new();
        context.insert("user.language".to_string(), Value::String("ko".to_string()));
        context.insert(
            "input.detected.language".to_string(),
            Value::String("ja".to_string()),
        );

        assert_eq!(get_user_language(&context), Some("ko".to_string()));
    }

    #[test]
    fn test_get_user_language_none() {
        let context = HashMap::new();
        assert_eq!(get_user_language(&context), None);
    }

    #[tokio::test]
    async fn test_resolve_approver_strategy() {
        let config = MessageLanguageConfig {
            strategy: MessageLanguageStrategy::Approver,
            fallback: vec![],
            explicit: None,
            llm_generate: None,
        };

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English message".to_string());
        messages.insert("ko".to_string(), "한국어 메시지".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::with_language("ko");
        let context = HashMap::new();

        let resolver = MessageResolver::new(&config);
        let result = resolver
            .resolve(&approval_message, None, &context, &handler)
            .await
            .unwrap();

        assert_eq!(result, "한국어 메시지");
    }

    #[tokio::test]
    async fn test_resolve_user_strategy() {
        let config = MessageLanguageConfig {
            strategy: MessageLanguageStrategy::User,
            fallback: vec![],
            explicit: None,
            llm_generate: None,
        };

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English message".to_string());
        messages.insert("ja".to_string(), "日本語メッセージ".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::new();
        let mut context = HashMap::new();
        context.insert("user.language".to_string(), Value::String("ja".to_string()));

        let resolver = MessageResolver::new(&config);
        let result = resolver
            .resolve(&approval_message, None, &context, &handler)
            .await
            .unwrap();

        assert_eq!(result, "日本語メッセージ");
    }

    #[tokio::test]
    async fn test_resolve_explicit_strategy() {
        let config = MessageLanguageConfig {
            strategy: MessageLanguageStrategy::Explicit,
            fallback: vec![],
            explicit: Some("en".to_string()),
            llm_generate: None,
        };

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English message".to_string());
        messages.insert("ko".to_string(), "한국어 메시지".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::with_language("ko");
        let context = HashMap::new();

        let resolver = MessageResolver::new(&config);
        let result = resolver
            .resolve(&approval_message, None, &context, &handler)
            .await
            .unwrap();

        assert_eq!(result, "English message");
    }

    #[tokio::test]
    async fn test_resolve_fallback_chain() {
        let config = MessageLanguageConfig {
            strategy: MessageLanguageStrategy::Approver,
            fallback: vec![
                MessageLanguageStrategy::User,
                MessageLanguageStrategy::Explicit,
            ],
            explicit: Some("en".to_string()),
            llm_generate: None,
        };

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English fallback".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::with_language("ko");
        let context = HashMap::new();

        let resolver = MessageResolver::new(&config);
        let result = resolver
            .resolve(&approval_message, None, &context, &handler)
            .await
            .unwrap();

        assert_eq!(result, "English fallback");
    }

    #[tokio::test]
    async fn test_resolve_simple_message() {
        let config = MessageLanguageConfig::default();
        let approval_message = ApprovalMessage::simple("Simple approval message");

        let handler = TestHandler::with_language("ko");
        let context = HashMap::new();

        let resolver = MessageResolver::new(&config);
        let result = resolver
            .resolve(&approval_message, None, &context, &handler)
            .await
            .unwrap();

        assert_eq!(result, "Simple approval message");
    }

    #[tokio::test]
    async fn test_resolve_with_template() {
        let config = MessageLanguageConfig {
            strategy: MessageLanguageStrategy::Explicit,
            fallback: vec![],
            explicit: Some("en".to_string()),
            llm_generate: None,
        };

        let mut messages = HashMap::new();
        messages.insert(
            "en".to_string(),
            "Approve {{ amount }} {{ currency }}?".to_string(),
        );
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::new();
        let mut context = HashMap::new();
        context.insert("amount".to_string(), Value::Number(500.into()));
        context.insert("currency".to_string(), Value::String("USD".to_string()));

        let resolver = MessageResolver::new(&config);
        let result = resolver
            .resolve(&approval_message, None, &context, &handler)
            .await
            .unwrap();

        assert_eq!(result, "Approve 500 USD?");
    }

    #[tokio::test]
    async fn test_resolve_local_config_override() {
        let global_config = MessageLanguageConfig {
            strategy: MessageLanguageStrategy::Approver,
            fallback: vec![],
            explicit: None,
            llm_generate: None,
        };

        let local_config = MessageLanguageConfig {
            strategy: MessageLanguageStrategy::Explicit,
            fallback: vec![],
            explicit: Some("ja".to_string()),
            llm_generate: None,
        };

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English".to_string());
        messages.insert("ja".to_string(), "日本語".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::with_language("en");
        let context = HashMap::new();

        let resolver = MessageResolver::new(&global_config);
        let result = resolver
            .resolve(&approval_message, Some(&local_config), &context, &handler)
            .await
            .unwrap();

        assert_eq!(result, "日本語");
    }

    #[test]
    fn test_resolve_best_language_approver_preferred() {
        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English".to_string());
        messages.insert("ko".to_string(), "한국어".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::with_language("ko");
        let context = HashMap::new();

        let result = resolve_best_language(&approval_message, &handler, &context);
        assert_eq!(result, Some("ko".to_string()));
    }

    #[test]
    fn test_resolve_best_language_user_fallback() {
        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English".to_string());
        messages.insert("ja".to_string(), "日本語".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::with_language("ko");
        let mut context = HashMap::new();
        context.insert("user.language".to_string(), Value::String("ja".to_string()));

        let result = resolve_best_language(&approval_message, &handler, &context);
        assert_eq!(result, Some("ja".to_string()));
    }

    #[test]
    fn test_resolve_best_language_english_default() {
        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "English".to_string());
        messages.insert("fr".to_string(), "Français".to_string());
        let approval_message = ApprovalMessage::multi_language(messages);

        let handler = TestHandler::with_language("ko");
        let context = HashMap::new();

        let result = resolve_best_language(&approval_message, &handler, &context);
        assert_eq!(result, Some("en".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_tool_message_empty() {
        let tool_config = ToolApprovalConfig::default();
        let global_config = MessageLanguageConfig::default();
        let context = HashMap::new();
        let handler = TestHandler::new();

        let result = resolve_tool_message(
            &tool_config,
            "test_tool",
            &global_config,
            &context,
            &handler,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result, "Approve execution of tool 'test_tool'?");
    }

    #[tokio::test]
    async fn test_resolve_tool_message_with_config() {
        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "Approve test?".to_string());

        let tool_config = ToolApprovalConfig {
            require_approval: true,
            approval_context: vec![],
            approval_message: ApprovalMessage::multi_language(messages),
            message_language: Some(MessageLanguageConfig {
                strategy: MessageLanguageStrategy::Explicit,
                fallback: vec![],
                explicit: Some("en".to_string()),
                llm_generate: None,
            }),
            timeout_seconds: None,
        };

        let global_config = MessageLanguageConfig::default();
        let context = HashMap::new();
        let handler = TestHandler::new();

        let result = resolve_tool_message(
            &tool_config,
            "test_tool",
            &global_config,
            &context,
            &handler,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result, "Approve test?");
    }
}
