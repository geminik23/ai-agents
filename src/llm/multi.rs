use async_trait::async_trait;
use std::sync::Arc;

use crate::llm::{
    ChatMessage, LLMCapability, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider,
    LLMResponse, TaskContext, ToolSelection,
};

#[derive(Clone)]
pub struct MultiLLMRouter {
    primary: Arc<dyn LLMProvider>,
    tool_selector: Option<Arc<dyn LLMProvider>>,
    guard_evaluator: Option<Arc<dyn LLMProvider>>,
    classifier: Option<Arc<dyn LLMProvider>>,
    enable_fallback: bool,
}

impl MultiLLMRouter {
    pub fn new(primary: Arc<dyn LLMProvider>) -> Self {
        Self {
            primary,
            tool_selector: None,
            guard_evaluator: None,
            classifier: None,
            enable_fallback: true,
        }
    }

    pub fn with_tool_selector(mut self, provider: Arc<dyn LLMProvider>) -> Self {
        self.tool_selector = Some(provider);
        self
    }

    pub fn with_guard_evaluator(mut self, provider: Arc<dyn LLMProvider>) -> Self {
        self.guard_evaluator = Some(provider);
        self
    }

    pub fn with_classifier(mut self, provider: Arc<dyn LLMProvider>) -> Self {
        self.classifier = Some(provider);
        self
    }

    pub fn with_fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }

    fn get_tool_selector(&self) -> Arc<dyn LLMProvider> {
        self.tool_selector
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.primary.clone())
    }

    fn get_guard_evaluator(&self) -> Arc<dyn LLMProvider> {
        self.guard_evaluator
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.primary.clone())
    }

    fn get_classifier(&self) -> Arc<dyn LLMProvider> {
        self.classifier
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.primary.clone())
    }

    async fn execute_with_fallback<F, Fut, T>(
        &self,
        primary_fn: F,
        fallback_provider: Arc<dyn LLMProvider>,
        operation: &str,
    ) -> Result<T, LLMError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, LLMError>>,
    {
        match primary_fn().await {
            Ok(result) => Ok(result),
            Err(e) if self.enable_fallback => {
                eprintln!(
                    "Multi-LLM: {} failed with specialized provider, falling back to primary: {}",
                    operation, e
                );

                // TODO: In future implementations, retry with fallback_provider
                // For now, just return the error since we can't generically retry
                Err(e)
            }
            Err(e) => Err(e),
        }
    }
}

#[async_trait]
impl LLMProvider for MultiLLMRouter {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        self.primary.complete(messages, config).await
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        self.primary.complete_stream(messages, config).await
    }

    fn provider_name(&self) -> &str {
        "multi-llm-router"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        self.primary.supports(feature)
    }
}

#[async_trait]
impl LLMCapability for MultiLLMRouter {
    async fn select_tool(
        &self,
        context: &TaskContext,
        user_input: &str,
    ) -> Result<ToolSelection, LLMError> {
        use crate::llm::capability::DefaultLLMCapability;

        let provider = self.get_tool_selector();
        let capability = DefaultLLMCapability::new(provider);
        capability.select_tool(context, user_input).await
    }

    async fn generate_tool_args(
        &self,
        tool_id: &str,
        user_input: &str,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value, LLMError> {
        use crate::llm::capability::DefaultLLMCapability;

        // Tool args generation can use the specialized tool selector or primary
        let provider = self.get_tool_selector();
        let capability = DefaultLLMCapability::new(provider);
        capability
            .generate_tool_args(tool_id, user_input, schema)
            .await
    }

    async fn evaluate_yesno(
        &self,
        question: &str,
        context: &TaskContext,
    ) -> Result<(bool, String), LLMError> {
        use crate::llm::capability::DefaultLLMCapability;

        let provider = self.get_guard_evaluator();
        let capability = DefaultLLMCapability::new(provider);
        capability.evaluate_yesno(question, context).await
    }

    async fn classify(
        &self,
        input: &str,
        categories: &[String],
    ) -> Result<(String, f32), LLMError> {
        use crate::llm::capability::DefaultLLMCapability;

        let provider = self.get_classifier();
        let capability = DefaultLLMCapability::new(provider);
        capability.classify(input, categories).await
    }

    async fn process_task(
        &self,
        context: &TaskContext,
        system_prompt: &str,
    ) -> Result<LLMResponse, LLMError> {
        use crate::llm::capability::DefaultLLMCapability;

        // Always use primary for main task processing
        let capability = DefaultLLMCapability::new(self.primary.clone());
        capability.process_task(context, system_prompt).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::mock::MockLLMProvider;
    use crate::llm::{ChatMessage, FinishReason, Role, TaskContext};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_router_with_primary_only() {
        let mut primary = MockLLMProvider::new("primary");
        primary.add_response(LLMResponse::new("Hello from primary", FinishReason::Stop));

        let router = MultiLLMRouter::new(Arc::new(primary));

        let messages = vec![ChatMessage {
            timestamp: None,
            role: Role::User,
            content: "Test".to_string(),
            name: None,
        }];

        let response = router.complete(&messages, None).await.unwrap();
        assert_eq!(response.content, "Hello from primary");
    }

    #[tokio::test]
    async fn test_router_with_specialized_providers() {
        let mut primary = MockLLMProvider::new("primary");
        primary.add_response(LLMResponse::new("Primary response", FinishReason::Stop));

        let mut tool_selector = MockLLMProvider::new("tool-selector");
        tool_selector.add_response(LLMResponse::new(
            r#"{"tool_id": "calculator", "confidence": 0.9}"#,
            FinishReason::Stop,
        ));

        let mut guard = MockLLMProvider::new("guard");
        guard.add_response(LLMResponse::new(
            r#"{"answer": true, "reasoning": "Approved"}"#,
            FinishReason::Stop,
        ));

        let router = MultiLLMRouter::new(Arc::new(primary))
            .with_tool_selector(Arc::new(tool_selector))
            .with_guard_evaluator(Arc::new(guard));

        // Test tool selection uses specialized provider
        let context = TaskContext {
            current_state: None,
            available_tools: vec!["calculator".to_string()],
            memory_slots: HashMap::new(),
            recent_messages: vec![],
        };

        let tool_selection = router.select_tool(&context, "Do math").await.unwrap();
        assert_eq!(tool_selection.tool_id, "calculator");
        assert_eq!(tool_selection.confidence, 0.9);

        // Test guard evaluation uses specialized provider
        let (answer, reasoning) = router
            .evaluate_yesno("Is it safe?", &context)
            .await
            .unwrap();
        assert!(answer);
        assert_eq!(reasoning, "Approved");
    }

    #[tokio::test]
    async fn test_router_fallback_to_primary() {
        let mut primary = MockLLMProvider::new("primary");
        primary.add_response(LLMResponse::new("Primary response", FinishReason::Stop));

        let router = MultiLLMRouter::new(Arc::new(primary)).with_fallback(true);

        // When no specialized provider is set, should use primary
        let messages = vec![ChatMessage {
            timestamp: None,
            role: Role::User,
            content: "Test".to_string(),
            name: None,
        }];

        let response = router.complete(&messages, None).await.unwrap();
        assert_eq!(response.content, "Primary response");
    }

    #[tokio::test]
    async fn test_router_provider_name() {
        let primary = MockLLMProvider::new("primary");
        let router = MultiLLMRouter::new(Arc::new(primary));

        assert_eq!(router.provider_name(), "multi-llm-router");
    }

    #[tokio::test]
    async fn test_router_supports() {
        let mut primary = MockLLMProvider::new("primary");
        primary.set_feature_support(LLMFeature::Streaming, true);

        let router = MultiLLMRouter::new(Arc::new(primary));

        assert!(router.supports(LLMFeature::Streaming));
    }

    #[tokio::test]
    async fn test_classify_with_specialized_provider() {
        let primary = MockLLMProvider::new("primary");

        let mut classifier = MockLLMProvider::new("classifier");
        classifier.add_response(LLMResponse::new(
            r#"{"category": "greeting", "confidence": 0.95}"#,
            FinishReason::Stop,
        ));

        let router = MultiLLMRouter::new(Arc::new(primary)).with_classifier(Arc::new(classifier));

        let categories = vec!["greeting".to_string(), "question".to_string()];
        let (category, confidence) = router.classify("Hello!", &categories).await.unwrap();

        assert_eq!(category, "greeting");
        assert_eq!(confidence, 0.95);
    }

    #[tokio::test]
    async fn test_process_task_uses_primary() {
        let mut primary = MockLLMProvider::new("primary");
        primary.add_response(LLMResponse::new(
            "Task processed by primary",
            FinishReason::Stop,
        ));

        let tool_selector = MockLLMProvider::new("tool-selector");

        let router =
            MultiLLMRouter::new(Arc::new(primary)).with_tool_selector(Arc::new(tool_selector));

        let context = TaskContext {
            current_state: None,
            available_tools: vec![],
            memory_slots: HashMap::new(),
            recent_messages: vec![],
        };

        let response = router
            .process_task(&context, "System prompt")
            .await
            .unwrap();

        // Should use primary, not tool selector
        assert_eq!(response.content, "Task processed by primary");
    }

    #[tokio::test]
    async fn test_generate_tool_args_with_specialized() {
        let primary = MockLLMProvider::new("primary");

        let mut tool_selector = MockLLMProvider::new("tool-selector");
        tool_selector.add_response(LLMResponse::new(
            r#"{"expression": "2 + 2"}"#,
            FinishReason::Stop,
        ));

        let router =
            MultiLLMRouter::new(Arc::new(primary)).with_tool_selector(Arc::new(tool_selector));

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            }
        });

        let result = router
            .generate_tool_args("calculator", "Calculate 2 + 2", &schema)
            .await
            .unwrap();

        assert_eq!(result["expression"], "2 + 2");
    }

    #[test]
    fn test_builder_pattern() {
        let primary = MockLLMProvider::new("primary");
        let tool_selector = MockLLMProvider::new("tool-selector");
        let guard = MockLLMProvider::new("guard");
        let classifier = MockLLMProvider::new("classifier");

        let router = MultiLLMRouter::new(Arc::new(primary))
            .with_tool_selector(Arc::new(tool_selector))
            .with_guard_evaluator(Arc::new(guard))
            .with_classifier(Arc::new(classifier))
            .with_fallback(false);

        assert_eq!(router.provider_name(), "multi-llm-router");
        assert!(!router.enable_fallback);
    }
}
