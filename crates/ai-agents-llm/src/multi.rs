use async_trait::async_trait;
use std::sync::Arc;

use ai_agents_core::{
    ChatMessage, LLMCapability, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider,
    LLMResponse, LLMToolRequest, TaskContext, ToolChoice, ToolSelection,
};

use super::capability::DefaultLLMCapability;

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

    fn is_fallback_eligible(error: &LLMError) -> bool {
        matches!(
            error,
            LLMError::Network(_)
                | LLMError::RateLimit { .. }
                | LLMError::API {
                    status: Some(408 | 429 | 500 | 502 | 503 | 504),
                    ..
                }
        )
    }

    fn should_fallback(&self, selected: &Arc<dyn LLMProvider>, error: &LLMError) -> bool {
        self.enable_fallback
            && !Arc::ptr_eq(selected, &self.primary)
            && !selected.is_terminal_error(error)
            && Self::is_fallback_eligible(error)
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

    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        request: &LLMToolRequest,
    ) -> Result<LLMResponse, LLMError> {
        self.primary
            .complete_with_tools(messages, config, request)
            .await
    }

    fn configured_tool_choice(&self) -> Option<ToolChoice> {
        self.primary.configured_tool_choice()
    }

    fn supports_tool_choice(&self, choice: &ToolChoice) -> bool {
        self.primary.supports_tool_choice(choice)
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

    fn is_terminal_error(&self, error: &LLMError) -> bool {
        self.primary.is_terminal_error(error)
    }
}

#[async_trait]
impl LLMCapability for MultiLLMRouter {
    async fn select_tool(
        &self,
        context: &TaskContext,
        user_input: &str,
    ) -> Result<ToolSelection, LLMError> {
        let selected = self.get_tool_selector();
        let result = DefaultLLMCapability::new(selected.clone())
            .select_tool(context, user_input)
            .await;
        match result {
            Err(error) if self.should_fallback(&selected, &error) => {
                DefaultLLMCapability::new(self.primary.clone())
                    .select_tool(context, user_input)
                    .await
            }
            other => other,
        }
    }

    async fn generate_tool_args(
        &self,
        tool_id: &str,
        user_input: &str,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value, LLMError> {
        let selected = self.get_tool_selector();
        let result = DefaultLLMCapability::new(selected.clone())
            .generate_tool_args(tool_id, user_input, schema)
            .await;
        match result {
            Err(error) if self.should_fallback(&selected, &error) => {
                DefaultLLMCapability::new(self.primary.clone())
                    .generate_tool_args(tool_id, user_input, schema)
                    .await
            }
            other => other,
        }
    }

    async fn evaluate_yesno(
        &self,
        question: &str,
        context: &TaskContext,
    ) -> Result<(bool, String), LLMError> {
        let selected = self.get_guard_evaluator();
        let result = DefaultLLMCapability::new(selected.clone())
            .evaluate_yesno(question, context)
            .await;
        match result {
            Err(error) if self.should_fallback(&selected, &error) => {
                DefaultLLMCapability::new(self.primary.clone())
                    .evaluate_yesno(question, context)
                    .await
            }
            other => other,
        }
    }

    async fn classify(
        &self,
        input: &str,
        categories: &[String],
    ) -> Result<(String, f32), LLMError> {
        let selected = self.get_classifier();
        let result = DefaultLLMCapability::new(selected.clone())
            .classify(input, categories)
            .await;
        match result {
            Err(error) if self.should_fallback(&selected, &error) => {
                DefaultLLMCapability::new(self.primary.clone())
                    .classify(input, categories)
                    .await
            }
            other => other,
        }
    }

    async fn process_task(
        &self,
        context: &TaskContext,
        system_prompt: &str,
    ) -> Result<LLMResponse, LLMError> {
        let capability = DefaultLLMCapability::new(self.primary.clone());
        capability.process_task(context, system_prompt).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockLLMProvider;
    use crate::providers::{ProviderType, UnifiedLLMProvider};
    use ai_agents_core::{FinishReason, LLMToolDefinition, LLMToolRequest, Role, ToolChoice};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy)]
    enum TestFailure {
        Network,
        Config,
        Api(Option<u16>),
    }

    struct CountingProvider {
        name: &'static str,
        response: Option<&'static str>,
        failure: Option<TestFailure>,
        terminal: bool,
        calls: AtomicUsize,
    }

    impl CountingProvider {
        fn success(name: &'static str, response: &'static str) -> Self {
            Self {
                name,
                response: Some(response),
                failure: None,
                terminal: false,
                calls: AtomicUsize::new(0),
            }
        }

        fn failure(name: &'static str, failure: TestFailure, terminal: bool) -> Self {
            Self {
                name,
                response: None,
                failure: Some(failure),
                terminal,
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        fn error(&self) -> LLMError {
            match self.failure.expect("configured failure") {
                TestFailure::Network => LLMError::Network("temporary".into()),
                TestFailure::Config => LLMError::Config("invalid".into()),
                TestFailure::Api(status) => LLMError::API {
                    message: "api failure".into(),
                    status,
                },
            }
        }
    }

    #[async_trait]
    impl LLMProvider for CountingProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, LLMError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.failure.is_some() {
                Err(self.error())
            } else {
                Ok(LLMResponse::new(
                    self.response.unwrap_or_default(),
                    FinishReason::Stop,
                ))
            }
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            Ok(Box::new(futures::stream::empty()))
        }

        fn provider_name(&self) -> &str {
            self.name
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }

        fn is_terminal_error(&self, _error: &LLMError) -> bool {
            self.terminal
        }
    }

    fn empty_task_context() -> TaskContext {
        TaskContext {
            current_state: None,
            available_tools: vec!["calculator".into()],
            memory_slots: HashMap::new(),
            recent_messages: vec![],
        }
    }

    #[derive(Clone, Copy)]
    enum CapabilityOperation {
        SelectTool,
        GenerateToolArgs,
        EvaluateYesNo,
        Classify,
    }

    impl CapabilityOperation {
        const ALL: [Self; 4] = [
            Self::SelectTool,
            Self::GenerateToolArgs,
            Self::EvaluateYesNo,
            Self::Classify,
        ];

        fn valid_response(self) -> &'static str {
            match self {
                Self::SelectTool => r#"{"tool_id":"calculator","confidence":0.9}"#,
                Self::GenerateToolArgs => r#"{"expression":"2+2"}"#,
                Self::EvaluateYesNo => r#"{"answer":true,"reasoning":"safe"}"#,
                Self::Classify => r#"{"category":"greeting","confidence":0.8}"#,
            }
        }

        fn malformed_response(self) -> &'static str {
            match self {
                Self::GenerateToolArgs => "not json",
                _ => "{}",
            }
        }
    }

    fn router_for_operation(
        operation: CapabilityOperation,
        primary: Arc<dyn LLMProvider>,
        specialized: Arc<dyn LLMProvider>,
    ) -> MultiLLMRouter {
        let router = MultiLLMRouter::new(primary);
        match operation {
            CapabilityOperation::SelectTool | CapabilityOperation::GenerateToolArgs => {
                router.with_tool_selector(specialized)
            }
            CapabilityOperation::EvaluateYesNo => router.with_guard_evaluator(specialized),
            CapabilityOperation::Classify => router.with_classifier(specialized),
        }
    }

    async fn invoke_operation(
        router: &MultiLLMRouter,
        operation: CapabilityOperation,
    ) -> Result<(), LLMError> {
        let context = empty_task_context();
        match operation {
            CapabilityOperation::SelectTool => {
                router.select_tool(&context, "calculate").await.map(|_| ())
            }
            CapabilityOperation::GenerateToolArgs => router
                .generate_tool_args("calculator", "calculate", &serde_json::json!({}))
                .await
                .map(|_| ()),
            CapabilityOperation::EvaluateYesNo => {
                router.evaluate_yesno("safe?", &context).await.map(|_| ())
            }
            CapabilityOperation::Classify => router
                .classify("hello", &["greeting".into()])
                .await
                .map(|_| ()),
        }
    }

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
    async fn test_router_delegates_native_tool_methods() {
        let mut primary = MockLLMProvider::new("primary");
        primary.add_response(LLMResponse::new("No call", FinishReason::Stop));
        let history = primary.clone();
        let router = MultiLLMRouter::new(Arc::new(primary));
        let request = LLMToolRequest {
            tools: vec![LLMToolDefinition {
                name: "calculator".to_string(),
                description: "Calculate an expression".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            choice: ToolChoice::Auto,
        };

        router
            .complete_with_tools(&[ChatMessage::user("Hello")], None, &request)
            .await
            .unwrap();

        assert!(router.supports_tool_choice(&ToolChoice::Required));
        assert_eq!(history.last_call().unwrap().request, Some(request));
    }

    #[test]
    fn test_router_delegates_terminal_error_classification_to_primary() {
        let primary = UnifiedLLMProvider::from_spec_config(
            ProviderType::Google,
            "gemini-3.7-flash",
            Some("test-key".to_string()),
            None,
            LLMConfig::default(),
        )
        .unwrap();
        let router = MultiLLMRouter::new(Arc::new(primary));
        assert!(router.is_terminal_error(&LLMError::Serialization(
            "invalid native history".to_string()
        )));
        assert!(!router.is_terminal_error(&LLMError::Network("temporary".to_string())));
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

        let context = TaskContext {
            current_state: None,
            available_tools: vec!["calculator".to_string()],
            memory_slots: HashMap::new(),
            recent_messages: vec![],
        };

        let tool_selection = router.select_tool(&context, "Do math").await.unwrap();
        assert_eq!(tool_selection.tool_id, "calculator");
        assert_eq!(tool_selection.confidence, 0.9);

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

    #[tokio::test]
    async fn transient_failures_fallback_for_all_specialized_capabilities() {
        let context = empty_task_context();

        let primary = Arc::new(CountingProvider::success(
            "primary",
            r#"{"tool_id":"calculator","confidence":0.9}"#,
        ));
        let specialized = Arc::new(CountingProvider::failure(
            "specialized",
            TestFailure::Network,
            false,
        ));
        let router = MultiLLMRouter::new(primary.clone()).with_tool_selector(specialized.clone());
        assert_eq!(
            router
                .select_tool(&context, "calculate")
                .await
                .unwrap()
                .tool_id,
            "calculator"
        );
        assert_eq!(specialized.call_count(), 1);
        assert_eq!(primary.call_count(), 1);

        let primary = Arc::new(CountingProvider::success(
            "primary",
            r#"{"expression":"2+2"}"#,
        ));
        let specialized = Arc::new(CountingProvider::failure(
            "specialized",
            TestFailure::Api(Some(503)),
            false,
        ));
        let router = MultiLLMRouter::new(primary.clone()).with_tool_selector(specialized.clone());
        assert_eq!(
            router
                .generate_tool_args("calculator", "calculate", &serde_json::json!({}))
                .await
                .unwrap()["expression"],
            "2+2"
        );
        assert_eq!(specialized.call_count(), 1);
        assert_eq!(primary.call_count(), 1);

        let primary = Arc::new(CountingProvider::success(
            "primary",
            r#"{"answer":true,"reasoning":"safe"}"#,
        ));
        let specialized = Arc::new(CountingProvider::failure(
            "specialized",
            TestFailure::Api(Some(429)),
            false,
        ));
        let router = MultiLLMRouter::new(primary.clone()).with_guard_evaluator(specialized.clone());
        assert!(router.evaluate_yesno("safe?", &context).await.unwrap().0);
        assert_eq!(specialized.call_count(), 1);
        assert_eq!(primary.call_count(), 1);

        let primary = Arc::new(CountingProvider::success(
            "primary",
            r#"{"category":"greeting","confidence":0.8}"#,
        ));
        let specialized = Arc::new(CountingProvider::failure(
            "specialized",
            TestFailure::Api(Some(500)),
            false,
        ));
        let router = MultiLLMRouter::new(primary.clone()).with_classifier(specialized.clone());
        assert_eq!(
            router
                .classify("hello", &["greeting".into()])
                .await
                .unwrap()
                .0,
            "greeting"
        );
        assert_eq!(specialized.call_count(), 1);
        assert_eq!(primary.call_count(), 1);
    }

    #[tokio::test]
    async fn parsing_config_and_non_transient_api_errors_do_not_fallback() {
        let primary = Arc::new(CountingProvider::success(
            "primary",
            r#"{"category":"greeting","confidence":0.8}"#,
        ));
        let malformed = Arc::new(CountingProvider::success("malformed", "{}"));
        let router = MultiLLMRouter::new(primary.clone()).with_classifier(malformed.clone());
        assert!(matches!(
            router.classify("hello", &["greeting".into()]).await,
            Err(LLMError::Serialization(_))
        ));
        assert_eq!(malformed.call_count(), 1);
        assert_eq!(primary.call_count(), 0);

        for failure in [
            TestFailure::Config,
            TestFailure::Api(Some(400)),
            TestFailure::Api(None),
        ] {
            let primary = Arc::new(CountingProvider::success(
                "primary",
                r#"{"category":"greeting","confidence":0.8}"#,
            ));
            let specialized = Arc::new(CountingProvider::failure("specialized", failure, false));
            let router = MultiLLMRouter::new(primary.clone()).with_classifier(specialized.clone());
            assert!(
                router
                    .classify("hello", &["greeting".into()])
                    .await
                    .is_err()
            );
            assert_eq!(specialized.call_count(), 1);
            assert_eq!(primary.call_count(), 0);
        }
    }

    #[tokio::test]
    async fn terminal_or_disabled_fallback_preserves_the_specialized_error() {
        let context = empty_task_context();
        let primary = Arc::new(CountingProvider::success(
            "primary",
            r#"{"answer":true,"reasoning":"safe"}"#,
        ));
        let terminal = Arc::new(CountingProvider::failure(
            "terminal",
            TestFailure::Network,
            true,
        ));
        let router = MultiLLMRouter::new(primary.clone()).with_guard_evaluator(terminal.clone());
        assert!(matches!(
            router.evaluate_yesno("safe?", &context).await,
            Err(LLMError::Network(_))
        ));
        assert_eq!(primary.call_count(), 0);

        let disabled = Arc::new(CountingProvider::failure(
            "disabled",
            TestFailure::Network,
            false,
        ));
        let router = MultiLLMRouter::new(primary.clone())
            .with_classifier(disabled.clone())
            .with_fallback(false);
        assert!(matches!(
            router.classify("hello", &["greeting".into()]).await,
            Err(LLMError::Network(_))
        ));
        assert_eq!(primary.call_count(), 0);
    }

    #[tokio::test]
    async fn every_specialized_capability_preserves_non_fallback_contracts() {
        for operation in CapabilityOperation::ALL {
            let primary = Arc::new(CountingProvider::success(
                "primary",
                operation.valid_response(),
            ));
            let malformed = Arc::new(CountingProvider::success(
                "malformed",
                operation.malformed_response(),
            ));
            let router = router_for_operation(operation, primary.clone(), malformed.clone());
            assert!(matches!(
                invoke_operation(&router, operation).await,
                Err(LLMError::Serialization(_))
            ));
            assert_eq!(malformed.call_count(), 1);
            assert_eq!(primary.call_count(), 0);

            let primary = Arc::new(CountingProvider::success(
                "primary",
                operation.valid_response(),
            ));
            let terminal = Arc::new(CountingProvider::failure(
                "terminal",
                TestFailure::Network,
                true,
            ));
            let router = router_for_operation(operation, primary.clone(), terminal.clone());
            assert!(matches!(
                invoke_operation(&router, operation).await,
                Err(LLMError::Network(_))
            ));
            assert_eq!(terminal.call_count(), 1);
            assert_eq!(primary.call_count(), 0);

            let primary = Arc::new(CountingProvider::success(
                "primary",
                operation.valid_response(),
            ));
            let disabled = Arc::new(CountingProvider::failure(
                "disabled",
                TestFailure::Network,
                false,
            ));
            let router = router_for_operation(operation, primary.clone(), disabled.clone())
                .with_fallback(false);
            assert!(matches!(
                invoke_operation(&router, operation).await,
                Err(LLMError::Network(_))
            ));
            assert_eq!(disabled.call_count(), 1);
            assert_eq!(primary.call_count(), 0);

            let primary = Arc::new(CountingProvider::success(
                "primary",
                operation.valid_response(),
            ));
            let router = router_for_operation(operation, primary.clone(), primary.clone());
            invoke_operation(&router, operation).await.unwrap();
            assert_eq!(primary.call_count(), 1);

            let primary = Arc::new(CountingProvider::failure(
                "primary",
                TestFailure::Config,
                false,
            ));
            let specialized = Arc::new(CountingProvider::failure(
                "specialized",
                TestFailure::Network,
                false,
            ));
            let router = router_for_operation(operation, primary.clone(), specialized.clone());
            assert!(matches!(
                invoke_operation(&router, operation).await,
                Err(LLMError::Config(_))
            ));
            assert_eq!(specialized.call_count(), 1);
            assert_eq!(primary.call_count(), 1);
        }
    }

    #[test]
    fn fallback_eligibility_is_an_explicit_transient_allowlist() {
        assert!(MultiLLMRouter::is_fallback_eligible(&LLMError::Network(
            "temporary".into()
        )));
        assert!(MultiLLMRouter::is_fallback_eligible(&LLMError::RateLimit {
            retry_after: None,
        }));
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(MultiLLMRouter::is_fallback_eligible(&LLMError::API {
                message: "temporary".into(),
                status: Some(status),
            }));
        }
        for status in [400, 401, 403, 404] {
            assert!(!MultiLLMRouter::is_fallback_eligible(&LLMError::API {
                message: "permanent".into(),
                status: Some(status),
            }));
        }
        assert!(!MultiLLMRouter::is_fallback_eligible(&LLMError::API {
            message: "unknown".into(),
            status: None,
        }));
        assert!(!MultiLLMRouter::is_fallback_eligible(
            &LLMError::Serialization("invalid JSON".into())
        ));
    }

    #[tokio::test]
    async fn the_same_specialized_and_primary_arc_is_called_once() {
        let primary = Arc::new(CountingProvider::success(
            "primary",
            r#"{"category":"greeting","confidence":0.8}"#,
        ));
        let router = MultiLLMRouter::new(primary.clone()).with_classifier(primary.clone());

        assert_eq!(
            router
                .classify("hello", &["greeting".into()])
                .await
                .unwrap()
                .0,
            "greeting"
        );
        assert_eq!(primary.call_count(), 1);
    }

    #[tokio::test]
    async fn primary_fallback_error_is_returned_without_more_attempts() {
        let primary = Arc::new(CountingProvider::failure(
            "primary",
            TestFailure::Config,
            false,
        ));
        let specialized = Arc::new(CountingProvider::failure(
            "specialized",
            TestFailure::Network,
            false,
        ));
        let router = MultiLLMRouter::new(primary.clone()).with_classifier(specialized.clone());

        assert!(matches!(
            router.classify("hello", &["greeting".into()]).await,
            Err(LLMError::Config(_))
        ));
        assert_eq!(specialized.call_count(), 1);
        assert_eq!(primary.call_count(), 1);
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
