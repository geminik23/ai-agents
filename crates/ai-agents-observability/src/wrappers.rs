use crate::event::{
    EventType, ObservationError, ObservationPurpose, ObservationTokenUsage, TokenUsageSource,
};
use crate::manager::ObservabilityManager;
use crate::span::SpanGuard;
use ai_agents_core::{
    ChatMessage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    LLMToolRequest, Tool, ToolCallClassification, ToolChoice, ToolExecutionContext,
    ToolPolicyBindings, ToolResult, ToolSafetyMetadata,
};
use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// LLMProvider wrapper that measures calls while preserving the inner provider API.
pub struct ObservedLLMProvider {
    inner: Arc<dyn LLMProvider>,
    manager: Arc<ObservabilityManager>,
    alias: Option<String>,
    provider: String,
    model: String,
}

impl ObservedLLMProvider {
    /// Creates a wrapper for one registry alias and model identity.
    pub fn new(
        inner: Arc<dyn LLMProvider>,
        manager: Arc<ObservabilityManager>,
        alias: Option<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            manager,
            alias,
            provider: provider.into(),
            model: model.into(),
        }
    }

    fn event_type(&self, streaming: bool) -> EventType {
        EventType::LlmCall {
            provider: self.provider.clone(),
            model: self.model.clone(),
            alias: self.alias.clone(),
            streaming,
        }
    }
}

#[async_trait]
impl LLMProvider for ObservedLLMProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        if !self.manager.config().latency.track_llm {
            return self.inner.complete(messages, config).await;
        }
        let mut span = self
            .manager
            .start_span(self.event_type(false), current_purpose());
        if self.manager.config().privacy.include_prompts {
            span.set_payload(serde_json::json!({"messages": messages}));
        } else if self.manager.config().privacy.hash_inputs {
            let text = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            span.set_payload(
                serde_json::json!({"input": self.manager.redactor().redact_text(&text)}),
            );
        }

        match self.inner.complete(messages, config).await {
            Ok(response) => {
                if let Some(tokens) = response.usage {
                    span.set_tokens(ObservationTokenUsage::new(
                        tokens.prompt_tokens as u64,
                        tokens.completion_tokens as u64,
                        TokenUsageSource::Provider,
                    ));
                } else if self.manager.config().tokens.estimate_when_missing {
                    span.set_tokens(estimate_usage(messages, &response.content));
                }
                if self.manager.config().privacy.include_responses {
                    span.set_payload(serde_json::json!({"response": response.content}));
                }
                Ok(response)
            }
            Err(error) => {
                span.set_error(ObservationError::new(
                    llm_error_kind(&error),
                    error.to_string(),
                ));
                Err(error)
            }
        }
    }

    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        request: &LLMToolRequest,
    ) -> std::result::Result<LLMResponse, LLMError> {
        if !self.manager.config().latency.track_llm {
            return self
                .inner
                .complete_with_tools(messages, config, request)
                .await;
        }
        let mut span = self
            .manager
            .start_span(self.event_type(false), current_purpose());
        if self.manager.config().privacy.include_prompts {
            span.set_payload(serde_json::json!({"messages": messages}));
        } else if self.manager.config().privacy.hash_inputs {
            let text = messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            span.set_payload(
                serde_json::json!({"input": self.manager.redactor().redact_text(&text)}),
            );
        }

        match self
            .inner
            .complete_with_tools(messages, config, request)
            .await
        {
            Ok(response) => {
                if let Some(tokens) = response.usage {
                    span.set_tokens(ObservationTokenUsage::new(
                        tokens.prompt_tokens as u64,
                        tokens.completion_tokens as u64,
                        TokenUsageSource::Provider,
                    ));
                } else if self.manager.config().tokens.estimate_when_missing {
                    span.set_tokens(estimate_usage(messages, &response.content));
                }
                if self.manager.config().privacy.include_responses {
                    span.set_payload(serde_json::json!({"response": response.content}));
                }
                Ok(response)
            }
            Err(error) => {
                span.set_error(ObservationError::new(
                    llm_error_kind(&error),
                    error.to_string(),
                ));
                Err(error)
            }
        }
    }

    fn configured_tool_choice(&self) -> Option<ToolChoice> {
        self.inner.configured_tool_choice()
    }

    fn supports_tool_choice(&self, choice: &ToolChoice) -> bool {
        self.inner.supports_tool_choice(choice)
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        if !self.manager.config().latency.track_llm {
            return self.inner.complete_stream(messages, config).await;
        }
        let mut span = self
            .manager
            .start_span(self.event_type(true), current_purpose());
        let estimated_input_tokens = estimate_messages(messages);
        let inner = match self.inner.complete_stream(messages, config).await {
            Ok(stream) => stream,
            Err(error) => {
                span.set_error(ObservationError::new(
                    llm_error_kind(&error),
                    error.to_string(),
                ));
                return Err(error);
            }
        };
        Ok(Box::new(ObservedLLMStream::new(
            inner,
            span,
            estimated_input_tokens,
            self.manager.config().tokens.estimate_when_missing,
        )))
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        self.inner.supports(feature)
    }
}

/// Tool wrapper that measures execution without changing tool identity or schema.
pub struct ObservedTool {
    inner: Arc<dyn Tool>,
    manager: Arc<ObservabilityManager>,
}

impl ObservedTool {
    /// Creates a wrapper around an already registered tool.
    pub fn new(inner: Arc<dyn Tool>, manager: Arc<ObservabilityManager>) -> Self {
        Self { inner, manager }
    }
}

#[async_trait]
impl Tool for ObservedTool {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        self.inner.safety_metadata()
    }

    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        self.inner.classify_call(args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        self.inner.policy_bindings()
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        if !self.manager.config().latency.track_tools {
            return self.inner.execute(args, ctx).await;
        }
        let mut span = self.manager.start_span(
            EventType::ToolCall {
                tool_id: self.inner.id().to_string(),
            },
            current_purpose(),
        );
        if self.manager.config().privacy.include_tool_args {
            span.set_payload(serde_json::json!({"args": args.clone()}));
        }
        let result = self.inner.execute(args, ctx).await;
        if !result.success {
            span.set_error(ObservationError::new("tool_error", result.output.clone()));
        }
        if self.manager.config().privacy.include_tool_outputs {
            span.set_payload(serde_json::json!({"output": result.output.clone()}));
        }
        result
    }
}

/// Streaming wrapper that records a span when the stream ends or is dropped.
struct ObservedLLMStream {
    inner: Box<dyn Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
    span: Option<SpanGuard>,
    estimated_input_tokens: u64,
    output_chars: usize,
    final_usage: Option<ObservationTokenUsage>,
    estimate_when_missing: bool,
}

impl ObservedLLMStream {
    fn new(
        inner: Box<dyn Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        span: SpanGuard,
        estimated_input_tokens: u64,
        estimate_when_missing: bool,
    ) -> Self {
        Self {
            inner,
            span: Some(span),
            estimated_input_tokens,
            output_chars: 0,
            final_usage: None,
            estimate_when_missing,
        }
    }

    fn finish(&mut self) {
        let Some(mut span) = self.span.take() else {
            return;
        };
        if let Some(usage) = self.final_usage.clone() {
            span.set_tokens(usage);
        } else if self.estimate_when_missing {
            let output_tokens = estimate_chars(self.output_chars);
            span.set_tokens(ObservationTokenUsage::new(
                self.estimated_input_tokens,
                output_tokens,
                TokenUsageSource::Estimated,
            ));
        }
        span.record_now();
    }
}

impl Stream for ObservedLLMStream {
    type Item = std::result::Result<LLMChunk, LLMError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                self.output_chars += chunk.delta.chars().count();
                if let Some(usage) = chunk.usage {
                    self.final_usage = Some(ObservationTokenUsage::new(
                        usage.prompt_tokens as u64,
                        usage.completion_tokens as u64,
                        TokenUsageSource::StreamFinalChunk,
                    ));
                }
                if chunk.is_final {
                    self.finish();
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(span) = self.span.as_mut() {
                    span.set_error(ObservationError::new(
                        llm_error_kind(&error),
                        error.to_string(),
                    ));
                }
                self.finish();
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                self.finish();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Unpin for ObservedLLMStream {}

impl Drop for ObservedLLMStream {
    fn drop(&mut self) {
        self.finish();
    }
}

fn current_purpose() -> ObservationPurpose {
    crate::context::current_observation_context()
        .map(|context| context.purpose)
        .unwrap_or_default()
}

fn estimate_usage(messages: &[ChatMessage], output: &str) -> ObservationTokenUsage {
    ObservationTokenUsage::new(
        estimate_messages(messages),
        estimate_chars(output.chars().count()),
        TokenUsageSource::Estimated,
    )
}

fn estimate_messages(messages: &[ChatMessage]) -> u64 {
    messages
        .iter()
        .map(|message| estimate_chars(message.content.chars().count()))
        .sum()
}

fn estimate_chars(chars: usize) -> u64 {
    ((chars as f64) / 4.0).ceil().max(1.0) as u64
}

fn llm_error_kind(error: &LLMError) -> &'static str {
    match error {
        LLMError::API { .. } => "api",
        LLMError::Network(_) => "network",
        LLMError::RateLimit { .. } => "rate_limit",
        LLMError::Config(_) => "config",
        LLMError::ModelNotFound(_) => "model_not_found",
        LLMError::ContentFiltered(_) => "content_filtered",
        LLMError::Serialization(_) => "serialization",
        LLMError::Other(_) => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ObservabilityConfig;
    use ai_agents_core::{FinishReason, LLMToolDefinition};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct NativeProvider {
        called: AtomicBool,
    }

    #[async_trait]
    impl LLMProvider for NativeProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse::new("plain", FinishReason::Stop))
        }

        async fn complete_with_tools(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
            _request: &LLMToolRequest,
        ) -> Result<LLMResponse, LLMError> {
            self.called.store(true, Ordering::SeqCst);
            Ok(LLMResponse::new("native", FinishReason::Stop))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<Box<dyn Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
        {
            Ok(Box::new(futures::stream::empty()))
        }

        fn provider_name(&self) -> &str {
            "native-test"
        }

        fn configured_tool_choice(&self) -> Option<ToolChoice> {
            Some(ToolChoice::Required)
        }

        fn supports_tool_choice(&self, choice: &ToolChoice) -> bool {
            matches!(choice, ToolChoice::Auto | ToolChoice::Required)
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn observed_provider_delegates_native_tool_methods() {
        let inner = Arc::new(NativeProvider {
            called: AtomicBool::new(false),
        });
        let observed = ObservedLLMProvider::new(
            inner.clone(),
            ObservabilityManager::new(ObservabilityConfig::default()),
            None,
            "native-test",
            "test-model",
        );
        let request = LLMToolRequest {
            tools: vec![LLMToolDefinition {
                name: "calculator".to_string(),
                description: "Calculate".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            choice: ToolChoice::Auto,
        };

        let response = observed
            .complete_with_tools(&[ChatMessage::user("calculate")], None, &request)
            .await
            .unwrap();

        assert_eq!(response.content, "native");
        assert!(inner.called.load(Ordering::SeqCst));
        assert_eq!(
            observed.configured_tool_choice(),
            Some(ToolChoice::Required)
        );
        assert!(observed.supports_tool_choice(&ToolChoice::Auto));
    }
}
