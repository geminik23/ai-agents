use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use ai_agents_core::{
    ChatMessage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    LLMToolRequest, TokenUsage, ToolChoice,
};
use ai_agents_observability::{CostEstimator, ObservationTokenUsage, TokenUsageSource};
use async_trait::async_trait;
use futures::Stream;

use crate::suite::ScenarioBudget;

pub(crate) const BUDGET_ERROR_PREFIX: &str = "eval budget exceeded";

#[derive(Debug, Clone)]
pub(crate) struct BudgetProviderConfig {
    pub provider: String,
    pub model: String,
    pub max_output_tokens: u32,
}

#[derive(Clone)]
pub(crate) struct ScenarioBudgetTracker {
    budget: ScenarioBudget,
    estimator: Option<CostEstimator>,
    state: Arc<Mutex<BudgetUsage>>,
}

#[derive(Debug, Default)]
struct BudgetUsage {
    llm_calls: u64,
    tokens_used: u64,
    tokens_reserved: u64,
    cost_used_usd: f64,
    cost_reserved_usd: f64,
    failure: Option<String>,
}

struct BudgetReservation {
    tokens: u64,
    cost_usd: f64,
}

// Provider futures can be dropped at any await, so reservation cleanup must be owned by a drop guard.
struct PendingBudgetReservation {
    tracker: ScenarioBudgetTracker,
    reservation: Option<BudgetReservation>,
}

impl PendingBudgetReservation {
    fn new(tracker: ScenarioBudgetTracker, reservation: BudgetReservation) -> Self {
        Self {
            tracker,
            reservation: Some(reservation),
        }
    }

    fn finish(
        &mut self,
        provider: &BudgetProviderConfig,
        usage: TokenUsage,
    ) -> Result<(), LLMError> {
        let Some(reservation) = self.reservation.take() else {
            return Ok(());
        };
        self.tracker.finish(provider, reservation, usage)
    }

    fn finish_unknown(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            self.tracker.finish_unknown(reservation);
        }
    }
}

impl Drop for PendingBudgetReservation {
    fn drop(&mut self) {
        self.finish_unknown();
    }
}

impl ScenarioBudgetTracker {
    pub(crate) fn new(budget: ScenarioBudget, estimator: Option<CostEstimator>) -> Self {
        Self {
            budget,
            estimator,
            state: Arc::new(Mutex::new(BudgetUsage::default())),
        }
    }

    pub(crate) fn wrap(
        &self,
        inner: Arc<dyn LLMProvider>,
        provider: BudgetProviderConfig,
    ) -> Arc<dyn LLMProvider> {
        Arc::new(BudgetedLlmProvider {
            inner,
            tracker: self.clone(),
            provider,
        })
    }

    pub(crate) fn has_failed(&self) -> bool {
        self.lock_state().failure.is_some()
    }

    fn reserve(
        &self,
        provider: &BudgetProviderConfig,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<BudgetReservation, LLMError> {
        self.reserve_with_additional_input(provider, messages, config, 0)
    }

    fn reserve_with_additional_input(
        &self,
        provider: &BudgetProviderConfig,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        additional_input_tokens: u64,
    ) -> Result<BudgetReservation, LLMError> {
        let input_tokens =
            upper_bound_message_tokens(messages).saturating_add(additional_input_tokens);
        let output_tokens = config
            .and_then(|config| config.max_tokens)
            .unwrap_or(provider.max_output_tokens) as u64;
        let tokens = input_tokens.saturating_add(output_tokens);
        let cost_usd = if self.budget.max_cost_usd.is_some() {
            self.estimate_cost(provider, input_tokens, output_tokens)?
        } else {
            0.0
        };

        let mut usage = self.lock_state();
        if let Some(message) = usage.failure.clone() {
            return Err(LLMError::Other(message));
        }
        if let Some(max) = self.budget.max_llm_calls
            && usage.llm_calls >= max
        {
            let blocked_call = usage.llm_calls + 1;
            return Err(mark_failed(
                &mut usage,
                format!(
                    "{BUDGET_ERROR_PREFIX}: max_llm_calls={max}; blocked provider call {}",
                    blocked_call
                ),
            ));
        }
        if let Some(max) = self.budget.max_total_tokens {
            let projected = usage
                .tokens_used
                .saturating_add(usage.tokens_reserved)
                .saturating_add(tokens);
            if projected > max {
                return Err(mark_failed(
                    &mut usage,
                    format!(
                        "{BUDGET_ERROR_PREFIX}: max_total_tokens={max}; projected reserved total={projected}"
                    ),
                ));
            }
        }
        if let Some(max) = self.budget.max_cost_usd {
            let projected = usage.cost_used_usd + usage.cost_reserved_usd + cost_usd;
            if projected > max {
                return Err(mark_failed(
                    &mut usage,
                    format!(
                        "{BUDGET_ERROR_PREFIX}: max_cost_usd={max:.6}; projected reserved cost=${projected:.6}"
                    ),
                ));
            }
        }

        usage.llm_calls += 1;
        usage.tokens_reserved = usage.tokens_reserved.saturating_add(tokens);
        usage.cost_reserved_usd += cost_usd;
        Ok(BudgetReservation { tokens, cost_usd })
    }

    fn finish(
        &self,
        provider: &BudgetProviderConfig,
        reservation: BudgetReservation,
        usage: TokenUsage,
    ) -> Result<(), LLMError> {
        let tokens = usage.total_tokens as u64;
        let cost_usd = if self.budget.max_cost_usd.is_some() {
            match self.estimate_cost(
                provider,
                usage.prompt_tokens as u64,
                usage.completion_tokens as u64,
            ) {
                Ok(cost_usd) => cost_usd,
                Err(error) => {
                    self.finish_unknown(reservation);
                    return Err(error);
                }
            }
        } else {
            0.0
        };
        let mut state = self.lock_state();
        state.tokens_reserved = state.tokens_reserved.saturating_sub(reservation.tokens);
        state.cost_reserved_usd = (state.cost_reserved_usd - reservation.cost_usd).max(0.0);
        state.tokens_used = state.tokens_used.saturating_add(tokens);
        state.cost_used_usd += cost_usd;

        if let Some(max) = self.budget.max_total_tokens
            && state.tokens_used.saturating_add(state.tokens_reserved) > max
        {
            let actual = state.tokens_used.saturating_add(state.tokens_reserved);
            return Err(mark_failed(
                &mut state,
                format!(
                    "{BUDGET_ERROR_PREFIX}: max_total_tokens={max}; actual and reserved total={actual}"
                ),
            ));
        }
        if let Some(max) = self.budget.max_cost_usd {
            let actual = state.cost_used_usd + state.cost_reserved_usd;
            if actual > max {
                return Err(mark_failed(
                    &mut state,
                    format!(
                        "{BUDGET_ERROR_PREFIX}: max_cost_usd={max:.6}; actual and reserved cost=${actual:.6}"
                    ),
                ));
            }
        }
        Ok(())
    }

    fn finish_unknown(&self, reservation: BudgetReservation) {
        let mut state = self.lock_state();
        state.tokens_reserved = state.tokens_reserved.saturating_sub(reservation.tokens);
        state.cost_reserved_usd = (state.cost_reserved_usd - reservation.cost_usd).max(0.0);
        state.tokens_used = state.tokens_used.saturating_add(reservation.tokens);
        state.cost_used_usd += reservation.cost_usd;
    }

    fn estimate_cost(
        &self,
        provider: &BudgetProviderConfig,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<f64, LLMError> {
        let usage =
            ObservationTokenUsage::new(input_tokens, output_tokens, TokenUsageSource::Provider);
        self.estimator
            .as_ref()
            .and_then(|estimator| {
                estimator.estimate(
                    Some(&provider.provider),
                    Some(&provider.model),
                    Some(&usage),
                )
            })
            .map(|cost| cost.total_usd)
            .ok_or_else(|| {
                let mut state = self.lock_state();
                mark_failed(
                    &mut state,
                    format!(
                        "{BUDGET_ERROR_PREFIX}: max_cost_usd requires pricing for {}/{}",
                        provider.provider, provider.model
                    ),
                )
            })
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, BudgetUsage> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn mark_failed(state: &mut BudgetUsage, message: String) -> LLMError {
    state.failure = Some(message.clone());
    LLMError::Other(message)
}

struct BudgetedLlmProvider {
    inner: Arc<dyn LLMProvider>,
    tracker: ScenarioBudgetTracker,
    provider: BudgetProviderConfig,
}

#[async_trait]
impl LLMProvider for BudgetedLlmProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        let reservation = self.tracker.reserve(&self.provider, messages, config)?;
        let mut pending = PendingBudgetReservation::new(self.tracker.clone(), reservation);
        match self.inner.complete(messages, config).await {
            Ok(response) => {
                let usage = response
                    .usage
                    .unwrap_or_else(|| estimate_usage(messages, &response.content));
                pending.finish(&self.provider, usage)?;
                Ok(response)
            }
            Err(error) => {
                pending.finish_unknown();
                Err(error)
            }
        }
    }

    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        request: &LLMToolRequest,
    ) -> Result<LLMResponse, LLMError> {
        let request_tokens = upper_bound_tool_request_tokens(request);
        let reservation = self.tracker.reserve_with_additional_input(
            &self.provider,
            messages,
            config,
            request_tokens,
        )?;
        let mut pending = PendingBudgetReservation::new(self.tracker.clone(), reservation);
        match self
            .inner
            .complete_with_tools(messages, config, request)
            .await
        {
            Ok(response) => {
                let usage = response
                    .usage
                    .unwrap_or_else(|| estimate_tool_usage(messages, request, &response));
                pending.finish(&self.provider, usage)?;
                Ok(response)
            }
            Err(error) => {
                pending.finish_unknown();
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
    ) -> Result<Box<dyn Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError> {
        let reservation = self.tracker.reserve(&self.provider, messages, config)?;
        let mut pending = PendingBudgetReservation::new(self.tracker.clone(), reservation);
        match self.inner.complete_stream(messages, config).await {
            Ok(stream) => Ok(Box::new(BudgetedLlmStream {
                inner: stream,
                provider: self.provider.clone(),
                pending,
                input_tokens: estimate_message_tokens(messages),
                output_chars: 0,
                done: false,
            })),
            Err(error) => {
                pending.finish_unknown();
                Err(error)
            }
        }
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        self.inner.supports(feature)
    }
}

struct BudgetedLlmStream {
    inner: Box<dyn Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>,
    provider: BudgetProviderConfig,
    pending: PendingBudgetReservation,
    input_tokens: u64,
    output_chars: usize,
    done: bool,
}

impl BudgetedLlmStream {
    fn finish(&mut self, usage: Option<TokenUsage>) -> Result<(), LLMError> {
        let usage = usage.unwrap_or_else(|| TokenUsage {
            prompt_tokens: self.input_tokens.min(u32::MAX as u64) as u32,
            completion_tokens: estimate_chars(self.output_chars).min(u32::MAX as u64) as u32,
            total_tokens: self
                .input_tokens
                .saturating_add(estimate_chars(self.output_chars))
                .min(u32::MAX as u64) as u32,
        });
        self.pending.finish(&self.provider, usage)
    }

    fn finish_unknown(&mut self) {
        self.pending.finish_unknown();
    }
}

impl Stream for BudgetedLlmStream {
    type Item = Result<LLMChunk, LLMError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                self.output_chars += chunk.delta.chars().count();
                if chunk.is_final {
                    self.done = true;
                    if let Err(error) = self.finish(chunk.usage) {
                        return Poll::Ready(Some(Err(error)));
                    }
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.done = true;
                self.finish_unknown();
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                self.done = true;
                match self.finish(None) {
                    Ok(()) => Poll::Ready(None),
                    Err(error) => Poll::Ready(Some(Err(error))),
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Unpin for BudgetedLlmStream {}

impl Drop for BudgetedLlmStream {
    fn drop(&mut self) {
        self.finish_unknown();
    }
}

fn upper_bound_tool_request_tokens(request: &LLMToolRequest) -> u64 {
    serde_json::to_vec(request)
        .map(|encoded| encoded.len() as u64)
        .unwrap_or(u64::MAX)
}

fn upper_bound_message_tokens(messages: &[ChatMessage]) -> u64 {
    messages.iter().fold(0_u64, |total, message| {
        total
            .saturating_add(message.content.len() as u64)
            .saturating_add(16)
    })
}

fn estimate_message_tokens(messages: &[ChatMessage]) -> u64 {
    messages.iter().fold(0_u64, |total, message| {
        total.saturating_add(estimate_chars(message.content.chars().count()))
    })
}

fn estimate_chars(chars: usize) -> u64 {
    ((chars as f64) / 4.0).ceil().max(1.0) as u64
}

fn estimate_usage(messages: &[ChatMessage], output: &str) -> TokenUsage {
    let prompt_tokens = estimate_message_tokens(messages).min(u32::MAX as u64) as u32;
    let completion_tokens = estimate_chars(output.chars().count()).min(u32::MAX as u64) as u32;
    TokenUsage::new(prompt_tokens, completion_tokens)
}

fn estimate_tool_usage(
    messages: &[ChatMessage],
    request: &LLMToolRequest,
    response: &LLMResponse,
) -> TokenUsage {
    let request_chars = serde_json::to_string(request)
        .map(|encoded| encoded.chars().count())
        .unwrap_or(usize::MAX);
    let response_chars = response.content.chars().count().saturating_add(
        response
            .tool_calls()
            .ok()
            .flatten()
            .and_then(|calls| serde_json::to_string(&calls).ok())
            .map(|encoded| encoded.chars().count())
            .unwrap_or(0),
    );
    let prompt_tokens = estimate_message_tokens(messages)
        .saturating_add(estimate_chars(request_chars))
        .min(u32::MAX as u64) as u32;
    let completion_tokens = estimate_chars(response_chars).min(u32::MAX as u64) as u32;
    TokenUsage::new(prompt_tokens, completion_tokens)
}

#[cfg(test)]
mod tests {
    use ai_agents_core::{FinishReason, LLMResponse};
    use ai_agents_llm::mock::MockLLMProvider;
    use ai_agents_observability::{CostConfig, ModelPricing};
    use futures::StreamExt;
    use std::collections::HashMap;
    use tokio::sync::Notify;

    use super::*;

    fn provider() -> Arc<dyn LLMProvider> {
        let mut provider = MockLLMProvider::new("budget-test");
        provider.add_response(LLMResponse::new("ok", FinishReason::Stop));
        Arc::new(provider)
    }

    fn provider_config() -> BudgetProviderConfig {
        BudgetProviderConfig {
            provider: "openai".to_string(),
            model: "test".to_string(),
            max_output_tokens: 10,
        }
    }

    struct PendingProvider {
        started: Arc<Notify>,
    }

    #[async_trait]
    impl LLMProvider for PendingProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, LLMError> {
            self.started.notify_one();
            std::future::pending().await
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<Box<dyn Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
        {
            self.started.notify_one();
            std::future::pending().await
        }

        fn provider_name(&self) -> &str {
            "pending"
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    struct PendingStreamProvider;

    #[async_trait]
    impl LLMProvider for PendingStreamProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, LLMError> {
            Ok(LLMResponse::new("ok", FinishReason::Stop))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> Result<Box<dyn Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
        {
            Ok(Box::new(futures::stream::pending()))
        }

        fn provider_name(&self) -> &str {
            "pending-stream"
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    fn cost_estimator() -> CostEstimator {
        CostEstimator::new(CostConfig {
            pricing: HashMap::from([(
                "openai/test".to_string(),
                ModelPricing {
                    input_per_1k: 1.0,
                    output_per_1k: 1.0,
                },
            )]),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn call_limit_is_shared_across_wrapped_providers() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_llm_calls: Some(1),
                ..Default::default()
            },
            None,
        );
        let first = tracker.wrap(provider(), provider_config());
        let second = tracker.wrap(provider(), provider_config());
        let messages = [ChatMessage::user("hello")];

        assert!(first.complete(&messages, None).await.is_ok());
        let error = second.complete(&messages, None).await.unwrap_err();
        assert!(error.to_string().contains("max_llm_calls=1"));
        assert!(tracker.has_failed());
    }

    #[tokio::test]
    async fn native_tool_calls_use_the_same_call_budget() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_llm_calls: Some(1),
                ..Default::default()
            },
            None,
        );
        let mut native = MockLLMProvider::new("native-budget-test");
        native.add_response(
            LLMResponse::new("", FinishReason::ToolCall)
                .with_tool_calls(vec![ai_agents_core::ToolCall {
                    id: "call-1".to_string(),
                    name: "calculator".to_string(),
                    arguments: serde_json::json!({"expression": "2 + 2"}),
                }])
                .unwrap(),
        );
        let wrapped = tracker.wrap(Arc::new(native), provider_config());
        let request = LLMToolRequest {
            tools: vec![ai_agents_core::LLMToolDefinition {
                name: "calculator".to_string(),
                description: "Calculate".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            choice: ToolChoice::Required,
        };
        let messages = [ChatMessage::user("calculate")];

        assert!(
            wrapped
                .complete_with_tools(&messages, None, &request)
                .await
                .is_ok()
        );
        let error = wrapped
            .complete_with_tools(&messages, None, &request)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("max_llm_calls=1"));
    }

    #[tokio::test]
    async fn native_tool_schema_is_reserved_before_provider_call() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_total_tokens: Some(100),
                ..Default::default()
            },
            None,
        );
        let native = MockLLMProvider::new("native-schema-budget-test");
        let observed = native.clone();
        let wrapped = tracker.wrap(Arc::new(native), provider_config());
        let request = LLMToolRequest {
            tools: vec![ai_agents_core::LLMToolDefinition {
                name: "large_tool".to_string(),
                description: "x".repeat(200),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            choice: ToolChoice::Auto,
        };

        let error = wrapped
            .complete_with_tools(&[ChatMessage::user("use it")], None, &request)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("projected reserved total"));
        assert_eq!(observed.call_count(), 0);
    }

    #[tokio::test]
    async fn token_reservation_blocks_before_provider_call() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_total_tokens: Some(5),
                ..Default::default()
            },
            None,
        );
        let wrapped = tracker.wrap(provider(), provider_config());
        let error = wrapped
            .complete(&[ChatMessage::user("hello")], None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("projected reserved total"));
    }

    #[tokio::test]
    async fn cost_reservation_uses_configured_pricing() {
        let mut pricing = HashMap::new();
        pricing.insert(
            "openai/test".to_string(),
            ModelPricing {
                input_per_1k: 1.0,
                output_per_1k: 1.0,
            },
        );
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_cost_usd: Some(0.001),
                ..Default::default()
            },
            Some(CostEstimator::new(CostConfig {
                pricing,
                ..Default::default()
            })),
        );
        let wrapped = tracker.wrap(provider(), provider_config());
        let error = wrapped
            .complete(&[ChatMessage::user("hello")], None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("projected reserved cost"));
    }

    #[tokio::test]
    async fn cost_budget_fails_closed_when_model_pricing_is_missing() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_cost_usd: Some(1.0),
                ..Default::default()
            },
            Some(CostEstimator::new(CostConfig::default())),
        );
        let wrapped = tracker.wrap(provider(), provider_config());
        let error = wrapped
            .complete(&[ChatMessage::user("hello")], None)
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("requires pricing for openai/test")
        );
        assert!(tracker.has_failed());
    }

    #[tokio::test]
    async fn provider_error_finalizes_reservation_once() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_llm_calls: Some(1),
                max_total_tokens: Some(100),
                ..Default::default()
            },
            None,
        );
        let mut inner = MockLLMProvider::new("budget-error");
        inner.set_error("provider failed");
        let wrapped = tracker.wrap(Arc::new(inner), provider_config());
        let messages = [ChatMessage::user("hello")];
        let expected_tokens = upper_bound_message_tokens(&messages) + 10;

        assert!(wrapped.complete(&messages, None).await.is_err());

        let state = tracker.lock_state();
        assert_eq!(state.llm_calls, 1);
        assert_eq!(state.tokens_reserved, 0);
        assert_eq!(state.tokens_used, expected_tokens);
    }

    #[tokio::test]
    async fn blocking_cancellation_finalizes_reservation_once() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_llm_calls: Some(2),
                max_total_tokens: Some(1_000),
                max_cost_usd: Some(10.0),
            },
            Some(cost_estimator()),
        );
        let started = Arc::new(Notify::new());
        let wrapped = tracker.wrap(
            Arc::new(PendingProvider {
                started: Arc::clone(&started),
            }),
            provider_config(),
        );
        let task =
            tokio::spawn(
                async move { wrapped.complete(&[ChatMessage::user("hello")], None).await },
            );
        started.notified().await;
        let (reserved_tokens, reserved_cost) = {
            let state = tracker.lock_state();
            assert_eq!(state.llm_calls, 1);
            assert!(state.tokens_reserved > 0);
            assert!(state.cost_reserved_usd > 0.0);
            (state.tokens_reserved, state.cost_reserved_usd)
        };

        task.abort();
        assert!(matches!(task.await, Err(error) if error.is_cancelled()));

        {
            let state = tracker.lock_state();
            assert_eq!(state.llm_calls, 1);
            assert_eq!(state.tokens_reserved, 0);
            assert_eq!(state.cost_reserved_usd, 0.0);
            assert_eq!(state.tokens_used, reserved_tokens);
            assert!((state.cost_used_usd - reserved_cost).abs() < f64::EPSILON);
        }

        let next = tracker.wrap(provider(), provider_config());
        assert!(
            next.complete(&[ChatMessage::user("next")], None)
                .await
                .is_ok()
        );
        assert_eq!(tracker.lock_state().tokens_reserved, 0);
    }

    #[tokio::test]
    async fn stream_creation_cancellation_finalizes_reservation_once() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_llm_calls: Some(2),
                max_total_tokens: Some(1_000),
                max_cost_usd: Some(10.0),
            },
            Some(cost_estimator()),
        );
        let started = Arc::new(Notify::new());
        let wrapped = tracker.wrap(
            Arc::new(PendingProvider {
                started: Arc::clone(&started),
            }),
            provider_config(),
        );
        let task = tokio::spawn(async move {
            wrapped
                .complete_stream(&[ChatMessage::user("hello")], None)
                .await
        });
        started.notified().await;
        let (reserved_tokens, reserved_cost) = {
            let state = tracker.lock_state();
            assert_eq!(state.llm_calls, 1);
            assert!(state.tokens_reserved > 0);
            assert!(state.cost_reserved_usd > 0.0);
            (state.tokens_reserved, state.cost_reserved_usd)
        };

        task.abort();
        assert!(matches!(task.await, Err(error) if error.is_cancelled()));

        let state = tracker.lock_state();
        assert_eq!(state.llm_calls, 1);
        assert_eq!(state.tokens_reserved, 0);
        assert_eq!(state.cost_reserved_usd, 0.0);
        assert_eq!(state.tokens_used, reserved_tokens);
        assert!((state.cost_used_usd - reserved_cost).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn dropping_returned_stream_finalizes_reservation_once() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_llm_calls: Some(1),
                max_total_tokens: Some(100),
                ..Default::default()
            },
            None,
        );
        let wrapped = tracker.wrap(Arc::new(PendingStreamProvider), provider_config());
        let stream = wrapped
            .complete_stream(&[ChatMessage::user("hello")], None)
            .await
            .unwrap();
        let reserved_tokens = tracker.lock_state().tokens_reserved;
        assert!(reserved_tokens > 0);

        drop(stream);

        let state = tracker.lock_state();
        assert_eq!(state.tokens_reserved, 0);
        assert_eq!(state.tokens_used, reserved_tokens);
    }

    #[tokio::test]
    async fn streaming_call_finishes_within_budget() {
        let tracker = ScenarioBudgetTracker::new(
            ScenarioBudget {
                max_llm_calls: Some(1),
                max_total_tokens: Some(100),
                ..Default::default()
            },
            None,
        );
        let wrapped = tracker.wrap(provider(), provider_config());
        let mut stream = wrapped
            .complete_stream(&[ChatMessage::user("hello")], None)
            .await
            .unwrap();
        while let Some(chunk) = stream.next().await {
            chunk.unwrap();
        }
        let used_after_completion = tracker.lock_state().tokens_used;
        assert!(used_after_completion > 0);
        assert_eq!(tracker.lock_state().tokens_reserved, 0);
        drop(stream);
        assert_eq!(tracker.lock_state().tokens_used, used_after_completion);
        assert!(!tracker.has_failed());
    }
}
