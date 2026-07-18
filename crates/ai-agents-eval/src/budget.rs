use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use ai_agents_core::{
    ChatMessage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse, TokenUsage,
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
        let input_tokens = upper_bound_message_tokens(messages);
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
            self.estimate_cost(
                provider,
                usage.prompt_tokens as u64,
                usage.completion_tokens as u64,
            )?
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
        match self.inner.complete(messages, config).await {
            Ok(response) => {
                let usage = response
                    .usage
                    .unwrap_or_else(|| estimate_usage(messages, &response.content));
                self.tracker.finish(&self.provider, reservation, usage)?;
                Ok(response)
            }
            Err(error) => {
                self.tracker.finish_unknown(reservation);
                Err(error)
            }
        }
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError> {
        let reservation = self.tracker.reserve(&self.provider, messages, config)?;
        match self.inner.complete_stream(messages, config).await {
            Ok(stream) => Ok(Box::new(BudgetedLlmStream {
                inner: stream,
                tracker: self.tracker.clone(),
                provider: self.provider.clone(),
                reservation: Some(reservation),
                input_tokens: estimate_message_tokens(messages),
                output_chars: 0,
                done: false,
            })),
            Err(error) => {
                self.tracker.finish_unknown(reservation);
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
    tracker: ScenarioBudgetTracker,
    provider: BudgetProviderConfig,
    reservation: Option<BudgetReservation>,
    input_tokens: u64,
    output_chars: usize,
    done: bool,
}

impl BudgetedLlmStream {
    fn finish(&mut self, usage: Option<TokenUsage>) -> Result<(), LLMError> {
        let Some(reservation) = self.reservation.take() else {
            return Ok(());
        };
        let usage = usage.unwrap_or_else(|| TokenUsage {
            prompt_tokens: self.input_tokens.min(u32::MAX as u64) as u32,
            completion_tokens: estimate_chars(self.output_chars).min(u32::MAX as u64) as u32,
            total_tokens: self
                .input_tokens
                .saturating_add(estimate_chars(self.output_chars))
                .min(u32::MAX as u64) as u32,
        });
        self.tracker.finish(&self.provider, reservation, usage)
    }

    fn finish_unknown(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            self.tracker.finish_unknown(reservation);
        }
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

#[cfg(test)]
mod tests {
    use ai_agents_core::{FinishReason, LLMResponse};
    use ai_agents_llm::mock::MockLLMProvider;
    use ai_agents_observability::{CostConfig, ModelPricing};
    use futures::StreamExt;
    use std::collections::HashMap;

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
        assert!(!tracker.has_failed());
    }
}
