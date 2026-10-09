//! Invocation-scoped admission shared by frozen provider handles without a runtime dependency.

use crate::{
    ChatMessage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    LLMToolRequest, ToolChoice,
};
use async_trait::async_trait;
use std::future::Future;
use std::sync::Arc;

use ai_agents_core::autonomy::{
    InvocationAdmission, ProviderCostBound, ProviderCostSettlement, ProviderRequest,
    validate_cost_settlement,
};

tokio::task_local! {
    static INVOCATION_ADMISSION: Arc<dyn InvocationAdmission>;
}

/// Captures explicit authority for transfer across Tokio task boundaries; IDs and labels cannot mint it.
pub fn current_invocation_admission() -> Option<Arc<dyn InvocationAdmission>> {
    INVOCATION_ADMISSION.try_with(Arc::clone).ok()
}

/// Scope belongs to one invocation tree, not to a globally shared provider or registry.
pub async fn scope_invocation_admission<F: Future>(
    admission: Arc<dyn InvocationAdmission>,
    future: F,
) -> F::Output {
    INVOCATION_ADMISSION.scope(admission, future).await
}

struct Attempt {
    admission: Arc<dyn InvocationAdmission>,
    id: String,
    settled: bool,
    bound: Option<ProviderCostBound>,
}

impl Attempt {
    // Acknowledgement precedes execution, including auxiliary consumers with frozen provider handles.
    async fn admit(
        provider: &dyn LLMProvider,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        tools: Option<&LLMToolRequest>,
        streaming: bool,
    ) -> Result<Option<Self>, LLMError> {
        let Some(admission) = current_invocation_admission() else {
            return Ok(None);
        };
        let bound = if admission.requires_priced_llm() {
            let request_id = uuid::Uuid::new_v4().to_string();
            let bound = provider
                .request_cost_bound(&ProviderRequest {
                    request_id: &request_id,
                    messages,
                    config,
                    tools,
                    streaming,
                })
                .inspect_err(|_| admission.reject_priced_request())?;
            if let Some(bound) = &bound {
                bound
                    .validate(&request_id)
                    .inspect_err(|_| admission.reject_priced_request())?;
            }
            bound
        } else {
            None
        };
        let id = admission.admit_priced_llm(bound.clone()).await?;
        Ok(Some(Self {
            admission,
            id,
            settled: false,
            bound,
        }))
    }

    // Failed settlement retains the admitted marker; it is never converted into a refund or cached success.
    async fn settle(&mut self, result: serde_json::Value) -> Result<(), LLMError> {
        self.admission
            .settle_priced_llm(&self.id, result, None)
            .await?;
        self.settled = true;
        Ok(())
    }

    // Billing comes only from the conforming implementation and must match the exact admitted request.
    async fn settle_response(
        &mut self,
        provider: &dyn LLMProvider,
        result: &Result<LLMResponse, LLMError>,
    ) -> Result<(), LLMError> {
        let settlement: Option<ProviderCostSettlement> = match (&self.bound, result) {
            (Some(bound), Ok(response)) => provider.settle_request_cost(bound, response)?,
            _ => None,
        };
        if let (Some(bound), Some(settlement)) = (&self.bound, &settlement) {
            validate_cost_settlement(bound, settlement)?;
        }
        let data = match result {
            Ok(response) => serde_json::json!({"response":response}),
            Err(error) => serde_json::json!({"error":error.to_string()}),
        };
        self.admission
            .settle_priced_llm(&self.id, data, settlement)
            .await?;
        self.settled = true;
        Ok(())
    }
}

impl Drop for Attempt {
    // Dispatch may have occurred, so abandoned work remains charged even when its speculative branch loses.
    fn drop(&mut self) {
        if !self.settled {
            self.admission.abandon_llm(&self.id);
        }
    }
}

// A deadline drop remains uncertain; only a returned provider result may be settled as known.
async fn bounded_provider<F: Future>(
    attempt: Option<&Attempt>,
    future: F,
) -> Result<F::Output, LLMError> {
    if let Some(attempt) = attempt {
        if attempt.admission.execution_stopped() {
            return Err(LLMError::Other("task execution stopped".into()));
        }
        let remaining = attempt.admission.remaining_duration();
        let monitored = async {
            let mut future = Box::pin(future);
            let mut tick = tokio::time::interval(std::time::Duration::from_millis(10));
            loop {
                tokio::select! {
                    result = &mut future => return Ok(result),
                    _ = tick.tick() => if attempt.admission.execution_stopped() {
                        return Err(LLMError::Other("task execution stopped".into()));
                    }
                }
            }
        };
        return if let Some(duration) = remaining {
            tokio::time::timeout(duration, monitored)
                .await
                .map_err(|_| LLMError::Other("task provider deadline exceeded".into()))?
        } else {
            monitored.await
        };
    }
    Ok(future.await)
}

/// Accounts direct borrowed auxiliary providers without replacing captured handles or charging an already managed layer twice.
pub async fn managed_completion(
    provider: &dyn LLMProvider,
    messages: &[ChatMessage],
    config: Option<&LLMConfig>,
) -> Result<LLMResponse, LLMError> {
    if current_invocation_admission().is_none() || provider.manages_invocation_admission() {
        return provider.complete(messages, config).await;
    }
    let mut attempt = Attempt::admit(provider, messages, config, None, false).await?;
    let result = bounded_provider(attempt.as_ref(), provider.complete(messages, config)).await?;
    if let Some(attempt) = &mut attempt {
        attempt.settle_response(provider, &result).await?;
    }
    result
}

struct ManagedProvider {
    inner: Arc<dyn LLMProvider>,
}

/// Installs one transparent admission layer before consumers freeze handles; ordinary calls remain unchanged.
pub fn managed_provider(provider: Arc<dyn LLMProvider>) -> Arc<dyn LLMProvider> {
    if provider.manages_invocation_admission() {
        provider
    } else {
        Arc::new(ManagedProvider { inner: provider })
    }
}

#[async_trait]
impl LLMProvider for ManagedProvider {
    /// Each retry/fallback call enters separately; cached provider configuration stays owned by the inner adapter.
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        let mut attempt =
            Attempt::admit(self.inner.as_ref(), messages, config, None, false).await?;
        let result =
            bounded_provider(attempt.as_ref(), self.inner.complete(messages, config)).await?;
        if let Some(attempt) = &mut attempt {
            attempt
                .settle_response(self.inner.as_ref(), &result)
                .await?;
        }
        result
    }

    /// Native definitions are passed unchanged through the same attempt boundary as normal completion.
    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
        request: &LLMToolRequest,
    ) -> Result<LLMResponse, LLMError> {
        let mut attempt =
            Attempt::admit(self.inner.as_ref(), messages, config, Some(request), false).await?;
        let result = bounded_provider(
            attempt.as_ref(),
            self.inner.complete_with_tools(messages, config, request),
        )
        .await?;
        if let Some(attempt) = &mut attempt {
            attempt
                .settle_response(self.inner.as_ref(), &result)
                .await?;
        }
        result
    }

    /// The attempt survives stream opening and settles only on EOF/error; dropping it retains uncertainty.
    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        let mut attempt = Attempt::admit(self.inner.as_ref(), messages, config, None, true).await?;
        let source = match bounded_provider(
            attempt.as_ref(),
            self.inner.complete_stream(messages, config),
        )
        .await?
        {
            Ok(source) => source,
            Err(error) => {
                if let Some(attempt) = &mut attempt {
                    attempt
                        .settle(serde_json::json!({"error":error.to_string()}))
                        .await?;
                }
                return Err(error);
            }
        };
        if attempt.is_none() {
            return Ok(source);
        }
        use futures::StreamExt;
        let stream = futures::stream::unfold(
            (source, attempt, false),
            |(mut source, mut attempt, done)| async move {
                if done {
                    return None;
                }
                let next = match bounded_provider(attempt.as_ref(), source.next()).await {
                    Ok(next) => next,
                    Err(error) => return Some((Err(error), (source, attempt, true))),
                };
                match next {
                    Some(Ok(chunk)) => Some((Ok(chunk), (source, attempt, false))),
                    Some(Err(error)) => {
                        let settlement = attempt
                            .as_mut()
                            .unwrap()
                            .settle(serde_json::json!({"error":error.to_string()}))
                            .await;
                        Some((
                            Err(settlement.err().unwrap_or(error)),
                            (source, attempt, true),
                        ))
                    }
                    None => {
                        match attempt
                            .as_mut()
                            .unwrap()
                            .settle(serde_json::json!({"stream_finished":true}))
                            .await
                        {
                            Ok(()) => None,
                            Err(error) => Some((Err(error), (source, attempt, true))),
                        }
                    }
                }
            },
        );
        Ok(Box::new(Box::pin(stream)))
    }

    /// Wrapper identity does not replace the underlying provider's public identity.
    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
    /// Capability probing is unchanged by invocation accounting.
    fn supports(&self, feature: LLMFeature) -> bool {
        self.inner.supports(feature)
    }
    /// Consumer-side wrapping is idempotent, including inherited and observed registries.
    fn manages_invocation_admission(&self) -> bool {
        true
    }
    fn priced_capability_identity(&self) -> Option<String> {
        self.inner.priced_capability_identity()
    }
    fn request_cost_bound(
        &self,
        request: &ProviderRequest<'_>,
    ) -> Result<Option<ProviderCostBound>, LLMError> {
        self.inner.request_cost_bound(request)
    }
    fn settle_request_cost(
        &self,
        bound: &ProviderCostBound,
        response: &LLMResponse,
    ) -> Result<Option<ProviderCostSettlement>, LLMError> {
        self.inner.settle_request_cost(bound, response)
    }

    /// Tool choice remains a provider selection contract, never an execution grant.
    fn configured_tool_choice(&self) -> Option<ToolChoice> {
        self.inner.configured_tool_choice()
    }
    /// Native-choice readiness remains owned by the actual provider implementation.
    fn supports_tool_choice(&self, choice: &ToolChoice) -> bool {
        self.inner.supports_tool_choice(choice)
    }
    /// Shared execution stops cannot be hidden by generic retry or static fallback responses.
    fn is_terminal_error(&self, error: &LLMError) -> bool {
        current_invocation_admission().is_some_and(|scope| scope.execution_stopped())
            || self.inner.is_terminal_error(error)
    }
}
