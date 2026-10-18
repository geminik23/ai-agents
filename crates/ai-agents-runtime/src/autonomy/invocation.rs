//! Live observation restrictions and mutation clocks cross existing execution boundaries without minting task owners.

use ai_agents_core::ToolExecutionRecord;
use ai_agents_llm::{InvocationBounds, current_invocation_bounds, scope_invocation_bounds};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

tokio::task_local! { static OBSERVATION_MUTATIONS: Arc<AtomicU64>; }

/// Captured live context is propagated explicitly across framework-spawned tasks, never serialized.
#[derive(Clone, Default)]
pub(crate) struct InvocationScope {
    bounds: Option<InvocationBounds>,
    mutations: Option<Arc<AtomicU64>>,
    evidence: Vec<Arc<dyn ai_agents_core::autonomy::TaskFinalAdmission>>,
}

impl InvocationScope {
    /// Captures the current restriction tree without replacing the run's ledger or fixed-operation authority.
    pub(crate) fn capture() -> Self {
        Self {
            bounds: current_invocation_bounds(),
            mutations: current_mutation_clock(),
            evidence: ai_agents_llm::current_invocation_evidence(),
        }
    }

    /// Independent production observations use their runtime's persistent clock, or their enclosing observation tree's shared clock.
    pub(crate) fn observation(clock: Arc<AtomicU64>) -> Self {
        let mut scope = Self::capture();
        scope.mutations = scope.mutations.or(Some(clock));
        scope
    }

    /// Nested restrictions intersect; inherited mutation tracking observes actual child/fallback hops rather than only the outer record.
    pub(crate) async fn scope<F: std::future::Future>(self, future: F) -> F::Output {
        let work = async {
            if let Some(clock) = self.mutations {
                OBSERVATION_MUTATIONS.scope(clock, future).await
            } else {
                future.await
            }
        };
        Box::pin(ai_agents_llm::scope_invocation_evidence(
            self.evidence,
            scope_invocation_bounds(self.bounds, Box::pin(work)),
        ))
        .await
    }
}

/// Looks up the independent observation tree's clock; live runners retain their existing authoritative run generation.
pub(crate) fn current_mutation_clock() -> Option<Arc<AtomicU64>> {
    OBSERVATION_MUTATIONS.try_with(Arc::clone).ok()
}

/// Every executed non-read-only hop invalidates earlier reads before publication hooks can reenter validation.
pub(crate) fn capture_observation_mutation(
    record: &ToolExecutionRecord,
    runtime_clock: &Arc<AtomicU64>,
) {
    if record.executed
        && record
            .metadata
            .get("classification")
            .and_then(|value| value.get("read_only"))
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        start_observation_mutation(runtime_clock);
    }
}

/// Actual mutation admission invalidates prior reads even if the caller drops before record finalization.
pub(crate) fn start_observation_mutation(runtime_clock: &Arc<AtomicU64>) {
    let advance = |clock: &AtomicU64| {
        // Saturation poisons later freshness capture instead of wrapping an old generation into eligibility.
        let _ = clock.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
            Some(value.saturating_add(1))
        });
    };
    advance(runtime_clock);
    if let Some(clock) = current_mutation_clock()
        && !Arc::ptr_eq(&clock, runtime_clock)
    {
        advance(&clock);
    }
}
