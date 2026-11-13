//! Live observation restrictions and mutation clocks cross existing execution boundaries without minting task owners.

use ai_agents_core::ToolExecutionRecord;
use ai_agents_llm::{InvocationBounds, current_invocation_bounds, scope_invocation_bounds};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

tokio::task_local! { static OBSERVATION_MUTATIONS: Arc<AtomicU64>; }
tokio::task_local! { static CAPACITY_WITNESS: Arc<CapacityWitness>; }

/// A live admission witness distinguishes a pure resource refusal from an observation that already admitted any work.
#[derive(Default)]
pub(crate) struct CapacityWitness {
    admitted: AtomicU64,
    denied: parking_lot::Mutex<Option<String>>,
}

impl CapacityWitness {
    /// Certification is restriction-only and requires zero acknowledged admissions in this observation; any admitted parent also excludes nested effects.
    pub(crate) fn noninvocation(&self) -> Option<String> {
        if self.admitted.load(Ordering::Acquire) != 0 {
            return None;
        }
        self.denied.lock().clone()
    }
}

/// Only the shared ledger records a capacity refusal, before dispatch and without a reservation write.
pub(crate) fn capacity_denied(reason: &str) {
    let _ = CAPACITY_WITNESS.try_with(|witness| *witness.denied.lock() = Some(reason.into()));
}

/// Any acknowledged admission poisons non-invocation certification, even when later settled or refunded.
pub(crate) fn capacity_admitted() {
    let _ = CAPACITY_WITNESS.try_with(|witness| {
        witness.admitted.fetch_add(1, Ordering::AcqRel);
    });
}

/// The witness is live request-local state; it cannot be decoded from a journal or supplied as execution authority.
pub(crate) async fn scope_capacity_witness<F: std::future::Future>(
    witness: Arc<CapacityWitness>,
    future: F,
) -> F::Output {
    CAPACITY_WITNESS.scope(witness, future).await
}

/// Rechecks continuing restrictions after dispatch without reopening admission-only gates or undoing acknowledged effects.
/// Host proof is sampled before clocks so synchronous validation cannot make a late result eligible.
pub(crate) fn check_dispatched_invocation() -> ai_agents_core::Result<()> {
    for guard in ai_agents_llm::current_invocation_evidence() {
        guard.validate_dispatched()?;
    }
    if super::current_execution().is_some_and(|execution| {
        ai_agents_core::autonomy::InvocationAdmission::execution_stopped(execution.as_ref())
    }) || current_invocation_bounds().is_some_and(|bounds| bounds.remaining().is_zero())
    {
        return Err(ai_agents_core::AgentError::Other(
            "invocation result exceeded its original authority".into(),
        ));
    }
    Ok(())
}

/// Captured live context is propagated explicitly across framework-spawned tasks, never serialized.
#[derive(Clone, Default)]
pub(crate) struct InvocationScope {
    bounds: Option<InvocationBounds>,
    mutations: Option<Arc<AtomicU64>>,
    evidence: Vec<Arc<dyn ai_agents_core::autonomy::TaskFinalAdmission>>,
    child_operation: Option<String>,
    child_required: Option<bool>,
    host_session: Option<super::TaskHostSession>,
}

impl InvocationScope {
    /// Captures the restriction tree and composition cut before a spawn without replacing ledger or fixed-operation authority.
    pub(crate) fn capture() -> Self {
        let bounds = super::composition::current_composition_deadline().map(|deadline| {
            let now = std::time::SystemTime::now();
            InvocationBounds::new(
                now.checked_add(deadline.saturating_duration_since(std::time::Instant::now()))
                    .unwrap_or(now),
                deadline,
            )
        });
        let bounds = match (current_invocation_bounds(), bounds) {
            (Some(parent), Some(child)) => Some(parent.intersect(child)),
            (parent, child) => parent.or(child),
        };
        Self {
            bounds,
            mutations: current_mutation_clock(),
            evidence: ai_agents_llm::current_invocation_evidence(),
            child_operation: super::current_child_operation(),
            child_required: Some(super::child_required()),
            host_session: super::host_session::current_session(),
        }
    }

    /// Independent production observations use their runtime's persistent clock, or their enclosing observation tree's shared clock.
    pub(crate) fn observation(clock: Arc<AtomicU64>) -> Self {
        let mut scope = Self::capture();
        scope.mutations = scope.mutations.or(Some(clock));
        scope
    }

    /// Nested restrictions intersect; child attribution, host delivery and failure policy survive spawned callbacks without minting another operation.
    /// Inherited mutation tracking observes actual child/fallback hops rather than only the outer record.
    pub(crate) async fn scope<F: std::future::Future>(self, future: F) -> F::Output {
        let future = Box::pin(future);
        let attributed = Box::pin(async {
            if let Some(operation) = super::current_child_operation().or(self.child_operation) {
                Box::pin(super::scope_child_operation(operation, future)).await
            } else {
                future.await
            }
        });
        let future = Box::pin(super::scope_child_requirement(
            super::participants::current_child_requirement()
                .or(self.child_required)
                .unwrap_or(true),
            attributed,
        ));
        let work = async {
            if let Some(clock) = self.mutations {
                OBSERVATION_MUTATIONS.scope(clock, future).await
            } else {
                future.await
            }
        };
        Box::pin(ai_agents_llm::scope_invocation_evidence(
            self.evidence,
            scope_invocation_bounds(
                self.bounds,
                Box::pin(super::host_session::scope_session(self.host_session, work)),
            ),
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
