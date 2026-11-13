//! Restriction-only invocation deadlines shared by tools and frozen provider consumers.

use ai_agents_core::autonomy::TaskFinalAdmission;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Live time bounds restrict an invocation tree without granting execution, accounting or restart authority.
#[derive(Debug, Clone, Copy)]
pub struct InvocationBounds {
    utc: SystemTime,
    monotonic: Instant,
}

impl InvocationBounds {
    /// Retains the original wall and monotonic limits; neither clock can extend the other.
    pub fn new(utc: SystemTime, monotonic: Instant) -> Self {
        Self { utc, monotonic }
    }

    /// Returns the smaller remaining allowance, including expiry after a backwards wall-clock adjustment.
    pub fn remaining(self) -> Duration {
        self.utc
            .duration_since(SystemTime::now())
            .unwrap_or_default()
            .min(self.monotonic.saturating_duration_since(Instant::now()))
    }

    /// Nested work can only narrow its parent's limits.
    pub fn intersect(self, other: Self) -> Self {
        Self {
            utc: self.utc.min(other.utc),
            monotonic: self.monotonic.min(other.monotonic),
        }
    }
}

tokio::task_local! {
    static INVOCATION_BOUNDS: InvocationBounds;
    static INVOCATION_EVIDENCE: Vec<Arc<dyn TaskFinalAdmission>>;
}

/// Captures restriction-only live proof guards for managed child tasks and returned streams.
pub fn current_invocation_evidence() -> Vec<Arc<dyn TaskFinalAdmission>> {
    INVOCATION_EVIDENCE
        .try_with(Clone::clone)
        .unwrap_or_default()
}

/// Rechecks pinned host proof at actual admission; success grants no operation or resource authority.
pub fn validate_invocation_evidence() -> ai_agents_core::Result<()> {
    for guard in current_invocation_evidence() {
        guard.validate()?;
    }
    Ok(())
}

/// Rechecks live proof after dispatch while admission-only restrictions leave consumed work eligible for settlement.
/// This never permits another invocation; its pre-dispatch path still calls each guard's full validation.
pub(crate) fn validate_dispatched_evidence() -> ai_agents_core::Result<()> {
    for guard in current_invocation_evidence() {
        guard.validate_dispatched()?;
    }
    Ok(())
}

/// Nested work retains all parent proof guards and cannot replace them by passing an empty set.
pub async fn scope_invocation_evidence<F: Future>(
    guards: Vec<Arc<dyn TaskFinalAdmission>>,
    future: F,
) -> F::Output {
    let mut combined = current_invocation_evidence();
    for guard in guards {
        if !combined.iter().any(|old| Arc::ptr_eq(old, &guard)) {
            combined.push(guard);
        }
    }
    INVOCATION_EVIDENCE.scope(combined, future).await
}

/// Captures live restrictions for explicit propagation across framework-spawned Tokio tasks.
pub fn current_invocation_bounds() -> Option<InvocationBounds> {
    INVOCATION_BOUNDS.try_with(|bounds| *bounds).ok()
}

/// Installs intersected restrictions without replacing the existing invocation admission owner.
pub async fn scope_invocation_bounds<F: Future>(
    bounds: Option<InvocationBounds>,
    future: F,
) -> F::Output {
    let bounds = match (current_invocation_bounds(), bounds) {
        (Some(parent), Some(child)) => Some(parent.intersect(child)),
        (parent, child) => parent.or(child),
    };
    if let Some(bounds) = bounds {
        INVOCATION_BOUNDS.scope(bounds, future).await
    } else {
        future.await
    }
}
