//! Host capability receipts qualify the actual implementation selected for this invocation.

use ai_agents_core::autonomy::{TaskFinalAdmission, ToolWriteAdmission, ToolWriteFootprint};
use ai_agents_core::{AgentError, Result, ToolExecutionContext};
use parking_lot::RwLock;
use std::sync::Arc;

struct HostBinding(Box<dyn Fn() -> bool + Send + Sync>);
impl std::fmt::Debug for HostBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HostWriteBinding")
    }
}
impl TaskFinalAdmission for HostBinding {
    fn validate(&self) -> Result<()> {
        if (self.0)() {
            Ok(())
        } else {
            Err(AgentError::Tool(
                "host implementation changed during write admission".into(),
            ))
        }
    }
}

// This process-local receipt is inert checkpoint data, never restoration authority; the paired guard retains the real allocation.
fn prefix<T: ?Sized>(provider: &Arc<T>) -> String {
    format!("host:{:p}:", Arc::as_ptr(provider))
}

fn bind<T: ?Sized>(provider: &Arc<T>, mut footprint: ToolWriteFootprint) -> ToolWriteFootprint {
    footprint.binding_identity = format!("{}{}", prefix(provider), footprint.binding_identity);
    footprint.targets.sort();
    footprint.targets.dedup();
    footprint
}

// The same captured Arc computes the quote and stays alive through admission, body and settlement.
pub(crate) fn prepare<T: ?Sized + Send + Sync + 'static>(
    slot: &Arc<RwLock<Arc<T>>>,
    captured: Arc<T>,
    footprint: Option<ToolWriteFootprint>,
) -> Option<ToolWriteAdmission> {
    footprint.map(|footprint| {
        let footprint = bind(&captured, footprint);
        let slot = slot.clone();
        let binding = Arc::new(HostBinding(Box::new(move || {
            Arc::ptr_eq(&slot.read(), &captured)
        })));
        ToolWriteAdmission {
            footprint,
            binding: Some(binding),
        }
    })
}

pub(crate) fn prepare_optional<T: ?Sized + Send + Sync + 'static>(
    slot: &Arc<RwLock<Option<Arc<T>>>>,
    captured: Option<Arc<T>>,
    footprint: Option<ToolWriteFootprint>,
) -> Option<ToolWriteAdmission> {
    footprint.map(|footprint| {
        let footprint = captured.as_ref().map_or_else(
            || footprint.clone(),
            |provider| bind(provider, footprint.clone()),
        );
        let slot = slot.clone();
        let binding = Arc::new(HostBinding(Box::new(move || {
            match (slot.read().as_ref(), captured.as_ref()) {
                (Some(current), Some(captured)) => Arc::ptr_eq(current, captured),
                (None, None) => true,
                _ => false,
            }
        })));
        ToolWriteAdmission {
            footprint,
            binding: Some(binding),
        }
    })
}

// Execution captures the selected Arc and never re-runs a potentially slow host quote after shared final admission.
pub(crate) fn verify_host<T: ?Sized>(ctx: &ToolExecutionContext, provider: &Arc<T>) -> Result<()> {
    if let Some(admitted) = &ctx.admitted_write_footprint {
        if !admitted.binding_identity.starts_with(&prefix(provider)) {
            return Err(changed());
        }
        // Proof sampling may itself take time; check clocks after it and immediately before the actual callback.
        let proof = ai_agents_llm::validate_invocation_evidence();
        let admission = ai_agents_llm::current_invocation_admission();
        let stopped = admission
            .as_ref()
            .is_some_and(|admission| admission.execution_stopped());
        let remaining = admission
            .as_ref()
            .and_then(|admission| admission.remaining_duration());
        if proof.is_err()
            || stopped
            || ctx.cancellation.is_cancelled()
            || remaining.is_some_and(|remaining| remaining.is_zero())
            || ai_agents_llm::current_invocation_bounds()
                .is_some_and(|bounds| bounds.remaining().is_zero())
            || ctx
                .deadline
                .is_some_and(|deadline| chrono::Utc::now() >= deadline)
        {
            return Err(AgentError::Tool(
                "host invocation exceeded its original authority".into(),
            ));
        }
    }
    Ok(())
}

fn changed() -> AgentError {
    AgentError::Tool("host footprint or actual implementation changed before invocation".into())
}

pub(crate) fn verify(
    ctx: &ToolExecutionContext,
    current: Option<ToolWriteFootprint>,
) -> Result<()> {
    if ctx
        .admitted_write_footprint
        .as_ref()
        .is_some_and(|admitted| current.as_ref() != Some(admitted))
    {
        return Err(changed());
    }
    Ok(())
}
