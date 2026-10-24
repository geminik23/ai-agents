//! Run-owned maintenance tails remain protected after provider settlement and across spawned task boundaries.

use super::*;
use ai_agents_core::{AgentError, Result};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

tokio::task_local! { static MAINTENANCE_EFFECTS: Arc<std::sync::atomic::AtomicBool>; }

/// Marks the first managed cache, persistence or host-callback effect before polling its implementation.
pub(crate) fn maintenance_effect() {
    let _ = MAINTENANCE_EFFECTS
        .try_with(|effects| effects.store(true, std::sync::atomic::Ordering::Release));
}

#[derive(Default)]
struct WorkState {
    active: usize,
    uncertain: bool,
    closed: bool,
    sealed: bool,
}

/// Tracks actual owned work, not task labels or the number of handles currently in a queue.
#[derive(Default)]
pub(crate) struct OwnedWork {
    state: parking_lot::Mutex<WorkState>,
    changed: tokio::sync::Notify,
}

impl OwnedWork {
    // Enrollment is synchronous with scheduling, so an unpolled queued body is not invisible to a pause or terminal barrier.
    fn enroll(self: &Arc<Self>) -> Result<WorkLease> {
        let mut state = self.state.lock();
        if state.closed || state.sealed {
            return Err(AgentError::Other("task work admission is closed".into()));
        }
        state.active = state
            .active
            .checked_add(1)
            .ok_or_else(|| AgentError::Other("task work count overflow".into()))?;
        Ok(WorkLease {
            work: self.clone(),
            dispatched: false,
            acknowledged: false,
        })
    }

    /// Quiescence includes queued and draining bodies and unacknowledged effect tails.
    pub(crate) fn unsettled(&self) -> bool {
        let state = self.state.lock();
        state.active != 0 || state.uncertain
    }

    /// Seals a quiescent snapshot-to-publication interval against late producers without releasing actual run ownership.
    pub(crate) fn seal(self: &Arc<Self>) -> Result<WorkBarrier> {
        let mut state = self.state.lock();
        if state.active != 0 || state.uncertain || state.closed || state.sealed {
            return Err(AgentError::Other(
                "task work is not acknowledged quiescent".into(),
            ));
        }
        state.sealed = true;
        Ok(WorkBarrier(self.clone()))
    }

    /// Terminal closure prevents a captured producer from enrolling after the final snapshot is sampled.
    pub(crate) fn close(&self) {
        self.state.lock().closed = true;
        self.changed.notify_waiters();
    }

    /// Explicit host reconciliation may retire uncertainty only after every caller-owned future has actually left its lease.
    pub(crate) fn acknowledge_reconciliation(&self) -> Result<()> {
        let mut state = self.state.lock();
        if state.active != 0 {
            return Err(AgentError::Other("task maintenance has not stopped".into()));
        }
        state.closed = true;
        state.uncertain = false;
        Ok(())
    }

    /// Closed admission cannot be reopened by a stale execution scope or a provider fallback.
    pub(crate) fn closed(&self) -> bool {
        self.state.lock().closed
    }
}

pub(crate) struct WorkBarrier(Arc<OwnedWork>);

impl Drop for WorkBarrier {
    // Releasing a local snapshot barrier grants no execution: the ledger's current claim and revision remain authoritative.
    fn drop(&mut self) {
        self.0.state.lock().sealed = false;
    }
}

struct WorkLease {
    work: Arc<OwnedWork>,
    dispatched: bool,
    acknowledged: bool,
}

impl Drop for WorkLease {
    // A dropped storage admission is ambiguous; a body never polled before admission has no effect to reconcile.
    fn drop(&mut self) {
        let mut state = self.work.state.lock();
        state.active -= 1;
        state.uncertain |= self.dispatched && !self.acknowledged;
        drop(state);
        self.work.changed.notify_waiters();
    }
}

/// Captures the existing ledger and restriction tree before spawning without creating another owner or invocation allowance.
/// The logical maintenance marker covers persistence and hooks; its inner provider calls retain their own counted attempts.
/// Erasing this optional tail keeps construction and polling frames bounded in recursive state and composed runtime paths.
pub(crate) fn owned_maintenance<'a, F, T>(
    label: &'static str,
    future: F,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send + 'a>>
where
    F: std::future::Future<Output = Result<T>> + Send + 'a,
    T: Send + 'a,
{
    let future = Box::pin(future);
    let execution = current_execution();
    let effects = MAINTENANCE_EFFECTS.try_with(Arc::clone).unwrap_or_default();
    let invocation = invocation::InvocationScope::capture();
    let lease = execution
        .as_ref()
        .map(|execution| execution.owner.work.enroll());
    Box::pin(async move {
        let Some(execution) = execution else {
            return future.await;
        };
        scope_inherited_execution(Some(execution.clone()), invocation, async move {
        let mut lease = lease.ok_or(TaskRunStorageError::InvalidCheckpoint)??;
        let snapshot = execution.load_owned().await?;
        let payload = serde_json::from_value(snapshot.payload)?;
        execution.check(&payload)?;
        let id = uuid::Uuid::new_v4().to_string();
        lease.dispatched = true;
        execution
            .update(|payload| {
                execution.check(payload)?;
                payload.adapters.push(TaskAdapterCheckpoint {
                    id: format!("maintenance:{id}"),
                    adapter: "runtime.maintenance".into(),
                    contract_version: 1,
                    config: json!({"purpose":label}),
                    state: serde_json::Value::Null,
                });
                payload.reservations.push(TaskReservation {
                    id: id.clone(),
                    state: TaskEffectState::Dispatched,
                    reserved_micro_usd: 0,
                    charged_micro_usd: 0,
                    write_targets: vec![],
                    result: None,
                });
                Ok(())
            })
            .await?;
        let run = execution.clone();
        let started_body = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let started = started_body.clone();
        let effects_for_body = effects.clone();
        let result = MAINTENANCE_EFFECTS.scope(effects_for_body, async move {
            // The storage acknowledgement may have outlived the original authority; no body poll races the first timer tick.
            if ai_agents_core::autonomy::InvocationAdmission::execution_stopped(run.as_ref())
                || ai_agents_llm::current_invocation_bounds().is_some_and(|bounds| bounds.remaining().is_zero())
            {
                return Err(AgentError::Other("task maintenance admission expired before body dispatch".into()));
            }
            ai_agents_llm::validate_invocation_evidence()?;
            started.store(true, std::sync::atomic::Ordering::Release);
            let mut body = Box::pin(future);
            let mut tick = tokio::time::interval(Duration::from_millis(10));
            let result = {
            let mut watched = std::pin::pin!(std::future::poll_fn(|cx| {
                if ai_agents_core::autonomy::InvocationAdmission::execution_stopped(run.as_ref())
                    || ai_agents_llm::current_invocation_bounds().is_some_and(|bounds| bounds.remaining().is_zero())
                    || ai_agents_llm::validate_invocation_evidence().is_err()
                {
                    return std::task::Poll::Ready(Err(AgentError::Other("task maintenance authority changed before polling its body".into())));
                }
                body.as_mut().poll(cx)
            }));
            loop {
                tokio::select! {
                    result = &mut watched => break result,
                    _ = tick.tick() => {
                        if ai_agents_core::autonomy::InvocationAdmission::execution_stopped(run.as_ref())
                            || ai_agents_llm::current_invocation_bounds().is_some_and(|bounds| bounds.remaining().is_zero())
                        {
                            break Err(AgentError::Other("task maintenance stopped before acknowledged completion".into()));
                        }
                    }
                }
            }
            };
            drop(body);
            if ai_agents_core::autonomy::InvocationAdmission::execution_stopped(run.as_ref())
                || ai_agents_llm::current_invocation_bounds().is_some_and(|bounds| bounds.remaining().is_zero())
            {
                return Err(AgentError::Other("task maintenance exceeded its active authority".into()));
            }
            ai_agents_llm::validate_invocation_evidence()?;
            result
        }).await;
        let data = match &result {
            Ok(_) => json!({"maintenance":label,"completed":true}),
            Err(error) => json!({"maintenance":label,"error":error.to_string()}),
        };
        // Returned failure is not proof that an opaque store did not commit; never replay the tail automatically.
        // This is a zero-count logical marker: retiring a proven unstarted body never refunds an inner provider/tool slot.
        execution
            .settle(
                &id,
                data,
                result.is_err()
                    && started_body.load(std::sync::atomic::Ordering::Acquire)
                    && effects.load(std::sync::atomic::Ordering::Acquire),
            )
            .await?;
        lease.acknowledged = true;
        if result.is_err() {
            execution.stop_maintenance(effects.load(std::sync::atomic::Ordering::Acquire));
        }
        result
        }).await
    })
}

#[cfg(test)]
#[path = "ad_work_tests.rs"]
mod ad_tests;

impl RunExecution {
    /// All runtime participants share this barrier, including a child's spawned maintenance after its model result returns.
    /// Cancellation or expiry stops new work but does not certify that an already admitted tail is quiescent.
    pub(crate) async fn await_owned_work(&self) -> Result<()> {
        loop {
            let notified = self.owner.work.changed.notified();
            if !self.owner.work.unsettled() {
                return Ok(());
            }
            if self.owner.work.state.lock().uncertain {
                self.stop("uncertain_maintenance");
                return Err(AgentError::Other(
                    "task maintenance acknowledgement is uncertain".into(),
                ));
            }
            if ai_agents_core::autonomy::InvocationAdmission::execution_stopped(self) {
                return Err(AgentError::Other(
                    "task maintenance requires recovery".into(),
                ));
            }
            tokio::select! { _ = notified => {}, _ = tokio::time::sleep(Duration::from_millis(10)) => {} }
        }
    }
}
