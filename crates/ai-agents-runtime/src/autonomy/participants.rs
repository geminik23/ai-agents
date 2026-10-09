//! Live child reservations and durable child outcomes share the coordinating run's admission and cleanup boundary.

use super::*;
use ai_agents_core::{AgentError, AgentResponse, Result};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

tokio::task_local! { static CHILD_REQUIRED: bool; static CHILD_OPERATION: String; }

/// Partial-failure policy is captured by the framework call site, not by a model-supplied source label.
pub(crate) async fn scope_child_requirement<F: std::future::Future>(
    required: bool,
    future: F,
) -> F::Output {
    CHILD_REQUIRED.scope(required, future).await
}

/// Dependency outcomes are mandatory unless an existing composition policy explicitly tolerates failure.
pub(crate) fn child_required() -> bool {
    CHILD_REQUIRED
        .try_with(|required| *required)
        .unwrap_or(true)
}

/// Attribution scopes keep native call IDs unchanged while distinguishing identical IDs in different child operations.
pub(crate) async fn scope_child_operation<F: std::future::Future>(
    operation: String,
    future: F,
) -> F::Output {
    CHILD_OPERATION.scope(operation, future).await
}

/// The execution-local operation identity is evidence attribution, not a serialized permission.
pub(crate) fn current_child_operation() -> Option<String> {
    CHILD_OPERATION.try_with(Clone::clone).ok()
}

struct Participant {
    owner: Arc<RunOwner>,
    slot: boundary::RunOwnerSlot,
}

#[derive(Default)]
pub(crate) struct Participants {
    members: parking_lot::Mutex<Vec<Participant>>,
    active: AtomicUsize,
    uncertain: AtomicBool,
    next_operation: parking_lot::Mutex<std::collections::BTreeMap<String, u64>>,
    changed: tokio::sync::Notify,
}

impl Participants {
    /// Only the exact live gate capability is an admitted participant; serialized child IDs never authorize entry.
    pub(crate) fn owner_for(&self, gate: &crate::runtime::RootTurnGate) -> Option<Arc<RunOwner>> {
        self.members
            .lock()
            .iter()
            .find(|member| Arc::ptr_eq(&member.owner.gate, gate))
            .map(|member| member.owner.clone())
    }

    /// Keeps child ownership across coordinating turns instead of exposing an inter-turn mutation gap.
    pub(crate) fn owns(&self, owner: &Arc<RunOwner>) -> bool {
        self.members
            .lock()
            .iter()
            .any(|member| Arc::ptr_eq(&member.owner, owner))
    }

    /// Enrollment is called under the shared admission mutex and the child's local gate.
    pub(crate) fn enroll(&self, owner: Arc<RunOwner>, slot: boundary::RunOwnerSlot) -> Result<()> {
        let mut members = self.members.lock();
        if let Some(member) = members
            .iter()
            .find(|member| Arc::ptr_eq(&member.owner.gate, &owner.gate))
        {
            if !Arc::ptr_eq(&member.owner, &owner) {
                return Err(AgentError::Other("child owner mismatch".into()));
            }
        } else {
            if members.len() >= MAX_TASK_CHECKPOINT_RECORDS {
                return Err(TaskRunStorageError::CheckpointTooLarge.into());
            }
            members.push(Participant {
                owner: owner.clone(),
                slot: slot.clone(),
            });
        }
        *slot.write() = Some(owner);
        self.active.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// A missing child result or unacknowledged child cleanup prevents a truthful successful terminal checkpoint.
    pub(crate) fn unsettled(&self) -> bool {
        self.active.load(Ordering::Acquire) != 0 || self.uncertain.load(Ordering::Acquire)
    }

    /// Operation IDs distinguish repeated legitimate requests to one runtime without overwriting completed results.
    pub(crate) fn next_operation(&self, runtime_id: &str, cycle: u64) -> String {
        let mut cursors = self.next_operation.lock();
        let ordinal = cursors
            .entry(json!({"runtime":runtime_id,"cycle":cycle}).to_string())
            .or_default();
        let id = json!({"runtime":runtime_id,"cycle":cycle,"ordinal":*ordinal}).to_string();
        *ordinal += 1;
        id
    }

    /// Release follows terminal acknowledgement and quiescence; uncertain owners remain reserved for recovery.
    pub(crate) async fn release(&self) -> Result<()> {
        if self.unsettled() {
            return Err(AgentError::Other(
                "child work is not acknowledged quiescent".into(),
            ));
        }
        let members: Vec<_> = self
            .members
            .lock()
            .iter()
            .map(|member| (member.owner.clone(), member.slot.clone()))
            .collect();
        for (owner, slot) in members {
            let _gate = owner.gate.lock().await;
            let mut current = slot.write();
            if current
                .as_ref()
                .is_none_or(|active| !Arc::ptr_eq(active, &owner))
            {
                return Err(AgentError::Other("child release owner mismatch".into()));
            }
            *current = None;
        }
        self.members.lock().clear();
        Ok(())
    }
}

pub(crate) struct ChildLease {
    execution: Arc<RunExecution>,
    acknowledged: bool,
}

impl ChildLease {
    /// Starts only after enrollment succeeds; cancellation cannot release an unrecorded dispatched child.
    pub(crate) fn new(execution: &Arc<RunExecution>) -> Self {
        Self {
            execution: execution.clone(),
            acknowledged: false,
        }
    }

    /// A durable child outcome must be acknowledged before the child ceases to be owned in-flight work.
    pub(crate) fn acknowledge(&mut self) {
        self.acknowledged = true;
    }
}

impl Drop for ChildLease {
    /// Drop cannot checkpoint; stop new work and retain every child reservation until explicit recovery.
    fn drop(&mut self) {
        if !self.acknowledged {
            self.execution
                .participants
                .uncertain
                .store(true, Ordering::Release);
            self.execution.stop("uncertain_child");
        }
        self.execution
            .participants
            .active
            .fetch_sub(1, Ordering::AcqRel);
        self.execution.participants.changed.notify_waiters();
    }
}

impl RunExecution {
    /// Child operations use the coordinator cycle and a runtime-local ordinal; concurrent different children do not reorder identities.
    pub(crate) fn next_child_operation(&self, runtime_id: &str) -> String {
        self.participants
            .next_operation(runtime_id, self.current_cycle())
    }

    /// Completed child results are reusable only for the same input and runtime binding, never by a free-form child ID alone.
    pub(crate) async fn cached_child(
        &self,
        operation: &str,
        input: &str,
        runtime_id: &str,
    ) -> Result<Option<Result<AgentResponse>>> {
        let _serial = self.serial.lock().await;
        let snapshot = self.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        let Some(child) = payload
            .children
            .iter()
            .find(|child| child.child_id == operation)
        else {
            return Ok(None);
        };
        let binding = payload
            .adapters
            .iter()
            .find(|binding| binding.id == format!("child-operation:{operation}"))
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        if binding.config != json!({"input":input,"runtime_id":runtime_id}) {
            return Err(AgentError::Config("child operation binding changed".into()));
        }
        let Some(result) = &child.result else {
            return Err(AgentError::Other(
                "child operation requires recovery, not replay".into(),
            ));
        };
        if let Some(response) = result.get("response") {
            return Ok(Some(Ok(serde_json::from_value(response.clone())?)));
        }
        Ok(Some(Err(AgentError::Other(
            result
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("child failed")
                .into(),
        ))))
    }

    /// Child setup cannot race the terminal transition or replace a foreign runtime owner.
    pub(crate) async fn enroll_child(
        self: &Arc<Self>,
        owner: Arc<RunOwner>,
        slot: boundary::RunOwnerSlot,
    ) -> Result<ChildLease> {
        let _serial = self.serial.lock().await;
        let snapshot = self.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        self.check(&payload)?;
        self.participants.enroll(owner, slot)?;
        Ok(ChildLease::new(self))
    }

    /// Stores the exact child runtime before its first effect, then an immutable result after its local turn settles.
    pub(crate) async fn checkpoint_child(
        &self,
        operation: &str,
        runtime: TaskRuntimeCheckpoint,
        input: &str,
        outcome: Option<&Result<AgentResponse>>,
    ) -> Result<()> {
        let result = outcome.map(|outcome| match outcome {
            Ok(response) => json!({"response":response}),
            Err(error) => json!({"error":error.to_string()}),
        });
        self.update(|payload| {
            if let Some(child) = payload
                .children
                .iter_mut()
                .find(|child| child.child_id == operation)
            {
                if child.result.is_some() {
                    return Err(TaskRunStorageError::Conflict.into());
                }
                child.runtime = runtime;
                child.result = result;
            } else {
                if outcome.is_some() {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
                payload.adapters.push(TaskAdapterCheckpoint {
                    id: format!("child-operation:{operation}"),
                    adapter: "runtime.child".into(),
                    contract_version: 1,
                    config: json!({"input":input,"runtime_id":runtime.snapshot.agent_id}),
                    state: Value::Null,
                });
                payload.children.push(TaskChildCheckpoint {
                    child_id: operation.into(),
                    parent_run_id: self.run_id.clone(),
                    config_identity: format!(
                        "{}:{}",
                        payload.config_identity, runtime.snapshot.agent_id
                    ),
                    runtime,
                    pending: None,
                    result: None,
                });
            }
            Ok(())
        })
        .await?;
        Ok(())
    }

    /// Waits for owned detached work before observations; deadline/cancellation never claims that unknown child work stopped.
    pub(crate) async fn await_children(&self) -> Result<()> {
        loop {
            let notified = self.participants.changed.notified();
            if self.participants.active.load(Ordering::Acquire) == 0 {
                return Ok(());
            }
            if ai_agents_core::autonomy::InvocationAdmission::execution_stopped(self) {
                self.participants.uncertain.store(true, Ordering::Release);
                return Err(AgentError::Other("child work requires recovery".into()));
            }
            tokio::select! { _ = notified => {}, _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {} }
        }
    }
}
