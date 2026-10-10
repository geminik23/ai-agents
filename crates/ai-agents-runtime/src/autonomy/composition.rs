//! Exact composition frames bind parent dispatch to stable child operations without replaying input preparation.

use super::*;
use ai_agents_core::{AgentError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    sync::Arc,
};

/// Captures the delegated state's post-input-processing location before its child can suspend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DelegateFrame {
    pub version: u32,
    pub id: String,
    pub runtime_id: String,
    pub input: String,
    pub input_context: HashMap<String, Value>,
    pub delegate_id: String,
    pub delegate_runtime_id: String,
    pub delegate_input: String,
    pub definition: Value,
    pub actor: crate::TurnActorContext,
    pub parent_actor: Option<crate::TurnActorContext>,
    pub source: AutonomyTurnSource,
    pub child_operation: String,
    pub user_message_committed: bool,
    pub native_exchanges: Vec<TaskNativeExchange>,
}

/// A coordinating request has its own identity and refers to one exact parked child request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskGroupState {
    pub version: u32,
    pub request_id: String,
    pub frame: DelegateFrame,
    pub child_operation: String,
    pub child_runtime_id: String,
    pub child_request: TaskPendingRequest,
    pub batch: TaskBatchState,
    pub frames: BTreeMap<String, DelegateFrame>,
}

/// The private response is propagated by the composition call site, not deserialized as execution authority.
struct GroupResponse {
    operation: String,
    request_id: String,
    response: super::suspension::BatchResponse,
}

tokio::task_local! {
    static CHILD_INVOCATION: (String, String);
    static RESUMING_DELEGATE: String;
    static GROUP_RESPONSE: Arc<parking_lot::Mutex<Option<GroupResponse>>>;
}

/// Marks only the parent's saved dispatch frame as a resumed composition, without rerunning input processing.
pub(crate) async fn scope_resuming_delegate<F: Future>(runtime_id: String, future: F) -> F::Output {
    RESUMING_DELEGATE.scope(runtime_id, future).await
}

/// A resumed parent uses its retained frame; ordinary later turns must allocate a new frame instead.
pub(crate) fn resuming_delegate(runtime_id: &str) -> bool {
    RESUMING_DELEGATE
        .try_with(|active| active == runtime_id)
        .unwrap_or(false)
}

/// Stable composition identity survives reentry into the child dispatcher without allocating a new operation ordinal.
pub(crate) async fn scope_child_invocation<F: Future>(
    operation: String,
    runtime_id: String,
    future: F,
) -> F::Output {
    CHILD_INVOCATION
        .scope((operation, runtime_id), future)
        .await
}

/// Only framework composition callers can select a persisted child operation.
pub(crate) fn current_child_invocation(runtime_id: &str) -> Option<String> {
    CHILD_INVOCATION
        .try_with(Clone::clone)
        .ok()
        .filter(|(_, target)| target == runtime_id)
        .map(|(operation, _)| operation)
}

/// Keeps a single response bound to its exact operation while the parent resumes its committed composition.
pub(crate) async fn scope_group_response<F: Future>(
    operation: String,
    request_id: String,
    response: super::suspension::BatchResponse,
    future: F,
) -> F::Output {
    GROUP_RESPONSE
        .scope(
            Arc::new(parking_lot::Mutex::new(Some(GroupResponse {
                operation,
                request_id,
                response,
            }))),
            future,
        )
        .await
}

/// Other siblings cannot consume the leaf response, and a second poll cannot reuse it.
pub(crate) fn take_group_response(
    operation: &str,
    request_id: &str,
) -> Option<super::suspension::BatchResponse> {
    GROUP_RESPONSE
        .try_with(|slot| {
            let mut slot = slot.lock();
            if slot.as_ref().is_some_and(|response| {
                response.operation == operation && response.request_id == request_id
            }) {
                slot.take().map(|response| response.response)
            } else {
                None
            }
        })
        .ok()
        .flatten()
}

impl TaskGroupState {
    /// Exact parent/child references are validated before a safe pause can be exposed or a response can claim it.
    pub(crate) fn validate_checkpoint(&self, payload: &TaskCheckpointPayload) -> Result<()> {
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        if self.version != 1
            || self.request_id.is_empty()
            || self.frame.version != 1
            || self.frame.runtime_id != payload.runtime.snapshot.agent_id
            || self.frame.child_operation != self.child_operation
            || self.frame.delegate_runtime_id != self.child_runtime_id
            || self.frames.get(&self.frame.runtime_id).is_none_or(|frame| {
                serde_json::to_value(frame).ok() != serde_json::to_value(&self.frame).ok()
            })
        {
            return Err(invalid());
        }
        let root = payload.pending.as_ref().ok_or_else(invalid)?;
        if root.id != self.request_id
            || root.reviewed_action
                != json!({"group_id":self.frame.id,"child_operation":self.child_operation,"request":self.child_request})
            || serde_json::to_value(root.kind)? != serde_json::to_value(self.child_request.kind)?
        {
            return Err(invalid());
        }
        let TaskContinuation::Suspended {
            request_id,
            turn_id,
            batch: None,
            ..
        } = &payload.runtime.continuation
        else {
            return Err(invalid());
        };
        if *request_id != self.request_id || *turn_id != self.frame.id {
            return Err(invalid());
        }
        let child = payload
            .children
            .iter()
            .find(|child| child.child_id == self.child_operation)
            .ok_or_else(invalid)?;
        if child.runtime.snapshot.agent_id != self.child_runtime_id
            || child.result.is_some()
            || child.pending.as_ref().is_none_or(|pending| {
                serde_json::to_value(pending).ok() != serde_json::to_value(&self.child_request).ok()
            })
        {
            return Err(invalid());
        }
        let mut projection = payload.clone();
        projection.runtime = child.runtime.clone();
        projection.pending = Some(self.child_request.clone());
        self.batch.validate_checkpoint(&projection)
    }
}

impl RunExecution {
    /// Captures a quiescent group together with its exact parent frame; unknown or still-polled work cannot be parked.
    pub(crate) async fn pause_group(
        &self,
        mut runtime: TaskRuntimeCheckpoint,
        runtime_id: &str,
        todos: Option<TaskTodoCheckpoint>,
    ) -> Result<TaskRunSnapshot> {
        let _serial = self.serial.lock().await;
        let previous = self.load_owned().await?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(previous.payload.clone())?;
        self.check(&payload)?;
        if self.participants.unsettled()
            || payload.reservations.iter().any(|reservation| {
                matches!(
                    reservation.state,
                    TaskEffectState::Dispatched | TaskEffectState::Uncertain
                )
            })
            || payload
                .children
                .iter()
                .any(|child| child.result.is_none() && child.pending.is_none())
        {
            return Err(AgentError::Other(
                "group work is not acknowledged quiescent".into(),
            ));
        }
        let frames = self.delegate_frames.lock().clone();
        let frame = frames
            .get(runtime_id)
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let child = payload
            .children
            .iter()
            .find(|child| child.child_id == frame.child_operation)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let child_request = child
            .pending
            .clone()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let batch: TaskBatchState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| {
                    adapter.id == format!("runtime.child_batch:{}", frame.child_operation)
                })
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                .state
                .clone(),
        )?;
        let request_id = uuid::Uuid::new_v4().to_string();
        let group = TaskGroupState {
            version: 1,
            request_id: request_id.clone(),
            child_operation: frame.child_operation.clone(),
            child_runtime_id: child.runtime.snapshot.agent_id.clone(),
            child_request,
            batch,
            frame,
            frames,
        };
        if let Some(old) = payload.pending.take() {
            payload.consumed_request_ids.push(old.id);
        }
        payload.pending = Some(TaskPendingRequest {
            id: request_id.clone(),
            issued_revision: previous.revision + 1,
            kind: group.child_request.kind,
            reviewed_action: json!({"group_id":group.frame.id,"child_operation":group.child_operation,"request":group.child_request}),
        });
        runtime.continuation = TaskContinuation::Suspended {
            request_id: request_id.clone(),
            turn_id: group.frame.id.clone(),
            user_message_committed: group.frame.user_message_committed,
            finalized: false,
            skill: None,
            batch: None,
        };
        runtime.validate()?;
        payload.runtime = runtime;
        payload.todos = todos;
        payload.pause_reason = Some("child_interaction".into());
        payload.clocks.active_millis = self.active_millis()?;
        let adapter = TaskAdapterCheckpoint {
            id: "runtime.group".into(),
            adapter: "runtime.group".into(),
            contract_version: 1,
            config: json!({"runtime_id":runtime_id}),
            state: serde_json::to_value(&group)?,
        };
        if let Some(existing) = payload
            .adapters
            .iter_mut()
            .find(|existing| existing.id == adapter.id)
        {
            *existing = adapter;
        } else {
            payload.adapters.push(adapter);
        }
        group.validate_checkpoint(&payload)?;
        let child_deadline = self
            .approval_deadlines
            .lock()
            .get(&group.child_request.id)
            .copied();
        if let Some(deadline) = child_deadline {
            self.approval_deadlines.lock().insert(request_id, deadline);
        }
        let anchored = self
            .store
            .mutate(
                &self.run_id,
                &TaskRunMutation::Checkpoint {
                    expected_revision: previous.revision,
                    owner_token: self.owner_token.clone(),
                    status: TaskRunStatus::Running,
                    payload: serde_json::to_value(&payload)?,
                    release: false,
                },
            )
            .await
            .inspect_err(|_| self.stop("storage_failure"))?;
        self.acknowledge_snapshot(&anchored);
        self.revision
            .store(anchored.revision, std::sync::atomic::Ordering::Release);
        payload.clocks.active_millis = self.active_millis()?.max(
            payload
                .clocks
                .active_millis
                .checked_add(1)
                .ok_or_else(|| AgentError::Config("group active clock overflow".into()))?,
        );
        payload.clocks.active_interval_started_at = None;
        let saved = self
            .store
            .mutate(
                &self.run_id,
                &TaskRunMutation::FinalCheckpoint {
                    expected_revision: anchored.revision,
                    owner_token: self.owner_token.clone(),
                    status: TaskRunStatus::Paused,
                    payload: serde_json::to_value(payload)?,
                    release: true,
                    expires_at: self.expiry(),
                    deadline: std::time::Instant::now()
                        .checked_add(self.remaining_duration())
                        .ok_or_else(|| {
                            AgentError::Config("group pause deadline overflow".into())
                        })?,
                },
            )
            .await
            .inspect_err(|_| self.stop("storage_failure"))?;
        self.acknowledge_snapshot(&saved);
        self.revision
            .store(saved.revision, std::sync::atomic::Ordering::Release);
        Ok(saved)
    }

    /// Captures an exact parent frame before child work, with immutable dispatch/configuration binding.
    pub(crate) async fn retain_delegate_frame(&self, frame: DelegateFrame) -> Result<()> {
        if frame.version != 1 || frame.id.is_empty() || frame.child_operation.is_empty() {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        self.update(|payload| {
            let adapter = TaskAdapterCheckpoint {
                id: format!("runtime.delegate:{}", frame.id),
                adapter: "runtime.delegate".into(),
                contract_version: 1,
                config: json!({"runtime_id":frame.runtime_id,"definition":frame.definition}),
                state: serde_json::to_value(&frame)?,
            };
            if let Some(existing) = payload
                .adapters
                .iter()
                .find(|existing| existing.id == adapter.id)
            {
                if existing.config != adapter.config || existing.state != adapter.state {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
            } else {
                payload.adapters.push(adapter);
            }
            Ok(())
        })
        .await?;
        self.delegate_frames
            .lock()
            .insert(frame.runtime_id.clone(), frame);
        Ok(())
    }

    /// Admission uses a live framework-owned frame; a stored child ID alone cannot authorize suspension.
    pub(crate) fn has_composition_child(&self, operation: &str) -> bool {
        self.delegate_frames
            .lock()
            .values()
            .any(|frame| frame.child_operation == operation)
    }

    /// A parked child batch is usable only for the same operation, runtime and prepared input binding.
    pub(crate) async fn parked_child_batch(
        &self,
        operation: &str,
        input: &str,
        runtime_id: &str,
    ) -> Result<Option<(TaskPendingRequest, TaskBatchState)>> {
        let snapshot = self.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        let Some(child) = payload
            .children
            .iter()
            .find(|child| child.child_id == operation)
        else {
            return Ok(None);
        };
        let Some(pending) = &child.pending else {
            return Ok(None);
        };
        let binding = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == format!("child-operation:{operation}"))
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        if binding.config != json!({"input":input,"runtime_id":runtime_id})
            || child.runtime.snapshot.agent_id != runtime_id
        {
            return Err(AgentError::Config("parked child binding changed".into()));
        }
        let adapter = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == format!("runtime.child_batch:{operation}"))
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let batch: TaskBatchState = serde_json::from_value(adapter.state.clone())?;
        let mut projection = payload.clone();
        projection.runtime = child.runtime.clone();
        projection.pending = Some(pending.clone());
        batch.validate_checkpoint(&projection)?;
        Ok(Some((pending.clone(), batch)))
    }

    /// Acknowledged parking ends the active child lease without claiming that an unknown effect stopped.
    pub(crate) async fn checkpoint_child_park(
        &self,
        operation: &str,
        mut runtime: TaskRuntimeCheckpoint,
        batch: TaskBatchState,
    ) -> Result<()> {
        batch.bind_runtime(&mut runtime, &self.run_id, self.current_cycle())?;
        let approval = batch
            .approvals
            .first()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .clone();
        self.update(|payload| {
            let child = payload
                .children
                .iter_mut()
                .find(|child| child.child_id == operation)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if child.result.is_some() {
                return Err(TaskRunStorageError::Conflict.into());
            }
            if let Some(old) = child.pending.take()
                && old.id != approval.request_id
            {
                payload.consumed_request_ids.push(old.id);
            }
            child.runtime = runtime;
            child.pending = Some(TaskPendingRequest {
                id: approval.request_id.clone(),
                issued_revision: self.revision.load(std::sync::atomic::Ordering::Acquire) + 1,
                kind: if approval.question.is_some() {
                    TaskPendingKind::UserQuestion
                } else {
                    TaskPendingKind::Approval
                },
                reviewed_action: serde_json::to_value(&approval)?,
            });
            let adapter = TaskAdapterCheckpoint {
                id: format!("runtime.child_batch:{operation}"),
                adapter: "runtime.child_batch".into(),
                contract_version: 1,
                config: json!({"operation":operation,"runtime_id":child.runtime.snapshot.agent_id}),
                state: serde_json::to_value(&batch)?,
            };
            if let Some(existing) = payload
                .adapters
                .iter_mut()
                .find(|existing| existing.id == adapter.id)
            {
                *existing = adapter;
            } else {
                payload.adapters.push(adapter);
            }
            Ok(())
        })
        .await?;
        Ok(())
    }
}
