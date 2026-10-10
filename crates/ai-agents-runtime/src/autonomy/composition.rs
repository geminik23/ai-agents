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
    #[serde(default)]
    pub dispatch: CompositionDispatch,
    #[serde(default)]
    pub children: Vec<CompositionChild>,
    #[serde(default)]
    pub calls: Vec<CompositionChild>,
    #[serde(default)]
    pub cursor: Value,
    #[serde(default)]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub parent_operation: Option<String>,
    #[serde(default = "required_by_default")]
    pub required: bool,
}

/// Older direct-delegation frames used the default mandatory-child policy.
fn required_by_default() -> bool {
    true
}

/// Dispatch coordinates are framework-owned; a serialized variant does not grant runtime access.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CompositionDispatch {
    #[default]
    Delegate,
    Concurrent,
    Pipeline,
    Handoff,
    GroupChat,
    ToolMessage,
    ToolBatch,
}

/// Stable slot identities distinguish repeated calls to the same runtime without allocating new operations on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompositionChild {
    pub registry_id: String,
    pub runtime_id: String,
    pub operation: String,
}

/// Every safely parked leaf remains bound to its own exact history and single-use request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ParkedCompositionChild {
    pub operation: String,
    pub runtime_id: String,
    pub request: TaskPendingRequest,
    pub batch: TaskBatchState,
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
    #[serde(default)]
    pub parked: Vec<ParkedCompositionChild>,
}

/// The private response is propagated by the composition call site, not deserialized as execution authority.
pub(crate) struct GroupResponse {
    operation: String,
    request_id: String,
    response: super::suspension::BatchResponse,
}

tokio::task_local! {
    static CHILD_INVOCATION: (String, String);
    static RESUMING_DELEGATE: String;
    static GROUP_RESPONSE: Arc<parking_lot::Mutex<Option<GroupResponse>>>;
    static COMPOSITION_SCOPE: CompositionScope;
    static COMPOSITION_DISPATCH: bool;
    static COMPOSITION_ENTRY: std::cell::Cell<bool>;
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

impl DelegateFrame {
    /// Reachability follows stable frame edges rather than granting authority to a free-form child operation.
    pub(crate) fn contains_operation(
        &self,
        operation: &str,
        frames: &BTreeMap<String, DelegateFrame>,
        ancestry: &mut Vec<String>,
    ) -> bool {
        if ancestry.len() >= 32 || ancestry.contains(&self.runtime_id) {
            return false;
        }
        ancestry.push(self.runtime_id.clone());
        let found = self.child_operation == operation
            || self
                .children
                .iter()
                .chain(&self.calls)
                .any(|slot| slot.operation == operation)
            || self
                .children
                .iter()
                .chain(&self.calls)
                .map(|slot| (&slot.operation, &slot.runtime_id))
                .chain(
                    (self.dispatch == CompositionDispatch::Delegate)
                        .then_some((&self.child_operation, &self.delegate_runtime_id)),
                )
                .filter_map(|(incoming, runtime)| {
                    frames
                        .get(runtime)
                        .filter(|frame| frame.parent_operation.as_ref() == Some(incoming))
                })
                .any(|frame| frame.contains_operation(operation, frames, ancestry));
        ancestry.pop();
        found
    }
}

/// An intermediate parent may resume only when the live single-use response targets one of its saved descendants.
pub(crate) fn response_targets_frame(
    frame: &DelegateFrame,
    frames: &BTreeMap<String, DelegateFrame>,
) -> bool {
    GROUP_RESPONSE
        .try_with(|custody| {
            custody.lock().as_ref().is_some_and(|response| {
                frame.contains_operation(&response.operation, frames, &mut Vec::new())
            })
        })
        .unwrap_or(false)
}

impl TaskGroupState {
    /// Exact parent/child references are validated before a safe pause can be exposed or a response can claim it.
    pub(crate) fn validate_checkpoint(&self, payload: &TaskCheckpointPayload) -> Result<()> {
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        if self.version != 1
            || self.request_id.is_empty()
            || self.frame.version != 1
            || self.frame.runtime_id != payload.runtime.snapshot.agent_id
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
            batch: parent_cursor,
            ..
        } = &payload.runtime.continuation
        else {
            return Err(invalid());
        };
        if self.frame.dispatch == CompositionDispatch::ToolBatch {
            let batch: TaskBatchState = serde_json::from_value(self.frame.cursor.clone())?;
            batch.validate_model_history(&payload.runtime)?;
            super::message::validate_message_batch(&self.frame, &self.frames, &batch, payload)?;
            let mut projection = payload.runtime.clone();
            batch.bind_runtime(&mut projection, "message", 0)?;
            let TaskContinuation::Suspended {
                batch: expected, ..
            } = projection.continuation
            else {
                return Err(invalid());
            };
            if serde_json::to_value(parent_cursor)? != serde_json::to_value(expected)? {
                return Err(invalid());
            }
        } else if parent_cursor.is_some() {
            return Err(invalid());
        }
        if *request_id != self.request_id
            || (self.frame.dispatch != CompositionDispatch::ToolBatch && *turn_id != self.frame.id)
        {
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
        self.batch.validate_checkpoint(&projection)?;
        if !self.parked.is_empty() {
            let mut operations = std::collections::HashSet::new();
            let mut requests = std::collections::HashSet::new();
            if self.parked.is_empty()
                || !self.parked.iter().any(|leaf| {
                    leaf.operation == self.child_operation
                        && leaf.runtime_id == self.child_runtime_id
                        && serde_json::to_value(&leaf.request).ok()
                            == serde_json::to_value(&self.child_request).ok()
                        && serde_json::to_value(&leaf.batch).ok()
                            == serde_json::to_value(&self.batch).ok()
                })
            {
                return Err(invalid());
            }
            for leaf in &self.parked {
                if !operations.insert(&leaf.operation)
                    || !requests.insert(&leaf.request.id)
                    || !self.frames.values().any(|frame| {
                        (frame.dispatch == CompositionDispatch::Delegate
                            && frame.child_operation == leaf.operation
                            && frame.delegate_runtime_id == leaf.runtime_id)
                            || frame.children.iter().chain(&frame.calls).any(|slot| {
                                slot.operation == leaf.operation
                                    && slot.runtime_id == leaf.runtime_id
                            })
                    })
                {
                    return Err(invalid());
                }
                let child = payload
                    .children
                    .iter()
                    .find(|child| child.child_id == leaf.operation)
                    .ok_or_else(invalid)?;
                if child.runtime.snapshot.agent_id != leaf.runtime_id
                    || child.result.is_some()
                    || serde_json::to_value(&child.pending)?
                        != serde_json::to_value(Some(&leaf.request))?
                {
                    return Err(invalid());
                }
                let mut projection = payload.clone();
                projection.runtime = child.runtime.clone();
                projection.pending = Some(leaf.request.clone());
                leaf.batch.validate_checkpoint(&projection)?;
            }
            let reachable =
                collect_parked_leaves(&self.frame, &self.frames, payload, &mut Vec::new())?;
            if serde_json::to_value(reachable)? != serde_json::to_value(&self.parked)? {
                return Err(invalid());
            }
        }
        Ok(())
    }
}

/// Walks only active pending frame edges, preserving declaration order and rejecting cycles or missing leaf continuation.
fn collect_parked_leaves(
    frame: &DelegateFrame,
    frames: &BTreeMap<String, DelegateFrame>,
    payload: &TaskCheckpointPayload,
    ancestry: &mut Vec<String>,
) -> Result<Vec<ParkedCompositionChild>> {
    if ancestry.len() >= 32 || ancestry.contains(&frame.runtime_id) {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    ancestry.push(frame.runtime_id.clone());
    let slots = if frame.dispatch == CompositionDispatch::Delegate {
        vec![CompositionChild {
            registry_id: frame.delegate_id.clone(),
            runtime_id: frame.delegate_runtime_id.clone(),
            operation: frame.child_operation.clone(),
        }]
    } else {
        frame.children.iter().chain(&frame.calls).cloned().collect()
    };
    let mut leaves = Vec::new();
    for slot in slots {
        let Some(child) = payload
            .children
            .iter()
            .find(|child| child.child_id == slot.operation)
        else {
            continue;
        };
        let Some(request) = &child.pending else {
            continue;
        };
        if child.runtime.snapshot.agent_id != slot.runtime_id || child.result.is_some() {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        if matches!(
            child.runtime.continuation,
            TaskContinuation::Suspended { batch: Some(_), .. }
        ) && request.reviewed_action.get("frame_id").is_none()
        {
            let adapter = payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == format!("runtime.child_batch:{}", slot.operation))
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            leaves.push(ParkedCompositionChild {
                operation: slot.operation,
                runtime_id: slot.runtime_id,
                request: request.clone(),
                batch: serde_json::from_value(adapter.state.clone())?,
            });
        } else {
            let nested = frames
                .get(&slot.runtime_id)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if nested.parent_operation.as_deref() != Some(slot.operation.as_str()) {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            leaves.extend(collect_parked_leaves(nested, frames, payload, ancestry)?);
        }
    }
    ancestry.pop();
    Ok(leaves)
}

/// Admission closure is local to one parent dispatch and never changes permanent cancellation state.
#[derive(Clone)]
pub(crate) struct CompositionScope {
    pub runtime_id: String,
    frame_key: String,
    closed: Arc<parking_lot::Mutex<bool>>,
    deadline: Option<std::time::Instant>,
}

/// Captures the same private response custody before a JoinSet boundary; it remains single-use across all siblings.
pub(crate) fn group_response_custody() -> Option<Arc<parking_lot::Mutex<Option<GroupResponse>>>> {
    GROUP_RESPONSE.try_with(Clone::clone).ok()
}

/// Closes new child admission at the first safe interaction point while admitted child turns drain normally.
pub(crate) fn close_composition_admission() {
    if let Ok(scope) = COMPOSITION_SCOPE.try_with(Clone::clone) {
        *scope.closed.lock() = true;
    }
}

/// Final child enrollment observes the same barrier as parking; waiting for a runtime gate does not authorize a later start.
pub(crate) fn composition_admission_denial() -> Option<String> {
    COMPOSITION_SCOPE
        .try_with(|scope| (*scope.closed.lock()).then(|| scope.runtime_id.clone()))
        .ok()
        .flatten()
}

/// Only a live parent scope can select a saved slot or cursor; persisted metadata does not establish authority.
pub(crate) fn current_composition() -> Option<CompositionScope> {
    if !COMPOSITION_DISPATCH
        .try_with(|active| *active)
        .unwrap_or(false)
    {
        return None;
    }
    COMPOSITION_SCOPE.try_with(Clone::clone).ok()
}

/// Creates a fresh admission interval for a retained parent; original timeouts stay in its saved cursor.
pub(crate) async fn scope_composition<F: Future>(runtime_id: String, future: F) -> F::Output {
    scope_composition_frame(runtime_id.clone(), runtime_id, future).await
}

/// Message calls select their own frame and deadline, never a sibling's runtime-wide coordinator.
pub(crate) async fn scope_message_composition<F: Future>(
    runtime_id: String,
    frame_id: String,
    future: F,
) -> F::Output {
    scope_composition_frame(runtime_id, frame_id, future).await
}

/// Private frame selection carries a distinct admission interval while inherited deadlines only narrow it.
async fn scope_composition_frame<F: Future>(
    runtime_id: String,
    frame_key: String,
    future: F,
) -> F::Output {
    let inherited = current_composition_deadline();
    let own = current_execution().and_then(|execution| {
        let frame = execution.delegate_frames.lock().get(&frame_key).cloned()?;
        execution
            .approval_deadlines
            .lock()
            .get(&format!("composition:{}", frame.id))
            .copied()
    });
    let deadline = match (inherited, own) {
        (Some(parent), Some(own)) => Some(parent.min(own)),
        (parent, own) => parent.or(own),
    };
    COMPOSITION_ENTRY
        .scope(
            std::cell::Cell::new(true),
            COMPOSITION_DISPATCH.scope(
                true,
                COMPOSITION_SCOPE.scope(
                    CompositionScope {
                        runtime_id,
                        frame_key,
                        closed: Arc::new(parking_lot::Mutex::new(false)),
                        deadline,
                    },
                    future,
                ),
            ),
        )
        .await
}

/// Captured parent deadlines can only narrow provider/tool admission, even inside callbacks without cursor authority.
pub(crate) fn current_composition_deadline() -> Option<std::time::Instant> {
    COMPOSITION_SCOPE
        .try_with(|scope| scope.deadline)
        .ok()
        .flatten()
}

/// A public orchestration entry consumes the parent's dispatch permission once, so provider callbacks cannot select its cursor again.
pub(crate) fn enter_composition_dispatch() -> bool {
    current_composition().is_some()
        && COMPOSITION_ENTRY
            .try_with(|entry| entry.replace(false))
            .unwrap_or(false)
}

/// Cursor authority is disabled for nested public orchestration without dropping inherited accounting or barrier custody.
pub(crate) async fn scope_dispatch_authority<F: Future>(authorized: bool, future: F) -> F::Output {
    COMPOSITION_DISPATCH.scope(authorized, future).await
}

/// Child admission and barrier closure share one lock; closure never aborts already admitted turns or their settlement.
pub(crate) async fn run_composition_child<
    F: Future<Output = Result<ai_agents_core::AgentResponse>>,
>(
    scope: Option<CompositionScope>,
    slot: Option<CompositionChild>,
    response: Option<Arc<parking_lot::Mutex<Option<GroupResponse>>>>,
    future: F,
) -> Result<ai_agents_core::AgentResponse> {
    let (Some(scope), Some(slot)) = (scope, slot) else {
        return future.await;
    };
    {
        let closed = scope.closed.lock();
        if *closed {
            return Err(AgentError::TaskSuspended(scope.runtime_id.clone()));
        }
    }
    let work = COMPOSITION_DISPATCH.scope(
        false,
        COMPOSITION_SCOPE.scope(
            scope,
            scope_child_invocation(slot.operation, slot.runtime_id, future),
        ),
    );
    let outcome = if let Some(response) = response {
        GROUP_RESPONSE.scope(response, work).await
    } else {
        work.await
    };
    if outcome.is_ok()
        && let Some(reason) = current_execution().and_then(|execution| execution.stop_reason())
    {
        return Err(AgentError::Other(reason));
    }
    outcome
}

impl RunExecution {
    /// Returns an acknowledged parent frame only inside that parent's live scope.
    pub(crate) fn composition_frame(&self) -> Result<Option<DelegateFrame>> {
        let Some(scope) = current_composition() else {
            return Ok(None);
        };
        self.delegate_frames
            .lock()
            .get(&scope.frame_key)
            .cloned()
            .map(Some)
            .ok_or_else(|| TaskRunStorageError::InvalidCheckpoint.into())
    }

    /// A serial orchestration call binds a deterministic coordinate to a catalogued live child before its first effect.
    pub(crate) async fn bind_composition_call(
        &self,
        runtime_id: &str,
        coordinate: &str,
        child_runtime_id: &str,
    ) -> Result<CompositionChild> {
        let mut frame = self
            .delegate_frames
            .lock()
            .get(runtime_id)
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let operation = format!("composition:{}:call:{coordinate}", frame.id);
        if let Some(slot) = frame.calls.iter().find(|slot| slot.operation == operation) {
            if slot.runtime_id != child_runtime_id {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            self.targets.resolve(&slot.operation)?;
            return Ok(slot.clone());
        }
        if frame.calls.len() >= MAX_TASK_CHECKPOINT_RECORDS {
            return Err(TaskRunStorageError::CheckpointTooLarge.into());
        }
        let catalog = frame
            .children
            .iter()
            .find(|slot| slot.runtime_id == child_runtime_id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let catalog_operation = catalog.operation.clone();
        let slot = CompositionChild {
            registry_id: catalog.registry_id.clone(),
            runtime_id: child_runtime_id.into(),
            operation,
        };
        frame.calls.push(slot.clone());
        self.update(|payload| {
            let adapter = payload
                .adapters
                .iter_mut()
                .find(|adapter| adapter.id == format!("runtime.delegate:{}", frame.id))
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            adapter.state = serde_json::to_value(&frame)?;
            Ok(())
        })
        .await?;
        self.targets
            .bind_call(&catalog_operation, &slot.operation)?;
        self.delegate_frames.lock().insert(runtime_id.into(), frame);
        Ok(slot)
    }

    /// Cursor replacement preserves immutable dispatch binding and follows the same conditional ledger as child outcomes.
    pub(crate) async fn checkpoint_composition_cursor(
        &self,
        runtime_id: &str,
        cursor: Value,
    ) -> Result<()> {
        let mut frame = self
            .delegate_frames
            .lock()
            .get(runtime_id)
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        frame.cursor = cursor;
        self.update(|payload| {
            let adapter = payload
                .adapters
                .iter_mut()
                .find(|adapter| adapter.id == format!("runtime.delegate:{}", frame.id))
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if adapter.config
                != json!({"runtime_id":frame.runtime_id,"definition":frame.definition})
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            adapter.state = serde_json::to_value(&frame)?;
            Ok(())
        })
        .await?;
        self.delegate_frames.lock().insert(runtime_id.into(), frame);
        Ok(())
    }

    /// Captures a quiescent group together with its exact parent frame; unknown or still-polled work cannot be parked.
    pub(crate) async fn pause_group(
        &self,
        mut runtime: TaskRuntimeCheckpoint,
        runtime_id: &str,
        todos: Option<TaskTodoCheckpoint>,
    ) -> Result<TaskRunSnapshot> {
        let _serial = self.serial.lock().await;
        let previous = self.load_owned_locked(&_serial).await?;
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
        let parked = collect_parked_leaves(&frame, &frames, &payload, &mut Vec::new())?;
        let selected_operation = parked
            .first()
            .map_or(frame.child_operation.as_str(), |leaf| {
                leaf.operation.as_str()
            });
        let child = payload
            .children
            .iter()
            .find(|child| child.child_id == selected_operation)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let child_request = child
            .pending
            .clone()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let batch: TaskBatchState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == format!("runtime.child_batch:{}", selected_operation))
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                .state
                .clone(),
        )?;
        let request_id = uuid::Uuid::new_v4().to_string();
        let group = TaskGroupState {
            version: 1,
            request_id: request_id.clone(),
            child_operation: selected_operation.into(),
            child_runtime_id: child.runtime.snapshot.agent_id.clone(),
            child_request,
            batch,
            frame,
            frames,
            parked,
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
        let parent_batch = if group.frame.dispatch == CompositionDispatch::ToolBatch {
            let batch: TaskBatchState = serde_json::from_value(group.frame.cursor.clone())?;
            batch.bind_runtime(&mut runtime, &self.run_id, self.current_cycle())?;
            match runtime.continuation {
                TaskContinuation::Suspended { batch, .. } => batch,
                _ => return Err(TaskRunStorageError::InvalidCheckpoint.into()),
            }
        } else {
            None
        };
        runtime.continuation = TaskContinuation::Suspended {
            request_id: request_id.clone(),
            turn_id: if group.frame.dispatch == CompositionDispatch::ToolBatch {
                format!("{}:{}", self.run_id, self.current_cycle())
            } else {
                group.frame.id.clone()
            },
            user_message_committed: group.frame.user_message_committed,
            finalized: false,
            skill: None,
            batch: parent_batch,
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

    /// Captures exact parent configuration and actual registry objects before child work; stored labels never create execution authority.
    pub(crate) async fn retain_delegate_frame(
        &self,
        frame: DelegateFrame,
        registry: Arc<crate::spawner::AgentRegistry>,
    ) -> Result<()> {
        if frame.version != 1 || frame.id.is_empty() || frame.child_operation.is_empty() {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let captured = self.targets.capture(&frame, &registry)?;
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
        self.targets.install(captured)?;
        let key = if frame.dispatch == CompositionDispatch::ToolMessage {
            frame.id.clone()
        } else {
            frame.runtime_id.clone()
        };
        self.delegate_frames.lock().insert(key, frame);
        Ok(())
    }

    /// Admission uses a live framework-owned frame; a stored child ID alone cannot authorize suspension.
    pub(crate) fn has_composition_child(&self, operation: &str) -> bool {
        self.delegate_frames.lock().values().any(|frame| {
            frame.child_operation == operation
                || frame
                    .children
                    .iter()
                    .chain(&frame.calls)
                    .any(|child| child.operation == operation)
        })
    }

    /// An intermediate parked parent is distinguished from a leaf batch and never restored from an unbound runtime label.
    pub(crate) async fn parked_child_composition(
        &self,
        operation: &str,
        input: &str,
        runtime_id: &str,
    ) -> Result<Option<DelegateFrame>> {
        let snapshot = self.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        let Some(child) = payload
            .children
            .iter()
            .find(|child| child.child_id == operation)
        else {
            return Ok(None);
        };
        if child.pending.is_none()
            || !(matches!(
                child.runtime.continuation,
                TaskContinuation::Suspended { batch: None, .. }
            ) || child
                .pending
                .as_ref()
                .is_some_and(|p| p.reviewed_action.get("frame_id").is_some()))
        {
            return Ok(None);
        }
        let binding = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == format!("child-operation:{operation}"))
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        if binding.config != json!({"input":input,"runtime_id":runtime_id})
            || child.runtime.snapshot.agent_id != runtime_id
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        self.delegate_frames
            .lock()
            .get(runtime_id)
            .cloned()
            .map(Some)
            .ok_or_else(|| TaskRunStorageError::InvalidCheckpoint.into())
    }

    /// Saves a nested parent's exact dispatch after its children have reached acknowledged safe points.
    pub(crate) async fn checkpoint_child_composition_park(
        &self,
        operation: &str,
        mut runtime: TaskRuntimeCheckpoint,
    ) -> Result<()> {
        let frames = self.delegate_frames.lock().clone();
        let frame = frames
            .get(&runtime.snapshot.agent_id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let request_id = uuid::Uuid::new_v4().to_string();
        self.update(|payload| {
            let leaves = collect_parked_leaves(frame, &frames, payload, &mut Vec::new())?;
            let leaf = leaves
                .first()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            let parent_batch = if frame.dispatch == CompositionDispatch::ToolBatch {
                let batch: TaskBatchState = serde_json::from_value(frame.cursor.clone())?;
                batch.bind_runtime(&mut runtime, &self.run_id, self.current_cycle())?;
                match runtime.continuation {
                    TaskContinuation::Suspended { batch, .. } => batch,
                    _ => return Err(TaskRunStorageError::InvalidCheckpoint.into()),
                }
            } else {
                None
            };
            runtime.continuation = TaskContinuation::Suspended {
                request_id: request_id.clone(),
                turn_id: if frame.dispatch == CompositionDispatch::ToolBatch {
                    format!("{}:{}", self.run_id, self.current_cycle())
                } else {
                    frame.id.clone()
                },
                user_message_committed: frame.user_message_committed,
                finalized: false,
                skill: None,
                batch: parent_batch,
            };
            runtime.validate()?;
            let child = payload
                .children
                .iter_mut()
                .find(|child| child.child_id == operation)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if child.result.is_some() {
                return Err(TaskRunStorageError::Conflict.into());
            }
            if let Some(pending) = child.pending.take() {
                payload.consumed_request_ids.push(pending.id);
            }
            child.runtime = runtime;
            child.pending = Some(TaskPendingRequest {
                id: request_id.clone(),
                issued_revision: self.revision.load(std::sync::atomic::Ordering::Acquire) + 1,
                kind: leaf.request.kind,
                reviewed_action: json!({"frame_id":frame.id}),
            });
            Ok(())
        })
        .await?;
        Ok(())
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
