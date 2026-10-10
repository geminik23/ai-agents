//! Typed message suspension keeps consumed invocation accounting separate from uncertain external effects.

use super::*;
use ai_agents_core::{AgentError, Result, Tool, ToolExecutionRecord};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{future::Future, sync::Arc};

/// Only the shared final executor can mint this live message dispatch authority.
#[derive(Clone)]
pub(crate) struct MessageInvocation {
    pub frame: Box<DelegateFrame>,
    pub tool: Arc<dyn Tool>,
}

/// Exact invocation and batch state are sensitive recovery data, not model-provided continuation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MessageState {
    pub attempt: String,
    pub record: ToolExecutionRecord,
    pub state_generation: Option<u64>,
    pub resource_lock_keys: Vec<String>,
    pub max_output_chars: Option<usize>,
    pub batch: Option<TaskBatchState>,
    #[serde(default)]
    pub response: MessageResponse,
    #[serde(default)]
    pub child_actor: Option<crate::TurnActorContext>,
    /// A repeated target waits for this exact incoming operation to finish before starting another root history.
    #[serde(default)]
    pub waiting_for: Option<String>,
}

/// Routing selection is a completed dispatch effect and is not repeated when formatting the eventual response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum MessageResponse {
    #[default]
    Message,
    Route {
        reason: String,
    },
}

tokio::task_local! {
    static MESSAGE_INVOCATION: std::cell::RefCell<Option<MessageInvocation>>;
}

/// Sender hooks and nested callbacks cannot reuse the parent invocation after dispatch consumes it.
pub(crate) async fn scope_message<F: Future>(
    invocation: Option<MessageInvocation>,
    future: F,
) -> F::Output {
    MESSAGE_INVOCATION
        .scope(std::cell::RefCell::new(invocation), future)
        .await
}

/// A matching framework message tool consumes its own dispatch authority once, before invoking any callback.
pub(crate) fn take_message() -> Option<MessageInvocation> {
    MESSAGE_INVOCATION
        .try_with(|slot| slot.borrow_mut().take())
        .ok()
        .flatten()
}

/// Serialized suspended markers require an exact recorded message frame and never authorize execution by themselves.
pub(crate) fn validate_suspended_attempts(
    payload: &TaskCheckpointPayload,
    status: TaskRunStatus,
) -> Result<()> {
    for reservation in payload
        .reservations
        .iter()
        .filter(|r| r.state == TaskEffectState::Suspended)
    {
        if status.is_terminal() || reservation.result.is_some() {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let mut bindings = 0;
        for adapter in payload
            .adapters
            .iter()
            .filter(|a| a.adapter == "runtime.delegate")
        {
            let frame: DelegateFrame = serde_json::from_value(adapter.state.clone())?;
            if frame.dispatch != composition::CompositionDispatch::ToolMessage {
                continue;
            }
            let state: MessageState = serde_json::from_value(frame.cursor)?;
            if state.attempt != reservation.id {
                continue;
            }
            bindings += 1;
            if frame.children.len() != 1
                || state.record.call_id.is_empty()
                || !state.record.executed
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            let child = payload
                .children
                .iter()
                .find(|child| child.child_id == frame.children[0].operation)
                .or_else(|| {
                    state.waiting_for.as_ref().and_then(|operation| {
                        payload
                            .children
                            .iter()
                            .find(|child| &child.child_id == operation)
                    })
                })
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if child.runtime.snapshot.agent_id != frame.children[0].runtime_id
                || (child.child_id == frame.children[0].operation
                    && child.pending.is_none()
                    && child.result.is_none())
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            if status == TaskRunStatus::Paused {
                state
                    .batch
                    .as_ref()
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                    .validate()?;
            }
        }
        if bindings != 1 {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    Ok(())
}

/// Acknowledged cancellation settles only known-safe message attempts, preserving their consumed charge and targets.
pub(crate) fn settle_cancelled_messages(payload: &mut TaskCheckpointPayload) -> Result<()> {
    let scope: EvaluationScope = serde_json::from_value(
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller")
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .state["scope"]
            .clone(),
    )?;
    for adapter in payload
        .adapters
        .iter()
        .filter(|adapter| adapter.adapter == "runtime.delegate")
    {
        let frame: DelegateFrame = serde_json::from_value(adapter.state.clone())?;
        if frame.dispatch != composition::CompositionDispatch::ToolMessage {
            continue;
        }
        let state: MessageState = serde_json::from_value(frame.cursor)?;
        if !payload
            .reservations
            .iter()
            .any(|r| r.id == state.attempt && r.state == TaskEffectState::Suspended)
        {
            continue;
        }
        let mut record = state.record;
        record.output = "message stopped after child suspension".into();
        record.success = false;
        record.cancelled = true;
        record.cancellation_reason = Some("task stopped".into());
        record
            .metadata
            .insert("_task_runtime_id".into(), json!(frame.runtime_id));
        record.metadata.insert(
            "_task_child_operation".into(),
            json!(frame.parent_operation),
        );
        let sequence = payload.evidence.tool_calls.len() as u64;
        record.metadata.insert(
            "_task_identity".into(),
            json!(EvidenceIdentity {
                key: scope.key.clone(),
                objective_revision: scope.objective_revision,
                cycle: scope.cycle,
                sequence,
                stage: scope.stage.clone(),
                attempt: None,
                config_identity: String::new(),
                target: None,
                target_revision: None,
                mutation_generation: scope.mutation_generation,
            }),
        );
        if !payload.evidence.tool_calls.iter().any(|capture| {
            capture.record.call_id == record.call_id
                && capture.record.metadata.get("_task_runtime_id")
                    == record.metadata.get("_task_runtime_id")
                && capture.record.metadata.get("_task_child_operation")
                    == record.metadata.get("_task_child_operation")
        }) {
            payload.evidence.tool_calls.push(TaskToolEvidence {
                run_id: scope.key.run_id.clone(),
                sequence,
                stage: scope.stage.clone(),
                attempt: None,
                origin: "shared_executor".into(),
                record,
            });
        }
    }
    for reservation in payload
        .reservations
        .iter_mut()
        .filter(|r| r.state == TaskEffectState::Suspended)
    {
        reservation.state = TaskEffectState::Completed;
        reservation.result =
            Some(json!({"success":false,"output":"message stopped after acknowledged suspension"}));
        let additional = reservation
            .reserved_micro_usd
            .checked_sub(reservation.charged_micro_usd)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        payload.counters.charged_micro_usd = payload
            .counters
            .charged_micro_usd
            .checked_add(additional)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        reservation.charged_micro_usd = reservation.reserved_micro_usd;
    }
    Ok(())
}

/// A coordinator is only a projection of acknowledged invocation frames, never an independent execution grant.
pub(crate) fn validate_message_batch(
    coordinator: &DelegateFrame,
    frames: &std::collections::BTreeMap<String, DelegateFrame>,
    batch: &TaskBatchState,
    payload: &TaskCheckpointPayload,
) -> Result<()> {
    let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
    let ids: Vec<String> = serde_json::from_value(coordinator.definition["messages"].clone())?;
    let declared: std::collections::HashSet<_> = ids.iter().cloned().collect();
    let mut expected = std::collections::HashSet::new();
    for adapter in payload
        .adapters
        .iter()
        .filter(|adapter| adapter.adapter == "runtime.delegate")
    {
        let frame: DelegateFrame = serde_json::from_value(adapter.state.clone())?;
        if frame.dispatch == composition::CompositionDispatch::ToolMessage
            && frame.runtime_id == coordinator.runtime_id
            && frame.parent_operation == coordinator.parent_operation
            && batch
                .approvals
                .iter()
                .any(|approval| approval.request_id == frame.id)
        {
            expected.insert(frame.id);
        }
    }
    if declared != expected || declared.len() != ids.len() {
        return Err(invalid());
    }
    let mut calls = std::collections::HashSet::new();
    let mut attempts = std::collections::HashSet::new();
    let mut children = Vec::new();
    if ids.is_empty() {
        return Err(invalid());
    }
    for id in ids {
        let frame = frames.get(&id).ok_or_else(invalid)?;
        let state: MessageState = serde_json::from_value(frame.cursor.clone())?;
        let adapter = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == format!("runtime.delegate:{id}"))
            .ok_or_else(invalid)?;
        if frame.id != id
            || frame.dispatch != composition::CompositionDispatch::ToolMessage
            || frame.runtime_id != coordinator.runtime_id
            || frame.parent_operation != coordinator.parent_operation
            || adapter.state != json!(frame)
            || !calls.insert(state.record.call_id.clone())
            || !attempts.insert(state.attempt.clone())
            || serde_json::to_value(&state.batch)? != serde_json::to_value(Some(batch))?
            || !batch.calls.iter().enumerate().any(|(index, call)| {
                let call = batch.pending_call(call);
                call.id == state.record.call_id
                    && call.name == state.record.requested_name
                    && call.arguments == state.record.arguments
                    && batch.results[index].is_none()
            })
            || !batch.approvals.iter().any(|approval| {
                approval.request_id == id && approval.call_id == state.record.call_id
            })
        {
            return Err(invalid());
        }
        if let Some(dependency) = &state.waiting_for {
            let source: DelegateFrame = payload
                .adapters
                .iter()
                .filter(|adapter| adapter.adapter == "runtime.delegate")
                .filter_map(|adapter| {
                    serde_json::from_value::<DelegateFrame>(adapter.state.clone()).ok()
                })
                .find(|source| source.child_operation == *dependency)
                .ok_or_else(invalid)?;
            let source_state: MessageState = serde_json::from_value(source.cursor.clone())?;
            let predecessor = batch
                .calls
                .iter()
                .position(|call| call.id == source_state.record.call_id)
                .ok_or_else(invalid)?;
            let current = batch
                .calls
                .iter()
                .position(|call| call.id == state.record.call_id)
                .ok_or_else(invalid)?;
            if source.dispatch != composition::CompositionDispatch::ToolMessage
                || source.runtime_id != frame.runtime_id
                || source.parent_operation != frame.parent_operation
                || source.delegate_runtime_id != frame.delegate_runtime_id
                || predecessor >= current
            {
                return Err(invalid());
            }
        }
        children.extend(frame.children.clone());
    }
    if json!(children) != json!(coordinator.children) {
        return Err(invalid());
    }
    Ok(())
}

impl RunExecution {
    /// Distinct calls may coexist, but one consumed attempt cannot publish a second dispatch capability.
    /// The caller holds message_admission through publication, so duplicate detection and installation are atomic.
    pub(crate) fn check_message_slot(&self, incoming: &DelegateFrame) -> Result<()> {
        let state: MessageState = serde_json::from_value(incoming.cursor.clone())?;
        let frames = self.delegate_frames.lock();
        if frames.len() >= MAX_TASK_CHECKPOINT_RECORDS {
            return Err(TaskRunStorageError::CheckpointTooLarge.into());
        }
        for frame in frames
            .values()
            .filter(|frame| frame.dispatch == composition::CompositionDispatch::ToolMessage)
        {
            let previous: MessageState = serde_json::from_value(frame.cursor.clone())?;
            if frame.id == incoming.id || previous.attempt == state.attempt {
                return Err(AgentError::Config(
                    "task message attempt already owns a cursor".into(),
                ));
            }
        }
        Ok(())
    }

    /// Settled message dispatch cannot classify a later local approval as pending child composition.
    pub(crate) fn retire_completed_message(&self, runtime_id: &str, attempt: &str) {
        let mut frames = self.delegate_frames.lock();
        frames.retain(|_, frame| {
            !(frame.runtime_id == runtime_id
                && frame.dispatch == composition::CompositionDispatch::ToolMessage
                && frame.cursor.get("attempt").and_then(Value::as_str) == Some(attempt))
        });
    }

    /// Parking requires an acknowledged child cursor or an exact dependency that prevents repeated-target history overlap.
    /// The parent attempt remains consumed; neither non-invocation refunds nor uncertainty settlement apply.
    pub(crate) async fn suspend_message(&self, runtime_id: &str, attempt: &str) -> Result<String> {
        let frame = self
            .delegate_frames
            .lock()
            .values()
            .find(|frame| {
                frame.runtime_id == runtime_id
                    && frame.dispatch == composition::CompositionDispatch::ToolMessage
                    && frame.cursor.get("attempt").and_then(Value::as_str) == Some(attempt)
            })
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let state: MessageState = serde_json::from_value(frame.cursor.clone())?;
        if frame.dispatch != composition::CompositionDispatch::ToolMessage
            || state.attempt != attempt
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let operation = &frame.children[0].operation;
        self.update(|payload| {
            let child = payload
                .children
                .iter()
                .find(|child| &child.child_id == operation)
                .or_else(|| {
                    state.waiting_for.as_ref().and_then(|dependency| {
                        payload
                            .children
                            .iter()
                            .find(|child| &child.child_id == dependency)
                    })
                })
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if child.child_id == *operation && child.pending.is_none() {
                return Err(AgentError::Other(
                    "message child did not acknowledge safe suspension".into(),
                ));
            }
            let reservation = payload
                .reservations
                .iter_mut()
                .find(|r| r.id == attempt)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if reservation.state != TaskEffectState::Dispatched {
                return Err(TaskRunStorageError::Conflict.into());
            }
            reservation.state = TaskEffectState::Suspended;
            Ok(())
        })
        .await?;
        self.parked_approvals
            .lock()
            .entry(runtime_id.into())
            .or_default()
            .push(suspension::BatchApproval {
                call_id: state.record.call_id,
                request_id: frame.id.clone(),
                trigger: Value::Null,
                context: state.record.executed_arguments,
                message: "Message dispatch is waiting for its child".into(),
                timeout_millis: None,
                expires_at: frame.expires_at,
                question: None,
            });
        Ok(frame.id)
    }

    /// Repeated calls retain distinct incoming operations but cannot start a second root while the target history is incomplete.
    pub(crate) fn message_dependency(&self, incoming: &DelegateFrame) -> Result<Option<String>> {
        let snapshot = self.acknowledged_snapshot();
        let frames = self.delegate_frames.lock();
        let predecessor = frames.values().find(|frame| {
            frame.dispatch == composition::CompositionDispatch::ToolMessage
                && frame.delegate_runtime_id == incoming.delegate_runtime_id
                && snapshot.payload["children"]
                    .as_array()
                    .is_some_and(|children| {
                        children.iter().any(|child| {
                            child["child_id"] == frame.child_operation
                                && !child["pending"].is_null()
                        })
                    })
        });
        if let Some(predecessor) = predecessor {
            if predecessor.runtime_id != incoming.runtime_id
                || predecessor.parent_operation != incoming.parent_operation
            {
                return Err(AgentError::Config(
                    "cross-parent pending message target requires composition continuation".into(),
                ));
            }
            return Ok(Some(predecessor.child_operation.clone()));
        }
        Ok(None)
    }

    /// Batch retention follows child parking and must be acknowledged before the root exposes a pause.
    pub(crate) async fn retain_message_batch(
        &self,
        runtime_id: &str,
        batch: TaskBatchState,
    ) -> Result<()> {
        let mut messages: Vec<_> = self
            .delegate_frames
            .lock()
            .values()
            .filter(|frame| {
                frame.runtime_id == runtime_id
                    && frame.dispatch == composition::CompositionDispatch::ToolMessage
                    && batch
                        .approvals
                        .iter()
                        .any(|approval| approval.request_id == frame.id)
            })
            .cloned()
            .collect();
        if messages.is_empty() {
            self.delegate_frames.lock().retain(|key, frame| {
                !(key == runtime_id
                    && frame.dispatch == composition::CompositionDispatch::ToolBatch)
            });
            return Ok(());
        }
        messages.sort_by_key(|frame| {
            batch.calls.iter().position(|call| {
                frame.cursor["record"]["call_id"].as_str() == Some(call.id.as_str())
            })
        });
        for frame in &mut messages {
            let mut state: MessageState = serde_json::from_value(frame.cursor.clone())?;
            state.batch = Some(batch.clone());
            frame.cursor = json!(state);
        }
        let mut coordinator = messages[0].clone();
        coordinator.id = format!("batch:{}", coordinator.id);
        coordinator.dispatch = composition::CompositionDispatch::ToolBatch;
        coordinator.children = messages
            .iter()
            .flat_map(|frame| frame.children.clone())
            .collect();
        coordinator.cursor = json!(batch);
        coordinator.definition =
            json!({"messages":messages.iter().map(|frame| &frame.id).collect::<Vec<_>>()});
        let frames = self.delegate_frames.lock().clone();
        if let Some(previous) = frames
            .get(runtime_id)
            .filter(|frame| frame.dispatch == composition::CompositionDispatch::ToolBatch)
        {
            coordinator.id = previous.id.clone();
        }
        self.update(|payload| {
            for frame in &messages {
                let adapter = payload
                    .adapters
                    .iter_mut()
                    .find(|adapter| adapter.id == format!("runtime.delegate:{}", frame.id))
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                adapter.state = json!(frame);
            }
            Ok(())
        })
        .await?;
        let mut frames = self.delegate_frames.lock();
        for frame in messages {
            frames.insert(frame.id.clone(), frame);
        }
        frames.insert(runtime_id.into(), coordinator);
        Ok(())
    }
}
