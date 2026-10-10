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
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if child.runtime.snapshot.agent_id != frame.children[0].runtime_id
                || (child.pending.is_none() && child.result.is_none())
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

impl RunExecution {
    /// Until a runtime owns a multi-invocation cursor, a second live dispatch cannot overwrite its first parked message.
    /// Previously settled messages do not restrict later ordinary delivery in the same model batch.
    pub(crate) fn check_message_slot(&self, runtime_id: &str) -> Result<()> {
        let frame = self.delegate_frames.lock().get(runtime_id).cloned();
        if let Some(frame) =
            frame.filter(|frame| frame.dispatch == composition::CompositionDispatch::ToolMessage)
        {
            let attempt = frame
                .cursor
                .get("attempt")
                .and_then(Value::as_str)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            let snapshot = self.acknowledged_snapshot();
            if snapshot.payload["reservations"]
                .as_array()
                .is_some_and(|reservations| {
                    reservations.iter().any(|reservation| {
                        reservation["id"].as_str() == Some(attempt)
                            && reservation["state"].as_str() != Some("completed")
                    })
                })
            {
                return Err(AgentError::Config(
                    "another task message cursor is pending in this runtime".into(),
                ));
            }
        }
        Ok(())
    }

    /// Settled message dispatch cannot classify a later local approval as pending child composition.
    pub(crate) fn retire_completed_message(&self, runtime_id: &str, attempt: &str) {
        let mut frames = self.delegate_frames.lock();
        if frames.get(runtime_id).is_some_and(|frame| {
            frame.dispatch == composition::CompositionDispatch::ToolMessage
                && frame.cursor.get("attempt").and_then(Value::as_str) == Some(attempt)
        }) {
            frames.remove(runtime_id);
        }
    }

    /// Parking is admitted only after the actual child has acknowledged its exact pending cursor.
    /// The parent attempt remains consumed; neither non-invocation refunds nor uncertainty settlement apply.
    pub(crate) async fn suspend_message(&self, runtime_id: &str, attempt: &str) -> Result<String> {
        let frame = self
            .delegate_frames
            .lock()
            .get(runtime_id)
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
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if child.pending.is_none() || child.result.is_some() {
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

    /// Batch retention follows child parking and must be acknowledged before the root exposes a pause.
    pub(crate) async fn retain_message_batch(
        &self,
        runtime_id: &str,
        batch: TaskBatchState,
    ) -> Result<()> {
        let frame = self.delegate_frames.lock().get(runtime_id).cloned();
        if let Some(frame) =
            frame.filter(|f| f.dispatch == composition::CompositionDispatch::ToolMessage)
        {
            let mut state: MessageState = serde_json::from_value(frame.cursor)?;
            if !batch.approvals.iter().any(|a| a.request_id == frame.id) {
                return Ok(());
            }
            state.batch = Some(batch);
            self.checkpoint_composition_cursor(runtime_id, json!(state))
                .await?;
        }
        Ok(())
    }
}
