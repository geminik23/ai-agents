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
    #[serde(default)]
    pub work: Value,
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
    Pipeline,
    Concurrent,
    Group,
    Handoff,
}

impl MessageResponse {
    /// Structured compositions use their existing algorithm cursors rather than a single message recipient.
    pub(crate) fn composed(&self) -> bool {
        matches!(
            self,
            Self::Pipeline | Self::Concurrent | Self::Group | Self::Handoff
        )
    }
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

// Separate decoding keeps optional validation data off nested message-checkpoint stacks.
fn validate_suspended_observation(payload: &TaskCheckpointPayload, id: &str) -> Result<()> {
    let mut matches = 0;
    for value in payload
        .adapters
        .iter()
        .filter_map(|adapter| adapter.state.get("_task_validation_driver"))
    {
        let state: Box<ValidationDriverState> = serde_json::from_value(value.clone())?;
        if let Some(parked) = &state.parked {
            super::authority::validate_observation_pause(parked)?;
            if state.in_flight.is_some() || state.pending_batch.first() != Some(&parked.observation)
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            if canonical_identity(
                &serde_json::json!({"check":state.check_id,"attempt":state.identity.attempt,"observation":parked.observation}),
            )? == id
            {
                matches += 1;
            }
        }
    }
    if matches != 1 {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    Ok(())
}

/// Serialized suspended markers require an exact recorded message or observation frame and never authorize execution by themselves.
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
        // Logical observation reservations need their own exact acknowledged cursor, not a fabricated message frame.
        if let Some(id) = reservation.id.strip_prefix("validation:") {
            validate_suspended_observation(payload, id)?;
            continue;
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
            if frame.children.is_empty()
                || state.record.call_id.is_empty()
                || !state.record.executed
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            if state.response.composed() {
                let mut parked = false;
                for slot in frame.children.iter().chain(&frame.calls) {
                    if let Some(child) = payload
                        .children
                        .iter()
                        .find(|child| child.child_id == slot.operation)
                    {
                        if child.runtime.snapshot.agent_id != slot.runtime_id
                            || (child.pending.is_none() && child.result.is_none())
                        {
                            return Err(TaskRunStorageError::InvalidCheckpoint.into());
                        }
                        parked |= child.pending.is_some();
                    }
                }
                if !parked
                    && state.waiting_for.as_ref().is_none_or(|dependency| {
                        !payload
                            .children
                            .iter()
                            .any(|child| &child.child_id == dependency)
                    })
                {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
                continue;
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
                .find(|source| {
                    source.child_operation == *dependency
                        || source
                            .children
                            .iter()
                            .any(|slot| &slot.operation == dependency)
                })
                .ok_or_else(invalid)?;
            let source_state: MessageState = serde_json::from_value(source.cursor.clone())?;
            let predecessor = batch
                .calls
                .iter()
                .position(|call| call.id == source_state.record.call_id);
            let current = batch
                .calls
                .iter()
                .position(|call| call.id == state.record.call_id)
                .ok_or_else(invalid)?;
            if source.dispatch != composition::CompositionDispatch::ToolMessage
                || (source.delegate_runtime_id != frame.delegate_runtime_id
                    && !source.children.iter().any(|previous| {
                        frame
                            .children
                            .iter()
                            .any(|next| next.runtime_id == previous.runtime_id)
                    }))
                || (source.runtime_id == frame.runtime_id
                    && source.parent_operation == frame.parent_operation
                    && predecessor.is_none_or(|predecessor| predecessor >= current))
            {
                return Err(invalid());
            }
        }
        children.extend(frame.children.iter().chain(&frame.calls).cloned());
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
            let selected_operation = if state.response.composed() {
                frame
                    .children
                    .iter()
                    .chain(&frame.calls)
                    .find(|slot| {
                        payload.children.iter().any(|child| {
                            child.child_id == slot.operation && child.pending.is_some()
                        })
                    })
                    .map(|slot| &slot.operation)
                    .or(state.waiting_for.as_ref())
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            } else {
                operation
            };
            let child = payload
                .children
                .iter()
                .find(|child| &child.child_id == selected_operation)
                .or_else(|| {
                    state.waiting_for.as_ref().and_then(|dependency| {
                        payload
                            .children
                            .iter()
                            .find(|child| &child.child_id == dependency)
                    })
                })
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if child.child_id == *selected_operation && child.pending.is_none() {
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

    /// A catalogue overlapping another parked operation defers the invocation before any new child root history.
    pub(crate) fn composition_dependency(
        &self,
        incoming: &DelegateFrame,
    ) -> Result<Option<String>> {
        let frames = self.delegate_frames.lock();
        if !frames.values().any(|frame| {
            frame.dispatch == composition::CompositionDispatch::ToolMessage
                && frame.children.iter().chain(&frame.calls).any(|previous| {
                    incoming
                        .children
                        .iter()
                        .any(|next| previous.runtime_id == next.runtime_id)
                })
        }) {
            return Ok(None);
        }
        drop(frames);
        self.composition_dependency_bound(incoming)
    }

    // Optional checkpoint decoding stays outside ordinary nested-message dispatch stack frames.
    fn composition_dependency_bound(&self, incoming: &DelegateFrame) -> Result<Option<String>> {
        let frames = self.delegate_frames.lock();
        let snapshot = self.acknowledged_snapshot();
        let payload: Box<TaskCheckpointPayload> = serde_json::from_value(snapshot.payload)?;
        if let Some(child) = payload.children.iter().find(|child| {
            child.pending.is_some()
                && incoming
                    .children
                    .iter()
                    .any(|slot| slot.runtime_id == child.runtime.snapshot.agent_id)
        }) {
            let source = frames
                .values()
                .find(|frame| {
                    frame.dispatch == composition::CompositionDispatch::ToolMessage
                        && frame
                            .children
                            .iter()
                            .any(|slot| slot.operation == child.child_id)
                })
                .ok_or_else(|| {
                    AgentError::Config("parked target needs exact parent dependency".into())
                })?;
            let destination = incoming
                .children
                .iter()
                .find(|slot| slot.runtime_id == child.runtime.snapshot.agent_id)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            let registered = self.targets.resolve(&child.child_id)?;
            if registered.info().id != destination.runtime_id || source.id == incoming.id {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            return Ok(Some(child.child_id.clone()));
        }
        for source in frames.values().filter(|source| {
            source.dispatch == composition::CompositionDispatch::ToolMessage
                && source.id != incoming.id
        }) {
            let state: MessageState = serde_json::from_value(source.cursor.clone())?;
            if payload.reservations.iter().any(|reservation| {
                reservation.id == state.attempt
                    && matches!(
                        reservation.state,
                        TaskEffectState::Dispatched | TaskEffectState::Suspended
                    )
            }) && let Some(slot) = source
                .children
                .iter()
                .chain(&source.calls)
                .find(|previous| {
                    incoming
                        .children
                        .iter()
                        .any(|next| previous.runtime_id == next.runtime_id)
                })
            {
                return Ok(Some(slot.operation.clone()));
            }
        }
        Ok(None)
    }

    /// Queued work waits outside message publication locks until its predecessor acknowledges a result or safe park.
    pub(crate) fn wait_message_dependency<'a>(
        &'a self,
        frame: &'a DelegateFrame,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let Some(dependency) = frame.cursor.get("waiting_for").and_then(Value::as_str) else {
                return Ok(false);
            };
            loop {
                let snapshot = self.load_owned().await?;
                let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
                self.check(&payload)?;
                if let Some(child) = payload
                    .children
                    .iter()
                    .find(|child| child.child_id == dependency)
                {
                    if child.pending.is_some() {
                        return Ok(true);
                    }
                    if child.result.is_some() {
                        return Ok(false);
                    }
                }
                let source = payload
                    .adapters
                    .iter()
                    .filter(|adapter| adapter.adapter == "runtime.delegate")
                    .filter_map(|adapter| {
                        serde_json::from_value::<DelegateFrame>(adapter.state.clone()).ok()
                    })
                    .find(|source| {
                        source
                            .children
                            .iter()
                            .chain(&source.calls)
                            .any(|slot| slot.operation == dependency)
                    })
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                let state: MessageState = serde_json::from_value(source.cursor.clone())?;
                let selected_runtime = source
                    .children
                    .iter()
                    .chain(&source.calls)
                    .find(|slot| slot.operation == dependency)
                    .map(|slot| slot.runtime_id.as_str())
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                if let Some(child) = payload.children.iter().find(|child| {
                    child.pending.is_some()
                        && child.runtime.snapshot.agent_id == selected_runtime
                        && source
                            .children
                            .iter()
                            .chain(&source.calls)
                            .any(|slot| slot.operation == child.child_id)
                }) {
                    let actual = child.child_id.clone();
                    let mut incoming = self
                        .delegate_frames
                        .lock()
                        .get(&frame.id)
                        .cloned()
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                    incoming.cursor["waiting_for"] = json!(actual);
                    self.update(|payload| {
                        let adapter = payload
                            .adapters
                            .iter_mut()
                            .find(|adapter| adapter.id == format!("runtime.delegate:{}", frame.id))
                            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                        adapter.state = serde_json::to_value(&incoming)?;
                        Ok(())
                    })
                    .await?;
                    self.delegate_frames
                        .lock()
                        .insert(frame.id.clone(), incoming);
                    return Ok(true);
                }
                if payload.reservations.iter().any(|reservation| {
                    reservation.id == state.attempt
                        && reservation.state == TaskEffectState::Completed
                }) {
                    return Ok(false);
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
    }

    /// A deferred invocation becomes runnable only after its exact predecessor settles and every selected actual target is free of parked work.
    pub(crate) fn deferred_message_ready(&self, frame: &DelegateFrame) -> bool {
        let Some(dependency) = frame.cursor.get("waiting_for").and_then(Value::as_str) else {
            return false;
        };
        let snapshot = self.acknowledged_snapshot();
        if self.delegate_frames.lock().values().any(|other| {
            other.id != frame.id
                && other.dispatch == composition::CompositionDispatch::ToolMessage
                && other.children.iter().any(|slot| {
                    frame
                        .children
                        .iter()
                        .any(|incoming| slot.runtime_id == incoming.runtime_id)
                })
                && snapshot.payload["reservations"]
                    .as_array()
                    .is_some_and(|reservations| {
                        reservations.iter().any(|reservation| {
                            reservation["id"] == other.cursor["attempt"]
                                && reservation["state"] == "dispatched"
                        })
                    })
        }) {
            return false;
        }
        snapshot.payload["children"]
            .as_array()
            .is_some_and(|children| {
                children
                    .iter()
                    .any(|child| child["child_id"] == dependency && !child["result"].is_null())
                    && !children.iter().any(|child| {
                        frame
                            .children
                            .iter()
                            .any(|slot| child["child_id"] == slot.operation)
                    })
                    && children.iter().all(|child| {
                        !frame
                            .children
                            .iter()
                            .any(|slot| child["runtime"]["snapshot"]["agent_id"] == slot.runtime_id)
                            || child["pending"].is_null()
                    })
            })
    }

    /// A retained batch can advance a dependency-only invocation without inventing a response for its unstarted child.
    pub(crate) fn deferred_batch_ready(&self, coordinator: &DelegateFrame) -> bool {
        let Some(ids) = coordinator
            .definition
            .get("messages")
            .and_then(Value::as_array)
        else {
            return false;
        };
        let frames = self.delegate_frames.lock().clone();
        ids.iter()
            .filter_map(Value::as_str)
            .filter_map(|id| frames.get(id))
            .any(|frame| self.deferred_message_ready(frame))
    }

    /// Drained composition siblings may release a dependency after its parent was polled; ready saved work must be driven before another pause is exposed.
    pub(crate) fn composition_has_ready_dependency(&self, frame: &DelegateFrame) -> bool {
        let frames = self.delegate_frames.lock().clone();
        let mut queue = vec![frame.clone()];
        let mut seen = std::collections::HashSet::new();
        while let Some(parent) = queue.pop() {
            if !seen.insert(parent.id.clone()) || seen.len() > 32 {
                continue;
            }
            if parent.dispatch == composition::CompositionDispatch::ToolBatch
                && self.deferred_batch_ready(&parent)
            {
                return true;
            }
            for slot in parent.children.iter().chain(&parent.calls) {
                if let Some(child) = frames.get(&slot.runtime_id).filter(|child| {
                    child.parent_operation.as_deref() == Some(slot.operation.as_str())
                }) {
                    queue.push(child.clone());
                }
            }
        }
        false
    }

    /// Final participant admission cannot start another root on a runtime whose different operation is still parked.
    pub(crate) fn target_has_pending_operation(
        &self,
        runtime: &str,
        incoming: Option<&str>,
    ) -> bool {
        let snapshot = self.acknowledged_snapshot();
        snapshot.payload["children"]
            .as_array()
            .is_some_and(|children| {
                children.iter().any(|child| {
                    child["runtime"]["snapshot"]["agent_id"] == runtime
                        && !child["pending"].is_null()
                        && child["child_id"].as_str() != incoming
                })
            })
    }

    /// Repeated calls retain distinct incoming operations but cannot start a second root while the target history is incomplete.
    pub(crate) fn message_dependency(&self, incoming: &DelegateFrame) -> Result<Option<String>> {
        let mut binding = incoming.clone();
        binding.children = vec![composition::CompositionChild {
            registry_id: incoming.delegate_id.clone(),
            runtime_id: incoming.delegate_runtime_id.clone(),
            operation: incoming.child_operation.clone(),
        }];
        self.composition_dependency(&binding)
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
            .flat_map(|frame| frame.children.iter().chain(&frame.calls).cloned())
            .collect();
        coordinator.calls.clear();
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
