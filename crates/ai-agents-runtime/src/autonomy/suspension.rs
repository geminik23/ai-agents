//! Reconstructible committed tool batches preserve out-of-order results separately from native history.

use super::*;
use ai_agents_core::{AgentError, Result, ToolCall, ToolExecutionRequest};
use ai_agents_hitl::{ApprovalRequest, ApprovalResult};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashMap, future::Future};

/// A response is bound to one acknowledged checkpoint revision and one pending request.
#[derive(Debug, Clone)]
pub enum TaskResumeInput {
    Approval {
        request_id: String,
        result: ApprovalResult,
    },
    UserAnswer {
        request_id: String,
        answer: Value,
    },
    QuestionTimeout {
        request_id: String,
    },
}

/// Keeps question data separate from approval authority while using the same exact batch cursor.
#[derive(Debug, Clone)]
pub(crate) enum BatchResponse {
    Approval(ApprovalResult),
    UserAnswer(Value),
    QuestionTimeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskLoopState {
    pub version: u32,
    pub processed_input: String,
    pub input_context: HashMap<String, Value>,
    pub reasoning_mode: ai_agents_reasoning::ReasoningMode,
    pub auto_detected: bool,
    pub iterations: u32,
    pub all_tool_calls: Vec<ToolCall>,
    pub thinking_content: Option<String>,
    #[serde(default)]
    pub native_exchanges: Vec<TaskNativeExchange>,
    #[serde(default)]
    pub deferred_final: Option<Box<super::location::TaskModelFinal>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BatchApproval {
    pub call_id: String,
    pub request_id: String,
    pub trigger: Value,
    pub context: Value,
    pub message: String,
    pub timeout_millis: Option<u64>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskBatchState {
    pub version: u32,
    pub content: String,
    pub calls: Vec<ToolCall>,
    pub results: Vec<Option<std::result::Result<String, String>>>,
    pub appended: usize,
    pub approvals: Vec<BatchApproval>,
    pub loop_state: Option<TaskLoopState>,
    #[serde(default)]
    pub authorized: Vec<BatchAuthorization>,
    #[serde(default)]
    pub rejected: bool,
    #[serde(default)]
    pub executor_cursors: Vec<TaskExecutorCursor>,
    #[serde(default)]
    pub location: Option<super::location::TaskLocation>,
}

/// A parked fallback keeps its actual request and canonical ancestry rather than replaying a failed ancestor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskExecutorCursor {
    pub request: ToolExecutionRequest,
    pub ancestry: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BatchAuthorization {
    pub call_id: String,
    pub trigger: Value,
    pub context: Value,
    pub result: ApprovalResult,
}

impl TaskBatchState {
    /// The committed model call stays immutable while a resumed fallback uses its exact pending request.
    pub(crate) fn pending_call(&self, call: &ToolCall) -> ToolCall {
        self.executor_cursors
            .iter()
            .find(|cursor| cursor.request.call_id == call.id)
            .map(|cursor| ToolCall {
                id: call.id.clone(),
                name: cursor.request.requested_name.clone(),
                arguments: cursor.request.arguments.clone(),
            })
            .unwrap_or_else(|| call.clone())
    }

    /// Parent and child snapshots use the same exact ordered native cursor and preserved batch results.
    pub(crate) fn bind_runtime(
        &self,
        runtime: &mut TaskRuntimeCheckpoint,
        run_id: &str,
        cycle: u64,
    ) -> Result<()> {
        self.validate()?;
        runtime.continuation = TaskContinuation::Suspended {
            request_id: self.approvals[0].request_id.clone(),
            turn_id: format!("{run_id}:{cycle}"),
            user_message_committed: self
                .location
                .as_ref()
                .is_none_or(|location| location.root_state().0),
            finalized: false,
            skill: self.location.as_ref().and_then(|location| match location {
                super::location::TaskLocation::Skill(state) => Some(TaskSkillCursor {
                    skill_id: state.cursor.definition.id.clone(),
                    next_step: state.cursor.next_step,
                    results: state
                        .cursor
                        .context
                        .step_results
                        .iter()
                        .map(|result| result.result.clone())
                        .collect(),
                }),
                super::location::TaskLocation::Actions(_) => None,
            }),
            batch: Some(TaskToolBatchCursor {
                messages: runtime.snapshot.memory.messages.clone(),
                call_ids: self.calls.iter().map(|call| call.id.clone()).collect(),
                next_call: self.appended,
                completed_results: self.prefix_outputs(),
            }),
        };
        runtime.validate()
    }

    /// Cross-checks executable calls and reviewed action against the committed history before any resume claim.
    pub(crate) fn validate_checkpoint(&self, payload: &TaskCheckpointPayload) -> Result<()> {
        self.validate()?;
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        let pending = payload.pending.as_ref().ok_or_else(invalid)?;
        if matches!(pending.kind, TaskPendingKind::UserQuestion)
            != self.approvals[0].question.is_some()
            || !matches!(
                pending.kind,
                TaskPendingKind::Approval | TaskPendingKind::UserQuestion
            )
            || pending.reviewed_action != serde_json::to_value(&self.approvals[0])?
            || pending.id != self.approvals[0].request_id
        {
            return Err(invalid());
        }
        let TaskContinuation::Suspended {
            batch: Some(cursor),
            request_id,
            ..
        } = &payload.runtime.continuation
        else {
            return Err(invalid());
        };
        if *request_id != pending.id
            || cursor.next_call != self.appended
            || cursor.call_ids
                != self
                    .calls
                    .iter()
                    .map(|call| call.id.clone())
                    .collect::<Vec<_>>()
            || cursor.completed_results != self.prefix_outputs()
            || (self.location.is_none()
                && payload
                    .runtime
                    .snapshot
                    .memory
                    .messages
                    .iter()
                    .rev()
                    .find(|message| message.role == ai_agents_core::Role::Assistant)
                    .is_none_or(|message| message.content != self.content))
        {
            return Err(invalid());
        }
        if self.location.is_some() {
            self.validate_model_history(&payload.runtime)?;
            return Ok(());
        }
        if self
            .loop_state
            .as_ref()
            .unwrap()
            .native_exchanges
            .iter()
            .any(|expected| {
                !payload
                    .runtime
                    .native_exchanges
                    .iter()
                    .any(|actual| actual == expected)
            })
        {
            return Err(invalid());
        }
        if let Some(native) =
            ai_agents_core::decode_native_tool_call_markers(&self.content).map_err(|_| invalid())?
            && serde_json::to_value(native.calls())? != serde_json::to_value(&self.calls)?
        {
            return Err(invalid());
        }
        Ok(())
    }

    /// Group parents retain the same normalized calls and signed expectations as foreground batches.
    pub(crate) fn validate_model_history(&self, runtime: &TaskRuntimeCheckpoint) -> Result<()> {
        self.validate()?;
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        if let Some(location) = &self.location {
            let request = location.request()?;
            if !self.content.is_empty()
                || self.calls.len() != 1
                || self.calls[0].id != request.call_id
                || self.calls[0].name != request.requested_name
                || self.calls[0].arguments != request.arguments
                || location
                    .root_state()
                    .1
                    .iter()
                    .any(|expected| !runtime.native_exchanges.contains(expected))
            {
                return Err(invalid());
            }
            return Ok(());
        }
        if runtime
            .snapshot
            .memory
            .messages
            .iter()
            .rev()
            .find(|message| message.role == ai_agents_core::Role::Assistant)
            .is_none_or(|message| message.content != self.content)
        {
            return Err(invalid());
        }
        if let Some(native) =
            ai_agents_core::decode_native_tool_call_markers(&self.content).map_err(|_| invalid())?
            && serde_json::to_value(native.calls())? != serde_json::to_value(&self.calls)?
        {
            return Err(invalid());
        }
        if self
            .loop_state
            .as_ref()
            .ok_or_else(invalid)?
            .native_exchanges
            .iter()
            .any(|expected| {
                !runtime
                    .native_exchanges
                    .iter()
                    .any(|actual| actual == expected)
            })
        {
            return Err(invalid());
        }
        Ok(())
    }

    /// Native results use the same JSON-or-string projection as the shared history encoder.
    fn prefix_outputs(&self) -> Vec<Value> {
        self.results[..self.appended]
            .iter()
            .map(|result| {
                let output = match result.as_ref().unwrap() {
                    Ok(output) => output.clone(),
                    Err(error) => format!("Error: {error}"),
                };
                serde_json::from_str(&output).unwrap_or(Value::String(output))
            })
            .collect()
    }

    /// The executable state is bounded independently of the opaque adapter payload and agrees with its native cursor.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.version != 1
            || self.calls.is_empty()
            || self.calls.len() > MAX_TASK_CHECKPOINT_RECORDS
            || self.results.len() != self.calls.len()
            || self.appended > self.calls.len()
            || self.results[..self.appended].iter().any(Option::is_none)
            || self
                .loop_state
                .as_ref()
                .is_some_and(|state| state.version != 1)
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        if self.loop_state.is_none() && self.location.is_none() {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        if let Some(location) = &self.location {
            location.validate()?;
        }
        let mut calls = std::collections::HashSet::new();
        let mut requests = std::collections::HashSet::new();
        for call in &self.calls {
            if call.id.is_empty() || !calls.insert(&call.id) {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
        }
        for approval in &self.approvals {
            if approval.request_id.is_empty()
                || !requests.insert(&approval.request_id)
                || !self
                    .calls
                    .iter()
                    .enumerate()
                    .any(|(i, call)| call.id == approval.call_id && self.results[i].is_none())
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
        }
        if self.approvals.is_empty()
            || self.calls.iter().enumerate().any(|(index, call)| {
                self.results[index].is_none()
                    && self
                        .approvals
                        .iter()
                        .filter(|approval| approval.call_id == call.id)
                        .count()
                        != 1
            })
            || self.authorized.len() > MAX_TASK_CHECKPOINT_RECORDS
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let mut executor_calls = std::collections::HashSet::new();
        for cursor in &self.executor_cursors {
            if !executor_calls.insert(&cursor.request.call_id)
                || cursor.ancestry.len() > 16
                || !self.calls.iter().enumerate().any(|(index, call)| {
                    call.id == cursor.request.call_id && self.results[index].is_none()
                })
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            match &cursor.request.source {
                ai_agents_core::ToolCallSource::Model => {
                    if !cursor.ancestry.is_empty()
                        || !self.calls.iter().any(|call| {
                            call.id == cursor.request.call_id
                                && call.name == cursor.request.requested_name
                                && call.arguments == cursor.request.arguments
                        })
                    {
                        return Err(TaskRunStorageError::InvalidCheckpoint.into());
                    }
                }
                ai_agents_core::ToolCallSource::Fallback { original_tool } => {
                    if cursor.ancestry.last() != Some(original_tool) {
                        return Err(TaskRunStorageError::InvalidCheckpoint.into());
                    }
                }
                _ if self.location.as_ref().is_some_and(|location| {
                    location.request().is_ok_and(|request| {
                        serde_json::to_value(request).ok()
                            == serde_json::to_value(&cursor.request).ok()
                    })
                }) && cursor.ancestry.is_empty() => {}
                _ => return Err(TaskRunStorageError::InvalidCheckpoint.into()),
            }
        }
        bounded_value(&serde_json::to_value(self)?, MAX_TASK_CHECKPOINT_BYTES)
    }
}

tokio::task_local! {
    static TASK_REQUEST: ToolExecutionRequest;
    static TASK_BATCH: bool;
    static RESUMED_APPROVAL: std::cell::RefCell<Vec<BatchAuthorization>>;
    static RESUMED_QUESTION: std::cell::RefCell<Option<(String, Value, Value)>>;
    static RESUMED_EXECUTOR: std::cell::RefCell<Vec<TaskExecutorCursor>>;
}

/// Message dispatch may park only inside a reconstructible model batch with a live task request.
pub(crate) fn message_request() -> Option<ToolExecutionRequest> {
    if !TASK_BATCH.try_with(|enabled| *enabled).unwrap_or(false) {
        return None;
    }
    TASK_REQUEST.try_with(Clone::clone).ok().filter(|request| {
        matches!(
            request.source,
            ai_agents_core::ToolCallSource::Model
                | ai_agents_core::ToolCallSource::Fallback { .. }
                | ai_agents_core::ToolCallSource::Skill { .. }
                | ai_agents_core::ToolCallSource::StateAction { .. }
                | ai_agents_core::ToolCallSource::Task
        )
    })
}

/// A private request scope ties an approval to its original call ID, not a model-selected label.
pub(crate) async fn scope_task_request<F: Future>(
    request: ToolExecutionRequest,
    future: F,
) -> F::Output {
    TASK_REQUEST.scope(request, future).await
}

/// Only batches whose caller can persist its exact loop state may transfer control by suspension.
pub(crate) async fn scope_task_batch<F: Future>(future: F) -> F::Output {
    TASK_BATCH.scope(true, future).await
}

/// Reauthorization encounters the approved action again and consumes this private receipt once.
pub(crate) async fn scope_resumed_approval<F: Future>(
    receipts: Vec<BatchAuthorization>,
    future: F,
) -> F::Output {
    RESUMED_APPROVAL
        .scope(std::cell::RefCell::new(receipts), future)
        .await
}

/// Returns a bound response only for the exact action rechecked by the executor; changed actions require new approval.
pub(crate) fn take_resumed_approval(request: &ApprovalRequest) -> Result<Option<ApprovalResult>> {
    let call = TASK_REQUEST
        .try_with(|request| request.call_id.clone())
        .ok();
    let trigger = serde_json::to_value(&request.trigger)?;
    let context = serde_json::to_value(&request.context)?;
    Ok(RESUMED_APPROVAL
        .try_with(|slot| {
            let mut receipt = slot.borrow_mut();
            receipt
                .iter()
                .position(|authorization| {
                    Some(&authorization.call_id) == call.as_ref()
                        && authorization.trigger == trigger
                        && authorization.context == context
                })
                .map(|index| receipt.remove(index).result)
        })
        .ok()
        .flatten())
}

/// Live resume custody supplies the actual pending executor request once; stored source labels cannot install it.
pub(crate) async fn scope_resumed_executor<F: Future>(
    cursors: Vec<TaskExecutorCursor>,
    future: F,
) -> F::Output {
    RESUMED_EXECUTOR
        .scope(std::cell::RefCell::new(cursors), future)
        .await
}

/// Only the exact original call can consume its saved fallback location.
pub(crate) fn take_resumed_executor(call_id: &str) -> Option<TaskExecutorCursor> {
    RESUMED_EXECUTOR
        .try_with(|slot| {
            let mut cursors = slot.borrow_mut();
            cursors
                .iter()
                .position(|cursor| cursor.request.call_id == call_id)
                .map(|index| cursors.remove(index))
        })
        .ok()
        .flatten()
}

/// Retains a validated response only while the exact resumed invocation is polled.
pub(crate) async fn scope_resumed_question<F: Future>(
    call_id: String,
    args: Value,
    answer: Value,
    future: F,
) -> F::Output {
    RESUMED_QUESTION
        .scope(
            std::cell::RefCell::new(Some((call_id, args, answer))),
            future,
        )
        .await
}

/// Question preparation checks availability without consuming the response before final invocation.
pub(crate) fn has_resumed_question(call_id: &str, args: &Value) -> bool {
    RESUMED_QUESTION
        .try_with(|slot| {
            slot.borrow()
                .as_ref()
                .is_some_and(|(id, reviewed, _)| id == call_id && reviewed == args)
        })
        .unwrap_or(false)
}

/// A sibling or changed argument set cannot consume another question's response.
pub(crate) fn take_resumed_question(call_id: &str, args: &Value) -> Option<Value> {
    RESUMED_QUESTION
        .try_with(|slot| {
            let mut response = slot.borrow_mut();
            if response
                .as_ref()
                .is_some_and(|(id, reviewed, _)| id == call_id && reviewed == args)
            {
                response.take().map(|(_, _, answer)| answer)
            } else {
                None
            }
        })
        .ok()
        .flatten()
}

impl RunExecution {
    /// Questions share exact batch safety requirements, but use their own configured interaction policy.
    pub(crate) fn pauses_question(&self, runtime_id: &str) -> bool {
        self.lifecycle_profile()
            .hitl
            .as_ref()
            .and_then(|policy| policy.on_user_question)
            == Some(InteractionAction::PauseRun)
            && TASK_BATCH.try_with(|enabled| *enabled).unwrap_or(false)
            && (self.is_coordinator(runtime_id)
                || current_child_operation()
                    .is_some_and(|operation| self.has_composition_child(&operation)))
            && TASK_REQUEST
                .try_with(|request| {
                    matches!(
                        request.source,
                        ai_agents_core::ToolCallSource::Model
                            | ai_agents_core::ToolCallSource::Fallback { .. }
                            | ai_agents_core::ToolCallSource::Skill { .. }
                            | ai_agents_core::ToolCallSource::StateAction { .. }
                            | ai_agents_core::ToolCallSource::Task
                    )
                })
                .unwrap_or(false)
    }

    /// Parks an implementation-validated question before attempt admission; ordinary handlers are not polled.
    pub(crate) fn park_question(
        &self,
        runtime_id: &str,
        call_id: &str,
        args: &Value,
        question: Value,
    ) -> Result<String> {
        super::composition::close_composition_admission();
        let request_id = uuid::Uuid::new_v4().to_string();
        let timeout_millis = question
            .get("timeout_seconds")
            .and_then(Value::as_u64)
            .map(|seconds| {
                seconds
                    .checked_mul(1000)
                    .ok_or_else(|| AgentError::Config("question timeout overflow".into()))
            })
            .transpose()?;
        let expires_at = timeout_millis
            .map(|millis| {
                let duration = chrono::Duration::milliseconds(
                    i64::try_from(millis)
                        .map_err(|_| AgentError::Config("question deadline overflow".into()))?,
                );
                let deadline = std::time::Instant::now()
                    .checked_add(std::time::Duration::from_millis(millis))
                    .ok_or_else(|| AgentError::Config("question deadline overflow".into()))?;
                let expiry = chrono::Utc::now()
                    .checked_add_signed(duration)
                    .ok_or_else(|| AgentError::Config("question deadline overflow".into()))?;
                self.approval_deadlines
                    .lock()
                    .insert(request_id.clone(), deadline);
                Ok::<_, AgentError>(expiry)
            })
            .transpose()?;
        let message = question
            .get("question")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.parked_approvals
            .lock()
            .entry(runtime_id.into())
            .or_default()
            .push(BatchApproval {
                call_id: call_id.into(),
                request_id: request_id.clone(),
                trigger: Value::Null,
                context: args.clone(),
                message,
                timeout_millis,
                expires_at,
                question: Some(question),
            });
        Ok(request_id)
    }

    /// Suspension is opt-in and never silently replaces ordinary host approval waits.
    pub(crate) fn pauses_approval(&self, runtime_id: &str) -> bool {
        if self.task_interaction_policy() != Some(InteractionAction::PauseRun)
            || !TASK_BATCH.try_with(|enabled| *enabled).unwrap_or(false)
        {
            return false;
        }
        let Ok(request) = TASK_REQUEST.try_with(Clone::clone) else {
            return false;
        };
        if !matches!(
            request.source,
            ai_agents_core::ToolCallSource::Model
                | ai_agents_core::ToolCallSource::Fallback { .. }
                | ai_agents_core::ToolCallSource::Skill { .. }
                | ai_agents_core::ToolCallSource::StateAction { .. }
                | ai_agents_core::ToolCallSource::Task
        ) {
            return false;
        }
        // Child suspension requires an exact live parent frame, not merely an inherited task label.
        self.is_coordinator(runtime_id)
            || current_child_operation()
                .is_some_and(|operation| self.has_composition_child(&operation))
    }

    /// Captures a request before hooks or external interaction; no implementation slot has been admitted here.
    pub(crate) fn park_approval(
        &self,
        runtime_id: &str,
        request: &ApprovalRequest,
    ) -> Result<String> {
        super::composition::close_composition_admission();
        let call_id = TASK_REQUEST
            .try_with(|request| request.call_id.clone())
            .map_err(|_| AgentError::Config("approval has no executable task request".into()))?;
        if let Some(timeout) = request.timeout {
            let deadline = std::time::Instant::now()
                .checked_add(timeout)
                .ok_or_else(|| AgentError::Config("approval monotonic deadline overflow".into()))?;
            self.approval_deadlines
                .lock()
                .insert(request.id.clone(), deadline);
        }
        let expires_at = request
            .timeout
            .map(|timeout| {
                let duration = chrono::Duration::from_std(timeout)
                    .map_err(|_| AgentError::Config("approval deadline overflow".into()))?;
                chrono::Utc::now()
                    .checked_add_signed(duration)
                    .ok_or_else(|| AgentError::Config("approval deadline overflow".into()))
            })
            .transpose()?;
        let approval = BatchApproval {
            question: None,
            expires_at,
            call_id,
            request_id: request.id.clone(),
            trigger: serde_json::to_value(&request.trigger)?,
            context: serde_json::to_value(&request.context)?,
            message: request.message.clone(),
            timeout_millis: request
                .timeout
                .map(|timeout| u64::try_from(timeout.as_millis()))
                .transpose()
                .map_err(|_| AgentError::Config("approval timeout overflow".into()))?,
        };
        self.parked_approvals
            .lock()
            .entry(runtime_id.into())
            .or_default()
            .push(approval);
        Ok(request.id.clone())
    }

    /// A completed parallel sibling remains recorded even when an earlier call still awaits approval.
    pub(crate) fn take_parked_approvals(&self, runtime_id: &str) -> Vec<BatchApproval> {
        self.parked_approvals
            .lock()
            .remove(runtime_id)
            .unwrap_or_default()
    }

    /// The admitted caller attaches loop state before persisting this batch; missing state fails closed.
    pub(crate) fn retain_batch(&self, runtime_id: &str, state: TaskBatchState) {
        self.parked_batches.lock().insert(runtime_id.into(), state);
    }

    /// A coordinating batch can park alongside acknowledged completed children, whose live owners remain retained.
    /// Pending child work still requires its own exact group cursor and cannot be mistaken for quiescence.
    pub(crate) async fn pause_batch(
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
            || payload
                .children
                .iter()
                .any(|child| child.result.is_none() || child.pending.is_some())
        {
            return Err(AgentError::Config(
                "group suspension requires exact composition continuation".into(),
            ));
        }
        let state = self
            .parked_batches
            .lock()
            .get(runtime_id)
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        state.validate()?;
        let pending = state
            .approvals
            .first()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        state.bind_runtime(&mut runtime, &self.run_id, self.current_cycle())?;
        payload.runtime = runtime;
        payload.todos = todos;
        if let Some(previous) = &payload.pending
            && previous.id != pending.request_id
        {
            payload.consumed_request_ids.push(previous.id.clone());
        }
        payload.pending = Some(TaskPendingRequest {
            id: pending.request_id.clone(),
            issued_revision: previous.revision + 1,
            kind: if pending.question.is_some() {
                TaskPendingKind::UserQuestion
            } else {
                TaskPendingKind::Approval
            },
            reviewed_action: json!(pending),
        });
        payload.pause_reason = Some(
            if pending.question.is_some() {
                "user_question"
            } else {
                "approval_required"
            }
            .into(),
        );
        payload.clocks.active_millis = self.active_millis()?;
        let adapter = TaskAdapterCheckpoint {
            id: "runtime.tool_batch".into(),
            adapter: "runtime.tool_batch".into(),
            contract_version: 1,
            config: json!({"runtime_id":runtime_id}),
            state: serde_json::to_value(state)?,
        };
        if let Some(existing) = payload
            .adapters
            .iter_mut()
            .find(|item| item.id == adapter.id)
        {
            *existing = adapter;
        } else {
            payload.adapters.push(adapter);
        }
        // Persist the parked continuation while ownership and the active clock remain held across storage acknowledgement.
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
        // Millisecond rounding must not erase a live interval when its last write and stop occur within the same tick.
        payload.clocks.active_millis = self.active_millis()?.max(
            payload
                .clocks
                .active_millis
                .checked_add(1)
                .ok_or_else(|| AgentError::Config("pause active clock overflow".into()))?,
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
                        .ok_or_else(|| AgentError::Config("pause deadline overflow".into()))?,
                },
            )
            .await
            .inspect_err(|_| self.stop("storage_failure"))?;
        self.acknowledge_snapshot(&saved);
        self.revision
            .store(saved.revision, std::sync::atomic::Ordering::Release);
        Ok(saved)
    }
}
