//! Non-tool interactions retain their own semantic/transition positions without inventing model calls.

use super::*;
use ai_agents_core::{AgentError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, future::Future};

/// Exact pre-tool model output survives a rejected transition without another provider request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskModelCalls {
    pub state: Box<TaskLoopState>,
    pub content: String,
    pub calls: Vec<ai_agents_core::ToolCall>,
}

/// This discriminant identifies the existing driver and never supplies execution authority by itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TaskInteractionKind {
    PlanReview {
        plan: ai_agents_reasoning::Plan,
        attempt: u32,
        processed_input: String,
        input_context: HashMap<String, Value>,
        auto_detected: bool,
        reasoning_config: Value,
    },
    StateApproval {
        from_state: String,
        target: String,
        reason: String,
        staged: Option<HashMap<String, Value>>,
        processed_input: Option<String>,
        definition: Value,
        return_to: Box<location::TaskStateReturn>,
    },
    Clarification {
        manager: Value,
        input: String,
        prepared: Option<ai_agents_process::ProcessData>,
        skill_id: Option<String>,
    },
    Escalation {
        input: String,
        reason: String,
        #[serde(default)]
        prepared: Option<ai_agents_process::ProcessData>,
        #[serde(default)]
        skill_id: Option<String>,
    },
}

/// Foreground non-tool cursors are not composition groups even though they have no native tool batch.
pub(crate) fn has_root_interaction(payload: &TaskCheckpointPayload) -> bool {
    matches!(
        payload.runtime.continuation,
        TaskContinuation::Suspended { .. }
    ) && payload.pending.is_some()
        && payload.pause_reason.as_deref() != Some("child_interaction")
        && payload.adapters.iter().any(|adapter| {
            adapter.id == "runtime.tool_batch"
                && adapter.config.get("runtime_id").and_then(Value::as_str)
                    == Some(payload.runtime.snapshot.agent_id.as_str())
                && adapter
                    .state
                    .get("interaction")
                    .is_some_and(Value::is_object)
        })
}

/// One live exchange retains root ownership and the generation that selected its operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskInteraction {
    pub id: String,
    pub kind: TaskInteractionKind,
    pub state_generation: Option<u64>,
    pub epoch: u64,
    pub runtime_control_version: u64,
    pub source: AutonomyTurnSource,
    pub user_message_committed: bool,
    pub native_exchanges: Vec<TaskNativeExchange>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl TaskInteraction {
    /// Non-tool answers are bounded natural-language data; approval and tool answer schemas remain separate.
    pub(crate) fn validate_answer(&self, answer: &Value) -> Result<()> {
        if !matches!(self.kind, TaskInteractionKind::Clarification { .. })
            || answer
                .as_str()
                .is_none_or(|answer| answer.trim().is_empty())
        {
            return Err(AgentError::Config(
                "clarification requires a nonempty text answer".into(),
            ));
        }
        bounded_value(answer, 65_536)
    }

    /// Structural progress is checked separately from the live manager, state definition and root owner.
    pub(crate) fn validate(&self, batch: &TaskBatchState) -> Result<()> {
        if self.id.is_empty()
            || self.request_id.as_ref().is_some_and(|id| {
                batch.approvals.first().is_none_or(|approval| {
                    &approval.request_id != id || approval.expires_at != self.expires_at
                })
            })
            || !batch.calls.is_empty()
            || !batch.results.is_empty()
            || !batch.content.is_empty()
            || batch.appended != 0
            || batch.approvals.len() != 1
            || batch.approvals[0].call_id != self.id
            || batch.location.is_some()
            || batch.loop_state.is_some()
            || !batch.executor_cursors.is_empty()
            || !batch.authorized.is_empty()
            || matches!(self.kind, TaskInteractionKind::Clarification { .. })
                != batch.approvals[0].question.is_some()
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        bounded_value(&serde_json::to_value(self)?, MAX_TASK_CHECKPOINT_BYTES)
    }
}

tokio::task_local! {
    static PREPARED_INPUT: Box<ai_agents_process::ProcessData>;
    static CLARIFICATION_RETURN: Box<TaskInteractionKind>;
    static SKIP_TOOL_TRANSITION: bool;
}

/// The admitted input driver retains processed data for skill clarification instead of replaying input processors.
pub(crate) fn scope_prepared_input<F: Future>(
    data: ai_agents_process::ProcessData,
    future: F,
) -> impl Future<Output = F::Output> {
    PREPARED_INPUT.scope(Box::new(data), future)
}

/// Semantic answer parsing may exhaust clarification; the original live return position remains available until dispatch settles.
pub(crate) fn scope_clarification_return<F: Future>(
    state: TaskInteractionKind,
    future: F,
) -> impl Future<Output = F::Output> {
    CLARIFICATION_RETURN.scope(Box::new(state), future)
}

/// Only the admitted clarification driver supplies escalation context, never a new answer or serialized control request.
pub(crate) fn clarification_return() -> Option<TaskInteractionKind> {
    CLARIFICATION_RETURN
        .try_with(|state| (**state).clone())
        .ok()
}

/// Only a framework-installed processed input can become a skill clarification return position.
pub(crate) fn prepared_input() -> Option<ai_agents_process::ProcessData> {
    PREPARED_INPUT.try_with(|data| (**data).clone()).ok()
}

/// Resolved skill execution keeps the original input-processing context instead of replacing it with an empty map.
pub(crate) fn prepared_context() -> HashMap<String, Value> {
    prepared_input()
        .map(|data| data.context)
        .unwrap_or_default()
}

/// Rejected pre-tool transitions return to the already-produced calls, not another semantic selection.
pub(crate) fn scope_skipped_tool_transition<F: Future>(
    future: F,
) -> impl Future<Output = F::Output> {
    SKIP_TOOL_TRANSITION.scope(true, future)
}

/// Regenerated model work must not reevaluate the already committed transition that selected this response driver.
pub(crate) fn regenerating_model() -> bool {
    match location::current_state_return() {
        Some(location::TaskStateReturn::ModelContinue { state }) => state.deferred_final.is_some(),
        Some(location::TaskStateReturn::BeforeCalls { saved }) => {
            saved.state.deferred_final.is_some()
        }
        _ => false,
    }
}

/// Skip selection only inside the exact saved pre-tool response driver.
pub(crate) fn skip_tool_transition() -> bool {
    SKIP_TOOL_TRANSITION.try_with(|skip| *skip).unwrap_or(false)
}
