//! Exact non-model locations share task owner and executor custody without manufacturing model history.

use super::*;
use ai_agents_core::{AgentError, Result, ToolExecutionRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// A script owns its actual pending request, completed template inputs and already prepared root input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskSkillLocation {
    pub cursor: ai_agents_skills::SkillExecutionCursor,
    pub input_context: HashMap<String, Value>,
    pub user_message_committed: bool,
    pub native_exchanges: Vec<TaskNativeExchange>,
    #[serde(default)]
    pub actions: Option<Box<TaskActionsLocation>>,
    pub source: AutonomyTurnSource,
    pub state_generation: Option<u64>,
}

/// A response already processed by the winning model retains metadata while state actions are unfinished.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskModelFinal {
    pub processed_input: String,
    pub input_context: HashMap<String, Value>,
    pub content: String,
    pub all_tool_calls: Vec<ai_agents_core::ToolCall>,
    pub reasoning_mode: ai_agents_reasoning::ReasoningMode,
    pub auto_detected: bool,
    pub iterations: u32,
    pub thinking: Option<String>,
    pub reflection_metadata: Option<ai_agents_reasoning::ReflectionMetadata>,
    pub finalize_on_resume: bool,
}

/// Return positions select existing runtime drivers without repeating input processing, semantic selection or finalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TaskStateReturn {
    Prepared {
        data: ai_agents_process::ProcessData,
    },
    ModelContinue {
        state: Box<TaskLoopState>,
    },
    ModelFinish {
        state: Box<TaskModelFinal>,
    },
    SkillFinish {
        response: ai_agents_core::AgentResponse,
    },
    ReadyContext {
        input: String,
    },
}

/// Transition commit follows all exit actions and precedes entry actions exactly once.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskTransitionLocation {
    pub from_state: String,
    pub target: String,
    pub reason: String,
    pub staged: Option<HashMap<String, Value>>,
    pub history_before: Vec<ai_agents_state::StateTransitionEvent>,
    pub entering: bool,
    pub expected_generation: u64,
    pub expected_epoch: u64,
}

/// A state action list retains completed effects and final template arguments, not an index into changing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskActionsLocation {
    pub actions: Vec<ai_agents_state::StateAction>,
    pub next_action: usize,
    pub completed: Vec<Value>,
    pub pending: Option<ToolExecutionRequest>,
    pub state: String,
    pub transition: TaskTransitionLocation,
    pub return_to: TaskStateReturn,
    pub source: AutonomyTurnSource,
    pub user_message_committed: bool,
    pub native_exchanges: Vec<TaskNativeExchange>,
}

impl TaskActionsLocation {
    /// Only a complete action prefix can authorize advancement to the next saved operation.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.actions.len() > MAX_TASK_CHECKPOINT_RECORDS
            || self.next_action >= self.actions.len()
            || self.completed.len() != self.next_action
            || self.state.is_empty()
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        if let Some(request) = &self.pending
            && (!matches!(&request.source, ai_agents_core::ToolCallSource::StateAction {state:Some(state),action_index} if state == &self.state && *action_index == self.next_action)
                || !matches!(self.actions.get(self.next_action), Some(ai_agents_state::StateAction::Tool {tool,..}) if tool == &request.requested_name))
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        bounded_value(&serde_json::to_value(self)?, MAX_TASK_CHECKPOINT_BYTES)
    }
}

tokio::task_local! {
    static STATE_RETURN: Box<TaskStateReturn>;
    static ACTIONS_PARENT: Box<TaskActionsLocation>;
}

/// Only the live caller can install a return position; provider callbacks cannot invent one from persisted labels.
pub(crate) fn scope_state_return<F: std::future::Future>(
    position: TaskStateReturn,
    future: F,
) -> impl std::future::Future<Output = F::Output> {
    STATE_RETURN.scope(Box::new(position), future)
}

/// Captured state return data is supplied by the currently admitted root driver.
pub(crate) fn current_state_return() -> Option<TaskStateReturn> {
    STATE_RETURN.try_with(|position| (**position).clone()).ok()
}

/// A nested skill inherits the exact state action and transition cursor, never another root owner.
pub(crate) fn scope_actions_parent<F: std::future::Future>(
    position: TaskActionsLocation,
    future: F,
) -> impl std::future::Future<Output = F::Output> {
    ACTIONS_PARENT.scope(Box::new(position), future)
}

/// Nested script retention captures its actual caller rather than inferring ownership from a skill ID.
pub(crate) fn current_actions_parent() -> Option<TaskActionsLocation> {
    ACTIONS_PARENT
        .try_with(|position| (**position).clone())
        .ok()
}

/// The discriminant selects the existing executable driver, never a serialized execution capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TaskLocation {
    Skill(Box<TaskSkillLocation>),
    Actions(Box<TaskActionsLocation>),
}

impl TaskLocation {
    /// Structural validation is independent of live implementation binding, which resume checks before claiming.
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Skill(state) => {
                state.cursor.validate(&state.cursor.definition)?;
                if state.cursor.pending.is_none() {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
                bounded_value(&serde_json::to_value(state)?, MAX_TASK_CHECKPOINT_BYTES)
            }
            Self::Actions(state) => state.validate(),
        }
    }

    /// A location binds an actual pending operation rather than synthesizing a model call from a label.
    pub(crate) fn request(&self) -> Result<&ToolExecutionRequest> {
        match self {
            Self::Skill(state) => state
                .cursor
                .pending
                .as_ref()
                .ok_or_else(|| AgentError::from(TaskRunStorageError::InvalidCheckpoint)),
            Self::Actions(state) => state
                .pending
                .as_ref()
                .ok_or_else(|| AgentError::from(TaskRunStorageError::InvalidCheckpoint)),
        }
    }

    /// Resumption inherits the original classified input, not a new user/root turn.
    pub(crate) fn source(&self) -> AutonomyTurnSource {
        match self {
            Self::Skill(state) => state.source,
            Self::Actions(state) => state.source,
        }
    }

    /// Root commit ownership and signed expectations survive unwinding the original private driver.
    pub(crate) fn root_state(&self) -> (bool, &[TaskNativeExchange]) {
        match self {
            Self::Skill(state) => (state.user_message_committed, &state.native_exchanges),
            Self::Actions(state) => (state.user_message_committed, &state.native_exchanges),
        }
    }
}
