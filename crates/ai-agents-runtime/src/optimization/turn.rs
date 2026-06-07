use std::collections::HashMap;

use ai_agents_state::Transition;
use serde_json::Value;
use uuid::Uuid;

use super::branch::RuntimeOptimizationKind;

/// Tracks lifecycle and staged writes for one optimized runtime turn.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TurnOptimizationContext {
    /// Stable ID shared by branch and maintenance work for the current turn.
    pub turn_id: Uuid,
    /// Processed user input after the input pipeline.
    pub processed_input: String,
    /// Context values produced by the input process pipeline.
    pub input_context: HashMap<String, Value>,
    /// Context writes that must only commit if the selected path wins.
    pub staged_context_writes: HashMap<String, Value>,
    /// Whether actor memory and relationship loading ran for this turn.
    pub pre_turn_lifecycle_completed: bool,
    /// Whether the user message has been written to memory.
    pub user_message_committed: bool,
    /// Whether post-turn maintenance has been scheduled or completed.
    pub post_turn_lifecycle_completed: bool,
    /// Redispatch depth used to avoid repeated lifecycle work.
    pub redispatch_depth: u32,
    /// Number of speculative LLM calls used in this turn.
    pub speculative_llm_calls_used: u32,
    /// Maximum speculative LLM calls allowed in this turn.
    pub max_speculative_llm_calls: u32,
}

#[allow(dead_code)]
impl TurnOptimizationContext {
    /// Creates a new turn context with no staged writes.
    pub fn new(
        processed_input: impl Into<String>,
        input_context: HashMap<String, Value>,
        max_speculative_llm_calls: u32,
    ) -> Self {
        Self {
            turn_id: Uuid::new_v4(),
            processed_input: processed_input.into(),
            input_context,
            staged_context_writes: HashMap::new(),
            pre_turn_lifecycle_completed: false,
            user_message_committed: false,
            post_turn_lifecycle_completed: false,
            redispatch_depth: 0,
            speculative_llm_calls_used: 0,
            max_speculative_llm_calls,
        }
    }

    /// Returns true when another speculative LLM call can be scheduled.
    pub fn reserve_speculative_llm_call(&mut self) -> bool {
        if self.speculative_llm_calls_used >= self.max_speculative_llm_calls {
            return false;
        }
        self.speculative_llm_calls_used += 1;
        true
    }

    pub fn reserve_speculative_llm_call_for(&mut self, _kind: RuntimeOptimizationKind) -> bool {
        self.reserve_speculative_llm_call()
    }

    pub fn release_or_mark_failed_reservation(&mut self, _kind: RuntimeOptimizationKind) {}

    pub fn can_schedule_branch(&self, active_tasks: usize, max_parallel_tasks: usize) -> bool {
        active_tasks < max_parallel_tasks
    }

    pub fn stage_context_write(&mut self, key: impl Into<String>, value: Value) {
        self.staged_context_writes.insert(key.into(), value);
    }

    pub fn take_staged_context_writes(&mut self) -> HashMap<String, Value> {
        std::mem::take(&mut self.staged_context_writes)
    }

    /// Returns true when the turn can schedule the requested number of branch calls.
    pub fn reserve_speculative_llm_calls(&mut self, count: u32) -> bool {
        if self.speculative_llm_calls_used + count > self.max_speculative_llm_calls {
            return false;
        }
        self.speculative_llm_calls_used += count;
        true
    }

    /// Marks the root user message as committed.
    pub fn mark_user_message_committed(&mut self) {
        self.user_message_committed = true;
    }

    /// Marks post-turn lifecycle work as completed.
    pub fn mark_post_turn_lifecycle_completed(&mut self) {
        self.post_turn_lifecycle_completed = true;
    }

    /// Enters a redispatch scope.
    pub fn enter_redispatch(&mut self) {
        self.redispatch_depth += 1;
    }

    /// Leaves a redispatch scope.
    pub fn exit_redispatch(&mut self) {
        self.redispatch_depth = self.redispatch_depth.saturating_sub(1);
    }
}

/// Selected transition with enough data to commit side effects later.
#[derive(Debug, Clone)]
pub struct TransitionCandidate {
    /// State path where the transition was selected.
    pub from_state: String,
    /// Selected transition definition.
    pub transition: Transition,
    /// Reason recorded in state history and hooks.
    pub reason: String,
}

impl TransitionCandidate {
    /// Creates a candidate from the current state and transition.
    pub fn new(
        from_state: impl Into<String>,
        transition: Transition,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            from_state: from_state.into(),
            transition,
            reason: reason.into(),
        }
    }

    /// Returns the transition target string from the YAML definition.
    pub fn target(&self) -> &str {
        &self.transition.to
    }
}
