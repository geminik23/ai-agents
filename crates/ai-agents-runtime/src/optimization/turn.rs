use std::collections::HashMap;

use ai_agents_state::Transition;
use serde_json::Value;
use uuid::Uuid;

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
