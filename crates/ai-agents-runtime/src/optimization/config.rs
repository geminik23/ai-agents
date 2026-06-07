use serde::{Deserialize, Serialize};

use ai_agents_core::{AgentError, Result};

/// Runtime-level configuration for latency optimization.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Policies that reduce latency without changing behavior by default.
    pub optimization: RuntimeOptimizationConfig,
}

/// Controls safe pre-response routing, maintenance concurrency, and runtime task limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeOptimizationConfig {
    /// Enables runtime optimization behavior. Disabled agents keep serial behavior.
    pub enabled: bool,
    /// Hard cap for additional speculative LLM calls in one turn.
    pub max_speculative_llm_calls_per_turn: u32,
    /// Runs explicitly marked guard and resolved-intent transitions before old-state response generation.
    pub pre_response_deterministic_transitions: bool,
    /// Runs current-state extractors before pre-response transition selection when requested.
    pub pre_response_extractors: bool,
    /// Enables response-independent transition branches beside a draft response.
    pub speculative_state_transitions: bool,
    /// Enables pure skill routing beside a draft response.
    pub speculative_skill_routing: bool,
    /// Enables auto reasoning decisions beside a plain draft response.
    pub speculative_reasoning_auto: bool,
    /// Allows facts and relationship maintenance to run concurrently when configured.
    pub parallel_post_turn_memory: bool,
    /// Allows orchestration vote extraction to run concurrently while preserving order.
    pub parallel_orchestration_vote_extraction: bool,
    /// Reserved for snapshot-based observability export outside the response path.
    pub background_observability_export: bool,
    /// Streaming safety policy used when optimization is enabled.
    pub streaming_policy: StreamingOptimizationPolicy,
    /// Maximum internal runtime tasks scheduled at once.
    pub max_parallel_runtime_tasks: usize,
    /// Post-turn maintenance policy for future-turn work.
    pub post_turn: PostTurnOptimizationConfig,
}

impl Default for RuntimeOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_speculative_llm_calls_per_turn: 0,
            pre_response_deterministic_transitions: false,
            pre_response_extractors: false,
            speculative_state_transitions: false,
            speculative_skill_routing: false,
            speculative_reasoning_auto: false,
            parallel_post_turn_memory: false,
            parallel_orchestration_vote_extraction: false,
            background_observability_export: false,
            streaming_policy: StreamingOptimizationPolicy::PreflightOnly,
            max_parallel_runtime_tasks: 4,
            post_turn: PostTurnOptimizationConfig::default(),
        }
    }
}

impl RuntimeOptimizationConfig {
    /// Validates optimization settings before the runtime is built.
    pub fn validate(&self) -> Result<()> {
        if self.max_parallel_runtime_tasks == 0 {
            return Err(AgentError::InvalidSpec(
                "runtime.optimization.max_parallel_runtime_tasks must be greater than 0".into(),
            ));
        }
        if self.post_turn.max_background_tasks == 0 && self.post_turn.any_background_tasks_enabled()
        {
            return Err(AgentError::InvalidSpec(
                "runtime.optimization.post_turn.max_background_tasks must be greater than 0 when background maintenance is enabled".into(),
            ));
        }
        if self.background_observability_export {
            return Err(AgentError::InvalidSpec(
                "runtime.optimization.background_observability_export requires snapshot export support and is not enabled yet".into(),
            ));
        }
        let any_speculative = self.speculative_state_transitions
            || self.speculative_skill_routing
            || self.speculative_reasoning_auto;
        if any_speculative {
            if !self.enabled {
                return Err(AgentError::InvalidSpec(
                    "runtime.optimization.enabled must be true when speculative branch settings are enabled".into(),
                ));
            }
            if self.max_speculative_llm_calls_per_turn == 0 {
                return Err(AgentError::InvalidSpec(
                    "runtime.optimization.max_speculative_llm_calls_per_turn must be greater than 0 when speculative branch settings are enabled".into(),
                ));
            }
        }
        if self.max_speculative_llm_calls_per_turn > self.max_parallel_runtime_tasks as u32 {
            return Err(AgentError::InvalidSpec(
                "runtime.optimization.max_speculative_llm_calls_per_turn must be less than or equal to max_parallel_runtime_tasks".into(),
            ));
        }
        if self.post_turn.sessions != MaintenanceTaskPolicy::default() {
            return Err(AgentError::InvalidSpec(
                "runtime.optimization.post_turn.sessions is reserved until session maintenance scheduling is enabled".into(),
            ));
        }
        if self.post_turn.memory_compression != MaintenanceTaskPolicy::default() {
            return Err(AgentError::InvalidSpec(
                "runtime.optimization.post_turn.memory_compression is reserved until compression scheduling is enabled".into(),
            ));
        }
        Ok(())
    }
}

/// Post-turn task policies for work that affects later turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PostTurnOptimizationConfig {
    /// Fact extraction policy.
    pub facts: MaintenanceTaskPolicy,
    /// Relationship update policy.
    pub relationships: MaintenanceTaskPolicy,
    /// Session metadata policy.
    pub sessions: MaintenanceTaskPolicy,
    /// Memory compression policy.
    pub memory_compression: MaintenanceTaskPolicy,
    /// Maximum number of queued background tasks.
    pub max_background_tasks: usize,
    /// Behavior when the background queue is full.
    pub on_background_overflow: BackgroundOverflowPolicy,
}

impl Default for PostTurnOptimizationConfig {
    fn default() -> Self {
        Self {
            facts: MaintenanceTaskPolicy {
                mode: MaintenanceMode::InlineSerial,
                await_before_next_turn: AwaitBeforeNextTurn::Always,
            },
            relationships: MaintenanceTaskPolicy {
                mode: MaintenanceMode::InlineSerial,
                await_before_next_turn: AwaitBeforeNextTurn::Always,
            },
            sessions: MaintenanceTaskPolicy::default(),
            memory_compression: MaintenanceTaskPolicy::default(),
            max_background_tasks: 16,
            on_background_overflow: BackgroundOverflowPolicy::RunInline,
        }
    }
}

impl PostTurnOptimizationConfig {
    /// Returns true when any maintenance task may run outside the response path.
    pub fn any_background_tasks_enabled(&self) -> bool {
        matches!(self.facts.mode, MaintenanceMode::Background)
            || matches!(self.relationships.mode, MaintenanceMode::Background)
            || matches!(self.sessions.mode, MaintenanceMode::Background)
            || matches!(self.memory_compression.mode, MaintenanceMode::Background)
    }
}

/// Policy for one post-turn maintenance task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct MaintenanceTaskPolicy {
    /// Whether the task runs serially, concurrently, or in the background.
    pub mode: MaintenanceMode,
    /// Whether a later turn waits for pending background work.
    pub await_before_next_turn: AwaitBeforeNextTurn,
}

impl Default for MaintenanceTaskPolicy {
    fn default() -> Self {
        Self {
            mode: MaintenanceMode::InlineSerial,
            await_before_next_turn: AwaitBeforeNextTurn::Always,
        }
    }
}

/// Execution mode for post-turn maintenance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaintenanceMode {
    /// Run in the existing serial response path.
    InlineSerial,
    /// Run with other independent maintenance tasks and await completion.
    InlineParallel,
    /// Queue work after the response and apply freshness policy later.
    Background,
}

impl Default for MaintenanceMode {
    fn default() -> Self {
        Self::InlineSerial
    }
}

/// Freshness policy for pending background maintenance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AwaitBeforeNextTurn {
    /// Never wait for this task before a new turn.
    Never,
    /// Wait before a turn from the same actor.
    SameActor,
    /// Wait before every new turn.
    Always,
}

impl Default for AwaitBeforeNextTurn {
    fn default() -> Self {
        Self::Always
    }
}

/// Behavior when a background queue cannot accept more tasks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundOverflowPolicy {
    /// Run the task inline instead of dropping it.
    RunInline,
    /// Drop the task and record skipped maintenance.
    Drop,
    /// Return an error to the caller.
    Error,
}

impl Default for BackgroundOverflowPolicy {
    fn default() -> Self {
        Self::RunInline
    }
}

/// Streaming behavior when optimized routing may run before output begins.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StreamingOptimizationPolicy {
    /// Run safe preflight routing before opening the stream.
    PreflightOnly,
    /// Buffer unresolved stream output until routing decisions finish.
    BufferUntilRoutingDone,
    /// Disable optimized streaming behavior.
    Disabled,
}

impl Default for StreamingOptimizationPolicy {
    fn default() -> Self {
        Self::PreflightOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_buffered_streaming_policy() {
        let config = RuntimeOptimizationConfig {
            enabled: true,
            streaming_policy: StreamingOptimizationPolicy::BufferUntilRoutingDone,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_reserved_session_maintenance_policy() {
        let mut config = RuntimeOptimizationConfig {
            enabled: true,
            ..Default::default()
        };
        config.post_turn.sessions.mode = MaintenanceMode::Background;
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_reserved_compression_maintenance_policy() {
        let mut config = RuntimeOptimizationConfig {
            enabled: true,
            ..Default::default()
        };
        config.post_turn.memory_compression.mode = MaintenanceMode::Background;
        assert!(config.validate().is_err());
    }

    #[test]
    fn speculative_flags_require_positive_cap() {
        let config = RuntimeOptimizationConfig {
            enabled: true,
            speculative_skill_routing: true,
            max_speculative_llm_calls_per_turn: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn speculative_cap_must_fit_parallel_limit() {
        let config = RuntimeOptimizationConfig {
            enabled: true,
            speculative_skill_routing: true,
            max_speculative_llm_calls_per_turn: 5,
            max_parallel_runtime_tasks: 4,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
