//! Runtime agent and builder for AI Agents framework

mod builder;
pub mod optimization;
mod runtime;
mod streaming;
mod turn_context;

pub mod orchestration;
pub mod spawner;
pub mod spec;

pub use builder::AgentBuilder;
pub use optimization::{
    AwaitBeforeNextTurn, BackgroundOverflowPolicy, MainResponseDraft, MaintenanceMode,
    MaintenanceTaskPolicy, PostTurnOptimizationConfig, RuntimeBranch, RuntimeBranchOutcome,
    RuntimeBranchResult, RuntimeBranchStatus, RuntimeCommitBehavior, RuntimeConfig,
    RuntimeOptimizationConfig, RuntimeOptimizationKind, RuntimeTaskPriority, RuntimeTaskPurpose,
    ScheduledBranchSet, SkillCandidate, StreamBranchBuffer, StreamingDraftResult,
    StreamingOptimizationPolicy, TurnBranchScheduler, TurnOptimizationContext,
};
pub use runtime::{RuntimeAgent, RuntimeControlHandle};
pub use streaming::{StreamChunk, StreamingConfig};
pub use turn_context::TurnActorContext;

pub use ai_agents_core::{AgentInfo, AgentResponse, Result, ToolCall};

// Retry only transient Windows SQLite sharing violations so test cleanup remains strict for every other error.
#[cfg(test)]
pub(crate) async fn remove_sqlite_test_directory(path: &std::path::Path) -> std::io::Result<()> {
    const MAX_ATTEMPTS: usize = 20;
    const RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(50);

    for attempt in 0..MAX_ATTEMPTS {
        match std::fs::remove_dir_all(path) {
            Ok(()) => return Ok(()),
            Err(error)
                if cfg!(windows)
                    && error.raw_os_error() == Some(32)
                    && attempt + 1 < MAX_ATTEMPTS =>
            {
                tokio::time::sleep(RETRY_DELAY).await;
            }
            Err(error) => return Err(error),
        }
    }

    unreachable!("the final cleanup attempt always returns")
}

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelToolsConfig {
    #[serde(default = "default_parallel_enabled")]
    pub enabled: bool,
    #[serde(default = "default_max_parallel")]
    pub max_parallel: usize,
}

fn default_parallel_enabled() -> bool {
    true
}

fn default_max_parallel() -> usize {
    5
}

impl Default for ParallelToolsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_parallel: 5,
        }
    }
}

#[async_trait]
pub trait Agent: Send + Sync {
    async fn chat(&self, input: &str) -> Result<AgentResponse>;
    fn info(&self) -> AgentInfo;
    async fn reset(&self) -> Result<()>;
}
