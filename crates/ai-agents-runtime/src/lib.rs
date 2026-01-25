//! Runtime agent and builder for AI Agents framework

mod builder;
mod runtime;
mod streaming;

pub mod spec;

pub use builder::AgentBuilder;
pub use runtime::RuntimeAgent;
pub use streaming::{StreamChunk, StreamingConfig};

pub use ai_agents_core::{AgentInfo, AgentResponse, Result, ToolCall};

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
