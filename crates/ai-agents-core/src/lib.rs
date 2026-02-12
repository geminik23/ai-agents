//! Core types and traits for AI Agents framework

pub mod error;
pub mod message;
pub mod traits;
pub mod types;

pub use error::{AgentError, Result};
pub use message::{ChatMessage, Role};
pub use traits::llm::{LLMCapability, LLMError, LLMProvider, TaskContext, ToolSelection};
pub use traits::memory::{Memory, MemorySnapshot};
pub use traits::storage::{AgentSnapshot, AgentStorage};
pub use traits::tool::{Tool, ToolInfo, ToolResult};
pub use types::{
    AgentInfo, AgentResponse, FinishReason, LLMChunk, LLMConfig, LLMFeature, LLMResponse,
    StateMachineSnapshot, StateTransitionEvent, TokenUsage, ToolCall,
};
