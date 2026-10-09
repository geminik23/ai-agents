//! LLM providers for AI Agents framework

pub mod capability;
mod managed;
pub use managed::{
    current_invocation_admission, managed_completion, managed_provider, scope_invocation_admission,
};
pub mod mock;
pub mod multi;
pub mod prompts;
pub mod providers;
pub mod registry;
pub mod routing;

pub use ai_agents_core::{
    ChatMessage, FinishReason, LLMCapability, LLMChunk, LLMConfig, LLMError, LLMFeature,
    LLMProvider, LLMResponse, LLMToolDefinition, LLMToolRequest, Role, TaskContext, TokenUsage,
    ToolChoice, ToolSelection,
};
pub use registry::LLMRegistry;
pub use routing::*;
