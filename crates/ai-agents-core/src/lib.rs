//! Core types and traits for AI Agents framework

pub mod dot_path;
pub mod error;
pub mod message;
pub mod traits;
pub mod types;

pub use dot_path::{get_dot_path, get_dot_path_from_map, set_dot_path};
pub use error::{AgentError, Result};
pub use message::{ChatMessage, Role};
pub use traits::llm::{LLMCapability, LLMError, LLMProvider, TaskContext, ToolSelection};
pub use traits::memory::{Memory, MemorySnapshot};
pub use traits::storage::{
    AgentSnapshot, AgentStorage, NoopStorage, SpawnedAgentEntry, StorageCapability,
};
pub use traits::tool::{Tool, ToolInfo, ToolInvoker, ToolResult};
pub use types::{
    AgentInfo, AgentResponse, CommandBindingKind, CommandPolicyBinding, DomainPolicyBinding,
    FactCategory, FactFilter, FinishReason, KeyFact, LLMChunk, LLMConfig, LLMFeature, LLMResponse,
    LLMToolDefinition, LLMToolRequest, MAX_TOOL_TIMEOUT_MS, PathAccessMode, PathBindingKind,
    PathPolicyBinding, PermissionOutcome, ResultLimitBinding, ResultLimitKind, SessionFilter,
    SessionMetadata, SessionSummary, StateMachineSnapshot, StateTransitionEvent, TokenUsage,
    ToolActorContext, ToolApprovalRecord, ToolApprovalStatus, ToolCall, ToolCallClassification,
    ToolCallSource, ToolCancellationToken, ToolChoice, ToolExecutionContext, ToolExecutionLimits,
    ToolExecutionRecord, ToolExecutionRequest, ToolOperationKind, ToolPolicyBindings,
    ToolPolicyDecisionRecord, ToolSafetyMetadata, ToolSideEffectLevel,
};
