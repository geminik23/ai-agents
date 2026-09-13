//! Core types and traits for AI Agents framework

pub mod dot_path;
pub mod error;
pub mod message;
pub mod native_history;
pub mod traits;
pub mod types;

pub use dot_path::{get_dot_path, get_dot_path_from_map, set_dot_path};
pub use error::{AgentError, Result};
pub use message::{ChatMessage, Role};
pub use native_history::{
    MAX_NATIVE_PROVIDER_STATE_BYTES, NATIVE_PROVIDER_STATE_METADATA_KEY,
    NATIVE_PROVIDER_STATE_VERSION, NATIVE_TOOL_CALL_MARKER_KEY, NATIVE_TOOL_RESULT_MARKER_KEY,
    NativeCallBinding, NativeExchange, NativeHistoryInspection, NativeProviderState,
    NativeProviderTarget, NativeToolCallBatch, NativeToolResult, decode_native_tool_call_markers,
    decode_native_tool_result_markers, encode_native_tool_call_markers,
    encode_native_tool_result_marker, inspect_native_history, native_execution_projection,
    native_observation_projection, native_readable_projection, response_provider_state,
    set_response_provider_state, take_response_provider_state,
};
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
