//! Tool system for AI Agents framework

pub mod builtin;
mod condition;
pub mod mcp;
mod provider;
mod registry;
pub mod security;
mod types;

pub use ai_agents_core::{
    CommandBindingKind, CommandPolicyBinding, DomainPolicyBinding, PathAccessMode, PathBindingKind,
    PathPolicyBinding, PermissionOutcome, ResultLimitBinding, ResultLimitKind, Tool,
    ToolActorContext, ToolApprovalRecord, ToolApprovalStatus, ToolCallClassification,
    ToolCallSource, ToolCancellationToken, ToolExecutionContext, ToolExecutionLimits,
    ToolExecutionRecord, ToolExecutionRequest, ToolInfo, ToolInvoker, ToolOperationKind,
    ToolPolicyBindings, ToolPolicyDecisionRecord, ToolResult, ToolSafetyMetadata,
    ToolSideEffectLevel,
};
pub use condition::{
    ConditionEvaluator, EvaluationContext, LLMGetter, SimpleLLMGetter, ToolCallRecord,
};
pub use provider::{ProviderHealth, ToolDescriptor, ToolProvider, ToolProviderError};
pub use registry::{ResolvedTool, ToolIdentity, ToolRegistry};
pub use types::{
    DiagnosticItem, DiagnosticSeverity, DiagnosticsProvider, DiagnosticsProviderSlot,
    DiagnosticsRequest, DiagnosticsResponse, QuestionHandler, QuestionHandlerSlot, QuestionRequest,
    QuestionResponse, StaticDiagnosticsProvider, TodoItem, TodoStatus, TodoStore, ToolAliases,
    ToolContext, ToolMetadata, ToolProviderType, TrustLevel, UnavailableDiagnosticsProvider,
};

pub use builtin::HttpTool;
pub use builtin::{
    AskUserTool, CalculatorTool, DateTimeTool, DiagnosticsTool, EchoTool, FileInfoTool,
    FileListTool, FileReadTool, FileTool, GitDiffTool, GitStatusTool, GlobTool, GrepTool, JsonTool,
    MathTool, RandomTool, SleepTool, TemplateTool, TextTool, TodoTool, WebFetchTool,
};

pub use security::{
    CommandPolicyConfig, DomainPolicyConfig, OperationPolicyConfig, PathPolicyConfig,
    SecurityCheckResult, ToolPolicyConfig, ToolSecurityConfig, ToolSecurityEngine,
};

use schemars::JsonSchema;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Tool already registered: {0}")]
    AlreadyRegistered(String),
    #[error("Duplicate: {0}")]
    Duplicate(String),
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("Provider error: {0}")]
    Provider(String),
}

pub fn generate_schema<T: JsonSchema>() -> serde_json::Value {
    let schema = schemars::schema_for!(T);
    serde_json::to_value(schema).unwrap_or_else(|_| serde_json::json!({}))
}

pub fn create_builtin_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register(Arc::new(CalculatorTool::new()))
        .expect("failed to register calculator");
    registry
        .register(Arc::new(EchoTool::new()))
        .expect("failed to register echo");
    registry
        .register(Arc::new(DateTimeTool::new()))
        .expect("failed to register datetime");
    registry
        .register(Arc::new(JsonTool::new()))
        .expect("failed to register json");
    registry
        .register(Arc::new(RandomTool::new()))
        .expect("failed to register random");
    registry
        .register(Arc::new(FileTool::new()))
        .expect("failed to register file");
    registry
        .register(Arc::new(GlobTool::new()))
        .expect("failed to register glob");
    registry
        .register(Arc::new(GrepTool::new()))
        .expect("failed to register grep");
    registry
        .register(Arc::new(FileReadTool::new()))
        .expect("failed to register file_read");
    registry
        .register(Arc::new(FileListTool::new()))
        .expect("failed to register file_list");
    registry
        .register(Arc::new(FileInfoTool::new()))
        .expect("failed to register file_info");
    registry
        .register(Arc::new(GitStatusTool::new()))
        .expect("failed to register git_status");
    registry
        .register(Arc::new(GitDiffTool::new()))
        .expect("failed to register git_diff");
    registry
        .register(Arc::new(DiagnosticsTool::new(
            registry.diagnostics_provider_slot(),
        )))
        .expect("failed to register diagnostics");
    registry
        .register(Arc::new(AskUserTool::new(registry.question_handler_slot())))
        .expect("failed to register ask_user");
    registry
        .register(Arc::new(TodoTool::new(registry.todo_store())))
        .expect("failed to register todo");
    registry
        .register(Arc::new(SleepTool::new()))
        .expect("failed to register sleep");
    registry
        .register(Arc::new(WebFetchTool::with_extractor_slot(
            registry.web_fetch_extractor_slot(),
        )))
        .expect("failed to register web_fetch");
    registry
        .register(Arc::new(TextTool::new()))
        .expect("failed to register text");
    registry
        .register(Arc::new(TemplateTool::new()))
        .expect("failed to register template");
    registry
        .register(Arc::new(MathTool::new()))
        .expect("failed to register math");
    registry
        .register(Arc::new(HttpTool::new()))
        .expect("failed to register http");
    registry
}
