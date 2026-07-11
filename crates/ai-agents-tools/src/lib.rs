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
pub use registry::{ResolvedTool, ToolIdentity, ToolRegistry, ToolSchemaPromptMode};
pub use types::{
    CommandRequest, CommandResponse, CommandRunner, CommandRunnerSlot, DiagnosticItem,
    DiagnosticSeverity, DiagnosticsProvider, DiagnosticsProviderSlot, DiagnosticsRequest,
    DiagnosticsResponse, FileVersionEvidence, FileVersionStore, ProcessCommandRunner,
    QuestionHandler, QuestionHandlerSlot, QuestionRequest, QuestionResponse, StaticCommandRunner,
    StaticDiagnosticsProvider, StaticWebSearchProvider, TodoItem, TodoStatus, TodoStore,
    ToolAliases, ToolContext, ToolMetadata, ToolProviderType, TrustLevel, UnavailableCommandRunner,
    UnavailableDiagnosticsProvider, UnavailableWebSearchProvider, WebSearchProvider,
    WebSearchProviderSlot, WebSearchRequest, WebSearchResponse, WebSearchResultItem,
    WebSearchSafeSearch, file_version_evidence,
};

pub use builtin::HttpTool;
pub use builtin::{
    AskUserTool, CalculatorTool, CommandTool, CopyPathTool, DateTimeTool, DeletePathTool,
    DiagnosticsTool, EchoTool, FileEditTool, FileInfoTool, FileListTool, FileReadTool, FileTool,
    FileWriteTool, GitDiffTool, GitStatusTool, GlobTool, GrepTool, JsonTool, MathTool,
    MovePathTool, PatchTool, RandomTool, SleepTool, TemplateTool, TextTool, TodoTool,
    WebFetchResolver, WebFetchTool, WebFetchTransport, WebFetchTransportRequest,
    WebFetchTransportResponse, WebSearchTool,
};

pub use security::{
    CommandPolicyConfig, CommandRuleConfig, CommandTemplateConfig, DomainPolicyConfig,
    NoWritePolicyBehavior, OperationPolicyConfig, PathPolicyConfig, SecurityCheckResult,
    ToolPolicyConfig, ToolSecurityConfig, ToolSecurityEngine,
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
    let file_versions = registry.file_version_store();
    registry
        .register(Arc::new(FileReadTool::with_version_store(
            file_versions.clone(),
        )))
        .expect("failed to register file_read");
    registry
        .register(Arc::new(FileWriteTool::with_version_store(
            file_versions.clone(),
        )))
        .expect("failed to register file_write");
    registry
        .register(Arc::new(FileEditTool::with_version_store(
            file_versions.clone(),
        )))
        .expect("failed to register file_edit");
    registry
        .register(Arc::new(PatchTool::with_version_store(file_versions)))
        .expect("failed to register patch");
    registry
        .register(Arc::new(CopyPathTool::new()))
        .expect("failed to register copy_path");
    registry
        .register(Arc::new(MovePathTool::new()))
        .expect("failed to register move_path");
    registry
        .register(Arc::new(DeletePathTool::new()))
        .expect("failed to register delete_path");
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
        .register(Arc::new(WebSearchTool::with_provider_slot(
            registry.web_search_provider_slot(),
        )))
        .expect("failed to register web_search");
    registry
        .register(Arc::new(CommandTool::new(registry.command_runner_slot())))
        .expect("failed to register command");
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
