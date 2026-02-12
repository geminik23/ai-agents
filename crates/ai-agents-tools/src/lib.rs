//! Tool system for AI Agents framework

pub mod builtin;
mod condition;
mod provider;
mod registry;
pub mod security;
mod types;

pub use ai_agents_core::{Tool, ToolInfo, ToolResult};
pub use condition::{
    ConditionEvaluator, EvaluationContext, LLMGetter, SimpleLLMGetter, ToolCallRecord,
};
pub use provider::{ProviderHealth, ToolDescriptor, ToolProvider, ToolProviderError};
pub use registry::ToolRegistry;
pub use types::{ToolAliases, ToolContext, ToolMetadata, ToolProviderType, TrustLevel};

#[cfg(feature = "http-tool")]
pub use builtin::HttpTool;
pub use builtin::{
    CalculatorTool, DateTimeTool, EchoTool, FileTool, JsonTool, MathTool, RandomTool, TemplateTool,
    TextTool,
};

pub use security::{SecurityCheckResult, ToolPolicyConfig, ToolSecurityConfig, ToolSecurityEngine};

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
        .register(Arc::new(TextTool::new()))
        .expect("failed to register text");
    registry
        .register(Arc::new(TemplateTool::new()))
        .expect("failed to register template");
    registry
        .register(Arc::new(MathTool::new()))
        .expect("failed to register math");
    #[cfg(feature = "http-tool")]
    registry
        .register(Arc::new(HttpTool::new()))
        .expect("failed to register http");
    registry
}
