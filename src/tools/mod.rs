use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

pub mod builtin;
mod registry;

pub use builtin::{CalculatorTool, EchoTool};
pub use registry::ToolRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            metadata: None,
        }
    }

    pub fn ok_with_metadata(output: impl Into<String>, metadata: HashMap<String, Value>) -> Self {
        Self {
            success: true,
            output: output.into(),
            metadata: Some(metadata),
        }
    }

    pub fn error(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: error.into(),
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool execution failed: {0}")]
    Execution(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Duplicate tool: {0}")]
    Duplicate(String),

    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;

    async fn execute(&self, args: Value) -> ToolResult;

    fn info(&self) -> ToolInfo {
        ToolInfo {
            id: self.id().to_string(),
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

pub fn generate_schema<T: JsonSchema>() -> Value {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_ok() {
        let result = ToolResult::ok("success");
        assert!(result.success);
        assert_eq!(result.output, "success");
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("failed");
        assert!(!result.success);
        assert_eq!(result.output, "failed");
    }

    #[test]
    fn test_create_builtin_registry() {
        let registry = create_builtin_registry();
        assert!(registry.get("calculator").is_some());
        assert!(registry.get("echo").is_some());
    }
}
