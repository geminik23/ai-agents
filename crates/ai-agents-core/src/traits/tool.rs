//! Tool trait for external capabilities

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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

/// Core tool trait for external capabilities.
///
/// Implement this to add custom tools that the agent can invoke during conversation.
/// Built-in tools use `generate_schema::<T>()` from `ai-agents-tools` with
/// `schemars::JsonSchema` to derive input schemas automatically.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique identifier for this tool (e.g. `"calculator"`).
    fn id(&self) -> &str;
    /// Human-readable display name.
    fn name(&self) -> &str;
    /// Description shown to the LLM for tool selection.
    fn description(&self) -> &str;
    /// JSON Schema describing expected input arguments.
    fn input_schema(&self) -> Value;

    /// Execute the tool with the given arguments and return a result.
    async fn execute(&self, args: Value) -> ToolResult;

    /// Returns a [`ToolInfo`] struct from the above methods.
    fn info(&self) -> ToolInfo {
        ToolInfo {
            id: self.id().to_string(),
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}
