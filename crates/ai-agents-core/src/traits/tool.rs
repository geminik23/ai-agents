//! Tool trait for external capabilities

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::Result;
use crate::types::{
    ToolCallClassification, ToolExecutionContext, ToolExecutionRecord, ToolExecutionRequest,
    ToolPolicyBindings, ToolSafetyMetadata,
};

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

/// Public descriptor for a registered tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Canonical tool ID used by YAML, policy, HITL, and eval.
    pub id: String,
    /// Display name shown to the model.
    pub name: String,
    /// Description shown to the model for tool selection.
    pub description: String,
    /// JSON schema for tool arguments.
    pub input_schema: Value,
    /// Safety metadata used by the runtime executor.
    #[serde(default)]
    pub safety: ToolSafetyMetadata,
    /// Policy bindings declared by the tool.
    #[serde(default)]
    pub policy_bindings: ToolPolicyBindings,
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

    /// Execute the tool with arguments and executor context.
    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult;

    /// Policy bindings used by the shared executor to apply configured policy.
    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings::default()
    }

    /// Safety metadata used by runtime policy, scheduling, and observability.
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata::conservative_unknown()
    }

    /// Classify a specific call when risk depends on arguments.
    fn classify_call(&self, _args: &Value) -> ToolCallClassification {
        ToolCallClassification::from_metadata(&self.safety_metadata())
    }

    /// Returns a [`ToolInfo`] struct from the above methods.
    fn info(&self) -> ToolInfo {
        ToolInfo {
            id: self.id().to_string(),
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
            safety: self.safety_metadata(),
            policy_bindings: self.policy_bindings(),
        }
    }
}

/// Runtime boundary for invoking tools through shared policy and evidence handling.
#[async_trait]
pub trait ToolInvoker: Send + Sync {
    /// Invokes a tool request and returns a structured execution record.
    async fn invoke_tool(&self, request: ToolExecutionRequest) -> Result<ToolExecutionRecord>;
}
