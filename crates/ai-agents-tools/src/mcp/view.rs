//! MCP view tool — a filtered projection of an MCPWrapperTool.
//!
//! Each view exposes a named subset of the parent's functions as an independent
//! `Tool` with its own registry entry. All views share the parent's MCP connection.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use ai_agents_core::{Tool, ToolResult};

use super::wrapper::MCPWrapperTool;

/// A filtered projection of an `MCPWrapperTool` that exposes a named subset
/// of the parent's functions as an independent `Tool` in the registry.
pub struct MCPViewTool {
    view_name: String,
    parent: Arc<MCPWrapperTool>,
    allowed_functions: Vec<String>,
    description: String,
    schema: Value,
}

impl MCPViewTool {
    /// Create a new view over `parent`, exposing only `allowed_functions`.
    ///
    /// Returns `Err` if `allowed_functions` is empty or contains names that
    /// do not exist in the parent's discovered function set.
    pub fn new(
        view_name: String,
        parent: Arc<MCPWrapperTool>,
        allowed_functions: Vec<String>,
        custom_description: Option<String>,
    ) -> Result<Self, String> {
        if allowed_functions.is_empty() {
            return Err(format!(
                "View '{}': functions list cannot be empty",
                view_name,
            ));
        }

        // Validate that every requested function actually exists on the parent.
        let parent_names = parent.function_names();
        let mut missing: Vec<&str> = Vec::new();
        for name in &allowed_functions {
            if !parent_names.iter().any(|p| p == name) {
                missing.push(name);
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "View '{}': unknown functions {:?}. Available from '{}': {:?}",
                view_name,
                missing,
                parent.id(),
                parent_names,
            ));
        }

        let discovered = parent.get_functions_filtered(&allowed_functions);
        let schema = MCPWrapperTool::build_schema(&view_name, &discovered);
        let description = MCPWrapperTool::build_description(
            &view_name,
            custom_description.as_deref(),
            &discovered,
        );

        Ok(Self {
            view_name,
            parent,
            allowed_functions,
            description,
            schema,
        })
    }

    /// Check whether a function name is in this view's allowed set.
    fn is_allowed(&self, function: &str) -> bool {
        self.allowed_functions.iter().any(|f| f == function)
    }
}

#[async_trait]
impl Tool for MCPViewTool {
    /// Returns the view name as the tool ID.
    fn id(&self) -> &str {
        &self.view_name
    }

    /// Returns the view name as the display name.
    fn name(&self) -> &str {
        &self.view_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        // Extract the `function` field from input.
        let function = match args.get("function").and_then(|v| v.as_str()) {
            Some(f) => f.to_string(),
            None => {
                return ToolResult::error(format!(
                    "'function' is required. Available functions: {}",
                    self.allowed_functions.join(", ")
                ));
            }
        };

        // Validate the function is within this view's allowed set.
        if !self.is_allowed(&function) {
            return ToolResult::error(format!(
                "Function '{}' is not available in view '{}'. Available functions: {}",
                function,
                self.view_name,
                self.allowed_functions.join(", ")
            ));
        }

        // Extract optional `params` field (defaults to empty object).
        let params = args
            .get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        // Per-function HITL: check via the parent and return metadata if needed.
        if self.parent.requires_hitl(&function) {
            return ToolResult::ok_with_metadata(
                format!(
                    "Function '{}' on view '{}' requires approval before execution.",
                    function, self.view_name
                ),
                HashMap::from([
                    ("_hitl_required".to_string(), serde_json::json!(true)),
                    ("_hitl_function".to_string(), serde_json::json!(function)),
                    ("_hitl_params".to_string(), params.clone()),
                    ("_hitl_tool".to_string(), serde_json::json!(self.view_name)),
                ]),
            );
        }

        self.parent.call_function(&function, params).await
    }
}
