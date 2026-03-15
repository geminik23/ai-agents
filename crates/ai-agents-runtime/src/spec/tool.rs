//! Tool configuration types for agent specification.
//!
//! `ToolEntry` supports plain strings, structured builtins, and MCP wrapper entries.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use ai_agents_tools::mcp::wrapper::MCPWrapperConfig;

/// A single entry in the agent-level `tools:` list.
///
/// Supports three YAML forms:
/// 1. Plain string:         `- datetime`
/// 2. Builtin with config:  `- name: http`
/// 3. MCP wrapper:          `- name: github\n  type: mcp\n  transport: stdio\n  ...`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolEntry {
    /// Plain string tool reference (e.g., `- datetime`).
    Simple(String),
    /// Structured entry — builtin with extra config or MCP wrapper.
    Structured(StructuredToolEntry),
}

/// A structured tool entry with a name, optional type, and extra configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredToolEntry {
    pub name: String,

    /// `"mcp"` for MCP wrapper tools, absent/null for builtin tools.
    #[serde(rename = "type", default)]
    pub tool_type: Option<String>,

    /// Extra fields — flattened from YAML. For builtins, holds additional config.
    /// For MCP tools, holds transport/env/security/etc.
    #[serde(flatten)]
    pub extra: Value,
}

impl ToolEntry {
    /// Get the tool name regardless of entry form.
    pub fn name(&self) -> &str {
        match self {
            ToolEntry::Simple(name) => name,
            ToolEntry::Structured(s) => &s.name,
        }
    }

    /// Check if this entry is an MCP wrapper tool.
    pub fn is_mcp(&self) -> bool {
        match self {
            ToolEntry::Simple(_) => false,
            ToolEntry::Structured(s) => s.tool_type.as_deref() == Some("mcp"),
        }
    }

    /// Extract MCP wrapper config from a structured MCP entry.
    ///
    /// Re-serializes the structured entry to JSON, then deserializes as
    /// `MCPWrapperConfig` to pick up all flattened MCP fields.
    pub fn to_mcp_config(&self) -> Option<MCPWrapperConfig> {
        if !self.is_mcp() {
            return None;
        }
        match self {
            ToolEntry::Structured(s) => {
                let value = serde_json::to_value(s).ok()?;
                serde_json::from_value(value).ok()
            }
            _ => None,
        }
    }
}

/// Backward compatibility alias — existing code using `ToolConfig` keeps working.
pub type ToolConfig = ToolEntry;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_entry_plain_string() {
        let yaml = "datetime";
        let entry: ToolEntry = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(entry.name(), "datetime");
        assert!(!entry.is_mcp());
    }

    #[test]
    fn test_tool_entry_structured_builtin() {
        let yaml = "name: http";
        let entry: ToolEntry = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(entry.name(), "http");
        assert!(!entry.is_mcp());
    }

    #[test]
    fn test_tool_entry_mcp() {
        let yaml = r#"
name: github
type: mcp
transport: stdio
command: npx
args: ["-y", "@modelcontextprotocol/server-github"]
env:
  GITHUB_TOKEN: "test"
"#;
        let entry: ToolEntry = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(entry.name(), "github");
        assert!(entry.is_mcp());
        let config = entry.to_mcp_config().unwrap();
        assert_eq!(config.name, "github");
    }

    #[test]
    fn test_tool_entry_mixed_list() {
        let yaml = r#"
- datetime
- name: http
- name: github
  type: mcp
  transport: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_TOKEN: "test"
"#;
        let entries: Vec<ToolEntry> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name(), "datetime");
        assert!(!entries[0].is_mcp());
        assert_eq!(entries[1].name(), "http");
        assert!(!entries[1].is_mcp());
        assert_eq!(entries[2].name(), "github");
        assert!(entries[2].is_mcp());
    }

    #[test]
    fn test_tool_config_backward_compat() {
        let config = ToolConfig::Simple("echo".to_string());
        assert_eq!(config.name(), "echo");
    }

    #[test]
    fn test_tool_entry_name_method() {
        let simple = ToolEntry::Simple("calculator".to_string());
        assert_eq!(simple.name(), "calculator");

        let structured = ToolEntry::Structured(StructuredToolEntry {
            name: "custom_tool".to_string(),
            tool_type: None,
            extra: serde_json::json!({}),
        });
        assert_eq!(structured.name(), "custom_tool");
    }
}
