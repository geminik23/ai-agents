use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use super::{Tool, ToolError, ToolInfo};

pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolError> {
        let mut tools = self.tools.write();
        let id = tool.id().to_string();

        if tools.contains_key(&id) {
            return Err(ToolError::Duplicate(id));
        }

        tools.insert(id, tool);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().get(id).cloned()
    }

    pub fn list_ids(&self) -> Vec<String> {
        self.tools.read().keys().cloned().collect()
    }

    pub fn list_infos(&self) -> Vec<ToolInfo> {
        self.tools.read().values().map(|tool| tool.info()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.read().is_empty()
    }

    /// Generate tools prompt for ALL registered tools
    pub fn generate_tools_prompt(&self) -> String {
        let tools = self.tools.read();
        if tools.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("Available tools:\n");

        for tool in tools.values() {
            let schema = tool.input_schema();
            let args_desc = if let Some(props) = schema.get("properties") {
                serde_json::to_string(props).unwrap_or_default()
            } else {
                "{}".to_string()
            };

            prompt.push_str(&format!(
                "- {}: {}. Arguments: {}\n",
                tool.id(),
                tool.description(),
                args_desc
            ));
        }

        prompt.push_str(
            "\nWhen you need to use a tool, respond ONLY with valid JSON in this exact format:\n",
        );
        prompt.push_str("{\"tool\": \"tool_name\", \"arguments\": {...}}\n");
        prompt.push_str("\nWhen you receive a tool result, summarize it naturally for the user.\n");
        prompt.push_str("If no tool is needed, respond normally.");

        prompt
    }

    /// Generate tools prompt for specific tool IDs only (for state-specific filtering)
    /// If tool_ids is empty, returns prompt for all tools (same as generate_tools_prompt)
    pub fn generate_filtered_prompt(&self, tool_ids: &[String]) -> String {
        if tool_ids.is_empty() {
            return self.generate_tools_prompt();
        }

        let tools = self.tools.read();
        let mut prompt = String::from("Available tools:\n");
        let mut found_any = false;

        for id in tool_ids {
            if let Some(tool) = tools.get(id) {
                found_any = true;
                let schema = tool.input_schema();
                let args_desc = if let Some(props) = schema.get("properties") {
                    serde_json::to_string(props).unwrap_or_default()
                } else {
                    "{}".to_string()
                };

                prompt.push_str(&format!(
                    "- {}: {}. Arguments: {}\n",
                    tool.id(),
                    tool.description(),
                    args_desc
                ));
            }
        }

        if !found_any {
            return String::new();
        }

        prompt.push_str(
            "\nWhen you need to use a tool, respond ONLY with valid JSON in this exact format:\n",
        );
        prompt.push_str("{\"tool\": \"tool_name\", \"arguments\": {...}}\n");
        prompt.push_str("\nWhen you receive a tool result, summarize it naturally for the user.\n");
        prompt.push_str("If no tool is needed, respond normally.");

        prompt
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolResult;
    use async_trait::async_trait;
    use serde_json::Value;

    struct TestTool {
        id: String,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn id(&self) -> &str {
            &self.id
        }
        fn name(&self) -> &str {
            "Test"
        }
        fn description(&self) -> &str {
            "A test tool"
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _args: Value) -> ToolResult {
            ToolResult::ok("test")
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ToolRegistry::new();
        let tool = Arc::new(TestTool {
            id: "test".to_string(),
        });

        registry.register(tool).unwrap();
        assert!(registry.get("test").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_duplicate_registration() {
        let mut registry = ToolRegistry::new();
        let tool1 = Arc::new(TestTool {
            id: "test".to_string(),
        });
        let tool2 = Arc::new(TestTool {
            id: "test".to_string(),
        });

        registry.register(tool1).unwrap();
        assert!(registry.register(tool2).is_err());
    }

    #[test]
    fn test_list_ids() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "b".to_string(),
            }))
            .unwrap();

        let ids = registry.list_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
    }

    #[test]
    fn test_generate_tools_prompt() {
        // Test empty registry
        let empty_registry = ToolRegistry::new();
        let empty_prompt = empty_registry.generate_tools_prompt();
        assert!(empty_prompt.is_empty());

        // Test with tools
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "test".to_string(),
            }))
            .unwrap();

        let prompt = registry.generate_tools_prompt();
        assert!(prompt.contains("Available tools:"));
        assert!(prompt.contains("test:"));
        assert!(prompt.contains("A test tool"));
        assert!(prompt.contains("tool_name"));
    }

    #[test]
    fn test_generate_filtered_prompt_with_filter() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_b".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_c".to_string(),
            }))
            .unwrap();

        // Filter to only tool_a and tool_c
        let prompt =
            registry.generate_filtered_prompt(&["tool_a".to_string(), "tool_c".to_string()]);

        assert!(prompt.contains("tool_a"));
        assert!(!prompt.contains("tool_b"));
        assert!(prompt.contains("tool_c"));
    }

    #[test]
    fn test_generate_filtered_prompt_empty_filter() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();
        registry
            .register(Arc::new(TestTool {
                id: "tool_b".to_string(),
            }))
            .unwrap();

        // Empty filter returns all tools
        let prompt = registry.generate_filtered_prompt(&[]);
        assert!(prompt.contains("tool_a"));
        assert!(prompt.contains("tool_b"));
    }

    #[test]
    fn test_generate_filtered_prompt_nonexistent_tools() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TestTool {
                id: "tool_a".to_string(),
            }))
            .unwrap();

        // Filter with nonexistent tool returns empty
        let prompt = registry.generate_filtered_prompt(&["nonexistent".to_string()]);
        assert!(prompt.is_empty());

        // Mix of existing and nonexistent
        let prompt2 =
            registry.generate_filtered_prompt(&["tool_a".to_string(), "nonexistent".to_string()]);
        assert!(prompt2.contains("tool_a"));
        assert!(!prompt2.contains("nonexistent"));
    }
}
