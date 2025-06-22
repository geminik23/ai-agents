use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod builder;
mod runtime;

pub use builder::AgentBuilder;
pub use runtime::RuntimeAgent;

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl AgentInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: version.into(),
            description: None,
            capabilities: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl AgentResponse {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: None,
            tool_calls: None,
        }
    }

    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(calls);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[async_trait]
pub trait Agent: Send + Sync {
    async fn chat(&self, input: &str) -> Result<AgentResponse>;
    fn info(&self) -> AgentInfo;
    async fn reset(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, Role};

    #[test]
    fn test_chat_message_constructors() {
        let msg = ChatMessage::system("test");
        assert_eq!(msg.role, Role::System);
        assert!(msg.timestamp.is_some());

        let msg = ChatMessage::user("hello");
        assert_eq!(msg.role, Role::User);

        let msg = ChatMessage::assistant("hi");
        assert_eq!(msg.role, Role::Assistant);

        let msg = ChatMessage::tool("calc", "42");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.name, Some("calc".to_string()));
    }

    #[test]
    fn test_agent_info_builder() {
        let info = AgentInfo::new("test-id", "Test", "1.0")
            .with_description("A test agent")
            .with_capability("chat");

        assert_eq!(info.id, "test-id");
        assert_eq!(info.description, Some("A test agent".to_string()));
        assert!(info.capabilities.contains(&"chat".to_string()));
    }

    #[test]
    fn test_agent_response() {
        let response = AgentResponse::new("Hello!");
        assert_eq!(response.content, "Hello!");
        assert!(response.tool_calls.is_none());
    }
}
