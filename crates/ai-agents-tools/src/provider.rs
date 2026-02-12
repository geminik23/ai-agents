use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use super::types::{ToolAliases, ToolMetadata, ToolProviderType};
use super::{Tool, ToolResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub id: String,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aliases: Option<ToolAliases>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ToolMetadata>,
}

impl ToolDescriptor {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            input_schema,
            aliases: None,
            metadata: None,
        }
    }

    pub fn with_aliases(mut self, aliases: ToolAliases) -> Self {
        self.aliases = Some(aliases);
        self
    }

    pub fn with_metadata(mut self, metadata: ToolMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn get_name(&self, lang: Option<&str>) -> &str {
        if let Some(lang) = lang {
            if let Some(ref aliases) = self.aliases {
                if let Some(name) = aliases.get_name(lang) {
                    return name;
                }
            }
        }
        &self.name
    }

    pub fn get_description(&self, lang: Option<&str>) -> &str {
        if let Some(lang) = lang {
            if let Some(ref aliases) = self.aliases {
                if let Some(desc) = aliases.get_description(lang) {
                    return desc;
                }
            }
        }
        &self.description
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ProviderHealth {
    #[default]
    Healthy,
    Degraded {
        message: String,
    },
    Unavailable {
        message: String,
    },
}

impl ProviderHealth {
    pub fn is_healthy(&self) -> bool {
        matches!(self, ProviderHealth::Healthy)
    }

    pub fn is_available(&self) -> bool {
        !matches!(self, ProviderHealth::Unavailable { .. })
    }

    pub fn degraded(message: impl Into<String>) -> Self {
        ProviderHealth::Degraded {
            message: message.into(),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        ProviderHealth::Unavailable {
            message: message.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolProviderError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Provider unavailable: {0}")]
    Unavailable(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Timeout after {0}ms")]
    Timeout(u64),

    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait ToolProvider: Send + Sync {
    fn id(&self) -> &str;

    fn name(&self) -> &str;

    fn provider_type(&self) -> ToolProviderType;

    async fn list_tools(&self) -> Vec<ToolDescriptor>;

    async fn get_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>>;

    async fn execute(&self, tool_id: &str, args: Value) -> Result<ToolResult, ToolProviderError> {
        if let Some(tool) = self.get_tool(tool_id).await {
            Ok(tool.execute(args).await)
        } else {
            Err(ToolProviderError::ToolNotFound(tool_id.to_string()))
        }
    }

    fn supports_refresh(&self) -> bool {
        false
    }

    async fn refresh(&self) -> Result<(), ToolProviderError> {
        Ok(())
    }

    async fn health_check(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_descriptor() {
        let desc = ToolDescriptor::new(
            "search",
            "Web Search",
            "Search the web",
            serde_json::json!({"type": "object"}),
        );

        assert_eq!(desc.id, "search");
        assert_eq!(desc.get_name(None), "Web Search");
        assert_eq!(desc.get_description(None), "Search the web");
    }

    #[test]
    fn test_tool_descriptor_with_aliases() {
        let aliases = ToolAliases::new()
            .with_name("ko", "검색")
            .with_description("ko", "웹 검색");

        let desc = ToolDescriptor::new(
            "search",
            "Web Search",
            "Search the web",
            serde_json::json!({}),
        )
        .with_aliases(aliases);

        assert_eq!(desc.get_name(Some("ko")), "검색");
        assert_eq!(desc.get_name(Some("en")), "Web Search");
        assert_eq!(desc.get_description(Some("ko")), "웹 검색");
    }

    #[test]
    fn test_provider_health() {
        let healthy = ProviderHealth::Healthy;
        assert!(healthy.is_healthy());
        assert!(healthy.is_available());

        let degraded = ProviderHealth::degraded("Some tools failing");
        assert!(!degraded.is_healthy());
        assert!(degraded.is_available());

        let unavailable = ProviderHealth::unavailable("Connection lost");
        assert!(!unavailable.is_healthy());
        assert!(!unavailable.is_available());
    }
}
