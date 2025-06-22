//! Tool configuration types

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    pub name: String,

    #[serde(flatten)]
    pub config: Option<Value>,
}

impl ToolConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: None,
        }
    }

    pub fn with_config(name: impl Into<String>, config: Value) -> Self {
        Self {
            name: name.into(),
            config: Some(config),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_config_new() {
        let config = ToolConfig::new("calculator");
        assert_eq!(config.name, "calculator");
        assert!(config.config.is_none());
    }

    #[test]
    fn test_tool_config_with_config() {
        let extra = json!({"param": "value"});
        let config = ToolConfig::with_config("custom_tool", extra.clone());
        assert_eq!(config.name, "custom_tool");
        assert_eq!(config.config, Some(extra));
    }

    #[test]
    fn test_tool_config_deserialize_simple() {
        let yaml = r#"
name: calculator
"#;
        let config: ToolConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "calculator");
        // Note: flatten may capture empty object instead of None
    }

    #[test]
    fn test_tool_config_deserialize_with_extra() {
        let yaml = r#"
name: web_search
api_key: "test_key"
timeout: 30
"#;
        let config: ToolConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "web_search");
        assert!(config.config.is_some());
    }

    #[test]
    fn test_tool_config_serialize() {
        let config = ToolConfig::new("echo");
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("name: echo"));
    }
}
