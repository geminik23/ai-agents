//! LLM configuration types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for LLM provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub provider: String,

    pub model: String,

    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Additional provider-specific configuration
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_max_tokens() -> u32 {
    2000
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "gpt-4".to_string(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            top_p: None,
            extra: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMSelector {
    #[serde(default = "default_alias")]
    pub default: String,
    #[serde(default)]
    pub router: Option<String>,
}

fn default_alias() -> String {
    "default".to_string()
}

impl Default for LLMSelector {
    fn default() -> Self {
        Self {
            default: default_alias(),
            router: None,
        }
    }
}

impl LLMSelector {
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            default: default.into(),
            router: None,
        }
    }

    pub fn with_router(mut self, router: impl Into<String>) -> Self {
        self.router = Some(router.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_default() {
        let config = LLMConfig::default();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 2000);
    }

    #[test]
    fn test_llm_config_deserialize() {
        let yaml = r#"
provider: openai
model: gpt-3.5-turbo
temperature: 0.5
max_tokens: 1000
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-3.5-turbo");
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 1000);
    }

    #[test]
    fn test_llm_config_with_defaults() {
        let yaml = r#"
provider: openai
model: gpt-4
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.temperature, 0.7); // default
        assert_eq!(config.max_tokens, 2000); // default
    }

    #[test]
    fn test_llm_config_extra_fields() {
        let yaml = r#"
provider: openai
model: gpt-4
custom_field: "value"
another_field: 123
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.extra.contains_key("custom_field"));
        assert!(config.extra.contains_key("another_field"));
    }

    #[test]
    fn test_llm_selector_default() {
        let selector = LLMSelector::default();
        assert_eq!(selector.default, "default");
        assert!(selector.router.is_none());
    }

    #[test]
    fn test_llm_selector_with_router() {
        let selector = LLMSelector::new("main").with_router("cheap");
        assert_eq!(selector.default, "main");
        assert_eq!(selector.router, Some("cheap".to_string()));
    }

    #[test]
    fn test_llm_selector_deserialize() {
        let yaml = r#"
default: main
router: router_llm
"#;
        let selector: LLMSelector = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(selector.default, "main");
        assert_eq!(selector.router, Some("router_llm".to_string()));
    }
}
