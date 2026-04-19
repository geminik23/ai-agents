//! LLM configuration types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CliMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub welcome: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hints: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_tools: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_state: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_timing: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_style: Option<CliPromptStyle>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_builtin_commands: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hitl: Option<CliHitlMetadata>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CliPromptStyle {
    Simple,
    WithState,
}

/// Controls how the CLI handles HITL approval requests at runtime.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CliHitlStyle {
    /// Interactive y/N prompt in the terminal (default).
    #[default]
    Prompt,
    /// Silently approve all requests.
    AutoApprove,
    /// Silently reject all requests.
    AutoReject,
}

/// CLI-specific HITL display settings from `metadata.cli.hitl`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CliHitlMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<CliHitlStyle>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_context: Option<bool>,
}

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

    /// Base URL for the LLM provider API.
    /// Required for `openai-compatible`; optional override for other providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Environment variable name containing the API key.
    /// Overrides the provider's default env var (e.g. OPENAI_API_KEY).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,

    /// Request timeout in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,

    /// Enable extended thinking / reasoning mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,

    /// Reasoning effort level: "low", "medium", or "high".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// Maximum token budget for reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_budget_tokens: Option<u32>,

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
            base_url: None,
            api_key_env: None,
            timeout_seconds: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
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
    fn test_cli_metadata_deserialize() {
        let yaml = r#"
welcome: "=== Demo ==="
hints:
  - "Try: hello"
  - "Try: help"
show_tools: true
show_state: false
show_timing: true
streaming: true
prompt_style: with_state
disable_builtin_commands: false
"#;
        let metadata: CliMetadata = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(metadata.welcome.as_deref(), Some("=== Demo ==="));
        assert_eq!(metadata.hints.len(), 2);
        assert_eq!(metadata.show_tools, Some(true));
        assert_eq!(metadata.show_state, Some(false));
        assert_eq!(metadata.show_timing, Some(true));
        assert_eq!(metadata.streaming, Some(true));
        assert_eq!(metadata.prompt_style, Some(CliPromptStyle::WithState));
        assert_eq!(metadata.disable_builtin_commands, Some(false));
        assert!(metadata.hitl.is_none());
    }

    #[test]
    fn test_llm_config_default() {
        let config = LLMConfig::default();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 2000);
        assert_eq!(config.base_url, None);
        assert_eq!(config.api_key_env, None);
    }

    #[test]
    fn test_llm_config_with_base_url() {
        let yaml = r#"
provider: openai-compatible
model: llama3.2
base_url: http://localhost:1234/v1
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.provider, "openai-compatible");
        assert_eq!(
            config.base_url,
            Some("http://localhost:1234/v1".to_string())
        );
    }

    #[test]
    fn test_llm_config_with_api_key_env() {
        let yaml = r#"
provider: openai-compatible
model: my-model
base_url: http://my-server:8080/v1
api_key_env: MY_SERVER_KEY
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.api_key_env, Some("MY_SERVER_KEY".to_string()));
    }

    #[test]
    fn test_llm_config_base_url_does_not_leak_to_extra() {
        let yaml = r#"
provider: openai-compatible
model: my-model
base_url: http://localhost:1234/v1
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!config.extra.contains_key("base_url"));
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
    fn test_cli_hitl_metadata_deserialize() {
        let yaml = r#"
style: auto_approve
show_context: false
"#;
        let meta: CliHitlMetadata = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(meta.style, Some(CliHitlStyle::AutoApprove));
        assert_eq!(meta.show_context, Some(false));
    }

    #[test]
    fn test_cli_hitl_style_default() {
        assert_eq!(CliHitlStyle::default(), CliHitlStyle::Prompt);
    }

    #[test]
    fn test_cli_metadata_with_hitl() {
        let yaml = r#"
welcome: "Hello"
hints: []
hitl:
  style: prompt
  show_context: true
"#;
        let meta: CliMetadata = serde_yaml::from_str(yaml).unwrap();
        let hitl = meta.hitl.unwrap();
        assert_eq!(hitl.style, Some(CliHitlStyle::Prompt));
        assert_eq!(hitl.show_context, Some(true));
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

    #[test]
    fn test_llm_config_reasoning_fields_deser() {
        let yaml = r#"
provider: openai
model: o3
timeout_seconds: 120
reasoning: true
reasoning_effort: high
reasoning_budget_tokens: 16384
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.timeout_seconds, Some(120));
        assert_eq!(config.reasoning, Some(true));
        assert_eq!(config.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(config.reasoning_budget_tokens, Some(16384));
        // Must NOT leak into extra
        assert!(!config.extra.contains_key("timeout_seconds"));
        assert!(!config.extra.contains_key("reasoning"));
        assert!(!config.extra.contains_key("reasoning_effort"));
        assert!(!config.extra.contains_key("reasoning_budget_tokens"));
    }
}
