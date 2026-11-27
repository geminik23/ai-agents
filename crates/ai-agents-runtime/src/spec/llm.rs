//! LLM configuration types

use ai_agents_core::ToolChoice;
use ai_agents_llm::providers::ProviderType;
pub use ai_agents_llm::{RouterRolesConfig, RouterSelector};
use serde::{Deserialize, Deserializer, Serialize};
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

    /// An explicit temperature, or no temperature for a preset that requires omission.
    #[serde(
        default,
        deserialize_with = "deserialize_explicit",
        skip_serializing_if = "Option::is_none"
    )]
    pub temperature: Option<f32>,

    /// Explicit output ceiling. Existing YAML transports apply their historical default at construction.
    #[serde(
        default,
        deserialize_with = "deserialize_explicit",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_tokens: Option<u32>,

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

    /// Provider reasoning effort. The first-party Chat path preserves the supplied string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// Maximum token budget for reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_budget_tokens: Option<u32>,

    /// Override whether the provider supports function/tool calling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_calling: Option<bool>,

    /// Opt in to provider-native or runtime-enforced tool selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Override whether the provider supports vision inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,

    /// Override whether the provider supports JSON mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<bool>,

    /// Additional provider-specific configuration
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

fn default_temperature() -> f32 {
    0.7
}

// Explicit null remains invalid in YAML; only an absent field means omission.
fn deserialize_explicit<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    d: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(d).map(Some)
}

fn default_max_tokens() -> u32 {
    2000
}

impl Default for LLMConfig {
    // Keep declaration defaults independent of the model selected by a Rust struct update.
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "gpt-4".to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            base_url: None,
            api_key_env: None,
            timeout_seconds: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
            function_calling: None,
            tool_choice: None,
            vision: None,
            json_mode: None,
            extra: HashMap::new(),
        }
    }
}

impl LLMConfig {
    /// Resolves the historical output ceiling only on declarations using a legacy or Chat path.
    /// Native non-Chat presets preserve absence unless the caller declares a ceiling.
    pub fn effective_max_tokens(&self) -> Option<u32> {
        if self.uses_messages() || self.provider.parse() == Ok(ProviderType::OllamaCloud) {
            return self.max_tokens;
        }
        self.max_tokens
            .or_else(|| match self.extra.get("protocol") {
                None => Some(default_max_tokens()),
                Some(protocol) if protocol == "chat-completions" => Some(default_max_tokens()),
                _ => None,
            })
    }

    // Keep Messages declaration defaults independent of legacy provider defaults.
    fn uses_messages(&self) -> bool {
        self.provider.parse::<ProviderType>().is_ok_and(|provider| {
            provider.uses_messages(
                self.extra
                    .get("protocol")
                    .and_then(serde_json::Value::as_str),
            )
        })
    }

    // Only framework presets with a historical sampling default materialize it for execution.
    fn effective_temperature(&self, provider: ProviderType) -> Option<f32> {
        self.temperature.or_else(|| {
            (!matches!(provider, ProviderType::OpenAI | ProviderType::OllamaCloud)
                && !self.uses_messages())
            .then(default_temperature)
        })
    }

    // Local native options use declaration presence to validate overlaps with legacy fields.
    fn uses_local_native_options(&self, provider: ProviderType) -> bool {
        provider == ProviderType::Ollama && self.extra.contains_key("provider_options")
    }

    /// Resolves declaration defaults for runtime and eval while preserving Local Ollama option presence for conflict checks.
    pub fn to_provider_config(&self, provider: ProviderType) -> ai_agents_core::LLMConfig {
        // Legacy Local Ollama omitted these fields on the wire. Native options only conflict with explicitly set values.
        let (temperature, max_tokens) = if self.uses_local_native_options(provider) {
            (self.temperature, self.max_tokens)
        } else {
            (
                self.effective_temperature(provider),
                self.effective_max_tokens(),
            )
        };
        ai_agents_core::LLMConfig {
            temperature,
            max_tokens,
            top_p: self.top_p,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            timeout_seconds: self.timeout_seconds,
            reasoning: self.reasoning,
            reasoning_effort: self.reasoning_effort.clone(),
            reasoning_budget_tokens: self.reasoning_budget_tokens,
            extra: self.extra.clone(),
        }
    }
}

impl super::AgentSpec {
    // Compare effective declarations while preserving the historical serialized identity of equivalent preset defaults.
    // Persistence itself keeps the original presence so an OpenAI omission is distinct from an explicit temperature.
    pub(crate) fn identity_value(&self) -> ai_agents_core::Result<serde_json::Value> {
        let mut value = serde_json::to_value(self)?;
        let normalize = |config: &LLMConfig, target: &mut serde_json::Value| {
            if config
                .provider
                .parse()
                .is_ok_and(|provider| config.uses_local_native_options(provider))
            {
                return;
            }
            if let Some(limit) = config.effective_max_tokens() {
                target["max_tokens"] = serde_json::json!(limit);
            }
            if config.temperature.is_none()
                && let Ok(provider) = config.provider.parse()
                && let Some(temperature) = config.effective_temperature(provider)
            {
                target["temperature"] = serde_json::json!(temperature);
            }
        };
        if let Some(config) = self.llm.as_config() {
            if self.llms.is_empty() {
                normalize(config, &mut value["llm"]);
            } else {
                // A named registry supplies the provider. Preserve the historical numeric defaults of its inactive inline declaration.
                let inline = &mut value["llm"];
                inline["temperature"] =
                    serde_json::json!(config.temperature.unwrap_or_else(default_temperature));
                inline["max_tokens"] =
                    serde_json::json!(config.max_tokens.unwrap_or_else(default_max_tokens));
            }
        }
        for (alias, config) in &self.llms {
            normalize(config, &mut value["llms"][alias]);
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LLMSelector {
    #[serde(default = "default_alias")]
    pub default: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router: Option<RouterSelector>,
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

    /// Installs a hierarchy while retaining explicit opt-in even when it is empty.
    pub fn with_router_roles(mut self, config: RouterRolesConfig) -> Self {
        self.router = Some(RouterSelector::Hierarchical(Box::new(config)));
        self
    }

    /// Preserves the scalar legacy constructor while storing the new selector type.
    pub fn with_router(mut self, router: impl Into<String>) -> Self {
        self.router = Some(RouterSelector::Alias(router.into()));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A Cloud task round trip must not inject the legacy Ollama sampling and token defaults.
    #[test]
    fn ollama_cloud_omitted_config_and_identity_survive_yaml_round_trip() {
        let spec = super::super::AgentSpec::from_yaml_strict(
            "name: Cloud\nsystem_prompt: test\nllm: {provider: ollama-cloud, model: chosen}",
        )
        .unwrap();
        let before = spec.identity_value().unwrap();
        let loaded =
            super::super::AgentSpec::from_yaml_strict(&serde_yaml::to_string(&spec).unwrap())
                .unwrap();
        assert_eq!(before, loaded.identity_value().unwrap());
        assert!(before["llm"].get("temperature").is_none());
        assert!(before["llm"].get("max_tokens").is_none());
        let cfg = loaded
            .llm
            .as_config()
            .unwrap()
            .to_provider_config(ProviderType::OllamaCloud);
        assert!(cfg.temperature.is_none() && cfg.max_tokens.is_none());
        let legacy = LLMConfig {
            provider: "ollama".into(),
            ..Default::default()
        };
        assert_eq!(
            legacy.to_provider_config(ProviderType::Ollama).temperature,
            Some(0.7)
        );
        assert_eq!(legacy.effective_max_tokens(), Some(2000));
    }

    #[test]
    fn ollama_local_explicit_options_survive_save_without_changing_legacy_identity() {
        let parse = |fields: &str| {
            super::super::AgentSpec::from_yaml_strict(&format!(
                "name: Local\nsystem_prompt: test\nllm: {{provider: ollama, model: chosen{fields}}}",
            ))
            .unwrap()
        };
        let legacy_identity = parse("").identity_value().unwrap();
        assert_eq!(
            legacy_identity,
            parse(", temperature: 0.7, max_tokens: 2000")
                .identity_value()
                .unwrap(),
        );
        let native = parse(", provider_options: {options: {temperature: 0.2, num_predict: 8}}");
        let before = native.identity_value().unwrap();
        assert_ne!(before, legacy_identity);
        let duplicate = parse(
            ", temperature: 0.7, max_tokens: 2000, provider_options: {options: {temperature: 0.2, num_predict: 8}}",
        );
        assert_ne!(duplicate.identity_value().unwrap(), before);
        let saved = serde_yaml::to_string(&native).unwrap();
        let loaded = super::super::AgentSpec::from_yaml_strict(&saved).unwrap();
        assert_eq!(loaded.identity_value().unwrap(), before);
        let config = loaded
            .llm
            .as_config()
            .unwrap()
            .to_provider_config(ProviderType::Ollama);
        assert!(config.temperature.is_none() && config.max_tokens.is_none());
        assert_eq!(
            config.extra["provider_options"]["options"]["num_predict"],
            8
        );
    }

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
        assert_eq!(config.temperature, None);
        assert_eq!(config.effective_max_tokens(), Some(2000));
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
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(1000));
    }

    #[test]
    fn test_llm_config_tool_choice_deserialize() {
        let required: LLMConfig =
            serde_yaml::from_str("provider: openai\nmodel: gpt-5.1-mini\ntool_choice: required\n")
                .unwrap();
        assert_eq!(required.tool_choice, Some(ToolChoice::Required));
        assert!(!required.extra.contains_key("tool_choice"));

        let specific: LLMConfig = serde_yaml::from_str(
            "provider: openai\nmodel: gpt-5.1-mini\ntool_choice:\n  specific: random\n",
        )
        .unwrap();
        assert_eq!(
            specific.tool_choice,
            Some(ToolChoice::Specific("random".to_string()))
        );
    }

    #[test]
    fn test_llm_config_with_defaults() {
        let yaml = r#"
provider: openai
model: gpt-4
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            config.to_provider_config(ProviderType::OpenAI).temperature,
            None
        );
        assert_eq!(config.effective_max_tokens(), Some(2000)); // default
    }

    #[test]
    fn agent_spec_round_trips_gpt_6_2_omission_and_legacy_defaults() {
        let yaml = "name: Sampling\nsystem_prompt: Test\nllm: {provider: openai, model: gpt-6.2-astra}\nllms:\n  omitted: {provider: openai, model: gpt-6.2-sol}\n  explicit: {provider: openai, model: gpt-6.2-sol, temperature: 0.7}\n  compatible: {provider: openai-compatible, model: gpt-6.2-sol}\n";
        let spec = crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap();
        let yaml = serde_yaml::to_string(&spec).unwrap();
        let restored = crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
        assert_eq!(restored.llm.as_config().unwrap().temperature, None);
        assert_eq!(restored.llms["omitted"].temperature, None);
        assert_eq!(restored.llms["explicit"].temperature, Some(0.7));
        assert_eq!(restored.llms["compatible"].temperature, None);

        let raw: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
        assert!(raw["llm"].get("temperature").is_none());
        assert!(raw["llms"]["omitted"].get("temperature").is_none());
        assert!(raw["llms"]["compatible"].get("temperature").is_none());
        assert!(
            serde_yaml::from_str::<LLMConfig>(
                "provider: openai\nmodel: gpt-6.2-sol\ntemperature: null\n"
            )
            .is_err()
        );
    }

    #[test]
    fn rust_sampling_declaration_and_task_scope_survive_yaml() {
        use crate::autonomy::TaskHostScope;
        use crate::spec::{AgentSpec, LLMConfigOrSelector};
        let scope = |spec: &AgentSpec| {
            TaskHostScope::from_spec(spec, std::path::Path::new("."), None)
                .unwrap()
                .identity()
                .to_owned()
        };
        for provider in [ProviderType::OpenAI, ProviderType::OpenAICompatible] {
            let config = LLMConfig {
                provider: if provider == ProviderType::OpenAI {
                    "openai"
                } else {
                    "openai-compatible"
                }
                .into(),
                model: "gpt-6.2-sol".into(),
                ..Default::default()
            };
            assert_eq!(config.temperature, None);
            assert_eq!(
                config.to_provider_config(provider).temperature,
                (provider == ProviderType::OpenAICompatible).then_some(0.7)
            );
            for named in [false, true] {
                let declaration = |config: LLMConfig| {
                    let mut spec = AgentSpec::default();
                    if named {
                        spec.llm = LLMConfigOrSelector::Selector(LLMSelector::default());
                        spec.llms.insert("default".into(), config);
                    } else {
                        spec.llm = LLMConfigOrSelector::Config(config);
                    }
                    spec
                };
                let spec = declaration(config.clone());
                let loaded =
                    AgentSpec::from_yaml_strict(&serde_yaml::to_string(&spec).unwrap()).unwrap();
                assert_eq!(scope(&spec), scope(&loaded));
                assert_eq!(
                    serde_json::to_value(&spec).unwrap(),
                    serde_json::to_value(&loaded).unwrap()
                );
                let explicit = declaration(LLMConfig {
                    max_tokens: Some(default_max_tokens()),
                    temperature: Some(0.7),
                    ..config.clone()
                });
                if provider == ProviderType::OpenAICompatible {
                    // Legacy declarations serialized the numeric default directly into the namespace hash.
                    assert_eq!(
                        loaded.identity_value().unwrap(),
                        serde_json::to_value(&explicit).unwrap()
                    );
                }
                assert_eq!(
                    scope(&explicit) == scope(&loaded),
                    provider == ProviderType::OpenAICompatible
                );
            }
        }
    }

    #[test]
    fn responses_output_limit_preserves_omission_and_an_explicit_historical_default() {
        let source = "name: Test\nsystem_prompt: hi\nllm: {default: selected}\nllms:\n  selected:\n    provider: openai\n    model: model\n    protocol: responses\n";
        let parse = |s: &str| crate::spec::AgentSpec::from_yaml_strict(s).unwrap();
        let omitted = parse(source);
        let explicit = parse(&format!("{source}    max_tokens: 2000\n"));
        for (spec, limit) in [(&omitted, None), (&explicit, Some(2000))] {
            let loaded = parse(&serde_yaml::to_string(spec).unwrap());
            assert_eq!(
                loaded.llms["selected"]
                    .to_provider_config(ProviderType::OpenAI)
                    .max_tokens,
                limit
            );
            assert_eq!(
                spec.identity_value().unwrap(),
                loaded.identity_value().unwrap()
            );
        }
        assert_ne!(
            omitted.identity_value().unwrap(),
            explicit.identity_value().unwrap()
        );
    }

    #[test]
    fn named_registry_preserves_legacy_scope_with_an_inactive_responses_declaration() {
        use crate::autonomy::TaskHostScope;
        use crate::spec::AgentSpec;

        let source = "name: Test\nsystem_prompt: hello\nllm: {provider: openai, model: unused, protocol: responses}\nllms:\n  default: {provider: openai, model: active, protocol: responses}\n";
        let scope = |source: &str| {
            let spec = AgentSpec::from_yaml_strict(source).unwrap();
            TaskHostScope::from_spec(&spec, std::path::Path::new("."), None)
                .unwrap()
                .identity()
                .to_owned()
        };
        let original = scope(source);
        let legacy_inline = source.replace("model: unused,", "model: unused, max_tokens: 2000,");
        assert_eq!(original, scope(&legacy_inline));
        let chat = source.replace("model: active, protocol: responses", "model: active");
        let saved_chat = legacy_inline.replace(
            "model: active, protocol: responses",
            "model: active, max_tokens: 2000",
        );
        assert_eq!(scope(&chat), scope(&saved_chat));
        assert_ne!(original, scope(&legacy_inline.replace("2000", "1000")));
        assert_ne!(
            original,
            scope(&source.replace("model: active,", "model: active, max_tokens: 2000,"))
        );
    }

    #[test]
    fn messages_omits_sampling_and_requires_a_limit_without_changing_legacy_identity() {
        let source = "name: Test\nsystem_prompt: hello\nllm: {provider: anthropic, model: claude-sonnet-4-6}";
        let parse = |yaml: &str| crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap();
        let legacy = parse(source);
        let saved = parse(&source.replace('}', ", temperature: 0.7, max_tokens: 2000}"));
        assert_eq!(
            legacy.identity_value().unwrap(),
            saved.identity_value().unwrap()
        );
        for yaml in [
            source.replace('}', ", protocol: messages}"),
            source.replace("provider: anthropic,", "provider: anthropic-compatible,"),
        ] {
            let messages = parse(&yaml);
            let config = messages.llm.as_config().unwrap();
            let effective = config.to_provider_config(config.provider.parse().unwrap());
            assert_eq!(effective.temperature, None);
            assert_eq!(effective.max_tokens, None);
        }
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
        assert_eq!(
            selector.router,
            Some(RouterSelector::Alias("cheap".to_string()))
        );
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
        assert_eq!(
            selector.router,
            Some(RouterSelector::Alias("router_llm".to_string()))
        );
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

    #[test]
    fn test_ollama_named_fields_land_in_extra() {
        let yaml = r#"
provider: ollama
model: llama3.1
num_ctx: 8192
num_gpu: -1
keep_alive: 5m
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.extra.get("num_ctx"), Some(&serde_json::json!(8192)));
        assert_eq!(config.extra.get("num_gpu"), Some(&serde_json::json!(-1)));
        assert_eq!(
            config.extra.get("keep_alive"),
            Some(&serde_json::json!("5m"))
        );
    }

    #[test]
    fn test_llm_config_feature_override_fields_deser() {
        let yaml = r#"
provider: openai-compatible
model: qwen3:8b
base_url: http://localhost:11434/v1
function_calling: true
vision: false
json_mode: true
"#;
        let config: LLMConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.function_calling, Some(true));
        assert_eq!(config.vision, Some(false));
        assert_eq!(config.json_mode, Some(true));
        assert!(!config.extra.contains_key("function_calling"));
        assert!(!config.extra.contains_key("vision"));
        assert!(!config.extra.contains_key("json_mode"));
    }
}
