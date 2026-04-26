//! Configuration types for session management and key facts extraction.

use serde::{Deserialize, Serialize};

/// Config for `memory.actor_memory:` YAML block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActorMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub identification: IdentificationConfig,
    #[serde(default)]
    pub injection: InjectionConfig,
    #[serde(default)]
    pub privacy: PrivacyConfig,
}

/// How to resolve the current actor ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationConfig {
    /// explicit = set via set_actor_id API. from_context = read from context path.
    #[serde(default = "default_explicit")]
    pub method: IdentificationMethod,
    /// Context path to read actor_id from (when method = from_context).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IdentificationMethod {
    #[serde(rename = "explicit")]
    Explicit,
    #[serde(rename = "from_context")]
    FromContext,
}

fn default_explicit() -> IdentificationMethod {
    IdentificationMethod::Explicit
}

impl Default for IdentificationConfig {
    fn default() -> Self {
        Self {
            method: IdentificationMethod::Explicit,
            context_path: None,
        }
    }
}

/// How facts are injected into the prompt context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// all = inject all facts. category = inject only specified categories. on_demand = no auto-inject.
    #[serde(default = "default_all")]
    pub mode: InjectionMode,
    /// Max tokens for injected facts.
    #[serde(default = "default_injection_tokens")]
    pub max_tokens: usize,
    /// Categories to inject (when mode = category).
    #[serde(default)]
    pub categories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InjectionMode {
    #[serde(rename = "all")]
    All,
    #[serde(rename = "category")]
    Category,
    #[serde(rename = "on_demand")]
    OnDemand,
}

fn default_all() -> InjectionMode {
    InjectionMode::All
}

fn default_injection_tokens() -> usize {
    800
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            mode: InjectionMode::All,
            max_tokens: 800,
            categories: vec![],
        }
    }
}

/// Privacy and data retention settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Number of days to retain actor facts. None = no expiry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_days: Option<u32>,
    /// Whether actors can request deletion of all their data.
    #[serde(default = "default_true")]
    pub allow_deletion: bool,
}

fn default_true() -> bool {
    true
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            retention_days: None,
            allow_deletion: true,
        }
    }
}

/// Config for `memory.facts:` YAML block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extractor_llm: Option<String>,
    #[serde(default = "default_true")]
    pub auto_extract: bool,
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub custom_categories: Vec<CategoryDefinition>,
    #[serde(default = "default_true")]
    pub inject_in_context: bool,
    #[serde(default = "default_max_facts")]
    pub max_facts: usize,
    #[serde(default)]
    pub dedup: DedupConfig,
    /// Custom extraction prompt. None = use default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction_prompt: Option<String>,
}

fn default_max_facts() -> usize {
    50
}

impl Default for FactsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            extractor_llm: None,
            auto_extract: true,
            categories: vec![],
            custom_categories: vec![],
            inject_in_context: true,
            max_facts: 50,
            dedup: DedupConfig::default(),
            extraction_prompt: None,
        }
    }
}

/// User-defined fact category with a description for the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryDefinition {
    pub name: String,
    pub description: String,
}

/// Deduplication strategy for extracted facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// exact = normalized string match. llm = LLM-based semantic dedup.
    #[serde(default = "default_dedup_method")]
    pub method: DedupMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DedupMethod {
    #[serde(rename = "llm")]
    Llm,
    #[serde(rename = "exact")]
    Exact,
}

fn default_dedup_method() -> DedupMethod {
    DedupMethod::Exact
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: DedupMethod::Exact,
        }
    }
}

/// Config for `memory.session:` YAML block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_memory_config_default() {
        let config = ActorMemoryConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.identification.method, IdentificationMethod::Explicit);
        assert_eq!(config.injection.mode, InjectionMode::All);
        assert_eq!(config.injection.max_tokens, 800);
        assert!(config.privacy.allow_deletion);
    }

    #[test]
    fn test_facts_config_default() {
        let config = FactsConfig::default();
        assert!(!config.enabled);
        assert!(config.auto_extract);
        assert!(config.inject_in_context);
        assert_eq!(config.max_facts, 50);
        assert!(config.dedup.enabled);
        assert_eq!(config.dedup.method, DedupMethod::Exact);
    }

    #[test]
    fn test_facts_config_deserialize() {
        let yaml = r#"
enabled: true
extractor_llm: router
auto_extract: true
categories:
  - user_preference
  - user_context
  - decision
custom_categories:
  - name: suspicion
    description: "Suspicious behavior observed"
inject_in_context: true
max_facts: 30
dedup:
  enabled: true
  method: llm
"#;
        let config: FactsConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.extractor_llm.as_deref(), Some("router"));
        assert_eq!(config.categories.len(), 3);
        assert_eq!(config.custom_categories.len(), 1);
        assert_eq!(config.custom_categories[0].name, "suspicion");
        assert_eq!(config.max_facts, 30);
        assert_eq!(config.dedup.method, DedupMethod::Llm);
    }

    #[test]
    fn test_actor_memory_config_deserialize() {
        let yaml = r#"
enabled: true
identification:
  method: from_context
  context_path: user.id
injection:
  mode: category
  max_tokens: 500
  categories:
    - user_preference
privacy:
  retention_days: 365
  allow_deletion: true
"#;
        let config: ActorMemoryConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
        assert_eq!(
            config.identification.method,
            IdentificationMethod::FromContext
        );
        assert_eq!(
            config.identification.context_path.as_deref(),
            Some("user.id")
        );
        assert_eq!(config.injection.mode, InjectionMode::Category);
        assert_eq!(config.injection.max_tokens, 500);
        assert_eq!(config.injection.categories, vec!["user_preference"]);
        assert_eq!(config.privacy.retention_days, Some(365));
    }

    #[test]
    fn test_session_config_deserialize() {
        let yaml = r#"
tags: [support, tier-1]
ttl_seconds: 86400
"#;
        let config: SessionConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.tags, vec!["support", "tier-1"]);
        assert_eq!(config.ttl_seconds, Some(86400));
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert!(config.tags.is_empty());
        assert!(config.ttl_seconds.is_none());
    }
}
