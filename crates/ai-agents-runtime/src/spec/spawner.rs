//! Spawner configuration types for YAML deserialization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::StorageConfig;

/// An agent to create at startup and register in the AgentRegistry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoSpawnEntry {
    /// Registry ID for this agent.
    pub id: String,
    /// Path to the agent YAML file (resolved relative to parent YAML directory).
    pub agent: String,
}

/// Orchestration tool selection: all tools or a specific subset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OrchestrationToolsConfig {
    /// `orchestration_tools: true` registers all orchestration tools.
    All(bool),
    /// `orchestration_tools: [route_to_agent, group_discussion]` registers listed tools.
    Selected(Vec<String>),
}

impl Default for OrchestrationToolsConfig {
    fn default() -> Self {
        Self::All(false)
    }
}

impl OrchestrationToolsConfig {
    /// Returns true if any orchestration tools are enabled.
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::All(v) => *v,
            Self::Selected(v) => !v.is_empty(),
        }
    }

    /// Returns true if the given tool name is included.
    pub fn includes(&self, tool_name: &str) -> bool {
        match self {
            Self::All(true) => true,
            Self::All(false) => false,
            Self::Selected(v) => v.iter().any(|t| t == tool_name),
        }
    }
}

/// Configuration for dynamic agent spawning declared in the `spawner:` YAML section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnerConfig {
    /// When true, spawned agents reuse the parent agent's LLM connections.
    #[serde(default)]
    pub shared_llms: bool,

    /// Shared storage backend for all spawned agents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_storage: Option<StorageConfig>,

    /// Context values injected into every spawned agent.
    #[serde(default)]
    pub shared_context: HashMap<String, serde_json::Value>,

    /// Maximum number of agents that can be spawned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_agents: Option<usize>,

    /// Auto-naming prefix for spawned agents (e.g. "npc_" -> "npc_001").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name_prefix: Option<String>,

    /// Named YAML templates -- inline strings or file path references.
    #[serde(default)]
    pub templates: HashMap<String, TemplateSource>,

    /// Tool names that spawned agents are allowed to use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,

    /// Agents to create at startup and register in the AgentRegistry.
    #[serde(default)]
    pub auto_spawn: Vec<AutoSpawnEntry>,

    /// Register orchestration tools (route_to_agent, group_discussion, etc.).
    #[serde(default)]
    pub orchestration_tools: OrchestrationToolsConfig,
}

/// A spawner template source: either an inline YAML string or a file path reference.
///
/// Untagged:
/// Serde tries `File` first (object with `path` key), falls back to `Inline` (plain string).
/// File paths are resolved against the parent YAML directory at config time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplateSource {
    /// File-based template: `{ path: "./templates/npc.yaml" }`.
    File { path: String },
    /// Inline YAML template string (backward compatible).
    Inline(String),
}

impl TemplateSource {
    /// Returns true if this is a file path reference.
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    /// Returns true if this is an inline template string.
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(_))
    }
}

impl Default for SpawnerConfig {
    fn default() -> Self {
        Self {
            shared_llms: false,
            shared_storage: None,
            shared_context: HashMap::new(),
            max_agents: None,
            name_prefix: None,
            templates: HashMap::new(),
            allowed_tools: None,
            auto_spawn: Vec::new(),
            orchestration_tools: OrchestrationToolsConfig::default(),
        }
    }
}

impl SpawnerConfig {
    /// Returns true if any spawner configuration is present.
    pub fn is_configured(&self) -> bool {
        self.shared_llms
            || self.shared_storage.is_some()
            || !self.shared_context.is_empty()
            || self.max_agents.is_some()
            || self.name_prefix.is_some()
            || !self.templates.is_empty()
            || !self.auto_spawn.is_empty()
            || self.orchestration_tools.is_enabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_not_configured() {
        let config = SpawnerConfig::default();
        assert!(!config.is_configured());
    }

    #[test]
    fn test_deserialize_inline_template() {
        let yaml = r#"
shared_llms: true
max_agents: 50
name_prefix: "npc_"
shared_context:
  world_name: "Medieval Fantasy"
  current_era: "Age of Dragons"
templates:
  npc_base: |
    name: "{{ name }}"
    system_prompt: "You are {{ name }}."
"#;
        let config: SpawnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.shared_llms);
        assert_eq!(config.max_agents, Some(50));
        assert_eq!(config.name_prefix.as_deref(), Some("npc_"));
        assert_eq!(config.shared_context.len(), 2);
        assert!(config.templates.contains_key("npc_base"));
        assert!(config.templates.get("npc_base").unwrap().is_inline());
        assert!(config.is_configured());
    }

    #[test]
    fn test_deserialize_auto_spawn() {
        let yaml = r#"
shared_llms: true
auto_spawn:
  - id: billing
    agent: agents/billing_agent.yaml
  - id: technical
    agent: agents/technical_agent.yaml
"#;
        let config: SpawnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.auto_spawn.len(), 2);
        assert_eq!(config.auto_spawn[0].id, "billing");
        assert_eq!(config.auto_spawn[0].agent, "agents/billing_agent.yaml");
    }

    #[test]
    fn test_deserialize_orchestration_tools_all() {
        let yaml = r#"
orchestration_tools: true
"#;
        let config: SpawnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.orchestration_tools.is_enabled());
        assert!(config.orchestration_tools.includes("route_to_agent"));
    }

    #[test]
    fn test_deserialize_orchestration_tools_selected() {
        let yaml = r#"
orchestration_tools:
  - route_to_agent
  - group_discussion
"#;
        let config: SpawnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.orchestration_tools.is_enabled());
        assert!(config.orchestration_tools.includes("route_to_agent"));
        assert!(config.orchestration_tools.includes("group_discussion"));
        assert!(!config.orchestration_tools.includes("concurrent_ask"));
    }

    #[test]
    fn test_auto_spawn_makes_configured() {
        let config = SpawnerConfig {
            auto_spawn: vec![AutoSpawnEntry {
                id: "test".to_string(),
                agent: "test.yaml".to_string(),
            }],
            ..SpawnerConfig::default()
        };
        assert!(config.is_configured());
    }

    #[test]
    fn test_deserialize_file_template() {
        let yaml = r#"
templates:
  npc_base:
    path: ./templates/npc_base.yaml
"#;
        let config: SpawnerConfig = serde_yaml::from_str(yaml).unwrap();
        match config.templates.get("npc_base") {
            Some(TemplateSource::File { path }) => {
                assert_eq!(path, "./templates/npc_base.yaml");
            }
            other => panic!("expected File variant, got {:?}", other),
        }
    }

    #[test]
    fn test_deserialize_mixed_templates() {
        let yaml = r#"
templates:
  inline_one: |
    name: "{{ name }}"
  file_one:
    path: ./templates/npc.yaml
  inline_two: "name: test"
"#;
        let config: SpawnerConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.templates.get("inline_one").unwrap().is_inline());
        assert!(config.templates.get("file_one").unwrap().is_file());
        assert!(config.templates.get("inline_two").unwrap().is_inline());
    }

    #[test]
    fn test_deserialize_absolute_path_template() {
        let yaml = r#"
templates:
  shared_guard:
    path: /opt/game/shared_templates/guard.yaml
"#;
        let config: SpawnerConfig = serde_yaml::from_str(yaml).unwrap();
        match config.templates.get("shared_guard") {
            Some(TemplateSource::File { path }) => {
                assert_eq!(path, "/opt/game/shared_templates/guard.yaml");
            }
            other => panic!("expected File variant, got {:?}", other),
        }
    }

    #[test]
    fn test_roundtrip_serde() {
        let config = SpawnerConfig {
            shared_llms: true,
            shared_storage: None,
            shared_context: HashMap::new(),
            max_agents: Some(100),
            name_prefix: Some("test_".to_string()),
            templates: HashMap::new(),
            allowed_tools: Some(vec!["echo".to_string()]),
            auto_spawn: Vec::new(),
            orchestration_tools: OrchestrationToolsConfig::default(),
        };
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: SpawnerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.max_agents, Some(100));
        assert_eq!(parsed.name_prefix.as_deref(), Some("test_"));
    }

    #[test]
    fn test_template_source_is_file() {
        let file = TemplateSource::File {
            path: "./test.yaml".to_string(),
        };
        let inline = TemplateSource::Inline("content".to_string());
        assert!(file.is_file());
        assert!(!file.is_inline());
        assert!(inline.is_inline());
        assert!(!inline.is_file());
    }
}
