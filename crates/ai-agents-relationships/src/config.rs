use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use ai_agents_core::{AgentError, Result};

use crate::defaults::{builtin_dimensions, default_dimension_names, fallback_dimension};
use crate::types::{RelationshipDimensionDefinition, RelationshipModel};

/// Top-level configuration for actor-scoped relationship memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipConfig {
    /// Enables relationship loading, prompt injection, evaluation, and persistence.
    #[serde(default)]
    pub enabled: bool,
    /// Selects one-sided or two-sided relationship semantics.
    #[serde(default)]
    pub model: RelationshipModel,
    /// Defines which relationship dimensions are tracked for each actor.
    #[serde(default)]
    pub dimensions: RelationshipDimensionsConfig,
    /// Controls automatic LLM-based relationship updates after successful turns.
    #[serde(default)]
    pub auto_update: AutoUpdateConfig,
    /// Controls context and prompt injection for the current actor relationship.
    #[serde(default)]
    pub injection: InjectionConfig,
    /// Controls whether relationship data is persisted through storage backends.
    #[serde(default)]
    pub persistence: PersistenceConfig,
    /// Controls compact notable event storage for relationship-relevant events.
    #[serde(default)]
    pub notable_events: NotableEventsConfig,
}

impl Default for RelationshipConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: RelationshipModel::OneSided,
            dimensions: RelationshipDimensionsConfig::default(),
            auto_update: AutoUpdateConfig::default(),
            injection: InjectionConfig::default(),
            persistence: PersistenceConfig::default(),
            notable_events: NotableEventsConfig::default(),
        }
    }
}

impl RelationshipConfig {
    /// Expand shorthand or explicit dimension config into the validated definition map used by the runtime and evaluator.
    pub fn dimension_definitions(
        &self,
    ) -> Result<HashMap<String, RelationshipDimensionDefinition>> {
        let builtins = builtin_dimensions();
        let definitions = match &self.dimensions {
            RelationshipDimensionsConfig::Shorthand(names) => {
                let names = if names.is_empty() {
                    default_dimension_names()
                } else {
                    names.clone()
                };
                names
                    .iter()
                    .map(|name| {
                        let def = builtins
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| fallback_dimension(name));
                        (name.clone(), def)
                    })
                    .collect()
            }
            RelationshipDimensionsConfig::Explicit(map) => {
                if map.is_empty() {
                    default_dimension_names()
                        .iter()
                        .map(|name| {
                            let def = builtins
                                .get(name)
                                .cloned()
                                .unwrap_or_else(|| fallback_dimension(name));
                            (name.clone(), def)
                        })
                        .collect()
                } else {
                    map.clone()
                }
            }
        };

        validate_definitions(&definitions)?;
        Ok(definitions)
    }
}

fn validate_definitions(
    definitions: &HashMap<String, RelationshipDimensionDefinition>,
) -> Result<()> {
    if definitions.is_empty() {
        return Err(AgentError::Config(
            "relationships.dimensions must contain at least one dimension".into(),
        ));
    }

    for (name, def) in definitions {
        if name.trim().is_empty() {
            return Err(AgentError::Config(
                "relationship dimension names cannot be empty".into(),
            ));
        }
        if def.min >= def.max {
            return Err(AgentError::Config(format!(
                "relationship dimension '{}' has invalid range: min must be less than max",
                name
            )));
        }
        if def.default < def.min || def.default > def.max {
            return Err(AgentError::Config(format!(
                "relationship dimension '{}' default is outside its range",
                name
            )));
        }
    }

    Ok(())
}

/// Dimension configuration accepted by the `memory.relationships.dimensions` YAML field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RelationshipDimensionsConfig {
    /// Shorthand list of dimension names using built-in definitions when available.
    Shorthand(Vec<String>),
    /// Explicit map of dimension names to full dimension definitions.
    Explicit(HashMap<String, RelationshipDimensionDefinition>),
}

impl Default for RelationshipDimensionsConfig {
    fn default() -> Self {
        RelationshipDimensionsConfig::Shorthand(default_dimension_names())
    }
}

/// Configuration for automatic relationship updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoUpdateConfig {
    /// Enables the relationship evaluator after successful turns.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional named LLM alias used for relationship evaluation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    /// Minimum evaluator confidence required before a proposed change is applied.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
    /// Maximum absolute dimension delta that can be applied from a single turn.
    #[serde(default = "default_max_delta_per_turn")]
    pub max_delta_per_turn: f64,
    /// Number of recent messages sent to the relationship evaluator.
    #[serde(default = "default_recent_messages")]
    pub recent_messages: usize,
}

impl Default for AutoUpdateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            llm: None,
            min_confidence: default_min_confidence(),
            max_delta_per_turn: default_max_delta_per_turn(),
            recent_messages: default_recent_messages(),
        }
    }
}

/// Configuration for relationship context and prompt injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// Enables relationship context and prompt variable injection.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Selects how the relationship is formatted for prompt injection.
    #[serde(default)]
    pub format: InjectionFormat,
    /// Maximum approximate token budget for formatted relationship prompt text.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Template variable name that receives formatted relationship prompt text.
    #[serde(default = "default_prompt_variable")]
    pub prompt_variable: String,
    /// Context path where the current actor relationship object is injected.
    #[serde(default = "default_context_path")]
    pub context_path: String,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            format: InjectionFormat::Summary,
            max_tokens: default_max_tokens(),
            prompt_variable: default_prompt_variable(),
            context_path: default_context_path(),
        }
    }
}

/// Prompt rendering format for relationship memory.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InjectionFormat {
    /// Natural language relationship summary.
    #[default]
    Summary,
    /// Detailed relationship data with dimensions, counts, timestamps, and recent events.
    Full,
    /// Compact comma-separated dimension scores.
    ScoresOnly,
}

/// Configuration for actor-keyed relationship persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Enables saving and loading relationship rows through configured storage.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Configuration for compact relationship-relevant event storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotableEventsConfig {
    /// Enables recording significant relationship events.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum number of notable events retained per actor.
    #[serde(default = "default_max_events")]
    pub max_per_actor: usize,
    /// Minimum event significance required before an event is stored.
    #[serde(default = "default_significance_threshold")]
    pub significance_threshold: f64,
    /// Strategy used when event count exceeds `max_per_actor`.
    #[serde(default)]
    pub eviction: EventEvictionStrategy,
}

impl Default for NotableEventsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_per_actor: default_max_events(),
            significance_threshold: default_significance_threshold(),
            eviction: EventEvictionStrategy::LowestSignificanceThenOldest,
        }
    }
}

/// Strategy for evicting old relationship events when a per-actor limit is exceeded.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventEvictionStrategy {
    /// Remove oldest events first.
    Oldest,
    /// Keep higher-significance events first, then older events as a tie breaker.
    #[default]
    LowestSignificanceThenOldest,
}

fn default_true() -> bool {
    true
}

fn default_min_confidence() -> f64 {
    0.6
}

fn default_max_delta_per_turn() -> f64 {
    0.3
}

fn default_recent_messages() -> usize {
    6
}

fn default_max_tokens() -> usize {
    400
}

fn default_prompt_variable() -> String {
    "relationship_memory".to_string()
}

fn default_context_path() -> String {
    "relationships.current_actor".to_string()
}

fn default_max_events() -> usize {
    50
}

fn default_significance_threshold() -> f64 {
    0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_relationship_config_has_dimensions() {
        let config = RelationshipConfig::default();
        assert_eq!(config.model, RelationshipModel::OneSided);
        let defs = config.dimension_definitions().unwrap();
        assert!(defs.contains_key("trust"));
        assert!(defs.contains_key("sentiment"));
        assert!(defs.contains_key("familiarity"));
        assert!(defs.contains_key("rapport"));
    }

    #[test]
    fn test_parse_shorthand_dimensions() {
        let yaml = r#"
enabled: true
dimensions:
  - trust
  - suspicion
"#;
        let config: RelationshipConfig = serde_yaml::from_str(yaml).unwrap();
        let defs = config.dimension_definitions().unwrap();
        assert!(defs.contains_key("trust"));
        assert!(defs.contains_key("suspicion"));
    }

    #[test]
    fn test_parse_explicit_dimensions() {
        let yaml = r#"
enabled: true
dimensions:
  motivation:
    description: "How motivated the actor seems"
    min: 0.0
    max: 1.0
    default: 0.5
"#;
        let config: RelationshipConfig = serde_yaml::from_str(yaml).unwrap();
        let defs = config.dimension_definitions().unwrap();
        assert_eq!(defs["motivation"].default, 0.5);
    }

    #[test]
    fn test_parse_two_sided_model() {
        let yaml = r#"
enabled: true
model: two_sided
dimensions:
  - trust
"#;
        let config: RelationshipConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model, RelationshipModel::TwoSided);
    }

    #[test]
    fn test_invalid_range_rejected() {
        let yaml = r#"
enabled: true
dimensions:
  broken:
    description: "bad"
    min: 1.0
    max: 0.0
    default: 0.0
"#;
        let config: RelationshipConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.dimension_definitions().is_err());
    }
}
