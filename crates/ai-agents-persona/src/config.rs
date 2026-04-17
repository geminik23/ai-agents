//! Persona configuration types parsed from YAML.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use ai_agents_state::ContextMatcher;

/// Top-level struct for the `persona:` YAML block.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersonaConfig {
    /// Core identity - name, role, backstory.
    /// Optional at parse time for template inheritance. Must be present after resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<PersonaIdentity>,

    /// Structured personality traits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traits: Option<PersonaTraits>,

    /// Agent goals (primary and hidden).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goals: Option<PersonaGoals>,

    /// Information the agent withholds until conditions are met.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secrets: Option<Vec<PersonaSecret>>,

    /// Evolution rules - which fields can change, whether to track history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evolution: Option<EvolutionConfig>,

    /// Template inheritance - base template name + field overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub templates: Option<PersonaTemplateRef>,

    /// Token cap for the persona section. Triggers condensed format when exceeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_prompt_tokens: Option<u32>,
}

impl PersonaConfig {
    /// Returns true if a persona is meaningfully configured.
    pub fn is_configured(&self) -> bool {
        self.identity.is_some() || self.templates.is_some()
    }
}

/// Core identity fields for an agent persona.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaIdentity {
    /// Display name.
    pub name: String,

    /// Functional role description.
    pub role: String,

    /// Short one-line description for UI, API responses, and agent listings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Rich backstory text.
    /// Supports Jinja2 templates rendered with ContextManager values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backstory: Option<String>,

    /// Group, organization, team, faction, or department.
    /// Domain-neutral - games use factions, corps use departments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub affiliation: Option<String>,
}

/// Structured personality traits for an agent.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersonaTraits {
    /// Core personality descriptors (e.g., ["disciplined", "suspicious", "loyal"]).
    #[serde(default)]
    pub personality: Vec<String>,

    /// What the agent cares about.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<String>>,

    /// What the agent is afraid of or avoids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fears: Option<Vec<String>>,

    /// Speaking style instruction included verbatim in the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaking_style: Option<String>,
}

/// Goals for an agent persona.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaGoals {
    /// Public goals the agent actively pursues.
    #[serde(default)]
    pub primary: Vec<String>,

    /// Goals excluded from the LLM prompt. Readable by application code only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden: Option<Vec<String>>,
}

/// A secret the agent knows but withholds until conditions are met.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSecret {
    /// The secret information.
    pub content: String,

    /// Conditions for revelation. When None, the secret is never auto-revealed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reveal_conditions: Option<SecretRevealCondition>,
}

/// Typed condition for secret revelation.
/// Evaluates against ContextManager values using the same matchers as state machine guards.
/// Custom Serialize/Deserialize uses single-key-map YAML format (context/all/any).
#[derive(Debug, Clone)]
pub enum SecretRevealCondition {
    /// Single context path comparison.
    Context(HashMap<String, ContextMatcher>),

    /// All conditions must be satisfied.
    All(Vec<SecretRevealCondition>),

    /// Any condition suffices.
    Any(Vec<SecretRevealCondition>),
}

// Manual Serialize/Deserialize implementation is required because serde's `untagged` representation cannot distinguish between the `All` and `Any` variants when deserializing.
impl Serialize for SecretRevealCondition {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;
        match self {
            SecretRevealCondition::Context(matchers) => {
                map.serialize_entry("context", matchers)?;
            }
            SecretRevealCondition::All(conditions) => {
                map.serialize_entry("all", conditions)?;
            }
            SecretRevealCondition::Any(conditions) => {
                map.serialize_entry("any", conditions)?;
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for SecretRevealCondition {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let map: HashMap<String, serde_json::Value> = HashMap::deserialize(deserializer)?;
        if map.len() != 1 {
            return Err(serde::de::Error::custom(
                "SecretRevealCondition must be a single-key map (context, all, or any)",
            ));
        }
        let (key, value) = map.into_iter().next().unwrap();
        match key.as_str() {
            "context" => {
                let matchers: HashMap<String, ContextMatcher> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(SecretRevealCondition::Context(matchers))
            }
            "all" => {
                let conditions: Vec<SecretRevealCondition> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(SecretRevealCondition::All(conditions))
            }
            "any" => {
                let conditions: Vec<SecretRevealCondition> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(SecretRevealCondition::Any(conditions))
            }
            other => Err(serde::de::Error::custom(format!(
                "Unknown SecretRevealCondition key '{}'. Expected context, all, or any",
                other
            ))),
        }
    }
}

/// Controls how the persona evolves over time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvolutionConfig {
    /// Allow evolve() calls from Rust API and hooks. Default false.
    #[serde(default)]
    pub enabled: bool,

    /// Dot-notation field paths that may be mutated. Validated against VALID_EVOLVE_PATHS.
    #[serde(default)]
    pub mutable_fields: Vec<String>,

    /// Keep a history of all changes.
    #[serde(default)]
    pub track_changes: bool,

    /// Register persona_evolve tool for LLM self-modification. Default false.
    #[serde(default)]
    pub allow_llm_evolve: bool,
}

/// Reference to a base persona template with optional overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaTemplateRef {
    /// Name of the base template to inherit from.
    pub base: String,

    /// Field overrides applied on top of the base template (dot-notation keys).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overrides: Option<HashMap<String, Value>>,
}

/// Known dot-paths allowed in mutable_fields. Any path not in this list is rejected.
pub const VALID_EVOLVE_PATHS: &[&str] = &[
    "identity.name",
    "identity.role",
    "identity.description",
    "identity.backstory",
    "identity.affiliation",
    "traits.personality",
    "traits.values",
    "traits.fears",
    "traits.speaking_style",
    "goals.primary",
    "goals.hidden",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_persona_config_parse() {
        let yaml = r#"
identity:
  name: "Captain Elira"
  role: "Harbor Guard Captain"
  description: "A disciplined former soldier"
  backstory: |
    Former soldier who served in the Eastern Campaign.
  affiliation: "Harbor Watch"
traits:
  personality: [disciplined, suspicious, loyal]
  values: [duty, order, justice]
  fears: [civil_unrest, betrayal]
  speaking_style: "formal military cadence"
goals:
  primary:
    - protect_harbor
    - investigate_smuggling
  hidden:
    - "Find the spy within the Watch"
secrets:
  - content: "Investigating a smuggling ring"
    reveal_conditions:
      all:
        - context:
            relationships.current_actor.trust:
              gte: 0.8
        - context:
            actor.is_watch_member:
              eq: true
evolution:
  enabled: true
  mutable_fields:
    - traits.personality
    - traits.speaking_style
  track_changes: true
  allow_llm_evolve: false
max_prompt_tokens: 400
"#;
        let config: PersonaConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_configured());

        let id = config.identity.as_ref().unwrap();
        assert_eq!(id.name, "Captain Elira");
        assert_eq!(id.role, "Harbor Guard Captain");
        assert_eq!(id.affiliation.as_deref(), Some("Harbor Watch"));

        let traits = config.traits.as_ref().unwrap();
        assert_eq!(traits.personality.len(), 3);
        assert_eq!(traits.values.as_ref().unwrap().len(), 3);
        assert_eq!(traits.fears.as_ref().unwrap().len(), 2);

        let goals = config.goals.as_ref().unwrap();
        assert_eq!(goals.primary.len(), 2);
        assert_eq!(goals.hidden.as_ref().unwrap().len(), 1);

        let secrets = config.secrets.as_ref().unwrap();
        assert_eq!(secrets.len(), 1);
        assert!(secrets[0].reveal_conditions.is_some());

        let evo = config.evolution.as_ref().unwrap();
        assert!(evo.enabled);
        assert_eq!(evo.mutable_fields.len(), 2);
        assert!(evo.track_changes);
        assert!(!evo.allow_llm_evolve);

        assert_eq!(config.max_prompt_tokens, Some(400));
    }

    #[test]
    fn test_minimal_persona_config_parse() {
        let yaml = r#"
identity:
  name: "Support Agent"
  role: "Customer Support"
"#;
        let config: PersonaConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_configured());
        assert!(config.traits.is_none());
        assert!(config.goals.is_none());
        assert!(config.secrets.is_none());
        assert!(config.evolution.is_none());
        assert!(config.max_prompt_tokens.is_none());
    }

    #[test]
    fn test_empty_persona_not_configured() {
        let config = PersonaConfig::default();
        assert!(!config.is_configured());
    }

    #[test]
    fn test_template_ref_only_is_configured() {
        let yaml = r#"
templates:
  base: "guard_base"
  overrides:
    identity.name: "Guard Tam"
goals:
  primary: [protect_harbor]
"#;
        let config: PersonaConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_configured());
        assert!(config.identity.is_none());
        assert!(config.templates.is_some());
        assert_eq!(config.templates.as_ref().unwrap().base, "guard_base");
    }

    #[test]
    fn test_persona_config_roundtrip() {
        let yaml = r#"
identity:
  name: "Alex"
  role: "Support Rep"
traits:
  personality: [empathetic, patient]
  speaking_style: "warm and professional"
"#;
        let config: PersonaConfig = serde_yaml::from_str(yaml).unwrap();
        let serialized = serde_yaml::to_string(&config).unwrap();
        let reparsed: PersonaConfig = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            reparsed.identity.as_ref().unwrap().name,
            config.identity.as_ref().unwrap().name
        );
    }

    #[test]
    fn test_secret_condition_context_parse() {
        let yaml = r#"
content: "secret info"
reveal_conditions:
  context:
    trust_level:
      gte: 0.8
"#;
        let secret: PersonaSecret = serde_yaml::from_str(yaml).unwrap();
        assert!(secret.reveal_conditions.is_some());
        match secret.reveal_conditions.as_ref().unwrap() {
            SecretRevealCondition::Context(matchers) => {
                assert!(matchers.contains_key("trust_level"));
            }
            _ => panic!("Expected Context variant"),
        }
    }

    #[test]
    fn test_secret_condition_all_parse() {
        let yaml = r#"
content: "hidden"
reveal_conditions:
  all:
    - context:
        trust:
          gte: 0.5
    - context:
        is_member:
          eq: true
"#;
        let secret: PersonaSecret = serde_yaml::from_str(yaml).unwrap();
        match secret.reveal_conditions.as_ref().unwrap() {
            SecretRevealCondition::All(conditions) => {
                assert_eq!(conditions.len(), 2);
            }
            _ => panic!("Expected All variant"),
        }
    }

    #[test]
    fn test_secret_no_conditions() {
        let yaml = r#"
content: "manual only"
"#;
        let secret: PersonaSecret = serde_yaml::from_str(yaml).unwrap();
        assert!(secret.reveal_conditions.is_none());
    }

    #[test]
    fn test_evolution_config_defaults() {
        let evo = EvolutionConfig::default();
        assert!(!evo.enabled);
        assert!(evo.mutable_fields.is_empty());
        assert!(!evo.track_changes);
        assert!(!evo.allow_llm_evolve);
    }

    #[test]
    fn test_customer_service_persona() {
        let yaml = r#"
identity:
  name: "Alex"
  role: "Senior Support Representative"
  description: "Experienced support rep specializing in account issues"
  affiliation: "Acme Corp Customer Success"
traits:
  personality: [empathetic, solution-oriented, patient]
  values: [customer_satisfaction, accuracy, transparency]
  speaking_style: "professional but warm, uses customer's name, avoids jargon"
goals:
  primary:
    - resolve_customer_issue
    - maintain_brand_voice
evolution:
  enabled: true
  mutable_fields:
    - traits.speaking_style
  track_changes: false
"#;
        let config: PersonaConfig = serde_yaml::from_str(yaml).unwrap();
        let id = config.identity.as_ref().unwrap();
        assert_eq!(id.name, "Alex");
        assert_eq!(
            id.affiliation.as_deref(),
            Some("Acme Corp Customer Success")
        );

        let evo = config.evolution.as_ref().unwrap();
        assert!(evo.enabled);
        assert!(!evo.track_changes);
    }

    #[test]
    fn test_valid_evolve_paths_contains_expected() {
        assert!(VALID_EVOLVE_PATHS.contains(&"traits.personality"));
        assert!(VALID_EVOLVE_PATHS.contains(&"identity.name"));
        assert!(VALID_EVOLVE_PATHS.contains(&"goals.primary"));
        assert!(!VALID_EVOLVE_PATHS.contains(&"secrets"));
        assert!(!VALID_EVOLVE_PATHS.contains(&"templates"));
    }

    #[test]
    fn test_secret_condition_any_parse() {
        let yaml = r#"
content: "any condition"
reveal_conditions:
  any:
    - context:
        trust:
          gte: 0.9
    - context:
        is_admin:
          eq: true
"#;
        let secret: PersonaSecret = serde_yaml::from_str(yaml).unwrap();
        match secret.reveal_conditions.as_ref().unwrap() {
            SecretRevealCondition::Any(conditions) => {
                assert_eq!(conditions.len(), 2);
            }
            _ => panic!("Expected Any variant"),
        }
    }

    #[test]
    fn test_persona_identity_serialize_skip_none() {
        let id = PersonaIdentity {
            name: "Test".into(),
            role: "Role".into(),
            description: None,
            backstory: None,
            affiliation: None,
        };
        let val = serde_json::to_value(&id).unwrap();
        assert!(!val.as_object().unwrap().contains_key("description"));
        assert!(!val.as_object().unwrap().contains_key("backstory"));
        assert!(!val.as_object().unwrap().contains_key("affiliation"));
    }
}
