//! Persona evolution - field mutation, validation, and change tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use ai_agents_core::{AgentError, Result, get_dot_path, set_dot_path};

use crate::config::{EvolutionConfig, PersonaConfig, VALID_EVOLVE_PATHS};

/// A single recorded change in persona evolution history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaChange {
    /// Dot-notation path of the field that changed.
    pub field: String,

    /// Previous value.
    pub old_value: Value,

    /// New value.
    pub new_value: Value,

    /// When the change occurred.
    pub timestamp: DateTime<Utc>,

    /// Optional reason for the change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Validate that all mutable_fields paths are allowed.
/// Rejects any path starting with "secrets" and any path not in VALID_EVOLVE_PATHS.
pub fn validate_mutable_fields(fields: &[String]) -> Result<()> {
    for field in fields {
        if field.starts_with("secrets") {
            return Err(AgentError::Config(format!(
                "Persona mutable_fields cannot include secrets paths ('{}').\
                 Mutating secrets would invalidate revealed_indices tracking.",
                field
            )));
        }
        if !VALID_EVOLVE_PATHS.contains(&field.as_str()) {
            return Err(AgentError::Config(format!(
                "Unknown persona mutable_fields path '{}'. Valid paths: {}",
                field,
                VALID_EVOLVE_PATHS.join(", ")
            )));
        }
    }
    Ok(())
}

/// Apply an evolution mutation to a PersonaConfig.
/// Validates that evolution is enabled, the field is in mutable_fields, and the field is not a secrets path.
/// Returns the PersonaChange record on success.
pub fn evolve_field(
    config: &PersonaConfig,
    evolution: &EvolutionConfig,
    field: &str,
    new_value: Value,
    reason: Option<&str>,
) -> Result<(PersonaConfig, PersonaChange)> {
    if !evolution.enabled {
        return Err(AgentError::Config(
            "Persona evolution is not enabled".into(),
        ));
    }

    if field.starts_with("secrets") {
        return Err(AgentError::Config(format!(
            "Cannot evolve secrets path '{}'. Secrets are immutable.",
            field
        )));
    }

    if !evolution.mutable_fields.contains(&field.to_string()) {
        return Err(AgentError::Config(format!(
            "Field '{}' is not in persona mutable_fields. Allowed: [{}]",
            field,
            evolution.mutable_fields.join(", ")
        )));
    }

    let serialized = serde_json::to_value(config)
        .map_err(|e| AgentError::Config(format!("Failed to serialize persona config: {}", e)))?;

    let old_value = get_dot_path(&serialized, field)
        .cloned()
        .unwrap_or(Value::Null);

    let updated = set_dot_path(serialized, field, new_value.clone())?;

    let new_config: PersonaConfig = serde_json::from_value(updated)
        .map_err(|e| AgentError::Config(format!("Failed to apply evolution to config: {}", e)))?;

    let change = PersonaChange {
        field: field.to_string(),
        old_value,
        new_value,
        timestamp: Utc::now(),
        reason: reason.map(|s| s.to_string()),
    };

    Ok((new_config, change))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use serde_json::json;

    fn make_evolvable_config() -> PersonaConfig {
        PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Guard".into(),
                role: "Patrol".into(),
                description: Some("A guard".into()),
                backstory: None,
                affiliation: None,
            }),
            traits: Some(PersonaTraits {
                personality: vec!["suspicious".into(), "stern".into()],
                values: Some(vec!["duty".into()]),
                fears: Some(vec!["chaos".into()]),
                speaking_style: Some("formal".into()),
            }),
            goals: Some(PersonaGoals {
                primary: vec!["patrol".into()],
                hidden: None,
            }),
            secrets: None,
            evolution: Some(EvolutionConfig {
                enabled: true,
                mutable_fields: vec![
                    "traits.personality".into(),
                    "traits.speaking_style".into(),
                    "goals.primary".into(),
                ],
                track_changes: true,
                allow_llm_evolve: false,
            }),
            templates: None,
            max_prompt_tokens: None,
        }
    }

    #[test]
    fn test_validate_mutable_fields_valid() {
        let fields = vec![
            "traits.personality".into(),
            "traits.speaking_style".into(),
            "goals.primary".into(),
        ];
        assert!(validate_mutable_fields(&fields).is_ok());
    }

    #[test]
    fn test_validate_mutable_fields_rejects_secrets() {
        let fields = vec!["secrets".into()];
        let err = validate_mutable_fields(&fields).unwrap_err();
        assert!(err.to_string().contains("secrets"));
    }

    #[test]
    fn test_validate_mutable_fields_rejects_secrets_subpath() {
        let fields = vec!["secrets.0.content".into()];
        let err = validate_mutable_fields(&fields).unwrap_err();
        assert!(err.to_string().contains("secrets"));
    }

    #[test]
    fn test_validate_mutable_fields_rejects_unknown() {
        let fields = vec!["traits.personailty".into()];
        let err = validate_mutable_fields(&fields).unwrap_err();
        assert!(err.to_string().contains("Unknown"));
        assert!(err.to_string().contains("traits.personailty"));
    }

    #[test]
    fn test_validate_mutable_fields_empty_ok() {
        assert!(validate_mutable_fields(&[]).is_ok());
    }

    #[test]
    fn test_evolve_personality() {
        let config = make_evolvable_config();
        let evo = config.evolution.as_ref().unwrap();

        let (new_config, change) = evolve_field(
            &config,
            evo,
            "traits.personality",
            json!(["confident", "assertive", "loyal"]),
            Some("Player proved trustworthy"),
        )
        .unwrap();

        assert_eq!(
            new_config.traits.as_ref().unwrap().personality,
            vec!["confident", "assertive", "loyal"]
        );
        assert_eq!(change.field, "traits.personality");
        assert_eq!(change.old_value, json!(["suspicious", "stern"]));
        assert_eq!(change.new_value, json!(["confident", "assertive", "loyal"]));
        assert_eq!(change.reason.as_deref(), Some("Player proved trustworthy"));
    }

    #[test]
    fn test_evolve_speaking_style() {
        let config = make_evolvable_config();
        let evo = config.evolution.as_ref().unwrap();

        let (new_config, change) = evolve_field(
            &config,
            evo,
            "traits.speaking_style",
            json!("casual and friendly"),
            None,
        )
        .unwrap();

        assert_eq!(
            new_config
                .traits
                .as_ref()
                .unwrap()
                .speaking_style
                .as_deref(),
            Some("casual and friendly")
        );
        assert_eq!(change.old_value, json!("formal"));
        assert!(change.reason.is_none());
    }

    #[test]
    fn test_evolve_goals() {
        let config = make_evolvable_config();
        let evo = config.evolution.as_ref().unwrap();

        let (new_config, _change) = evolve_field(
            &config,
            evo,
            "goals.primary",
            json!(["protect_harbor", "investigate"]),
            Some("New mission"),
        )
        .unwrap();

        assert_eq!(
            new_config.goals.as_ref().unwrap().primary,
            vec!["protect_harbor", "investigate"]
        );
    }

    #[test]
    fn test_evolve_disabled() {
        let mut config = make_evolvable_config();
        config.evolution.as_mut().unwrap().enabled = false;
        let evo = config.evolution.as_ref().unwrap();

        let err =
            evolve_field(&config, evo, "traits.personality", json!(["new"]), None).unwrap_err();
        assert!(err.to_string().contains("not enabled"));
    }

    #[test]
    fn test_evolve_field_not_in_mutable() {
        let config = make_evolvable_config();
        let evo = config.evolution.as_ref().unwrap();

        let err = evolve_field(&config, evo, "identity.name", json!("New Name"), None).unwrap_err();
        assert!(err.to_string().contains("not in persona mutable_fields"));
    }

    #[test]
    fn test_evolve_rejects_secrets_path() {
        let mut config = make_evolvable_config();
        config
            .evolution
            .as_mut()
            .unwrap()
            .mutable_fields
            .push("secrets".into());
        let evo = config.evolution.as_ref().unwrap();

        let err = evolve_field(&config, evo, "secrets", json!([]), None).unwrap_err();
        assert!(err.to_string().contains("Secrets are immutable"));
    }

    #[test]
    fn test_evolve_preserves_other_fields() {
        let config = make_evolvable_config();
        let evo = config.evolution.as_ref().unwrap();

        let (new_config, _) = evolve_field(
            &config,
            evo,
            "traits.personality",
            json!(["new_trait"]),
            None,
        )
        .unwrap();

        // Identity unchanged.
        assert_eq!(new_config.identity.as_ref().unwrap().name, "Guard");
        assert_eq!(new_config.identity.as_ref().unwrap().role, "Patrol");

        // Other trait fields unchanged.
        assert_eq!(
            new_config
                .traits
                .as_ref()
                .unwrap()
                .speaking_style
                .as_deref(),
            Some("formal")
        );
        assert_eq!(
            new_config.traits.as_ref().unwrap().values,
            Some(vec!["duty".to_string()])
        );

        // Goals unchanged.
        assert_eq!(
            new_config.goals.as_ref().unwrap().primary,
            vec!["patrol".to_string()]
        );
    }

    #[test]
    fn test_evolve_null_old_value_for_missing_field() {
        let mut config = make_evolvable_config();
        config.traits.as_mut().unwrap().fears = None;
        let mut evo = config.evolution.clone().unwrap();
        evo.mutable_fields.push("traits.fears".into());

        let (_new_config, change) =
            evolve_field(&config, &evo, "traits.fears", json!(["new_fear"]), None).unwrap();

        assert_eq!(change.old_value, Value::Null);
        assert_eq!(change.new_value, json!(["new_fear"]));
    }

    #[test]
    fn test_persona_change_serialization() {
        let change = PersonaChange {
            field: "traits.personality".into(),
            old_value: json!(["shy"]),
            new_value: json!(["bold"]),
            timestamp: Utc::now(),
            reason: Some("Character growth".into()),
        };

        let json = serde_json::to_value(&change).unwrap();
        assert_eq!(json["field"], "traits.personality");
        assert!(json.get("reason").is_some());

        let deserialized: PersonaChange = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.field, change.field);
    }

    #[test]
    fn test_persona_change_no_reason_skips() {
        let change = PersonaChange {
            field: "traits.personality".into(),
            old_value: json!(["a"]),
            new_value: json!(["b"]),
            timestamp: Utc::now(),
            reason: None,
        };

        let json = serde_json::to_string(&change).unwrap();
        assert!(!json.contains("reason"));
    }
}
