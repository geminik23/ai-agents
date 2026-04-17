//! Persona snapshot for persistence.
use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use ai_agents_core::{AgentError, Result};

use crate::config::PersonaConfig;
use crate::evolution::PersonaChange;

/// Serializable snapshot of persona state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSnapshot {
    /// Full persona config at time of snapshot.
    pub config: PersonaConfig,

    /// Evolution history up to this point.
    pub history: Vec<PersonaChange>,

    /// Set of secret indices that have been revealed.
    pub revealed_indices: HashSet<usize>,

    /// When the snapshot was taken.
    pub snapshot_at: DateTime<Utc>,
}

impl PersonaSnapshot {
    /// Create a new snapshot from current state.
    pub fn new(
        config: PersonaConfig,
        history: Vec<PersonaChange>,
        revealed_indices: HashSet<usize>,
    ) -> Self {
        Self {
            config,
            history,
            revealed_indices,
            snapshot_at: Utc::now(),
        }
    }

    /// Serialize this snapshot to a serde_json::Value for storage in AgentSnapshot.
    pub fn to_value(&self) -> Result<Value> {
        serde_json::to_value(self)
            .map_err(|e| AgentError::Config(format!("Failed to serialize persona snapshot: {}", e)))
    }

    /// Deserialize a PersonaSnapshot from a serde_json::Value.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|e| {
            AgentError::Config(format!("Failed to deserialize persona snapshot: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use serde_json::json;

    fn make_test_config() -> PersonaConfig {
        PersonaConfig {
            identity: Some(PersonaIdentity {
                name: "Test Agent".into(),
                role: "Tester".into(),
                description: Some("A test agent".into()),
                backstory: None,
                affiliation: None,
            }),
            traits: Some(PersonaTraits {
                personality: vec!["curious".into()],
                values: None,
                fears: None,
                speaking_style: Some("concise".into()),
            }),
            goals: None,
            secrets: None,
            evolution: Some(EvolutionConfig {
                enabled: true,
                mutable_fields: vec!["traits.personality".into()],
                track_changes: true,
                allow_llm_evolve: false,
            }),
            templates: None,
            max_prompt_tokens: None,
        }
    }

    fn make_test_change() -> PersonaChange {
        PersonaChange {
            field: "traits.personality".into(),
            old_value: json!(["curious"]),
            new_value: json!(["curious", "bold"]),
            timestamp: chrono::Utc::now(),
            reason: Some("Gained confidence".into()),
        }
    }

    #[test]
    fn test_snapshot_new() {
        let config = make_test_config();
        let history = vec![make_test_change()];
        let mut revealed = HashSet::new();
        revealed.insert(0);

        let snapshot = PersonaSnapshot::new(config.clone(), history.clone(), revealed.clone());

        assert_eq!(
            snapshot.config.identity.as_ref().unwrap().name,
            "Test Agent"
        );
        assert_eq!(snapshot.history.len(), 1);
        assert!(snapshot.revealed_indices.contains(&0));
        assert!(snapshot.snapshot_at <= Utc::now());
    }

    #[test]
    fn test_snapshot_to_value_roundtrip() {
        let config = make_test_config();
        let history = vec![make_test_change()];
        let mut revealed = HashSet::new();
        revealed.insert(1);
        revealed.insert(3);

        let snapshot = PersonaSnapshot::new(config, history, revealed);

        let value = snapshot.to_value().unwrap();
        assert!(value.is_object());

        let restored = PersonaSnapshot::from_value(value).unwrap();
        assert_eq!(
            restored.config.identity.as_ref().unwrap().name,
            "Test Agent"
        );
        assert_eq!(restored.history.len(), 1);
        assert_eq!(restored.history[0].field, "traits.personality");
        assert!(restored.revealed_indices.contains(&1));
        assert!(restored.revealed_indices.contains(&3));
        assert_eq!(restored.revealed_indices.len(), 2);
    }

    #[test]
    fn test_snapshot_empty_history() {
        let config = make_test_config();
        let snapshot = PersonaSnapshot::new(config, vec![], HashSet::new());

        let value = snapshot.to_value().unwrap();
        let restored = PersonaSnapshot::from_value(value).unwrap();
        assert!(restored.history.is_empty());
        assert!(restored.revealed_indices.is_empty());
    }

    #[test]
    fn test_snapshot_from_invalid_value() {
        let bad_value = json!({"invalid": "data"});
        let result = PersonaSnapshot::from_value(bad_value);
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_preserves_evolution_config() {
        let config = make_test_config();
        let snapshot = PersonaSnapshot::new(config, vec![], HashSet::new());

        let value = snapshot.to_value().unwrap();
        let restored = PersonaSnapshot::from_value(value).unwrap();

        let evo = restored.config.evolution.as_ref().unwrap();
        assert!(evo.enabled);
        assert_eq!(evo.mutable_fields, vec!["traits.personality".to_string()]);
        assert!(evo.track_changes);
        assert!(!evo.allow_llm_evolve);
    }

    #[test]
    fn test_snapshot_with_multiple_changes() {
        let config = make_test_config();
        let changes = vec![
            PersonaChange {
                field: "traits.personality".into(),
                old_value: json!(["curious"]),
                new_value: json!(["curious", "bold"]),
                timestamp: chrono::Utc::now(),
                reason: Some("First change".into()),
            },
            PersonaChange {
                field: "traits.personality".into(),
                old_value: json!(["curious", "bold"]),
                new_value: json!(["curious", "bold", "wise"]),
                timestamp: chrono::Utc::now(),
                reason: Some("Second change".into()),
            },
        ];

        let snapshot = PersonaSnapshot::new(config, changes, HashSet::new());
        let value = snapshot.to_value().unwrap();
        let restored = PersonaSnapshot::from_value(value).unwrap();

        assert_eq!(restored.history.len(), 2);
        assert_eq!(restored.history[0].reason.as_deref(), Some("First change"));
        assert_eq!(restored.history[1].reason.as_deref(), Some("Second change"));
    }

    #[test]
    fn test_snapshot_serde_json_roundtrip() {
        let config = make_test_config();
        let snapshot = PersonaSnapshot::new(config, vec![make_test_change()], HashSet::new());

        let json_str = serde_json::to_string(&snapshot).unwrap();
        let deserialized: PersonaSnapshot = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            deserialized.config.identity.as_ref().unwrap().name,
            "Test Agent"
        );
        assert_eq!(deserialized.history.len(), 1);
    }
}
