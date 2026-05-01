use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::Value;

use ai_agents_core::{AgentError, ChatMessage, Result};

use crate::config::{EventEvictionStrategy, RelationshipConfig};
use crate::evaluator::RelationshipEvaluatorTrait;
use crate::injection::{format_relationship, relationship_to_context_value};
use crate::snapshot::{RelationshipSnapshot, relationship_from_value, relationship_to_value};
use crate::types::{
    DimensionChange, Relationship, RelationshipDimensionDefinition, RelationshipEvaluation,
    RelationshipEvent, RelationshipUpdate,
};

pub struct RelationshipManager {
    config: RelationshipConfig,
    definitions: HashMap<String, RelationshipDimensionDefinition>,
    relationships: RwLock<HashMap<String, Relationship>>,
    evaluator: Option<Arc<dyn RelationshipEvaluatorTrait>>,
}

impl RelationshipManager {
    pub fn from_config(config: RelationshipConfig) -> Result<Self> {
        Self::from_config_with_evaluator(config, None)
    }

    pub fn from_config_with_evaluator(
        config: RelationshipConfig,
        evaluator: Option<Arc<dyn RelationshipEvaluatorTrait>>,
    ) -> Result<Self> {
        let definitions = config.dimension_definitions()?;
        Ok(Self {
            config,
            definitions,
            relationships: RwLock::new(HashMap::new()),
            evaluator,
        })
    }

    pub fn config(&self) -> &RelationshipConfig {
        &self.config
    }

    pub fn definitions(&self) -> &HashMap<String, RelationshipDimensionDefinition> {
        &self.definitions
    }

    pub fn get_or_create(&self, actor_id: &str, actor_name: Option<&str>) -> Relationship {
        let mut relationships = self.relationships.write();
        relationships
            .entry(actor_id.to_string())
            .or_insert_with(|| {
                Relationship::new(
                    actor_id,
                    actor_name.map(|s| s.to_string()),
                    &self.definitions,
                )
            })
            .clone()
    }

    pub fn insert(&self, relationship: Relationship) {
        self.relationships
            .write()
            .insert(relationship.actor_id.clone(), relationship);
    }

    pub fn get(&self, actor_id: &str) -> Option<Relationship> {
        self.relationships.read().get(actor_id).cloned()
    }

    pub fn list_actors(&self) -> Vec<String> {
        let mut actors: Vec<_> = self.relationships.read().keys().cloned().collect();
        actors.sort();
        actors
    }

    pub fn remove(&self, actor_id: &str) -> Option<Relationship> {
        self.relationships.write().remove(actor_id)
    }

    pub fn touch_interaction(&self, actor_id: &str, actor_name: Option<&str>) -> Relationship {
        let mut relationships = self.relationships.write();
        let relationship = relationships
            .entry(actor_id.to_string())
            .or_insert_with(|| {
                Relationship::new(
                    actor_id,
                    actor_name.map(|s| s.to_string()),
                    &self.definitions,
                )
            });
        relationship.touch(actor_name);
        relationship.clone()
    }

    pub fn update_dimension(
        &self,
        actor_id: &str,
        dimension: &str,
        delta: f64,
        confidence: f64,
        reason: impl Into<String>,
    ) -> Result<DimensionChange> {
        let def = self.definitions.get(dimension).ok_or_else(|| {
            AgentError::Config(format!("Unknown relationship dimension '{}'", dimension))
        })?;

        let mut relationships = self.relationships.write();
        let relationship = relationships
            .entry(actor_id.to_string())
            .or_insert_with(|| Relationship::new(actor_id, None, &self.definitions));
        let previous = relationship
            .dimensions
            .get(dimension)
            .copied()
            .unwrap_or(def.default);
        let current = def.clamp(previous + delta);
        relationship
            .dimensions
            .insert(dimension.to_string(), current);
        relationship.last_interaction = chrono::Utc::now();

        Ok(DimensionChange {
            dimension: dimension.to_string(),
            previous,
            current,
            delta: current - previous,
            confidence,
            reason: reason.into(),
        })
    }

    pub fn apply_evaluation(
        &self,
        actor_id: &str,
        evaluation: RelationshipEvaluation,
    ) -> Result<RelationshipUpdate> {
        let mut changes = Vec::new();

        for proposed in evaluation.changes {
            if proposed.confidence < self.config.auto_update.min_confidence {
                continue;
            }
            if !self.definitions.contains_key(&proposed.dimension) {
                continue;
            }
            let max_delta = self.config.auto_update.max_delta_per_turn.abs();
            let delta = proposed.delta.max(-max_delta).min(max_delta);
            if delta.abs() < f64::EPSILON {
                continue;
            }
            let change = self.update_dimension(
                actor_id,
                &proposed.dimension,
                delta,
                proposed.confidence,
                proposed.reason,
            )?;
            if change.delta.abs() >= f64::EPSILON {
                changes.push(change);
            }
        }

        let event = if let Some(proposed) = evaluation.notable_event {
            if self.config.notable_events.enabled
                && proposed.significance >= self.config.notable_events.significance_threshold
                && !proposed.description.trim().is_empty()
            {
                let event = RelationshipEvent::new(
                    proposed.description,
                    changes.clone(),
                    proposed.significance.clamp(0.0, 1.0),
                );
                self.record_event(actor_id, event.clone())?;
                Some(event)
            } else {
                None
            }
        } else {
            None
        };

        let relationship = self.get_or_create(actor_id, None);
        Ok(RelationshipUpdate {
            relationship,
            changes,
            event,
        })
    }

    pub async fn auto_update(
        &self,
        actor_id: &str,
        messages: &[ChatMessage],
    ) -> Result<RelationshipUpdate> {
        let relationship = self.get_or_create(actor_id, None);
        let Some(ref evaluator) = self.evaluator else {
            return Ok(RelationshipUpdate {
                relationship,
                changes: Vec::new(),
                event: None,
            });
        };

        let evaluation = evaluator
            .evaluate_turn(&relationship, messages, &self.definitions)
            .await?;
        self.apply_evaluation(actor_id, evaluation)
    }

    pub fn record_event(&self, actor_id: &str, event: RelationshipEvent) -> Result<()> {
        if !self.config.notable_events.enabled {
            return Ok(());
        }
        if event.significance < self.config.notable_events.significance_threshold {
            return Ok(());
        }

        let mut relationships = self.relationships.write();
        let relationship = relationships
            .entry(actor_id.to_string())
            .or_insert_with(|| Relationship::new(actor_id, None, &self.definitions));
        relationship.notable_events.push(event);
        enforce_event_limit(
            &mut relationship.notable_events,
            self.config.notable_events.max_per_actor,
            &self.config.notable_events.eviction,
        );
        Ok(())
    }

    pub fn format_for_prompt(&self, actor_id: &str) -> String {
        if !self.config.injection.enabled {
            return String::new();
        }
        match self.get(actor_id) {
            Some(relationship) => format_relationship(
                &relationship,
                &self.config.injection.format,
                self.config.injection.max_tokens,
            ),
            None => String::new(),
        }
    }

    pub fn to_context_value(&self, actor_id: &str) -> Option<Value> {
        self.get(actor_id)
            .map(|relationship| relationship_to_context_value(&relationship))
    }

    pub fn relationship_as_value(&self, actor_id: &str) -> Result<Option<Value>> {
        self.get(actor_id)
            .map(|relationship| relationship_to_value(&relationship))
            .transpose()
    }

    pub fn insert_from_value(&self, value: Value) -> Result<Relationship> {
        let relationship = relationship_from_value(value)?;
        self.insert(relationship.clone());
        Ok(relationship)
    }

    pub fn snapshot(&self) -> RelationshipSnapshot {
        RelationshipSnapshot::new(self.relationships.read().values().cloned().collect())
    }

    pub fn snapshot_as_value(&self) -> Result<Value> {
        self.snapshot().to_value()
    }

    pub fn restore_from_value(&self, value: Value) -> Result<()> {
        let snapshot = RelationshipSnapshot::from_value(value)?;
        let mut relationships = self.relationships.write();
        relationships.clear();
        for relationship in snapshot.relationships {
            relationships.insert(relationship.actor_id.clone(), relationship);
        }
        Ok(())
    }
}

fn enforce_event_limit(
    events: &mut Vec<RelationshipEvent>,
    max_events: usize,
    strategy: &EventEvictionStrategy,
) {
    if max_events == 0 {
        events.clear();
        return;
    }
    if events.len() <= max_events {
        return;
    }

    match strategy {
        EventEvictionStrategy::Oldest => {
            events.sort_by_key(|event| event.timestamp);
            let remove_count = events.len().saturating_sub(max_events);
            events.drain(0..remove_count);
        }
        EventEvictionStrategy::LowestSignificanceThenOldest => {
            events.sort_by(|a, b| {
                b.significance
                    .partial_cmp(&a.significance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.timestamp.cmp(&a.timestamp))
            });
            events.truncate(max_events);
            events.sort_by_key(|event| event.timestamp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RelationshipConfig;
    use crate::types::{ProposedDimensionChange, ProposedRelationshipEvent};

    #[test]
    fn test_get_or_create_defaults() {
        let manager = RelationshipManager::from_config(RelationshipConfig::default()).unwrap();
        let rel = manager.get_or_create("actor_1", Some("Alice"));
        assert_eq!(rel.actor_name.as_deref(), Some("Alice"));
        assert!(rel.dimensions.contains_key("trust"));
    }

    #[test]
    fn test_update_dimension_clamps() {
        let manager = RelationshipManager::from_config(RelationshipConfig::default()).unwrap();
        let change = manager
            .update_dimension("actor_1", "trust", 5.0, 1.0, "test")
            .unwrap();
        assert_eq!(change.current, 1.0);
    }

    #[test]
    fn test_apply_evaluation_filters_low_confidence() {
        let manager = RelationshipManager::from_config(RelationshipConfig::default()).unwrap();
        manager.get_or_create("actor_1", None);
        let update = manager
            .apply_evaluation(
                "actor_1",
                RelationshipEvaluation {
                    changes: vec![ProposedDimensionChange {
                        dimension: "trust".to_string(),
                        delta: 0.2,
                        confidence: 0.1,
                        reason: "weak".to_string(),
                    }],
                    notable_event: None,
                },
            )
            .unwrap();
        assert!(update.changes.is_empty());
    }

    #[test]
    fn test_record_event_respects_limit() {
        let mut config = RelationshipConfig::default();
        config.notable_events.max_per_actor = 1;
        let manager = RelationshipManager::from_config(config).unwrap();
        manager.get_or_create("actor_1", None);
        manager
            .record_event("actor_1", RelationshipEvent::new("low", vec![], 0.6))
            .unwrap();
        manager
            .record_event("actor_1", RelationshipEvent::new("high", vec![], 0.9))
            .unwrap();
        let rel = manager.get("actor_1").unwrap();
        assert_eq!(rel.notable_events.len(), 1);
        assert_eq!(rel.notable_events[0].description, "high");
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let manager = RelationshipManager::from_config(RelationshipConfig::default()).unwrap();
        manager.get_or_create("actor_1", None);
        let value = manager.snapshot_as_value().unwrap();
        let restored = RelationshipManager::from_config(RelationshipConfig::default()).unwrap();
        restored.restore_from_value(value).unwrap();
        assert!(restored.get("actor_1").is_some());
    }

    #[test]
    fn test_apply_evaluation_records_event() {
        let manager = RelationshipManager::from_config(RelationshipConfig::default()).unwrap();
        manager.get_or_create("actor_1", None);
        let update = manager
            .apply_evaluation(
                "actor_1",
                RelationshipEvaluation {
                    changes: vec![ProposedDimensionChange {
                        dimension: "trust".to_string(),
                        delta: 0.2,
                        confidence: 0.9,
                        reason: "helpful".to_string(),
                    }],
                    notable_event: Some(ProposedRelationshipEvent {
                        description: "Actor helped resolve a problem".to_string(),
                        significance: 0.8,
                    }),
                },
            )
            .unwrap();
        assert_eq!(update.changes.len(), 1);
        assert!(update.event.is_some());
    }
}
