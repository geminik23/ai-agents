use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Definition for one relationship dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipDimensionDefinition {
    /// Human-readable explanation sent to the relationship evaluator.
    pub description: String,
    /// Minimum allowed score for this dimension.
    pub min: f64,
    /// Maximum allowed score for this dimension.
    pub max: f64,
    /// Default score for a newly created relationship.
    pub default: f64,
}

impl RelationshipDimensionDefinition {
    pub fn new(description: impl Into<String>, min: f64, max: f64, default: f64) -> Self {
        Self {
            description: description.into(),
            min,
            max,
            default,
        }
    }

    pub fn clamp(&self, value: f64) -> f64 {
        value.max(self.min).min(self.max)
    }
}

/// Relationship state owned by one agent for one actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Stable actor identifier used for cross-session relationship lookup.
    pub actor_id: String,
    /// Optional display name for the actor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_name: Option<String>,
    /// Current dimension scores keyed by dimension name.
    #[serde(default)]
    pub dimensions: HashMap<String, f64>,
    /// Compact history of significant relationship-relevant events.
    #[serde(default)]
    pub notable_events: Vec<RelationshipEvent>,
    /// Number of interactions observed by this relationship memory.
    #[serde(default)]
    pub interaction_count: u32,
    /// Timestamp of the first recorded interaction.
    pub first_interaction: DateTime<Utc>,
    /// Timestamp of the most recent recorded interaction.
    pub last_interaction: DateTime<Utc>,
    /// Extension metadata for application-specific relationship data.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl Relationship {
    pub fn new(
        actor_id: impl Into<String>,
        actor_name: Option<String>,
        definitions: &HashMap<String, RelationshipDimensionDefinition>,
    ) -> Self {
        let now = Utc::now();
        let dimensions = definitions
            .iter()
            .map(|(name, def)| (name.clone(), def.clamp(def.default)))
            .collect();

        Self {
            actor_id: actor_id.into(),
            actor_name,
            dimensions,
            notable_events: Vec::new(),
            interaction_count: 0,
            first_interaction: now,
            last_interaction: now,
            metadata: HashMap::new(),
        }
    }

    pub fn touch(&mut self, actor_name: Option<&str>) {
        if let Some(name) = actor_name {
            self.actor_name = Some(name.to_string());
        }
        self.interaction_count = self.interaction_count.saturating_add(1);
        self.last_interaction = Utc::now();
    }
}

/// Significant relationship-relevant event kept as compact relationship history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEvent {
    /// Unique event identifier.
    pub event_id: String,
    /// Short event description.
    pub description: String,
    /// Dimension changes associated with this event.
    #[serde(default)]
    pub changes: Vec<DimensionChange>,
    /// Event significance from 0.0 to 1.0.
    pub significance: f64,
    /// Event timestamp.
    pub timestamp: DateTime<Utc>,
    /// Optional future link to an episodic memory record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub episode_id: Option<String>,
    /// Extension metadata for application-specific event data.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl RelationshipEvent {
    pub fn new(
        description: impl Into<String>,
        changes: Vec<DimensionChange>,
        significance: f64,
    ) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            changes,
            significance,
            timestamp: Utc::now(),
            episode_id: None,
            metadata: HashMap::new(),
        }
    }
}

/// Applied change to one relationship dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionChange {
    /// Dimension that changed.
    pub dimension: String,
    /// Score before the change.
    pub previous: f64,
    /// Score after validation and clamping.
    pub current: f64,
    /// Applied delta after validation and clamping.
    pub delta: f64,
    /// Evaluator confidence from 0.0 to 1.0.
    pub confidence: f64,
    /// Short reason for the change.
    pub reason: String,
}

/// Parsed relationship evaluator result before validation is applied.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelationshipEvaluation {
    /// Proposed dimension changes from the evaluator.
    #[serde(default)]
    pub changes: Vec<ProposedDimensionChange>,
    /// Optional notable event proposed by the evaluator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notable_event: Option<ProposedRelationshipEvent>,
}

/// Proposed dimension change returned by the evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedDimensionChange {
    /// Dimension name proposed by the evaluator.
    pub dimension: String,
    /// Proposed score delta before runtime clamping.
    pub delta: f64,
    /// Evaluator confidence from 0.0 to 1.0.
    #[serde(default)]
    pub confidence: f64,
    /// Short reason for the proposed change.
    #[serde(default)]
    pub reason: String,
}

/// Proposed relationship event returned by the evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedRelationshipEvent {
    /// Short event description.
    pub description: String,
    /// Proposed event significance from 0.0 to 1.0.
    pub significance: f64,
}

/// Result of applying a relationship update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipUpdate {
    /// Updated relationship state.
    pub relationship: Relationship,
    /// Dimension changes that were actually applied.
    #[serde(default)]
    pub changes: Vec<DimensionChange>,
    /// Notable event that was actually recorded, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<RelationshipEvent>,
}
