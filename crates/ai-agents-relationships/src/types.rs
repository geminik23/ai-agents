use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Relationship semantics used when storing and interpreting relationship scores.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipModel {
    #[default]
    OneSided,
    TwoSided,
}

/// Which side of the relationship a change applies to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipPerspective {
    #[default]
    AgentToActor,
    PerceivedActorToAgent,
    Mutual,
}

impl std::fmt::Display for RelationshipPerspective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgentToActor => write!(f, "agent_to_actor"),
            Self::PerceivedActorToAgent => write!(f, "perceived_actor_to_agent"),
            Self::Mutual => write!(f, "mutual"),
        }
    }
}

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
    /// Create a relationship dimension definition with an explanation and numeric bounds.
    pub fn new(description: impl Into<String>, min: f64, max: f64, default: f64) -> Self {
        Self {
            description: description.into(),
            min,
            max,
            default,
        }
    }

    /// Clamp a score to the configured min/max range.
    pub fn clamp(&self, value: f64) -> f64 {
        value.max(self.min).min(self.max)
    }
}

/// Relationship state owned by one agent for one actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Stable actor identifier used for cross-session relationship lookup.
    pub actor_id: String,
    /// One-sided or two-sided relationship semantics.
    #[serde(default)]
    pub model: RelationshipModel,
    /// Optional display name for the actor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_name: Option<String>,
    /// Current dimension scores keyed by dimension name.
    /// In two-sided mode this remains the canonical `agent_to_actor` perspective.
    #[serde(default)]
    pub dimensions: HashMap<String, f64>,
    /// In two-sided mode, the agent's inferred view of the actor's stance toward the agent.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub perceived_actor_to_agent: HashMap<String, f64>,
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
    /// Create a new relationship initialized from the provided dimension definitions
    /// and relationship model.
    pub fn new(
        actor_id: impl Into<String>,
        actor_name: Option<String>,
        definitions: &HashMap<String, RelationshipDimensionDefinition>,
        model: RelationshipModel,
    ) -> Self {
        let now = Utc::now();
        let dimensions = definitions
            .iter()
            .map(|(name, def)| (name.clone(), def.clamp(def.default)))
            .collect();
        let perceived_actor_to_agent = if matches!(model, RelationshipModel::TwoSided) {
            definitions
                .iter()
                .map(|(name, def)| (name.clone(), def.clamp(def.default)))
                .collect()
        } else {
            HashMap::new()
        };

        Self {
            actor_id: actor_id.into(),
            model,
            actor_name,
            dimensions,
            perceived_actor_to_agent,
            notable_events: Vec::new(),
            interaction_count: 0,
            first_interaction: now,
            last_interaction: now,
            metadata: HashMap::new(),
        }
    }

    /// Return the stored values for one perspective.
    ///
    /// `mutual` is derived, so it returns `None` here and should be accessed through
    /// [`Self::mutual_dimensions`].
    pub fn perspective_values(
        &self,
        perspective: RelationshipPerspective,
    ) -> Option<&HashMap<String, f64>> {
        match perspective {
            RelationshipPerspective::AgentToActor => Some(&self.dimensions),
            RelationshipPerspective::PerceivedActorToAgent => {
                if matches!(self.model, RelationshipModel::TwoSided) {
                    Some(&self.perceived_actor_to_agent)
                } else {
                    None
                }
            }
            RelationshipPerspective::Mutual => None,
        }
    }

    /// Return mutable access to one stored perspective.
    ///
    /// `mutual` is derived, so it does not expose a mutable backing map.
    pub fn perspective_values_mut(
        &mut self,
        perspective: RelationshipPerspective,
    ) -> Option<&mut HashMap<String, f64>> {
        match perspective {
            RelationshipPerspective::AgentToActor => Some(&mut self.dimensions),
            RelationshipPerspective::PerceivedActorToAgent => {
                if matches!(self.model, RelationshipModel::TwoSided) {
                    Some(&mut self.perceived_actor_to_agent)
                } else {
                    None
                }
            }
            RelationshipPerspective::Mutual => None,
        }
    }

    /// Compute the derived mutual view for this relationship.
    ///
    /// In one-sided mode this is equivalent to `agent_to_actor`.
    pub fn mutual_dimensions(&self) -> HashMap<String, f64> {
        if !matches!(self.model, RelationshipModel::TwoSided) {
            return self.dimensions.clone();
        }

        let mut merged = HashMap::new();
        for (name, value) in &self.dimensions {
            let other = self
                .perceived_actor_to_agent
                .get(name)
                .copied()
                .unwrap_or(*value);
            merged.insert(name.clone(), (value + other) / 2.0);
        }
        for (name, value) in &self.perceived_actor_to_agent {
            merged.entry(name.clone()).or_insert(*value);
        }
        merged
    }

    /// Update timestamps, interaction count, and optional display name after an
    /// observed interaction.
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
    /// Create a new notable relationship event.
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
    /// Perspective that changed.
    #[serde(default)]
    pub perspective: RelationshipPerspective,
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
    /// Perspective proposed by the evaluator.
    ///
    /// This is intentionally required during deserialization so evaluator output cannot silently default to `agent_to_actor` in two-sided mode.
    pub perspective: RelationshipPerspective,
    /// Dimension name proposed by the evaluator.
    pub dimension: String,
    /// Proposed score delta before runtime clamping.
    pub delta: f64,
    /// Evaluator confidence from 0.0 to 1.0.
    ///
    /// This is intentionally required during deserialization so missing confidence cannot silently become `0.0` and disappear during filtering.
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
