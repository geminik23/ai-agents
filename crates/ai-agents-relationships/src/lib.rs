//! Actor-scoped relationship memory for AI Agents.

pub mod config;
pub mod defaults;
pub mod evaluator;
pub mod injection;
pub mod manager;
pub mod snapshot;
pub mod types;

pub use config::{
    AutoUpdateConfig, EventEvictionStrategy, InjectionConfig, InjectionFormat, NotableEventsConfig,
    PersistenceConfig, RelationshipConfig, RelationshipDimensionsConfig,
};
pub use evaluator::{RelationshipEvaluator, RelationshipEvaluatorTrait};
pub use injection::{format_relationship, relationship_to_context_value};
pub use manager::RelationshipManager;
pub use snapshot::{RelationshipSnapshot, relationship_from_value, relationship_to_value};
pub use types::{
    DimensionChange, ProposedDimensionChange, ProposedRelationshipEvent, Relationship,
    RelationshipDimensionDefinition, RelationshipEvaluation, RelationshipEvent, RelationshipUpdate,
};
