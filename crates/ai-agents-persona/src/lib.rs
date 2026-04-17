//! Agent persona system for structured identity, evolution, and secrets.

pub mod conditions;
pub mod config;
pub mod evolution;
pub mod manager;
pub mod prompt;
pub mod snapshot;
pub mod templates;
pub mod tool;

pub use config::{
    EvolutionConfig, PersonaConfig, PersonaGoals, PersonaIdentity, PersonaSecret,
    PersonaTemplateRef, PersonaTraits, SecretRevealCondition, VALID_EVOLVE_PATHS,
};
pub use evolution::PersonaChange;
pub use manager::{PersonaManager, PersonaRenderResult};
pub use snapshot::PersonaSnapshot;
pub use templates::PersonaTemplateRegistry;
pub use tool::{PERSONA_CHANGE_METADATA_KEY, PersonaEvolveTool};
