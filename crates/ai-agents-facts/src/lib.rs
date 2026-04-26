//! Key facts extraction and actor memory for AI Agents framework.

pub mod config;
pub mod dedup;
pub mod extractor;
pub mod store;

pub use config::{
    ActorMemoryConfig, CategoryDefinition, DedupConfig, DedupMethod, FactsConfig,
    IdentificationConfig, IdentificationMethod, InjectionConfig, InjectionMode, PrivacyConfig,
    SessionConfig,
};
pub use dedup::deduplicate_exact;
pub use extractor::{FactExtractor, LLMFactExtractor};
pub use store::FactStore;

// Re-export core types for convenience.
pub use ai_agents_core::{FactCategory, FactFilter, KeyFact};
