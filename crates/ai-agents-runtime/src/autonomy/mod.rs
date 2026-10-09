//! Strict autonomy configuration resolution; task execution is added separately.

mod config;

pub use ai_agents_core::autonomy::*;
pub use config::{AutonomyHostCeilings, AutonomyScope, EffectiveAutonomyProfile, resolve_profile};
pub(crate) use config::{has_enabled_declaration, validate_agent_config, validate_loaded_skills};
