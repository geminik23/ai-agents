//! Strict autonomy configuration resolution; task execution is added separately.

#[cfg(test)]
mod checkpoint_tests;
mod config;
mod run;
mod store;
#[cfg(test)]
mod tests;
mod todo;

pub use run::*;
pub use store::{ScopedTaskRunStore, TaskRunStore};
pub use todo::RunTodoAdapter;

pub use ai_agents_core::autonomy::*;
pub use config::{AutonomyHostCeilings, AutonomyScope, EffectiveAutonomyProfile, resolve_profile};
pub(crate) use config::{has_enabled_declaration, validate_agent_config, validate_loaded_skills};
