//! Strict autonomy configuration resolution; task execution is added separately.

mod authority;
mod builtins;
#[cfg(test)]
mod checkpoint_tests;
mod config;
mod control;
mod journal;
mod lifecycle;
pub use authority::{HostValidationContext, RuntimeObservationExecutor, validation_judge_provider};
pub(crate) use authority::{
    invoke_bound, validation_arguments_valid, validation_extra_grant, validation_metadata,
    validation_requires_approval,
};
pub use control::HostControlAction;
pub(crate) use control::prepare_host_control;
pub use journal::TaskValidationJournal;
pub use lifecycle::*;
#[cfg(test)]
mod driver_regression_tests;
#[cfg(test)]
mod evaluation_tests;
mod evidence;
mod extensions;
mod gate;
#[cfg(test)]
mod integration_tests;
mod observations;
mod progress;
#[cfg(test)]
mod registry_tests;
mod replanning;
mod run;
mod validation;

pub use evidence::*;
pub use extensions::*;
pub use gate::*;
pub use progress::*;
pub use validation::*;
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
