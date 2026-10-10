//! Development autonomy primitives and guarded standalone execution; full task integration remains unfinished.

#[cfg(test)]
mod admission_tests;
mod authority;
mod boundary;
pub(crate) mod composition;
mod execution;
mod memory;
mod participants;
mod targets;
pub(crate) use composition::{
    DelegateFrame, TaskGroupState, current_child_invocation, scope_child_invocation,
};
pub(crate) use participants::{
    Participants, child_required, current_child_operation, scope_child_operation,
    scope_child_requirement,
};
pub(crate) use targets::CompositionTargets;
mod runner;
pub(crate) mod suspension;
#[cfg(test)]
mod suspension_tests;
pub use suspension::TaskResumeInput;
pub(crate) use suspension::{TaskBatchState, TaskLoopState, scope_task_batch, scope_task_request};
#[cfg(test)]
mod runner_additional_tests;
#[cfg(test)]
mod runner_tests;
pub(crate) use boundary::{
    AutonomyTurnInput, AutonomyTurnSource, OwnedTurnCleanup, RunOwner, RunOwnerSlot,
    current_turn_input, scope_turn,
};
pub(crate) use execution::{
    RunExecution, current_execution, scope_execution, scope_inherited_execution,
};
pub(crate) use memory::TaskMemory;
pub use runner::AutonomyRunner;
mod builtins;
#[cfg(test)]
mod checkpoint_tests;
#[cfg(test)]
mod child_tests;
#[cfg(test)]
mod composition_tests;
mod config;
#[cfg(test)]
mod continuation_tests;
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
