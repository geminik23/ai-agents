//! Development autonomy primitives and guarded standalone execution; full task integration remains unfinished.

#[cfg(test)]
mod ad_maintenance_fault_tests;
#[cfg(test)]
mod admission_tests;
mod authority;
mod boundary;
pub(crate) mod composition;
mod execution;
pub(crate) mod speculation;
pub(crate) mod work;
pub(crate) use work::owned_maintenance;
pub(crate) mod invocation;
mod memory;
pub(crate) mod message;
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
pub use runner::TaskRestoreParticipant;
pub use runner::{
    AutonomyRunner, TaskReconstructionCheckpoint, TaskRestorationReview, TaskRestoreBinding,
    TaskRestoreComposition, TaskRestoreEntry, TaskRestoreGenerations, TaskRestoreGraph,
    TaskRestoreHost, TaskRestoreManifest, TaskRestoreNode, TaskRestoreRegistry, TaskRestoreRequest,
    TaskRestoreResponse, TaskRestoreTarget,
};
mod builtins;
#[cfg(test)]
mod checkpoint_tests;
#[cfg(test)]
pub(crate) mod child_tests;
#[cfg(test)]
mod co_completion_tests;
#[cfg(test)]
mod co_generated_tests;
#[cfg(test)]
mod co_routing_tests;
#[cfg(test)]
mod composition_tests;
mod config;
#[cfg(test)]
pub(crate) mod continuation_tests;
mod control;
mod controller;
#[cfg(test)]
mod controller_failure_tests;
#[cfg(test)]
mod controller_tests;
#[cfg(test)]
mod restoration_failure_tests;
#[cfg(test)]
mod restoration_test_support;
#[cfg(test)]
mod restoration_tests;
pub(crate) use controller::{ControllerContinuation, controller_checkpoint, retire_controller};
pub(crate) mod interaction;
#[cfg(test)]
mod interaction_tests;
mod journal;
#[cfg(test)]
mod journal_resume_tests;
mod lifecycle;
pub(crate) mod location;
#[cfg(test)]
mod location_tests;
#[cfg(test)]
mod lp_controller_tests;
#[cfg(test)]
mod message_batch_tests;
#[cfg(test)]
mod shared_target_tests;
#[cfg(test)]
mod tool_composition_tests;
pub use authority::{HostValidationContext, RuntimeObservationExecutor, validation_judge_provider};
pub(crate) use authority::{
    invoke_bound, scope_message_recovery_authority, validation_arguments_valid,
    validation_extra_grant, validation_metadata, validation_requires_approval,
};
pub use control::HostControlAction;
pub(crate) use control::prepare_host_control;
pub use journal::TaskValidationJournal;
pub use lifecycle::*;
#[cfg(test)]
pub(crate) mod continuation_test_support;
#[cfg(test)]
mod driver_regression_tests;
#[cfg(test)]
mod evaluation_tests;
mod evidence;
mod extensions;
mod host_evidence;
#[cfg(test)]
mod host_evidence_tests;
#[cfg(test)]
mod retained_admission_tests;
#[cfg(test)]
mod retained_boundary_tests;
#[cfg(test)]
mod retained_proof_tests;
pub use host_evidence::{ExternalUnavailablePolicy, HostRevisionSource};
#[cfg(test)]
mod fallback_acceptance_tests;
mod gate;
#[cfg(test)]
mod host_control_tests;
#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod model_consumer_tests;
#[cfg(test)]
mod model_dependency_tests;
#[cfg(test)]
mod observation_message_tests;
mod observations;
#[cfg(test)]
mod paused_limit_tests;
mod progress;
#[cfg(test)]
mod registry_tests;
mod replanning;
pub(crate) mod response;
mod run;
#[cfg(test)]
mod state_acceptance_tests;
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
