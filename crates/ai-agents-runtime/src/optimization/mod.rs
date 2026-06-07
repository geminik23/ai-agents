//! Runtime optimization support.

pub mod branch;
pub mod config;
pub mod maintenance;
pub mod observability;
pub mod reasoning;
pub mod response;
pub mod scheduler;
pub mod skill;
pub mod streaming;
pub mod transition;
mod turn;

pub use branch::{
    RuntimeBranch, RuntimeBranchOutcome, RuntimeBranchResult, RuntimeBranchStatus,
    RuntimeCommitBehavior, RuntimeOptimizationKind, RuntimeTaskPriority,
};
pub use config::{
    AwaitBeforeNextTurn, BackgroundOverflowPolicy, MaintenanceMode, MaintenanceTaskPolicy,
    PostTurnOptimizationConfig, RuntimeConfig, RuntimeOptimizationConfig,
    StreamingOptimizationPolicy,
};
pub use maintenance::{BackgroundMaintenanceQueue, MaintenanceSequenceKey, RuntimeTaskPurpose};
pub use reasoning::ReasoningBranchResult;
pub use response::MainResponseDraft;
pub use scheduler::{BranchFuture, ScheduledBranchSet, TurnBranchScheduler};
pub use skill::SkillCandidate;
pub use streaming::{StreamBranchBuffer, StreamingDraftResult};
pub use transition::TransitionBranchResult;
pub use turn::{TransitionCandidate, TurnOptimizationContext};
