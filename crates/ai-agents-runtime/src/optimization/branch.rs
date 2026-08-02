use ai_agents_core::{AgentError, Result};
use ai_agents_reasoning::ReasoningMode;
use uuid::Uuid;

use super::maintenance::RuntimeTaskPurpose;
use super::response::MainResponseDraft;
use super::skill::SkillCandidate;
use super::turn::TransitionCandidate;

/// Runtime branch category used for observation and winner selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeOptimizationKind {
    ParallelStateTransition,
    SpeculativeSkillRouting,
    SpeculativeReasoningAuto,
    BufferedStreamingRouting,
}

impl RuntimeOptimizationKind {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::ParallelStateTransition => "parallel_state_transition",
            Self::SpeculativeSkillRouting => "speculative_skill_routing",
            Self::SpeculativeReasoningAuto => "speculative_reasoning_auto",
            Self::BufferedStreamingRouting => "buffered_streaming_routing",
        }
    }
}

/// Describes what can commit if a branch wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeCommitBehavior {
    FinalResponse,
    TransitionDecision,
    SkillSelection,
    ReasoningDecision,
    DiscardOnly,
}

impl RuntimeCommitBehavior {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::FinalResponse => "final_response",
            Self::TransitionDecision => "transition_decision",
            Self::SkillSelection => "skill_selection",
            Self::ReasoningDecision => "reasoning_decision",
            Self::DiscardOnly => "discard_only",
        }
    }
}

/// Lifecycle state for one runtime branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeBranchStatus {
    Scheduled,
    Completed,
    Committed,
    Discarded,
    Failed,
    Cancelled,
}

impl RuntimeBranchStatus {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Scheduled => "scheduled",
            Self::Completed => "completed",
            Self::Committed => "committed",
            Self::Discarded => "discarded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Committed | Self::Discarded | Self::Failed | Self::Cancelled
        )
    }
}

/// Priority hint used when several branch results are available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeTaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Result produced by one speculative branch before commit or discard.
/// Boxing the draft would break frozen public v1 variant construction, so this enum keeps its current representation.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum RuntimeBranchResult {
    MainDraft(MainResponseDraft),
    Transition(Option<TransitionCandidate>),
    Skill(Option<SkillCandidate>),
    Reasoning(ReasoningMode),
    Failed(AgentError),
    Cancelled,
}

/// Metadata and result for one completed runtime branch.
#[derive(Debug)]
pub struct RuntimeBranchOutcome {
    pub branch: RuntimeBranch,
    pub result: RuntimeBranchResult,
}

/// Metadata for one branch scheduled during a turn.
#[derive(Debug, Clone)]
pub struct RuntimeBranch {
    pub id: Uuid,
    pub purpose: RuntimeTaskPurpose,
    pub optimization: RuntimeOptimizationKind,
    pub priority: RuntimeTaskPriority,
    pub commit_behavior: RuntimeCommitBehavior,
    pub status: RuntimeBranchStatus,
}

impl RuntimeBranch {
    pub fn new(
        purpose: RuntimeTaskPurpose,
        optimization: RuntimeOptimizationKind,
        priority: RuntimeTaskPriority,
        commit_behavior: RuntimeCommitBehavior,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            purpose,
            optimization,
            priority,
            commit_behavior,
            status: RuntimeBranchStatus::Scheduled,
        }
    }

    pub fn transition_to(&mut self, next: RuntimeBranchStatus) -> Result<()> {
        if self.status.is_terminal() && self.status != next {
            return Err(AgentError::Other(format!(
                "runtime branch {} cannot move from terminal status {} to {}",
                self.id,
                self.status.as_label(),
                next.as_label()
            )));
        }
        self.status = next;
        Ok(())
    }

    pub fn complete(&mut self) -> Result<()> {
        self.transition_to(RuntimeBranchStatus::Completed)
    }

    pub fn commit(&mut self) -> Result<()> {
        self.transition_to(RuntimeBranchStatus::Committed)
    }

    pub fn discard(&mut self) -> Result<()> {
        self.transition_to(RuntimeBranchStatus::Discarded)
    }

    pub fn fail(&mut self) -> Result<()> {
        self.transition_to(RuntimeBranchStatus::Failed)
    }

    pub fn cancel(&mut self) -> Result<()> {
        self.transition_to(RuntimeBranchStatus::Cancelled)
    }

    pub fn branch_id(&self) -> String {
        self.id.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_branch_status_cannot_change() {
        let mut branch = RuntimeBranch::new(
            RuntimeTaskPurpose::MainResponse,
            RuntimeOptimizationKind::SpeculativeSkillRouting,
            RuntimeTaskPriority::Normal,
            RuntimeCommitBehavior::FinalResponse,
        );
        branch
            .transition_to(RuntimeBranchStatus::Committed)
            .unwrap();
        assert!(
            branch
                .transition_to(RuntimeBranchStatus::Discarded)
                .is_err()
        );
    }

    #[test]
    fn repeated_terminal_status_is_allowed() {
        let mut branch = RuntimeBranch::new(
            RuntimeTaskPurpose::MainResponse,
            RuntimeOptimizationKind::SpeculativeSkillRouting,
            RuntimeTaskPriority::Normal,
            RuntimeCommitBehavior::FinalResponse,
        );
        branch
            .transition_to(RuntimeBranchStatus::Discarded)
            .unwrap();
        assert!(branch.transition_to(RuntimeBranchStatus::Discarded).is_ok());
    }
}
