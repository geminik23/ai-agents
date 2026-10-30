//! Pure lifecycle and checkpoint decision reducers; execution remains owned by the forthcoming runner.

use super::{GateOutcome, ProgressState};
use ai_agents_core::autonomy::{
    AutonomyProfile, ExecutionStrategy, StagnationAction, TaskRunStatus,
};
use ai_agents_core::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleState {
    pub strategy: ExecutionStrategy,
    pub completed_stages: BTreeSet<String>,
    pub current_stage: usize,
    pub skill_segment_completed: bool,
    pub fix_cycles: u32,
}

impl LifecycleState {
    /// Strategies share the same mandatory stages; skill scripts are executed once, not replayed as continuation.
    pub fn new(profile: &AutonomyProfile, skill_scope: bool) -> Result<Self> {
        let strategy = profile
            .execution
            .as_ref()
            .and_then(|e| e.strategy)
            .unwrap_or(if skill_scope {
                ExecutionStrategy::Skill
            } else if profile.lifecycle.as_ref().is_some_and(|s| !s.is_empty()) {
                ExecutionStrategy::Lifecycle
            } else {
                ExecutionStrategy::ModelLed
            });
        if strategy == ExecutionStrategy::Skill && !skill_scope {
            return Err(AgentError::Config(
                "skill strategy needs skill scope".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for stage in profile.lifecycle.as_deref().unwrap_or_default() {
            if stage.id.is_empty() || !ids.insert(&stage.id) {
                return Err(AgentError::Config(
                    "invalid lifecycle stage identity".into(),
                ));
            }
        }
        Ok(Self {
            strategy,
            completed_stages: BTreeSet::new(),
            current_stage: 0,
            skill_segment_completed: false,
            fix_cycles: 0,
        })
    }
    /// Early task completion cannot skip required stages or the initial scripted skill segment.
    pub fn requirements_met(&self, profile: &AutonomyProfile) -> bool {
        self.validate(profile).is_ok()
            && (self.strategy != ExecutionStrategy::Skill || self.skill_segment_completed)
            && profile
                .lifecycle
                .as_deref()
                .unwrap_or_default()
                .iter()
                .filter(|stage| stage.required.unwrap_or(true))
                .all(|stage| self.completed_stages.contains(&stage.id))
    }
    /// Lifecycle restrictions only narrow existing authority; final executor admission will consume this predicate.
    pub fn allows_tool(&self, profile: &AutonomyProfile, tool: &str) -> bool {
        self.validate(profile).is_ok()
            && !profile
                .lifecycle
                .as_deref()
                .unwrap_or_default()
                .iter()
                .any(|stage| {
                    !self.completed_stages.contains(&stage.id)
                        && stage
                            .required_before_tools
                            .as_ref()
                            .is_some_and(|tools| tools.iter().any(|id| id == tool))
                })
    }
    /// A stage advances only after its own eligible gate and required validations pass.
    pub fn complete_stage(
        &mut self,
        profile: &AutonomyProfile,
        stage_id: &str,
        gate: GateOutcome,
        validations: GateOutcome,
    ) -> Result<bool> {
        let stage = profile
            .lifecycle
            .as_deref()
            .unwrap_or_default()
            .get(self.current_stage)
            .ok_or_else(|| AgentError::Config("lifecycle has no active stage".into()))?;
        if stage.id != stage_id {
            return Err(AgentError::Config("out-of-order stage completion".into()));
        }
        if gate != GateOutcome::Pass || validations != GateOutcome::Pass {
            return Ok(false);
        }
        self.completed_stages.insert(stage.id.clone());
        self.current_stage += 1;
        Ok(true)
    }
    /// A failed validation may revisit an earlier declared stage within the same bounded fix history.
    pub fn validation_failure(&mut self, profile: &AutonomyProfile) -> Result<bool> {
        let stages = profile.lifecycle.as_deref().unwrap_or_default();
        let stage = stages
            .get(self.current_stage)
            .ok_or_else(|| AgentError::Config("no stage for validation failure".into()))?;
        if !stage.retry_on_failure.unwrap_or(false)
            || self.fix_cycles >= stage.max_fix_cycles.unwrap_or(0)
        {
            return Ok(false);
        }
        let target = stage
            .on_failure_stage
            .as_ref()
            .ok_or_else(|| AgentError::Config("missing validation fix target".into()))?;
        let index = stages
            .iter()
            .position(|stage| &stage.id == target)
            .ok_or_else(|| AgentError::Config("unknown validation fix target".into()))?;
        if index >= self.current_stage {
            return Err(AgentError::Config(
                "fix target must precede validation stage".into(),
            ));
        }
        for stage in &stages[index..] {
            self.completed_stages.remove(&stage.id);
        }
        self.current_stage = index;
        self.fix_cycles += 1;
        Ok(true)
    }
    /// Once the scripted segment has completed, subsequent work uses the normal model loop.
    pub fn needs_skill_segment(&self) -> bool {
        self.strategy == ExecutionStrategy::Skill && !self.skill_segment_completed
    }
    /// Restore validates every retained stage identity before execution decisions can consume it.
    pub fn validate(&self, profile: &AutonomyProfile) -> Result<()> {
        let stages = profile.lifecycle.as_deref().unwrap_or_default();
        if stages.iter().enumerate().any(|(index, stage)| {
            self.completed_stages.contains(&stage.id) != (index < self.current_stage)
        }) || self.current_stage > stages.len()
            || self
                .completed_stages
                .iter()
                .any(|id| !stages.iter().any(|s| &s.id == id))
        {
            return Err(AgentError::Config("invalid lifecycle checkpoint".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum CheckpointDecision {
    Terminal {
        status: TaskRunStatus,
        reason: String,
    },
    Wait,
    Replan,
    AskUser,
    Continue,
    Limit,
}

/// Counts exactly at a cap are not violations; unavailable additional work is a separate input.
pub struct DecisionInputs {
    pub uncertain_effect: bool,
    pub cancel_requested: bool,
    pub unrecoverable_error: bool,
    pub actual_time_violation: bool,
    pub mandatory_wait: bool,
    pub gates: GateOutcome,
    pub required_validation: GateOutcome,
    pub children_settled: bool,
    pub scope_ended: bool,
    pub additional_work_capacity: bool,
    pub interaction_available: bool,
}

/// Selects exactly one action in the documented precedence order, before premature-finish continuation.
pub fn checkpoint_decision(
    profile: &AutonomyProfile,
    lifecycle: &LifecycleState,
    progress: &mut ProgressState,
    input: &DecisionInputs,
) -> Result<CheckpointDecision> {
    if progress.decided_sequence == Some(progress.sequence)
        && !input.uncertain_effect
        && !input.cancel_requested
        && !input.unrecoverable_error
        && !input.actual_time_violation
    {
        return Err(AgentError::Config(
            "checkpoint decision already selected".into(),
        ));
    }
    let decision = select_checkpoint_decision(profile, lifecycle, progress, input)?;
    if decision != CheckpointDecision::Wait {
        progress.decided_sequence = Some(progress.sequence);
    }
    Ok(decision)
}

// Terminal precedence and mandatory barriers are evaluated before requesting any new work.
fn select_checkpoint_decision(
    profile: &AutonomyProfile,
    lifecycle: &LifecycleState,
    progress: &mut ProgressState,
    input: &DecisionInputs,
) -> Result<CheckpointDecision> {
    let terminal = |status, reason: &str| CheckpointDecision::Terminal {
        status,
        reason: reason.into(),
    };
    if input.uncertain_effect {
        return Ok(terminal(
            TaskRunStatus::RecoveryRequired,
            "uncertain_effect",
        ));
    }
    if input.cancel_requested {
        return Ok(terminal(TaskRunStatus::Cancelled, "cancel_acknowledged"));
    }
    if input.unrecoverable_error {
        return Ok(terminal(TaskRunStatus::Failed, "execution_error"));
    }
    if input.actual_time_violation {
        return Ok(terminal(TaskRunStatus::LimitReached, "time_violation"));
    }
    if input.mandatory_wait {
        return Ok(CheckpointDecision::Wait);
    }
    let complete = input.gates == GateOutcome::Pass
        && input.required_validation == GateOutcome::Pass
        && input.children_settled
        && lifecycle.requirements_met(profile);
    if complete {
        return Ok(terminal(TaskRunStatus::Completed, "gates_passed"));
    }
    if input.scope_ended {
        return Ok(terminal(TaskRunStatus::Incomplete, "scope_ended"));
    }
    if !input.additional_work_capacity {
        return Ok(CheckpointDecision::Limit);
    }
    let policy = profile
        .progress
        .as_ref()
        .and_then(|p| p.stagnation.as_ref());
    if let Some(action) = progress.stagnation(policy, false, input.interaction_available)? {
        return Ok(match action {
            StagnationAction::Replan => CheckpointDecision::Replan,
            StagnationAction::AskUser => CheckpointDecision::AskUser,
            StagnationAction::Stop => terminal(TaskRunStatus::Incomplete, "no_progress"),
            StagnationAction::Fail => terminal(TaskRunStatus::Failed, "no_progress"),
            StagnationAction::Continue => CheckpointDecision::Continue,
        });
    }
    Ok(CheckpointDecision::Continue)
}
