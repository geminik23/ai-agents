//! Exact post-turn controller positions are independent of model tool batches and root-turn finalization.

use super::*;
use ai_agents_core::{AgentError, AgentResponse, Result};
use serde::{Deserialize, Serialize};

/// A position identifies controller-owned work, not another executor or a saved Rust call stack.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ControllerPhase {
    #[default]
    EachCycle,
    StageEnd,
    Completion,
    Decision,
    Replan,
}

/// Saved validation work resumes after the completed turn, not through another runtime response driver.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ControllerContinuation {
    pub scope: EvaluationScope,
    pub lifecycle: LifecycleState,
    pub progress: ProgressState,
    pub evidence: EvaluationEvidence,
    pub response: AgentResponse,
    pub instruction: String,
    pub source: AutonomyTurnSource,
    pub forced_continuation: bool,
    pub checks: Vec<String>,
    pub next_check: usize,
    #[serde(default)]
    pub phase: ControllerPhase,
    pub driver: Option<Box<ValidationDriverState>>,
    #[serde(default)]
    pub review: Option<ControllerReview>,
}

/// A frozen host decision returns to one controller boundary, never an obsolete tool or skill cursor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ControllerReview {
    pub id: String,
    pub proposal: ControllerProposal,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub state_generation: Option<u64>,
    pub control_binding: (u64, u64),
}

/// Stage and plan proposals are advisory until approved; Continue carries no execution permission from model data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ControllerProposal {
    Continue,
    Stage {
        lifecycle: LifecycleState,
    },
    Plan {
        instruction: String,
    },
    Question {
        reason: String,
        prompt: String,
        limit: Option<Box<ControllerLimit>>,
    },
    Capacity {
        limit: Box<ControllerLimit>,
    },
}

impl ControllerProposal {
    /// Only stage and plan adoption consume approval authority; questions and capacity waits carry no execution grant.
    pub(crate) fn supervision(&self) -> bool {
        matches!(self, Self::Stage { .. } | Self::Plan { .. })
    }
}

/// A capacity wait retains the exhausted ceiling; Continue is not a resource grant or an implicit limit revision.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ControllerLimit {
    pub reason: String,
    pub original: TaskRunLimits,
    pub resume_turn: bool,
    pub skill: Option<Box<super::location::TaskSkillLocation>>,
    pub model: Option<Box<super::TaskLoopState>>,
    pub tool: Option<String>,
}

/// Nested group requests retain the leaf's typed capacity requirement rather than trusting the outer pending label.
pub(crate) fn pending_capacity(pending: &TaskPendingRequest) -> Result<ControllerLimit> {
    let mut value = &pending.reviewed_action;
    for _ in 0..32 {
        if let Some(limit) = value
            .get("trigger")
            .and_then(|trigger| trigger.get("capacity"))
        {
            return Ok(serde_json::from_value(limit.clone())?);
        }
        value = value
            .get("request")
            .and_then(|request| request.get("reviewed_action"))
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
    }
    Err(TaskRunStorageError::InvalidCheckpoint.into())
}

impl ControllerLimit {
    /// Checks the exact exhausted resource before consuming the pending request; final invocation admission remains authoritative.
    pub(crate) fn available(&self, payload: &TaskCheckpointPayload) -> bool {
        let limits = &payload.limits;
        match self.reason.as_str() {
            "turn_capacity" => u64::from(limits.max_turns) > payload.counters.turns,
            "invocation_capacity" | "llm_capacity" => {
                u64::from(limits.max_llm_calls) > payload.counters.llm_attempts
                    && (self.resume_turn || u64::from(limits.max_turns) > payload.counters.turns)
            }
            "tool_capacity" => {
                u64::from(limits.max_tool_calls) > payload.counters.tool_attempts
                    && (self.tool.as_deref() != Some("command")
                        || u64::from(limits.max_command_calls) > payload.counters.command_attempts)
            }
            "priced_capacity" => limits.max_micro_usd > self.original.max_micro_usd,
            "write_capacity" => {
                limits.max_declared_write_paths > self.original.max_declared_write_paths
            }
            _ => false,
        }
    }
}

impl ControllerContinuation {
    /// Captures a controller-only wait after a decision without inventing a validator or a model-tool request.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decision(
        scope: &EvaluationScope,
        lifecycle: &LifecycleState,
        progress: &ProgressState,
        evidence: &EvaluationEvidence,
        response: Option<&AgentResponse>,
        instruction: &str,
        source: AutonomyTurnSource,
        forced_continuation: bool,
    ) -> Self {
        Self {
            scope: scope.clone(),
            lifecycle: lifecycle.clone(),
            progress: progress.clone(),
            evidence: evidence.clone(),
            response: response.cloned().unwrap_or_else(|| AgentResponse::new("")),
            instruction: instruction.into(),
            source,
            forced_continuation,
            checks: Vec::new(),
            next_check: 0,
            phase: ControllerPhase::Decision,
            driver: None,
            review: None,
        }
    }

    /// Reconstructs only the check selected at this exact position; dynamic replans retain their intervention identity.
    pub(crate) fn driver_check(
        &self,
        bound: &BoundAutonomyProfile,
        objective: &str,
    ) -> Result<BoundValidationCheck> {
        let driver = self
            .driver
            .as_ref()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let check = if self.phase == ControllerPhase::Replan {
            bound.replan_check(&self.progress, objective)?
        } else {
            bound
                .checks
                .iter()
                .find(|check| check.check.id == driver.check_id)
                .cloned()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
        };
        if check.check.id != driver.check_id
            || check.config_identity != driver.identity.config_identity
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        Ok(check)
    }

    /// Structural checks bind a pending wait to the exact enclosing run and controller position.
    pub(crate) fn validate(&self, payload: &TaskCheckpointPayload) -> Result<()> {
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        let pending = payload.pending.as_ref().ok_or_else(invalid)?;
        if self.phase == ControllerPhase::Replan
            && (self.checks.len() != 1
                || self.next_check != 0
                || self.driver.as_ref().is_none_or(|driver| {
                    driver.check_id != format!("__replan.{}", self.progress.sequence)
                })
                || self.progress.decided_sequence != Some(self.progress.sequence)
                || self.progress.handled_sequence != Some(self.progress.sequence))
        {
            return Err(invalid());
        }
        if let Some(review) = &self.review {
            self.lifecycle.validate(&payload.settings)?;
            if (self.driver.is_some()
                && !matches!(
                    review.proposal,
                    ControllerProposal::Capacity { .. }
                        | ControllerProposal::Question { limit: Some(_), .. }
                ))
                || review.id != pending.id
                || review.id.is_empty()
                || pending.reviewed_action != serde_json::to_value(review)?
                || self.scope.objective_revision != payload.objective_revision
                || !matches!(payload.runtime.continuation, TaskContinuation::BetweenTurns)
                || !matches!(
                    (&review.proposal, pending.kind),
                    (ControllerProposal::Continue, TaskPendingKind::HostContinue)
                        | (
                            ControllerProposal::Stage { .. } | ControllerProposal::Plan { .. },
                            TaskPendingKind::Approval
                        )
                        | (
                            ControllerProposal::Question { .. },
                            TaskPendingKind::UserQuestion
                        )
                        | (ControllerProposal::Capacity { .. }, TaskPendingKind::Limit)
                )
            {
                return Err(invalid());
            }
            if let Some(driver) = &self.driver
                && (self.checks.get(self.next_check) != Some(&driver.check_id)
                    || driver.capacity_reason.is_none()
                    || driver.in_flight.is_some()
                    || driver.identity.key != self.scope.key
                    || driver.identity.attempt.as_ref()
                        != self.scope.validation_attempts.get(&driver.check_id))
            {
                return Err(invalid());
            }
            if let ControllerProposal::Stage { lifecycle } = &review.proposal {
                lifecycle.validate(&payload.settings)?;
            }
            return bounded_value(&serde_json::to_value(self)?, MAX_TASK_CHECKPOINT_BYTES);
        }
        let driver = self.driver.as_ref().ok_or_else(invalid)?;
        if self.checks.get(self.next_check) != Some(&driver.check_id)
            || self.scope.key != driver.identity.key
            || self.scope.objective_revision != payload.objective_revision
            || self.scope.cycle != driver.identity.cycle
            || self.scope.stage != driver.identity.stage
            || self
                .evidence
                .current
                .as_ref()
                .is_none_or(|current| current.value.response != self.response.content)
            || self.scope.validation_attempts.get(&driver.check_id)
                != driver.identity.attempt.as_ref()
            || !matches!(payload.runtime.continuation, TaskContinuation::BetweenTurns)
        {
            return Err(invalid());
        }
        if let Some(wait) = &driver.wait {
            if !matches!(pending.kind, TaskPendingKind::ExternalCondition)
                || pending.id != wait.request_id
                || pending.reviewed_action != serde_json::to_value(wait)?
            {
                return Err(invalid());
            }
        } else if let Some(parked) = &driver.parked {
            if let Some(group) = &parked.group {
                let group: Box<TaskGroupState> = serde_json::from_value(group.clone())?;
                super::authority::validate_observation_group(&group, payload, pending)?;
            }
            if pending.id != parked.pending.id
                || serde_json::to_value(pending)? != serde_json::to_value(&parked.pending)?
            {
                return Err(invalid());
            }
        } else {
            return Err(invalid());
        }
        bounded_value(&serde_json::to_value(self)?, MAX_TASK_CHECKPOINT_BYTES)
    }
}

/// Retires executable progress without deleting the adapter's immutable configuration binding.
pub(crate) fn retire_controller(payload: &mut TaskCheckpointPayload) -> Result<()> {
    if let Some(adapter) = payload
        .adapters
        .iter_mut()
        .find(|a| a.id == "runtime.validation")
    {
        if adapter.state.get("retired_request").is_some() {
            return Ok(());
        }
        let state: ControllerContinuation = serde_json::from_value(adapter.state.clone())?;
        let id = if let Some(review) = &state.review {
            review.id.clone()
        } else {
            let driver = state
                .driver
                .as_ref()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            driver
                .wait
                .as_ref()
                .map(|wait| wait.request_id.clone())
                .or_else(|| {
                    driver
                        .parked
                        .as_ref()
                        .map(|parked| parked.pending.id.clone())
                })
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
        };
        if !payload.consumed_request_ids.contains(&id) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        adapter.state = serde_json::json!({"retired_request":id});
    }
    Ok(())
}

/// Only one controller position exists for a run; it is never inferred from a pending response label.
pub(crate) fn controller_checkpoint(
    payload: &TaskCheckpointPayload,
) -> Result<Option<Box<ControllerContinuation>>> {
    payload
        .adapters
        .iter()
        .find(|adapter| adapter.id == "runtime.validation")
        .map(|adapter| {
            if adapter.adapter != "runtime.validation" || adapter.contract_version != 1 {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            if let Some(id) = adapter
                .state
                .get("retired_request")
                .and_then(serde_json::Value::as_str)
            {
                if !payload
                    .consumed_request_ids
                    .iter()
                    .any(|consumed| consumed == id)
                {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
                return Ok(None);
            }
            let state: Box<ControllerContinuation> = serde_json::from_value(adapter.state.clone())?;
            state.validate(payload)?;
            Ok(Some(state))
        })
        .transpose()
        .map(Option::flatten)
}
