//! Exact post-turn controller positions are independent of model tool batches and root-turn finalization.

use super::*;
use ai_agents_core::{AgentError, AgentResponse, Result};
use serde::{Deserialize, Serialize};

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
    Stage { lifecycle: LifecycleState },
    Plan { instruction: String },
}

impl ControllerContinuation {
    /// Structural checks bind a pending wait to the exact enclosing run and controller position.
    pub(crate) fn validate(&self, payload: &TaskCheckpointPayload) -> Result<()> {
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        let pending = payload.pending.as_ref().ok_or_else(invalid)?;
        if let Some(review) = &self.review {
            self.lifecycle.validate(&payload.settings)?;
            if self.driver.is_some()
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
                )
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
