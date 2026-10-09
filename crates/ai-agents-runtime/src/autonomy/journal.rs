//! Conditional storage bridge for exact validator continuation, not a separate task persistence engine.

use super::*;
use ai_agents_core::autonomy::{TaskRunMutation, TaskRunStatus};
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// One validation owner journals under the existing task claim and shares its latest revision with execution evidence.
pub struct TaskValidationJournal {
    store: Arc<dyn TaskRunStore>,
    run_id: String,
    owner_token: String,
    revision: Arc<AtomicU64>,
    serial: Arc<tokio::sync::Mutex<()>>,
    invocation_accounting: bool,
}

impl TaskValidationJournal {
    /// The caller retains the actual task owner; this adapter never steals or refreshes a claim.
    pub fn new(
        store: Arc<ScopedTaskRunStore>,
        run_id: String,
        owner_token: String,
        revision: u64,
    ) -> Self {
        Self {
            store,
            run_id,
            owner_token,
            revision: Arc::new(AtomicU64::new(revision)),
            serial: Arc::new(tokio::sync::Mutex::new(())),
            invocation_accounting: false,
        }
    }
    /// The controller journal shares revision and serialization with actual provider/tool admission.
    pub(crate) fn for_execution(execution: &Arc<RunExecution>) -> Self {
        Self {
            store: execution.store.clone(),
            run_id: execution.run_id.clone(),
            owner_token: execution.owner_token.clone(),
            revision: execution.revision.clone(),
            serial: execution.serial.clone(),
            invocation_accounting: true,
        }
    }

    /// Invocation evidence uses the acknowledged revision, not a stale attempt-start revision.
    pub fn revision_handle(&self) -> Arc<AtomicU64> {
        self.revision.clone()
    }
}

// Unambiguous run/check/attempt/observation identity is retained across checkpoint revisions.
fn observation_reservation_id(state: &ValidationDriverState, id: &str) -> Result<String> {
    Ok(format!(
        "validation:{}",
        canonical_identity(
            &serde_json::json!({"check":state.check_id,"attempt":state.identity.attempt,"observation":id})
        )?
    ))
}

#[async_trait]
impl ValidationJournal for TaskValidationJournal {
    /// Checkpoint failure stops before dispatch, while a saved in-flight record remains crash-uncertain.
    async fn checkpoint(&self, state: &ValidationDriverState) -> Result<()> {
        let _serial = self.serial.lock().await;
        let execution = super::current_execution()
            .filter(|run| self.invocation_accounting && Arc::ptr_eq(&run.revision, &self.revision));
        if let Some(execution) = &execution {
            execution.load_owned().await?;
        }
        let current = self
            .store
            .load(&self.run_id)
            .await?
            .ok_or_else(|| AgentError::Persistence("task journal run missing".into()))?;
        if current.revision != self.revision.load(Ordering::Acquire)
            || current.status != TaskRunStatus::Running
            || current.owner_token.as_deref() != Some(&self.owner_token)
            || (current.cancel_requested
                && (!self.invocation_accounting || state.in_flight.is_some()))
            || current.key != state.identity.key
        {
            return Err(AgentError::Persistence(
                "task journal owner/revision/cancellation conflict".into(),
            ));
        }
        let mut payload: TaskCheckpointPayload = serde_json::from_value(current.payload.clone())?;
        if payload.objective_revision != state.identity.objective_revision {
            return Err(AgentError::Config(
                "validation objective revision changed".into(),
            ));
        }
        let checkpoint = TaskAdapterCheckpoint {
            id: state.check_id.clone(),
            adapter: state.adapter.clone(),
            contract_version: state.contract_version,
            config: serde_json::from_str(&state.identity.config_identity).map_err(|_| {
                AgentError::Config("invalid exact validator config identity".into())
            })?,
            state: serde_json::json!({"_task_validation_driver":state}),
        };
        if let Some(existing) = payload.adapters.iter_mut().find(|a| a.id == state.check_id) {
            if existing.adapter != checkpoint.adapter
                || existing.contract_version != checkpoint.contract_version
                || existing.config != checkpoint.config
            {
                return Err(AgentError::Config("journal adapter binding changed".into()));
            }
            if let Some(old) = existing.state.get("_task_validation_driver") {
                let old: ValidationDriverState = serde_json::from_value(old.clone())?;
                state.validate_journal_successor(&old)?;
            }
            *existing = checkpoint;
        } else {
            payload.adapters.push(checkpoint);
        }
        if let Some(id) = &state.in_flight {
            let reservation_id = observation_reservation_id(state, id)?;
            if payload.reservations.iter().any(|r| r.id == reservation_id)
                || payload
                    .reservations
                    .iter()
                    .any(|r| r.state == TaskEffectState::Uncertain)
            {
                return Err(AgentError::Config(
                    "validation reservation already admitted or requires recovery".into(),
                ));
            }
            {
                let request = state.requests.get(id).ok_or_else(|| {
                    AgentError::Config("journal observation request missing".into())
                })?;
                if !self.invocation_accounting {
                    match request {
                        ValidationObservationRequest::Tool { tool, .. } => {
                            payload.counters.tool_attempts = payload
                                .counters
                                .tool_attempts
                                .checked_add(1)
                                .ok_or_else(|| {
                                    AgentError::Config("tool counter overflow".into())
                                })?;
                            if payload.counters.tool_attempts
                                > u64::from(payload.limits.max_tool_calls)
                            {
                                return Err(AgentError::Config(
                                    "validation tool capacity exhausted".into(),
                                ));
                            }
                            if tool == "command" {
                                payload.counters.command_attempts = payload
                                    .counters
                                    .command_attempts
                                    .checked_add(1)
                                    .ok_or_else(|| {
                                        AgentError::Config("command counter overflow".into())
                                    })?;
                                if payload.counters.command_attempts
                                    > u64::from(payload.limits.max_command_calls)
                                {
                                    return Err(AgentError::Config(
                                        "validation command capacity exhausted".into(),
                                    ));
                                }
                            }
                        }
                        ValidationObservationRequest::Judge { .. }
                        | ValidationObservationRequest::Plan { .. } => {
                            payload.counters.llm_attempts = payload
                                .counters
                                .llm_attempts
                                .checked_add(1)
                                .ok_or_else(|| AgentError::Config("LLM counter overflow".into()))?;
                            if payload.counters.llm_attempts
                                > u64::from(payload.limits.max_llm_calls)
                            {
                                return Err(AgentError::Config(
                                    "validation judge capacity exhausted".into(),
                                ));
                            }
                        }
                    }
                }
                payload.reservations.push(TaskReservation {
                    id: reservation_id,
                    state: TaskEffectState::Dispatched,
                    reserved_micro_usd: 0,
                    charged_micro_usd: 0,
                    write_targets: vec![],
                    result: None,
                });
            }
        }
        for (id, completed) in &state.completed {
            let reservation_id = observation_reservation_id(state, id)?;
            if let Some(reservation) = payload
                .reservations
                .iter_mut()
                .find(|r| r.id == reservation_id)
            {
                reservation.state = match &completed.result {
                    ObservationResult::Tool { record }
                        if record.executed
                            && (record.cancelled || record.timed_out)
                            && record
                                .metadata
                                .get("classification")
                                .and_then(|c| c.get("read_only"))
                                .and_then(serde_json::Value::as_bool)
                                != Some(true) =>
                    {
                        TaskEffectState::Uncertain
                    }
                    _ => TaskEffectState::Completed,
                };
                reservation.result = Some(serde_json::to_value(&completed.result)?);
            } else {
                return Err(AgentError::Config(
                    "completed observation has no admitted reservation".into(),
                ));
            }
        }
        if let Some(wait) = &state.wait {
            if payload
                .pending
                .as_ref()
                .is_some_and(|p| p.id != wait.request_id)
            {
                return Err(AgentError::Config(
                    "another pending request owns the task".into(),
                ));
            }
            if payload.pending.is_none() {
                payload.pending = Some(TaskPendingRequest {
                    id: wait.request_id.clone(),
                    issued_revision: current.revision + 1,
                    kind: TaskPendingKind::ExternalCondition,
                    reviewed_action: serde_json::to_value(wait)?,
                });
            }
        } else if let Some(pending) = &payload.pending
            && matches!(pending.kind, TaskPendingKind::ExternalCondition)
            && pending
                .reviewed_action
                .get("check_id")
                .and_then(serde_json::Value::as_str)
                == Some(&state.check_id)
            && pending.reviewed_action.get("identity")
                == serde_json::to_value(&state.identity).ok().as_ref()
        {
            payload.consumed_request_ids.push(pending.id.clone());
            payload.pending = None;
        }
        let saved = self
            .store
            .mutate(
                &self.run_id,
                &TaskRunMutation::Checkpoint {
                    expected_revision: current.revision,
                    owner_token: self.owner_token.clone(),
                    status: TaskRunStatus::Running,
                    payload: serde_json::to_value(payload)?,
                    release: false,
                },
            )
            .await?;
        self.revision.store(saved.revision, Ordering::Release);
        if let Some(execution) = &execution {
            execution.acknowledge_snapshot(&saved);
        }
        Ok(())
    }
}
