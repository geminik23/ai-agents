//! Foreground standalone execution using the existing runtime, conditional store and pure decision reducers.

use super::*;
use ai_agents_core::{AgentError, AgentResponse, Result, ToolCancellationToken};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};

/// Development API for explicit host invocation; it does not enable automatic state/skill task entry.
pub struct AutonomyRunner {
    agent: Arc<crate::RuntimeAgent>,
    config: AutonomyConfig,
    host: AutonomyHostCeilings,
    store: Arc<dyn TaskRunStore>,
    config_identity: String,
    active: parking_lot::Mutex<Option<Arc<RunExecution>>>,
    paused: parking_lot::Mutex<BTreeMap<String, PausedTask>>,
    resume_gate: tokio::sync::Mutex<()>,
}

struct PausedTask {
    owner: Arc<RunOwner>,
    participants: Arc<super::participants::Participants>,
    targets: Arc<CompositionTargets>,
    todo: Option<RunTodoAdapter>,
    batch: Option<TaskBatchState>,
    group: Option<TaskGroupState>,
    whole_expiry: Option<std::time::Instant>,
    approval_deadlines: BTreeMap<String, std::time::Instant>,
}

impl Drop for PausedTask {
    /// Losing the host's retained continuation is abandonment, never implicit ownership transfer.
    fn drop(&mut self) {
        self.owner.abandon();
    }
}

impl AutonomyRunner {
    /// Retains only the exact acknowledged cursor that agrees with private live batch and composition authority.
    fn capture_pause(
        &self,
        execution: &RunExecution,
    ) -> Result<(Option<TaskBatchState>, Option<TaskGroupState>)> {
        let snapshot = execution.acknowledged_snapshot();
        if snapshot.status != TaskRunStatus::Paused {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        if matches!(
            payload.runtime.continuation,
            TaskContinuation::Suspended { batch: Some(_), .. }
        ) {
            let batch = execution
                .parked_batches
                .lock()
                .get(&self.agent.info().id)
                .cloned()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            batch.validate_checkpoint(&payload)?;
            Ok((Some(batch), None))
        } else {
            let group: TaskGroupState = serde_json::from_value(
                payload
                    .adapters
                    .iter()
                    .find(|adapter| adapter.id == "runtime.group")
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                    .state
                    .clone(),
            )?;
            group.validate_checkpoint(&payload)?;
            if group.parked.iter().any(|leaf| {
                execution
                    .parked_batches
                    .lock()
                    .get(&leaf.runtime_id)
                    .is_none_or(|batch| {
                        serde_json::to_value(batch).ok() != serde_json::to_value(&leaf.batch).ok()
                    })
            }) || execution
                .parked_batches
                .lock()
                .get(&group.child_runtime_id)
                .is_none_or(|batch| {
                    serde_json::to_value(batch).ok() != serde_json::to_value(&group.batch).ok()
                })
                || serde_json::to_value(execution.delegate_frames.lock().clone())?
                    != serde_json::to_value(&group.frames)?
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            Ok((None, Some(group)))
        }
    }

    /// Refreshes trusted parked state and non-extending request deadlines after another acknowledged safe pause.
    fn refresh_pause(&self, paused: &mut PausedTask, execution: &RunExecution) -> Result<()> {
        (paused.batch, paused.group) = self.capture_pause(execution)?;
        for (id, deadline) in execution.approval_deadlines.lock().iter() {
            paused
                .approval_deadlines
                .entry(id.clone())
                .or_insert(*deadline);
        }
        paused.approval_deadlines.retain(|id, _| {
            paused.batch.as_ref().is_some_and(|batch| {
                batch
                    .approvals
                    .iter()
                    .any(|approval| &approval.request_id == id)
            }) || paused.group.as_ref().is_some_and(|group| {
                &group.request_id == id
                    || group
                        .frames
                        .values()
                        .any(|frame| format!("composition:{}", frame.id) == *id)
                    || group.parked.iter().any(|leaf| {
                        leaf.batch
                            .approvals
                            .iter()
                            .any(|approval| &approval.request_id == id)
                    })
                    || group
                        .batch
                        .approvals
                        .iter()
                        .any(|approval| &approval.request_id == id)
            })
        });
        Ok(())
    }

    /// Claims an exact retained composition and resumes its child action before running the parent's saved finalization path.
    async fn resume_group(
        &self,
        snapshot: TaskRunSnapshot,
        payload: TaskCheckpointPayload,
        request_id: String,
        mut response: super::suspension::BatchResponse,
    ) -> Result<TaskRunResult> {
        let group: TaskGroupState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.group")
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                .state
                .clone(),
        )?;
        group.validate_checkpoint(&payload)?;
        let (owner, whole_expiry, request_deadline, composition_expired, targets) = {
            let retained = self.paused.lock();
            let retained = retained
                .get(&snapshot.key.run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            if serde_json::to_value(
                retained
                    .group
                    .as_ref()
                    .ok_or(TaskRunStorageError::NotResumable)?,
            )? != serde_json::to_value(&group)?
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            (
                retained.owner.clone(),
                retained.whole_expiry,
                retained.approval_deadlines.get(&request_id).copied(),
                group
                    .frames
                    .values()
                    .filter(|frame| {
                        frame.contains_operation(
                            &group.child_operation,
                            &group.frames,
                            &mut Vec::new(),
                        )
                    })
                    .any(|frame| {
                        frame
                            .expires_at
                            .is_some_and(|expiry| chrono::Utc::now() >= expiry)
                            || retained
                                .approval_deadlines
                                .get(&format!("composition:{}", frame.id))
                                .is_some_and(|expiry| std::time::Instant::now() >= *expiry)
                    }),
                retained.targets.clone(),
            )
        };
        owner.check()?;
        let child = self.agent.task_group_child(&group, &targets)?;
        child.preflight_priced_autonomy(payload.limits.max_micro_usd.is_some())?;
        if let super::suspension::BatchResponse::UserAnswer(answer) = &response {
            let pending = &group.batch.approvals[0];
            let mut call = group
                .batch
                .calls
                .iter()
                .find(|call| call.id == pending.call_id)
                .cloned()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            call.arguments = pending.context.clone();
            child.validate_task_question_answer(&call, &pending.question, answer)?;
        }
        if whole_expiry.is_some_and(|expiry| std::time::Instant::now() >= expiry) {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        let expired = request_deadline.is_some_and(|expiry| std::time::Instant::now() >= expiry)
            || group.batch.approvals[0]
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry);
        if expired {
            match response {
                super::suspension::BatchResponse::Approval(
                    ai_agents_hitl::ApprovalResult::Timeout,
                ) => {
                    response = super::suspension::BatchResponse::Approval(
                        child.task_approval_timeout_result()?,
                    )
                }
                super::suspension::BatchResponse::QuestionTimeout => {}
                _ => return Err(AgentError::HITLTimeout),
            }
        } else if matches!(
            response,
            super::suspension::BatchResponse::Approval(ai_agents_hitl::ApprovalResult::Timeout)
                | super::suspension::BatchResponse::QuestionTimeout
        ) {
            return Err(AgentError::HITLTimeout);
        }
        let profile = resolve_profile(
            &self.config,
            AutonomyScope::Task,
            payload.profile.as_deref(),
            None,
            None,
            Some(&payload.objective),
            &self.host,
        )?;
        if profile.settings != payload.settings {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let bound = self
            .agent
            .autonomy_extensions()
            .bind_for_agent(&self.agent, &profile)?;
        let controller = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller")
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let lifecycle: LifecycleState =
            serde_json::from_value(controller.state["lifecycle"].clone())?;
        lifecycle.validate(&profile.settings)?;
        let claimed = self
            .store
            .mutate(
                &snapshot.key.run_id,
                &TaskRunMutation::Claim {
                    expected_revision: snapshot.revision,
                    owner_token: uuid::Uuid::new_v4().to_string(),
                },
            )
            .await?;
        let mut paused = self
            .paused
            .lock()
            .remove(&snapshot.key.run_id)
            .ok_or(TaskRunStorageError::Conflict)?;
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let mut execution = RunExecution::new(
            self.store.clone(),
            &claimed,
            lifecycle.clone(),
            owner.clone(),
            std::time::Instant::now(),
        )?;
        {
            let live = Arc::get_mut(&mut execution).ok_or(TaskRunStorageError::Conflict)?;
            live.participants = paused.participants.clone();
            live.targets = paused.targets.clone();
            *live.delegate_frames.lock() = group.frames.clone();
            for leaf in &group.parked {
                live.parked_batches
                    .lock()
                    .insert(leaf.runtime_id.clone(), leaf.batch.clone());
            }
            *live.approval_deadlines.lock() = paused.approval_deadlines.clone();
            if let Some(expiry) = whole_expiry {
                live.expiry_projection = Some(
                    live.expiry_projection
                        .map_or(expiry, |current| current.min(expiry)),
                );
            }
        }
        if let Some(todo) = &paused.todo {
            execution.install_todos(self.agent.todo_store(), todo.checkpoint()?.binding);
        }
        execution.set_scope(&serde_json::from_value(controller.state["scope"].clone())?);
        *self.active.lock() = Some(execution.clone());
        execution
            .update(|payload| {
                payload.clocks.active_interval_started_at = Some(chrono::Utc::now());
                Ok(())
            })
            .await?;
        let outcome = scope_execution(
            execution.clone(),
            Box::pin(async {
                if composition_expired {
                    execution.stop("composition_timeout");
                    return Box::pin(self.finish_stopped_group(
                        &execution,
                        &owner,
                        &group,
                        "composition_timeout".into(),
                    ))
                    .await;
                }
                let response = self
                    .agent
                    .resume_task_group(
                        AutonomyTurnInput {
                            owner: owner.clone(),
                            objective: payload.objective.clone(),
                            controller_message: payload.controller_state.to_string(),
                            source: group.frame.source,
                        },
                        group.clone(),
                        response,
                    )
                    .await;
                if let Some(reason) = execution.stop_reason() {
                    return Box::pin(self.finish_stopped_group(&execution, &owner, &group, reason))
                        .await;
                }
                if matches!(&response, Err(AgentError::TaskSuspended(_))) {
                    return self.pause(&execution, paused.todo.as_ref()).await;
                }
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        if let Some(reason) = execution.stop_reason() {
                            let runtime = TaskRuntimeCheckpoint::between_turns(
                                self.agent.save_state_full().await?,
                            )?;
                            execution
                                .update(|payload| {
                                    if let Some(pending) = payload.pending.take() {
                                        payload.consumed_request_ids.push(pending.id);
                                    }
                                    payload.runtime = runtime;
                                    payload.pause_reason = None;
                                    Ok(())
                                })
                                .await?;
                            return self
                                .finish(&execution, Self::stopped_status(&execution), reason, None)
                                .await;
                        }
                        return Err(error);
                    }
                };
                self.execute(
                    &owner,
                    &execution,
                    &bound,
                    &profile,
                    &snapshot.key,
                    &payload.objective,
                    lifecycle,
                    paused.todo.as_ref(),
                    Some(response),
                )
                .await
            }),
        )
        .await;
        if outcome
            .as_ref()
            .is_ok_and(|result| result.run.status == TaskRunStatus::Paused)
        {
            self.refresh_pause(&mut paused, &execution)?;
            self.paused.lock().insert(snapshot.key.run_id, paused);
            cleanup.finish();
        } else if outcome
            .as_ref()
            .is_ok_and(|result| result.run.status != TaskRunStatus::RecoveryRequired)
        {
            if let Some(todo) = &paused.todo {
                todo.release()?;
            }
            execution.participants.release().await?;
            self.agent.release_autonomy_run(&owner).await?;
            self.retire_owner(&owner);
            cleanup.finish();
        }
        outcome
    }

    /// A stop wins over re-pause; acknowledged parked leaves are closed without replay while genuinely uncertain work retains custody.
    async fn finish_stopped_group(
        &self,
        execution: &RunExecution,
        owner: &Arc<RunOwner>,
        group: &TaskGroupState,
        reason: String,
    ) -> Result<TaskRunResult> {
        if execution.participants.unsettled() {
            return self
                .finish(execution, TaskRunStatus::RecoveryRequired, reason, None)
                .await;
        }
        let saved = execution.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload)?;
        let leaves = if group.parked.is_empty() {
            vec![super::composition::ParkedCompositionChild {
                operation: group.child_operation.clone(),
                runtime_id: group.child_runtime_id.clone(),
                request: group.child_request.clone(),
                batch: group.batch.clone(),
            }]
        } else {
            group.parked.clone()
        };
        let mut pending = Vec::new();
        for mut leaf in leaves {
            if let Some(child) = payload
                .children
                .iter()
                .find(|child| child.child_id == leaf.operation)
                && let Some(request) = &child.pending
            {
                let adapter = payload
                    .adapters
                    .iter()
                    .find(|adapter| adapter.id == format!("runtime.child_batch:{}", leaf.operation))
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                leaf.batch = serde_json::from_value(adapter.state.clone())?;
                leaf.request = request.clone();
                pending.push(leaf);
            }
        }
        let retired = if pending.is_empty() {
            Vec::new()
        } else {
            let mut cancellation = group.clone();
            cancellation.parked = pending;
            Box::pin(self.agent.cancel_task_group(
                owner,
                &cancellation,
                &execution.participants,
                &execution.targets,
            ))
            .await?
        };
        let runtime = TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        execution
            .update(|payload| {
                for (operation, runtime) in retired {
                    let child = payload
                        .children
                        .iter_mut()
                        .find(|child| child.child_id == operation)
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                    child.runtime = runtime;
                    if let Some(pending) = child.pending.take()
                        && !payload.consumed_request_ids.contains(&pending.id)
                    {
                        payload.consumed_request_ids.push(pending.id);
                    }
                    child.result = Some(json!({"error":"task stopped before child continuation"}));
                }
                if let Some(pending) = payload.pending.take() {
                    payload.consumed_request_ids.push(pending.id);
                }
                payload.runtime = runtime;
                payload.pause_reason = None;
                Ok(())
            })
            .await?;
        self.finish(execution, Self::stopped_status(execution), reason, None)
            .await
    }

    /// Cancels a known-safe group without polling any pending child or aggregation work.
    async fn cancel_group(
        &self,
        snapshot: TaskRunSnapshot,
        mut payload: TaskCheckpointPayload,
    ) -> Result<TaskRunResult> {
        let run_id = snapshot.key.run_id.clone();
        let group: TaskGroupState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.group")
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                .state
                .clone(),
        )?;
        group.validate_checkpoint(&payload)?;
        let (owner, participants, targets) = {
            let retained = self.paused.lock();
            let retained = retained
                .get(&run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            if serde_json::to_value(
                retained
                    .group
                    .as_ref()
                    .ok_or(TaskRunStorageError::NotResumable)?,
            )? != serde_json::to_value(&group)?
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            (
                retained.owner.clone(),
                retained.participants.clone(),
                retained.targets.clone(),
            )
        };
        owner.check()?;
        if participants.unsettled() {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let requested = if snapshot.cancel_requested {
            snapshot
        } else {
            self.store
                .mutate(
                    &run_id,
                    &TaskRunMutation::RequestCancel {
                        expected_revision: snapshot.revision,
                    },
                )
                .await?
        };
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let retired = self
            .agent
            .cancel_task_group(&owner, &group, &participants, &targets)
            .await?;
        for (operation, child_runtime) in retired {
            let child = payload
                .children
                .iter_mut()
                .find(|child| child.child_id == operation)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            child.runtime = child_runtime;
            if let Some(pending) = child.pending.take()
                && !payload.consumed_request_ids.contains(&pending.id)
            {
                payload.consumed_request_ids.push(pending.id);
            }
            child.result = Some(json!({"error":"task cancelled before child continuation"}));
        }
        if let Some(pending) = payload.pending.take() {
            payload.consumed_request_ids.push(pending.id);
        }
        for approval in group.batch.approvals.iter().chain(
            group
                .parked
                .iter()
                .flat_map(|leaf| leaf.batch.approvals.iter()),
        ) {
            if !payload.consumed_request_ids.contains(&approval.request_id) {
                payload
                    .consumed_request_ids
                    .push(approval.request_id.clone());
            }
        }
        payload.runtime =
            TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        payload.pause_reason = None;
        payload.stop_reason = Some("cancel_requested".into());
        let acknowledged = self
            .store
            .mutate(
                &run_id,
                &TaskRunMutation::AcknowledgeCancel {
                    expected_revision: requested.revision,
                    payload: serde_json::to_value(payload)?,
                },
            )
            .await?;
        let paused = self
            .paused
            .lock()
            .remove(&run_id)
            .ok_or(TaskRunStorageError::Conflict)?;
        if let Some(todo) = &paused.todo {
            todo.release()?;
        }
        participants.release().await?;
        self.agent.release_autonomy_run(&owner).await?;
        self.retire_owner(&owner);
        cleanup.finish();
        Ok(TaskRunResult {
            run: TaskRun::from_checkpoint(&acknowledged, &self.config_identity)?,
            final_response: None,
        })
    }

    /// Host identity represents the prepared providers, tools and bindings, not merely a mutable profile name.
    pub fn try_new(
        agent: Arc<crate::RuntimeAgent>,
        config: AutonomyConfig,
        host: AutonomyHostCeilings,
        store: Arc<dyn TaskRunStore>,
        config_identity: String,
    ) -> Result<Self> {
        if config_identity.trim().is_empty() {
            return Err(AgentError::Config(
                "empty task configuration identity".into(),
            ));
        }
        super::validate_agent_config(&config, None, &[])?;
        agent.preflight_standalone_autonomy()?;
        Ok(Self {
            agent,
            config,
            host,
            store,
            config_identity,
            active: parking_lot::Mutex::new(None),
            paused: Default::default(),
            resume_gate: Default::default(),
        })
    }

    /// Acknowledged cleanup always retires this owner's catalogue, while active-slot clearing still requires exact pointer identity.
    fn retire_owner(&self, owner: &Arc<RunOwner>) {
        owner.targets.release();
        let mut active = self.active.lock();
        if active
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(&current.owner, owner))
        {
            *active = None;
        }
    }

    /// Requests cancellation of this retained live owner; running-work receipts are not cleanup acknowledgement or rollback.
    /// Known-safe paused work uses awaited cancellation acknowledgement before releasing runtime protection.
    pub async fn request_cancel(&self, run_id: &str) -> Result<TaskRunSummary> {
        if self.paused.lock().contains_key(run_id) {
            let snapshot = self
                .store
                .load(run_id)
                .await?
                .ok_or(TaskRunStorageError::NotFound)?;
            return self
                .cancel_paused(run_id, snapshot.revision)
                .await
                .map(|result| TaskRunSummary {
                    key: result.run.key,
                    actor_id: snapshot.actor_id,
                    revision: result.run.revision,
                    status: result.run.status,
                    owned: false,
                    cancel_requested: true,
                    created_at: snapshot.created_at,
                    updated_at: result.run.updated_at,
                });
        }
        let execution = self
            .active
            .lock()
            .as_ref()
            .cloned()
            .filter(|execution| execution.run_id == run_id)
            .ok_or(TaskRunStorageError::NotResumable)?;
        execution.request_cancel().await
    }

    /// Releases abandoned live protection only after the host has durably reconciled every uncertain outcome.
    /// Calling this method asserts that unobservable external work has stopped; it neither infers rollback nor performs replay.
    /// A temporary CAS claim protects nonterminal cleanup; failed cleanup or acknowledgement remains explicitly recoverable.
    pub async fn acknowledge_recovery_release(
        &self,
        run_id: &str,
        expected_revision: u64,
    ) -> Result<TaskRunSummary> {
        let _resume = self.resume_gate.lock().await;
        let execution = self
            .active
            .lock()
            .as_ref()
            .filter(|execution| execution.run_id == run_id)
            .cloned()
            .ok_or(TaskRunStorageError::NotResumable)?;
        if !execution.owner.is_abandoned() {
            return Err(TaskRunStorageError::Owned.into());
        }
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;
        if snapshot.revision != expected_revision
            || snapshot.owner_token.is_some()
            || !matches!(
                snapshot.status,
                TaskRunStatus::Paused
                    | TaskRunStatus::Completed
                    | TaskRunStatus::Cancelled
                    | TaskRunStatus::Incomplete
                    | TaskRunStatus::Failed
                    | TaskRunStatus::LimitReached
            )
            || payload.pending.is_some()
            || payload.clocks.active_interval_started_at.is_some()
            || !matches!(payload.runtime.continuation, TaskContinuation::BetweenTurns)
            || payload.reservations.iter().any(|reservation| {
                matches!(
                    reservation.state,
                    TaskEffectState::Reserved
                        | TaskEffectState::Dispatched
                        | TaskEffectState::Uncertain
                )
            })
            || payload.children.iter().any(|child| {
                child.pending.is_some()
                    || child.result.is_none()
                    || !matches!(child.runtime.continuation, TaskContinuation::BetweenTurns)
            })
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        execution.participants.acknowledge_reconciliation()?;
        // A conditional claim excludes competing resume/control writes while live protection is retired.
        // Terminal snapshots are already immutable and need no temporary claim.
        let claimed = if snapshot.status == TaskRunStatus::Paused {
            Some(
                self.store
                    .mutate(
                        run_id,
                        &TaskRunMutation::Claim {
                            expected_revision,
                            owner_token: uuid::Uuid::new_v4().to_string(),
                        },
                    )
                    .await?,
            )
        } else {
            None
        };
        execution.participants.release().await?;
        execution.release_reconciled_todos()?;
        self.agent
            .release_reconciled_autonomy_owner(&execution.owner)
            .await?;
        let acknowledged = if let Some(claimed) = claimed {
            self.store
                .mutate(
                    run_id,
                    &TaskRunMutation::Checkpoint {
                        expected_revision: claimed.revision,
                        owner_token: claimed.owner_token.ok_or(TaskRunStorageError::Owned)?,
                        status: TaskRunStatus::Paused,
                        payload: snapshot.payload.clone(),
                        release: true,
                    },
                )
                .await?
        } else {
            snapshot
        };
        self.paused.lock().remove(run_id);
        self.retire_owner(&execution.owner);
        Ok(acknowledged.summary())
    }

    /// Awaits cleanup of a known-safe same-process pause; request receipt alone never releases runtime protection.
    pub async fn cancel_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
    ) -> Result<TaskRunResult> {
        let _resume = self.resume_gate.lock().await;
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != expected_revision || snapshot.status != TaskRunStatus::Paused {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        if matches!(
            payload.runtime.continuation,
            TaskContinuation::Suspended { batch: None, .. }
        ) {
            return Box::pin(self.cancel_group(snapshot, payload)).await;
        }
        let batch: TaskBatchState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.tool_batch")
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                .state
                .clone(),
        )?;
        batch.validate_checkpoint(&payload)?;
        let owner = self
            .paused
            .lock()
            .get(run_id)
            .map(|paused| paused.owner.clone())
            .ok_or(TaskRunStorageError::NotResumable)?;
        let requested = if snapshot.cancel_requested {
            snapshot
        } else {
            self.store
                .mutate(
                    run_id,
                    &TaskRunMutation::RequestCancel { expected_revision },
                )
                .await?
        };
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        payload.runtime = self.agent.cancel_task_batch(&owner, batch.clone()).await?;
        payload.pending = None;
        payload.pause_reason = None;
        payload.stop_reason = Some("cancel_requested".into());
        for approval in &batch.approvals {
            if !payload.consumed_request_ids.contains(&approval.request_id) {
                payload
                    .consumed_request_ids
                    .push(approval.request_id.clone());
            }
        }
        let acknowledged = self
            .store
            .mutate(
                run_id,
                &TaskRunMutation::AcknowledgeCancel {
                    expected_revision: requested.revision,
                    payload: serde_json::to_value(payload)?,
                },
            )
            .await?;
        let paused = self
            .paused
            .lock()
            .remove(run_id)
            .ok_or(TaskRunStorageError::Conflict)?;
        if let Some(todo) = &paused.todo {
            todo.release()?;
        }
        paused.participants.release().await?;
        self.agent.release_autonomy_run(&owner).await?;
        self.retire_owner(&owner);
        cleanup.finish();
        Ok(TaskRunResult {
            run: TaskRun::from_checkpoint(&acknowledged, &self.config_identity)?,
            final_response: None,
        })
    }

    /// Returns a pause only after exact continuation and owner release are acknowledged by the selected store.
    /// Boxed group ledger futures keep optional composition state out of every nested controller polling frame.
    async fn pause(
        &self,
        execution: &Arc<RunExecution>,
        todo: Option<&RunTodoAdapter>,
    ) -> Result<TaskRunResult> {
        let runtime = self.agent.task_suspension_snapshot().await?;
        let todos = todo.map(RunTodoAdapter::checkpoint).transpose()?;
        let snapshot = if execution
            .parked_batches
            .lock()
            .contains_key(&self.agent.info().id)
        {
            Box::pin(execution.pause_batch(runtime, &self.agent.info().id, todos)).await?
        } else {
            Box::pin(execution.pause_group(runtime, &self.agent.info().id, todos)).await?
        };
        Ok(TaskRunResult {
            run: TaskRun::from_checkpoint(&snapshot, &self.config_identity)?,
            final_response: None,
        })
    }

    /// Validates one bound approval or question answer before claiming its same-process exact continuation.
    /// The implementation-owned question contract is checked before claim and final authorization still runs on resume.
    /// A losing or invalid response cannot mutate the runtime, and failed execution retains the owner for recovery.
    pub async fn resume(
        &self,
        run_id: &str,
        expected_revision: u64,
        input: TaskResumeInput,
    ) -> Result<TaskRunResult> {
        let _resume = self.resume_gate.lock().await;
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != expected_revision
            || snapshot.status != TaskRunStatus::Paused
            || snapshot.cancel_requested
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;
        if payload
            .clocks
            .active_millis
            .checked_add(payload.clocks.interrupted_interval_millis)
            .is_none_or(|used| used >= payload.limits.max_active_time_seconds.saturating_mul(1000))
            || payload
                .clocks
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        let (request_id, mut result, question_response) = match input {
            TaskResumeInput::Approval { request_id, result } => (
                request_id,
                super::suspension::BatchResponse::Approval(result),
                false,
            ),
            TaskResumeInput::UserAnswer { request_id, answer } => (
                request_id,
                super::suspension::BatchResponse::UserAnswer(answer),
                true,
            ),
            TaskResumeInput::QuestionTimeout { request_id } => (
                request_id,
                super::suspension::BatchResponse::QuestionTimeout,
                true,
            ),
        };
        if payload.pending.as_ref().is_none_or(|pending| {
            pending.id != request_id
                || if question_response {
                    !matches!(pending.kind, TaskPendingKind::UserQuestion)
                } else {
                    !matches!(pending.kind, TaskPendingKind::Approval)
                }
        }) || payload.consumed_request_ids.contains(&request_id)
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        if matches!(
            payload.runtime.continuation,
            TaskContinuation::Suspended { batch: None, .. }
        ) {
            return Box::pin(self.resume_group(snapshot, payload, request_id, result)).await;
        }
        let adapter = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.tool_batch")
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let batch: TaskBatchState = serde_json::from_value(adapter.state.clone())?;
        batch.validate_checkpoint(&payload)?;
        if let super::suspension::BatchResponse::UserAnswer(answer) = &result {
            let pending = &batch.approvals[0];
            let call = batch
                .calls
                .iter()
                .find(|call| call.id == pending.call_id)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            let mut final_call = call.clone();
            final_call.arguments = pending.context.clone();
            self.agent
                .validate_task_question_answer(&final_call, &pending.question, answer)?;
        }
        {
            let retained = self.paused.lock();
            let retained = retained
                .get(run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            if serde_json::to_value(
                retained
                    .batch
                    .as_ref()
                    .ok_or(TaskRunStorageError::NotResumable)?,
            )? != serde_json::to_value(&batch)?
            {
                return Err(AgentError::Config(
                    "retained task batch differs from acknowledged checkpoint".into(),
                ));
            }
        }
        let (whole_expiry, approval_deadline) = {
            let retained = self.paused.lock();
            let retained = retained
                .get(run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            (
                retained.whole_expiry,
                retained.approval_deadlines.get(&request_id).copied(),
            )
        };
        if whole_expiry.is_some_and(|expiry| std::time::Instant::now() >= expiry) {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        if approval_deadline.is_some_and(|expiry| std::time::Instant::now() >= expiry)
            || batch.approvals[0]
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        {
            match result {
                super::suspension::BatchResponse::Approval(
                    ai_agents_hitl::ApprovalResult::Timeout,
                ) => {
                    result = super::suspension::BatchResponse::Approval(
                        self.agent.task_approval_timeout_result()?,
                    );
                }
                super::suspension::BatchResponse::QuestionTimeout => {}
                _ => return Err(AgentError::HITLTimeout),
            }
        } else if matches!(
            result,
            super::suspension::BatchResponse::Approval(ai_agents_hitl::ApprovalResult::Timeout)
                | super::suspension::BatchResponse::QuestionTimeout
        ) {
            return Err(AgentError::HITLTimeout);
        }
        if batch
            .approvals
            .first()
            .is_none_or(|approval| approval.request_id != request_id)
            || adapter.config != json!({"runtime_id":self.agent.info().id})
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let profile = resolve_profile(
            &self.config,
            AutonomyScope::Task,
            payload.profile.as_deref(),
            None,
            None,
            Some(&payload.objective),
            &self.host,
        )?;
        if profile.settings != payload.settings {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let bound = self
            .agent
            .autonomy_extensions()
            .bind_for_agent(&self.agent, &profile)?;
        let lifecycle = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller")
            .map(|adapter| serde_json::from_value(adapter.state["lifecycle"].clone()))
            .transpose()?
            .unwrap_or(LifecycleState::new(&profile.settings, false)?);
        let owner = self
            .paused
            .lock()
            .get(run_id)
            .map(|paused| paused.owner.clone())
            .ok_or_else(|| {
                AgentError::Config("this checkpoint needs compatible live-owner restoration".into())
            })?;
        owner.check()?;
        let claimed = self
            .store
            .mutate(
                run_id,
                &TaskRunMutation::Claim {
                    expected_revision,
                    owner_token: uuid::Uuid::new_v4().to_string(),
                },
            )
            .await?;
        let mut paused = self
            .paused
            .lock()
            .remove(run_id)
            .ok_or(TaskRunStorageError::Conflict)?;
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let mut execution = RunExecution::new(
            self.store.clone(),
            &claimed,
            lifecycle.clone(),
            owner.clone(),
            std::time::Instant::now(),
        )?;
        {
            let live = Arc::get_mut(&mut execution).ok_or(TaskRunStorageError::Conflict)?;
            live.participants = paused.participants.clone();
            live.targets = paused.targets.clone();
        }
        if let Some(expiry) = whole_expiry {
            let execution = Arc::get_mut(&mut execution).ok_or(TaskRunStorageError::Conflict)?;
            execution.expiry_projection = Some(
                execution
                    .expiry_projection
                    .map_or(expiry, |current| current.min(expiry)),
            );
        }
        *self.active.lock() = Some(execution.clone());
        if let Some(todo) = &paused.todo {
            execution.install_todos(self.agent.todo_store(), todo.checkpoint()?.binding);
        }
        if let Some(controller) = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller")
        {
            execution.set_scope(&serde_json::from_value(controller.state["scope"].clone())?);
        }
        execution
            .update(|payload| {
                payload.clocks.active_interval_started_at = Some(chrono::Utc::now());
                Ok(())
            })
            .await?;
        let outcome = scope_execution(
            execution.clone(),
            Box::pin(async {
                let response = self
                    .agent
                    .resume_task_batch(
                        AutonomyTurnInput {
                            owner: owner.clone(),
                            objective: payload.objective.clone(),
                            controller_message: payload.controller_state.to_string(),
                            source: AutonomyTurnSource::ResumeAfterApproval,
                        },
                        batch,
                        &request_id,
                        result,
                    )
                    .await;
                if matches!(&response, Err(AgentError::TaskSuspended(_))) {
                    return self.pause(&execution, paused.todo.as_ref()).await;
                }
                let response = response?;
                let runtime =
                    TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
                execution
                    .update(|payload| {
                        payload.runtime = runtime;
                        payload.pending = None;
                        payload.pause_reason = None;
                        if !payload.consumed_request_ids.contains(&request_id) {
                            payload.consumed_request_ids.push(request_id.clone());
                        }
                        Ok(())
                    })
                    .await?;
                self.execute(
                    &owner,
                    &execution,
                    &bound,
                    &profile,
                    &snapshot.key,
                    &payload.objective,
                    lifecycle,
                    paused.todo.as_ref(),
                    Some(response),
                )
                .await
            }),
        )
        .await;
        if outcome
            .as_ref()
            .is_ok_and(|result| result.run.status == TaskRunStatus::Paused)
        {
            self.refresh_pause(&mut paused, &execution)?;
            self.paused.lock().insert(run_id.into(), paused);
            cleanup.finish();
        } else if outcome
            .as_ref()
            .is_ok_and(|result| result.run.status != TaskRunStatus::RecoveryRequired)
        {
            if let Some(todo) = &paused.todo {
                todo.release()?;
            }
            execution.participants.release().await?;
            self.agent.release_autonomy_run(&owner).await?;
            self.retire_owner(&owner);
            cleanup.finish();
        }
        outcome
    }

    /// Runs a bounded foreground objective; unsuccessful storage writes never publish a successful task result.
    pub async fn run(&self, objective: &str, selected: Option<&str>) -> Result<TaskRunResult> {
        let profile = resolve_profile(
            &self.config,
            AutonomyScope::Task,
            selected,
            None,
            None,
            Some(objective),
            &self.host,
        )?;
        if !profile.enabled {
            return Err(AgentError::Config(
                "selected autonomy profile is disabled".into(),
            ));
        }
        self.agent
            .preflight_priced_autonomy(profile.max_cost_usd.is_some())?;
        let objective = if objective.trim().is_empty() {
            profile.settings.objective.as_deref().unwrap_or(objective)
        } else {
            objective
        }
        .to_string();
        let objective = if let Some(template) = &profile.settings.objective_template {
            self.agent.render_autonomy_objective(template, &objective)?
        } else {
            objective
        };
        if objective.trim().is_empty() {
            return Err(AgentError::Config(
                "standalone runner requires a nonempty resolved objective".into(),
            ));
        }
        let bound = self
            .agent
            .autonomy_extensions()
            .bind_for_agent(&self.agent, &profile)?;

        if profile
            .settings
            .progress
            .as_ref()
            .and_then(|progress| progress.stagnation.as_ref())
            .is_some_and(|policy| {
                (policy.action == Some(StagnationAction::AskUser)
                    || policy.on_exhausted == Some(StagnationAction::AskUser))
                    && policy.on_unavailable.is_none()
            })
        {
            return Err(AgentError::Config(
                "unavailable task interaction requires an explicit stop/fail fallback".into(),
            ));
        }
        if profile.settings.persistence.is_some()
            || profile
                .settings
                .premature_finish
                .as_ref()
                .is_some_and(|policy| policy.action == Some(PrematureFinishAction::AskUser))
        {
            return Err(AgentError::Config(
                "task persistence/resume and host-interaction profile controls are not installed"
                    .into(),
            ));
        }
        if bound
            .checks
            .iter()
            .any(|check| !check.adapter.descriptor().conditions.is_empty())
            || profile.settings.hitl.as_ref().is_some_and(|hitl| {
                hitl.on_user_question
                    .is_some_and(|action| action != InteractionAction::PauseRun)
                    || hitl
                        .on_approval_required
                        .is_some_and(|action| action != InteractionAction::PauseRun)
            })
            || profile.settings.supervision.is_some()
            || profile.settings.on_limit.as_ref().is_some_and(|limit| {
                !matches!(
                    limit.action,
                    None | Some(LimitAction::SummarizeAndStop | LimitAction::Fail)
                )
            })
        {
            return Err(AgentError::Config(
                "live task suspension is not installed for this profile".into(),
            ));
        }
        let lifecycle = LifecycleState::new(&profile.settings, false)?;
        let run_id = uuid::Uuid::new_v4().to_string();
        let owner = self.agent.reserve_autonomy_run(run_id.clone()).await?;
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let started = std::time::Instant::now();
        let now = chrono::Utc::now();
        let key = TaskRunKey {
            agent_id: self.agent.info().id,
            run_id: run_id.clone(),
        };
        let mut todo = None;
        // Known pre-execution setup errors release the reservation; drop during an ambiguous write still requires recovery.
        let setup = Box::pin(async {
            let mut payload = TaskCheckpointPayload::new(
                objective.clone(),
                self.config_identity.clone(),
                &profile,
                TaskRuntimeCheckpoint::between_turns(self.agent.save_state().await?)?,
            );
            payload.clocks.active_interval_started_at = Some(now);
            payload.clocks.expires_at = profile
                .max_wall_time_seconds
                .map(|seconds| {
                    i64::try_from(seconds)
                        .ok()
                        .and_then(chrono::Duration::try_seconds)
                        .and_then(|duration| now.checked_add_signed(duration))
                        .ok_or_else(|| AgentError::Config("task expiry overflow".into()))
                })
                .transpose()?;
            let envelope = TaskRunSnapshot {
                schema_version: TASK_RUN_SCHEMA_VERSION,
                key: key.clone(),
                actor_id: owner.actor_id.clone(),
                revision: 0,
                status: TaskRunStatus::Running,
                owner_token: Some(uuid::Uuid::new_v4().to_string()),
                cancel_requested: false,
                created_at: now,
                updated_at: now,
                payload: json!({}),
            };
            // Validate large objective/history data before replacing the prior session's canonical list.
            payload.clone().bind(envelope.clone())?;
            if profile.settings.progress.as_ref().is_some_and(|p| {
                p.todo_tool.is_some()
                    || p.adapter.as_deref() == Some("builtin.todo")
                    || p.require_todos == Some(true)
                    || p.require_active_todo == Some(true)
                    || p.max_open_todos.is_some()
            }) {
                todo = Some(RunTodoAdapter::begin(self.agent.todo_store(), &key)?);
                todo.as_ref().unwrap().set_open_limit(
                    profile
                        .settings
                        .progress
                        .as_ref()
                        .and_then(|progress| progress.max_open_todos),
                )?;
                payload.todos = Some(todo.as_ref().unwrap().checkpoint()?);
            }
            let snapshot = payload.bind(envelope)?;
            self.store.create(&snapshot).await?;
            Ok::<_, AgentError>(snapshot)
        })
        .await;
        let snapshot = match setup {
            Ok(snapshot) => snapshot,
            Err(error) => {
                if let Some(todo) = &todo {
                    todo.release()?;
                }
                self.agent.release_autonomy_run(&owner).await?;
                cleanup.finish();
                return Err(error);
            }
        };
        let execution = RunExecution::new(
            self.store.clone(),
            &snapshot,
            lifecycle.clone(),
            owner.clone(),
            started,
        )?;
        if let Some(todo) = &todo {
            execution.install_todos(self.agent.todo_store(), todo.checkpoint()?.binding);
        }
        *self.active.lock() = Some(execution.clone());
        // Box the controller before layering task scopes so default-stack polling matches the ordinary runtime entry.
        let result = scope_execution(
            execution.clone(),
            Box::pin(self.execute(
                &owner,
                &execution,
                &bound,
                &profile,
                &key,
                &objective,
                lifecycle,
                todo.as_ref(),
                None,
            )),
        )
        .await;
        let result = match result {
            Err(error)
                if !matches!(
                    &error,
                    AgentError::Persistence(_) | AgentError::TaskRunStorage(_)
                ) =>
            {
                self.finish(
                    &execution,
                    Self::stopped_status(&execution),
                    execution
                        .stop_reason()
                        .unwrap_or_else(|| "execution_error".into()),
                    None,
                )
                .await
            }
            result => result,
        };
        if result
            .as_ref()
            .is_ok_and(|result| result.run.status == TaskRunStatus::Paused)
        {
            let (batch, group) = self.capture_pause(&execution)?;
            self.paused.lock().insert(
                run_id.clone(),
                PausedTask {
                    participants: execution.participants.clone(),
                    targets: execution.targets.clone(),
                    owner,
                    todo,
                    whole_expiry: execution.expiry_projection,
                    approval_deadlines: execution.approval_deadlines.lock().clone(),
                    batch,
                    group,
                },
            );
            cleanup.finish();
            return result;
        }
        // A failed write may leave a durable dispatched effect; keep the live runtime reserved for recovery.
        if result
            .as_ref()
            .is_ok_and(|result| result.run.status != TaskRunStatus::RecoveryRequired)
        {
            if let Some(todo) = &todo {
                todo.release()?;
            }
            execution.participants.release().await?;
            self.agent.release_autonomy_run(&owner).await?;
            self.retire_owner(&owner);
            cleanup.finish();
        }
        result
    }

    /// The controller invokes internal turns, not public chat, and shares one ledger with validation observations.
    #[allow(clippy::too_many_arguments)]
    async fn execute(
        &self,
        owner: &Arc<RunOwner>,
        execution: &Arc<RunExecution>,
        bound: &BoundAutonomyProfile,
        profile: &EffectiveAutonomyProfile,
        key: &TaskRunKey,
        objective: &str,
        mut lifecycle: LifecycleState,
        todo: Option<&RunTodoAdapter>,
        mut resumed_response: Option<AgentResponse>,
    ) -> Result<TaskRunResult> {
        let restored = execution.load_owned().await?;
        let restored: TaskCheckpointPayload = serde_json::from_value(restored.payload)?;
        let controller = restored
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller");
        let mut scope = EvaluationScope {
            key: key.clone(),
            objective_revision: 0,
            cycle: 0,
            stage: None,
            mutation_generation: 0,
            target_revisions: BTreeMap::new(),
            target_generations: BTreeMap::new(),
            validation_bindings: BTreeMap::new(),
            validation_attempts: BTreeMap::new(),
        };
        if let Some(controller) = controller {
            scope = serde_json::from_value(controller.state["scope"].clone())?;
            lifecycle = serde_json::from_value(controller.state["lifecycle"].clone())?;
        }
        bound.prepare_scope(&mut scope);
        let mut progress = if let Some(controller) = controller {
            serde_json::from_value(controller.state["progress"].clone())?
        } else {
            ProgressState::new(scope.clone())
        };
        let mut evidence = EvaluationEvidence {
            required_checks: bound
                .checks
                .iter()
                .filter(|check| check.check.required.unwrap_or(true))
                .map(|check| check.check.id.clone())
                .collect(),
            ..Default::default()
        };
        let journal = TaskValidationJournal::for_execution(execution);
        let executor = RuntimeObservationExecutor {
            agent: self.agent.clone(),
            profile: profile.profile.clone(),
            scope_mode: "task".into(),
            run_revision: execution.revision.clone(),
            validator: String::new(),
            contract_version: 1,
        };
        if let Some(controller) = controller {
            evidence = serde_json::from_value(controller.state["evidence"].clone())?;
        }
        let mut last_response = None;
        let mut instruction = String::new();
        let mut source = AutonomyTurnSource::InitialObjective;
        let mut forced_continuation = false;
        loop {
            let snapshot = execution.load_owned().await?;
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
            if execution.check(&payload).is_err()
                || (resumed_response.is_none()
                    && payload.counters.turns >= u64::from(profile.max_turns))
            {
                return self
                    .finish(
                        execution,
                        if execution.stop_reason().is_some() {
                            Self::stopped_status(execution)
                        } else {
                            TaskRunStatus::LimitReached
                        },
                        execution
                            .stop_reason()
                            .unwrap_or_else(|| "turn_capacity".into()),
                        last_response,
                    )
                    .await;
            }
            if resumed_response.is_none() {
                scope.cycle += 1;
            }
            scope.mutation_generation = execution.mutation_generation();
            scope.stage = profile
                .settings
                .lifecycle
                .as_ref()
                .and_then(|stages| stages.get(lifecycle.current_stage))
                .map(|stage| stage.id.clone());

            execution.set_scope(&scope);
            // This is an advisory view; live admission remains authoritative if time passes before dispatch.
            let controller = json!({"objective":objective,"stage":scope.stage,"instruction":instruction,
                "todos":todo.map(RunTodoAdapter::checkpoint).transpose()?,
                "remaining_turns":profile.max_turns as u64-payload.counters.turns,
                "remaining_llm_calls":(profile.max_llm_calls as u64).saturating_sub(payload.counters.llm_attempts),
                "remaining_tool_calls":(profile.max_tool_calls as u64).saturating_sub(payload.counters.tool_attempts),
                "remaining_command_calls":(profile.max_command_calls as u64).saturating_sub(payload.counters.command_attempts),
                "remaining_active_seconds":profile.max_active_time_seconds.saturating_mul(1000).saturating_sub(execution.consumed_active_millis()?)/1000,
                "remaining_wall_seconds":payload.clocks.expires_at.map(|expiry|(expiry-chrono::Utc::now()).num_seconds().max(0)),
                "remaining_micro_usd":payload.limits.max_micro_usd.map(|limit| limit.saturating_sub(payload.counters.charged_micro_usd).saturating_sub(payload.reservations.iter().filter(|reservation| matches!(reservation.state, TaskEffectState::Reserved | TaskEffectState::Dispatched | TaskEffectState::Uncertain)).map(|reservation| reservation.reserved_micro_usd.saturating_sub(reservation.charged_micro_usd)).sum::<u64>())),
                "remaining_declared_write_paths":payload.limits.max_declared_write_paths.map(|limit| (limit as usize).saturating_sub(payload.declared_write_targets.len())),
                                "failed_gates":payload.evidence.records.last().and_then(|record| record.data.get("completion")).cloned(),
                "progress_instruction":profile.settings.progress.as_ref().and_then(|progress| progress.auto_create_todo_prompt.clone()).unwrap_or_else(|| "Before progress-gated work, maintain the run todo list and select exactly one active item when required.".into()),
                "stage_instruction":profile.settings.lifecycle.as_ref().and_then(|stages| stages.get(lifecycle.current_stage)).and_then(|stage| stage.instruction.clone())});
            execution
                .update(|payload| {
                    if resumed_response.is_none() { payload.counters.turns += 1; }
                    if forced_continuation && resumed_response.is_none() {
                        payload.counters.continuations += 1;
                    }
                    payload.stage = scope.stage.clone();
                    payload.controller_state = controller.clone();
                    let checkpoint = TaskAdapterCheckpoint { id: "runtime.controller".into(), adapter: "runtime.controller".into(),
                        contract_version: 1, config: json!({"runtime_id":key.agent_id}),
                        state: json!({"scope":scope,"lifecycle":lifecycle,"progress":progress,"evidence":evidence}) };
                    if let Some(existing) = payload.adapters.iter_mut().find(|adapter| adapter.id == checkpoint.id) { *existing = checkpoint; }
                    else { payload.adapters.push(checkpoint); }
                    Ok(())
                })
                .await?;
            let response = if let Some(response) = resumed_response.take() {
                Ok(response)
            } else {
                self.agent
                    .run_autonomy_turn(AutonomyTurnInput {
                        owner: owner.clone(),
                        objective: objective.into(),
                        controller_message: controller.to_string(),
                        source,
                    })
                    .await
            };
            match response {
                Ok(response) => last_response = Some(response),
                Err(AgentError::TaskSuspended(_)) => {
                    return self.pause(execution, todo).await;
                }
                Err(_error) => {
                    return self
                        .finish(
                            execution,
                            Self::stopped_status(execution),
                            execution
                                .stop_reason()
                                .unwrap_or_else(|| "execution_error".into()),
                            last_response,
                        )
                        .await;
                }
            }
            if let Some(reason) = execution.stop_reason() {
                return self
                    .finish(
                        execution,
                        Self::stopped_status(execution),
                        reason,
                        last_response,
                    )
                    .await;
            }
            execution.await_children().await?;
            self.agent.flush_background_tasks().await?;
            let runtime =
                TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
            let todos = todo.map(RunTodoAdapter::checkpoint).transpose()?;
            execution
                .update(|payload| {
                    payload.runtime = runtime;
                    payload.todos = todos.clone();
                    payload.last_response = last_response.as_ref().map(|r| r.content.clone());
                    Ok(())
                })
                .await?;
            scope.mutation_generation = execution.mutation_generation();
            let identity = Self::identity(&scope, scope.cycle, None, String::new());
            evidence.current = Some(ScopedObservation {
                identity: identity.clone(),
                complete: true,
                value: CurrentTaskObservation {
                    state: self.agent.current_state(),
                    context: serde_json::to_value(self.agent.context_manager().get_all())?,
                    response: last_response.as_ref().unwrap().content.clone(),
                    todos,
                },
            });
            // Shared executor records remain authoritative; capturing a logical request does not charge another attempt.
            scope.mutation_generation = execution.mutation_generation();
            Self::collect_tools(execution, &mut evidence).await?;
            let schedules = [
                ValidationSchedule::EachCycle,
                ValidationSchedule::StageEnd,
                ValidationSchedule::Completion,
            ];
            for schedule in schedules {
                for check in bound.scheduled_checks(&scope, schedule) {
                    let evaluator = CompletionGateEvaluator::default();
                    if schedule == ValidationSchedule::Completion
                        && !check.check.required.unwrap_or(true)
                        && profile.settings.completion.as_ref().is_some_and(|gate| {
                            evaluator
                                .evaluate(gate, &scope, &evidence)
                                .is_ok_and(|result| result.outcome == GateOutcome::Pass)
                        })
                    {
                        continue;
                    }
                    self.validate(
                        check,
                        &mut scope,
                        &mut evidence,
                        execution,
                        bound,
                        &executor,
                        &journal,
                    )
                    .await?;
                }
            }
            scope.mutation_generation = execution.mutation_generation();
            Self::collect_tools(execution, &mut evidence).await?;
            let observation = bound.observe_progress(&scope, &evidence)?;
            progress.observe(&scope, &observation)?;
            evidence.progress =
                Some(observation.scoped(Self::identity(&scope, scope.cycle, None, String::new())));
            let evaluator = CompletionGateEvaluator::default();
            let previous_stage = lifecycle.current_stage;
            if let Some(stage) = profile
                .settings
                .lifecycle
                .as_ref()
                .and_then(|stages| stages.get(lifecycle.current_stage))
            {
                let gate = stage
                    .completion
                    .as_ref()
                    .map(|gate| evaluator.evaluate(gate, &scope, &evidence))
                    .transpose()?
                    .map_or(GateOutcome::Pass, |r| r.outcome);
                if lifecycle.complete_stage(
                    &profile.settings,
                    &stage.id,
                    gate,
                    bound.required_stage_outcome(&scope, &evidence)?,
                )? {
                    execution.set_lifecycle(&profile.settings, &lifecycle);
                } else if (gate == GateOutcome::Fail
                    || bound.required_stage_outcome(&scope, &evidence)? == GateOutcome::Fail)
                    && lifecycle.validation_failure(&profile.settings)?
                {
                    source = AutonomyTurnSource::ValidationFix;
                    execution.set_lifecycle(&profile.settings, &lifecycle);
                }
            }
            let completion = profile
                .settings
                .completion
                .as_ref()
                .map(|gate| evaluator.evaluate(gate, &scope, &evidence))
                .transpose()?;
            let mut gate = completion
                .as_ref()
                .map_or(GateOutcome::Unknown, |result| result.outcome);
            // A nonempty-list requirement is distinct from the explicitly configured todos_done completion gate.
            if gate == GateOutcome::Pass
                && profile
                    .settings
                    .progress
                    .as_ref()
                    .is_some_and(|progress| progress.require_todos == Some(true))
                && todo
                    .map(RunTodoAdapter::checkpoint)
                    .transpose()?
                    .is_none_or(|checkpoint| checkpoint.items.is_empty())
            {
                gate = GateOutcome::Fail;
            }
            let current = execution.load_owned().await?;
            let payload: TaskCheckpointPayload = serde_json::from_value(current.payload)?;
            let decision = checkpoint_decision(
                &profile.settings,
                &lifecycle,
                &mut progress,
                &DecisionInputs {
                    uncertain_effect: payload.reservations.iter().any(|r| {
                        matches!(
                            r.state,
                            TaskEffectState::Dispatched | TaskEffectState::Uncertain
                        )
                    }),
                    cancel_requested: execution
                        .cancellation
                        .load(std::sync::atomic::Ordering::Acquire),
                    unrecoverable_error: execution.stop_reason().is_some_and(|reason| {
                        !matches!(reason.as_str(), "time_violation" | "cancel_requested")
                    }),
                    actual_time_violation: execution.stop_reason().as_deref()
                        == Some("time_violation"),
                    mandatory_wait: payload.pending.is_some(),
                    gates: gate,
                    required_validation: {
                        let mut task_scope = scope.clone();
                        task_scope.stage = None;
                        bound.required_outcome(&task_scope, &evidence)?
                    },
                    children_settled: !execution.participants.unsettled()
                        && payload.children.iter().all(|child| child.result.is_some()),
                    scope_ended: false,
                    additional_work_capacity: execution.stop_reason().is_none()
                        && payload.counters.turns < u64::from(profile.max_turns)
                        && payload.counters.llm_attempts < u64::from(profile.max_llm_calls),
                    interaction_available: false,
                },
            )?;
            execution.update(|payload| {
                payload.progress.observation_sequence = progress.sequence;
                payload.progress.cycles_without_progress = progress.cycles_without_progress;
                payload.progress.replans = progress.replans;
                payload.progress.high_water_marks = serde_json::to_value(&progress)?;
                payload.evidence.records.push(TaskEvidenceRecord { run_id: key.run_id.clone(), sequence: scope.cycle,
                    stage: scope.stage.clone(), attempt: None, origin: "controller".into(),
                    data: json!({"scope":scope,"gate":gate,"completion":completion,"validations":evidence.validations.iter().filter(|result| result.identity.cycle == scope.cycle).collect::<Vec<_>>(),"progress":progress.last_delta,"decision":decision}) });
                Ok(())
            }).await?;
            match decision {
                CheckpointDecision::Terminal { status, reason } => {
                    return self.finish(execution, status, reason, last_response).await;
                }
                CheckpointDecision::Limit => {
                    return self
                        .finish(
                            execution,
                            TaskRunStatus::LimitReached,
                            "invocation_capacity".into(),
                            last_response,
                        )
                        .await;
                }
                CheckpointDecision::Replan => {
                    forced_continuation = false;
                    let check = bound.replan_check(&progress, objective)?;
                    scope
                        .validation_bindings
                        .insert(check.check.id.clone(), check.config_identity.clone());
                    self.validate(
                        &check,
                        &mut scope,
                        &mut evidence,
                        execution,
                        bound,
                        &executor,
                        &journal,
                    )
                    .await?;
                    instruction = evidence
                        .validations
                        .last()
                        .map(|result| result.metrics.to_string())
                        .unwrap_or_default();
                }
                CheckpointDecision::Wait | CheckpointDecision::AskUser => {
                    return Err(AgentError::Config(
                        "task suspension is not installed".into(),
                    ));
                }
                CheckpointDecision::Continue => {
                    if lifecycle.current_stage != previous_stage
                        && lifecycle.current_stage
                            < profile.settings.lifecycle.as_ref().map_or(0, Vec::len)
                    {
                        forced_continuation = false;
                        source = AutonomyTurnSource::StageInstruction;
                        instruction = String::new();
                        continue;
                    }
                    if source == AutonomyTurnSource::ValidationFix {
                        forced_continuation = false;
                        instruction = "Required validation failed. Repair the reopened stage without resetting completed effects or resource usage.".into();
                        continue;
                    }
                    forced_continuation = progress.handled_sequence != Some(progress.sequence);
                    let policy = profile.settings.premature_finish.as_ref();
                    let action = policy
                        .and_then(|p| p.action)
                        .unwrap_or(PrematureFinishAction::Continue);
                    if progress.handled_sequence != Some(progress.sequence) {
                        match action {
                            PrematureFinishAction::SummarizeAndStop => {
                                return self
                                    .finish(
                                        execution,
                                        TaskRunStatus::Incomplete,
                                        "premature_finish".into(),
                                        last_response,
                                    )
                                    .await;
                            }
                            PrematureFinishAction::Fail => {
                                return self
                                    .finish(
                                        execution,
                                        TaskRunStatus::Failed,
                                        "premature_finish".into(),
                                        last_response,
                                    )
                                    .await;
                            }
                            PrematureFinishAction::AskUser => {
                                return self
                                    .finish(
                                        execution,
                                        TaskRunStatus::Incomplete,
                                        "interaction_unavailable".into(),
                                        last_response,
                                    )
                                    .await;
                            }
                            PrematureFinishAction::Continue => {}
                        }
                        if payload.counters.continuations
                            >= u64::from(
                                policy
                                    .and_then(|p| p.max_continuations)
                                    .unwrap_or(profile.max_turns),
                            )
                        {
                            return self
                                .finish(
                                    execution,
                                    TaskRunStatus::LimitReached,
                                    "continuation_capacity".into(),
                                    last_response,
                                )
                                .await;
                        }
                    }
                    instruction = policy.and_then(|p| p.prompt.clone()).unwrap_or_else(|| "Completion is not proven. Continue the objective using the failed gates and remaining allowance.".into());
                }
            }
            if source != AutonomyTurnSource::ValidationFix {
                source = AutonomyTurnSource::Continuation;
            }
        }
    }

    /// Binds one selected check attempt to the same acknowledged revision and invocation ledger as runtime turns.
    #[allow(clippy::too_many_arguments)]
    async fn validate(
        &self,
        check: &BoundValidationCheck,
        scope: &mut EvaluationScope,
        evidence: &mut EvaluationEvidence,
        execution: &Arc<RunExecution>,
        bound: &BoundAutonomyProfile,
        executor: &RuntimeObservationExecutor,
        journal: &TaskValidationJournal,
    ) -> Result<()> {
        let attempt = format!("{}:{}", scope.cycle, check.check.id);
        scope
            .validation_attempts
            .insert(check.check.id.clone(), attempt.clone());
        let capacity = execution.load_owned().await?;
        let capacity: TaskCheckpointPayload = serde_json::from_value(capacity.payload)?;
        if (check.adapter.descriptor().needs_judge || check.adapter.descriptor().needs_planner)
            && capacity.counters.llm_attempts >= u64::from(capacity.limits.max_llm_calls)
        {
            execution.stop("llm_capacity");
            return Err(AgentError::Other(
                "validation has no remaining provider allowance".into(),
            ));
        }
        scope.mutation_generation = execution.mutation_generation();
        let identity = Self::identity(
            scope,
            scope.cycle,
            Some(attempt),
            check.config_identity.clone(),
        );
        let mut driver = ValidationDriverState::new(check, identity)?;
        let snapshot = execution.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        let millis = payload
            .limits
            .max_active_time_seconds
            .saturating_mul(1000)
            .saturating_sub(execution.consumed_active_millis()?);
        let deadline = chrono::Utc::now()
            .checked_add_signed(chrono::Duration::milliseconds(
                i64::try_from(millis)
                    .map_err(|_| AgentError::Config("deadline overflow".into()))?,
            ))
            .ok_or_else(|| AgentError::Config("deadline overflow".into()))?;
        let deadline = payload
            .clocks
            .expires_at
            .map_or(deadline, |expiry| deadline.min(expiry));
        let cancellation = ToolCancellationToken::new(
            execution.cancellation.clone(),
            Some("task cancellation".into()),
        );
        let executor = RuntimeObservationExecutor {
            agent: executor.agent.clone(),
            profile: executor.profile.clone(),
            scope_mode: executor.scope_mode.clone(),
            run_revision: executor.run_revision.clone(),
            validator: check.check.adapter.clone(),
            contract_version: check.adapter.descriptor().contract_version,
        };
        match driver
            .drive(
                check,
                &ValidationDriveContext {
                    scope,
                    evidence,
                    extensions: &bound.extensions,
                    executor: &executor,
                    journal,
                    limits: ValidationDriverLimits::default(),
                    cancellation: &cancellation,
                    deadline,
                },
            )
            .await?
        {
            ValidationDriverOutcome::Complete(_) => {
                scope.mutation_generation = execution.mutation_generation();
                evidence.collect_validation(&driver)
            }
            ValidationDriverOutcome::AwaitExternal(_) => Err(AgentError::Config(
                "external task suspension is not installed".into(),
            )),
        }
    }

    /// Replaces duplicate driver captures with the acknowledged shared-executor records.
    async fn collect_tools(
        execution: &RunExecution,
        evidence: &mut EvaluationEvidence,
    ) -> Result<()> {
        let snapshot = execution.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        evidence.tools = payload
            .evidence
            .tool_calls
            .into_iter()
            .map(|capture| {
                let identity: EvidenceIdentity = serde_json::from_value(
                    capture
                        .record
                        .metadata
                        .get("_task_identity")
                        .cloned()
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)?,
                )?;
                Ok(ScopedObservation {
                    identity,
                    complete: true,
                    value: capture.record,
                })
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Invocation exhaustion and unanswered question timeout remain distinct from execution or ownership failure.
    fn stopped_status(execution: &RunExecution) -> TaskRunStatus {
        match execution.stop_reason().as_deref() {
            Some(
                "llm_capacity" | "tool_capacity" | "priced_capacity" | "write_capacity"
                | "time_violation",
            ) => TaskRunStatus::LimitReached,
            Some("cancel_requested") => TaskRunStatus::Cancelled,
            Some("question_timeout" | "pipeline_timeout" | "composition_timeout") => {
                TaskRunStatus::Incomplete
            }
            _ => TaskRunStatus::Failed,
        }
    }

    /// Fresh observations use one objective/cycle/stage identity; string labels do not grant execution.
    fn identity(
        scope: &EvaluationScope,
        sequence: u64,
        attempt: Option<String>,
        config_identity: String,
    ) -> EvidenceIdentity {
        EvidenceIdentity {
            key: scope.key.clone(),
            objective_revision: scope.objective_revision,
            cycle: scope.cycle,
            sequence,
            stage: scope.stage.clone(),
            attempt,
            config_identity,
            target: None,
            target_revision: None,
            mutation_generation: scope.mutation_generation,
        }
    }

    /// Returns only the committed status; limit summaries are deterministic and consume no extra provider attempt.
    async fn finish(
        &self,
        execution: &RunExecution,
        status: TaskRunStatus,
        reason: String,
        response: Option<AgentResponse>,
    ) -> Result<TaskRunResult> {
        let snapshot = execution.terminal(status, reason).await?;
        let run = TaskRun::from_checkpoint(&snapshot, &self.config_identity)?;
        let final_response = if run.status == TaskRunStatus::Completed {
            response
        } else {
            Some(AgentResponse::new(format!(
                "Task stopped ({:?}); reason: {}; turns: {}; provider attempts: {}; tool attempts: {}",
                run.status,
                run.stop_reason.as_deref().unwrap_or("execution_error"),
                run.counters.turns,
                run.counters.llm_attempts,
                run.counters.tool_attempts
            )))
        };
        Ok(TaskRunResult {
            run,
            final_response,
        })
    }
}

#[cfg(test)]
mod catalogue_race_tests {
    use super::*;
    use crate::spawner::{AgentRegistry, SpawnedAgent};

    // Delayed old cleanup must release its own pins even after a newer execution has claimed the controller's active slot.
    #[tokio::test]
    async fn old_cleanup_retires_catalogue_without_erasing_new_execution() {
        let provider = super::super::runner_tests::RecordingProvider::gated();
        let (_, runner, _) = super::super::runner_tests::fixture(provider.clone(), 2);
        let yaml = "name: CatalogueChild\nsystem_prompt: test\n";
        let child = crate::AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(super::super::runner_tests::RecordingProvider::new(&[
                "done",
            ]))
            .build()
            .unwrap();
        let registry = Arc::new(AgentRegistry::new());
        Box::pin(registry.register(SpawnedAgent::from_runtime(
            "child".into(),
            child,
            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
        )))
        .await
        .unwrap();
        let old_owner =
            RunOwner::new("old".into(), Arc::new(tokio::sync::Mutex::new(())), None).unwrap();
        old_owner
            .targets
            .install(vec![(
                "old-operation".into(),
                registry.pin_task_target("child", "CatalogueChild").unwrap(),
            )])
            .unwrap();
        let work = runner.run("objective", None);
        tokio::pin!(work);
        tokio::select! {
            _ = provider.entered.as_ref().unwrap().acquire() => {},
            result = &mut work => panic!("provider did not park: {result:?}"),
        }
        let new_execution = runner.active.lock().as_ref().unwrap().clone();
        new_execution
            .targets
            .install(vec![(
                "new-operation".into(),
                registry.pin_task_target("child", "CatalogueChild").unwrap(),
            )])
            .unwrap();
        runner.retire_owner(&old_owner);
        assert!(old_owner.targets.resolve("old-operation").is_err());
        assert!(Arc::ptr_eq(
            &runner.active.lock().as_ref().unwrap().owner,
            &new_execution.owner
        ));
        assert!(registry.remove("child").await.is_none());
        provider.release.as_ref().unwrap().add_permits(1);
        let finished = work.await.unwrap();
        assert_eq!(finished.run.status, TaskRunStatus::Completed);
        assert!(new_execution.targets.resolve("new-operation").is_err());
        assert!(registry.remove("child").await.is_some());
    }
}
