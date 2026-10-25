//! Foreground standalone execution using the existing runtime, conditional store and pure decision reducers.

use super::*;
use ai_agents_core::{AgentError, AgentResponse, Result, ToolCancellationToken};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};

pub use restoration::TaskRestoreParticipant;
pub use restoration::{
    TaskReconstructionCheckpoint, TaskRestorationReview, TaskRestoreBinding,
    TaskRestoreComposition, TaskRestoreEntry, TaskRestoreGenerations, TaskRestoreGraph,
    TaskRestoreHost, TaskRestoreManifest, TaskRestoreNode, TaskRestoreRegistry, TaskRestoreRequest,
    TaskRestoreResponse, TaskRestoreTarget,
};
#[cfg(test)]
#[path = "hosting_delivery_tests.rs"]
mod hosting_delivery_tests;
#[path = "persistence.rs"]
mod persistence;
#[path = "recovery.rs"]
mod recovery;
#[path = "restoration.rs"]
mod restoration;
pub use persistence::TaskStartupRequest;
pub use recovery::{
    TaskRecoveryCheckpoint, TaskRecoveryDecision, TaskRecoveryDisposition, TaskRecoveryHost,
};

#[path = "controller_runtime.rs"]
mod controller_runtime;
#[path = "host_runtime.rs"]
mod host_runtime;
#[path = "hosting.rs"]
mod hosting;
pub use hosting::{
    CancellationDisposition, CancellationOutcome, MAX_TASK_DISPLAY_BYTES, TaskHostDetails,
    TaskHostInteraction, TaskHostPrompt, TaskRunDisplay,
};
#[path = "live_controller.rs"]
mod live_controller;
#[path = "paused_control.rs"]
mod paused_control;

type ControllerFuture<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send + 'a>>;

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
    interaction_delivery: bool,
    restore_host: Option<Arc<dyn TaskRestoreHost>>,
    automatic_outcomes:
        parking_lot::Mutex<BTreeMap<String, std::sync::Weak<super::automatic::AutomaticOutcome>>>,
    pub(crate) actor_scope: Option<String>,
}

struct PausedTask {
    owner: Arc<RunOwner>,
    participants: Arc<super::participants::Participants>,
    targets: Arc<CompositionTargets>,
    todo: Option<RunTodoAdapter>,
    batch: Option<TaskBatchState>,
    group: Option<TaskGroupState>,
    controller: Option<Box<ControllerContinuation>>,
    whole_expiry: Option<std::time::Instant>,
    approval_deadlines: BTreeMap<String, std::time::Instant>,
    restoration: Option<restoration::LiveRestoreReview>,
}

impl Drop for PausedTask {
    /// Losing the host's retained continuation is abandonment, never implicit ownership transfer.
    fn drop(&mut self) {
        self.owner.abandon();
    }
}

impl AutonomyRunner {
    /// Retains only the exact acknowledged cursor that agrees with private live controller, batch or composition authority.
    fn capture_pause(
        &self,
        execution: &RunExecution,
    ) -> Result<(Option<TaskBatchState>, Option<TaskGroupState>)> {
        let snapshot = execution.acknowledged_snapshot();
        if snapshot.status != TaskRunStatus::Paused {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        if let Some(state) = controller_checkpoint(&payload)? {
            if serde_json::to_value(&state)? != serde_json::to_value(&*execution.controller.lock())?
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            return Ok((None, None));
        }
        if (matches!(
            payload.runtime.continuation,
            TaskContinuation::Suspended { batch: Some(_), .. }
        ) || super::interaction::has_root_interaction(&payload))
            && payload.pause_reason.as_deref() != Some("child_interaction")
        {
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

    /// Refreshes acknowledged parked state and original clocks, retaining only live pending/prior-gate review bindings rather than all historical receipts.
    fn refresh_pause(&self, paused: &mut PausedTask, execution: &RunExecution) -> Result<()> {
        (paused.batch, paused.group) = self.capture_pause(execution)?;
        paused.controller = execution.controller.lock().clone();
        for (id, deadline) in execution.approval_deadlines.lock().iter() {
            paused
                .approval_deadlines
                .entry(id.clone())
                .or_insert(*deadline);
        }
        let observation_group = paused
            .controller
            .as_ref()
            .and_then(|state| state.driver.as_ref())
            .and_then(|driver| driver.parked.as_ref())
            .and_then(|parked| parked.group.as_ref())
            .map(|value| serde_json::from_value::<Box<TaskGroupState>>(value.clone()))
            .transpose()?;
        paused.approval_deadlines.retain(|id, _| {
            observation_group.as_ref().is_some_and(|group| {
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
            }) || paused.controller.as_ref().is_some_and(|state| {
                state.review.as_ref().is_some_and(|review| &review.id == id)
                    || state
                        .driver
                        .as_ref()
                        .and_then(|driver| driver.parked.as_ref())
                        .is_some_and(|parked| &parked.pending.id == id)
            }) || paused.batch.as_ref().is_some_and(|batch| {
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
        super::response::retain_response_bindings(
            &paused.owner,
            &serde_json::to_value((&paused.batch, &paused.group, &paused.controller))?,
        );
        Ok(())
    }

    /// Claims an exact retained composition and resumes its child action before running the parent's saved finalization path.
    /// Only a fully returned, non-pending group can transfer a configured capacity wait to the controller; rejection still settles siblings.
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
        child.validate_task_batch_response(&group.batch, &response)?;
        if whole_expiry.is_some_and(|expiry| std::time::Instant::now() >= expiry) {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        let expired = request_deadline.is_some_and(|expiry| std::time::Instant::now() >= expiry)
            || group.batch.approvals[0]
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry);
        let timeout_id = (expired
            && matches!(
                &response,
                super::suspension::BatchResponse::Approval(ai_agents_hitl::ApprovalResult::Timeout)
            ))
        .then(|| group.batch.approvals[0].request_id.clone());
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
        let profile = self.retained_profile(&payload)?;
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
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let claimed = self.claim_retained(&snapshot, &mut cleanup).await?;
        let mut paused = self
            .paused
            .lock()
            .remove(&snapshot.key.run_id)
            .ok_or(TaskRunStorageError::Conflict)?;
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
                let late = request_deadline
                    .is_some_and(|expiry| std::time::Instant::now() >= expiry)
                    || group.batch.approvals[0]
                        .expires_at
                        .is_some_and(|expiry| chrono::Utc::now() >= expiry);
                if late && !expired {
                    execution.stop("response_timeout");
                    return Box::pin(self.finish_stopped_group(
                        &execution,
                        &owner,
                        &group,
                        "response_timeout".into(),
                    ))
                    .await;
                }
                let response = super::suspension::scope_timeout_approval(
                    timeout_id,
                    self.agent.resume_task_group(
                        AutonomyTurnInput {
                            owner: owner.clone(),
                            objective: payload.objective.clone(),
                            controller_message: payload.controller_state.to_string(),
                            source: group.frame.source,
                        },
                        group.clone(),
                        response,
                    ),
                )
                .await;
                if let Some(reason) = execution.stop_reason() {
                    // A real rejection still settles parked siblings; only a completed group with no pending leaf can hand capacity back to the controller.
                    let latest = execution.load_owned().await?;
                    let current: TaskCheckpointPayload = serde_json::from_value(latest.payload)?;
                    if matches!(
                        reason.as_str(),
                        "llm_capacity" | "tool_capacity" | "priced_capacity" | "write_capacity"
                    ) && current.pending.is_none()
                        && current.settings.on_limit.as_ref().is_some_and(|policy| {
                            matches!(
                                policy.action,
                                Some(LimitAction::PauseRun | LimitAction::AskUser)
                            )
                        })
                    {
                        return Box::pin(self.pause_admission_limit(
                            &execution,
                            paused.todo.as_ref(),
                            &reason,
                        ))
                        .await;
                    }
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
        // A resumed intermediate parent can replace its old descendant leaf with a local approval.
        // Discover current acknowledged leaf batches rather than trusting the original group's leaf list.
        let mut pending = Vec::new();
        for child in &payload.children {
            let Some(request) = &child.pending else {
                continue;
            };
            if request.reviewed_action.get("frame_id").is_some() {
                continue;
            }
            let adapter = payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == format!("runtime.child_batch:{}", child.child_id))
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            pending.push(super::composition::ParkedCompositionChild {
                operation: child.child_id.clone(),
                runtime_id: child.runtime.snapshot.agent_id.clone(),
                request: request.clone(),
                batch: serde_json::from_value(adapter.state.clone())?,
            });
        }
        let retired = if pending.is_empty()
            && !execution
                .parked_batches
                .lock()
                .contains_key(&self.agent.info().id)
        {
            Vec::new()
        } else {
            let mut cancellation = group.clone();
            // An empty parked list must not fall back to a completed original leaf during cancellation.
            cancellation.parked = pending;
            // Resume may have settled siblings or advanced a child request; cleanup uses the newest acknowledged batch prefix.
            cancellation.frames = execution.delegate_frames.lock().clone();
            if let Some(frame) = cancellation.frames.get(&cancellation.frame.runtime_id) {
                cancellation.frame = frame.clone();
            }
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
                super::message::settle_cancelled_messages(payload)?;
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
        let requested = self.admit_paused_cleanup(snapshot).await?;
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
        super::message::settle_cancelled_messages(&mut payload)?;
        payload.runtime =
            TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        payload.pause_reason = None;
        payload.stop_reason = Some("cancel_requested".into());
        let acknowledged = self.acknowledge_paused_cleanup(&requested, payload).await?;
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
        self.acknowledged_automatic_result(TaskRunResult {
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
        super::config::validate_runner_config(&config)?;
        agent.preflight_standalone_autonomy()?;
        let actor_scope = agent.effective_actor_id();
        Ok(Self {
            agent,
            config,
            host,
            store,
            config_identity,
            active: parking_lot::Mutex::new(None),
            paused: Default::default(),
            resume_gate: Default::default(),
            interaction_delivery: true,
            restore_host: None,
            automatic_outcomes: Default::default(),
            actor_scope,
        })
    }

    /// Binds delivery to the newly admitted automatic run without replacing the configured runner or its scoped store.
    /// Prior live delivery slots survive later admissions because an owner can release before its public result reaches its parent.
    pub(crate) fn bind_automatic_outcome(&self, outcome: Arc<super::automatic::AutomaticOutcome>) {
        if let Some(key) = &outcome.key {
            let mut outcomes = self.automatic_outcomes.lock();
            outcomes.retain(|_, slot| slot.strong_count() > 0);
            outcomes.insert(key.run_id.clone(), Arc::downgrade(&outcome));
        }
    }

    // Live callers own delivery slots; this lookup neither retains completed callers nor transfers one run's result to another.
    fn automatic_outcome(&self, run_id: &str) -> Option<Arc<super::automatic::AutomaticOutcome>> {
        self.automatic_outcomes
            .lock()
            .get(run_id)
            .and_then(std::sync::Weak::upgrade)
    }

    /// Publishes only the bound run after public-result preparation; other explicit executions cannot overwrite its delivery slot.
    /// Foreground hosts may expose pending prompts only after this boundary, never from an intermediate pause checkpoint.
    fn publish_automatic(&self, result: &TaskRunResult) {
        if let Some(session) = crate::autonomy::host_session::current_session() {
            session.publish(&result.run.key.run_id, result.run.revision);
        }
        if let Some(outcome) = self.automatic_outcome(&result.run.key.run_id) {
            outcome.publish(result);
        }
    }

    /// Captures weak delivery custody for one exact operation so a dropped response future cannot strand its ordinary composing parent.
    fn automatic_delivery_guard(
        &self,
        run_id: &str,
    ) -> Option<super::automatic::AutomaticDeliveryGuard> {
        let outcome = self.automatic_outcome(run_id)?;
        let active = self.active.lock();
        let execution = active
            .as_ref()
            .filter(|execution| execution.run_id == run_id)?;
        Some(super::automatic::AutomaticDeliveryGuard {
            owner: Arc::downgrade(&execution.owner),
            outcome: Arc::downgrade(&outcome),
        })
    }

    /// Invalid responses preserve a safe pause; only an actually abandoned or storage-uncertain bound execution fails ordinary delivery.
    fn automatic_operation_result(&self, result: Result<TaskRunResult>) -> Result<TaskRunResult> {
        if result.is_err() {
            let failed_run = self
                .active
                .lock()
                .as_ref()
                .filter(|execution| {
                    execution.owner.is_abandoned() || execution.has_storage_failure()
                })
                .map(|execution| execution.run_id.clone());
            if let Some(run_id) = failed_run
                && let Some(outcome) = self.automatic_outcome(&run_id)
            {
                outcome.failure();
            }
        }
        result
    }

    /// Returns a fully acknowledged control result after owner release, waking only that run's ordinary consumers.
    fn acknowledged_automatic_result(&self, result: TaskRunResult) -> Result<TaskRunResult> {
        self.publish_automatic(&result);
        Ok(result)
    }

    // A foreground host may withdraw delivery after EOF; absence of such a scope preserves the original embedded contract.
    pub(super) fn interaction_available(&self) -> bool {
        self.interaction_delivery
            && crate::autonomy::host_session::current_session()
                .is_none_or(|session| session.interaction_available())
    }

    /// Returns the actual scoped storage capability for trusted inspection and reviewed recovery after ambiguous operation errors.
    /// Namespace scoping is not caller authentication, and exact recovery payloads must not be exported as display events.
    pub fn task_store(&self) -> &Arc<dyn TaskRunStore> {
        &self.store
    }

    /// Returns the actual root for host control/inspection, including roots created by explicit reconstruction.
    /// Holding this handle does not bypass foreground ownership, fresh restoration review or shared execution admission.
    pub fn runtime(&self) -> &Arc<crate::RuntimeAgent> {
        &self.agent
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

    /// Requests cancellation on the actual running execution's serial boundary, without rebinding a stale display revision.
    /// None means no running handle remains; the caller must still poll its original future, and no cleanup or success is implied.
    /// An acknowledged pause/terminal winner is preserved, while errors against a still-running execution retain uncertainty and are never retried.
    pub async fn request_running_cancel(&self, run_id: &str) -> Result<Option<TaskRunSummary>> {
        let execution = self
            .active
            .lock()
            .as_ref()
            .filter(|execution| execution.run_id == run_id)
            .cloned();
        let Some(execution) = execution else {
            return Ok(None);
        };
        if execution.acknowledged_snapshot().status != TaskRunStatus::Running {
            return Ok(None);
        }
        match execution.request_cancel().await {
            Err(_) if execution.acknowledged_snapshot().status != TaskRunStatus::Running => {
                Ok(None)
            }
            result => result.map(Some),
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
                        | TaskEffectState::Suspended
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
        execution.owner.work.acknowledge_reconciliation()?;
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
    /// Every successful branch publishes only after cleanup acknowledgement, including controller waits consumed by ordinary composing parents.
    pub async fn cancel_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
    ) -> Result<TaskRunResult> {
        let _delivery = self.automatic_delivery_guard(run_id);
        let result = Box::pin(self.cancel_paused_inner(run_id, expected_revision)).await;
        let result = self.automatic_operation_result(result)?;
        self.acknowledged_automatic_result(result)
    }

    /// Performs conditional typed cleanup; only unavailable-host retirement may accept its exact cancellation-only successor before admission.
    /// The public boundary separately reports uncertain delivery to a composing host; arbitrary revision changes still reject.
    async fn cancel_paused_inner(
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
        if snapshot.status != TaskRunStatus::Paused {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let cancellation_only = snapshot.cancel_requested
            && paused_control::accept_retirement_cancellation(&snapshot, expected_revision)?;
        if snapshot.revision != expected_revision && !cancellation_only {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        if controller_checkpoint(&payload)?.is_some() {
            return self.cancel_controller(snapshot, payload).await;
        }
        if (matches!(
            payload.runtime.continuation,
            TaskContinuation::Suspended { batch: None, .. }
        ) && !super::interaction::has_root_interaction(&payload))
            || payload.pause_reason.as_deref() == Some("child_interaction")
        {
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
        let requested = self.admit_paused_cleanup(snapshot).await?;
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
        let acknowledged = self.acknowledge_paused_cleanup(&requested, payload).await?;
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
        self.acknowledged_automatic_result(TaskRunResult {
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
        let message_group = execution
            .delegate_frames
            .lock()
            .get(&self.agent.info().id)
            .is_some_and(|frame| {
                frame.dispatch == super::composition::CompositionDispatch::ToolBatch
            });
        let snapshot = if execution
            .parked_batches
            .lock()
            .contains_key(&self.agent.info().id)
            && !message_group
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

    /// Validates one bound approval, question answer or external signal before claiming its same-process exact continuation.
    /// The implementation-owned question contract is checked before claim and final authorization still runs on resume.
    /// A losing or invalid response cannot mutate the runtime, and failed execution retains the owner for recovery.
    /// Opted-in persistence prepares any returned safe pause only after the retained resume gate has been released.
    pub async fn resume(
        &self,
        run_id: &str,
        expected_revision: u64,
        input: TaskResumeInput,
    ) -> Result<TaskRunResult> {
        let _delivery = self.automatic_delivery_guard(run_id);
        let result = async {
            let result = Box::pin(self.resume_inner(run_id, expected_revision, input)).await?;
            self.persist_returned_pause(result).await
        }
        .await;
        self.automatic_operation_result(result)
    }

    // The retained gate is released before optional restart preparation reacquires it; pending work remains owner-protected throughout.
    async fn resume_inner(
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
        self.ensure_restoration_review(&payload, run_id)?;
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
        if controller_checkpoint(&payload)?.is_some() {
            return Box::pin(self.resume_controller(snapshot, payload, input)).await;
        }
        let capacity = payload
            .pending
            .as_ref()
            .filter(|pending| matches!(pending.kind, TaskPendingKind::Limit));
        let (request_id, mut result, question_response) = if let Some(pending) = capacity {
            self.validate_capacity_response(&payload, &input)?;
            (
                pending.id.clone(),
                super::suspension::BatchResponse::Capacity,
                false,
            )
        } else {
            match input {
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
                TaskResumeInput::Continue | TaskResumeInput::ExternalSignal(_) => {
                    return Err(TaskRunStorageError::NotResumable.into());
                }
            }
        };
        if payload.pending.as_ref().is_none_or(|pending| {
            pending.id != request_id
                || if matches!(result, super::suspension::BatchResponse::Capacity) {
                    !matches!(pending.kind, TaskPendingKind::Limit)
                } else if question_response {
                    !matches!(pending.kind, TaskPendingKind::UserQuestion)
                } else {
                    !matches!(pending.kind, TaskPendingKind::Approval)
                }
        }) || payload.consumed_request_ids.contains(&request_id)
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        if (matches!(
            payload.runtime.continuation,
            TaskContinuation::Suspended { batch: None, .. }
        ) && !super::interaction::has_root_interaction(&payload))
            || payload.pause_reason.as_deref() == Some("child_interaction")
        {
            return Box::pin(self.resume_group(snapshot, payload, request_id, result)).await;
        }
        let adapter = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.tool_batch")
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let batch: TaskBatchState = serde_json::from_value(adapter.state.clone())?;
        batch.validate_checkpoint(&payload)?;
        self.agent.validate_task_batch_response(&batch, &result)?;
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
        let original_expired = approval_deadline
            .is_some_and(|expiry| std::time::Instant::now() >= expiry)
            || batch.approvals[0]
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry);
        let timeout_id = (original_expired
            && matches!(
                &result,
                super::suspension::BatchResponse::Approval(ai_agents_hitl::ApprovalResult::Timeout)
            ))
        .then(|| batch.approvals[0].request_id.clone());
        if original_expired {
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
        let profile = self.retained_profile(&payload)?;
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
            .map(Ok)
            .unwrap_or_else(|| {
                LifecycleState::new(
                    &profile.settings,
                    profile.strategy == ExecutionStrategy::Skill,
                )
            })?;
        let owner = self
            .paused
            .lock()
            .get(run_id)
            .map(|paused| paused.owner.clone())
            .ok_or_else(|| {
                AgentError::Config("this checkpoint needs compatible live-owner restoration".into())
            })?;
        owner.check()?;
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let claimed = self.claim_retained(&snapshot, &mut cleanup).await?;
        let mut paused = self
            .paused
            .lock()
            .remove(run_id)
            .ok_or(TaskRunStorageError::Conflict)?;
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
            *live.approval_deadlines.lock() = paused.approval_deadlines.clone();
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
                // A claim or active-clock acknowledgement may cross the original pending deadline without admitting the operation.
                let late = approval_deadline
                    .is_some_and(|expiry| std::time::Instant::now() >= expiry)
                    || batch.approvals[0]
                        .expires_at
                        .is_some_and(|expiry| chrono::Utc::now() >= expiry);
                if late && !original_expired {
                    let runtime = self.agent.cancel_task_batch(&owner, batch.clone()).await?;
                    execution
                        .update(|payload| {
                            payload.runtime = runtime;
                            payload.pending = None;
                            payload.pause_reason = None;
                            for approval in &batch.approvals {
                                if !payload.consumed_request_ids.contains(&approval.request_id) {
                                    payload
                                        .consumed_request_ids
                                        .push(approval.request_id.clone());
                                }
                            }
                            Ok(())
                        })
                        .await?;
                    return self
                        .finish(
                            &execution,
                            TaskRunStatus::Incomplete,
                            "response_timeout".into(),
                            None,
                        )
                        .await;
                }
                let response = super::suspension::scope_timeout_approval(
                    timeout_id,
                    self.agent.resume_task_batch(
                        AutonomyTurnInput {
                            owner: owner.clone(),
                            objective: payload.objective.clone(),
                            controller_message: payload.controller_state.to_string(),
                            source: AutonomyTurnSource::ResumeAfterApproval,
                        },
                        batch.clone(),
                        &request_id,
                        result,
                    ),
                )
                .await;
                if matches!(&response, Err(AgentError::TaskSuspended(_))) {
                    return self.pause(&execution, paused.todo.as_ref()).await;
                }
                if let Some(reason) = execution.stop_reason()
                    && matches!(
                        reason.as_str(),
                        "llm_capacity" | "tool_capacity" | "priced_capacity" | "write_capacity"
                    )
                {
                    if !self.interaction_available() {
                        let latest = execution.load_owned().await?;
                        let current: TaskCheckpointPayload =
                            serde_json::from_value(latest.payload)?;
                        if current.pending.is_some() {
                            let runtime =
                                self.agent.cancel_task_batch(&owner, batch.clone()).await?;
                            execution
                                .update(|payload| {
                                    payload.runtime = runtime;
                                    if let Some(pending) = payload.pending.take() {
                                        payload.consumed_request_ids.push(pending.id);
                                    }
                                    payload.pause_reason = None;
                                    Ok(())
                                })
                                .await?;
                        }
                    }
                    return Box::pin(self.pause_admission_limit(
                        &execution,
                        paused.todo.as_ref(),
                        &reason,
                    ))
                    .await;
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

    /// Runs a bounded foreground objective after sampling required host targets; unsuccessful writes never publish success.
    /// Revision sources are sampled state, not providers or execution permits, and are rechecked at continuation/publication boundaries.
    /// Opted-in persistence prepares the acknowledged safe pause before returning it; failed preparation never reports a durable restart point.
    pub async fn run(&self, objective: &str, selected: Option<&str>) -> Result<TaskRunResult> {
        let result = Box::pin(self.run_entry(objective, selected, None, None)).await?;
        self.persist_returned_pause(result).await
    }

    /// Executes one admitted entry under the same controller; automatic routing never creates another run here.
    async fn run_entry(
        &self,
        objective: &str,
        selected: Option<&str>,
        skill: Option<ai_agents_skills::SkillDefinition>,
        automatic: Option<Box<super::automatic::AutomaticEntry>>,
    ) -> Result<TaskRunResult> {
        let binding = automatic.as_ref().map(|entry| &entry.binding);
        let profile = resolve_profile(
            &self.config,
            binding.map_or(
                if skill.is_some() {
                    AutonomyScope::Skill
                } else {
                    AutonomyScope::Task
                },
                |binding| binding.scope(),
            ),
            selected,
            binding
                .map(|binding| &binding.settings)
                .or_else(|| skill.as_ref().and_then(|skill| skill.autonomy.as_ref())),
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
        let objective = if let Some(binding) = binding {
            self.agent
                .render_automatic_objective(&profile.settings, binding)?
        } else {
            let objective = if objective.trim().is_empty() {
                profile.settings.objective.as_deref().unwrap_or(objective)
            } else {
                objective
            };
            if let Some(template) = &profile.settings.objective_template {
                self.agent.render_autonomy_objective(template, objective)?
            } else {
                objective.to_string()
            }
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
        self.preflight_persistence(&profile.settings)?;

        let lifecycle = LifecycleState::new(
            &profile.settings,
            binding.map_or(skill.is_some(), |binding| {
                binding.scope() == AutonomyScope::Skill
            }),
        )?;
        let mut preflight_scope = EvaluationScope {
            key: TaskRunKey {
                agent_id: self.agent.info().id,
                run_id: String::new(),
            },
            objective_revision: 0,
            cycle: 0,
            stage: None,
            mutation_generation: 0,
            target_revisions: BTreeMap::new(),
            target_generations: BTreeMap::new(),
            validation_bindings: BTreeMap::new(),
            validation_attempts: BTreeMap::new(),
            validation_dependencies: BTreeMap::new(),
        };
        bound.capture_targets(&mut preflight_scope)?;
        let (owner, activation) = if let Some(entry) = automatic {
            (entry.owner, Some(entry.binding))
        } else {
            (
                self.agent
                    .reserve_autonomy_run(uuid::Uuid::new_v4().to_string())
                    .await?,
                None,
            )
        };
        let run_id = owner.run_id.clone();
        *owner.skill.write() = if activation
            .as_ref()
            .is_some_and(|binding| binding.scope() != AutonomyScope::Skill)
        {
            None
        } else {
            skill.clone()
        };
        *owner.prepared.lock() = activation.clone().map(|mut binding| {
            if binding.scope() == AutonomyScope::Skill
                && profile.strategy != ExecutionStrategy::Skill
            {
                binding.skill = None;
            }
            binding
        });
        owner.interaction_delivery.store(
            self.interaction_available(),
            std::sync::atomic::Ordering::Release,
        );
        // The scoped host catalogue retains only this exact configured runner; registration inside setup remains covered by pre-execution cleanup.
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
            if let Some(session) = crate::autonomy::host_session::current_session()
                && let Some(runner) = self.agent.automatic_task_runner()
                && std::ptr::eq(self, Arc::as_ptr(&runner))
            {
                session.register(runner, key.clone())?;
            }
            let mut payload = TaskCheckpointPayload::new(
                objective.clone(),
                self.config_identity.clone(),
                &profile,
                TaskRuntimeCheckpoint::between_turns(self.agent.save_state().await?)?,
            );
            if let Some(binding) = &activation {
                payload.adapters.push(TaskAdapterCheckpoint {
                    id: "runtime.activation".into(),
                    adapter: "runtime.activation".into(),
                    contract_version: 1,
                    config: serde_json::to_value(binding)?,
                    state: json!({}),
                });
            }
            if let Some(skill) = &skill {
                payload.adapters.push(TaskAdapterCheckpoint {
                    id: "runtime.entry".into(),
                    adapter: "runtime.entry".into(),
                    contract_version: 1,
                    config: serde_json::to_value(skill)?,
                    state: json!({}),
                });
            }
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
        if let Some(session) = crate::autonomy::host_session::current_session() {
            session.admitted(&run_id);
        }
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
                if !execution.has_storage_failure()
                    && !matches!(
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
                    restoration: None,
                    participants: execution.participants.clone(),
                    targets: execution.targets.clone(),
                    owner,
                    todo,
                    whole_expiry: execution.expiry_projection,
                    approval_deadlines: execution.approval_deadlines.lock().clone(),
                    batch,
                    group,
                    controller: execution.controller.lock().clone(),
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

    /// The controller shares one ledger with validation; current target samples invalidate old proof without relabelling completed effects.
    /// Required stage review and final completion recheck frozen host sources after storage waits; run CAS is not an external-target transaction.
    /// Future erasure isolates retained validation data from nested model-message polling stacks.
    /// Per-schedule prefixes and selected replans resume after committed work without repeating progress or decision accounting.
    #[allow(clippy::too_many_arguments)]
    fn execute<'a>(
        &'a self,
        owner: &'a Arc<RunOwner>,
        execution: &'a Arc<RunExecution>,
        bound: &'a BoundAutonomyProfile,
        profile: &'a EffectiveAutonomyProfile,
        key: &'a TaskRunKey,
        objective: &'a str,
        mut lifecycle: LifecycleState,
        todo: Option<&'a RunTodoAdapter>,
        mut resumed_response: Option<AgentResponse>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TaskRunResult>> + Send + 'a>>
    {
        Box::pin(async move {
            let restored = execution.load_owned().await?;
            let restored: TaskCheckpointPayload = serde_json::from_value(restored.payload)?;
            super::events::emit(execution, if restored.counters.turns == 0 { super::events::TaskRunEventKind::TaskStarted } else { super::events::TaskRunEventKind::TaskResumed }, || json!({"objective_revision":restored.objective_revision,"counters":restored.counters})).await;
            let controller = restored
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.controller");
            let mut scope = EvaluationScope {
                key: key.clone(),
                objective_revision: restored.objective_revision,
                cycle: 0,
                stage: None,
                mutation_generation: 0,
                target_revisions: BTreeMap::new(),
                target_generations: BTreeMap::new(),
                validation_bindings: BTreeMap::new(),
                validation_attempts: BTreeMap::new(),
                validation_dependencies: BTreeMap::new(),
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
                scope_mode: super::automatic::binding(&restored)?.map_or_else(
                    || "task".to_string(),
                    |binding| {
                        if binding.scope() == AutonomyScope::State {
                            "state".into()
                        } else {
                            "skill".into()
                        }
                    },
                ),
                run_revision: execution.revision.clone(),
                validator: String::new(),
                contract_version: 1,
            };
            if let Some(controller) = controller {
                evidence = serde_json::from_value(controller.state["evidence"].clone())?;
            }
            let mut continuation = execution.controller.lock().take();
            let mut phase = continuation
                .as_ref()
                .map_or(super::controller::ControllerPhase::EachCycle, |saved| {
                    saved.phase
                });
            let mut last_response = None;
            let mut instruction = String::new();
            let mut source = if restored.objective_revision == 0 {
                AutonomyTurnSource::InitialObjective
            } else {
                AutonomyTurnSource::Continuation
            };
            let mut forced_continuation = false;
            if let Some(saved) = &continuation {
                scope = saved.scope.clone();
                lifecycle = saved.lifecycle.clone();
                progress = saved.progress.clone();
                evidence = saved.evidence.clone();
                last_response = Some(saved.response.clone());
                instruction = saved.instruction.clone();
                source = saved.source;
                forced_continuation = saved.forced_continuation;
                execution.set_scope(&scope);
            }
            // Required-check inventory is structural configuration, not old-objective completion credit.
            evidence.required_checks = bound
                .checks
                .iter()
                .filter(|check| check.check.required.unwrap_or(true))
                .map(|check| check.check.id.clone())
                .collect();
            let reviewed = continuation.as_ref().and_then(|saved| saved.review.clone());
            let capacity = reviewed.as_ref().and_then(|review| match &review.proposal {
                super::controller::ControllerProposal::Capacity { limit }
                | super::controller::ControllerProposal::Question {
                    limit: Some(limit), ..
                } => Some(limit.clone()),
                _ => None,
            });
            let mut resuming_turn = capacity.as_ref().is_some_and(|limit| limit.resume_turn);
            let mut pending_model = capacity.as_ref().and_then(|limit| limit.model.clone());
            let mut pending_skill = capacity.and_then(|limit| limit.skill);
            if let Some(review) = &reviewed {
                match &review.proposal {
                    super::controller::ControllerProposal::Continue => {
                        continuation = None;
                        last_response = None;
                        source = AutonomyTurnSource::Continuation;
                        phase = super::controller::ControllerPhase::EachCycle;
                    }
                    super::controller::ControllerProposal::Plan {
                        instruction: approved,
                    } => {
                        instruction = approved.clone();
                        continuation = None;
                        source = AutonomyTurnSource::Continuation;
                        phase = super::controller::ControllerPhase::EachCycle;
                    }
                    super::controller::ControllerProposal::Question { .. }
                    | super::controller::ControllerProposal::Capacity { .. } => {
                        if continuation
                            .as_ref()
                            .is_some_and(|saved| saved.driver.is_some())
                        {
                            continuation.as_mut().unwrap().review = None;
                        } else {
                            continuation = None;
                            phase = super::controller::ControllerPhase::EachCycle;
                        }
                        source = AutonomyTurnSource::Continuation;
                    }
                    super::controller::ControllerProposal::Stage { .. } => {}
                }
            }
            let reviewed_scope = continuation.as_ref().map(|saved| saved.scope.clone());
            let reviewed_stage_admission = reviewed.clone();
            let mut reviewed_plan_admission = reviewed
                .as_ref()
                .filter(|review| {
                    matches!(
                        review.proposal,
                        super::controller::ControllerProposal::Plan { .. }
                    )
                })
                .cloned();
            let mut reviewed_stage = reviewed.and_then(|review| match review.proposal {
                super::controller::ControllerProposal::Stage { lifecycle } => Some(lifecycle),
                _ => None,
            });
            loop {
                // Replanning is already this observation's selected decision; resume only its managed planner work.
                if phase == super::controller::ControllerPhase::Replan {
                    let mut state = continuation
                        .take()
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                    if let Some(result) = self
                        .continue_replan(
                            execution, todo, bound, &executor, &journal, objective, &mut state,
                        )
                        .await?
                    {
                        return Ok(result);
                    }
                    scope = state.scope;
                    evidence = state.evidence;
                    instruction = state.instruction;
                    resuming_turn = false;
                    phase = super::controller::ControllerPhase::EachCycle;
                    source = AutonomyTurnSource::Continuation;
                }
                // Validation resumes after the committed turn and must not repeat its counters, observation or hooks.
                if continuation.is_none() {
                    let snapshot = execution.load_owned().await?;
                    let payload: TaskCheckpointPayload =
                        serde_json::from_value(snapshot.payload.clone())?;
                    if execution.stop_reason().is_none()
                        && resumed_response.is_none()
                        && !resuming_turn
                        && payload.counters.turns >= u64::from(payload.limits.max_turns)
                    {
                        let state = ControllerContinuation::decision(
                            &scope,
                            &lifecycle,
                            &progress,
                            &evidence,
                            last_response.as_ref(),
                            &instruction,
                            source,
                            forced_continuation,
                        );
                        return Box::pin(self.park_limit(execution, todo, state, "turn_capacity"))
                            .await;
                    }
                    if execution.check(&payload).is_err() {
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
                    if resumed_response.is_none() && !resuming_turn {
                        scope.cycle += 1;
                    }
                    scope.mutation_generation = execution.mutation_generation();
                    scope.stage = profile
                        .settings
                        .lifecycle
                        .as_ref()
                        .and_then(|stages| stages.get(lifecycle.current_stage))
                        .map(|stage| stage.id.clone());

                    bound.capture_targets(&mut scope)?;
                    execution.set_scope(&scope);
                    let starting_stage =
                        payload.stage != scope.stage || payload.counters.turns == 0;
                    // This is an advisory view; live admission remains authoritative if time passes before dispatch.
                    let controller = json!({"objective":objective,"stage":scope.stage,"instruction":instruction,
                "todos":todo.map(RunTodoAdapter::checkpoint).transpose()?,
                "remaining_turns":(payload.limits.max_turns as u64).saturating_sub(payload.counters.turns),
                "remaining_llm_calls":(payload.limits.max_llm_calls as u64).saturating_sub(payload.counters.llm_attempts),
                "remaining_tool_calls":(payload.limits.max_tool_calls as u64).saturating_sub(payload.counters.tool_attempts),
                "remaining_command_calls":(payload.limits.max_command_calls as u64).saturating_sub(payload.counters.command_attempts),
                "remaining_active_seconds":payload.limits.max_active_time_seconds.saturating_mul(1000).saturating_sub(execution.consumed_active_millis()?)/1000,
                "remaining_wall_seconds":payload.clocks.expires_at.map(|expiry|(expiry-chrono::Utc::now()).num_seconds().max(0)),
                "remaining_micro_usd":payload.limits.max_micro_usd.map(|limit| limit.saturating_sub(payload.counters.charged_micro_usd).saturating_sub(payload.reservations.iter().filter(|reservation| matches!(reservation.state, TaskEffectState::Reserved | TaskEffectState::Dispatched | TaskEffectState::Suspended | TaskEffectState::Uncertain)).map(|reservation| reservation.reserved_micro_usd.saturating_sub(reservation.charged_micro_usd)).sum::<u64>())),
                "remaining_declared_write_paths":payload.limits.max_declared_write_paths.map(|limit| (limit as usize).saturating_sub(payload.declared_write_targets.len())),
                                "failed_gates":payload.evidence.records.last().and_then(|record| record.data.get("completion")).cloned(),
                "skill_results":super::automatic::skill_results(&payload),
                "progress_instruction":profile.settings.progress.as_ref().and_then(|progress| progress.auto_create_todo_prompt.clone()).unwrap_or_else(|| "Before progress-gated work, maintain the run todo list and select exactly one active item when required.".into()),
                "stage_instruction":profile.settings.lifecycle.as_ref().and_then(|stages| stages.get(lifecycle.current_stage)).and_then(|stage| stage.instruction.clone())});
                    execution
                .update(|payload| {
                    if resumed_response.is_none() && !resuming_turn { payload.counters.turns += 1; }
                    if forced_continuation && resumed_response.is_none() && !resuming_turn {
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
                    // Delivery follows the turn checkpoint and holds no ledger lock; a slow consumer cannot extend its execution authority.
                    if starting_stage && scope.stage.is_some() {
                        super::events::emit(execution, super::events::TaskRunEventKind::StageStarted, || json!({"stage":scope.stage.as_deref().map(super::events::safe_text),"cycle":scope.cycle})).await;
                    }
                    if forced_continuation {
                        super::events::emit(
                            execution,
                            super::events::TaskRunEventKind::ContinuationForced,
                            || json!({"cycle":scope.cycle}),
                        )
                        .await;
                    }
                    let response = if let Some(response) = resumed_response.take() {
                        Ok(response)
                    } else {
                        if let Some(review) = reviewed_plan_admission.take()
                            && !self.controller_review_current(execution, &review)?
                        {
                            return self
                                .finish(
                                    execution,
                                    TaskRunStatus::Incomplete,
                                    "supervision_timeout".into(),
                                    last_response,
                                )
                                .await;
                        }
                        let input = AutonomyTurnInput {
                            owner: owner.clone(),
                            objective: objective.into(),
                            controller_message: controller.to_string(),
                            source: pending_skill.as_ref().map_or(source, |skill| skill.source),
                        };
                        if pending_skill.is_some() || pending_model.is_some() {
                            self.agent
                                .run_autonomy_turn_with_skill(
                                    input,
                                    pending_skill.take(),
                                    pending_model.take(),
                                )
                                .await
                        } else {
                            self.agent.run_autonomy_turn(input).await
                        }
                    };
                    match response {
                        Ok(response) => {
                            super::events::emit(execution, super::events::TaskRunEventKind::TurnCompleted, || json!({"cycle":scope.cycle,"response":super::events::safe_text(&response.content)})).await;
                            last_response = Some(response);
                        }
                        Err(AgentError::TaskSuspended(_)) => {
                            return self.pause(execution, todo).await;
                        }
                        Err(_)
                            if matches!(
                                execution.stop_reason().as_deref(),
                                Some(
                                    "llm_capacity"
                                        | "tool_capacity"
                                        | "priced_capacity"
                                        | "write_capacity"
                                )
                            ) => {}
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
                        if matches!(
                            reason.as_str(),
                            "llm_capacity" | "tool_capacity" | "priced_capacity" | "write_capacity"
                        ) {
                            let runtime = TaskRuntimeCheckpoint::between_turns(
                                self.agent.save_state_full().await?,
                            )?;
                            execution
                                .update(|payload| {
                                    payload.runtime = runtime;
                                    Ok(())
                                })
                                .await?;
                            let state = ControllerContinuation::decision(
                                &scope,
                                &lifecycle,
                                &progress,
                                &evidence,
                                last_response.as_ref(),
                                &instruction,
                                source,
                                forced_continuation,
                            );
                            return Box::pin(self.park_limit(execution, todo, state, &reason))
                                .await;
                        }
                        return self
                            .finish(
                                execution,
                                Self::stopped_status(execution),
                                reason,
                                last_response,
                            )
                            .await;
                    }
                    // The next action will be selected afresh; a repair source must not leak into later ordinary cycles.
                    if source == AutonomyTurnSource::ValidationFix {
                        source = AutonomyTurnSource::Continuation;
                    }
                    lifecycle.skill_segment_completed |= execution.skill_segment_completed();
                    execution.await_children().await?;
                    execution.await_owned_work().await?;
                    Box::pin(self.agent.flush_background_tasks()).await?;
                    let runtime =
                        TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
                    let todos = todo.map(RunTodoAdapter::checkpoint).transpose()?;
                    execution
                        .update(|payload| {
                            payload.runtime = runtime;
                            payload.todos = todos.clone();
                            payload.last_response =
                                last_response.as_ref().map(|r| r.content.clone());
                            Ok(())
                        })
                        .await?;
                    if let Some(todos) = &todos {
                        super::events::emit(execution, super::events::TaskRunEventKind::TodoUpdated, || json!({"count":todos.items.len(),"items":super::events::safe_value(&serde_json::to_value(&todos.items).unwrap_or(serde_json::Value::Null))})).await;
                    }
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
                    self.checkpoint_persistence_step(execution, &profile.settings, scope.cycle)
                        .await?;
                }
                // Each schedule has its own saved prefix; stage/task eligibility is evaluated after earlier observations.
                use super::controller::ControllerPhase;
                for check_phase in [
                    ControllerPhase::EachCycle,
                    ControllerPhase::StageEnd,
                    ControllerPhase::Completion,
                ] {
                    if phase == ControllerPhase::Decision {
                        break;
                    }
                    if continuation
                        .as_ref()
                        .is_some_and(|saved| saved.phase != check_phase)
                    {
                        continue;
                    }
                    let checks = if let Some(saved) = &continuation {
                        let schedule = match check_phase {
                            ControllerPhase::EachCycle => ValidationSchedule::EachCycle,
                            ControllerPhase::StageEnd => ValidationSchedule::StageEnd,
                            _ => ValidationSchedule::Completion,
                        };
                        let available = bound.scheduled_checks(&scope, schedule);
                        saved
                            .checks
                            .iter()
                            .map(|id| {
                                available
                                    .iter()
                                    .find(|check| &check.check.id == id)
                                    .copied()
                                    .ok_or(TaskRunStorageError::InvalidCheckpoint)
                            })
                            .collect::<std::result::Result<Vec<_>, _>>()?
                    } else {
                        bound.controller_checks(&scope, &lifecycle, &evidence, check_phase)?
                    };
                    let ids: Vec<_> = checks.iter().map(|check| check.check.id.clone()).collect();
                    let next = continuation.as_ref().map_or(0, |saved| saved.next_check);
                    for (index, check) in checks.into_iter().enumerate().skip(next) {
                        let evaluator = CompletionGateEvaluator::default();
                        let candidate_gate = check
                            .stage
                            .as_ref()
                            .and_then(|id| {
                                profile
                                    .settings
                                    .lifecycle
                                    .as_ref()?
                                    .iter()
                                    .find(|stage| &stage.id == id)?
                                    .completion
                                    .as_ref()
                            })
                            .or_else(|| {
                                if check.stage.is_none() {
                                    profile.settings.completion.as_ref()
                                } else {
                                    None
                                }
                            });
                        if check_phase == ControllerPhase::Completion
                            && !check.check.required.unwrap_or(true)
                            && candidate_gate.is_some_and(|gate| {
                                evaluator
                                    .evaluate(gate, &scope, &evidence)
                                    .is_ok_and(|result| result.outcome == GateOutcome::Pass)
                            })
                        {
                            continue;
                        }
                        let driver = continuation.as_mut().and_then(|saved| saved.driver.take());
                        if let Some(driver) = Box::pin(self.validate(
                            check,
                            &mut scope,
                            &mut evidence,
                            execution,
                            bound,
                            &executor,
                            &journal,
                            driver,
                        ))
                        .await?
                        {
                            let capacity_reason = driver.capacity_reason.clone();
                            *execution.controller.lock() = Some(Box::new(ControllerContinuation {
                                scope: scope.clone(),
                                lifecycle: lifecycle.clone(),
                                progress: progress.clone(),
                                evidence: evidence.clone(),
                                response: last_response
                                    .clone()
                                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?,
                                instruction: instruction.clone(),
                                source,
                                forced_continuation,
                                checks: ids.clone(),
                                next_check: index,
                                phase: check_phase,
                                driver: Some(driver),
                                review: None,
                            }));
                            if let Some(reason) = capacity_reason {
                                let state = *execution
                                    .controller
                                    .lock()
                                    .take()
                                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                                return Box::pin(self.park_limit(execution, todo, state, &reason))
                                    .await;
                            }
                            return Box::pin(self.pause_controller(execution, todo)).await;
                        }
                    }
                    continuation = None;
                }
                continuation = None;
                resuming_turn = false;
                phase = ControllerPhase::EachCycle;
                execution
                    .update(|payload| {
                        retire_controller(payload)?;
                        payload.pause_reason = None;
                        Ok(())
                    })
                    .await?;
                scope.mutation_generation = execution.mutation_generation();
                bound.capture_targets(&mut scope)?;
                Self::collect_tools(execution, &mut evidence).await?;
                // The existing post-validation journal ACK seals numeric collector events; telemetry never substitutes for admission.
                let telemetry_snapshot = execution.load_owned().await?;
                let telemetry_payload: TaskCheckpointPayload =
                    serde_json::from_value(telemetry_snapshot.payload)?;
                evidence.observability = super::telemetry::observation(
                    &telemetry_payload,
                    Self::identity(&scope, scope.cycle, None, String::new()),
                )?;
                bound.extensions.ensure_targets_current(&scope)?;
                if reviewed_stage.is_none() {
                    let observation = bound.observe_progress(&scope, &evidence)?;
                    progress.observe(&scope, &observation)?;
                    evidence.progress = Some(observation.scoped(Self::identity(
                        &scope,
                        scope.cycle,
                        None,
                        String::new(),
                    )));
                }
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
                    let approved = reviewed_stage.take();
                    let mut proposed = approved.clone().unwrap_or_else(|| lifecycle.clone());
                    let advances = approved.is_some()
                        || proposed.complete_stage(
                            &profile.settings,
                            &stage.id,
                            gate,
                            bound.required_stage_outcome(&scope, &evidence)?,
                        )?;
                    let rewinds = !advances
                        && (gate == GateOutcome::Fail
                            || bound.required_stage_outcome(&scope, &evidence)?
                                == GateOutcome::Fail)
                        && proposed.validation_failure(&profile.settings)?;
                    if advances || rewinds {
                        if approved.is_none()
                            && (profile.settings.supervision.as_ref().is_some_and(|policy| {
                                policy.stage_changes == Some(ReviewPolicy::RequireApproval)
                            }) || profile.settings.hitl.as_ref().is_some_and(|policy| {
                                policy.on_approval_required == Some(InteractionAction::AskEachTime)
                            }))
                        {
                            return Box::pin(
                                self.park_review(
                                    execution,
                                    todo,
                                    ControllerContinuation {
                                        scope: scope.clone(),
                                        lifecycle: lifecycle.clone(),
                                        progress: progress.clone(),
                                        evidence: evidence.clone(),
                                        response: last_response
                                            .clone()
                                            .ok_or(TaskRunStorageError::InvalidCheckpoint)?,
                                        instruction: instruction.clone(),
                                        source,
                                        forced_continuation,
                                        checks: Vec::new(),
                                        next_check: 0,
                                        phase: ControllerPhase::Decision,
                                        driver: None,
                                        review: None,
                                    },
                                    super::controller::ControllerProposal::Stage {
                                        lifecycle: proposed,
                                    },
                                ),
                            )
                            .await;
                        }
                        if approved.is_some() {
                            if let Some(original) = &reviewed_scope {
                                bound.extensions.ensure_targets_current(original)?;
                            }
                            bound.extensions.ensure_targets_current(&scope)?;
                            let review = reviewed_stage_admission
                                .as_ref()
                                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                            if !self.controller_review_current(execution, review)? {
                                return self
                                    .finish(
                                        execution,
                                        TaskRunStatus::Incomplete,
                                        "supervision_timeout".into(),
                                        last_response,
                                    )
                                    .await;
                            }
                        }
                        if proposed.current_stage < lifecycle.current_stage {
                            source = AutonomyTurnSource::ValidationFix;
                        }
                        lifecycle = proposed;
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
                        scope_ended: self.agent.automatic_scope_ended(),
                        additional_work_capacity: execution.stop_reason().is_none()
                            && payload.counters.turns < u64::from(payload.limits.max_turns)
                            && payload.counters.llm_attempts
                                < u64::from(payload.limits.max_llm_calls),
                        interaction_available: self.interaction_available(),
                    },
                )?;
                // Publish the consumed allowance and selected decision, not the advisory pre-turn values, into the result checkpoint.
                let consumed_active = execution.consumed_active_millis()?;
                execution.update(|payload| {
                payload.controller_state["remaining_turns"] = json!((payload.limits.max_turns as u64).saturating_sub(payload.counters.turns));
                payload.controller_state["remaining_llm_calls"] = json!((payload.limits.max_llm_calls as u64).saturating_sub(payload.counters.llm_attempts));
                payload.controller_state["remaining_tool_calls"] = json!((payload.limits.max_tool_calls as u64).saturating_sub(payload.counters.tool_attempts));
                payload.controller_state["remaining_command_calls"] = json!((payload.limits.max_command_calls as u64).saturating_sub(payload.counters.command_attempts));
                let reserved = payload.reservations.iter().filter(|reservation| matches!(reservation.state, TaskEffectState::Reserved | TaskEffectState::Dispatched | TaskEffectState::Suspended | TaskEffectState::Uncertain))
                    .try_fold(0u64, |total, reservation| total.checked_add(reservation.reserved_micro_usd.saturating_sub(reservation.charged_micro_usd)))
                    .ok_or_else(|| AgentError::Config("controller commitment overflow".into()))?;
                payload.controller_state["remaining_micro_usd"] = json!(payload.limits.max_micro_usd.map(|limit| limit.saturating_sub(payload.counters.charged_micro_usd).saturating_sub(reserved)));
                payload.controller_state["remaining_declared_write_paths"] = json!(payload.limits.max_declared_write_paths.map(|limit| (limit as usize).saturating_sub(payload.declared_write_targets.len())));
                payload.controller_state["remaining_active_seconds"] = json!(payload.limits.max_active_time_seconds.saturating_mul(1000).saturating_sub(consumed_active)/1000);
                payload.controller_state["remaining_wall_seconds"] = json!(payload.clocks.expires_at.map(|expiry| (expiry-chrono::Utc::now()).num_seconds().max(0)));
                payload.controller_state["lifecycle"] = serde_json::to_value(&lifecycle)?;
                payload.controller_state["decision"] = serde_json::to_value(&decision)?;
                payload.controller_state["progress"] = serde_json::to_value(&progress)?;
                payload.controller_state["failed_gates"] = serde_json::to_value(&completion)?;
                if let Some(controller) = payload.adapters.iter_mut().find(|adapter| adapter.id == "runtime.controller") {
                    controller.state = json!({"scope":scope,"lifecycle":lifecycle,"progress":progress,"evidence":evidence});
                }
                payload.progress.observation_sequence = progress.sequence;
                payload.progress.cycles_without_progress = progress.cycles_without_progress;
                payload.progress.replans = progress.replans;
                payload.progress.high_water_marks = serde_json::to_value(&progress)?;
                payload.evidence.records.push(TaskEvidenceRecord { run_id: key.run_id.clone(), sequence: scope.cycle,
                    stage: scope.stage.clone(), attempt: None, origin: "controller".into(),
                    data: json!({"scope":scope,"gate":gate,"completion":completion,"validations":evidence.validations.iter().filter(|result| result.identity.cycle == scope.cycle).collect::<Vec<_>>(),"progress":progress.last_delta,"decision":decision}) });
                Ok(())
            }).await?;
                // Only the acknowledged controller decision is exported; delivery never selects another intervention.
                super::events::emit(
                    execution,
                    super::events::TaskRunEventKind::ProgressObserved,
                    || json!({"observation_sequence":progress.sequence,"delta":progress.last_delta,"metrics":evidence.progress.as_ref().map(|p|super::events::safe_value(&p.value))}),
                )
                .await;
                super::events::emit(
                    execution,
                    super::events::TaskRunEventKind::CompletionGateChecked,
                    || json!({"outcome":gate,"cycle":scope.cycle,"details":completion.as_ref().map(|c|super::events::safe_value(&serde_json::to_value(c).unwrap_or(serde_json::Value::Null)))}),
                )
                .await;
                super::events::emit(execution, super::events::TaskRunEventKind::InterventionSelected, || json!({"decision":match &decision { CheckpointDecision::Terminal { .. } => "terminal", CheckpointDecision::Wait => "wait", CheckpointDecision::Replan => "replan", CheckpointDecision::AskUser => "ask_user", CheckpointDecision::Continue => "continue", CheckpointDecision::Limit => "limit" },"cycle":scope.cycle})).await;
                if lifecycle.current_stage > previous_stage {
                    super::events::emit(
                        execution,
                        super::events::TaskRunEventKind::StageCompleted,
                        || json!({"stage_index":previous_stage,"stage":profile.settings.lifecycle.as_ref().and_then(|stages|stages.get(previous_stage)).map(|stage|super::events::safe_text(&stage.id)),"cycle":scope.cycle}),
                    )
                    .await;
                }
                bound.extensions.ensure_targets_current(&scope)?;
                match decision {
                    CheckpointDecision::Terminal { status, reason } => {
                        if status == TaskRunStatus::Completed {
                            execution.set_final_evidence(Arc::new(
                                super::host_evidence::HostEvidenceAdmission {
                                    extensions: bound.extensions.clone(),
                                    scope: scope.clone(),
                                },
                            ));
                        }
                        return self.finish(execution, status, reason, last_response).await;
                    }
                    CheckpointDecision::Limit => {
                        let state = ControllerContinuation::decision(
                            &scope,
                            &lifecycle,
                            &progress,
                            &evidence,
                            last_response.as_ref(),
                            &instruction,
                            source,
                            forced_continuation,
                        );
                        let reason =
                            if payload.counters.turns >= u64::from(payload.limits.max_turns) {
                                "turn_capacity"
                            } else {
                                "invocation_capacity"
                            };
                        return Box::pin(self.park_limit(execution, todo, state, reason)).await;
                    }
                    CheckpointDecision::Replan => {
                        forced_continuation = false;
                        phase = ControllerPhase::Replan;
                        continuation = Some(Box::new(ControllerContinuation::decision(
                            &scope,
                            &lifecycle,
                            &progress,
                            &evidence,
                            last_response.as_ref(),
                            &instruction,
                            source,
                            false,
                        )));
                        continue;
                    }
                    CheckpointDecision::AskUser => {
                        let state = ControllerContinuation::decision(
                            &scope,
                            &lifecycle,
                            &progress,
                            &evidence,
                            last_response.as_ref(),
                            &instruction,
                            source,
                            false,
                        );
                        return Box::pin(self.park_question(execution, todo, state, "no_progress",
                            "Progress has stalled. Provide task guidance, or use an explicit host action to stop, cancel or revise the objective or limits.".into(), None)).await;
                    }
                    CheckpointDecision::Wait => {
                        return Err(TaskRunStorageError::InvalidCheckpoint.into());
                    }
                    CheckpointDecision::Continue => {
                        if source == AutonomyTurnSource::ValidationFix {
                            forced_continuation = false;
                            instruction = "Required validation failed. Repair the reopened stage without resetting completed effects or resource usage.".into();
                            continue;
                        }
                        if lifecycle.current_stage != previous_stage
                            && lifecycle.current_stage
                                < profile.settings.lifecycle.as_ref().map_or(0, Vec::len)
                        {
                            forced_continuation = false;
                            source = AutonomyTurnSource::StageInstruction;
                            instruction = String::new();
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
                                    progress.handled_sequence = Some(progress.sequence);
                                    let state = ControllerContinuation::decision(
                                        &scope,
                                        &lifecycle,
                                        &progress,
                                        &evidence,
                                        last_response.as_ref(),
                                        &instruction,
                                        source,
                                        false,
                                    );
                                    let blocked = evidence.tools.iter().any(|tool| {
                                        tool.value.policy.outcome
                                            == ai_agents_core::PermissionOutcome::Deny
                                    });
                                    return Box::pin(self.park_question(execution, todo, state,
                                        if blocked { "policy_blocked" } else { "premature_finish" },
                                        "Completion is not proven. Provide guidance for the unmet gates; an answer does not waive validation or authorize denied work.".into(), None)).await;
                                }
                                PrematureFinishAction::Continue => {}
                            }
                            if payload.counters.continuations
                                >= u64::from(
                                    policy
                                        .and_then(|p| p.max_continuations)
                                        .unwrap_or(payload.limits.max_turns),
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
        })
    }

    /// Binds one check to its frozen host target and the shared ledger without resetting a retained attempt or its deadline.
    /// Erasure keeps optional observation future construction out of nested runtime turn polling frames.
    #[allow(clippy::too_many_arguments)]
    fn validate<'a>(
        &'a self,
        check: &'a BoundValidationCheck,
        scope: &'a mut EvaluationScope,
        evidence: &'a mut EvaluationEvidence,
        execution: &'a Arc<RunExecution>,
        bound: &'a BoundAutonomyProfile,
        executor: &'a RuntimeObservationExecutor,
        journal: &'a TaskValidationJournal,
        resumed: Option<Box<ValidationDriverState>>,
    ) -> ControllerFuture<'a, Option<Box<ValidationDriverState>>> {
        Box::pin(async move {
            // A replan is bound to its selected intervention rather than a generic cycle/check attempt.
            let attempt = if check.check.adapter == "internal.replan" {
                check.check.id.clone()
            } else {
                format!("{}:{}", scope.cycle, check.check.id)
            };
            scope
                .validation_attempts
                .insert(check.check.id.clone(), attempt.clone());

            // A resumed attempt retains its original identity; capture_scope records current invocation/publication freshness separately.
            if resumed.is_none() {
                scope.mutation_generation = execution.mutation_generation();
                bound.capture_targets(scope)?;
            }
            let mut identity = Self::identity(
                scope,
                scope.cycle,
                Some(attempt),
                check.config_identity.clone(),
            );
            if let Some(target) = bound.extensions.validation_target(&check.check.id) {
                identity.target = Some(target.into());
                identity.target_revision = scope.target_revisions.get(target).cloned();
            }
            let resuming_validation = resumed.is_some();
            let mut driver = Box::new(if let Some(driver) = resumed {
                ValidationDriverState::restore_retained(
                    serde_json::to_value(&driver)?,
                    &driver,
                    check,
                    scope,
                    &bound.extensions,
                )?
            } else {
                ValidationDriverState::new(check, identity)?
            });
            if !resuming_validation {
                super::events::emit(execution, super::events::TaskRunEventKind::ValidationStarted, || json!({"check":super::events::safe_text(&check.check.id),"attempt":super::events::safe_text(driver.identity.attempt.as_deref().unwrap_or(""))})).await;
            }
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
                    bound.extensions.ensure_targets_current(scope)?;
                    scope.mutation_generation = execution.mutation_generation();
                    evidence.collect_validation(&driver)?;
                    evidence.collect_artifacts(&driver, check)?;
                    // Normalization may retire older artifacts, so select the acknowledged attempt rather than slicing by the old vector length.
                    for artifact in evidence.artifacts.iter().filter(|artifact| {
                        artifact.identity.attempt == driver.identity.attempt
                            && artifact.identity.cycle == driver.identity.cycle
                            && artifact.identity.config_identity == driver.identity.config_identity
                            && artifact
                                .identity
                                .validation_dependencies
                                .contains_key(&check.check.id)
                    }) {
                        super::events::emit(execution, super::events::TaskRunEventKind::ArtifactObserved, || json!({"path":super::events::safe_text(&artifact.value.path),"exists":artifact.value.exists,"check":super::events::safe_text(&check.check.id)})).await;
                    }

                    Ok(None)
                }
                ValidationDriverOutcome::AwaitExternal(_) | ValidationDriverOutcome::Suspended => {
                    Ok(Some(driver))
                }
            }
        })
    }

    /// Replaces duplicate driver captures with acknowledged shared-executor records, retaining late effects as ineligible audit evidence.
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
                    complete: capture.record.metadata.get("_task_result_eligible")
                        != Some(&serde_json::Value::Bool(false)),
                    value: capture.record,
                })
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Capacity exhaustion and unanswered or exhausted clarification remain distinct from execution or ownership failure.
    fn stopped_status(execution: &RunExecution) -> TaskRunStatus {
        match execution.stop_reason().as_deref() {
            Some(
                "llm_capacity"
                | "tool_capacity"
                | "priced_capacity"
                | "write_capacity"
                | "time_violation"
                | "maintenance_capacity",
            ) => {
                if execution
                    .lifecycle_profile()
                    .on_limit
                    .as_ref()
                    .and_then(|policy| policy.action)
                    == Some(LimitAction::Fail)
                {
                    TaskRunStatus::Failed
                } else {
                    TaskRunStatus::LimitReached
                }
            }
            Some("cancel_requested") => TaskRunStatus::Cancelled,
            Some(
                "question_timeout"
                | "supervision_rejected"
                | "response_timeout"
                | "pipeline_timeout"
                | "composition_timeout"
                | "scope_ended"
                | "interaction_unavailable"
                | "interaction_rejected"
                | "disambiguation_exhausted",
            ) => TaskRunStatus::Incomplete,
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
            validation_dependencies: BTreeMap::new(),
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
