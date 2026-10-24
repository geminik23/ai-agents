//! Same-process validation resumes retain the controller rather than synthesize another model turn.

use super::super::suspension::{
    BatchApproval, BatchAuthorization, BatchResponse, TaskExecutorCursor,
};
use super::*;
use ai_agents_core::{ToolCall, ToolExecutionRequest};
use ai_agents_hitl::ApprovalResult;

impl AutonomyRunner {
    /// Anchors exact controller progress only after run-owned maintenance has settled, before releasing the claim and active clock.
    pub(super) async fn pause_controller(
        &self,
        execution: &Arc<RunExecution>,
        todo: Option<&RunTodoAdapter>,
    ) -> Result<TaskRunResult> {
        execution.await_owned_work().await?;
        let _work_barrier = execution.owner.work.seal()?;
        let state = execution
            .controller
            .lock()
            .clone()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let runtime = TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        execution
            .update(|payload| {
                payload.runtime = runtime;
                // An objective handoff has already installed its canonical checkpoint before this pause.
                if let Some(todo) = todo {
                    payload.todos = Some(todo.checkpoint()?);
                }
                payload.pause_reason = Some("validation".into());
                let adapter = TaskAdapterCheckpoint {
                    id: "runtime.validation".into(),
                    adapter: "runtime.validation".into(),
                    contract_version: 1,
                    config: json!({"runtime_id":self.agent.info().id}),
                    state: serde_json::to_value(&state)?,
                };
                payload.adapters.retain(|a| a.id != adapter.id);
                payload.adapters.push(adapter);
                state.validate(payload)
            })
            .await?;
        let _serial = execution.serial.lock().await;
        let anchored = execution.load_owned_locked(&_serial).await?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(anchored.payload.clone())?;
        // Pending publication waits remain active work; cancellation and real expiry cannot be exposed as a fresh executable pause.
        if anchored.cancel_requested || execution.check(&payload).is_err() {
            drop(_serial);
            if let Some(driver) = &state.driver {
                self.settle_controller_cancel(execution, &mut driver.as_ref().clone())
                    .await?;
            } else {
                execution
                    .update(|payload| {
                        if let Some(pending) = payload.pending.take() {
                            payload.consumed_request_ids.push(pending.id);
                        }
                        retire_controller(payload)?;
                        payload.pause_reason = None;
                        Ok(())
                    })
                    .await?;
            }
            return self
                .finish(
                    execution,
                    Self::stopped_status(execution),
                    execution
                        .stop_reason()
                        .unwrap_or_else(|| "cancel_requested".into()),
                    Some(state.response.clone()),
                )
                .await;
        }
        payload.clocks.active_millis = execution.active_millis()?.max(
            payload
                .clocks
                .active_millis
                .checked_add(1)
                .ok_or_else(|| AgentError::Config("pause clock overflow".into()))?,
        );
        payload.clocks.active_interval_started_at = None;
        let exact_payload = serde_json::to_value(&payload)?;
        let saved = execution
            .store
            .mutate(
                &execution.run_id,
                &TaskRunMutation::FinalCheckpoint {
                    expected_revision: anchored.revision,
                    owner_token: execution.owner_token.clone(),
                    status: TaskRunStatus::Paused,
                    payload: serde_json::to_value(payload)?,
                    release: true,
                    expires_at: execution.expiry(),
                    evidence: None,
                    deadline: std::time::Instant::now()
                        .checked_add(execution.remaining_duration())
                        .ok_or_else(|| AgentError::Config("pause deadline overflow".into()))?,
                },
            )
            .await
            .inspect_err(|_| execution.storage_failure())?;
        if saved.key != anchored.key
            || saved.actor_id != anchored.actor_id
            || saved.schema_version != anchored.schema_version
            || saved.created_at != anchored.created_at
            || saved.updated_at < anchored.updated_at
            || saved.revision != anchored.revision + 1
            || saved.status != TaskRunStatus::Paused
            || saved.owner_token.is_some()
            || saved.cancel_requested != anchored.cancel_requested
            || saved.payload != exact_payload
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        execution.acknowledge_snapshot(&saved);
        execution
            .revision
            .store(saved.revision, std::sync::atomic::Ordering::Release);
        Ok(TaskRunResult {
            run: TaskRun::from_checkpoint(&saved, &self.config_identity)?,
            final_response: None,
        })
    }

    /// Checks checkpoint equality and returns the actual retained cursor, including nonserialized monotonic deadlines.
    pub(super) fn retained_controller(
        &self,
        snapshot: &TaskRunSnapshot,
        payload: &TaskCheckpointPayload,
    ) -> Result<Box<ControllerContinuation>> {
        let state = controller_checkpoint(payload)?.ok_or(TaskRunStorageError::NotResumable)?;
        let retained = self.paused.lock();
        let retained = retained
            .get(&snapshot.key.run_id)
            .ok_or(TaskRunStorageError::NotResumable)?;
        retained.owner.check()?;
        if serde_json::to_value(&state)? != serde_json::to_value(&retained.controller)? {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        retained
            .controller
            .clone()
            .ok_or_else(|| TaskRunStorageError::NotResumable.into())
    }

    /// Signals remain data; target freshness is checked before claim and after acknowledgements without weakening cleanup authority.
    /// Erasure prevents the optional controller driver from enlarging unrelated nested model resume frames.
    pub(super) fn resume_controller(
        &self,
        snapshot: TaskRunSnapshot,
        payload: TaskCheckpointPayload,
        input: TaskResumeInput,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TaskRunResult>> + Send + '_>>
    {
        Box::pin(async move {
            let mut state = self.retained_controller(&snapshot, &payload)?;
            if state.review.is_some() {
                return Box::pin(self.resume_host_boundary(snapshot, payload, state, input)).await;
            }
            let profile = self.retained_profile(&payload)?;
            if profile.settings != payload.settings {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            let bound = self
                .agent
                .autonomy_extensions()
                .bind_for_agent(&self.agent, &profile)?;
            if self
                .paused
                .lock()
                .get(&snapshot.key.run_id)
                .and_then(|paused| paused.whole_expiry)
                .is_some_and(|expiry| std::time::Instant::now() >= expiry)
            {
                return Err(TaskRunStorageError::AdmissionExpired.into());
            }
            let driver = state
                .driver
                .as_ref()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if driver.expired() {
                return Err(TaskRunStorageError::AdmissionExpired.into());
            }
            let check = state.driver_check(&bound, &payload.objective)?;
            let mut restored = ValidationDriverState::restore_retained(
                serde_json::to_value(driver)?,
                driver,
                &check,
                &state.scope,
                &bound.extensions,
            )?;
            bound.extensions.ensure_targets_current(&state.scope)?;
            let pending = payload
                .pending
                .as_ref()
                .ok_or(TaskRunStorageError::NotResumable)?;
            let mut receipts = Vec::new();
            let mut cursor = Vec::new();
            let mut question = None;
            let mut external = None;
            let mut observation_group = None;
            let mut timed_out = false;
            let mut timeout_id = None;
            let capacity_response = self.validate_capacity_response(&payload, &input)?;
            match input {
                TaskResumeInput::ExternalSignal(signal) => {
                    if !matches!(pending.kind, TaskPendingKind::ExternalCondition) {
                        return Err(TaskRunStorageError::NotResumable.into());
                    }
                    restored.validate_external(&signal, &bound.extensions)?;
                    external = Some(signal);
                }
                response => {
                    let parked = restored
                        .parked
                        .as_ref()
                        .ok_or(TaskRunStorageError::NotResumable)?;
                    receipts = parked
                        .authorized
                        .iter()
                        .map(|value| serde_json::from_value::<BatchAuthorization>(value.clone()))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let group = parked
                        .group
                        .as_ref()
                        .map(|value| serde_json::from_value::<TaskGroupState>(value.clone()))
                        .transpose()?;
                    if let Some(group) = &group {
                        super::super::authority::validate_observation_group(
                            group, &payload, pending,
                        )?;
                        let retained = self.paused.lock();
                        let retained = retained
                            .get(&snapshot.key.run_id)
                            .ok_or(TaskRunStorageError::NotResumable)?;
                        if group
                            .frames
                            .values()
                            .filter(|frame| {
                                frame.id == group.frame.id
                                    || frame.contains_operation(
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
                            })
                        {
                            return Err(TaskRunStorageError::AdmissionExpired.into());
                        }
                    }
                    let approval: BatchApproval = if let Some(group) = &group {
                        group
                            .batch
                            .approvals
                            .first()
                            .cloned()
                            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                    } else {
                        serde_json::from_value(parked.pending.reviewed_action.clone())?
                    };
                    let (id, response) = match response {
                        TaskResumeInput::Continue if capacity_response => {
                            (pending.id.clone(), BatchResponse::Capacity)
                        }
                        TaskResumeInput::UserAnswer { request_id, .. } if capacity_response => {
                            (request_id, BatchResponse::Capacity)
                        }
                        TaskResumeInput::Approval { request_id, result } => {
                            (request_id, BatchResponse::Approval(result))
                        }
                        TaskResumeInput::UserAnswer { request_id, answer } => {
                            (request_id, BatchResponse::UserAnswer(answer))
                        }
                        TaskResumeInput::QuestionTimeout { request_id } => {
                            (request_id, BatchResponse::QuestionTimeout)
                        }
                        TaskResumeInput::Continue => {
                            return Err(TaskRunStorageError::NotResumable.into());
                        }
                        TaskResumeInput::ExternalSignal(_) => unreachable!(),
                    };
                    if id != pending.id || payload.consumed_request_ids.contains(&id) {
                        return Err(TaskRunStorageError::NotResumable.into());
                    }
                    let mut pending_child = None;
                    if let Some(group) = &group {
                        let targets = self
                            .paused
                            .lock()
                            .get(&snapshot.key.run_id)
                            .map(|paused| paused.targets.clone())
                            .ok_or(TaskRunStorageError::NotResumable)?;
                        let child = self.agent.task_group_child(group, &targets)?;
                        child.validate_task_batch_response(&group.batch, &response)?;
                        pending_child = Some(child);
                        observation_group = Some((
                            parked.request.call_id.clone(),
                            group.clone(),
                            response.clone(),
                        ));
                    }
                    let retained = self.paused.lock();
                    let deadline = retained
                        .get(&snapshot.key.run_id)
                        .and_then(|p| {
                            p.approval_deadlines.get(&id).or_else(|| {
                                group.as_ref().and_then(|group| {
                                    p.approval_deadlines.get(&group.child_request.id)
                                })
                            })
                        })
                        .copied();
                    drop(retained);
                    let expired = deadline
                        .is_some_and(|deadline| std::time::Instant::now() >= deadline)
                        || approval
                            .expires_at
                            .is_some_and(|expiry| chrono::Utc::now() >= expiry);
                    if let Some(saved) = &parked.executor {
                        cursor.push(serde_json::from_value::<TaskExecutorCursor>(saved.clone())?);
                    }
                    match response {
                        BatchResponse::Capacity
                            if matches!(pending.kind, TaskPendingKind::Limit) => {}
                        BatchResponse::Approval(mut result)
                            if matches!(pending.kind, TaskPendingKind::Approval) =>
                        {
                            if expired {
                                if !matches!(result, ApprovalResult::Timeout) {
                                    return Err(AgentError::HITLTimeout);
                                }
                                // Timeout authority belongs to this exact leaf, not the coordinating validator's profile.
                                result = pending_child
                                    .as_deref()
                                    .unwrap_or(&self.agent)
                                    .task_approval_timeout_result()?;
                                timeout_id = Some(approval.request_id.clone());
                            } else if matches!(result, ApprovalResult::Timeout) {
                                return Err(AgentError::HITLTimeout);
                            }
                            if let Some((_, _, response)) = &mut observation_group {
                                *response = BatchResponse::Approval(result.clone());
                            }
                            receipts.push(BatchAuthorization {
                                request_id: approval.request_id.clone(),
                                call_id: approval.call_id,
                                trigger: approval.trigger,
                                context: approval.context,
                                result,
                            });
                        }
                        BatchResponse::UserAnswer(answer)
                            if matches!(pending.kind, TaskPendingKind::UserQuestion) =>
                        {
                            if expired {
                                return Err(AgentError::HITLTimeout);
                            }
                            let request: ToolExecutionRequest = cursor
                                .first()
                                .map(|c| c.request.clone())
                                .unwrap_or_else(|| parked.request.clone());
                            if group.is_none() {
                                self.agent.validate_task_question_answer(
                                    &ToolCall {
                                        id: request.call_id.clone(),
                                        name: request.requested_name,
                                        arguments: approval.context.clone(),
                                    },
                                    &approval.question,
                                    &answer,
                                )?;
                            }
                            question = Some((
                                approval.request_id,
                                request.call_id,
                                approval.context,
                                answer,
                            ));
                        }
                        BatchResponse::QuestionTimeout
                            if matches!(pending.kind, TaskPendingKind::UserQuestion) && expired =>
                        {
                            timed_out = true
                        }
                        _ => return Err(TaskRunStorageError::NotResumable.into()),
                    }
                }
            }
            if let Some(parked) = &mut restored.parked {
                parked.authorized = receipts
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
            }
            bounded_value(&serde_json::to_value(&restored)?, 262_144)?;
            let owner = self
                .paused
                .lock()
                .get(&snapshot.key.run_id)
                .map(|paused| paused.owner.clone())
                .ok_or(TaskRunStorageError::NotResumable)?;
            let mut cleanup = OwnedTurnCleanup::new(owner.clone());
            let claimed = self.claim_retained(&snapshot, &mut cleanup).await?;
            let mut paused = self
                .paused
                .lock()
                .remove(&snapshot.key.run_id)
                .ok_or(TaskRunStorageError::Conflict)?;
            let owner = paused.owner.clone();
            let mut execution = RunExecution::new(
                self.store.clone(),
                &claimed,
                state.lifecycle.clone(),
                owner.clone(),
                std::time::Instant::now(),
            )?;
            let live = Arc::get_mut(&mut execution).ok_or(TaskRunStorageError::Conflict)?;
            live.participants = paused.participants.clone();
            live.targets = paused.targets.clone();
            live.expiry_projection = match (live.expiry_projection, paused.whole_expiry) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            *execution.approval_deadlines.lock() = paused.approval_deadlines.clone();
            if let Some((_, group, _)) = &observation_group {
                *execution.delegate_frames.lock() = group.frames.clone();
            }
            execution.set_scope(&state.scope);
            if let Some(todo) = &paused.todo {
                execution.install_todos(self.agent.todo_store(), todo.checkpoint()?.binding);
            }
            *self.active.lock() = Some(execution.clone());
            let outcome = scope_execution(
                execution.clone(),
                Box::pin(async {
                    if let Some(result) = self
                        .finish_controller_interruption(&execution, &restored, &state.response)
                        .await?
                    {
                        return Ok(result);
                    }
                    execution
                        .update(|payload| {
                            payload.pause_reason = None;
                            payload.clocks.active_interval_started_at = Some(chrono::Utc::now());
                            Ok(())
                        })
                        .await?;
                    if let Some(result) = self
                        .finish_controller_interruption(&execution, &restored, &state.response)
                        .await?
                    {
                        return Ok(result);
                    }
                    if let Some(signal) = external {
                        let journal = TaskValidationJournal::for_execution(&execution);
                        restored.acknowledge(&journal).await?;
                        if let Some(result) = self
                            .finish_controller_interruption(&execution, &restored, &state.response)
                            .await?
                        {
                            return Ok(result);
                        }
                        restored.accept_external(signal, &bound.extensions)?;
                        // Persist acceptance and event deduplication before any callback observes the signal.
                        restored.acknowledge(&journal).await?;
                    } else {
                        execution
                            .update(|payload| {
                                payload.pending = None;
                                payload.consumed_request_ids.push(pending.id.clone());
                                retire_controller(payload)?;
                                payload.pause_reason = None;
                                Ok(())
                            })
                            .await?;
                    }
                    if let Some(result) = self
                        .finish_controller_interruption(&execution, &restored, &state.response)
                        .await?
                    {
                        return Ok(result);
                    }
                    if timed_out {
                        self.settle_controller_cancel(&execution, &mut restored)
                            .await?;
                        return self
                            .finish(
                                &execution,
                                TaskRunStatus::Failed,
                                "question_timeout".into(),
                                Some(state.response.clone()),
                            )
                            .await;
                    }
                    state.driver = Some(Box::new(restored));
                    *execution.controller.lock() = Some(state.clone());
                    let future = super::super::suspension::scope_resumed_executor(
                        cursor,
                        super::super::suspension::scope_resumed_approval(
                            receipts,
                            Box::pin(self.execute(
                                &owner,
                                &execution,
                                &bound,
                                &profile,
                                &snapshot.key,
                                &payload.objective,
                                state.lifecycle.clone(),
                                paused.todo.as_ref(),
                                None,
                            )),
                        ),
                    );
                    let future =
                        super::super::authority::scope_observation_group(observation_group, future);
                    let result =
                        super::super::suspension::scope_timeout_approval(timeout_id, async {
                            if let Some((request_id, call, args, answer)) = question {
                                super::super::suspension::scope_resumed_question(
                                    request_id, call, args, answer, future,
                                )
                                .await
                            } else {
                                future.await
                            }
                        })
                        .await;
                    // An ambiguous storage failure is not a clean clock interruption and cannot release retained custody.
                    if !execution.has_storage_failure()
                        && result.as_ref().is_err_and(|error| {
                            !matches!(
                                error,
                                AgentError::Persistence(_) | AgentError::TaskRunStorage(_)
                            )
                        })
                    {
                        let driver = state
                            .driver
                            .as_ref()
                            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                        if let Some(stopped) = self
                            .finish_controller_interruption(&execution, driver, &state.response)
                            .await?
                        {
                            return Ok(stopped);
                        }
                    }
                    result
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
        })
    }

    /// Cancellation of an acknowledged controller wait makes no tool, model or validation callback and releases owners only after acknowledgement.
    pub(super) async fn cancel_controller(
        &self,
        snapshot: TaskRunSnapshot,
        mut payload: TaskCheckpointPayload,
    ) -> Result<TaskRunResult> {
        let state = self.retained_controller(&snapshot, &payload)?;
        if let Some(driver) = &state.driver {
            driver.retire_wait_reservation(&mut payload)?;
        }
        if let Some(pending) = payload.pending.take() {
            payload.consumed_request_ids.push(pending.id);
        }
        payload.pause_reason = None;
        payload.stop_reason = Some("cancel_requested".into());
        retire_controller(&mut payload)?;
        let owner = self
            .paused
            .lock()
            .get(&snapshot.key.run_id)
            .map(|paused| paused.owner.clone())
            .ok_or(TaskRunStorageError::NotResumable)?;
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let requested = self.admit_paused_cleanup(snapshot.clone()).await?;
        if let Some(group) = state
            .driver
            .as_ref()
            .and_then(|driver| driver.parked.as_ref())
            .and_then(|parked| parked.group.as_ref())
        {
            let group: TaskGroupState = serde_json::from_value(group.clone())?;
            let (participants, targets) = self
                .paused
                .lock()
                .get(&snapshot.key.run_id)
                .map(|paused| (paused.participants.clone(), paused.targets.clone()))
                .ok_or(TaskRunStorageError::NotResumable)?;
            self.cleanup_observation_group(&owner, &group, &participants, &targets, &mut payload)
                .await?;
        }
        let acknowledged = self.acknowledge_paused_cleanup(&requested, payload).await?;
        let paused = self
            .paused
            .lock()
            .remove(&snapshot.key.run_id)
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

    /// Known-safe observation groups close each exact child history and consumed parent attempt without invoking a pending implementation.
    pub(super) async fn cleanup_observation_group(
        &self,
        owner: &Arc<RunOwner>,
        group: &TaskGroupState,
        participants: &Arc<super::super::participants::Participants>,
        targets: &Arc<CompositionTargets>,
        payload: &mut TaskCheckpointPayload,
    ) -> Result<()> {
        if participants.unsettled() {
            return Err(AgentError::Config(
                "observation group cleanup requires actual quiescence".into(),
            ));
        }
        let retired = self
            .agent
            .cancel_task_group(owner, group, participants, targets)
            .await?;
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
            child.result = Some(json!({"error":"observation child stopped before continuation"}));
        }
        super::super::message::settle_cancelled_messages(payload)?;
        payload.runtime =
            TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        Ok(())
    }

    /// Successful acknowledgements can race expiry or host target changes; only proven-safe work can be retired without execution.
    async fn finish_controller_interruption(
        &self,
        execution: &Arc<RunExecution>,
        driver: &ValidationDriverState,
        response: &AgentResponse,
    ) -> Result<Option<TaskRunResult>> {
        if execution.has_storage_failure() {
            return Err(AgentError::Persistence(
                "controller storage acknowledgement requires recovery".into(),
            ));
        }
        let snapshot = execution.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        let reason = if snapshot.cancel_requested {
            Some("cancel_requested")
        } else if driver.expired() {
            Some("validation_timeout")
        } else if self
            .agent
            .autonomy_extensions()
            .ensure_driver_target(driver)
            .is_err()
        {
            Some("validation_target_changed")
        } else if let Err(error) = execution.check(&payload) {
            if matches!(
                execution.stop_reason().as_deref(),
                Some("time_violation" | "composition_timeout")
            ) {
                Some("time_violation")
            } else {
                return Err(error);
            }
        } else {
            None
        };
        let Some(reason) = reason else {
            return Ok(None);
        };
        let mut state = driver.clone();
        self.settle_controller_cancel(execution, &mut state).await?;
        Ok(Some(
            self.finish(
                execution,
                if reason == "cancel_requested" {
                    TaskRunStatus::Cancelled
                } else if reason == "validation_target_changed" {
                    TaskRunStatus::Incomplete
                } else {
                    TaskRunStatus::LimitReached
                },
                reason.into(),
                Some(response.clone()),
            )
            .await?,
        ))
    }

    /// Known-safe controller cancellation retires only its logical parked observation and preserves actual executor outcomes.
    async fn settle_controller_cancel(
        &self,
        execution: &Arc<RunExecution>,
        driver: &mut ValidationDriverState,
    ) -> Result<()> {
        if let Some(group) = driver
            .parked
            .as_ref()
            .and_then(|parked| parked.group.as_ref())
        {
            let group: TaskGroupState = serde_json::from_value(group.clone())?;
            let snapshot = execution.load_owned().await?;
            let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
            let logical = driver.parked.as_ref().map(|parked| canonical_identity(&json!({"check":driver.check_id,"attempt":driver.identity.attempt,"observation":parked.observation})).map(|id| format!("validation:{id}"))).transpose()?;
            if payload.reservations.iter().any(|reservation| {
                Some(&reservation.id) != logical.as_ref()
                    && matches!(
                        reservation.state,
                        TaskEffectState::Dispatched | TaskEffectState::Uncertain
                    )
            }) {
                return Err(AgentError::Config(
                    "observation group interruption has uncertain invocation work".into(),
                ));
            }
            self.cleanup_observation_group(
                &execution.owner,
                &group,
                &execution.participants,
                &execution.targets,
                &mut payload,
            )
            .await?;
            if let Some(logical) = &logical {
                let reservation = payload
                    .reservations
                    .iter_mut()
                    .find(|reservation| &reservation.id == logical)
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                reservation.state = TaskEffectState::Completed;
                reservation.result = Some(json!({"cancelled":true,"origin":"validation"}));
            }
            if let Some(pending) = payload.pending.take() {
                payload.consumed_request_ids.push(pending.id);
            }
            payload.pause_reason = None;
            retire_controller(&mut payload)?;
            execution
                .update(|current| {
                    *current = payload;
                    Ok(())
                })
                .await?;
            return Ok(());
        }
        execution
            .update(|payload| {
                let parked_id = driver.wait_reservation()?;
                if execution.participants.unsettled()
                    || payload.children.iter().any(|child| child.result.is_none())
                    || payload.reservations.iter().any(|reservation| {
                        Some(&reservation.id) != parked_id.as_ref()
                            && matches!(
                                reservation.state,
                                TaskEffectState::Dispatched
                                    | TaskEffectState::Suspended
                                    | TaskEffectState::Uncertain
                            )
                    })
                {
                    return Err(AgentError::Config(
                        "controller stop requires actual invocation quiescence".into(),
                    ));
                }
                driver.retire_wait_reservation(payload)?;
                if let Some(pending) = payload.pending.take() {
                    payload.consumed_request_ids.push(pending.id);
                }
                payload.pause_reason = None;
                retire_controller(payload)?;
                Ok(())
            })
            .await?;
        Ok(())
    }
}
