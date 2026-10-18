//! Retained host decisions preserve live custody while changing only the selected controller boundary.

use super::super::controller::{ControllerProposal, ControllerReview};
use super::*;
use ai_agents_hitl::ApprovalResult;

impl AutonomyRunner {
    /// Keeps abandonment armed across claim acknowledgement, except when the store certifies that no claim was written.
    pub(super) async fn claim_retained(
        &self,
        snapshot: &TaskRunSnapshot,
        cleanup: &mut OwnedTurnCleanup,
    ) -> Result<TaskRunSnapshot> {
        let token = uuid::Uuid::new_v4().to_string();
        let claimed = self
            .store
            .claim_paused(&snapshot.key.run_id, snapshot.revision, token.clone())
            .await?;
        let Some(claimed) = claimed else {
            cleanup.finish();
            return Err(TaskRunStorageError::Conflict.into());
        };
        if claimed.key != snapshot.key
            || claimed.actor_id != snapshot.actor_id
            || claimed.schema_version != snapshot.schema_version
            || claimed.created_at != snapshot.created_at
            || claimed.updated_at < snapshot.updated_at
            || claimed.payload != snapshot.payload
            || claimed.revision != snapshot.revision + 1
            || claimed.status != TaskRunStatus::Running
            || claimed.owner_token.as_deref() != Some(token.as_str())
            || claimed.cancel_requested != snapshot.cancel_requested
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        Ok(claimed)
    }

    /// Final proposal admission preserves its original clocks and control binding, including an explicit bound timeout policy.
    pub(super) fn controller_review_current(
        &self,
        execution: &RunExecution,
        review: &ControllerReview,
    ) -> Result<bool> {
        if review.state_generation
            != self
                .agent
                .state_machine()
                .map(|machine| machine.generation())
            || (!matches!(review.proposal, ControllerProposal::Continue)
                && review.control_binding != self.agent.task_review_binding())
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let expired = review
            .expires_at
            .is_some_and(|expiry| chrono::Utc::now() >= expiry)
            || execution
                .approval_deadlines
                .lock()
                .get(&review.id)
                .is_some_and(|expiry| std::time::Instant::now() >= *expiry);
        Ok(!expired || super::super::suspension::timeout_approved(&review.id))
    }

    /// Freezes one actual stage or plan proposal before application; approval is not a tool grant.
    pub(super) async fn park_review(
        &self,
        execution: &Arc<RunExecution>,
        todo: Option<&RunTodoAdapter>,
        mut state: ControllerContinuation,
        proposal: ControllerProposal,
    ) -> Result<TaskRunResult> {
        let timeout = self.agent.task_review_timeout();
        let review = ControllerReview {
            id: uuid::Uuid::new_v4().to_string(),
            proposal,
            control_binding: self.agent.task_review_binding(),
            expires_at: timeout
                .map(|duration| {
                    chrono::Utc::now()
                        .checked_add_signed(
                            chrono::Duration::from_std(duration)
                                .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?,
                        )
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)
                })
                .transpose()?,
            state_generation: self
                .agent
                .state_machine()
                .map(|machine| machine.generation()),
        };
        if let Some(duration) = timeout {
            execution.approval_deadlines.lock().insert(
                review.id.clone(),
                std::time::Instant::now()
                    .checked_add(duration)
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?,
            );
        }
        state.review = Some(review.clone());
        execution
            .update(|payload| {
                payload.pending = Some(TaskPendingRequest {
                    id: review.id.clone(),
                    issued_revision: execution
                        .revision
                        .load(std::sync::atomic::Ordering::Acquire)
                        + 1,
                    kind: TaskPendingKind::Approval,
                    reviewed_action: serde_json::to_value(&review)?,
                });
                Ok(())
            })
            .await?;
        *execution.controller.lock() = Some(Box::new(state));
        Box::pin(self.pause_controller(execution, todo)).await
    }

    /// Retires known-safe old-objective work without executing it, then returns a new revision awaiting explicit Continue.
    /// Once cleanup or a conditional write may have happened, any failure retains recovery protection instead of reviving the old cursor.
    pub async fn modify_objective_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
        objective: &str,
        reason: &str,
    ) -> Result<TaskRunResult> {
        let _resume = self.resume_gate.lock().await;
        if !self.store.supports_objective_revision() {
            return Err(AgentError::Config(
                "retained objective revision is unsupported by this task store".into(),
            ));
        }
        if objective.trim().is_empty() || reason.trim().is_empty() || reason.chars().count() > 512 {
            return Err(AgentError::Config(
                "objective revision requires a nonempty objective and bounded audit reason".into(),
            ));
        }
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;
        if snapshot.revision != expected_revision
            || snapshot.status != TaskRunStatus::Paused
            || snapshot.cancel_requested
            || payload
                .clocks
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
            || payload.clocks.active_interval_started_at.is_some()
            || payload.reservations.iter().any(|reservation| {
                matches!(
                    reservation.state,
                    TaskEffectState::Dispatched | TaskEffectState::Uncertain
                )
            })
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let controller = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller")
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let mut lifecycle: LifecycleState =
            serde_json::from_value(controller.state["lifecycle"].clone())?;
        lifecycle.validate(&payload.settings)?;
        if lifecycle.strategy == ExecutionStrategy::Skill && !lifecycle.skill_segment_completed {
            return Err(AgentError::Config("objective revision cannot restart or falsely complete a required partial skill segment".into()));
        }
        let mut scope: EvaluationScope = serde_json::from_value(controller.state["scope"].clone())?;
        let mut progress: ProgressState =
            serde_json::from_value(controller.state["progress"].clone())?;
        let (owner, participants, targets, batch, group, retained_controller, whole_expiry) = {
            let retained = self.paused.lock();
            let retained = retained
                .get(run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            retained.owner.check()?;
            if retained.participants.unsettled()
                || retained
                    .whole_expiry
                    .is_some_and(|expiry| std::time::Instant::now() >= expiry)
                || serde_json::to_value(
                    retained
                        .todo
                        .as_ref()
                        .map(RunTodoAdapter::checkpoint)
                        .transpose()?,
                )? != serde_json::to_value(&payload.todos)?
            {
                return Err(TaskRunStorageError::NotResumable.into());
            }
            (
                retained.owner.clone(),
                retained.participants.clone(),
                retained.targets.clone(),
                retained.batch.clone(),
                retained.group.clone(),
                retained.controller.clone(),
                retained.whole_expiry,
            )
        };
        if serde_json::to_value(controller_checkpoint(&payload)?)?
            != serde_json::to_value(&retained_controller)?
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        if let Some(state) = &retained_controller {
            lifecycle = state.lifecycle.clone();
            scope = state.scope.clone();
            progress = state.progress.clone();
        }
        if let Some(batch) = &batch {
            batch.validate_checkpoint(&payload)?;
        }
        if let Some(group) = &group {
            group.validate_checkpoint(&payload)?;
        }
        // Retired executable data remains audit-only even after a later attempt replaces the live adapter checkpoint.
        let retired_validation: Vec<_> = payload
            .adapters
            .iter()
            .filter_map(|adapter| {
                adapter
                    .state
                    .get("_task_validation_driver")
                    .map(|driver| json!({"id":adapter.id,"driver":driver}))
            })
            .collect();
        let retired_execution = json!({"objective_revision":payload.objective_revision,"controller":retained_controller,"batch":batch,"group":group,"validation":retired_validation});
        // This preflight rejects oversized audit growth before claiming or changing the live pause.
        let mut audit_probe = payload.clone();
        audit_probe.evidence.records.push(TaskEvidenceRecord {
            run_id: run_id.into(),
            sequence: audit_probe.evidence.records.len() as u64,
            stage: payload.stage.clone(),
            attempt: None,
            origin: "host.continuation.supersede".into(),
            data: retired_execution.clone(),
        });
        audit_probe.evidence.records.push(TaskEvidenceRecord {
            run_id: run_id.into(), sequence: 0, stage: None, attempt: None, origin: "host.control.preflight".into(),
            data: json!({"objective":objective,"reason":reason,"retired_todos":payload.todos,"retired_controller":controller.state}),
        });
        audit_probe.bind(snapshot.clone())?;
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        let claimed = self.claim_retained(&snapshot, &mut cleanup).await?;
        let mut execution = RunExecution::new(
            self.store.clone(),
            &claimed,
            lifecycle.clone(),
            owner.clone(),
            std::time::Instant::now(),
        )?;
        let live = Arc::get_mut(&mut execution).ok_or(TaskRunStorageError::Conflict)?;
        live.participants = participants.clone();
        live.targets = targets.clone();
        live.expiry_projection = whole_expiry;
        *self.active.lock() = Some(execution.clone());
        if let Some(todos) = &payload.todos {
            execution.install_todos(self.agent.todo_store(), todos.binding.clone());
        }

        if let Some(state) = &retained_controller {
            if let Some(parked) = state
                .driver
                .as_ref()
                .and_then(|driver| driver.parked.as_ref())
            {
                let driver = state
                    .driver
                    .as_ref()
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                let id = format!(
                    "validation:{}",
                    canonical_identity(
                        &json!({"check":driver.check_id,"attempt":driver.identity.attempt,"observation":parked.observation})
                    )?
                );
                let reservation = payload
                    .reservations
                    .iter_mut()
                    .find(|reservation| reservation.id == id)
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                if reservation.state != TaskEffectState::Suspended {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
                reservation.state = TaskEffectState::Completed;
                reservation.result = Some(json!({"superseded":true,"origin":"validation"}));
                // Settle the logical observation before generic child-message cleanup settles suspended invocation reservations.
                if let Some(group) = &parked.group {
                    let group: TaskGroupState = serde_json::from_value(group.clone())?;
                    self.cleanup_observation_group(
                        &owner,
                        &group,
                        &participants,
                        &targets,
                        &mut payload,
                    )
                    .await?;
                }
            }
        } else if let Some(group) = &group {
            let retired = self
                .agent
                .cancel_task_group(&owner, group, &participants, &targets)
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
                child.result = Some(json!({"superseded":true}));
            }
            super::super::message::settle_cancelled_messages(&mut payload)?;
        } else if let Some(batch) = &batch {
            payload.runtime = self.agent.cancel_task_batch(&owner, batch.clone()).await?;
        } else {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        for pending in
            batch
                .iter()
                .flat_map(|batch| batch.approvals.iter())
                .chain(group.iter().flat_map(|group| {
                    group.batch.approvals.iter().chain(
                        group
                            .parked
                            .iter()
                            .flat_map(|leaf| leaf.batch.approvals.iter()),
                    )
                }))
        {
            if !payload.consumed_request_ids.contains(&pending.request_id) {
                payload
                    .consumed_request_ids
                    .push(pending.request_id.clone());
            }
        }
        if let Some(pending) = payload.pending.take()
            && !payload.consumed_request_ids.contains(&pending.id)
        {
            payload.consumed_request_ids.push(pending.id);
        }
        retire_controller(&mut payload)?;
        // Archived validation attempts are not successful results and cannot be resumed under another objective.
        for adapter in &mut payload.adapters {
            if let Some(driver) = adapter.state.get("_task_validation_driver").cloned() {
                adapter.state = json!({"_retired_validation_attempt": driver["identity"]["attempt"], "superseded_objective_revision": payload.objective_revision});
            }
        }
        payload.runtime =
            TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        payload.pause_reason = None;
        payload.stop_reason = None;
        payload.evidence.records.push(TaskEvidenceRecord {
            run_id: run_id.into(),
            sequence: payload.evidence.records.len() as u64,
            stage: payload.stage.clone(),
            attempt: None,
            origin: "host.continuation.supersede".into(),
            data: retired_execution,
        });
        execution
            .update(|current| {
                *current = payload;
                Ok(())
            })
            .await?;
        let safe = execution.load_owned().await?;
        let (mut exact, _) = super::super::control::prepare_host_control(
            &safe,
            HostControlAction::ModifyObjective {
                objective: objective.into(),
                reason: reason.into(),
            },
            &self.host,
        )?;
        let revised = self
            .store
            .apply_host_control(
                run_id,
                safe.revision,
                HostControlAction::ModifyObjective {
                    objective: objective.into(),
                    reason: reason.into(),
                },
                &self.host,
                whole_expiry,
            )
            .await?
            .ok_or(TaskRunStorageError::Conflict)?;
        let revised_payload: TaskCheckpointPayload =
            serde_json::from_value(revised.payload.clone())?;
        if revised.key != safe.key
            || revised.owner_token != safe.owner_token
            || revised.status != TaskRunStatus::Running
            || revised.revision != safe.revision + 1
            || revised.cancel_requested
            || revised_payload.objective != objective
            || revised_payload.objective_revision != scope.objective_revision + 1
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        // The scoped store mints a fresh todo token; every other payload field must match the exact control proposal.
        if let (Some(expected), Some(actual)) = (&mut exact.todos, &revised_payload.todos) {
            if actual.binding.token.is_empty() {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            expected.binding.token = actual.binding.token.clone();
        }
        if revised_payload.config_identity != self.config_identity
            || revised.actor_id != safe.actor_id
            || revised.schema_version != safe.schema_version
            || revised.created_at != safe.created_at
            || revised.updated_at < safe.updated_at
            || serde_json::to_value(&revised_payload)? != serde_json::to_value(&exact)?
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        revised_payload.validate(&revised, &self.config_identity)?;
        revised_payload.validate_successor_with_control(
            &serde_json::from_value(safe.payload)?,
            revised.revision,
            false,
            true,
        )?;
        execution.acknowledge_snapshot(&revised);
        execution
            .revision
            .store(revised.revision, std::sync::atomic::Ordering::Release);
        if let Some(todos) = &revised_payload.todos {
            let mut retained = self.paused.lock();
            let retained = retained
                .get_mut(run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            let adapter = retained
                .todo
                .as_ref()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                .rebind(todos.binding.clone())?;
            execution.install_todos(self.agent.todo_store(), todos.binding.clone());
            retained.todo = Some(adapter);
        }
        scope.objective_revision = revised_payload.objective_revision;
        scope.stage = None;
        scope.validation_attempts.clear();
        scope.validation_bindings.clear();
        scope.mutation_generation = execution.mutation_generation();
        progress.revise_objective(scope.clone());
        lifecycle.current_stage = 0;
        lifecycle.completed_stages.clear();
        let profile = resolve_profile(
            &self.config,
            AutonomyScope::Task,
            revised_payload.profile.as_deref(),
            None,
            None,
            Some(objective),
            &self.host,
        )?;
        let bound = self
            .agent
            .autonomy_extensions()
            .bind_for_agent(&self.agent, &profile)?;
        bound.prepare_scope(&mut scope);
        let review = ControllerReview {
            id: uuid::Uuid::new_v4().to_string(),
            proposal: ControllerProposal::Continue,
            control_binding: self.agent.task_review_binding(),
            expires_at: None,
            state_generation: self
                .agent
                .state_machine()
                .map(|machine| machine.generation()),
        };
        let state = Box::new(ControllerContinuation {
            scope: scope.clone(),
            lifecycle: lifecycle.clone(),
            progress: progress.clone(),
            evidence: EvaluationEvidence {
                required_checks: bound
                    .checks
                    .iter()
                    .filter(|check| check.check.required.unwrap_or(true))
                    .map(|check| check.check.id.clone())
                    .collect(),
                ..Default::default()
            },
            response: AgentResponse::new(""),
            instruction: String::new(),
            source: AutonomyTurnSource::Continuation,
            forced_continuation: false,
            checks: Vec::new(),
            next_check: 0,
            driver: None,
            review: Some(review.clone()),
        });
        execution.update(|payload| {
            payload.last_response = None;
            payload.stage = None;
            payload.stage_attempt = None;
            payload.controller_state = serde_json::Value::Null;
            payload.pending = Some(TaskPendingRequest {
                id: review.id.clone(), issued_revision: execution.revision.load(std::sync::atomic::Ordering::Acquire) + 1,
                kind: TaskPendingKind::HostContinue, reviewed_action: serde_json::to_value(&review)?,
            });
            payload.progress.high_water_marks = serde_json::to_value(&progress)?;
            let adapter = payload.adapters.iter_mut().find(|adapter| adapter.id == "runtime.controller").ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            adapter.state = json!({"scope":scope,"lifecycle":lifecycle,"progress":progress,"evidence":EvaluationEvidence::default()});
            Ok(())
        }).await?;
        *execution.controller.lock() = Some(state);
        let todo_checkpoint = self
            .paused
            .lock()
            .get(run_id)
            .and_then(|paused| paused.todo.as_ref())
            .map(RunTodoAdapter::checkpoint)
            .transpose()?;
        // The adapter stays in the map across awaits so interrupted handoff never silently drops its binding.
        let result = self.pause_controller(&execution, None).await?;
        if let Some(todos) = todo_checkpoint
            && execution.acknowledged_snapshot().payload["todos"]["binding"]
                != serde_json::to_value(todos.binding)?
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let mut retained = self.paused.lock();
        let retained = retained
            .get_mut(run_id)
            .ok_or(TaskRunStorageError::NotResumable)?;
        retained.batch = None;
        retained.group = None;
        retained.controller = execution.controller.lock().clone();
        retained.approval_deadlines.clear();
        cleanup.finish();
        Ok(result)
    }

    /// Consumes only the frozen host decision and then reenters its controller, preserving original clocks and counted work.
    pub(super) async fn resume_host_boundary(
        &self,
        snapshot: TaskRunSnapshot,
        payload: TaskCheckpointPayload,
        state: Box<ControllerContinuation>,
        input: TaskResumeInput,
    ) -> Result<TaskRunResult> {
        let review = state
            .review
            .as_ref()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let pending = payload
            .pending
            .as_ref()
            .ok_or(TaskRunStorageError::NotResumable)?;
        if review.state_generation
            != self
                .agent
                .state_machine()
                .map(|machine| machine.generation())
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        if !matches!(review.proposal, ControllerProposal::Continue)
            && review.control_binding != self.agent.task_review_binding()
        {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let (owner, expired) = {
            let retained = self.paused.lock();
            let paused = retained
                .get(&snapshot.key.run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            paused.owner.check()?;
            if paused
                .whole_expiry
                .is_some_and(|expiry| std::time::Instant::now() >= expiry)
            {
                return Err(TaskRunStorageError::AdmissionExpired.into());
            }
            (
                paused.owner.clone(),
                review
                    .expires_at
                    .is_some_and(|expiry| chrono::Utc::now() >= expiry)
                    || paused
                        .approval_deadlines
                        .get(&review.id)
                        .is_some_and(|expiry| std::time::Instant::now() >= *expiry),
            )
        };
        let timeout_id = (expired
            && matches!(
                &input,
                TaskResumeInput::Approval {
                    result: ApprovalResult::Timeout,
                    ..
                }
            ))
        .then(|| review.id.clone());
        let accepted = match (&review.proposal, input) {
            (ControllerProposal::Continue, TaskResumeInput::Continue) => true,
            (
                ControllerProposal::Stage { .. } | ControllerProposal::Plan { .. },
                TaskResumeInput::Approval {
                    request_id,
                    mut result,
                },
            ) if request_id == pending.id => {
                if matches!(result, ApprovalResult::Timeout) {
                    if !expired {
                        return Err(AgentError::HITLTimeout);
                    }
                    result = self.agent.task_approval_timeout_result()?;
                } else if expired {
                    return Err(AgentError::HITLTimeout);
                }
                match result {
                    ApprovalResult::Approved => true,
                    ApprovalResult::Modified { changes } if changes.is_empty() => true,
                    ApprovalResult::Rejected { .. } | ApprovalResult::Timeout => false,
                    _ => {
                        return Err(AgentError::Config(
                            "supervision approval cannot replace the frozen proposal".into(),
                        ));
                    }
                }
            }
            _ => return Err(TaskRunStorageError::NotResumable.into()),
        };
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
        if accepted && !matches!(review.proposal, ControllerProposal::Continue) {
            bound.extensions.ensure_targets_current(&state.scope)?;
        }
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
        execution.set_scope(&state.scope);
        if let Some(todo) = &paused.todo {
            execution.install_todos(self.agent.todo_store(), todo.checkpoint()?.binding);
        }
        *self.active.lock() = Some(execution.clone());
        let outcome = scope_execution(
            execution.clone(),
            Box::pin(async {
                if review.state_generation
                    != self
                        .agent
                        .state_machine()
                        .map(|machine| machine.generation())
                {
                    return Err(TaskRunStorageError::NotResumable.into());
                }
                execution
                    .update(|payload| {
                        payload.pending = None;
                        payload.consumed_request_ids.push(pending.id.clone());
                        retire_controller(payload)?;
                        payload.pause_reason = None;
                        payload.clocks.active_interval_started_at = Some(chrono::Utc::now());
                        Ok(())
                    })
                    .await?;
                if !matches!(review.proposal, ControllerProposal::Continue)
                    && review.control_binding != self.agent.task_review_binding()
                {
                    return Err(TaskRunStorageError::NotResumable.into());
                }
                // Storage waits do not extend approval deadlines or permit work after a competing cancellation.
                let latest = execution.load_owned().await?;
                let current: TaskCheckpointPayload = serde_json::from_value(latest.payload)?;
                let expired_now = review
                    .expires_at
                    .is_some_and(|expiry| chrono::Utc::now() >= expiry)
                    || execution
                        .approval_deadlines
                        .lock()
                        .get(&review.id)
                        .is_some_and(|expiry| std::time::Instant::now() >= *expiry);
                let interruption = if latest.cancel_requested {
                    Some((TaskRunStatus::Cancelled, "cancel_requested"))
                } else if execution.check(&current).is_err() {
                    Some((Self::stopped_status(&execution), "time_violation"))
                } else if accepted
                    && !matches!(review.proposal, ControllerProposal::Continue)
                    && bound
                        .extensions
                        .ensure_targets_current(&state.scope)
                        .is_err()
                {
                    Some((TaskRunStatus::Incomplete, "validation_target_changed"))
                } else if expired_now && !expired {
                    Some((TaskRunStatus::Incomplete, "supervision_timeout"))
                } else {
                    None
                };
                if let Some((status, reason)) = interruption {
                    return self.finish(&execution, status, reason.into(), None).await;
                }
                if !accepted {
                    return self
                        .finish(
                            &execution,
                            TaskRunStatus::Incomplete,
                            "supervision_rejected".into(),
                            None,
                        )
                        .await;
                }
                *execution.controller.lock() = Some(state.clone());
                super::super::suspension::scope_timeout_approval(
                    timeout_id,
                    self.execute(
                        &owner,
                        &execution,
                        &bound,
                        &profile,
                        &snapshot.key,
                        &payload.objective,
                        state.lifecycle.clone(),
                        paused.todo.as_ref(),
                        None,
                    ),
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
}
