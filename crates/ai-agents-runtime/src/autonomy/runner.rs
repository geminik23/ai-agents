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
    active: parking_lot::Mutex<Option<std::sync::Weak<RunExecution>>>,
}

impl AutonomyRunner {
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
        })
    }

    /// Requests cancellation of this retained live owner; the receipt is not cleanup acknowledgement or rollback.
    pub async fn request_cancel(&self, run_id: &str) -> Result<TaskRunSummary> {
        let execution = self
            .active
            .lock()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .filter(|execution| execution.run_id == run_id)
            .ok_or(TaskRunStorageError::NotResumable)?;
        execution.request_cancel().await
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
        if bound.checks.iter().any(|check| {
            !check.adapter.descriptor().conditions.is_empty()
                || !check.adapter.descriptor().tools.is_empty()
        }) || profile.settings.hitl.is_some()
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
        *self.active.lock() = Some(Arc::downgrade(&execution));
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
    ) -> Result<TaskRunResult> {
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
        bound.prepare_scope(&mut scope);
        let mut progress = ProgressState::new(scope.clone());
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
        let mut last_response = None;
        let mut instruction = String::new();
        let mut source = AutonomyTurnSource::InitialObjective;
        let mut forced_continuation = false;
        loop {
            let snapshot = execution.load_owned().await?;
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
            if execution.check(&payload).is_err()
                || payload.counters.turns >= u64::from(profile.max_turns)
            {
                return self
                    .finish(
                        execution,
                        TaskRunStatus::LimitReached,
                        execution
                            .stop_reason()
                            .unwrap_or_else(|| "turn_capacity".into()),
                        last_response,
                    )
                    .await;
            }
            scope.cycle += 1;
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
                "remaining_active_seconds":profile.max_active_time_seconds.saturating_mul(1000).saturating_sub(execution.active_millis()?)/1000,
                "remaining_wall_seconds":payload.clocks.expires_at.map(|expiry|(expiry-chrono::Utc::now()).num_seconds().max(0)),
                "remaining_micro_usd":payload.limits.max_micro_usd.map(|limit| limit.saturating_sub(payload.counters.charged_micro_usd).saturating_sub(payload.reservations.iter().filter(|reservation| matches!(reservation.state, TaskEffectState::Reserved | TaskEffectState::Dispatched | TaskEffectState::Uncertain)).map(|reservation| reservation.reserved_micro_usd.saturating_sub(reservation.charged_micro_usd)).sum::<u64>())),
                "remaining_declared_write_paths":payload.limits.max_declared_write_paths.map(|limit| (limit as usize).saturating_sub(payload.declared_write_targets.len())),
                                "failed_gates":payload.evidence.records.last().and_then(|record| record.data.get("completion")).cloned(),
                "progress_instruction":profile.settings.progress.as_ref().and_then(|progress| progress.auto_create_todo_prompt.clone()).unwrap_or_else(|| "Before progress-gated work, maintain the run todo list and select exactly one active item when required.".into()),
                "stage_instruction":profile.settings.lifecycle.as_ref().and_then(|stages| stages.get(lifecycle.current_stage)).and_then(|stage| stage.instruction.clone())});
            execution
                .update(|payload| {
                    payload.counters.turns += 1;
                    if forced_continuation {
                        payload.counters.continuations += 1;
                    }
                    payload.stage = scope.stage.clone();
                    payload.controller_state = controller.clone();
                    Ok(())
                })
                .await?;
            let response = self
                .agent
                .run_autonomy_turn(AutonomyTurnInput {
                    owner: owner.clone(),
                    objective: objective.into(),
                    controller_message: controller.to_string(),
                    source,
                })
                .await;
            match response {
                Ok(response) => last_response = Some(response),
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
            evidence.progress = Some(observation.scoped(identity));
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
            .saturating_sub(execution.active_millis()?);
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
            ValidationDriverOutcome::Complete(_) => evidence.collect_validation(&driver),
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

    /// Invocation exhaustion differs from storage, evidence or ownership failure.
    fn stopped_status(execution: &RunExecution) -> TaskRunStatus {
        match execution.stop_reason().as_deref() {
            Some(
                "llm_capacity" | "tool_capacity" | "priced_capacity" | "write_capacity"
                | "time_violation",
            ) => TaskRunStatus::LimitReached,
            Some("cancel_requested") => TaskRunStatus::Cancelled,
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
