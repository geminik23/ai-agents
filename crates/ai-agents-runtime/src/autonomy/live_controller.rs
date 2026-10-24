//! Explicit skill admission and controller-origin decisions reuse the retained owner and conditional pause protocol.

use super::super::controller::{ControllerLimit, ControllerProposal};
use super::*;

impl AutonomyRunner {
    /// Runs one explicitly selected loaded skill without semantic rerouting or automatic YAML task startup.
    /// Nested skills inherit the existing execution through the ordinary runtime instead of recursively invoking this API.
    /// Explicit skill profiles use the same opted-in safe-pause persistence as model-led tasks.
    pub async fn run_skill(
        &self,
        objective: &str,
        skill_id: &str,
        selected: Option<&str>,
    ) -> Result<TaskRunResult> {
        let skill = self.agent.admit_autonomy_skill(skill_id)?;
        let result = Box::pin(self.run_entry(objective, selected, Some(skill))).await?;
        self.persist_returned_pause(result).await
    }

    /// Manual resume is a delivery endpoint by default; hosts can explicitly declare that no usable endpoint is exposed.
    pub fn with_interaction_delivery(mut self, available: bool) -> Self {
        self.interaction_delivery = available;
        self
    }

    /// Re-resolves the original entry scope against its actual live definition, never against a serialized skill ID alone.
    pub(super) fn retained_profile(
        &self,
        payload: &TaskCheckpointPayload,
    ) -> Result<EffectiveAutonomyProfile> {
        let entry = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.entry");
        let skill = entry
            .map(|entry| {
                serde_json::from_value::<ai_agents_skills::SkillDefinition>(entry.config.clone())
            })
            .transpose()?;
        if let Some(skill) = &skill {
            self.agent.validate_autonomy_skill_binding(skill)?;
        }
        resolve_profile(
            &self.config,
            if skill.is_some() {
                AutonomyScope::Skill
            } else {
                AutonomyScope::Task
            },
            payload.profile.as_deref(),
            skill.as_ref().and_then(|skill| skill.autonomy.as_ref()),
            None,
            Some(&payload.objective),
            &self.host,
        )
    }

    /// Validates capacity responses before claim; answers and Continue cannot refill any allowance or authorize a pending operation.
    pub(super) fn validate_capacity_response(
        &self,
        payload: &TaskCheckpointPayload,
        input: &TaskResumeInput,
    ) -> Result<bool> {
        let Some(pending) = payload
            .pending
            .as_ref()
            .filter(|pending| matches!(pending.kind, TaskPendingKind::Limit))
        else {
            return Ok(false);
        };
        if !super::super::controller::pending_capacity(pending)?.available(payload) {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        match (
            input,
            payload
                .settings
                .on_limit
                .as_ref()
                .and_then(|policy| policy.action),
        ) {
            (TaskResumeInput::Continue, Some(LimitAction::PauseRun)) => {}
            (TaskResumeInput::UserAnswer { request_id, answer }, Some(LimitAction::AskUser))
                if request_id == &pending.id =>
            {
                bounded_value(answer, 65_536)?;
                if answer.as_str().is_none_or(|text| text.trim().is_empty()) {
                    return Err(TaskRunStorageError::NotResumable.into());
                }
            }
            _ => return Err(TaskRunStorageError::NotResumable.into()),
        }
        Ok(true)
    }

    /// Freezes a post-decision position; an answer continues work and cannot grant tools, change objectives or refill budgets.
    pub(super) async fn park_question(
        &self,
        execution: &Arc<RunExecution>,
        todo: Option<&RunTodoAdapter>,
        state: ControllerContinuation,
        reason: &str,
        prompt: String,
        limit: Option<ControllerLimit>,
    ) -> Result<TaskRunResult> {
        if !self.interaction_delivery {
            if let Some(driver) = &state.driver {
                execution
                    .update(|payload| driver.retire_wait_reservation(payload))
                    .await?;
            }
            let snapshot = execution.load_owned().await?;
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
            let fallback = if limit.is_some() {
                payload
                    .settings
                    .on_limit
                    .as_ref()
                    .and_then(|policy| policy.on_unavailable)
            } else {
                payload
                    .settings
                    .premature_finish
                    .as_ref()
                    .and_then(|policy| policy.on_unavailable)
            };
            return self
                .finish(
                    execution,
                    if fallback == Some(InteractionUnavailableAction::Fail) {
                        TaskRunStatus::Failed
                    } else {
                        TaskRunStatus::Incomplete
                    },
                    "interaction_unavailable".into(),
                    Some(state.response),
                )
                .await;
        }
        Box::pin(self.park_review(
            execution,
            todo,
            state,
            ControllerProposal::Question {
                reason: reason.into(),
                prompt,
                limit: limit.map(Box::new),
            },
        ))
        .await
    }

    /// A refused invocation returns to its saved controller position only after native history and owned effects prove a safe boundary.
    pub(super) async fn pause_admission_limit(
        &self,
        execution: &Arc<RunExecution>,
        todo: Option<&RunTodoAdapter>,
        reason: &str,
    ) -> Result<TaskRunResult> {
        let snapshot = execution.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        if payload.pending.is_some() {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let controller = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller")
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let mut state = ControllerContinuation::decision(
            &serde_json::from_value(controller.state["scope"].clone())?,
            &serde_json::from_value(controller.state["lifecycle"].clone())?,
            &serde_json::from_value(controller.state["progress"].clone())?,
            &serde_json::from_value(controller.state["evidence"].clone())?,
            None,
            "",
            AutonomyTurnSource::Continuation,
            false,
        );
        state.lifecycle.skill_segment_completed |= execution.skill_segment_completed();
        let runtime = TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        execution
            .update(|payload| {
                payload.runtime = runtime;
                Ok(())
            })
            .await?;
        Box::pin(self.park_limit(execution, todo, state, reason)).await
    }

    /// Drives the already selected intervention without observing or deciding its cycle again.
    /// Boxing keeps optional planner/pause construction out of ordinary nested model-call stack frames.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn continue_replan<'a>(
        &'a self,
        execution: &'a Arc<RunExecution>,
        todo: Option<&'a RunTodoAdapter>,
        bound: &'a BoundAutonomyProfile,
        executor: &'a RuntimeObservationExecutor,
        journal: &'a TaskValidationJournal,
        objective: &'a str,
        state: &'a mut ControllerContinuation,
    ) -> ControllerFuture<'a, Option<TaskRunResult>> {
        Box::pin(async move {
            let check = bound.replan_check(&state.progress, objective)?;
            state
                .scope
                .validation_bindings
                .insert(check.check.id.clone(), check.config_identity.clone());
            let resumed = state.driver.take();
            if let Some(driver) = self
                .validate(
                    &check,
                    &mut state.scope,
                    &mut state.evidence,
                    execution,
                    bound,
                    executor,
                    journal,
                    resumed,
                )
                .await?
            {
                let reason = driver.capacity_reason.clone();
                state.phase = super::super::controller::ControllerPhase::Replan;
                state.checks = vec![check.check.id.clone()];
                state.next_check = 0;
                state.driver = Some(driver);
                state.review = None;
                if let Some(reason) = reason {
                    return Box::pin(self.park_limit(execution, todo, state.clone(), &reason))
                        .await
                        .map(Some);
                }
                *execution.controller.lock() = Some(Box::new(state.clone()));
                return Box::pin(self.pause_controller(execution, todo))
                    .await
                    .map(Some);
            }
            let plan =
                latest_validation(&state.evidence.validations, &check.check.id, &state.scope)
                    .filter(|result| result.outcome == GateOutcome::Pass)
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            state.instruction = plan.metrics.to_string();
            state.phase = super::super::controller::ControllerPhase::Decision;
            state.checks.clear();
            state.next_check = 0;
            state.review = None;
            state.source = AutonomyTurnSource::Continuation;
            if bound
                .profile
                .supervision
                .as_ref()
                .is_some_and(|policy| policy.plan_changes == Some(ReviewPolicy::RequireApproval))
            {
                return Box::pin(self.park_review(
                    execution,
                    todo,
                    state.clone(),
                    ControllerProposal::Plan {
                        instruction: state.instruction.clone(),
                    },
                ))
                .await
                .map(Some);
            }
            Ok(None)
        })
    }

    /// Capacity decisions are deterministic and consume no provider or tool slot; expiry and uncertainty still take precedence.
    pub(super) async fn park_limit(
        &self,
        execution: &Arc<RunExecution>,
        todo: Option<&RunTodoAdapter>,
        state: ControllerContinuation,
        reason: &str,
    ) -> Result<TaskRunResult> {
        let snapshot = execution.load_owned().await?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
        let action = payload
            .settings
            .on_limit
            .as_ref()
            .and_then(|policy| policy.action)
            .unwrap_or(LimitAction::SummarizeAndStop);
        let logical_wait = state
            .driver
            .as_ref()
            .map(|driver| driver.wait_reservation())
            .transpose()?
            .flatten();
        let safe = !execution.participants.unsettled()
            && !execution.owner.work.unsettled()
            && payload.children.iter().all(|child| child.result.is_some())
            && payload.reservations.iter().all(|r| {
                matches!(
                    r.state,
                    TaskEffectState::Completed | TaskEffectState::Reserved
                ) || (r.state == TaskEffectState::Suspended && logical_wait.as_ref() == Some(&r.id))
            })
            && !execution.has_storage_failure()
            && execution.remaining_duration() > std::time::Duration::ZERO;
        if !safe || !matches!(action, LimitAction::AskUser | LimitAction::PauseRun) {
            if safe && let Some(driver) = &state.driver {
                execution
                    .update(|payload| driver.retire_wait_reservation(payload))
                    .await?;
            }
            return self
                .finish(
                    execution,
                    if action == LimitAction::Fail {
                        TaskRunStatus::Failed
                    } else {
                        TaskRunStatus::LimitReached
                    },
                    reason.into(),
                    Some(state.response),
                )
                .await;
        }
        let resume_turn = state.driver.is_some()
            || matches!(
                execution.stop_reason().as_deref(),
                Some("llm_capacity" | "tool_capacity" | "priced_capacity" | "write_capacity")
            );
        if let Some(stopped) = execution.stop_reason() {
            execution.clear_capacity_stop(&stopped);
        }
        let skill = if resume_turn && state.driver.is_none() {
            payload
                .adapters
                .iter()
                .rev()
                .filter(|adapter| adapter.adapter == "runtime.skill")
                .find_map(|adapter| {
                    serde_json::from_value::<super::super::location::TaskSkillLocation>(
                        adapter.state.clone(),
                    )
                    .ok()
                    .filter(|state| {
                        state.actions.is_none()
                            && state.cursor.next_step < state.cursor.definition.steps.len()
                    })
                })
                .map(Box::new)
        } else {
            None
        };
        if let Some(skill) = &skill {
            self.agent.validate_capacity_skill(skill)?;
        }
        let limit = ControllerLimit {
            reason: reason.into(),
            original: payload.limits,
            tool: state
                .driver
                .as_ref()
                .and_then(|driver| {
                    driver
                        .pending_batch
                        .first()
                        .and_then(|id| driver.requests.get(id))
                })
                .and_then(|request| match request {
                    ValidationObservationRequest::Tool { tool, .. } => Some(tool.clone()),
                    _ => None,
                }),
            resume_turn,
            model: if resume_turn && skill.is_none() && state.driver.is_none() {
                execution.capacity_model.lock().clone()
            } else {
                None
            },
            skill,
        };
        if action == LimitAction::AskUser {
            Box::pin(self.park_question(execution, todo, state, "limit_reached",
                format!("The task needs more capacity ({reason}). The host may revise limits within its ceilings, stop or cancel. An answer alone does not increase capacity."), Some(limit))).await
        } else if !self.interaction_delivery {
            if let Some(driver) = &state.driver {
                execution
                    .update(|payload| driver.retire_wait_reservation(payload))
                    .await?;
            }
            let fallback = payload
                .settings
                .on_limit
                .as_ref()
                .and_then(|policy| policy.on_unavailable);
            self.finish(
                execution,
                if fallback == Some(InteractionUnavailableAction::Fail) {
                    TaskRunStatus::Failed
                } else {
                    TaskRunStatus::Incomplete
                },
                "interaction_unavailable".into(),
                Some(state.response),
            )
            .await
        } else {
            Box::pin(self.park_review(
                execution,
                todo,
                state,
                ControllerProposal::Capacity {
                    limit: Box::new(limit),
                },
            ))
            .await
        }
    }
}
