//! Exact state-action continuation preserves transition commit order and the caller's return position.

use super::*;
use crate::autonomy::location::{
    TaskActionsLocation, TaskLocation, TaskStateReturn, TaskTransitionLocation,
};

impl RuntimeAgent {
    /// Exit work finishes before one transition commit; entry work and hooks follow that acknowledged position.
    pub(super) async fn apply_task_transition_target(
        &self,
        from: &str,
        target: &str,
        reason: &str,
        staged: Option<&HashMap<String, Value>>,
    ) -> Result<bool> {
        let machine = self
            .state_machine
            .as_ref()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let return_to = crate::autonomy::location::current_state_return().ok_or_else(|| {
            AgentError::Config("state transition has no exact task return position".into())
        })?;
        if machine.current() != from {
            return Ok(false);
        }
        let Some(_reservation) = self.reserve_state_transition() else {
            return Ok(false);
        };
        let transition = TaskTransitionLocation {
            from_state: from.into(),
            target: target.into(),
            reason: reason.into(),
            staged: staged.cloned(),
            history_before: machine.history(),
            entering: false,
            expected_generation: machine.generation(),
            expected_epoch: self.disambiguation_epoch.load(Ordering::SeqCst),
        };
        Box::pin(self.continue_task_transition(transition, return_to)).await?;
        Ok(true)
    }

    /// Serial action lists use the existing executor and captured configuration, with no completed-effect replay.
    async fn continue_task_transition(
        &self,
        mut transition: TaskTransitionLocation,
        return_to: TaskStateReturn,
    ) -> Result<()> {
        let machine = self
            .state_machine
            .as_ref()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let source = crate::autonomy::current_turn_input(&self.root_turn_gate)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?
            .source;
        if !transition.entering {
            let definition = machine
                .get_definition(&transition.from_state)
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            let mut actions = TaskActionsLocation {
                actions: definition.on_exit.clone(),
                next_action: 0,
                completed: Vec::new(),
                pending: None,
                state: transition.from_state.clone(),
                transition: transition.clone(),
                return_to: return_to.clone(),
                source,
                user_message_committed: self.root_user_message_committed.load(Ordering::SeqCst),
                native_exchanges: self.task_native_expectations(),
            };
            Box::pin(self.drive_task_actions(&mut actions)).await?;
            let admission = self.disambiguation_admission.write().await;
            if machine.current() != transition.from_state
                || machine.generation() != transition.expected_generation
                || self.disambiguation_epoch.load(Ordering::SeqCst) != transition.expected_epoch
            {
                return Err(AgentError::Other(
                    "state transition binding changed during task exit actions".into(),
                ));
            }
            machine.transition_to(&transition.target, &transition.reason)?;
            self.invalidate_pending_confirmation("state_transition")
                .await;
            machine.reset_no_transition();
            if let Some(staged) = &transition.staged {
                self.commit_staged_context_writes(staged);
            }
            transition.entering = true;
            transition.expected_generation = machine.generation();
            transition.expected_epoch = self.disambiguation_epoch.load(Ordering::SeqCst);
            drop(admission);
        }
        let entered = machine.current();
        let definition = machine
            .get_definition(&entered)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let reentry = Self::state_was_previously_entered(
            &entered,
            &transition.from_state,
            &transition.history_before,
        );
        let selected = if reentry && !definition.on_reenter.is_empty() {
            &definition.on_reenter
        } else {
            &definition.on_enter
        };
        let mut actions = TaskActionsLocation {
            actions: selected.clone(),
            next_action: 0,
            completed: Vec::new(),
            pending: None,
            state: entered.clone(),
            transition: transition.clone(),
            return_to,
            source,
            user_message_committed: self.root_user_message_committed.load(Ordering::SeqCst),
            native_exchanges: self.task_native_expectations(),
        };
        Box::pin(self.drive_task_actions(&mut actions)).await?;
        self.hooks
            .on_state_transition(Some(&transition.from_state), &entered, &transition.reason)
            .await;
        Ok(())
    }

    /// The saved phase binds the action definition, state generation and disambiguation epoch before response claim.
    pub(super) fn validate_task_actions(&self, state: &TaskActionsLocation) -> Result<()> {
        state.validate()?;
        let machine = self
            .state_machine
            .as_ref()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let definition = machine
            .get_definition(&state.state)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let expected = if !state.transition.entering {
            &definition.on_exit
        } else if Self::state_was_previously_entered(
            &state.state,
            &state.transition.from_state,
            &state.transition.history_before,
        ) && !definition.on_reenter.is_empty()
        {
            &definition.on_reenter
        } else {
            &definition.on_enter
        };
        if machine.current() != state.state
            || machine.generation() != state.transition.expected_generation
            || self.disambiguation_epoch.load(Ordering::SeqCst) != state.transition.expected_epoch
            || serde_json::to_value(expected)? != serde_json::to_value(&state.actions)?
        {
            return Err(AgentError::Config(
                "state action continuation binding changed".into(),
            ));
        }
        Ok(())
    }

    /// Results and rendered tool arguments are checkpointed at each action boundary under the shared task writer.
    async fn drive_task_actions(&self, state: &mut TaskActionsLocation) -> Result<()> {
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        while state.next_action < state.actions.len() {
            if let Some(reason) = execution.stop_reason() {
                return Err(AgentError::Other(reason));
            }
            let action = state.actions[state.next_action].clone();
            let value = match action {
                StateAction::Tool { tool, args } => {
                    if state.pending.is_none() {
                        state.pending = Some(ToolExecutionRequest::new(
                            uuid::Uuid::new_v4().to_string(),
                            tool,
                            self.render_action_args(&args.unwrap_or_else(|| serde_json::json!({}))),
                            ToolCallSource::StateAction {
                                state: Some(state.state.clone()),
                                action_index: state.next_action,
                            },
                        ));
                    }
                    Box::pin(self.checkpoint_task_actions(state)).await?;
                    let result = crate::autonomy::scope_task_batch(
                        self.execute_tool_record(state.pending.as_ref().unwrap().clone()),
                    )
                    .await;
                    let record = match result {
                        Err(error @ AgentError::TaskSuspended(_)) => {
                            Box::pin(self.retain_location_batch(TaskLocation::Actions(Box::new(
                                state.clone(),
                            ))))
                            .await?;
                            return Err(error);
                        }
                        result => result?,
                    };
                    if record.success {
                        self.context_manager.set(
                            "last_tool_result",
                            Value::String(record.model_output_string()),
                        )?;
                        self.context_manager
                            .set("last_tool_record", serde_json::to_value(&record)?)?;
                    } else if matches!(record.policy.outcome, PermissionOutcome::RequiresApproval) {
                        execution.stop("approval_rejected");
                        return Err(AgentError::HITLRejected(record.model_output_string()));
                    }
                    state.pending = None;
                    serde_json::to_value(record)?
                }
                StateAction::Skill { skill } => {
                    let definition = self
                        .skills
                        .iter()
                        .find(|definition| definition.id == skill)
                        .ok_or_else(|| AgentError::Skill("state action skill not found".into()))?;
                    let response = crate::autonomy::location::scope_actions_parent(
                        state.clone(),
                        Box::pin(self.execute_task_skill(definition, "")),
                    )
                    .await?;
                    Value::String(response)
                }
                StateAction::SetContext { set_context } => {
                    for (key, value) in &set_context {
                        self.context_manager.set(key, value.clone())?;
                    }
                    serde_json::to_value(set_context)?
                }
                StateAction::Prompt {
                    prompt,
                    llm,
                    store_as,
                } => {
                    Box::pin(self.checkpoint_task_actions(state)).await?;
                    let provider = match llm {
                        Some(alias) => self.llm_registry.get(&alias)?,
                        None => self.llm_registry.default()?,
                    };
                    let rendered = self
                        .template_renderer
                        .render(&prompt, &self.build_context_with_overlays())
                        .unwrap_or(prompt);
                    let mut messages = self.memory.get_messages(Some(5)).await?;
                    messages.push(ChatMessage::user(rendered));
                    let response = self
                        .observe_purpose(
                            ObservationPurpose::StateAction,
                            provider.complete(&messages, None),
                        )
                        .await
                        .map_err(|error| AgentError::LLM(error.to_string()))?;
                    if let Some(key) = store_as {
                        self.context_manager
                            .set(&key, Value::String(response.content.clone()))?;
                    }
                    Value::String(response.content)
                }
            };
            state.completed.push(value);
            state.next_action += 1;
            Box::pin(self.checkpoint_task_actions(state)).await?;
        }
        Ok(())
    }

    /// Private action receipts follow the current claim and do not replace tool admission or effect settlement.
    async fn checkpoint_task_actions(&self, state: &TaskActionsLocation) -> Result<()> {
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let id = format!(
            "runtime.actions:{}:{}:{}:{}",
            self.info.id,
            crate::autonomy::current_child_operation().unwrap_or_default(),
            state.transition.expected_generation,
            if state.transition.entering {
                "enter"
            } else {
                "exit"
            }
        );
        let value = serde_json::to_value(state)?;
        Box::pin(execution.update(|payload| {
            let adapter = crate::autonomy::TaskAdapterCheckpoint {
                id: id.clone(),
                adapter: "runtime.actions".into(),
                contract_version: 1,
                config: serde_json::to_value(&state.actions)?,
                state: value,
            };
            if let Some(old) = payload.adapters.iter_mut().find(|old| old.id == id) {
                *old = adapter;
            } else {
                payload.adapters.push(adapter);
            }
            Ok(())
        }))
        .await
        .map(|_| ())
    }

    /// Completes only unfinished actions, commits a saved exit transition once and returns to the already prepared caller.
    pub(super) async fn resume_task_actions(
        &self,
        mut state: TaskActionsLocation,
    ) -> Result<AgentResponse> {
        let reservation = self.reserve_state_transition().ok_or_else(|| {
            AgentError::Other("state continuation already has an active transition".into())
        })?;
        Box::pin(self.drive_task_actions(&mut state)).await?;
        if state.transition.entering {
            self.hooks
                .on_state_transition(
                    Some(&state.transition.from_state),
                    &state.state,
                    &state.transition.reason,
                )
                .await;
        } else {
            let machine = self
                .state_machine
                .as_ref()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            let admission = self.disambiguation_admission.write().await;
            if machine.current() != state.state
                || machine.generation() != state.transition.expected_generation
                || self.disambiguation_epoch.load(Ordering::SeqCst)
                    != state.transition.expected_epoch
            {
                return Err(AgentError::Other(
                    "state exit continuation lost its commit position".into(),
                ));
            }
            machine.transition_to(&state.transition.target, &state.transition.reason)?;
            self.invalidate_pending_confirmation("state_transition")
                .await;
            machine.reset_no_transition();
            if let Some(staged) = &state.transition.staged {
                self.commit_staged_context_writes(staged);
            }
            state.transition.entering = true;
            state.transition.expected_generation = machine.generation();
            state.transition.expected_epoch = self.disambiguation_epoch.load(Ordering::SeqCst);
            drop(admission);
            Box::pin(self.continue_task_transition(state.transition, state.return_to.clone()))
                .await?;
        }
        drop(reservation);
        Box::pin(self.resume_state_return(state.return_to)).await
    }

    /// Return-position dispatch never starts another root turn or repeats completed semantic/input processing.
    async fn resume_state_return(&self, position: TaskStateReturn) -> Result<AgentResponse> {
        match position {
            TaskStateReturn::Prepared { data } => Box::pin(self.run_accepted_input(data)).await,
            TaskStateReturn::ReadyContext { input } => {
                self.context_manager.refresh_per_turn().await?;
                self.context_manager.validate()?;
                self.clear_disambiguation_context();
                Box::pin(self.run_after_ready_context(&input)).await
            }
            TaskStateReturn::ModelContinue { mut state } => {
                if let Some(finalization) = &mut state.deferred_final {
                    finalization.finalize_on_resume = true;
                }
                self.memory
                    .add_message(ChatMessage::assistant(
                        "(Transitioned to new state - tool call handled by workflow)",
                    ))
                    .await?;
                Box::pin(self.run_committed_task_loop(*state)).await
            }
            TaskStateReturn::SkillFinish { response } => {
                self.finish_turn_if_root(&response).await?;
                Ok(response)
            }
            TaskStateReturn::ModelFinish { state } => {
                let state = *state;
                let result = crate::autonomy::location::scope_state_return(
                    TaskStateReturn::ModelFinish {
                        state: Box::new(state.clone()),
                    },
                    Box::pin(self.post_transition_processing(
                        &state.processed_input,
                        state.content.clone(),
                        true,
                    )),
                )
                .await?;
                let applied = self
                    .apply_post_loop_result(&state.processed_input, result)
                    .await?;
                let response = self.build_agent_response(AgentResponseParts {
                    content: applied.content,
                    all_tool_calls: applied.tool_calls.unwrap_or(state.all_tool_calls),
                    reasoning_mode: state.reasoning_mode,
                    auto_detected: state.auto_detected,
                    iterations: state.iterations,
                    thinking: state.thinking,
                    reflection_metadata: state.reflection_metadata,
                });
                self.finish_turn_if_root(&response).await?;
                Ok(response)
            }
        }
    }
}
