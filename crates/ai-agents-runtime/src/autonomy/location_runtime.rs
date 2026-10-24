//! Owner-authorized non-model continuation uses the same guarded executor and exact script cursor.

use super::*;
use crate::autonomy::TaskBatchState;
use crate::autonomy::location::{TaskLocation, TaskSkillLocation};
use crate::autonomy::suspension::{BatchAuthorization, BatchResponse};
use ai_agents_core::ToolInvoker;
use ai_agents_skills::{SkillExecutionCursor, SkillExecutionObserver};
use async_trait::async_trait;

struct TaskSkillInvoker<'a>(&'a RuntimeAgent);
#[async_trait]
impl ToolInvoker for TaskSkillInvoker<'_> {
    /// The caller owns a reconstructible script location; its private scope does not widen tool grants.
    async fn invoke_tool(&self, request: ToolExecutionRequest) -> Result<ToolExecutionRecord> {
        crate::autonomy::scope_task_batch(self.0.execute_tool_record(request)).await
    }
}

// The original script generation restricts real dispatch as well as checkpoints; current tool scope alone cannot adopt an old cursor.
struct SkillGeneration {
    machine: Option<Arc<StateMachine>>,
    expected: Option<u64>,
}
impl std::fmt::Debug for SkillGeneration {
    // Avoid exposing script or host state in admission diagnostics.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SkillGeneration")
    }
}
impl crate::autonomy::TaskFinalAdmission for SkillGeneration {
    // Awaiting storage or provider preparation cannot replace the generation that admitted this script.
    fn validate(&self) -> Result<()> {
        if self.machine.as_ref().map(|machine| machine.generation()) != self.expected {
            return Err(AgentError::Config(
                "skill continuation state or scope changed".into(),
            ));
        }
        Ok(())
    }
}

struct TaskSkillJournal<'a> {
    agent: &'a RuntimeAgent,
    context: HashMap<String, Value>,
    actions: Option<Box<crate::autonomy::location::TaskActionsLocation>>,
    generation: Option<u64>,
}
#[async_trait]
impl SkillExecutionObserver for TaskSkillJournal<'_> {
    /// Completed results retain their original binding before publication and after storage acknowledgement, including final prompt results.
    async fn checkpoint(&self, cursor: &SkillExecutionCursor) -> Result<()> {
        self.agent.validate_skill_cursor_binding(
            cursor,
            self.actions.as_deref(),
            self.generation,
        )?;
        if cursor.context.step_results.len() > crate::autonomy::MAX_TASK_CHECKPOINT_RECORDS {
            return Err(crate::autonomy::TaskRunStorageError::CheckpointTooLarge.into());
        }
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let state = TaskSkillLocation {
            cursor: cursor.clone(),
            input_context: self.context.clone(),
            user_message_committed: self
                .agent
                .root_user_message_committed
                .load(Ordering::SeqCst),
            native_exchanges: self.agent.task_native_expectations(),
            actions: self.actions.clone(),
            source: crate::autonomy::current_turn_input(&self.agent.root_turn_gate)
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?
                .source,
            state_generation: self.generation,
        };
        let id = format!(
            "runtime.skill:{}:{}:{}",
            self.agent.info.id,
            crate::autonomy::current_child_operation().unwrap_or_default(),
            cursor.execution_id
        );
        Box::pin(execution.update(|payload| {
            self.agent.validate_skill_cursor_binding(
                cursor,
                self.actions.as_deref(),
                self.generation,
            )?;
            let adapter = crate::autonomy::TaskAdapterCheckpoint {
                id: id.clone(),
                adapter: "runtime.skill".into(),
                contract_version: 1,
                config: serde_json::to_value(&cursor.definition)?,
                state: serde_json::to_value(&state)?,
            };
            if let Some(old) = payload.adapters.iter_mut().find(|old| old.id == id) {
                *old = adapter;
            } else {
                payload.adapters.push(adapter);
            }
            Ok(())
        }))
        .await?;
        self.agent
            .validate_skill_cursor_binding(cursor, self.actions.as_deref(), self.generation)
    }
}

impl RuntimeAgent {
    // Only an executing task can supply the original script finishing restriction; ordinary chat has no added guard.
    pub(super) fn task_skill_finishing_guard(
        &self,
    ) -> Option<Arc<dyn crate::autonomy::TaskFinalAdmission>> {
        crate::autonomy::current_execution().and_then(|execution| {
            execution
                .owner
                .skill_finishing
                .lock()
                .get(&self.info.id)
                .cloned()
        })
    }

    // Reflection is still response-dependent script work and must not dispatch a retry after original-generation invalidation.
    pub(super) async fn reflect_skill_response(
        &self,
        input: &str,
        mut response: String,
        config: &ReflectionConfig,
    ) -> Result<String> {
        let guard = self.task_skill_finishing_guard();
        if let Some(guard) = &guard {
            guard.validate()?;
        }
        ai_agents_llm::scope_invocation_evidence(guard.iter().cloned().collect(), async {
            if config.requires_evaluation()
                && config.is_enabled()
                && self
                    .should_reflect_with_config(input, &response, config)
                    .await?
            {
                response = self
                    .evaluate_and_retry_with_config(input, response, config)
                    .await?;
            }
            if let Some(guard) = &guard {
                guard.validate()?;
            }
            Ok(response)
        })
        .await
    }

    // Pending-location shape is separate from live binding so completed final-prompt cursors can be checked too.
    fn validate_skill_cursor_binding(
        &self,
        cursor: &SkillExecutionCursor,
        actions: Option<&crate::autonomy::location::TaskActionsLocation>,
        generation: Option<u64>,
    ) -> Result<()> {
        use crate::autonomy::TaskFinalAdmission;
        SkillGeneration {
            machine: self.state_machine.clone(),
            expected: generation,
        }
        .validate()?;
        if let Some(actions) = actions {
            self.validate_task_actions(actions)?;
        } else if !self
            .get_available_skills()
            .iter()
            .any(|skill| skill.id == cursor.definition.id)
        {
            return Err(AgentError::Config(
                "skill continuation state or scope changed".into(),
            ));
        }
        let actual = self
            .skills
            .iter()
            .find(|skill| skill.id == cursor.definition.id)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        cursor.validate(actual)
    }

    /// Explicit entry selects an installed skill within the current state scope without invoking the semantic router.
    pub(crate) fn admit_autonomy_skill(&self, skill_id: &str) -> Result<SkillDefinition> {
        self.get_available_skills()
            .into_iter()
            .find(|skill| skill.id == skill_id)
            .cloned()
            .ok_or_else(|| {
                AgentError::Skill(
                    "selected autonomy skill is unavailable in the current scope".into(),
                )
            })
    }

    /// Retained entry data must agree with both the loaded definition and the private owner that admitted it.
    pub(crate) fn validate_autonomy_skill_binding(&self, skill: &SkillDefinition) -> Result<()> {
        let owner = self
            .autonomy_owner
            .read()
            .clone()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)?;
        owner.check()?;
        let admitted = owner.skill.read();
        let actual = self
            .skills
            .iter()
            .find(|actual| actual.id == skill.id)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if serde_json::to_value(actual)? != serde_json::to_value(skill)?
            || serde_json::to_value(&*admitted)? != serde_json::to_value(Some(skill))?
        {
            return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
        }
        Ok(())
    }

    /// Capacity may stop before a prompt as well as a tool, so its exact prefix is validated without requiring a pending tool request.
    pub(crate) fn validate_capacity_skill(&self, state: &TaskSkillLocation) -> Result<()> {
        let actual = self
            .skills
            .iter()
            .find(|skill| skill.id == state.cursor.definition.id)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        state.cursor.validate(actual)?;
        if state.actions.is_some()
            || state.cursor.next_step >= actual.steps.len()
            || self
                .state_machine
                .as_ref()
                .map(|machine| machine.generation())
                != state.state_generation
            || !self
                .get_available_skills()
                .iter()
                .any(|skill| skill.id == actual.id)
        {
            return Err(crate::autonomy::TaskRunStorageError::NotResumable.into());
        }
        crate::autonomy::bounded_value(
            &serde_json::to_value(state)?,
            crate::autonomy::MAX_TASK_CHECKPOINT_BYTES,
        )
    }

    /// A capacity wait has no approval receipt; its pending script operation reenters ordinary authorization and shared resource admission.
    pub(crate) fn resume_capacity_skill(
        &self,
        mut state: Box<TaskSkillLocation>,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + '_>> {
        Box::pin(async move {
            self.validate_capacity_skill(&state)?;
            self.begin_root_turn();
            let _cleanup = RootTurnCleanup::new(self);
            self.root_user_message_committed
                .store(state.user_message_committed, Ordering::SeqCst);
            *self.active_native_exchanges.write() = state
                .native_exchanges
                .iter()
                .map(|exchange| ActiveNativeExchange {
                    exchange_id: exchange.exchange_id.clone(),
                    call_ids: exchange.call_ids.clone(),
                })
                .collect();
            let mut content = Box::pin(self.drive_task_skill(
                &mut state.cursor,
                state.input_context.clone(),
                None,
                state.state_generation,
            ))
            .await?;
            content = self
                .reflect_skill_response(
                    &state.cursor.context.user_input,
                    content,
                    &self.get_skill_reflection_config(&state.cursor.definition),
                )
                .await?;
            self.commit_root_user_message(&state.cursor.context.user_input)
                .await?;
            self.handle_skill_response(
                &state.cursor.context.user_input,
                &state.cursor.definition.id,
                content,
                &state.input_context,
            )
            .await
        })
    }

    /// Script execution uses an exact local cursor; initial routing and completed template/prompt work are never replayed.
    pub(super) async fn execute_task_skill(
        &self,
        skill: &SkillDefinition,
        input: &str,
    ) -> Result<String> {
        let actions = crate::autonomy::location::current_actions_parent().map(Box::new);
        let generation = actions
            .as_ref()
            .map(|actions| actions.transition.expected_generation)
            .or_else(|| {
                self.state_machine
                    .as_ref()
                    .map(|machine| machine.generation())
            });
        if actions.is_none() {
            self.commit_root_user_message(input).await?;
        }
        let mut cursor = SkillExecutionCursor::new(skill, input, serde_json::json!({}));
        let context = self
            .active_turn_context
            .read()
            .as_ref()
            .map(|turn| turn.input_context.clone())
            .unwrap_or_default();
        Box::pin(self.drive_task_skill(&mut cursor, context, actions, generation)).await
    }

    /// Runs an originally bound cursor with pre-dispatch guards and rejects stale result advancement or segment credit after awaited work.
    /// Suspension preserves the same generation rather than rebinding old script progress to the latest state.
    async fn drive_task_skill(
        &self,
        cursor: &mut SkillExecutionCursor,
        context: HashMap<String, Value>,
        actions: Option<Box<crate::autonomy::location::TaskActionsLocation>>,
        generation: Option<u64>,
    ) -> Result<String> {
        let executor = self
            .skill_executor
            .as_ref()
            .ok_or_else(|| AgentError::Skill("No skill executor configured".into()))?;
        self.validate_skill_cursor_binding(cursor, actions.as_deref(), generation)?;
        let guard: Arc<dyn crate::autonomy::TaskFinalAdmission> = Arc::new(SkillGeneration {
            machine: self.state_machine.clone(),
            expected: generation,
        });
        if actions.is_none()
            && let Some(execution) = crate::autonomy::current_execution()
        {
            execution
                .owner
                .skill_finishing
                .lock()
                .insert(self.info.id.clone(), guard.clone());
        }
        let skill = cursor.definition.clone();
        let result = ai_agents_llm::scope_invocation_evidence(
            vec![guard.clone()],
            self.observe_purpose(
                ObservationPurpose::SkillPrompt,
                Box::pin(executor.execute_resumable_with_invoker(
                    &skill,
                    cursor,
                    &TaskSkillInvoker(self),
                    &TaskSkillJournal {
                        agent: self,
                        context: context.clone(),
                        actions: actions.clone(),
                        generation,
                    },
                )),
            ),
        )
        .await;
        self.validate_skill_cursor_binding(cursor, actions.as_deref(), generation)?;
        if let Err(error @ AgentError::TaskSuspended(_)) = result {
            let location = TaskLocation::Skill(Box::new(TaskSkillLocation {
                cursor: cursor.clone(),
                input_context: context,
                user_message_committed: self.root_user_message_committed.load(Ordering::SeqCst),
                native_exchanges: self.task_native_expectations(),
                actions,
                source: crate::autonomy::current_turn_input(&self.root_turn_gate)
                    .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?
                    .source,
                state_generation: generation,
            }));
            Box::pin(self.retain_location_batch(location)).await?;
            return Err(error);
        }
        if result.is_ok()
            && actions.is_none()
            && crate::autonomy::current_child_operation().is_none()
        {
            let execution = crate::autonomy::current_execution()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            if execution.is_coordinator(&self.info.id) {
                execution
                    .complete_skill_segment(&cursor.definition, guard.as_ref())
                    .await?;
            }
        }
        result
    }

    /// A non-model tool batch carries its actual source and location rather than inventing model loop/history state.
    pub(super) async fn retain_location_batch(&self, location: TaskLocation) -> Result<()> {
        location.validate()?;
        let request = location.request()?.clone();
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let approvals = execution.take_parked_approvals(&self.info.id);
        let executor_cursors = execution
            .parked_executor_cursors
            .lock()
            .remove(&(self.info.id.clone(), request.call_id.clone()))
            .into_iter()
            .collect();
        let batch = TaskBatchState {
            version: 1,
            content: String::new(),
            calls: vec![ToolCall {
                id: request.call_id,
                name: request.requested_name,
                arguments: request.arguments,
            }],
            results: vec![None],
            appended: 0,
            approvals,
            loop_state: None,
            authorized: Vec::new(),
            rejected: false,
            executor_cursors,
            location: Some(location),
            interaction: None,
        };
        batch.validate()?;
        Box::pin(execution.retain_message_batch(&self.info.id, batch.clone())).await?;
        execution.retain_batch(&self.info.id, batch);
        Ok(())
    }

    /// Private live configuration is checked before claim, not inferred from the serialized location kind or skill ID.
    pub(crate) fn validate_task_location(&self, location: &TaskLocation) -> Result<()> {
        location.validate()?;
        match location {
            TaskLocation::Actions(state) => self.validate_task_actions(state),
            TaskLocation::Observation(state) => {
                if state.identity.key.agent_id != self.info.id
                    || self
                        .autonomy_owner
                        .read()
                        .as_ref()
                        .is_none_or(|owner| owner.run_id != state.identity.key.run_id)
                {
                    return Err(AgentError::Config(
                        "observation runtime owner changed".into(),
                    ));
                }
                Ok(())
            }
            TaskLocation::Skill(state) => {
                if self
                    .state_machine
                    .as_ref()
                    .map(|machine| machine.generation())
                    != state.state_generation
                    || (state.actions.is_none()
                        && !self
                            .get_available_skills()
                            .iter()
                            .any(|skill| skill.id == state.cursor.definition.id))
                {
                    return Err(AgentError::Config(
                        "skill continuation state or scope changed".into(),
                    ));
                }
                if let Some(actions) = &state.actions {
                    self.validate_task_actions(actions)?;
                }
                let actual = self
                    .skills
                    .iter()
                    .find(|skill| skill.id == state.cursor.definition.id)
                    .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                state.cursor.validate(actual)
            }
        }
    }

    /// Resumes one exact non-model operation under its retained root owner, then advances the original driver.
    /// Location binding is rechecked under the root gate before effects; erasure bounds nested polling stack construction.
    pub(super) fn resume_task_location_batch(
        &self,
        mut input: crate::autonomy::AutonomyTurnInput,
        batch: TaskBatchState,
        request_id: &str,
        response: BatchResponse,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + '_>> {
        let request_id = request_id.to_string();
        Box::pin(async move {
            let mut location = batch
                .location
                .clone()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            self.validate_task_location(&location)?;
            input.source = location.source();
            let pending = batch
                .approvals
                .iter()
                .find(|pending| pending.request_id == request_id)
                .cloned()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            let _gate = self.root_turn_gate.clone().lock_owned().await;
            if self
                .autonomy_owner
                .read()
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &input.owner))
            {
                return Err(AgentError::Other("location resume owner changed".into()));
            }
            input.owner.check()?;
            self.validate_task_location(&location)?;
            let mut identities = current_runtime_gate_identity_stack().to_vec();
            if identities
                .iter()
                .any(|gate| Arc::ptr_eq(gate, &self.root_turn_gate))
            {
                return Err(AgentError::Other("reentrant location resume".into()));
            }
            identities.push(self.root_turn_gate.clone());
            let identities: RootTurnGateIdentityStack = identities.into();
            scope_runtime_gate_identity_stack(
                &identities,
                crate::autonomy::scope_turn(
                    input,
                    Box::pin(async {
                        self.begin_root_turn();
                        let _cleanup = RootTurnCleanup::new(self);
                        self.root_user_message_committed
                            .store(location.root_state().0, Ordering::SeqCst);
                        *self.active_native_exchanges.write() = location
                            .root_state()
                            .1
                            .iter()
                            .map(|exchange| ActiveNativeExchange {
                                exchange_id: exchange.exchange_id.clone(),
                                call_ids: exchange.call_ids.clone(),
                            })
                            .collect();
                        let execution = crate::autonomy::current_execution()
                            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                        self.validate_active_native_history(
                            &self.memory.get_messages(None).await?,
                            false,
                        )?;
                        if matches!(response, BatchResponse::QuestionTimeout) {
                            self.retire_location_observations(&mut location);
                            execution.stop("question_timeout");
                            self.acknowledge_completed_task_batch().await?;
                            execution.parked_batches.lock().remove(&self.info.id);
                            return Box::pin(self.finish_task_location_stop(
                                "Question expired without an answer".into(),
                            ))
                            .await;
                        }
                        let mut receipts = batch.authorized.clone();
                        if let BatchResponse::Approval(result) = &response {
                            receipts.push(BatchAuthorization {
                                request_id: pending.request_id.clone(),
                                call_id: pending.call_id.clone(),
                                trigger: pending.trigger.clone(),
                                context: pending.context.clone(),
                                result: result.clone(),
                            });
                        }
                        let request = location.request()?.clone();
                        let invocation = crate::autonomy::suspension::scope_resumed_executor(
                            batch.executor_cursors.clone(),
                            crate::autonomy::suspension::scope_resumed_approval(
                                receipts.clone(),
                                crate::autonomy::scope_task_batch(
                                    ai_agents_llm::scope_invocation_evidence(
                                        match &location {
                                            TaskLocation::Skill(state) => {
                                                vec![Arc::new(SkillGeneration {
                                                    machine: self.state_machine.clone(),
                                                    expected: state.state_generation,
                                                })
                                                    as Arc<dyn crate::autonomy::TaskFinalAdmission>]
                                            }
                                            _ => vec![],
                                        },
                                        self.execute_tool_record(request),
                                    ),
                                ),
                            ),
                        );
                        let record = match response {
                            BatchResponse::UserAnswer(answer) => {
                                crate::autonomy::suspension::scope_resumed_question(
                                    pending.request_id.clone(),
                                    pending.call_id.clone(),
                                    pending.context.clone(),
                                    answer,
                                    invocation,
                                )
                                .await
                            }
                            _ => invocation.await,
                        };
                        let record = match record {
                            Err(error @ AgentError::TaskSuspended(_)) => {
                                self.retain_location_batch(location).await?;
                                if let Some(retained) =
                                    execution.parked_batches.lock().get_mut(&self.info.id)
                                {
                                    retained.authorized = receipts;
                                }
                                return Err(error);
                            }
                            record => record?,
                        };
                        self.acknowledge_completed_task_batch().await?;
                        execution.parked_batches.lock().remove(&self.info.id);
                        Box::pin(self.finish_task_location_record(location, record)).await
                    }),
                ),
            )
            .await
        })
    }

    /// Known-safe terminal interactions record one response and complete the original root lifecycle without another model call.
    async fn finish_task_location_stop(&self, content: String) -> Result<AgentResponse> {
        self.memory
            .add_message(ChatMessage::assistant(&content))
            .await?;
        let response = AgentResponse::new(content);
        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    /// Advances the bound caller only after its original tool or message has settled; root finalization remains single-owner.
    /// A changed state-action binding cannot publish its settled result into a newer transition.
    /// The boxed driver prevents unused non-model branches from multiplying the model-message stack.
    pub(super) fn finish_task_location_record(
        &self,
        mut location: TaskLocation,
        record: ToolExecutionRecord,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + '_>> {
        Box::pin(async move {
            let execution = crate::autonomy::current_execution()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            match &mut location {
                TaskLocation::Observation(_) => {
                    Ok(AgentResponse::new(serde_json::to_string(&record)?))
                }
                TaskLocation::Actions(state) => {
                    self.validate_task_actions(state)?;
                    if record.success {
                        self.context_manager.set(
                            "last_tool_result",
                            Value::String(record.model_output_string()),
                        )?;
                        self.context_manager
                            .set("last_tool_record", serde_json::to_value(&record)?)?;
                    } else if matches!(record.policy.outcome, PermissionOutcome::RequiresApproval) {
                        self.retire_speculative_actions(state);
                        execution.stop("approval_rejected");
                        return Box::pin(
                            self.finish_task_location_stop("State action rejected".into()),
                        )
                        .await;
                    }
                    state.completed.push(serde_json::to_value(&record)?);
                    state.next_action += 1;
                    state.pending = None;
                    Box::pin(self.resume_task_actions((**state).clone())).await
                }
                TaskLocation::Skill(state) => {
                    self.validate_skill_cursor_binding(
                        &state.cursor,
                        state.actions.as_deref(),
                        state.state_generation,
                    )?;
                    if let Err(error) = state.cursor.accept_tool_result(record) {
                        if let Some(actions) = &mut state.actions {
                            self.retire_speculative_actions(actions);
                        }
                        execution.stop("skill_step_failed");
                        return Box::pin(self.finish_task_location_stop(error.to_string())).await;
                    }
                    let mut content = Box::pin(self.drive_task_skill(
                        &mut state.cursor,
                        state.input_context.clone(),
                        state.actions.clone(),
                        state.state_generation,
                    ))
                    .await?;
                    if let Some(mut actions) = state.actions.take() {
                        actions.completed.push(Value::String(content));
                        actions.next_action += 1;
                        return Box::pin(self.resume_task_actions(*actions)).await;
                    }
                    content = self
                        .reflect_skill_response(
                            &state.cursor.context.user_input,
                            content,
                            &self.get_skill_reflection_config(&state.cursor.definition),
                        )
                        .await?;
                    self.commit_root_user_message(&state.cursor.context.user_input)
                        .await?;
                    self.handle_skill_response(
                        &state.cursor.context.user_input,
                        &state.cursor.definition.id,
                        content,
                        &state.input_context,
                    )
                    .await
                }
            }
        })
    }
}
