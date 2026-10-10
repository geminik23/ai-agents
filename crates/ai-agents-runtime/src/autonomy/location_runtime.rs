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

struct TaskSkillJournal<'a> {
    agent: &'a RuntimeAgent,
    context: HashMap<String, Value>,
    actions: Option<Box<crate::autonomy::location::TaskActionsLocation>>,
}
#[async_trait]
impl SkillExecutionObserver for TaskSkillJournal<'_> {
    /// Completed prompt/tool results and rendered pending arguments are acknowledged before the next effect.
    async fn checkpoint(&self, cursor: &SkillExecutionCursor) -> Result<()> {
        cursor.validate(&cursor.definition)?;
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
            state_generation: self
                .agent
                .state_machine
                .as_ref()
                .map(|machine| machine.generation()),
        };
        let id = format!(
            "runtime.skill:{}:{}:{}",
            self.agent.info.id,
            crate::autonomy::current_child_operation().unwrap_or_default(),
            cursor.execution_id
        );
        Box::pin(execution.update(|payload| {
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
        .await
        .map(|_| ())
    }
}

impl RuntimeAgent {
    /// Script execution uses an exact local cursor; initial routing and completed template/prompt work are never replayed.
    pub(super) async fn execute_task_skill(
        &self,
        skill: &SkillDefinition,
        input: &str,
    ) -> Result<String> {
        let actions = crate::autonomy::location::current_actions_parent().map(Box::new);
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
        Box::pin(self.drive_task_skill(&mut cursor, context, actions)).await
    }

    /// Captures the precise pending tool step when the shared executor transfers control; no fake assistant call is committed.
    async fn drive_task_skill(
        &self,
        cursor: &mut SkillExecutionCursor,
        context: HashMap<String, Value>,
        actions: Option<Box<crate::autonomy::location::TaskActionsLocation>>,
    ) -> Result<String> {
        let executor = self
            .skill_executor
            .as_ref()
            .ok_or_else(|| AgentError::Skill("No skill executor configured".into()))?;
        let skill = cursor.definition.clone();
        let result = self
            .observe_purpose(
                ObservationPurpose::SkillPrompt,
                Box::pin(executor.execute_resumable_with_invoker(
                    &skill,
                    cursor,
                    &TaskSkillInvoker(self),
                    &TaskSkillJournal {
                        agent: self,
                        context: context.clone(),
                        actions: actions.clone(),
                    },
                )),
            )
            .await;
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
                state_generation: self
                    .state_machine
                    .as_ref()
                    .map(|machine| machine.generation()),
            }));
            Box::pin(self.retain_location_batch(location)).await?;
            return Err(error);
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
    /// Erasure keeps optional script/action future construction out of nested model polling frames.
    pub(super) fn resume_task_location_batch(
        &self,
        mut input: crate::autonomy::AutonomyTurnInput,
        batch: TaskBatchState,
        request_id: &str,
        response: BatchResponse,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + '_>> {
        let request_id = request_id.to_string();
        Box::pin(async move {
            let location = batch
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
                                    self.execute_tool_record(request),
                                ),
                            ),
                        );
                        let record = match response {
                            BatchResponse::UserAnswer(answer) => {
                                crate::autonomy::suspension::scope_resumed_question(
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
                TaskLocation::Actions(state) => {
                    if record.success {
                        self.context_manager.set(
                            "last_tool_result",
                            Value::String(record.model_output_string()),
                        )?;
                        self.context_manager
                            .set("last_tool_record", serde_json::to_value(&record)?)?;
                    } else if matches!(record.policy.outcome, PermissionOutcome::RequiresApproval) {
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
                    if let Err(error) = state.cursor.accept_tool_result(record) {
                        execution.stop("skill_step_failed");
                        return Box::pin(self.finish_task_location_stop(error.to_string())).await;
                    }
                    let mut content = Box::pin(self.drive_task_skill(
                        &mut state.cursor,
                        state.input_context.clone(),
                        state.actions.clone(),
                    ))
                    .await?;
                    if let Some(mut actions) = state.actions.take() {
                        actions.completed.push(Value::String(content));
                        actions.next_action += 1;
                        return Box::pin(self.resume_task_actions(*actions)).await;
                    }
                    let config = self.get_skill_reflection_config(&state.cursor.definition);
                    if config.requires_evaluation()
                        && config.is_enabled()
                        && self
                            .should_reflect_with_config(
                                &state.cursor.context.user_input,
                                &content,
                                &config,
                            )
                            .await?
                    {
                        content = self
                            .evaluate_and_retry_with_config(
                                &state.cursor.context.user_input,
                                content,
                                &config,
                            )
                            .await?;
                    }
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
