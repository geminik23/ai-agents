//! Owner-authorized committed continuation never reroutes or repeats assistant batch commit.

use super::*;
use crate::autonomy::{TaskBatchState, TaskContinuation, TaskLoopState, TaskRuntimeCheckpoint};
use serde_json::json;

impl RuntimeAgent {
    /// Checks the live parent's definition and registered target before a group response can claim execution.
    pub(crate) fn task_group_child(
        &self,
        group: &crate::autonomy::TaskGroupState,
    ) -> Result<Arc<RuntimeAgent>> {
        let definition = self
            .state_machine
            .as_ref()
            .and_then(|machine| machine.current_definition())
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if group.frame.runtime_id != self.info.id
            || serde_json::to_value(&definition)? != group.frame.definition
            || definition.delegate.as_deref() != Some(group.frame.delegate_id.as_str())
        {
            return Err(AgentError::Config(
                "parent composition configuration changed".into(),
            ));
        }
        let child = self
            .spawner_registry
            .as_ref()
            .and_then(|registry| registry.get(&group.frame.delegate_id))
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if child.info.id != group.child_runtime_id {
            return Err(AgentError::Config(
                "group child implementation binding changed".into(),
            ));
        }
        child.preflight_standalone_autonomy()?;
        Ok(child)
    }

    /// Resumes only the saved parent dispatch frame; the child consumes its exact request through inherited live authority.
    pub(crate) async fn resume_task_group(
        &self,
        input: crate::autonomy::AutonomyTurnInput,
        group: crate::autonomy::TaskGroupState,
        response: crate::autonomy::suspension::BatchResponse,
    ) -> Result<AgentResponse> {
        self.task_group_child(&group)?;
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|owner| !Arc::ptr_eq(owner, &input.owner))
        {
            return Err(AgentError::Other("group resume owner mismatch".into()));
        }
        let identities = current_runtime_gate_identity_stack();
        if identities
            .iter()
            .any(|gate| Arc::ptr_eq(gate, &self.root_turn_gate))
        {
            return Err(AgentError::Other("reentrant group resume".into()));
        }
        let _gate = self.root_turn_gate.clone().lock_owned().await;
        input.owner.check()?;
        let mut ancestry = identities.to_vec();
        ancestry.push(self.root_turn_gate.clone());
        let ancestry: RootTurnGateIdentityStack = ancestry.into();
        let definition = self
            .state_machine
            .as_ref()
            .and_then(|machine| machine.current_definition())
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let frame = group.frame.clone();
        let run = scope_runtime_gate_identity_stack(
            &ancestry,
            crate::autonomy::scope_turn(
                input,
                Box::pin(async {
                    self.begin_root_turn();
                    let _cleanup = RootTurnCleanup::new(self);
                    self.root_user_message_committed
                        .store(frame.user_message_committed, Ordering::SeqCst);
                    self.update_active_turn_context(&frame.input, frame.input_context.clone());
                    *self.active_native_exchanges.write() = frame
                        .native_exchanges
                        .iter()
                        .map(|exchange| ActiveNativeExchange {
                            exchange_id: exchange.exchange_id.clone(),
                            call_ids: exchange.call_ids.clone(),
                        })
                        .collect();
                    self.validate_active_native_history(
                        &self.memory.get_messages(None).await?,
                        false,
                    )?;
                    crate::autonomy::composition::scope_group_response(
                        group.child_operation,
                        group.child_request.id,
                        response,
                        crate::autonomy::composition::scope_resuming_delegate(
                            self.info.id.clone(),
                            Box::pin(self.handle_delegated_state(
                                &frame.input,
                                &frame.delegate_id,
                                &definition,
                            )),
                        ),
                    )
                    .await
                }),
            ),
        );
        if let Some(actor) = frame.parent_actor.clone() {
            scope_actor_context(actor, run).await
        } else {
            run.await
        }
    }

    /// Cancels parked child native history without invoking tools; parent acknowledgement still owns final release.
    pub(crate) async fn cancel_task_group(
        &self,
        owner: &Arc<crate::autonomy::RunOwner>,
        group: &crate::autonomy::TaskGroupState,
        participants: &crate::autonomy::Participants,
    ) -> Result<TaskRuntimeCheckpoint> {
        owner.check()?;
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|active| !Arc::ptr_eq(active, owner))
        {
            return Err(AgentError::Other(
                "group cancellation owner mismatch".into(),
            ));
        }
        let child = self.task_group_child(group)?;
        let child_owner = participants
            .owner_for(&child.root_turn_gate)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        child
            .cancel_task_batch(&child_owner, group.batch.clone())
            .await
    }

    /// Resumes a parked participant's committed batch rather than running its original input pipeline again.
    /// Enrollment and the retained owner protect child-local gates; unacknowledged outcomes retain recovery custody.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn resume_parked_task_child(
        &self,
        input: &str,
        actor: Option<crate::TurnActorContext>,
        execution: Arc<crate::autonomy::RunExecution>,
        operation: String,
        pending: crate::autonomy::TaskPendingRequest,
        batch: TaskBatchState,
        response: crate::autonomy::suspension::BatchResponse,
    ) -> Result<AgentResponse> {
        let owner = execution
            .participants
            .owner_for(&self.root_turn_gate)
            .ok_or_else(|| AgentError::Config("parked child has no retained owner".into()))?;
        owner.check()?;
        if actor
            .as_ref()
            .and_then(|actor| actor.effective_actor_id())
            .is_some_and(|actor| owner.actor_id.as_deref() != Some(actor))
        {
            return Err(AgentError::Other(
                "parked child actor binding changed".into(),
            ));
        }
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|current| !Arc::ptr_eq(current, &owner))
        {
            return Err(AgentError::Other("parked child owner changed".into()));
        }
        let guard = tokio::time::timeout(
            execution.remaining_duration(),
            self.root_turn_gate.clone().lock_owned(),
        )
        .await
        .map_err(|_| AgentError::Other("child resume gate deadline exceeded".into()))?;
        let mut lease = execution
            .enroll_child(owner.clone(), self.autonomy_owner.clone())
            .await?;
        drop(guard);
        let mut cleanup = crate::autonomy::OwnedTurnCleanup::new(owner.clone());
        let turn = crate::autonomy::AutonomyTurnInput {
            owner,
            objective: input.into(),
            controller_message: json!({"delegated_objective":input,"operation":operation})
                .to_string(),
            source: crate::autonomy::AutonomyTurnSource::ResumeAfterApproval,
        };
        let run = crate::autonomy::scope_child_operation(
            operation.clone(),
            Box::pin(self.resume_task_batch(turn, batch, &pending.id, response)),
        );
        let outcome = if let Some(actor) = actor {
            scope_actor_context(actor, run).await
        } else {
            run.await
        };
        if matches!(&outcome, Err(AgentError::TaskSuspended(_))) {
            let batch = execution
                .parked_batches
                .lock()
                .get(&self.info.id)
                .cloned()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            execution
                .checkpoint_child_park(&operation, self.task_suspension_snapshot().await?, batch)
                .await?;
            lease.acknowledge_park();
        } else {
            let runtime = TaskRuntimeCheckpoint::between_turns(self.save_state_full().await?)?;
            execution
                .checkpoint_child(&operation, runtime, input, Some(&outcome))
                .await?;
            lease.acknowledge();
        }
        cleanup.finish();
        outcome
    }

    /// Cleanup may be retried after terminal acknowledgement, but can never clear a newer runtime owner.
    pub(crate) async fn release_reconciled_autonomy_owner(
        &self,
        owner: &Arc<crate::autonomy::RunOwner>,
    ) -> Result<()> {
        let _gate = self.root_turn_gate.lock().await;
        let mut current = self.autonomy_owner.write();
        if current
            .as_ref()
            .is_some_and(|active| !Arc::ptr_eq(active, owner))
        {
            return Err(AgentError::Other("recovery runtime owner mismatch".into()));
        }
        *current = None;
        Ok(())
    }

    /// Checks an implementation-owned answer shape without invoking a host or admitting tool work.
    pub(crate) fn validate_task_question_answer(
        &self,
        call: &ToolCall,
        question: &Option<Value>,
        answer: &Value,
    ) -> Result<()> {
        let tool = self
            .tools
            .get(&call.name)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if tool.task_question(&call.arguments)? != *question {
            return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
        }
        tool.task_question_result(&call.arguments, answer)?;
        Ok(())
    }

    /// Appends only the completed ordered prefix; later parallel results remain in the exact batch checkpoint.
    pub(super) async fn append_task_batch_prefix(
        &self,
        state: &mut TaskBatchState,
        all_calls: &mut Vec<ToolCall>,
    ) -> Result<()> {
        let native = Self::is_native_tool_call_content(&state.content)?;
        while state.appended < state.calls.len() {
            let Some(result) = &state.results[state.appended] else {
                break;
            };
            let output = match result {
                Ok(output) => output.clone(),
                Err(error) => format!("Error: {error}"),
            };
            let call = &state.calls[state.appended];
            self.memory
                .add_message(Self::tool_result_message(call, &output, native)?)
                .await?;
            all_calls.push(call.clone());
            state.appended += 1;
        }
        Ok(())
    }

    /// Captures turn-local values before the caller unwinds its ordinary root bookkeeping.
    pub(super) fn retain_task_loop(&self, mut state: TaskLoopState) -> Result<()> {
        state.native_exchanges = self.task_native_expectations();
        let execution = crate::autonomy::current_execution()
            .ok_or_else(|| AgentError::Config("task loop has no owner".into()))?;
        let mut batches = execution.parked_batches.lock();
        let batch = batches
            .get_mut(&self.info.id)
            .ok_or_else(|| AgentError::Config("task batch continuation missing".into()))?;
        batch.loop_state = Some(state);
        batch.validate()
    }

    /// Captures all current-root signed exchange expectations, including earlier completed batches in this turn.
    pub(super) fn task_native_expectations(&self) -> Vec<crate::autonomy::TaskNativeExchange> {
        self.active_native_exchanges
            .read()
            .iter()
            .map(|exchange| crate::autonomy::TaskNativeExchange {
                exchange_id: exchange.exchange_id.clone(),
                call_ids: exchange.call_ids.clone(),
            })
            .collect()
    }

    /// Applies the configured ordinary timeout policy only after the acknowledged request deadline has expired.
    pub(crate) fn task_approval_timeout_result(&self) -> Result<ApprovalResult> {
        match self
            .hitl_engine
            .as_ref()
            .map(|engine| &engine.config().on_timeout)
        {
            Some(TimeoutAction::Approve) => Ok(ApprovalResult::Approved),
            Some(TimeoutAction::Error) => Err(AgentError::HITLTimeout),
            _ => Ok(ApprovalResult::Rejected {
                reason: Some("Timeout".into()),
            }),
        }
    }

    /// An incomplete native exchange is legal only with an exact suspended cursor, never BetweenTurns.
    pub(crate) async fn task_suspension_snapshot(&self) -> Result<TaskRuntimeCheckpoint> {
        let snapshot = self.save_state_full().await?;
        let inspection = inspect_native_history(&snapshot.memory.messages)
            .map_err(|error| AgentError::LLM(error.to_string()))?;
        Ok(TaskRuntimeCheckpoint {
            native_exchanges: inspection
                .exchanges()
                .iter()
                .map(|exchange| crate::autonomy::TaskNativeExchange {
                    exchange_id: exchange.state().exchange_id().into(),
                    call_ids: exchange.call_ids().to_vec(),
                })
                .collect(),
            snapshot,
            continuation: TaskContinuation::BetweenTurns,
        })
    }

    /// Completes cancelled native result history without polling any pending implementation or provider.
    pub(crate) async fn cancel_task_batch(
        &self,
        owner: &Arc<crate::autonomy::RunOwner>,
        mut batch: TaskBatchState,
    ) -> Result<TaskRuntimeCheckpoint> {
        let _gate = self.root_turn_gate.lock().await;
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|active| !Arc::ptr_eq(active, owner))
        {
            return Err(AgentError::Other("task cancellation owner mismatch".into()));
        }
        owner.check()?;
        let input = crate::autonomy::AutonomyTurnInput {
            owner: owner.clone(),
            objective: String::new(),
            controller_message: String::new(),
            source: crate::autonomy::AutonomyTurnSource::ResumeAfterApproval,
        };
        crate::autonomy::scope_turn(input, async {
            for result in &mut batch.results {
                if result.is_none() {
                    *result = Some(Err("task cancelled before invocation".into()));
                }
            }
            let mut calls = Vec::new();
            self.append_task_batch_prefix(&mut batch, &mut calls)
                .await?;
            self.end_root_turn();
            TaskRuntimeCheckpoint::between_turns(self.save_state_full().await?)
        })
        .await
    }

    /// Resumes one bound interaction under a fresh gate; timeout closes history without polling pending work.
    /// Persisted call IDs and answers never bypass current shared authorization.
    pub(crate) async fn resume_task_batch(
        &self,
        input: crate::autonomy::AutonomyTurnInput,
        mut batch: TaskBatchState,
        request_id: &str,
        result: crate::autonomy::suspension::BatchResponse,
    ) -> Result<AgentResponse> {
        batch.validate()?;
        let approval = batch
            .approvals
            .iter()
            .find(|approval| approval.request_id == request_id)
            .cloned()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let active = self.autonomy_owner.read().clone();
        if active
            .as_ref()
            .is_none_or(|owner| !Arc::ptr_eq(owner, &input.owner))
        {
            return Err(AgentError::Other("task resume owner mismatch".into()));
        }
        let _gate = self.root_turn_gate.clone().lock_owned().await;
        input.owner.check()?;
        let mut identities = current_runtime_gate_identity_stack().to_vec();
        if identities
            .iter()
            .any(|gate| Arc::ptr_eq(gate, &self.root_turn_gate))
        {
            return Err(AgentError::Other("reentrant task resume".into()));
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
                        .store(true, Ordering::SeqCst);
                    self.remember_active_native_exchange(&batch.content)?;
                    let mut state = batch
                        .loop_state
                        .take()
                        .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                    *self.active_native_exchanges.write() = state
                        .native_exchanges
                        .iter()
                        .map(|exchange| ActiveNativeExchange {
                            exchange_id: exchange.exchange_id.clone(),
                            call_ids: exchange.call_ids.clone(),
                        })
                        .collect();
                    self.remember_active_native_exchange(&batch.content)?;
                    self.validate_active_native_history(
                        &self.memory.get_messages(None).await?,
                        false,
                    )?;
                    let index = batch
                        .calls
                        .iter()
                        .position(|call| call.id == approval.call_id)
                        .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                    let question_timed_out = matches!(
                        result,
                        crate::autonomy::suspension::BatchResponse::QuestionTimeout
                    );
                    batch.rejected |= question_timed_out;
                    if let crate::autonomy::suspension::BatchResponse::Approval(result) = &result {
                        batch.rejected |= matches!(result, ApprovalResult::Rejected { .. });
                        batch
                            .authorized
                            .push(crate::autonomy::suspension::BatchAuthorization {
                                call_id: approval.call_id.clone(),
                                trigger: approval.trigger,
                                context: approval.context.clone(),
                                result: result.clone(),
                            });
                    }
                    let invocation = crate::autonomy::suspension::scope_resumed_approval(
                        batch
                            .authorized
                            .iter()
                            .filter(|receipt| receipt.call_id == approval.call_id)
                            .cloned()
                            .collect(),
                        crate::autonomy::scope_task_batch(
                            self.execute_tool_smart(&batch.calls[index]),
                        ),
                    );
                    let output = match result {
                        crate::autonomy::suspension::BatchResponse::QuestionTimeout => {
                            drop(invocation);
                            Err(AgentError::HITLTimeout)
                        }
                        crate::autonomy::suspension::BatchResponse::Approval(_) => invocation.await,
                        crate::autonomy::suspension::BatchResponse::UserAnswer(answer) => {
                            crate::autonomy::suspension::scope_resumed_question(
                                approval.call_id.clone(),
                                approval.context.clone(),
                                answer,
                                invocation,
                            )
                            .await
                        }
                    };
                    match output {
                        Err(AgentError::TaskSuspended(id)) => {
                            batch
                                .approvals
                                .retain(|pending| pending.call_id != approval.call_id);
                            let mut replacements = crate::autonomy::current_execution()
                                .unwrap()
                                .take_parked_approvals(&self.info.id);
                            replacements.append(&mut batch.approvals);
                            batch.approvals = replacements;
                            batch.loop_state = Some(state);
                            crate::autonomy::current_execution()
                                .unwrap()
                                .retain_batch(&self.info.id, batch);
                            return Err(AgentError::TaskSuspended(id));
                        }
                        output => {
                            if matches!(&output, Err(AgentError::HITLRejected(_))) {
                                batch.rejected = true;
                            }
                            batch.results[index] = Some(output.map_err(|error| error.to_string()))
                        }
                    }
                    batch
                        .approvals
                        .retain(|approval| approval.request_id != request_id);
                    if batch.rejected {
                        for result in &mut batch.results {
                            if result.is_none() {
                                *result = Some(Err("cancelled after approval rejection".into()));
                            }
                        }
                        let retired: Vec<_> = batch
                            .approvals
                            .drain(..)
                            .map(|pending| pending.request_id)
                            .collect();
                        crate::autonomy::current_execution()
                            .unwrap()
                            .update(|payload| {
                                for id in retired {
                                    if !payload.consumed_request_ids.contains(&id) {
                                        payload.consumed_request_ids.push(id);
                                    }
                                }
                                Ok(())
                            })
                            .await?;
                    }
                    self.append_task_batch_prefix(&mut batch, &mut state.all_tool_calls)
                        .await?;
                    if let Some(next) = batch.approvals.first() {
                        let id = next.request_id.clone();
                        batch.loop_state = Some(state);
                        crate::autonomy::current_execution()
                            .unwrap()
                            .retain_batch(&self.info.id, batch);
                        return Err(AgentError::TaskSuspended(id));
                    }
                    if batch.appended != batch.calls.len() {
                        return Err(AgentError::Config(
                            "task batch has unbound pending work".into(),
                        ));
                    }
                    // Complete native history and retire this batch before another provider call can create a new batch.
                    self.acknowledge_completed_task_batch().await?;
                    if batch.rejected {
                        crate::autonomy::current_execution()
                            .unwrap()
                            .stop(if question_timed_out {
                                "question_timeout"
                            } else {
                                "approval_rejected"
                            });
                        let response = AgentResponse {
                            content: if question_timed_out {
                                "Question expired without an answer"
                            } else {
                                "Operation cancelled by the approver"
                            }
                            .into(),
                            metadata: None,
                            tool_calls: Some(state.all_tool_calls),
                        };
                        self.memory
                            .add_message(ChatMessage::assistant(&response.content))
                            .await?;
                        self.finish_turn_if_root(&response).await?;
                        return Ok(response);
                    }
                    self.run_committed_task_loop(state).await
                }),
            ),
        )
        .await
    }

    /// A completed batch is acknowledged before model continuation so a later pause cannot replace an active cursor.
    async fn acknowledge_completed_task_batch(&self) -> Result<()> {
        let execution = crate::autonomy::current_execution()
            .ok_or_else(|| AgentError::Config("completed batch has no owner".into()))?;
        let runtime = TaskRuntimeCheckpoint::between_turns(self.save_state_full().await?)?;
        let child_operation = crate::autonomy::current_child_operation();
        execution
            .update(|payload| {
                if let Some(operation) = &child_operation {
                    let child = payload
                        .children
                        .iter_mut()
                        .find(|child| child.child_id == *operation)
                        .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                    if let Some(pending) = child.pending.take() {
                        payload.consumed_request_ids.push(pending.id);
                    }
                    child.runtime = runtime;
                } else {
                    if let Some(pending) = payload.pending.take() {
                        payload.consumed_request_ids.push(pending.id);
                    }
                    payload.runtime = runtime;
                    payload.pause_reason = None;
                }
                Ok(())
            })
            .await?;
        Ok(())
    }

    /// Continues an existing turn without resetting iterations, reasoning, accumulated calls or user commit.
    pub(super) async fn run_committed_task_loop(
        &self,
        mut state: TaskLoopState,
    ) -> Result<AgentResponse> {
        let llm = self.get_state_llm()?;
        loop {
            let max = if state.reasoning_mode != ReasoningMode::None {
                self.max_iterations
                    .min(self.get_effective_reasoning_config().max_iterations)
            } else {
                self.max_iterations
            };
            if state.iterations >= max {
                return Err(AgentError::Other(format!(
                    "Max iterations ({max}) exceeded"
                )));
            }
            state.iterations += 1;
            *self.iteration_count.write() = state.iterations;
            let protocol = self.main_tool_protocol(llm.as_ref(), false).await?;
            let mut messages = self
                .build_messages_internal(true, None, protocol.choice.is_none())
                .await?;
            self.inject_reasoning_prompt(
                &mut messages,
                &state.reasoning_mode,
                state.iterations == 1,
            );
            self.hooks.on_llm_start(&messages).await;
            let start = Instant::now();
            let response = self
                .complete_main_llm_with_recovery(llm.clone(), &messages, &protocol)
                .await?;
            self.hooks
                .on_llm_complete(&response, start.elapsed().as_millis() as u64)
                .await;
            let content = response.content.trim();
            if let Some(calls) = self.parse_main_tool_calls(content, &protocol)? {
                match self
                    .handle_tool_calls(
                        &state.processed_input,
                        content,
                        calls,
                        &mut state.all_tool_calls,
                        None,
                    )
                    .await
                {
                    Err(AgentError::TaskSuspended(id)) => {
                        self.retain_task_loop(state)?;
                        return Err(AgentError::TaskSuspended(id));
                    }
                    Err(error) => return Err(error),
                    Ok(ToolCallOutcome::Continue | ToolCallOutcome::TransitionFired) => continue,
                    Ok(ToolCallOutcome::Rejected(response)) => {
                        self.finish_turn_if_root(&response).await?;
                        return Ok(response);
                    }
                }
            }
            let (thinking, answer) = self.extract_thinking(content);
            if thinking.is_some() {
                state.thinking_content = thinking;
            }
            return self
                .finish_text_response_from_model(CommittedTextResponse {
                    processed_input: &state.processed_input,
                    input_context: &state.input_context,
                    answer,
                    reasoning_mode: state.reasoning_mode,
                    auto_detected: state.auto_detected,
                    iterations: state.iterations,
                    thinking_content: state.thinking_content,
                    all_tool_calls: state.all_tool_calls,
                })
                .await;
        }
    }
}
