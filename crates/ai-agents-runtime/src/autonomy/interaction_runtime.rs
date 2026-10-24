//! Exact non-tool runtime interactions reuse the admitted root and existing semantic/state drivers.

use super::*;
use crate::autonomy::TaskBatchState;
use crate::autonomy::interaction::{self, TaskInteraction, TaskInteractionKind, TaskModelCalls};
use crate::autonomy::suspension::{BatchApproval, BatchResponse};
use ai_agents_hitl::ApprovalRequest;

impl RuntimeAgent {
    /// A framework-selected interaction retains its return position before any host response or transition effect.
    /// Clarification questions use the configured HITL deadline so timeout settlement can retire the exact exchange without semantic replay.
    pub(super) async fn park_task_interaction(
        &self,
        kind: TaskInteractionKind,
        approval: Option<ApprovalRequest>,
        question: Option<Value>,
    ) -> Result<bool> {
        let Some(turn) = crate::autonomy::current_turn_input(&self.root_turn_gate) else {
            return Ok(false);
        };
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let enabled = if matches!(kind, TaskInteractionKind::PlanReview { .. }) {
            execution
                .task_interaction_policy()
                .or(Some(crate::autonomy::InteractionAction::PauseRun))
        } else if question.is_some() {
            execution
                .lifecycle_profile()
                .hitl
                .and_then(|h| h.on_user_question)
        } else {
            execution.task_interaction_policy()
        };
        // A restrictive task profile stops an unresolved semantic exchange instead of silently invoking a host handler.
        if matches!(
            enabled,
            Some(
                crate::autonomy::InteractionAction::Reject
                    | crate::autonomy::InteractionAction::AutoIfPolicy
            )
        ) {
            execution.stop("interaction_rejected");
            return Err(AgentError::Config(
                "task profile rejects required interaction".into(),
            ));
        }
        if !matches!(
            enabled,
            Some(
                crate::autonomy::InteractionAction::PauseRun
                    | crate::autonomy::InteractionAction::AskEachTime
            )
        ) {
            return Ok(false);
        }
        let id = uuid::Uuid::new_v4().to_string();
        let (request_id, trigger, context, message, timeout) = if let Some(approval) = approval {
            (
                approval.id,
                serde_json::to_value(approval.trigger)?,
                serde_json::to_value(approval.context)?,
                approval.message,
                approval.timeout,
            )
        } else {
            (
                uuid::Uuid::new_v4().to_string(),
                Value::Null,
                Value::Null,
                question
                    .as_ref()
                    .and_then(|q| q.get("question"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
                self.hitl_engine.as_ref().map(|hitl| {
                    std::time::Duration::from_secs(hitl.config().default_timeout_seconds)
                }),
            )
        };
        let expires_at = if let Some(timeout) = timeout {
            let expiry = chrono::Utc::now()
                .checked_add_signed(
                    chrono::Duration::from_std(timeout)
                        .map_err(|_| AgentError::Config("interaction deadline overflow".into()))?,
                )
                .ok_or_else(|| AgentError::Config("interaction deadline overflow".into()))?;
            execution.approval_deadlines.lock().insert(
                request_id.clone(),
                Instant::now()
                    .checked_add(timeout)
                    .ok_or_else(|| AgentError::Config("interaction deadline overflow".into()))?,
            );
            Some(expiry)
        } else {
            None
        };
        let state = TaskInteraction {
            id: id.clone(),
            kind,
            state_generation: self.state_machine.as_ref().map(|m| m.generation()),
            epoch: self.disambiguation_epoch.load(Ordering::SeqCst),
            runtime_control_version: self.runtime_control.version.load(Ordering::SeqCst),
            source: turn.source,
            user_message_committed: self.root_user_message_committed.load(Ordering::SeqCst),
            native_exchanges: self.task_native_expectations(),
            request_id: Some(request_id.clone()),
            expires_at,
        };
        let batch = TaskBatchState {
            version: 1,
            content: String::new(),
            calls: Vec::new(),
            results: Vec::new(),
            appended: 0,
            approvals: vec![BatchApproval {
                call_id: id,
                request_id: request_id.clone(),
                trigger,
                context,
                message,
                timeout_millis: timeout
                    .map(|d| u64::try_from(d.as_millis()))
                    .transpose()
                    .map_err(|_| AgentError::Config("interaction timeout overflow".into()))?,
                expires_at,
                question,
            }],
            loop_state: None,
            authorized: Vec::new(),
            rejected: false,
            executor_cursors: Vec::new(),
            location: None,
            interaction: Some(Box::new(state)),
        };
        batch.validate()?;
        crate::autonomy::composition::close_composition_admission();
        execution.retain_batch(&self.info.id, batch);
        Err(AgentError::TaskSuspended(request_id))
    }

    /// Existing manager progress is captured after semantic work, without committing another root question response.
    pub(super) async fn park_task_clarification(
        &self,
        input: &str,
        response: &AgentResponse,
    ) -> Result<bool> {
        if crate::autonomy::current_turn_input(&self.root_turn_gate).is_none() {
            return Ok(false);
        }
        let manager = self
            .disambiguation_manager
            .as_ref()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let checkpoint = manager
            .pending_checkpoint()
            .await?
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if !matches!(
            execution
                .lifecycle_profile()
                .hitl
                .and_then(|policy| policy.on_user_question),
            Some(
                crate::autonomy::InteractionAction::PauseRun
                    | crate::autonomy::InteractionAction::AskEachTime
            )
        ) {
            manager.clear_pending_checkpoint(&checkpoint).await?;
            *self.pending_skill_id.write() = None;
            execution.stop("interaction_unavailable");
            return Ok(false);
        }
        let skill_id = self.pending_skill_id.read().clone();
        self.park_task_interaction(TaskInteractionKind::Clarification { manager: checkpoint, input: input.into(), prepared: interaction::prepared_input(), skill_id },
            None, Some(serde_json::json!({"kind":"clarification","question":response.content,"metadata":response.metadata}))).await
    }

    /// Binding checks happen before response claim and again under the root gate; no serialized exchange can install manager state.
    pub(crate) fn validate_task_interaction(&self, state: &TaskInteraction) -> Result<()> {
        if self.state_machine.as_ref().map(|m| m.generation()) != state.state_generation
            || self.disambiguation_epoch.load(Ordering::SeqCst) != state.epoch
        {
            return Err(AgentError::Config(
                "interaction state or reset ownership changed".into(),
            ));
        }
        match &state.kind {
            TaskInteractionKind::PlanReview {
                reasoning_config, ..
            } => {
                if *reasoning_config != serde_json::to_value(self.get_effective_reasoning_config())?
                {
                    return Err(AgentError::Config(
                        "reviewed plan reasoning binding changed".into(),
                    ));
                }
            }
            TaskInteractionKind::StateApproval {
                from_state,
                definition,
                ..
            } => {
                let machine = self
                    .state_machine
                    .as_ref()
                    .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                if machine.current() != *from_state
                    || serde_json::to_value(machine.current_definition())? != *definition
                {
                    return Err(AgentError::Config(
                        "transition review binding changed".into(),
                    ));
                }
            }
            TaskInteractionKind::Clarification {
                manager, skill_id, ..
            } => {
                if self.pending_skill_id.read().as_ref() != skill_id.as_ref()
                    || !self
                        .disambiguation_manager
                        .as_ref()
                        .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?
                        .matches_pending_checkpoint(manager)?
                {
                    return Err(AgentError::Config("clarification exchange changed".into()));
                }
                if let Some(skill) = skill_id
                    && !self.get_available_skills().iter().any(|s| &s.id == skill)
                {
                    return Err(AgentError::Config(
                        "clarification skill scope changed".into(),
                    ));
                }
            }
            TaskInteractionKind::Escalation { skill_id, .. } => {
                if let Some(skill) = skill_id
                    && !self.get_available_skills().iter().any(|s| &s.id == skill)
                {
                    return Err(AgentError::Config("escalation skill scope changed".into()));
                }
            }
        }
        Ok(())
    }

    /// Approval changes cannot silently retarget an already-reviewed transition or escalation.
    pub(crate) fn validate_task_interaction_response(
        &self,
        state: &TaskInteraction,
        response: &BatchResponse,
    ) -> Result<()> {
        self.validate_task_interaction(state)?;
        if self.runtime_control.emergency_deny.load(Ordering::SeqCst)
            || self.runtime_control.version.load(Ordering::SeqCst) != state.runtime_control_version
        {
            return Err(AgentError::Config(
                "interaction runtime control authority changed".into(),
            ));
        }
        if let BatchResponse::Approval(result) = response {
            crate::autonomy::bounded_value(&serde_json::to_value(result)?, 65_536)?;
        }
        match response {
            BatchResponse::UserAnswer(answer) => state.validate_answer(answer),
            BatchResponse::QuestionTimeout
                if matches!(state.kind, TaskInteractionKind::Clarification { .. }) =>
            {
                Ok(())
            }
            BatchResponse::Approval(ApprovalResult::Modified { changes })
                if !changes.is_empty() =>
            {
                Err(AgentError::Config(
                    "non-tool approval does not grant modified transition or escalation authority"
                        .into(),
                ))
            }
            BatchResponse::Approval(_)
                if !matches!(state.kind, TaskInteractionKind::Clarification { .. }) =>
            {
                Ok(())
            }
            _ => Err(crate::autonomy::TaskRunStorageError::NotResumable.into()),
        }
    }

    /// A frozen executable plan keeps its review binding and both original clocks through the final adoption write.
    pub(super) fn validate_task_plan_review(&self, state: &TaskInteraction) -> Result<()> {
        self.validate_task_interaction_response(
            state,
            &BatchResponse::Approval(ApprovalResult::Approved),
        )?;
        let id = state
            .request_id
            .as_ref()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let expired = state
            .expires_at
            .is_some_and(|expiry| chrono::Utc::now() >= expiry)
            || crate::autonomy::current_execution().is_some_and(|execution| {
                execution
                    .approval_deadlines
                    .lock()
                    .get(id)
                    .is_some_and(|expiry| Instant::now() >= *expiry)
            });
        if expired && !crate::autonomy::suspension::timeout_approved(id) {
            return Err(AgentError::HITLTimeout);
        }
        Ok(())
    }

    /// A known-safe expired decision finishes without applying its proposal or making another provider/tool call.
    pub(super) async fn finish_expired_task_response(&self) -> Result<AgentResponse> {
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        execution.stop("response_timeout");
        let response = AgentResponse::new("Pending response expired before application");
        self.finish_turn_if_root(&response).await?;
        Ok(response)
    }

    /// Common response validation dispatches by actual location, never by a fabricated tool name.
    pub(crate) fn validate_task_batch_response(
        &self,
        batch: &TaskBatchState,
        response: &BatchResponse,
    ) -> Result<()> {
        if matches!(response, BatchResponse::Capacity)
            && batch
                .approvals
                .first()
                .is_none_or(|pending| pending.capacity().ok().flatten().is_none())
        {
            return Err(crate::autonomy::TaskRunStorageError::NotResumable.into());
        }
        if let Some(interaction) = &batch.interaction {
            return self.validate_task_interaction_response(interaction, response);
        }
        if let Some(location) = &batch.location {
            self.validate_task_location(location)?;
        }
        if let BatchResponse::UserAnswer(answer) = response {
            let pending = batch
                .approvals
                .first()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            let call = batch
                .calls
                .iter()
                .find(|call| call.id == pending.call_id)
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            let mut call = batch.pending_call(call);
            call.arguments = pending.context.clone();
            self.validate_task_question_answer(&call, &pending.question, answer)?;
        }
        Ok(())
    }

    /// Resumes the admitted non-tool location, retaining selected skill/input context through escalation without repeating routing or processing.
    /// Future erasure keeps optional semantic/transition drivers out of nested model polling frames.
    pub(super) fn resume_task_interaction_batch(
        &self,
        mut input: crate::autonomy::AutonomyTurnInput,
        batch: TaskBatchState,
        response: BatchResponse,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + '_>> {
        Box::pin(async move {
            let state = batch
                .interaction
                .as_ref()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?
                .clone();
            self.validate_task_interaction_response(&state, &response)?;
            let originally_expired = self.task_response_expired(&batch);
            input.source = state.source;
            let _gate = self.root_turn_gate.clone().lock_owned().await;
            if self
                .autonomy_owner
                .read()
                .as_ref()
                .is_none_or(|owner| !Arc::ptr_eq(owner, &input.owner))
            {
                return Err(AgentError::Other("interaction resume owner changed".into()));
            }
            input.owner.check()?;
            self.validate_task_interaction_response(&state, &response)?;
            let mut ancestry = current_runtime_gate_identity_stack().to_vec();
            if ancestry
                .iter()
                .any(|gate| Arc::ptr_eq(gate, &self.root_turn_gate))
            {
                return Err(AgentError::Other("reentrant interaction resume".into()));
            }
            ancestry.push(self.root_turn_gate.clone());
            let ancestry: RootTurnGateIdentityStack = ancestry.into();
            scope_runtime_gate_identity_stack(
                &ancestry,
                crate::autonomy::scope_turn(
                    input,
                    Box::pin(async {
                        self.begin_root_turn();
                        let _cleanup = RootTurnCleanup::new(self);
                        self.root_user_message_committed
                            .store(state.user_message_committed, Ordering::SeqCst);
                        if let TaskInteractionKind::Clarification {
                            prepared: Some(data),
                            ..
                        }
                        | TaskInteractionKind::Escalation {
                            prepared: Some(data),
                            ..
                        } = &state.kind
                        {
                            self.update_active_turn_context(&data.content, data.context.clone());
                        }
                        if let TaskInteractionKind::StateApproval { return_to, .. } = &state.kind
                            && let crate::autonomy::location::TaskStateReturn::SpeculativeSelection { selection } = return_to.as_ref()
                        {
                            self.restore_speculative_context(selection);
                        }
                        *self.active_native_exchanges.write() = state
                            .native_exchanges
                            .iter()
                            .map(|e| ActiveNativeExchange {
                                exchange_id: e.exchange_id.clone(),
                                call_ids: e.call_ids.clone(),
                            })
                            .collect();
                        self.validate_active_native_history(
                            &self.memory.get_messages(None).await?,
                            false,
                        )?;
                        self.acknowledge_completed_task_batch().await?;
                        let execution = crate::autonomy::current_execution()
                            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                        execution.parked_batches.lock().remove(&self.info.id);
                        self.validate_task_interaction_response(&state, &response)?;
                        // Consuming the request is not permission to apply a proposal after its original deadline.
                        if !originally_expired && self.task_response_expired(&batch) {
                            self.clear_task_interaction(&state).await?;
                            execution.stop("response_timeout");
                            let response =
                                AgentResponse::new("Pending response expired before application");
                            self.finish_turn_if_root(&response).await?;
                            return Ok(response);
                        }
                        match (&state.kind, response) {
                            (
                                TaskInteractionKind::PlanReview {
                                    plan,
                                    attempt,
                                    processed_input,
                                    input_context,
                                    auto_detected,
                                    ..
                                },
                                BatchResponse::Approval(result),
                            ) => {
                                if matches!(
                                    result,
                                    ApprovalResult::Approved | ApprovalResult::Modified { .. }
                                ) {
                                    Box::pin(self.continue_plan_and_execute(
                                        processed_input,
                                        input_context,
                                        *auto_detected,
                                        plan.clone(),
                                        *attempt,
                                        Some((*state).clone()),
                                    ))
                                    .await
                                } else {
                                    execution.stop("supervision_rejected");
                                    let response = AgentResponse::new("Plan review declined");
                                    self.finish_turn_if_root(&response).await?;
                                    Ok(response)
                                }
                            }
                            (
                                TaskInteractionKind::Clarification { .. },
                                BatchResponse::QuestionTimeout,
                            ) => {
                                self.clear_task_interaction(&state).await?;
                                execution.stop("question_timeout");
                                let response =
                                    AgentResponse::new("Clarification expired without an answer");
                                self.finish_turn_if_root(&response).await?;
                                Ok(response)
                            }
                            (
                                TaskInteractionKind::Clarification { prepared, .. },
                                BatchResponse::UserAnswer(answer),
                            ) => {
                                let answer = answer
                                    .as_str()
                                    .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)?;
                                let run = Box::pin(async {
                                    match self.resolve_disambiguation(answer).await? {
                                        DisambiguationDispatch::Terminal(response) => Ok(response),
                                        DisambiguationDispatch::Proceed(input) => {
                                            Box::pin(self.run_loop_internal(&input)).await
                                        }
                                        DisambiguationDispatch::RecheckSkill {
                                            skill_id,
                                            enriched_input,
                                            disambiguation_epoch,
                                            state_generation,
                                        } => {
                                            Box::pin(self.recheck_skill_disambiguation(
                                                &skill_id,
                                                &enriched_input,
                                                disambiguation_epoch,
                                                state_generation,
                                            ))
                                            .await
                                        }
                                    }
                                });
                                let run = interaction::scope_clarification_return(
                                    state.kind.clone(),
                                    run,
                                );
                                if let Some(data) = prepared {
                                    interaction::scope_prepared_input(data.clone(), run).await
                                } else {
                                    run.await
                                }
                            }
                            (
                                TaskInteractionKind::StateApproval {
                                    from_state,
                                    target,
                                    reason,
                                    staged,
                                    processed_input,
                                    return_to,
                                    ..
                                },
                                BatchResponse::Approval(result),
                            ) => {
                                let approved = matches!(
                                    result,
                                    ApprovalResult::Approved | ApprovalResult::Modified { .. }
                                );
                                let transitioned = if approved {
                                    if let Some(input) = processed_input {
                                        self.commit_root_user_message(input).await?;
                                    }
                                    crate::autonomy::location::scope_state_return(
                                        (**return_to).clone(),
                                        Box::pin(self.apply_transition_target(
                                            from_state,
                                            target,
                                            reason,
                                            staged.as_ref(),
                                        )),
                                    )
                                    .await?
                                } else {
                                    false
                                };
                                Box::pin(self.resume_state_return_outcome(
                                    (**return_to).clone(),
                                    transitioned,
                                ))
                                .await
                            }
                            (
                                TaskInteractionKind::Escalation {
                                    input,
                                    reason,
                                    prepared,
                                    skill_id,
                                },
                                BatchResponse::Approval(result),
                            ) => {
                                if matches!(
                                    result,
                                    ApprovalResult::Approved | ApprovalResult::Modified { .. }
                                ) {
                                    if let Some(skill) = skill_id {
                                        // Human approval keeps the selected skill; restarting routing would repeat completed semantic work.
                                        let content =
                                            Box::pin(self.execute_skill_by_id(skill, input))
                                                .await?;
                                        self.commit_rechecked_skill_input(input).await?;
                                        let context = prepared
                                            .as_ref()
                                            .map(|data| data.context.clone())
                                            .unwrap_or_default();
                                        Box::pin(
                                            self.handle_skill_response(
                                                input, skill, content, &context,
                                            ),
                                        )
                                        .await
                                    } else if let Some(data) = prepared {
                                        Box::pin(self.run_accepted_input(data.clone())).await
                                    } else {
                                        Box::pin(self.run_loop_internal(input)).await
                                    }
                                } else {
                                    execution.stop("approval_rejected");
                                    let response = AgentResponse::new(format!(
                                        "Human escalation was declined: {reason}"
                                    ));
                                    self.finish_turn_if_root(&response).await?;
                                    Ok(response)
                                }
                            }
                            _ => Err(crate::autonomy::TaskRunStorageError::NotResumable.into()),
                        }
                    }),
                ),
            )
            .await
        })
    }

    /// Cancellation retires deferred branch observations and clears only the exact manager exchange, without semantic or transition work.
    pub(super) async fn clear_task_interaction(&self, state: &TaskInteraction) -> Result<()> {
        if let TaskInteractionKind::StateApproval { return_to, .. } = &state.kind {
            self.cancel_speculative_return(&mut return_to.as_ref().clone());
        }
        if let TaskInteractionKind::Clarification {
            manager, skill_id, ..
        } = &state.kind
        {
            let actual = self
                .disambiguation_manager
                .as_ref()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            if actual.matches_pending_checkpoint(manager)? {
                actual.clear_pending_checkpoint(manager).await?;
                if self.pending_skill_id.read().as_ref() == skill_id.as_ref() {
                    *self.pending_skill_id.write() = None;
                }
            }
        }
        Ok(())
    }

    /// Completed composition output retains metadata and finalization ownership while transition approval or actions are unfinished.
    pub(super) async fn post_composed_processing(
        &self,
        input: &str,
        content: String,
    ) -> Result<PostLoopResult> {
        if crate::autonomy::current_turn_input(&self.root_turn_gate).is_none() {
            return self.post_loop_processing(input, content).await;
        }
        let mut metadata = HashMap::new();
        if let Some(orchestration) = self.context_manager.get("orchestration") {
            metadata.insert("orchestration".into(), orchestration);
        }
        let input_context = self
            .active_turn_context
            .read()
            .as_ref()
            .map(|turn| turn.input_context.clone())
            .unwrap_or_default();
        let state = crate::autonomy::location::TaskModelFinal {
            processed_input: input.into(),
            input_context,
            content: content.clone(),
            all_tool_calls: Vec::new(),
            reasoning_mode: ReasoningMode::None,
            auto_detected: false,
            iterations: 0,
            thinking: None,
            reflection_metadata: None,
            finalize_on_resume: false,
            response_metadata: Some(metadata),
        };
        crate::autonomy::location::scope_state_return(
            crate::autonomy::location::TaskStateReturn::ModelFinish {
                state: Box::new(state),
            },
            Box::pin(self.post_loop_processing(input, content)),
        )
        .await
    }

    /// Task skill rechecks retain the original root user commit; ordinary separate chat turns retain their existing user record.
    pub(super) async fn commit_rechecked_skill_input(&self, input: &str) -> Result<()> {
        if crate::autonomy::current_turn_input(&self.root_turn_gate).is_some() {
            self.commit_root_user_message(input).await
        } else {
            self.memory.add_message(ChatMessage::user(input)).await
        }
    }

    /// Rejected pre-tool selection executes the already-produced native/prompt calls once, skipping only the completed selection.
    pub(super) async fn resume_rejected_tool_transition(
        &self,
        saved: TaskModelCalls,
    ) -> Result<AgentResponse> {
        let mut state = *saved.state;
        match interaction::scope_skipped_tool_transition(Box::pin(self.handle_tool_calls(
            &state.processed_input,
            &saved.content,
            saved.calls,
            &mut state.all_tool_calls,
            None,
        )))
        .await
        {
            Err(AgentError::TaskSuspended(id)) => {
                self.retain_task_loop(state).await?;
                Err(AgentError::TaskSuspended(id))
            }
            Err(error) => Err(error),
            Ok(ToolCallOutcome::Rejected(response)) => {
                self.finish_turn_if_root(&response).await?;
                Ok(response)
            }
            Ok(ToolCallOutcome::Continue | ToolCallOutcome::TransitionFired) => {
                Box::pin(self.run_committed_task_loop(state)).await
            }
        }
    }
}
