//! Exact parent message reentry reauthorizes the saved invocation without repeating delivery or admission.

use super::*;
use crate::autonomy::TaskRuntimeCheckpoint;
use crate::autonomy::composition::{CompositionDispatch, DelegateFrame};
use crate::autonomy::message::{MessageInvocation, MessageState};
use serde_json::json;

impl RuntimeAgent {
    /// Only model batches can install an implementation-bound message cursor at the final invocation boundary.
    pub(super) fn prepare_message_invocation(
        &self,
        tool: Arc<dyn ai_agents_core::Tool>,
        args: &Value,
        ctx: &ToolExecutionContext,
        attempt: Option<&String>,
    ) -> Result<Option<MessageInvocation>> {
        if !tool.supports_task_messages() {
            return Ok(None);
        }
        let Some(request) = crate::autonomy::suspension::message_request() else {
            return Ok(None);
        };
        let Some(attempt) = attempt else {
            return Ok(None);
        };
        let turn = crate::autonomy::current_turn_input(&self.root_turn_gate)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let id = uuid::Uuid::new_v4().to_string();
        let state_generation = self
            .state_machine
            .as_ref()
            .map(|machine| machine.generation());
        let mut metadata = HashMap::new();
        metadata.insert("state_generation_snapshot".into(), json!(state_generation));
        metadata.insert("classification".into(), json!(ctx.classification));
        metadata.insert("effective_limits".into(), json!(ctx.limits));
        metadata.insert("policy_snapshot".into(), ctx.policy_snapshot.clone());
        let record = self.record_from_parts_at(
            &request,
            ctx.canonical_id.clone(),
            args.clone(),
            ctx.started_at,
            Instant::now(),
            true,
            false,
            String::new(),
            metadata,
            ctx.permission.clone(),
            ctx.approval.clone(),
            false,
            false,
            ToolDecisionVersions {
                policy: ctx.policy_version,
                registry: ctx.registry_version,
                runtime_control: ctx.runtime_control_version,
                state: state_generation,
            },
        );
        let state = MessageState {
            attempt: attempt.clone(),
            record,
            state_generation,
            resource_lock_keys: Vec::new(),
            max_output_chars: ctx.limits.max_output_chars,
            batch: None,
            response: Default::default(),
            child_actor: None,
            waiting_for: None,
        };
        let frame = DelegateFrame {
            version: 1,
            id: id.clone(),
            runtime_id: self.info.id.clone(),
            input: turn.objective,
            input_context: self.context_manager.get_all(),
            delegate_id: String::new(),
            delegate_runtime_id: String::new(),
            delegate_input: String::new(),
            definition: json!({"tool":ctx.canonical_id,"arguments":args,"call_id":ctx.call_id}),
            actor: self.outbound_actor_context(),
            parent_actor: current_turn_actor_context(),
            source: turn.source,
            child_operation: format!("message:{id}"),
            user_message_committed: self.root_user_message_committed.load(Ordering::SeqCst),
            native_exchanges: self.task_native_expectations(),
            dispatch: CompositionDispatch::ToolMessage,
            children: Vec::new(),
            calls: Vec::new(),
            cursor: json!(state),
            expires_at: ctx.deadline,
            parent_operation: crate::autonomy::current_child_operation(),
            required: crate::autonomy::child_required(),
        };
        Ok(Some(MessageInvocation {
            frame: Box::new(frame),
            tool,
        }))
    }

    /// Exact tool and target identities are checked before a coordinating response can claim execution.
    pub(super) fn check_message_frame(
        &self,
        frame: &DelegateFrame,
        targets: &crate::autonomy::CompositionTargets,
    ) -> Result<MessageState> {
        let state: MessageState = serde_json::from_value(frame.cursor.clone())?;
        let resolved = self
            .tools
            .resolve(&state.record.requested_name)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if frame.runtime_id != self.info.id
            || frame.children.len() != 1
            || frame.definition
                != json!({"tool":state.record.canonical_id,"arguments":state.record.executed_arguments,"call_id":state.record.call_id})
            || resolved.identity.canonical_id != state.record.canonical_id
            || !resolved.tool.supports_task_messages()
            || !targets.matches_tool(&frame.id, &resolved.tool)
        {
            return Err(AgentError::Config(
                "message tool implementation binding changed".into(),
            ));
        }
        let batch = state
            .batch
            .as_ref()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        batch.validate()?;
        if let Some(location) = &batch.location {
            self.validate_task_location(location)?;
        }
        Ok(state)
    }

    /// Resume checks current scope and policy under fresh locks but never consumes a second invocation or rate slot.
    async fn reauthorize_message(
        &self,
        frame: &DelegateFrame,
        state: &MessageState,
    ) -> Result<ToolResourceGuards> {
        let safety = self.runtime_safety_snapshot();
        let tool = self
            .tools
            .get(&state.record.canonical_id)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let arguments = &state.record.executed_arguments;
        let scope = self
            .get_available_tool_ids_with_validation_grant(
                safety.tool_scope_override.as_deref(),
                None,
            )
            .await?;
        let security = safety
            .tool_security
            .validate_tool_execution_with_bindings(
                &state.record.canonical_id,
                arguments,
                &tool.policy_bindings(),
            )
            .await?;
        let current_state = self
            .state_machine
            .as_ref()
            .map(|machine| machine.generation());
        let version_changed = safety.version != state.record.runtime_config_version
            || safety.tool_security.policy_version() != state.record.policy_version
            || self.tools.version() != state.record.registry_version
            || current_state != state.state_generation;
        if version_changed
            || safety.emergency_deny
            || !scope.tool_ids.contains(&state.record.canonical_id)
            || matches!(
                security,
                SecurityCheckResult::Block { .. } | SecurityCheckResult::Unavailable { .. }
            )
            || (matches!(security, SecurityCheckResult::RequireConfirmation { .. })
                && state.record.approval.is_none())
            || safety
                .tool_security
                .classification_approval_message(
                    &state.record.canonical_id,
                    &tool.classify_call(arguments),
                )
                .is_some()
                && state.record.approval.is_none()
            || frame
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        {
            return Err(AgentError::Tool(
                "message continuation authorization or deadline changed".into(),
            ));
        }
        let guards = self
            .acquire_tool_resource_locks(&state.resource_lock_keys)
            .await
            .ok_or_else(|| {
                AgentError::Tool("message continuation lock admission cancelled".into())
            })?;
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if self.tools.version() != state.record.registry_version
            || !execution.targets.matches_tool(&frame.id, &tool)
            || self.runtime_control.version.load(Ordering::SeqCst) != safety.version
            || self
                .runtime_safety_snapshot()
                .tool_security
                .policy_version()
                != state.record.policy_version
            || self
                .state_machine
                .as_ref()
                .map(|machine| machine.generation())
                != state.state_generation
            || self.runtime_control.emergency_deny.load(Ordering::SeqCst)
            || execution.stop_reason().is_some()
            || !execution.allows_tool(&state.record.canonical_id)
        {
            return Err(AgentError::Tool(
                "message continuation final authorization changed".into(),
            ));
        }
        Ok(guards)
    }

    /// Only the invocation owning the selected leaf response resumes; other calls retain their attempts and results.
    pub(super) async fn resume_message_batch(
        &self,
        coordinator: DelegateFrame,
    ) -> Result<AgentResponse> {
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let frames = execution.delegate_frames.lock().clone();
        let ids: Vec<String> = serde_json::from_value(coordinator.definition["messages"].clone())?;
        let frame = ids
            .iter()
            .filter_map(|id| frames.get(id))
            .find(|frame| crate::autonomy::composition::response_targets_frame(frame, &frames))
            .cloned()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        Box::pin(self.resume_message_frame(frame)).await
    }

    /// Completes the original parent result after its child settles; delivery hooks and the tool body are never replayed.
    pub(super) async fn resume_message_frame(&self, frame: DelegateFrame) -> Result<AgentResponse> {
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let mut state = self.check_message_frame(&frame, &execution.targets)?;
        let original_deadline = execution
            .approval_deadlines
            .lock()
            .get(&format!("composition:{}", frame.id))
            .copied()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let allowance = original_deadline
            .saturating_duration_since(std::time::Instant::now())
            .min(execution.remaining_duration());
        let guards = tokio::time::timeout(
            allowance,
            Box::pin(self.reauthorize_message(&frame, &state)),
        )
        .await
        .unwrap_or_else(|_| {
            Err(AgentError::Tool(
                "message continuation deadline expired during authorization".into(),
            ))
        });
        let guards = match guards {
            Ok(guards) => guards,
            Err(error) => {
                execution.stop("message_authorization_changed");
                return Err(error);
            }
        };
        let child = execution.targets.resolve(&frame.child_operation)?;
        // Resuming owned work restores the dispatched marker without reserving another slot.
        // Until the child acknowledges its next result or park, drop retains conservative effect custody.
        Box::pin(execution.update(|payload| {
            let reservation = payload
                .reservations
                .iter_mut()
                .find(|r| r.id == state.attempt)
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            if reservation.state != crate::autonomy::TaskEffectState::Suspended {
                return Err(crate::autonomy::TaskRunStorageError::Conflict.into());
            }
            reservation.state = crate::autonomy::TaskEffectState::Dispatched;
            Ok(())
        }))
        .await?;
        // Storage acknowledgement can await; recheck actual registry and control authority immediately before child polling.
        let resolved = self.tools.resolve(&state.record.requested_name);
        if self.tools.version() != state.record.registry_version
            || resolved
                .as_ref()
                .is_none_or(|resolved| !execution.targets.matches_tool(&frame.id, &resolved.tool))
            || self.runtime_control.version.load(Ordering::SeqCst)
                != state.record.runtime_config_version
            || self
                .runtime_safety_snapshot()
                .tool_security
                .policy_version()
                != state.record.policy_version
            || self
                .state_machine
                .as_ref()
                .map(|machine| machine.generation())
                != state.state_generation
            || self.runtime_control.emergency_deny.load(Ordering::SeqCst)
            || execution.stop_reason().is_some()
            || original_deadline <= std::time::Instant::now()
        {
            Box::pin(execution.suspend_message(&self.info.id, &state.attempt)).await?;
            execution.take_parked_approvals(&self.info.id);
            execution.stop("message_authorization_changed");
            return Err(AgentError::Tool("message final admission changed".into()));
        }
        guards.effect_custody.store(true, Ordering::SeqCst);
        let work = crate::autonomy::composition::scope_message_composition(
            frame.runtime_id.clone(),
            frame.id.clone(),
            crate::autonomy::scope_child_invocation(
                frame.child_operation.clone(),
                child.info.id.clone(),
                Box::pin(async {
                    if let Some(actor) = &state.child_actor {
                        child
                            .chat_with_actor_context(&frame.delegate_input, actor.clone())
                            .await
                    } else {
                        child.chat(&frame.delegate_input).await
                    }
                }),
            ),
        );
        let timer = tokio::time::sleep(
            original_deadline
                .saturating_duration_since(std::time::Instant::now())
                .min(execution.remaining_duration()),
        );
        tokio::pin!(timer);
        let mut work = Box::pin(work);
        let mut cancellation = tokio::time::interval(std::time::Duration::from_millis(10));
        let result = loop {
            tokio::select! {
                result = &mut work => break result,
                _ = &mut timer => {
                    drop(work);
                    Box::pin(execution.settle(&state.attempt, json!({"timed_out":true}), true)).await?;
                    return Err(AgentError::Other("message continuation timed out with unacknowledged child work".into()));
                }
                _ = cancellation.tick() => {
                    // Non-cancellation control stops still drain the child's acknowledgement and history cleanup.
                    if self.runtime_control.emergency_deny.load(Ordering::SeqCst) || execution.cancellation.load(Ordering::Acquire) {
                        drop(work);
                        Box::pin(execution.settle(&state.attempt, json!({"cancelled":true}), true)).await?;
                        return Err(AgentError::Other("message continuation cancelled with unacknowledged child work".into()));
                    }
                }
            }
        };
        drop(work);
        if matches!(result, Err(AgentError::TaskSuspended(_))) {
            Box::pin(execution.suspend_message(&self.info.id, &state.attempt)).await?;
            execution.take_parked_approvals(&self.info.id);
            guards.effect_custody.store(false, Ordering::SeqCst);
            drop(guards);
            let batch = state
                .batch
                .take()
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            Box::pin(execution.retain_message_batch(&self.info.id, batch.clone())).await?;
            execution.retain_batch(&self.info.id, batch);
            return Err(AgentError::TaskSuspended(frame.id));
        }
        // An intermediate parent's own lease is still active here; only this dispatch's child outcome establishes quiescence.
        let acknowledged: crate::autonomy::TaskCheckpointPayload =
            serde_json::from_value(execution.acknowledged_snapshot().payload)?;
        if acknowledged
            .children
            .iter()
            .find(|child| child.child_id == frame.child_operation)
            .is_none_or(|child| child.result.is_none() || child.pending.is_some())
        {
            return Err(AgentError::Other("message child requires recovery".into()));
        }
        let tool_result = match result {
            Ok(response) => ToolResult::ok(match &state.response {
                crate::autonomy::message::MessageResponse::Message => json!({"from":frame.delegate_id,"response":response.content}),
                crate::autonomy::message::MessageResponse::Route { reason } => json!({"selected_agent":frame.delegate_id,"response":response.content,"reason":reason}),
            }.to_string()),
            Err(error) => ToolResult::error(match state.response {
                crate::autonomy::message::MessageResponse::Message => format!("send failed: {error}"),
                crate::autonomy::message::MessageResponse::Route { .. } => format!("routing failed: {error}"),
            }),
        };
        Box::pin(execution.settle(&state.attempt, serde_json::to_value(&tool_result)?, false))
            .await?;
        guards.effect_custody.store(false, Ordering::SeqCst);
        let (output, truncated) =
            Self::truncate_tool_output(tool_result.output, state.max_output_chars);
        state.record.output = output;
        state.record.output_truncated = truncated;
        state.record.success = tool_result.success;
        state.record.duration_ms = u64::try_from(
            (chrono::Utc::now() - state.record.started_at)
                .num_milliseconds()
                .max(0),
        )
        .unwrap_or(u64::MAX);
        self.finish_tool_record_after_resource_guards(guards, &state.record)
            .await;
        let mut batch = state
            .batch
            .take()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if let Some(location) = batch.location.take() {
            execution.retire_completed_message(&self.info.id, &state.attempt);
            self.acknowledge_completed_task_batch().await?;
            execution.parked_batches.lock().remove(&self.info.id);
            execution.delegate_frames.lock().remove(&self.info.id);
            return Box::pin(self.finish_task_location_record(location, state.record)).await;
        }
        let mut loop_state = batch
            .loop_state
            .take()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let index = batch
            .calls
            .iter()
            .position(|call| call.id == state.record.call_id)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        batch
            .executor_cursors
            .retain(|cursor| cursor.request.call_id != state.record.call_id);
        batch.results[index] = Some(if state.record.success {
            Ok(state.record.model_output_string())
        } else {
            Err(state.record.model_output_string())
        });
        batch
            .approvals
            .retain(|approval| approval.request_id != frame.id);
        self.append_task_batch_prefix(&mut batch, &mut loop_state.all_tool_calls)
            .await?;
        execution.retire_completed_message(&self.info.id, &state.attempt);
        if let Some(next) = batch.approvals.first() {
            let id = next.request_id.clone();
            batch.loop_state = Some(loop_state);
            Box::pin(execution.retain_message_batch(&self.info.id, batch.clone())).await?;
            let order: Vec<_> = batch.calls.iter().map(|call| call.id.clone()).collect();
            execution.retain_batch(&self.info.id, batch);
            // A repeated-target call already owns its parent attempt but has not started another child history.
            let frames = execution.delegate_frames.lock().clone();
            let snapshot = execution.acknowledged_snapshot();
            let deferred =
                frames
                    .values()
                    .filter(|frame| {
                        frame.runtime_id == self.info.id
                            && frame.dispatch == CompositionDispatch::ToolMessage
                            && frame.cursor["waiting_for"]
                                .as_str()
                                .is_some_and(|dependency| {
                                    snapshot.payload["children"].as_array().is_some_and(
                                        |children| {
                                            !children.iter().any(|child| {
                                                child["child_id"] == frame.child_operation
                                            }) && children.iter().all(|child| {
                                                child["runtime"]["snapshot"]["agent_id"]
                                                    != frame.delegate_runtime_id
                                                    || !child["result"].is_null()
                                            }) && children.iter().any(|child| {
                                                child["child_id"] == dependency
                                                    && !child["result"].is_null()
                                            })
                                        },
                                    )
                                })
                    })
                    .min_by_key(|frame| {
                        order
                            .iter()
                            .position(|id| frame.cursor["record"]["call_id"] == *id)
                    })
                    .cloned();
            if let Some(deferred) = deferred {
                return Box::pin(self.resume_message_frame(deferred)).await;
            }
            return Err(AgentError::TaskSuspended(id));
        }
        self.acknowledge_completed_task_batch().await?;
        execution.parked_batches.lock().remove(&self.info.id);
        execution.delegate_frames.lock().remove(&self.info.id);
        if execution.stop_reason().is_some() {
            return Ok(AgentResponse {
                content: "Message child stopped".into(),
                metadata: None,
                tool_calls: Some(loop_state.all_tool_calls),
            });
        }
        self.run_committed_task_loop(loop_state).await
    }

    /// Cancellation closes each runtime's native batch once while publishing every suspended invocation's own record.
    /// Completed out-of-order results stay intact; no sender, selector, child tool or provider is polled.
    pub(super) async fn close_message_batch(
        &self,
        owner: &Arc<crate::autonomy::RunOwner>,
        coordinator: &DelegateFrame,
        frames: &std::collections::BTreeMap<String, DelegateFrame>,
    ) -> Result<TaskRuntimeCheckpoint> {
        let execution = crate::autonomy::current_execution();
        let mut batch: crate::autonomy::TaskBatchState = execution
            .as_ref()
            .and_then(|execution| execution.parked_batches.lock().get(&self.info.id).cloned())
            .map(Ok)
            .unwrap_or_else(|| serde_json::from_value(coordinator.cursor.clone()))?;
        let mut records = Vec::new();
        let ids: Vec<String> = serde_json::from_value(coordinator.definition["messages"].clone())?;
        for id in ids {
            let frame = frames
                .get(&id)
                .cloned()
                .or_else(|| {
                    execution.as_ref().and_then(|execution| {
                        execution.acknowledged_snapshot().payload["adapters"]
                            .as_array()
                            .and_then(|adapters| {
                                adapters.iter().find(|adapter| {
                                    adapter["id"] == format!("runtime.delegate:{id}")
                                })
                            })
                            .and_then(|adapter| {
                                serde_json::from_value(adapter["state"].clone()).ok()
                            })
                    })
                })
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            let mut state: MessageState = serde_json::from_value(frame.cursor.clone())?;
            if execution.as_ref().is_some_and(|execution| {
                execution.acknowledged_snapshot().payload["reservations"]
                    .as_array()
                    .is_some_and(|reservations| {
                        reservations.iter().any(|reservation| {
                            reservation["id"] == state.attempt
                                && reservation["state"] == "completed"
                        })
                    })
            }) {
                continue;
            }
            let index = batch
                .calls
                .iter()
                .position(|call| call.id == state.record.call_id)
                .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
            if batch.results[index].is_none() {
                batch.results[index] = Some(Err("message stopped after child suspension".into()));
            }
            state.record.output = "message stopped after child suspension".into();
            state.record.success = false;
            state.record.cancelled = true;
            state.record.cancellation_reason = Some("task stopped".into());
            state.record.duration_ms = u64::try_from(
                (chrono::Utc::now() - state.record.started_at)
                    .num_milliseconds()
                    .max(0),
            )
            .unwrap_or(u64::MAX);
            records.push(state.record);
        }
        let runtime = self.cancel_task_batch(owner, batch).await?;
        for record in records {
            // The cancellation checkpoint owns attribution; hooks do not append a second root-attributed record.
            self.publish_tool_record(&record).await;
        }
        Ok(runtime)
    }
}
