//! Parent dispatch capture and retirement preserve prepared input and immutable child bindings across safe pauses.

use super::*;
use crate::autonomy::composition::{CompositionChild, CompositionDispatch, DelegateFrame};

impl RuntimeAgent {
    /// A resumed dispatch uses the private retained frame rather than preparing input or allocating child operations again.
    pub(super) fn retained_composition_frame(&self) -> Result<Option<DelegateFrame>> {
        if !crate::autonomy::composition::resuming_delegate(&self.info.id) {
            return Ok(None);
        }
        let execution = crate::autonomy::current_execution()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let frame = execution
            .delegate_frames
            .lock()
            .get(&self.info.id)
            .cloned()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        Ok(Some(frame))
    }

    /// Captures state dispatch after input preparation and before any child can act; saved slots bind registry targets by identity.
    pub(super) async fn prepare_composition_frame(
        &self,
        input: &str,
        prepared: &str,
        dispatch: CompositionDispatch,
        definition: Value,
        registry_ids: &[String],
        cursor: Value,
    ) -> Result<Option<DelegateFrame>> {
        let Some(execution) = crate::autonomy::current_execution() else {
            return Ok(None);
        };
        if let Some(frame) = self.retained_composition_frame()? {
            if frame.input != input
                || frame.dispatch != dispatch
                || frame.definition != definition
                || frame
                    .children
                    .iter()
                    .map(|slot| &slot.registry_id)
                    .ne(registry_ids.iter())
            {
                return Err(AgentError::Config(
                    "composition dispatch binding changed".into(),
                ));
            }
            return Ok(Some(frame));
        }
        let registry = self
            .spawner_registry
            .as_ref()
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let turn = crate::autonomy::current_turn_input(&self.root_turn_gate)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let id = uuid::Uuid::new_v4().to_string();
        let expires_at = definition
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .map(|millis| {
                let deadline = std::time::Instant::now()
                    .checked_add(std::time::Duration::from_millis(millis))
                    .ok_or_else(|| {
                        AgentError::Config("composition monotonic deadline overflow".into())
                    })?;
                let millis = i64::try_from(millis)
                    .map_err(|_| AgentError::Config("composition deadline overflow".into()))?;
                let expiry = chrono::Utc::now()
                    .checked_add_signed(chrono::Duration::milliseconds(millis))
                    .ok_or_else(|| AgentError::Config("composition deadline overflow".into()))?;
                execution
                    .approval_deadlines
                    .lock()
                    .insert(format!("composition:{id}"), deadline);
                Ok::<_, AgentError>(expiry)
            })
            .transpose()?;
        let children = registry_ids
            .iter()
            .enumerate()
            .map(|(index, registry_id)| {
                let child = registry.get(registry_id).ok_or_else(|| {
                    AgentError::Config(format!("composition child not registered: {registry_id}"))
                })?;
                child.preflight_standalone_autonomy()?;
                child.preflight_priced_autonomy(
                    ai_agents_core::autonomy::InvocationAdmission::requires_priced_llm(
                        execution.as_ref(),
                    ),
                )?;
                Ok(CompositionChild {
                    registry_id: registry_id.clone(),
                    runtime_id: child.info.id.clone(),
                    operation: format!("composition:{id}:{index}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Box::pin(self.commit_root_user_message(input)).await?;

        let frame = DelegateFrame {
            version: 1,
            id: id.clone(),
            runtime_id: self.info.id.clone(),
            input: input.into(),
            input_context: self.context_manager.get_all(),
            delegate_id: String::new(),
            delegate_runtime_id: String::new(),
            delegate_input: prepared.into(),
            definition,
            actor: self.outbound_actor_context(),
            parent_actor: current_turn_actor_context(),
            source: turn.source,
            child_operation: format!("composition:{id}"),
            user_message_committed: self.root_user_message_committed.load(Ordering::SeqCst),
            native_exchanges: self.task_native_expectations(),
            dispatch,
            children,
            calls: Vec::new(),
            cursor,
            expires_at,
            parent_operation: crate::autonomy::current_child_operation(),
            required: crate::autonomy::child_required(),
        };
        Box::pin(execution.retain_delegate_frame(frame.clone())).await?;

        Ok(Some(frame))
    }

    /// Resumes an intermediate participant from its saved parent location; it never restarts the child input pipeline.
    pub(super) async fn resume_parked_task_composition(
        &self,
        input: &str,
        actor: Option<crate::TurnActorContext>,
        execution: Arc<crate::autonomy::RunExecution>,
        operation: String,
        frame: DelegateFrame,
    ) -> Result<AgentResponse> {
        let owner = execution
            .participants
            .owner_for(&self.root_turn_gate)
            .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        owner.check()?;
        let actor_bound = Box::pin(execution.verify_child_actor(&operation, &actor)).await?;
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|current| !Arc::ptr_eq(current, &owner))
            || (!actor_bound
                && actor
                    .as_ref()
                    .and_then(|actor| actor.effective_actor_id())
                    .is_some_and(|actor| owner.actor_id.as_deref() != Some(actor)))
        {
            return Err(AgentError::Other(
                "nested resume owner binding changed".into(),
            ));
        }
        let guard = tokio::time::timeout(
            execution.remaining_duration(),
            self.root_turn_gate.clone().lock_owned(),
        )
        .await
        .map_err(|_| AgentError::Other("nested resume gate deadline exceeded".into()))?;
        let mut lease =
            Box::pin(execution.enroll_child(owner.clone(), self.autonomy_owner.clone())).await?;
        drop(guard);
        let mut cleanup = crate::autonomy::OwnedTurnCleanup::new(owner.clone());
        let turn = crate::autonomy::AutonomyTurnInput {
            owner,
            objective: input.into(),
            controller_message:
                serde_json::json!({"delegated_objective":input,"operation":operation}).to_string(),
            source: frame.source,
        };
        let outcome = crate::autonomy::scope_child_operation(
            operation.clone(),
            Box::pin(self.resume_task_parent_frame(turn, frame)),
        )
        .await;
        if matches!(&outcome, Err(AgentError::TaskSuspended(_))) {
            Box::pin(execution.checkpoint_child_composition_park(
                &operation,
                self.task_suspension_snapshot().await?,
            ))
            .await?;
            lease.acknowledge_park();
        } else {
            let runtime = crate::autonomy::TaskRuntimeCheckpoint::between_turns(
                self.save_state_full().await?,
            )?;
            Box::pin(execution.checkpoint_child(&operation, runtime, input, Some(&outcome)))
                .await?;
            lease.acknowledge();
        }
        cleanup.finish();
        outcome
    }

    /// Retires the parent interaction only after the composition returns a settled result, never while children are parked.
    pub(super) async fn retire_composition_pending(&self) -> Result<()> {
        let Some(execution) = crate::autonomy::current_execution() else {
            return Ok(());
        };
        let runtime =
            crate::autonomy::TaskRuntimeCheckpoint::between_turns(self.save_state_full().await?)?;
        let operation = crate::autonomy::current_child_operation();
        Box::pin(execution.update(|payload| {
            if let Some(operation) = operation {
                let child = payload
                    .children
                    .iter_mut()
                    .find(|child| child.child_id == operation)
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
        }))
        .await?;
        Ok(())
    }
}
