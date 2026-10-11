//! Task broadcasts retain recipient operations and drain admitted siblings before propagating suspension.

use super::AgentRegistry;
use crate::autonomy::{self, composition, message::MessageState};
use crate::orchestration::AgentResult;
use crate::runtime::{current_runtime_gate_identity_stack, scope_runtime_gate_identity_stack};
use crate::{Agent, TurnActorContext};
use ai_agents_core::{AgentError, AgentResponse, Result, ToolResult};
use ai_agents_observability::{current_observation_context, with_observation_context};
use serde_json::json;
use std::sync::Arc;
use tokio::task::JoinSet;

impl AgentRegistry {
    /// A task-aware tool may propagate typed suspension from this broadcast without converting it into recipient failure.
    /// Only the shared executor's live message permit installs a cursor; ordinary host calls keep blocking broadcast semantics.
    pub async fn broadcast_task(
        self: &Arc<Self>,
        from: &str,
        message: &str,
        actor_context: Option<TurnActorContext>,
    ) -> Result<ToolResult> {
        let Some(invocation) = autonomy::message::take_message() else {
            return Ok(ToolResult::ok(
                crate::orchestration::task_tools::broadcast_output(
                    self.broadcast_inner(from, message, actor_context).await,
                )
                .to_string(),
            ));
        };
        let execution = autonomy::current_execution()
            .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let admission = execution.message_admission.lock().await;
        execution.check_message_slot(&invocation.frame)?;
        let mut targets = self.list();
        targets.retain(|target| target.id != from);
        targets.sort_by(|left, right| left.id.cmp(&right.id));
        if targets.is_empty() {
            return Ok(ToolResult::ok("[]"));
        }
        if targets.len() > autonomy::MAX_TASK_CHECKPOINT_RECORDS {
            return Err(autonomy::TaskRunStorageError::CheckpointTooLarge.into());
        }
        let mut frame = *invocation.frame;
        frame.required = autonomy::child_required();
        frame.children = targets
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let target = self
                    .get(&entry.id)
                    .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                target.preflight_standalone_autonomy()?;
                target.reject_task_ancestor()?;
                Ok(composition::CompositionChild {
                    registry_id: entry.id.clone(),
                    runtime_id: target.info().id,
                    operation: format!("{}:{index}", frame.child_operation),
                })
            })
            .collect::<Result<_>>()?;
        frame.delegate_input = if self.send_with_context {
            format!("[From {from}]: {message}")
        } else {
            message.into()
        };
        let mut state: MessageState = serde_json::from_value(frame.cursor.clone())?;
        state.response = autonomy::message::MessageResponse::Broadcast;
        state.child_actor = actor_context;
        state.waiting_for = execution.composition_dependency(&frame)?;
        let deferred = state.waiting_for.is_some();
        frame.cursor = serde_json::to_value(state)?;
        execution
            .retain_delegate_frame(frame.clone(), self.clone())
            .await?;
        execution.targets.bind_tool(&frame.id, invocation.tool)?;
        execution
            .targets
            .bind_composition(&frame.id, self.clone(), None)?;
        drop(admission);
        if deferred && execution.wait_message_dependency(&frame).await? {
            return Err(AgentError::TaskSuspended(frame.id));
        }
        composition::scope_message_composition(
            frame.runtime_id,
            frame.id,
            Box::pin(self.resume_task_broadcast()),
        )
        .await
        .map(|results| {
            ToolResult::ok(crate::orchestration::task_tools::broadcast_output(results).to_string())
        })
    }

    /// Reentry uses captured slots and acknowledged results rather than recataloguing or invoking completed recipients.
    /// The shared barrier closes new admission at a pause while every already admitted sibling drains normally.
    pub(crate) async fn resume_task_broadcast(
        &self,
    ) -> Result<Vec<(String, Result<AgentResponse>)>> {
        let authorized = composition::enter_composition_dispatch();
        composition::scope_dispatch_authority(authorized, Box::pin(self.drive_task_broadcast()))
            .await
    }

    // Private cursor access is scoped to the exact admitted dispatch, not arbitrary nested callbacks.
    async fn drive_task_broadcast(&self) -> Result<Vec<(String, Result<AgentResponse>)>> {
        let execution = autonomy::current_execution()
            .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let frame = execution
            .composition_frame()?
            .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let scope = composition::current_composition();
        let custody = composition::group_response_custody();
        let mut results: Vec<AgentResult> = frame
            .cursor
            .get("results")
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()?
            .unwrap_or_default();
        let live: MessageState = serde_json::from_value(
            execution
                .delegate_frames
                .lock()
                .get(&frame.id)
                .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?
                .cursor
                .clone(),
        )?;
        let mut tasks = JoinSet::new();
        let ancestry = current_runtime_gate_identity_stack();
        let required = frame.required;
        for (index, slot) in frame.children.iter().enumerate() {
            if results.iter().any(|result| result.agent_index == index) {
                continue;
            }
            let target = execution.targets.resolve(&slot.operation)?;
            let input = frame.delegate_input.clone();
            let actor = live.child_actor.clone();
            let scope = scope.clone();
            let slot = slot.clone();
            let custody = custody.clone();
            let execution = execution.clone();
            let ancestry = ancestry.clone();
            let observation = current_observation_context();
            tasks.spawn(Box::pin(async move {
                autonomy::scope_child_requirement(
                    required,
                    autonomy::scope_inherited_execution(
                        Some(execution),
                        Box::pin(scope_runtime_gate_identity_stack(&ancestry, async move {
                            let started = std::time::Instant::now();
                            let work = composition::run_composition_child(
                                scope,
                                Some(slot.clone()),
                                custody,
                                async {
                                    if let Some(actor) = actor {
                                        target.chat_with_actor_context(&input, actor).await
                                    } else {
                                        target.chat(&input).await
                                    }
                                },
                            );
                            let outcome = if let Some(observation) = observation {
                                with_observation_context(observation, work).await
                            } else {
                                work.await
                            };
                            let (response, error) = match outcome {
                                Err(error @ AgentError::TaskSuspended(_)) => return Err(error),
                                Ok(response) => (Some(response), None),
                                Err(error) => (None, Some(error.to_string())),
                            };
                            Ok(AgentResult {
                                agent_index: index,
                                agent_id: slot.registry_id,
                                success: response.is_some(),
                                response,
                                duration_ms: u64::try_from(started.elapsed().as_millis())
                                    .unwrap_or(u64::MAX),
                                error,
                            })
                        })),
                    ),
                )
                .await
            }));
        }
        let mut suspension = None;
        let mut join_failed = false;
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(error @ AgentError::TaskSuspended(_))) => suspension = Some(error),
                Ok(Err(error)) => return Err(error),
                Err(_) => join_failed = true,
            }
        }
        if join_failed {
            execution.stop("uncertain_child");
            return Err(AgentError::Other(
                "broadcast child requires recovery".into(),
            ));
        }
        results.sort_by_key(|result| result.agent_index);
        execution
            .checkpoint_composition_cursor(&frame.runtime_id, json!({"results":results}))
            .await?;
        if let Some(suspension) = suspension {
            if execution.composition_has_ready_dependency(&frame) {
                return composition::scope_message_composition(
                    frame.runtime_id,
                    frame.id,
                    Box::pin(self.resume_task_broadcast()),
                )
                .await;
            }
            return Err(suspension);
        }
        Ok(results
            .into_iter()
            .map(|result| {
                let outcome = result.response.ok_or_else(|| {
                    AgentError::Other(
                        result
                            .error
                            .unwrap_or_else(|| "broadcast recipient failed".into()),
                    )
                });
                (result.agent_id, outcome)
            })
            .collect())
    }
}
