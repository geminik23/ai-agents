//! Observation message calls retain the actual invocation batch and return to validation, never a synthetic model turn.

use super::*;
use crate::autonomy::{
    self,
    composition::{CompositionDispatch, TaskGroupState},
    location::{TaskLocation, TaskObservationLocation},
};
use ai_agents_core::ToolExecutionRequest;

impl RuntimeAgent {
    /// Post-turn observation tools use the existing private run owner and gate without user commits or response finalization.
    pub(crate) async fn scope_observation_invocation<F, T>(&self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>> + Send,
    {
        let Some(execution) = autonomy::current_execution() else {
            return future.await;
        };
        if current_runtime_gate_identity_stack()
            .iter()
            .any(|gate| Arc::ptr_eq(gate, &self.root_turn_gate))
        {
            return Err(AgentError::Config(
                "reentrant observation invocation".into(),
            ));
        }
        let _gate = self.root_turn_gate.clone().lock_owned().await;
        execution.owner.check()?;
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|owner| !Arc::ptr_eq(owner, &execution.owner))
        {
            return Err(AgentError::Config(
                "observation owner binding changed".into(),
            ));
        }
        let mut ancestry = current_runtime_gate_identity_stack().to_vec();
        ancestry.push(self.root_turn_gate.clone());
        let ancestry: RootTurnGateIdentityStack = ancestry.into();
        scope_runtime_gate_identity_stack(
            &ancestry,
            autonomy::scope_turn(
                autonomy::AutonomyTurnInput {
                    owner: execution.owner.clone(),
                    objective: String::new(),
                    controller_message: String::new(),
                    source: autonomy::AutonomyTurnSource::ValidationFix,
                },
                future,
            ),
        )
        .await
    }

    /// Safe child suspension installs the real Task-origin request as an observation location, not as a model-produced call.
    pub(crate) async fn retain_observation_message(
        &self,
        request: ToolExecutionRequest,
        identity: autonomy::EvidenceIdentity,
    ) -> Result<TaskGroupState> {
        let execution = autonomy::current_execution()
            .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        if !execution.parked_batches.lock().contains_key(&self.info.id) {
            Box::pin(
                self.retain_location_batch(TaskLocation::Observation(Box::new(
                    TaskObservationLocation {
                        request,
                        identity,
                        native_exchanges: self.task_native_expectations(),
                    },
                ))),
            )
            .await?;
        }
        let frames = execution.delegate_frames.lock().clone();
        let frame = frames
            .get(&self.info.id)
            .filter(|frame| frame.dispatch == CompositionDispatch::ToolBatch)
            .cloned()
            .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let payload: autonomy::TaskCheckpointPayload =
            serde_json::from_value(execution.acknowledged_snapshot().payload)?;
        let parked = autonomy::composition::collect_parked_leaves(
            &frame,
            &frames,
            &payload,
            &mut Vec::new(),
        )?;
        let leaf = parked
            .first()
            .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        Ok(TaskGroupState {
            version: 1,
            request_id: uuid::Uuid::new_v4().to_string(),
            frame,
            child_operation: leaf.operation.clone(),
            child_runtime_id: leaf.runtime_id.clone(),
            child_request: leaf.request.clone(),
            batch: leaf.batch.clone(),
            frames,
            parked,
        })
    }

    /// Resume reuses captured child operations and tool admission and returns the exact executor record to the observation driver.
    pub(crate) async fn resume_observation_message(
        &self,
        group: TaskGroupState,
        response: autonomy::suspension::BatchResponse,
    ) -> Result<ToolExecutionRecord> {
        let execution = autonomy::current_execution()
            .ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
        let response = self
            .resume_task_group(
                autonomy::AutonomyTurnInput {
                    owner: execution.owner.clone(),
                    objective: String::new(),
                    controller_message: String::new(),
                    source: autonomy::AutonomyTurnSource::ValidationFix,
                },
                group,
                response,
            )
            .await?;
        Ok(serde_json::from_str(&response.content)?)
    }
}
