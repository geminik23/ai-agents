//! Private live catalogue guards supplement serialized frame identities and span final child enrollment.

use super::*;
use crate::spawner::AgentRegistry;
use crate::spawner::registry::PinnedTaskAgent;
use ai_agents_core::{AgentError, Result};
use parking_lot::Mutex;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Default)]
pub(crate) struct CompositionTargets {
    bindings: Mutex<BTreeMap<String, Arc<PinnedTaskAgent>>>,
    retired: std::sync::atomic::AtomicBool,
    tools: Mutex<BTreeMap<String, Arc<dyn ai_agents_core::Tool>>>,
}

impl CompositionTargets {
    /// Captures the actual registered objects before any child effect; discarded preparation releases its own pins.
    pub(crate) fn capture(
        &self,
        frame: &DelegateFrame,
        registry: &Arc<AgentRegistry>,
    ) -> Result<Vec<(String, Arc<PinnedTaskAgent>)>> {
        if self.retired.load(std::sync::atomic::Ordering::Acquire) {
            return Err(AgentError::Other("task catalogue has been retired".into()));
        }
        let slots = if frame.dispatch == composition::CompositionDispatch::Delegate {
            vec![composition::CompositionChild {
                registry_id: frame.delegate_id.clone(),
                runtime_id: frame.delegate_runtime_id.clone(),
                operation: frame.child_operation.clone(),
            }]
        } else {
            frame.children.clone()
        };
        let mut captured = Vec::with_capacity(slots.len());
        for slot in slots {
            let pinned = registry.pin_task_target(&slot.registry_id, &slot.runtime_id)?;
            if self
                .bindings
                .lock()
                .get(&slot.operation)
                .is_some_and(|old| !Arc::ptr_eq(&old.agent, &pinned.agent))
            {
                return Err(AgentError::Config(
                    "task catalogue implementation changed".into(),
                ));
            }
            captured.push((slot.operation, pinned));
        }
        Ok(captured)
    }

    /// Publication follows frame acknowledgement, so a stored operation never gains authority from serialization alone.
    pub(crate) fn install(&self, captured: Vec<(String, Arc<PinnedTaskAgent>)>) -> Result<()> {
        let mut bindings = self.bindings.lock();
        if self.retired.load(std::sync::atomic::Ordering::Acquire) {
            return Err(AgentError::Other("task catalogue has been retired".into()));
        }
        if bindings.len().saturating_add(captured.len()) > MAX_TASK_CHECKPOINT_RECORDS {
            return Err(TaskRunStorageError::CheckpointTooLarge.into());
        }
        if captured.iter().any(|(operation, pinned)| {
            bindings
                .get(operation)
                .is_some_and(|old| !Arc::ptr_eq(&old.agent, &pinned.agent))
        }) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        bindings.extend(captured);
        Ok(())
    }

    /// Dynamic serial calls inherit only their catalogue object's live guard, never a fresh lookup by a reusable label.
    pub(crate) fn bind_call(&self, source: &str, operation: &str) -> Result<()> {
        let mut bindings = self.bindings.lock();
        if self.retired.load(std::sync::atomic::Ordering::Acquire) {
            return Err(AgentError::Other("task catalogue has been retired".into()));
        }
        let pinned = bindings
            .get(source)
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        if let Some(old) = bindings.get(operation) {
            if !Arc::ptr_eq(&old.agent, &pinned.agent) {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
        } else {
            if bindings.len() >= MAX_TASK_CHECKPOINT_RECORDS {
                return Err(TaskRunStorageError::CheckpointTooLarge.into());
            }
            bindings.insert(operation.into(), pinned);
        }
        Ok(())
    }

    /// Resume preflight verifies every captured implementation without consuming a claim or invoking a participant.
    pub(crate) fn resolve(&self, operation: &str) -> Result<Arc<crate::RuntimeAgent>> {
        let pinned = self
            .bindings
            .lock()
            .get(operation)
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        pinned.with_binding(|actual| Ok(actual.clone()))
    }

    /// Registry validation and owner installation are one non-awaiting critical section after durable run admission.
    pub(crate) fn enroll(
        &self,
        operation: &str,
        owner: Arc<RunOwner>,
        slot: boundary::RunOwnerSlot,
        participants: &Participants,
    ) -> Result<()> {
        let pinned = self
            .bindings
            .lock()
            .get(operation)
            .cloned()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        pinned.with_binding(|actual| {
            if !actual.matches_task_gate(&owner.gate) {
                return Err(AgentError::Config(
                    "child gate differs from captured implementation".into(),
                ));
            }
            participants.enroll(owner, slot)
        })
    }

    /// Message resume is tied to the actual reviewed tool implementation, not its reusable canonical name.
    pub(crate) fn bind_tool(
        &self,
        frame_id: &str,
        tool: Arc<dyn ai_agents_core::Tool>,
    ) -> Result<()> {
        let _bindings = self.bindings.lock();
        if self.retired.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let mut tools = self.tools.lock();
        if tools.len() >= MAX_TASK_CHECKPOINT_RECORDS {
            return Err(TaskRunStorageError::CheckpointTooLarge.into());
        }
        if tools
            .get(frame_id)
            .is_some_and(|old| !Arc::ptr_eq(old, &tool))
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        tools.insert(frame_id.into(), tool);
        Ok(())
    }

    /// Retained live capability must still agree with current resolution before a resume claim can execute.
    pub(crate) fn matches_tool(
        &self,
        frame_id: &str,
        tool: &Arc<dyn ai_agents_core::Tool>,
    ) -> bool {
        !self.retired.load(std::sync::atomic::Ordering::Acquire)
            && self
                .tools
                .lock()
                .get(frame_id)
                .is_some_and(|old| Arc::ptr_eq(old, tool))
    }

    /// Only acknowledged terminal or explicit reconciled cleanup retires catalogue protection; stale scopes cannot recreate it.
    pub(crate) fn release(&self) {
        let mut bindings = self.bindings.lock();
        self.retired
            .store(true, std::sync::atomic::Ordering::Release);
        bindings.clear();
        self.tools.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::spawner::SpawnedAgent;

    // Real registry instances exercise pin lifetime without substituting a membership table.
    async fn fixture() -> (Arc<AgentRegistry>, Arc<crate::RuntimeAgent>) {
        let yaml = "name: Pinned\nsystem_prompt: test\n";
        let agent = crate::AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(super::super::runner_tests::RecordingProvider::new(&[
                "done",
            ]))
            .build()
            .unwrap();
        let registry = Arc::new(AgentRegistry::new());
        Box::pin(registry.register(SpawnedAgent::from_runtime(
            "target".into(),
            agent,
            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
        )))
        .await
        .unwrap();
        let agent = registry.get("target").unwrap();
        (registry, agent)
    }

    // Cloned references share one guard; one run's retirement never releases another run's pin.
    #[tokio::test]
    async fn catalogue_pins_protect_unstarted_instances_and_retire_independently() {
        let (registry, agent) = fixture().await;
        let first = registry
            .pin_task_target("target", &agent.info().id)
            .unwrap();
        let clone = first.clone();
        let second = registry
            .pin_task_target("target", &agent.info().id)
            .unwrap();
        assert!(registry.remove("target").await.is_none());
        assert!(
            registry
                .reconcile(&std::collections::HashSet::new(), Vec::new())
                .await
                .is_err()
        );
        drop(first);
        drop(second);
        assert!(registry.remove("target").await.is_none());
        drop(clone);
        assert!(registry.remove("target").await.is_some());
    }

    // The public removal tool reports a refused live reservation truthfully rather than pretending the target disappeared.
    #[tokio::test]
    async fn removal_tool_distinguishes_reserved_target_from_missing_id() {
        use ai_agents_core::Tool;
        let (registry, agent) = fixture().await;
        let pinned = registry
            .pin_task_target("target", &agent.info().id)
            .unwrap();
        let tool = crate::spawner::RemoveAgentTool::new(registry.clone());
        let refused = tool
            .execute(
                serde_json::json!({"id":"target"}),
                ai_agents_core::ToolExecutionContext::test("remove_agent"),
            )
            .await;
        assert!(!refused.success);
        assert!(refused.output.contains("reserved"));
        assert!(registry.contains("target"));
        drop(pinned);
        let removed = tool
            .execute(
                serde_json::json!({"id":"target"}),
                ai_agents_core::ToolExecutionContext::test("remove_agent"),
            )
            .await;
        assert!(removed.success);
        assert!(!registry.contains("target"));
        let missing = tool
            .execute(
                serde_json::json!({"id":"target"}),
                ai_agents_core::ToolExecutionContext::test("remove_agent"),
            )
            .await;
        assert!(!missing.success);
        assert!(missing.output.contains("not found"));
    }

    // A delayed publication cannot recreate catalogue protection after acknowledged retirement.
    #[tokio::test]
    async fn retired_catalogue_rejects_delayed_binding_publication() {
        let (registry, agent) = fixture().await;
        let delayed = registry
            .pin_task_target("target", &agent.info().id)
            .unwrap();
        let targets = CompositionTargets::default();
        targets.release();
        assert!(targets.install(vec![("late".into(), delayed)]).is_err());
        assert!(targets.resolve("late").is_err());
        assert!(registry.remove("target").await.is_some());
    }

    // Even the same display identity cannot authorize a child gate that belongs to a different runtime.
    #[tokio::test]
    async fn final_enrollment_rejects_foreign_gate_before_owner_installation() {
        let (registry, agent) = fixture().await;
        let targets = CompositionTargets::default();
        targets
            .install(vec![(
                "operation".into(),
                registry
                    .pin_task_target("target", &agent.info().id)
                    .unwrap(),
            )])
            .unwrap();
        let owner =
            RunOwner::new("run".into(), Arc::new(tokio::sync::Mutex::new(())), None).unwrap();
        let slot = Arc::new(parking_lot::RwLock::new(None));
        let participants = Participants::default();
        assert!(
            targets
                .enroll("operation", owner, slot.clone(), &participants)
                .is_err()
        );
        assert!(slot.read().is_none());
        assert!(!participants.unsettled());
        targets.release();
        assert!(registry.remove("target").await.is_some());
    }

    // Registry retirement and independent run owner installation share the existing mutation barrier.
    #[tokio::test]
    async fn independent_owner_admission_and_registry_retirement_are_exclusive() {
        let (registry, agent) = fixture().await;
        let guard = agent.try_registry_retirement().unwrap();
        assert!(agent.reserve_autonomy_run("blocked".into()).await.is_err());
        assert!(!agent.has_task_owner());
        drop(guard);
        let owner = agent.reserve_autonomy_run("admitted".into()).await.unwrap();
        assert!(registry.remove("target").await.is_none());
        assert!(
            registry
                .reconcile(&std::collections::HashSet::new(), Vec::new())
                .await
                .is_err()
        );
        agent.release_autonomy_run(&owner).await.unwrap();
        assert!(registry.remove("target").await.is_some());
    }
}
