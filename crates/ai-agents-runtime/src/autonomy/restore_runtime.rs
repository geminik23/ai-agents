//! Private reconstruction seams require an installed actual owner; ordinary session restore remains excluded during tasks.

use super::*;
use crate::autonomy::{TaskRestoreGenerations, TaskRuntimeCheckpoint};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

struct RestoreRuntimeGuard {
    // Owners retain response guards themselves; weak backreferences prevent cycles after an unpublished reconstruction fails.
    owner: std::sync::Weak<crate::autonomy::RunOwner>,
    slot: std::sync::Weak<parking_lot::RwLock<Option<Arc<crate::autonomy::RunOwner>>>>,
    control: Arc<RuntimeControlState>,
    state: Option<Arc<StateMachine>>,
    expected_state: Option<u64>,
    expected_control: u64,
    tools: Arc<ToolRegistry>,
    expected_tools: u64,
    reviewed: Option<Arc<AtomicBool>>,
}
impl std::fmt::Debug for RestoreRuntimeGuard {
    // Diagnostics expose the boundary kind, never runtime memory or host configuration.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RestoreRuntimeGuard")
    }
}
impl ai_agents_core::autonomy::TaskFinalAdmission for RestoreRuntimeGuard {
    // Fresh object/generation binding is validated at backend publication and every restored response admission.
    fn validate(&self) -> Result<()> {
        let owner = self
            .owner
            .upgrade()
            .ok_or(crate::autonomy::TaskRunStorageError::Owned)?;
        let slot = self
            .slot
            .upgrade()
            .ok_or(crate::autonomy::TaskRunStorageError::Owned)?;
        owner.check()?;
        if slot
            .read()
            .as_ref()
            .is_none_or(|current| !Arc::ptr_eq(current, &owner))
            || self.state.as_ref().map(|state| state.generation()) != self.expected_state
            || self.control.version.load(Ordering::SeqCst) != self.expected_control
            || self.tools.version() != self.expected_tools
            || self.control.emergency_deny.load(Ordering::SeqCst)
            || self
                .reviewed
                .as_ref()
                .is_some_and(|reviewed| !reviewed.load(Ordering::Acquire))
        {
            return Err(AgentError::Config(
                "restoration binding or fresh review is invalid".into(),
            ));
        }
        Ok(())
    }
}

impl RuntimeAgent {
    /// Process-local generations are captured only to validate and translate cursor bindings, never copied into new counters.
    pub(crate) fn task_restore_generations(&self) -> TaskRestoreGenerations {
        let safety = self.runtime_safety_snapshot();
        TaskRestoreGenerations {
            state: self
                .state_machine
                .as_ref()
                .map(|machine| machine.generation()),
            control: safety.version,
            policy: safety.tool_security.policy_version(),
            tools: self.tools.version(),
            epoch: self.disambiguation_epoch.load(Ordering::SeqCst),
        }
    }

    /// Hashes framework-owned prepared behavior; opaque providers/tools/services additionally require the installed host binding contract.
    /// Only the digest is persisted, not raw policy configuration or credentials.
    pub(crate) fn task_restore_configuration(&self) -> Result<String> {
        let safety = self.runtime_safety_snapshot();
        let providers: BTreeMap<_, _> = self
            .llm_registry
            .aliases()
            .into_iter()
            .map(|alias| {
                let provider = self.llm_registry.get(&alias)?;
                Ok((alias, provider.provider_name().to_string()))
            })
            .collect::<Result<_>>()?;
        let mut value = serde_json::json!({
            "prompt": self.base_system_prompt, "iterations":self.max_iterations,
            "context_tokens":self.max_context_tokens, "skills":self.skills,
            "states":self.state_machine.as_ref().map(|machine| machine.config()),
            "reasoning":self.reasoning_config, "reflection":self.reflection_config,
            "recovery":self.recovery_manager.config(), "runtime":self.runtime_config,
            "parallel_tools":self.parallel_tools,"streaming":self.streaming,
            "grant":self.declared_tool_ids,"tools":self.validation_capabilities().tools,
            "policy":safety.tool_security.config(),"scope":safety.tool_scope_override,
            "providers":providers,"process":self.process_processor.as_ref().map(|processor| processor.config()),
            "hitl":self.hitl_engine.as_ref().map(|engine| engine.config()),
            "disambiguation":self.disambiguation_manager.as_ref().map(|manager| manager.config()),
            "actor_memory":self.actor_memory_config,"facts":self.facts_config,
        });
        if let Some(manager) = &self.relationship_manager {
            value["relationships"] = serde_json::to_value(manager.config())?;
        }
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
    }

    /// A safe source capture excludes root admission and public host mutation while recording one authoritative current image.
    pub(crate) async fn capture_task_reconstruction(
        &self,
        continuation: crate::autonomy::TaskContinuation,
        expected: Option<&Arc<crate::autonomy::RunOwner>>,
    ) -> Result<(TaskRuntimeCheckpoint, Value, TaskRestoreGenerations)> {
        let _gate = self.root_turn_gate.lock().await;
        let _mutations = self.autonomy_mutations.try_write().map_err(|_| {
            AgentError::Config("source reconstruction capture has an active host operation".into())
        })?;
        let current = self.autonomy_owner.read().clone();
        if current.as_ref().map(Arc::as_ptr) != expected.map(Arc::as_ptr) {
            return Err(AgentError::Config(
                "source runtime belongs to another owner".into(),
            ));
        }
        if let Some(owner) = expected {
            owner.check()?;
        }
        let generations = self.task_restore_generations();
        let snapshot = self.save_state_full().await?;
        let native_exchanges = inspect_native_history(&snapshot.memory.messages)?
            .exchanges()
            .iter()
            .map(|exchange| crate::autonomy::TaskNativeExchange {
                exchange_id: exchange.state().exchange_id().to_owned(),
                call_ids: exchange.call_ids().to_vec(),
            })
            .collect();
        let runtime = TaskRuntimeCheckpoint {
            snapshot,
            native_exchanges,
            continuation,
        };
        runtime.validate()?;
        let session = self.task_restore_session().await?;
        if self.task_restore_generations() != generations {
            return Err(AgentError::Config(
                "runtime changed during restore capture".into(),
            ));
        }
        Ok((runtime, session, generations))
    }

    /// Session and local bookkeeping are exact restoration state, not conversation replay or semantic callbacks.
    /// Actor caches and extraction progress survive a prepared safe pause without repeating completed maintenance.
    pub(crate) async fn task_restore_session(&self) -> Result<Value> {
        let manager = match &self.disambiguation_manager {
            Some(manager) => manager.pending_checkpoint().await?,
            None => None,
        };
        let mut value = serde_json::json!({
            "id":*self.current_session_id.read(), "metadata":*self.session_metadata.read(),
            "actor":*self.actor_id.read(), "initialized":self.context_initialized.load(Ordering::SeqCst),
            "pending_skill":*self.pending_skill_id.read(),"plan":*self.current_plan.read(),
            "manager":manager,
            "mutations":self.observation_mutations.load(Ordering::SeqCst),
        });
        if self.facts_config.is_some()
            || self.actor_memory_config.is_some()
            || self.relationship_manager.is_some()
        {
            value["actor_facts"] = serde_json::to_value(&*self.actor_facts_cache.read())?;
            value["extraction_count"] =
                serde_json::to_value(*self.messages_since_extraction.read())?;
        }
        Ok(value)
    }

    /// Captures the prepared provider registry as an actual host binding, not a deserialized alias map.
    pub(crate) fn task_restore_providers(&self) -> Arc<LLMRegistry> {
        self.llm_registry.clone()
    }

    /// Resolves the current actual tool and provider registry without exposing private permits to serialized data.
    pub(crate) fn task_restore_tool(&self, name: &str) -> Result<Arc<dyn ai_agents_core::Tool>> {
        self.tools
            .get(name)
            .ok_or_else(|| AgentError::Config("restored tool is not installed".into()))
    }

    /// A successful reconstruction claim may reserve only a fresh unowned staged runtime, never adopt an abandoned or foreign slot.
    pub(crate) async fn reserve_reconstructed_owner(
        &self,
        run_id: String,
        actor: Option<String>,
    ) -> Result<Arc<crate::autonomy::RunOwner>> {
        let admission = self.acquire_root_turn().await?;
        let _mutations = self
            .autonomy_mutations
            .try_write()
            .map_err(|_| AgentError::Config("restoration target has active mutation".into()))?;
        if self.autonomy_owner.read().is_some() {
            return Err(AgentError::Config(
                "restoration target is already owned".into(),
            ));
        }
        let owner = crate::autonomy::RunOwner::new(run_id, self.root_turn_gate.clone(), actor)?;
        *self.autonomy_owner.write() = Some(owner.clone());
        drop(admission);
        Ok(owner)
    }

    /// Unstarted staged nodes have no admitted effects; after publication they retain registration pins rather than fabricated task membership.
    /// Slot release and membership retirement share the node gate, so a new owner cannot interleave between them.
    pub(crate) fn release_reconstruction_temporary(
        &self,
        owner: &Arc<crate::autonomy::RunOwner>,
        participants: &crate::autonomy::Participants,
    ) -> Result<()> {
        let _gate = self
            .root_turn_gate
            .try_lock()
            .map_err(|_| AgentError::Config("unstarted restoration target is busy".into()))?;
        let mut current = self.autonomy_owner.write();
        if current
            .as_ref()
            .is_none_or(|active| !Arc::ptr_eq(active, owner))
        {
            return Err(crate::autonomy::TaskRunStorageError::Owned.into());
        }
        *current = None;
        drop(current);
        participants.retire_reconstructed_member(owner)
    }

    /// A reconstructed runtime is reserved before component mutation; this does not bypass ordinary public restore guards.
    /// Native suspended history is installed exactly and later validated by the existing continuation drivers.
    pub(crate) async fn apply_task_reconstruction(
        &self,
        owner: &Arc<crate::autonomy::RunOwner>,
        runtime: &TaskRuntimeCheckpoint,
        session: &Value,
    ) -> Result<()> {
        runtime.validate()?;
        if runtime.snapshot.agent_id != self.info.id {
            return Err(AgentError::Config(
                "restored runtime identity differs".into(),
            ));
        }
        let _gate = self.root_turn_gate.lock().await;
        let _mutations = self.autonomy_mutations.try_write().map_err(|_| {
            AgentError::Config("staged reconstruction target has an active host operation".into())
        })?;
        owner.check()?;
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|active| !Arc::ptr_eq(active, owner))
        {
            return Err(AgentError::Config(
                "task reconstruction lacks actual owner".into(),
            ));
        }
        self.memory.restore(runtime.snapshot.memory.clone()).await?;
        if let (Some(machine), Some(state)) = (&self.state_machine, &runtime.snapshot.state_machine)
            && !state.current_state.is_empty()
        {
            machine.restore(state.clone())?;
        }
        self.context_manager
            .restore(runtime.snapshot.context.clone());
        if let (Some(value), Some(manager)) = (&runtime.snapshot.persona, &self.persona_manager) {
            manager.restore_from_value(value.clone())?;
        }
        if let (Some(value), Some(manager)) =
            (&runtime.snapshot.relationships, &self.relationship_manager)
        {
            manager.restore_from_value(value.clone())?;
        }
        *self.current_session_id.write() = serde_json::from_value(session["id"].clone())?;
        *self.session_metadata.write() = serde_json::from_value(session["metadata"].clone())?;
        *self.actor_id.write() = serde_json::from_value(session["actor"].clone())?;
        if let Some(facts) = session.get("actor_facts") {
            *self.actor_facts_cache.write() = serde_json::from_value(facts.clone())?;
        }
        if let Some(count) = session.get("extraction_count") {
            *self.messages_since_extraction.write() = serde_json::from_value(count.clone())?;
        }
        *self.pending_skill_id.write() = serde_json::from_value(session["pending_skill"].clone())?;
        *self.current_plan.write() = serde_json::from_value(session["plan"].clone())?;
        self.context_initialized.store(
            session["initialized"].as_bool().unwrap_or(false),
            Ordering::SeqCst,
        );
        self.observation_mutations.store(
            session["mutations"]
                .as_u64()
                .ok_or_else(|| AgentError::Config("missing mutation lineage".into()))?,
            Ordering::SeqCst,
        );
        *self.active_native_exchanges.write() = runtime
            .native_exchanges
            .iter()
            .map(|exchange| ActiveNativeExchange {
                exchange_id: exchange.exchange_id.clone(),
                call_ids: exchange.call_ids.clone(),
            })
            .collect();
        if !session["manager"].is_null() {
            let manager = self.disambiguation_manager.as_ref().ok_or_else(|| {
                AgentError::Config("restored clarification manager is absent".into())
            })?;
            let mut pending = session["manager"].clone();
            pending["origin_state_generation"] = serde_json::to_value(
                self.state_machine
                    .as_ref()
                    .map(|machine| machine.generation()),
            )?;
            manager.restore_pending_checkpoint(pending).await?;
        }
        Ok(())
    }

    /// Reconstructs a fresh generation guard over the actual installed runtime; optional review gates still require explicit host approval.
    /// The guard cannot keep its containing owner or owner slot alive; dropped authority denies admission instead of leaking a self-cycle.
    pub(crate) fn task_reconstruction_guard(
        &self,
        owner: Arc<crate::autonomy::RunOwner>,
        reviewed: Option<Arc<AtomicBool>>,
    ) -> Arc<dyn ai_agents_core::autonomy::TaskFinalAdmission> {
        Arc::new(RestoreRuntimeGuard {
            owner: Arc::downgrade(&owner),
            slot: Arc::downgrade(&self.autonomy_owner),
            control: self.runtime_control.clone(),
            state: self.state_machine.clone(),
            expected_state: self.state_machine.as_ref().map(|state| state.generation()),
            expected_control: self.runtime_control.version.load(Ordering::SeqCst),
            tools: self.tools.clone(),
            expected_tools: self.tools.version(),
            reviewed,
        })
    }

    /// Recreates only the live context for an already suspended message; ordinary reauthorization still checks policy, locks and final admission.
    /// Raw policy/custom settings come from current host configuration rather than persisted credentials.
    pub(crate) fn reconstruct_message_context(
        &self,
        value: &Value,
    ) -> Result<ToolExecutionContext> {
        macro_rules! field {
            ($name:ident) => {
                serde_json::from_value(value[stringify!($name)].clone())?
            };
        }
        let canonical_id: String = field!(canonical_id);
        let safety = self.runtime_safety_snapshot();
        let policy_snapshot = safety.tool_security.policy_snapshot(&canonical_id);
        let custom_config = safety.tool_security.custom_config(&canonical_id);
        Ok(ToolExecutionContext {
            requested_name: field!(requested_name),
            canonical_id,
            display_name: field!(display_name),
            provider_id: field!(provider_id),
            registry_version: self.tools.version(),
            policy_version: safety.tool_security.policy_version(),
            runtime_control_version: safety.version,
            call_id: field!(call_id),
            source: field!(source),
            actor: field!(actor),
            cancellation: ToolCancellationToken::new(
                self.runtime_control.emergency_deny.clone(),
                Some("runtime control cancellation".into()),
            ),
            started_at: field!(started_at),
            deadline: field!(deadline),
            permission: field!(permission),
            approval: field!(approval),
            classification: field!(classification),
            safety: field!(safety),
            limits: field!(limits),
            policy_snapshot,
            custom_config,
            // Persisted footprint data is audit-only; current shared admission must derive a fresh live binding.
            admitted_write_footprint: None,
        })
    }

    /// Restores quiescent membership without claiming a live invocation or producing a new accounting reservation.
    pub(crate) fn install_reconstructed_owner(
        &self,
        owner: Arc<crate::autonomy::RunOwner>,
        participants: &crate::autonomy::Participants,
    ) -> Result<()> {
        if self
            .autonomy_owner
            .read()
            .as_ref()
            .is_none_or(|active| !Arc::ptr_eq(active, &owner))
        {
            return Err(AgentError::Config("reconstructed owner changed".into()));
        }
        participants.restore_member(owner, self.autonomy_owner.clone())
    }

    /// Exact lookup of retained owner state allows capture to distinguish actual members from unstarted pinned targets.
    pub(crate) fn captured_task_owner(&self) -> Option<Arc<crate::autonomy::RunOwner>> {
        self.autonomy_owner.read().clone()
    }
}
