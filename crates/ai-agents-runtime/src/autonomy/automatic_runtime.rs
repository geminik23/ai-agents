//! Root promotion is a live capability transfer, not a second chat loop or a relaxed public gate.

use super::*;
use crate::autonomy::automatic::{
    self, AutomaticBinding, AutomaticEntry, AutomaticTaskHost, AutomaticTaskOptions,
};
use crate::autonomy::{
    AutonomyConfig, AutonomyRunner, AutonomyScope, TaskRunResult, TaskRunStatus,
    TaskStreamOperation,
};

struct PromotionCleanup(automatic::Promotion);
impl Drop for PromotionCleanup {
    // An interrupted handoff keeps the runtime reserved and makes the unpublished owner explicitly abandoned.
    fn drop(&mut self) {
        if let Some(entry) = self.0.lock().as_ref() {
            entry.owner.abandon();
        }
    }
}

impl RuntimeAgent {
    /// Validates final component and declaration bindings without provider calls, storage construction or task admission.
    /// None defers only the configured store check to explicit preparation; an installed restore host is still required for persistent automatic entry.
    pub(crate) fn preflight_task_hosting(
        &self,
        config: &AutonomyConfig,
        options: &AutomaticTaskOptions,
        reconstruction: Option<bool>,
    ) -> Result<()> {
        self.preflight_standalone_autonomy()?;
        let validate = |scope,
                        settings: &ai_agents_core::autonomy::AutonomyOverride|
         -> Result<()> {
            let profile = crate::autonomy::resolve_profile(
                config,
                scope,
                None,
                Some(settings),
                None,
                Some("invocation objective"),
                &options.ceilings,
            )?;
            if profile.enabled {
                self.preflight_priced_autonomy(profile.max_cost_usd.is_some())?;
                self.autonomy_extensions.bind_for_agent(self, &profile)?;
                if profile
                    .settings
                    .persistence
                    .as_ref()
                    .is_some_and(|p| p.enabled != Some(false))
                    && (options.restore_host.is_none() || reconstruction == Some(false))
                {
                    return Err(AgentError::Config("automatic task persistence requires an explicit reconstruction store and restore host".into()));
                }
            }
            Ok(())
        };
        // Validate every reachable declaration without rendering an objective or constructing/calling a provider.
        fn states(
            defs: &HashMap<String, ai_agents_state::StateDefinition>,
            validate: &impl Fn(AutonomyScope, &ai_agents_core::autonomy::AutonomyOverride) -> Result<()>,
        ) -> Result<()> {
            for definition in defs.values() {
                if let Some(settings) = &definition.autonomy {
                    validate(AutonomyScope::State, settings)?;
                }
                if let Some(children) = &definition.states {
                    states(children, validate)?;
                }
            }
            Ok(())
        }
        if let Some(machine) = &self.state_machine {
            states(&machine.config().states, &validate)?;
        }
        for skill in &self.skills {
            if let Some(settings) = &skill.autonomy {
                validate(AutonomyScope::Skill, settings)?;
            }
        }
        Ok(())
    }

    /// Prepares only a selected task-capable session backend; snapshot-only configurations do not trigger hidden service connections.
    /// The unpublished runtime retains the same backend for session and task work, while every runner still receives its own fixed actor scope.
    pub(crate) async fn prepare_task_storage(
        &self,
    ) -> Result<Option<Arc<dyn ai_agents_core::AgentStorage>>> {
        let injected = self.storage.read().clone();
        if let Some(storage) = injected {
            return Ok(storage
                .supports(ai_agents_core::StorageCapability::TaskRuns)
                .then_some(storage));
        }
        if !matches!(self.storage_config, crate::spec::StorageConfig::Sqlite(_)) {
            return Ok(None);
        }
        self.init_storage().await?;
        let storage = self.storage.read().clone();
        match storage {
            Some(storage) if storage.supports(ai_agents_core::StorageCapability::TaskRuns) => {
                Ok(Some(storage))
            }
            _ => Err(AgentError::UnsupportedStorageCapability(
                ai_agents_core::StorageCapability::TaskRuns,
            )),
        }
    }

    /// Publishes one configured service only after final capability checks and optional explicit storage preparation.
    /// Reconstruction capability is not process-exit durability, and installing this service never starts a task or consumes a saved decision.
    pub(crate) fn install_automatic_tasks(
        mut self,
        config: AutonomyConfig,
        options: AutomaticTaskOptions,
        storage: Option<Arc<dyn ai_agents_core::AgentStorage>>,
        interaction_delivery: bool,
    ) -> Result<Self> {
        let reconstruction = options.store.as_ref().map_or_else(
            || storage.is_some(),
            |store| store.supports_reconstruction(),
        );
        self.preflight_task_hosting(&config, &options, Some(reconstruction))?;
        let host = Arc::new(AutomaticTaskHost {
            config,
            options,
            interaction_delivery,
            storage: storage
                .unwrap_or_else(|| Arc::new(ai_agents_storage::InMemoryTaskStorage::default())),
            runner: Default::default(),
            outcome: Default::default(),
        });
        *self.inner.automatic_host.write() = Arc::downgrade(&host);
        self.automatic_host = Some(host);
        Ok(self)
    }

    /// Upgrades a service lookup only; execution still needs the actual owner, root gate and current admission.
    pub(super) fn automatic_host(&self) -> Option<Arc<AutomaticTaskHost>> {
        self.automatic_host
            .clone()
            .or_else(|| self.inner.automatic_host.read().upgrade())
    }

    /// Exposes the retained controller to trusted hosts for task events and explicit review, never as serialized authority.
    pub fn automatic_task_runner(&self) -> Option<Arc<AutonomyRunner>> {
        self.automatic_host()
            .and_then(|host| host.runner.lock().clone())
    }

    /// Resolves the configured runner for explicit and automatic work without starting a task or cloning runtime state.
    /// An actor change creates a new scoped runner only when the original owner permits access; retained work cannot migrate actors.
    pub fn task_runner(&self) -> Result<Arc<AutonomyRunner>> {
        if let Some(runner) = self.automatic_task_runner()
            && runner.actor_scope == self.effective_actor_id()
        {
            return Ok(runner);
        }
        let _mutations = self.autonomy_mutations.try_write().map_err(|_| {
            AgentError::Config("task host construction conflicts with a live host operation".into())
        })?;
        self.configured_task_runner()
    }

    // Construction runs under the caller's mutation boundary; automatic promotion already holds that boundary and its root gate.
    fn configured_task_runner(&self) -> Result<Arc<AutonomyRunner>> {
        let host = self.automatic_host().ok_or_else(|| {
            AgentError::Config(
                "task hosting is not configured; supply autonomy settings and task options".into(),
            )
        })?;
        let mut slot = host.runner.lock();
        let actor = self.effective_actor_id();
        if let Some(runner) = slot.as_ref()
            && runner.actor_scope == actor
        {
            return Ok(runner.clone());
        }
        self.check_autonomy_access()?;
        let identity = host
            .options
            .config_identity
            .clone()
            .unwrap_or(format!("automatic:{}", self.task_restore_configuration()?));
        let store = match &host.options.store {
            Some(store) => store.clone(),
            None => Arc::new(crate::autonomy::ScopedTaskRunStore::new(
                host.storage.clone(),
                self.info.id.clone(),
                actor,
                identity.clone(),
            )?),
        };
        let mut runner = AutonomyRunner::try_new(
            self.shared_task_handle(),
            host.config.clone(),
            host.options.ceilings.clone(),
            store,
            identity,
        )?
        .with_interaction_delivery(host.interaction_delivery);
        if let Some(restore) = &host.options.restore_host {
            runner = runner.with_restore_host(restore.clone());
        }
        let runner = Arc::new(runner);
        *slot = Some(runner.clone());
        Ok(runner)
    }

    /// Executes an explicit objective through the same configured runner used by automatic scoped entry.
    /// Profile resolution, live ownership, admission and acknowledged pause preparation remain controller responsibilities.
    pub async fn run_task(&self, objective: &str, profile: Option<&str>) -> Result<TaskRunResult> {
        self.task_runner()?.run(objective, profile).await
    }

    /// Resumes only the exact acknowledged revision on the retained runner; the trusted host authenticates the response caller.
    /// Turn-local actor context may have ended, so control must retain the admitted store scope rather than reconstruct it from session defaults.
    pub async fn resume_task(
        &self,
        run_id: &str,
        revision: u64,
        input: crate::autonomy::TaskResumeInput,
    ) -> Result<TaskRunResult> {
        let runner = self
            .automatic_task_runner()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)?;
        runner.resume(run_id, revision, input).await
    }

    /// Cancels the exact retained task revision and distinguishes a request receipt from acknowledged cleanup.
    /// Running hosts must keep polling their original foreground operation; this method never detaches or fabricates terminal events.
    pub async fn cancel_task(
        &self,
        run_id: &str,
        revision: u64,
    ) -> Result<crate::autonomy::CancellationOutcome> {
        self.retained_task_runner()?
            .cancel_task(run_id, revision)
            .await
    }

    /// Reads a bounded default-redacted snapshot in the retained run's original scope without constructing or executing work.
    /// This is an acknowledged display observation, not a current completion permit or caller authentication.
    pub async fn task_display(&self, run_id: &str) -> Result<crate::autonomy::TaskRunDisplay> {
        self.retained_task_runner()?.task_display(run_id).await
    }

    /// Returns exact typed prompt data only to a trusted host; diagnostics and display snapshots omit its sensitive content.
    /// Request and revision must be returned unchanged, and any later policy/binding checks still run at execution admission.
    pub async fn task_interaction(
        &self,
        run_id: &str,
        revision: u64,
    ) -> Result<Option<crate::autonomy::TaskHostInteraction>> {
        self.retained_task_runner()?
            .task_interaction(run_id, revision)
            .await
    }

    /// Applies a typed operation through the configured controller, including controls which must not execute another model turn.
    /// Only new execution resolves infrastructure; control keeps the original retained actor/store and acknowledged result publication.
    pub async fn task_operation(&self, operation: TaskStreamOperation) -> Result<TaskRunResult> {
        let runner = if matches!(
            &operation,
            TaskStreamOperation::Run { .. } | TaskStreamOperation::Skill { .. }
        ) {
            self.task_runner()?
        } else {
            self.retained_task_runner()?
        };
        let mut stream = runner.stream_operation(operation, Default::default())?;
        while stream.next_event().await.is_some() {}
        stream
            .take_result()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)?
    }

    // A control lookup must not re-scope a paused actor or silently construct a replacement for a missing live continuation.
    fn retained_task_runner(&self) -> Result<Arc<AutonomyRunner>> {
        self.automatic_task_runner()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable.into())
    }

    /// Returns exact trusted-host data for the most recently published automatic operation, not a safe export DTO.
    pub fn automatic_task_result(&self) -> Option<TaskRunResult> {
        self.automatic_host()
            .and_then(|host| host.outcome.lock().result.lock().clone())
    }

    /// Applies explicit task control on the same retained controller and wakes ordinary composing callers after acknowledgement.
    /// The host authenticates its caller; model answers and display identifiers do not authorize this method.
    pub async fn automatic_task_operation(
        &self,
        operation: TaskStreamOperation,
    ) -> Result<TaskRunResult> {
        let host = self
            .automatic_host()
            .ok_or_else(|| AgentError::Config("automatic task hosting is not installed".into()))?;
        let outcome = host.outcome.lock().clone();
        let runner = host
            .runner
            .lock()
            .clone()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)?;
        let mut stream = runner.stream_operation(operation, Default::default())?;
        while stream.next_event().await.is_some() {}
        let result = stream
            .take_result()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)??;
        outcome.publish(&result);
        Ok(result)
    }

    /// Uses the nearest explicit state declaration; a child false suppresses automatic startup rather than disabling an outer task.
    fn automatic_state(&self) -> Option<(String, ai_agents_core::autonomy::AutonomyOverride)> {
        let machine = self.state_machine.as_ref()?;
        let mut path = machine.current();
        loop {
            if let Some(settings) = machine
                .get_definition(&path)
                .and_then(|definition| definition.autonomy.clone())
            {
                return Some((path, settings));
            }
            if machine
                .get_definition(&path)
                .is_some_and(|definition| !definition.inherit_parent)
            {
                return None;
            }
            let (parent, _) = path.rsplit_once('.')?;
            path = parent.to_string();
        }
    }

    /// Resolves objective data once from accepted input and the captured scope, keeping templates compatible with ordinary skill inputs.
    pub(crate) fn render_automatic_objective(
        &self,
        settings: &ai_agents_core::autonomy::AutonomyProfile,
        binding: &AutomaticBinding,
    ) -> Result<String> {
        if let Some(template) = &settings.objective_template {
            let mut environment = minijinja::Environment::new();
            environment.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
            return environment.render_str(template, serde_json::json!({
                "input":binding.prepared.content,"user_input":binding.prepared.content,
                "objective":settings.objective.as_deref().unwrap_or(&binding.prepared.content),
                "context":self.context_manager.get_all(),"state":binding.state.as_ref().cloned().or_else(||self.current_state()),
                "skill":binding.skill,
            })).map_err(|error| AgentError::TemplateError(error.to_string()));
        }
        Ok(match &settings.objective {
            Some(objective) => format!("{objective}\n\n{}", binding.prepared.content),
            None => binding.prepared.content.clone(),
        })
    }

    /// Validates the original declaration rather than reselecting from the current state after exit or restoration.
    pub(crate) fn validate_automatic_binding(&self, binding: &AutomaticBinding) -> Result<()> {
        if let Some(state) = &binding.state
            && self
                .state_machine
                .as_ref()
                .and_then(|machine| machine.get_definition(state))
                .and_then(|definition| definition.autonomy.clone())
                .as_ref()
                != Some(&binding.settings)
        {
            return Err(AgentError::Config(
                "automatic state declaration changed".into(),
            ));
        }
        if let Some(skill) = &binding.skill
            && !self
                .skills
                .iter()
                .any(|actual| serde_json::to_value(actual).ok() == serde_json::to_value(skill).ok())
        {
            return Err(AgentError::Config(
                "automatic skill declaration changed".into(),
            ));
        }
        Ok(())
    }

    /// Retrieves only the live owner for an unstarted prepared segment; checkpoint labels cannot manufacture this binding.
    pub(crate) fn autonomy_owner_for_prepared(&self) -> Result<Arc<crate::autonomy::RunOwner>> {
        self.autonomy_owner
            .read()
            .clone()
            .ok_or_else(|| crate::autonomy::TaskRunStorageError::NotResumable.into())
    }

    /// Restores scope metadata only onto the actual reserved owner; it cannot create new ownership or consume a pending request.
    pub(crate) fn restore_automatic_scope(
        &self,
        binding: &AutomaticBinding,
        mode: crate::autonomy::AutonomyMode,
    ) -> Result<()> {
        let owner = self
            .autonomy_owner
            .read()
            .clone()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)?;
        *owner.state_scope.write() = if mode == crate::autonomy::AutonomyMode::StateUntilExit {
            binding.state.clone()
        } else {
            None
        };
        Ok(())
    }

    /// A coordinating state owns its subtree; transitions within it do not end the task and nested children have separate gates.
    pub(crate) fn automatic_scope_ended(&self) -> bool {
        let Some(owner) = self.autonomy_owner.read().clone() else {
            return false;
        };
        let scope = owner.state_scope.read().clone();
        scope.is_some_and(|scope| {
            self.current_state().is_none_or(|current| {
                current != scope && !current.starts_with(&format!("{scope}."))
            })
        })
    }

    /// Promotes accepted state work before any actual work; erasure keeps unused automatic preparation off nested task stacks.
    pub(super) fn promote_automatic_state<'a>(
        &'a self,
        data: &'a ProcessData,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        if crate::autonomy::current_execution().is_some() || self.automatic_host().is_none() {
            return Box::pin(async { Ok(()) });
        }
        Box::pin(async move {
            let Some((state, settings)) = self.automatic_state() else {
                return Ok(());
            };
            self.promote_automatic(Box::new(AutomaticBinding {
                state: Some(state),
                settings,
                skill: None,
                prepared: data.clone(),
                user_committed: self.root_user_message_committed.load(Ordering::SeqCst),
                state_generation: self
                    .state_machine
                    .as_ref()
                    .map(|machine| machine.generation()),
            }))
            .await
        })
    }

    /// Retains the selected definition and prepared clarification context; unused routing adds no large future to existing task polling.
    pub(super) fn promote_automatic_skill<'a>(
        &'a self,
        skill: &'a SkillDefinition,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        if crate::autonomy::current_execution().is_some() || self.automatic_host().is_none() {
            return Box::pin(async { Ok(()) });
        }
        Box::pin(async move {
            let mut data = crate::autonomy::interaction::prepared_input()
                .or_else(|| self.automatic_skill_prepared(&skill.id, input))
                .unwrap_or_else(|| ProcessData::new(input));
            data.content = input.into();
            if data.context.is_empty() {
                data.context = self
                    .active_turn_context
                    .read()
                    .as_ref()
                    .map(|context| context.input_context.clone())
                    .unwrap_or_default();
            }
            let generation = self
                .state_machine
                .as_ref()
                .map(|machine| machine.generation());
            if let Some((state, settings)) = self.automatic_state() {
                self.promote_automatic(Box::new(AutomaticBinding {
                    state: Some(state),
                    settings,
                    skill: Some(skill.clone()),
                    prepared: data.clone(),
                    user_committed: self.root_user_message_committed.load(Ordering::SeqCst),
                    state_generation: generation,
                }))
                .await?;
            }
            let Some(settings) = &skill.autonomy else {
                return Ok(());
            };
            self.promote_automatic(Box::new(AutomaticBinding {
                state: None,
                settings: settings.clone(),
                skill: Some(skill.clone()),
                prepared: data,
                user_committed: self.root_user_message_committed.load(Ordering::SeqCst),
                state_generation: generation,
            }))
            .await
        })
    }

    /// Publishes a live owner while the original root gate still excludes competitors, then unwinds to its designated caller.
    /// Ordinary preparation is retained but never relabelled as task invocation evidence or replayed for admission.
    async fn promote_automatic(&self, binding: Box<AutomaticBinding>) -> Result<()> {
        let Some(slot) = automatic::promotion(&self.root_turn_gate) else {
            return Ok(());
        };
        let Some(host) = self.automatic_host() else {
            return Ok(());
        };
        let profile = crate::autonomy::resolve_profile(
            &host.config,
            binding.scope(),
            None,
            Some(&binding.settings),
            None,
            Some(&binding.prepared.content),
            &host.options.ceilings,
        )?;
        if !profile.enabled {
            return Ok(());
        }
        self.preflight_priced_autonomy(profile.max_cost_usd.is_some())?;
        self.autonomy_extensions.bind_for_agent(self, &profile)?;
        self.check_autonomy_access()?;
        let _mutations = self
            .autonomy_mutations
            .try_write()
            .map_err(|_| AgentError::Other("runtime has an active host operation".into()))?;
        let runner = self.configured_task_runner()?;
        let owner = crate::autonomy::RunOwner::new(
            uuid::Uuid::new_v4().to_string(),
            self.root_turn_gate.clone(),
            self.effective_actor_id(),
        )?;
        *owner.state_scope.write() =
            if profile.mode == crate::autonomy::AutonomyMode::StateUntilExit {
                binding.state.clone()
            } else {
                None
            };
        owner.telemetry.register(self.observability());
        *self.autonomy_owner.write() = Some(owner.clone());
        *self.pending_automatic_input.write() = None;
        let outcome = Arc::new(automatic::AutomaticOutcome {
            key: Some(crate::autonomy::TaskRunKey {
                agent_id: self.info.id.clone(),
                run_id: owner.run_id.clone(),
            }),
            ..Default::default()
        });
        runner.bind_automatic_outcome(outcome.clone());
        *host.outcome.lock() = outcome.clone();
        *slot.lock() = Some(Box::new(AutomaticEntry {
            owner,
            binding,
            outcome,
        }));
        Err(AgentError::TaskSuspended("automatic root handoff".into()))
    }

    /// Only semantic routing and its clarification return qualify for automatic entry; erasure preserves default nested polling stacks.
    pub(super) fn execute_selected_skill<'a>(
        &'a self,
        skill: &'a SkillDefinition,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            self.promote_automatic_skill(skill, input).await?;
            Box::pin(self.execute_skill(skill, input)).await
        })
    }

    /// Retains one ordinary clarification's original preparation without changing selection or creating task authority.
    pub(super) fn retain_automatic_skill_input(&self, skill_id: &str, input: &str) {
        if self.automatic_host().is_none() || crate::autonomy::current_execution().is_some() {
            return;
        }
        let data = crate::autonomy::interaction::prepared_input().unwrap_or_else(|| {
            let mut data = ProcessData::new(input);
            data.context = self
                .active_turn_context
                .read()
                .as_ref()
                .map(|context| context.input_context.clone())
                .unwrap_or_default();
            data
        });
        *self.pending_automatic_input.write() = Some(automatic::PreparedSkillInput {
            skill: skill_id.into(),
            epoch: self.disambiguation_epoch.load(Ordering::SeqCst),
            generation: self
                .state_machine
                .as_ref()
                .map(|machine| machine.generation()),
            data: Box::new(data),
        });
    }

    /// Reads preparation only for the same pending skill and state/clarification lineage; a new answer changes content, not prior extraction.
    fn automatic_skill_prepared(&self, skill_id: &str, input: &str) -> Option<ProcessData> {
        let pending = self.pending_automatic_input.read();
        let pending = pending.as_ref()?;
        if pending.skill != skill_id
            || pending.epoch != self.disambiguation_epoch.load(Ordering::SeqCst)
            || pending.generation
                != self
                    .state_machine
                    .as_ref()
                    .map(|machine| machine.generation())
        {
            return None;
        }
        let mut data = (*pending.data).clone();
        data.content = input.into();
        Some(data)
    }

    /// Ordinary clarification returns use the original prepared context in both blocking and streaming dispatch without extracting it again.
    pub(super) fn recheck_skill_disambiguation<'a>(
        &'a self,
        skill_id: &'a str,
        input: &'a str,
        epoch: u64,
        generation: Option<u64>,
    ) -> Pin<Box<dyn Future<Output = Result<AgentResponse>> + Send + 'a>> {
        if crate::autonomy::current_execution().is_some() {
            return Box::pin(
                self.recheck_skill_disambiguation_inner(skill_id, input, epoch, generation),
            );
        }
        Box::pin(async move {
            if let Some(data) = self.automatic_skill_prepared(skill_id, input) {
                self.update_active_turn_context(&data.content, data.context.clone());
                crate::autonomy::interaction::scope_prepared_input(
                    data,
                    Box::pin(
                        self.recheck_skill_disambiguation_inner(skill_id, input, epoch, generation),
                    ),
                )
                .await
            } else {
                Box::pin(
                    self.recheck_skill_disambiguation_inner(skill_id, input, epoch, generation),
                )
                .await
            }
        })
    }

    /// Resumes the already prepared initial segment without repeating receive hooks, extraction, semantic selection or user commit.
    pub(crate) async fn run_automatic_prepared(
        &self,
        binding: Box<AutomaticBinding>,
    ) -> Result<AgentResponse> {
        if binding.state_generation
            != self
                .state_machine
                .as_ref()
                .map(|machine| machine.generation())
        {
            return Err(AgentError::Config(
                "automatic entry state generation changed before initial execution".into(),
            ));
        }
        self.begin_root_turn();
        let _cleanup = RootTurnCleanup::new(self);
        self.root_user_message_committed
            .store(binding.user_committed, Ordering::SeqCst);
        self.update_active_turn_context(
            &binding.prepared.content,
            binding.prepared.context.clone(),
        );
        if let Some(skill) = binding.skill {
            let content = self
                .execute_skill(&skill, &binding.prepared.content)
                .await?;
            self.commit_root_user_message(&binding.prepared.content)
                .await?;
            return self
                .handle_skill_response(
                    &binding.prepared.content,
                    &skill.id,
                    content,
                    &binding.prepared.context,
                )
                .await;
        }
        self.run_accepted_input(binding.prepared).await
    }

    /// Waits in the original ordinary parent's call frame while a trusted host services its child's retained pause.
    /// No parent stage, route or completed sibling is replayed; dropping the wait does not cancel an acknowledged child pause.
    async fn deliver_automatic_result(
        &self,
        mut result: TaskRunResult,
        composed: bool,
        outcome: Arc<automatic::AutomaticOutcome>,
    ) -> Result<AgentResponse> {
        outcome.publish(&result);
        if !composed {
            return Ok(automatic::response(&result));
        }
        loop {
            let changed = outcome.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if outcome.failed.load(Ordering::Acquire) {
                return Err(AgentError::Other(
                    "automatic child requires explicit recovery after interrupted acknowledgement"
                        .into(),
                ));
            }
            if let Some(current) = outcome.result.lock().clone() {
                result = current;
            }
            if result.run.status != TaskRunStatus::Paused {
                return automatic::completed_child(&result);
            }
            changed.await;
        }
    }

    /// Settles earlier ordinary maintenance before accepting a new input; callbacks are never charged to the later automatic task.
    async fn acquire_automatic_root(&self) -> Result<RootTurnAdmission> {
        loop {
            let admission = self.acquire_root_turn().await?;
            scope_runtime_gate_identity_stack(&admission.identity_stack, self.init_storage())
                .await?;
            if self.background_maintenance.is_empty() {
                return Ok(admission);
            }
            drop(admission);
            self.flush_background_tasks().await?;
        }
    }

    /// Owns the ordinary root until promotion, then releases only its gate while the minted task reservation excludes interleaving.
    pub(super) async fn automatic_chat(
        &self,
        input: &str,
        actor: Option<crate::TurnActorContext>,
    ) -> Result<AgentResponse> {
        let ancestry = current_runtime_gate_identity_stack();
        let composed = !ancestry.is_empty() || automatic::composing();
        let RootTurnAdmission {
            guard,
            identity_stack,
        } = self.acquire_automatic_root().await?;
        let slot = Arc::new(parking_lot::Mutex::new(None));
        let _handoff_cleanup = PromotionCleanup(slot.clone());
        let observation = self.build_observation_context(
            actor
                .as_ref()
                .and_then(|actor| actor.effective_actor_id().map(str::to_string)),
        );
        let work = automatic::scope_promotion(self.root_turn_gate.clone(), slot.clone(), async {
            match actor.clone() {
                Some(actor) => scope_actor_context(actor, Box::pin(self.run_loop(input))).await,
                None => Box::pin(self.run_loop(input)).await,
            }
        });
        let result = scope_runtime_gate_identity_stack(&identity_stack, async {
            if let Some(context) = &observation {
                with_observation_context(context.clone(), work).await
            } else {
                work.await
            }
        })
        .await;
        drop(guard);
        let entry = slot.lock().take();
        let Some(entry) = entry else {
            self.export_observability_if_configured().await;
            return result;
        };
        let runner = self
            .automatic_task_runner()
            .ok_or(crate::autonomy::TaskRunStorageError::NotResumable)?;
        let outcome = entry.outcome.clone();
        let work = async {
            let work = runner.run_automatic(entry);
            match actor {
                Some(actor) => scope_actor_context(actor, work).await,
                None => work.await,
            }
        };
        let result = if let Some(context) = observation {
            with_observation_context(context, work).await
        } else {
            work.await
        };
        self.export_observability_if_configured().await;
        self.deliver_automatic_result(result?, composed, outcome)
            .await
    }

    /// Ordinary streams retain token streaming until an actual promotion; task execution then follows the same foreground controller.
    /// The stream owns both phases, and its authoritative Final reports the task status without exposing raw pending data.
    pub(super) async fn automatic_event_stream<'a>(
        &'a self,
        input: &'a str,
        actor: Option<crate::TurnActorContext>,
        legacy_errors: bool,
    ) -> Result<Pin<Box<dyn Stream<Item = AgentStreamEvent> + Send + 'a>>> {
        let RootTurnAdmission {
            guard,
            identity_stack,
        } = self.acquire_automatic_root().await?;
        let terminal = new_runtime_stream_terminal_slot();
        let observation = self.build_observation_context(
            actor
                .as_ref()
                .and_then(|actor| actor.effective_actor_id().map(str::to_string)),
        );
        Ok(Box::pin(async_stream::stream! {
            let mut guard = Some(guard);
            let slot = Arc::new(parking_lot::Mutex::new(None));
            let _handoff_cleanup = PromotionCleanup(slot.clone());
            let mut inner = self.run_loop_stream(input, terminal.clone());
            loop {
                let next = automatic::scope_promotion(self.root_turn_gate.clone(), slot.clone(), poll_scoped_chunk(&mut inner, &identity_stack, observation.as_ref(), actor.as_ref())).await;
                let promoted = slot.lock().is_some();
                if promoted {
                    // The internal control-transfer error is not a provider error or provisional user output.
                    while automatic::scope_promotion(self.root_turn_gate.clone(), slot.clone(), poll_scoped_chunk(&mut inner, &identity_stack, observation.as_ref(), actor.as_ref())).await.is_some() {}
                    drop(inner);
                    drop(guard.take());
                    let entry = slot.lock().take().expect("live promotion slot");
                    let runner = self.automatic_task_runner().expect("promoted runner");
                    let outcome = entry.outcome.clone();
                    let mut task_events = runner.stream_automatic(entry).expect("default bounded task stream");
                    loop {
                        let next = async {
                            match actor.clone() {
                                Some(actor) => scope_actor_context(actor, task_events.next_event()).await,
                                None => task_events.next_event().await,
                            }
                        };
                        let event = if let Some(observation) = &observation {
                            with_observation_context(observation.clone(), next).await
                        } else { next.await };
                        if let Some(event) = event { yield AgentStreamEvent::Task(event); } else { break; }
                    }
                    let result = task_events.take_result().expect("drained automatic task result");
                    match result {
                        Ok(result) => match self.deliver_automatic_result(result, false, outcome).await {
                            Ok(response) => { if !response.content.is_empty() { yield AgentStreamEvent::Chunk(StreamChunk::content(&response.content)); } yield AgentStreamEvent::Final(response); },
                            Err(error) => yield AgentStreamEvent::Chunk(StreamChunk::error(error.to_string())),
                        },
                        Err(error) => yield AgentStreamEvent::Chunk(StreamChunk::error(error.to_string())),
                    }
                    return;
                }
                match next {
                    Some(StreamChunk::Done {}) => {
                        let response = terminal.write().take();
                        while poll_scoped_chunk(&mut inner, &identity_stack, observation.as_ref(), actor.as_ref()).await.is_some() {}
                        drop(inner);
                        drop(guard.take());
                        if let Some(response) = response { yield AgentStreamEvent::Final(response); }
                        return;
                    },
                    Some(chunk @ StreamChunk::Error { .. }) => {
                        if terminal.read().is_some() {
                            if legacy_errors { yield AgentStreamEvent::Chunk(chunk); }
                            continue;
                        }
                        while poll_scoped_chunk(&mut inner, &identity_stack, observation.as_ref(), actor.as_ref()).await.is_some() {}
                        drop(inner);
                        drop(guard.take());
                        yield AgentStreamEvent::Chunk(chunk);
                        return;
                    },
                    Some(chunk) => yield AgentStreamEvent::Chunk(chunk),
                    None => { drop(inner); drop(guard.take()); return; },
                }
            }
        }))
    }
}
