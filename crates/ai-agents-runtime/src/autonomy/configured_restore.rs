//! Declarative construction binds actual runtime objects to an installed restoration host.

use super::*;
use crate::spec::PreparedAgentSpec;
use crate::{AgentBuilder, RuntimeAgent};
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

type NodeSource = (PreparedAgentSpec, PathBuf, Option<(String, String)>);

/// A host-installed component factory supplies application tools, providers and domain state without replacing the task engine.
/// Validation is pure; installation constructs unpublished objects and must not execute model, tool or business work.
pub trait TaskRestoreComponents: Send + Sync {
    /// Versions the installed component implementation, not an identifier supplied by model output.
    fn identity(&self) -> (String, u32);
    /// Checks current immutable configuration and saved state before reconstruction claims ownership.
    fn validate(&self, prepared: &PreparedAgentSpec, binding: &TaskRestoreBinding) -> Result<()>;
    /// Configures providers and features through the existing builder chain, then installs application components and their saved state.
    /// The host preserves its storage, scope and ceilings; the installer must not replace those infrastructure settings.
    /// When shared providers are supplied, reuse the already installed handles and do not construct child provider declarations.
    fn install(
        &self,
        prepared: &PreparedAgentSpec,
        builder: AgentBuilder,
        saved: Option<&TaskRestoreBinding>,
        shared_providers: Option<&ai_agents_llm::LLMRegistry>,
    ) -> Result<(AgentBuilder, Arc<dyn TaskRestoreComponentState>)>;
    /// Certifies that interrupted component preparation has no unresolved external work; absence of proof remains an explicit refusal.
    fn validate_interrupted_preparation(&self, _snapshot: &TaskRunSnapshot) -> Result<()> {
        Err(AgentError::Config(
            "component preparation requires explicit stop and effect evidence".into(),
        ))
    }
}

/// The actual installed component instances expose their configuration and state for one runtime's restoration receipt.
pub trait TaskRestoreComponentState: Send + Sync {
    /// Captures pure, non-secret recovery state; telemetry alone is not the component's behavioral state.
    fn binding(&self) -> Result<TaskRestoreBinding>;
}

// First-party construction issues this source descriptor before the runtime is published.
pub(crate) struct ConfiguredTaskSource {
    pub prepared: PreparedAgentSpec,
    pub base_dir: PathBuf,
    pub state: Option<ConfiguredTaskState>,
    pub process_commands: bool,
    pub components: Option<Arc<dyn TaskRestoreComponents>>,
    pub component_state: Option<Arc<dyn TaskRestoreComponentState>>,
}

// Framework-owned mutable tool state is recovery data, not a new authorization or a claim of external freshness.
#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConfiguredTaskState {
    pub route_position: Option<usize>,
    pub file_reads: Vec<ai_agents_tools::FileVersionEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub components: Option<TaskRestoreBinding>,
}

// The receipt retains actual handles and immutable configuration alongside the trusted source recipe.
pub(crate) struct ConfiguredTaskRecipe {
    pub source: Box<ConfiguredTaskSource>,
    pub configuration: String,
    pub providers: Arc<ai_agents_llm::LLMRegistry>,
    pub tools_version: u64,
    pub route: Option<Arc<crate::orchestration::tools::RouteToAgentTool>>,
    pub services_match: Box<dyn Fn() -> bool + Send + Sync>,
    pub component_binding: Option<TaskRestoreBinding>,
}

/// A configured host keeps construction authority outside serialized task data.
/// Callers choose a reconstruction work bound separately from task iteration and time limits.
#[derive(Clone)]
pub struct ConfiguredTaskRestoreHost {
    prepared: PreparedAgentSpec,
    base_dir: PathBuf,
    scope: TaskHostScope,
    budget: Duration,
    configuration: String,
    working_dir: PathBuf,
    declared: BTreeMap<String, (PreparedAgentSpec, PathBuf)>,
    instances: Arc<Mutex<Vec<crate::runtime::WeakRuntimeAgent>>>,
    approval: Option<Arc<dyn ai_agents_hitl::ApprovalHandler>>,
    questions: Option<Arc<dyn ai_agents_tools::QuestionHandler>>,
    components: Option<Arc<dyn TaskRestoreComponents>>,
    clock: Arc<Mutex<Option<TaskUtcEvidence>>>,
    ceilings: AutonomyHostCeilings,
    component_identity: Option<(String, u32)>,
}

impl ConfiguredTaskRestoreHost {
    /// Resolves the declared namespace without constructing providers or opening task storage.
    /// The prepared input is trusted host configuration, never a specification taken from an arbitrary checkpoint.
    /// The positive whole-millisecond work bound cannot exceed an explicit declarative restoration timeout.
    pub fn new(
        prepared: PreparedAgentSpec,
        base_dir: PathBuf,
        actor: Option<String>,
        budget: Duration,
    ) -> Result<Self> {
        let budget = prepared
            .spec()
            .autonomy
            .restoration_timeout_ms
            .map_or(budget, |limit| {
                budget.min(Duration::from_millis(limit.get()))
            });
        let millis = u64::try_from(budget.as_millis()).map_err(|_| {
            AgentError::Config("reconstruction work budget is not representable".into())
        })?;
        if millis == 0 || Duration::from_millis(millis) != budget {
            return Err(AgentError::Config(
                "reconstruction work budget must be positive whole milliseconds".into(),
            ));
        }
        let base_dir = if base_dir.as_os_str().is_empty() {
            std::env::current_dir()?
        } else {
            base_dir.canonicalize()?
        };
        let working_dir = std::env::current_dir()?;
        let mut declared = BTreeMap::new();
        if let Some(spawner) = &prepared.spec().spawner {
            for entry in &spawner.auto_spawn {
                let path = base_dir.join(&entry.agent);
                let child = PreparedAgentSpec::from_yaml_strict(&std::fs::read_to_string(&path)?)?;
                let directory = path
                    .parent()
                    .ok_or_else(|| AgentError::Config("child source has no directory".into()))?
                    .canonicalize()?;
                if declared
                    .insert(entry.id.clone(), (child, directory))
                    .is_some()
                {
                    return Err(AgentError::Config("duplicate declared child".into()));
                }
            }
        }
        let declarations = declared.iter().map(|(id, (child, directory))| {
            Ok((id, json!({"spec": child.spec().identity_value()?, "policy": child.task_iteration_policy(), "directory": directory})))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        let scope = TaskHostScope::from_prepared_spec(&prepared, &base_dir, actor)?;
        let configuration = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&json!({
                "spec": prepared.spec().identity_value()?,
                "policy": prepared.task_iteration_policy(),
                "base_dir": base_dir,
            "working_dir": working_dir,
            "declarations": declarations,
                "scope": scope.identity(),
            }))?)
        );
        Ok(Self {
            prepared,
            base_dir,
            scope,
            budget,
            configuration,
            working_dir,
            declared,
            instances: Arc::new(Mutex::new(Vec::new())),
            approval: None,
            questions: None,
            components: None,
            clock: Default::default(),
            ceilings: Default::default(),
            component_identity: None,
        })
    }

    /// Selects application component construction while retaining the framework's configured topology, clocks and recovery protocol.
    pub fn components(mut self, components: Arc<dyn TaskRestoreComponents>) -> Result<Self> {
        if self.components.is_some() {
            return Err(AgentError::Config(
                "configured component factory is already selected".into(),
            ));
        }
        let identity = components.identity();
        if identity.0.trim().is_empty() || identity.1 == 0 {
            return Err(AgentError::Config(
                "component factory requires a versioned identity".into(),
            ));
        }
        self.configuration = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(self.configuration, &identity))?)
        );
        self.component_identity = Some(identity);
        self.components = Some(components);
        Ok(self)
    }

    /// Binds the current authenticated approval endpoint before constructing any runtime.
    pub fn approval_handler(mut self, handler: Arc<dyn ai_agents_hitl::ApprovalHandler>) -> Self {
        self.approval = Some(handler);
        self
    }

    /// Binds current ordinary question delivery; restored task decisions still use exact typed requests and fresh review.
    pub fn question_handler(mut self, handler: Arc<dyn ai_agents_tools::QuestionHandler>) -> Self {
        self.questions = Some(handler);
        self
    }

    /// Supplies current application ceilings without resetting saved per-run limits or granting any component capability.
    pub fn host_ceilings(mut self, ceilings: AutonomyHostCeilings) -> Self {
        self.ceilings = ceilings;
        self
    }

    /// Returns the checked namespace used by both initial execution and explicit restoration selection.
    pub fn scope(&self) -> &TaskHostScope {
        &self.scope
    }

    // Serialize fresh OS observations so concurrent callers cannot reorder clock evidence; stored bounds never substitute for a new sample.
    fn sample_clock(&self, previous: Option<&TaskUtcEvidence>) -> Result<TaskUtcEvidence> {
        let mut last = self.clock.lock();
        let mut sample = SystemTaskClock::sample()?;
        // Check the fresh interval before widening either upper bound, so one historical bound cannot hide rollback against the other.
        if let Some(previous) = previous {
            sample.validate_after(previous)?;
        }
        if let Some(previous) = last.as_ref() {
            sample.validate_after(previous)?;
        }
        if let Some(previous) = previous {
            sample = sample.retain_upper_after(previous)?;
        }
        if let Some(previous) = last.as_ref() {
            sample = sample.retain_upper_after(previous)?;
        }
        *last = Some(sample.clone());
        Ok(sample)
    }

    // A receipt is tied to the actual runtime identity and provider handles, not a matching display name.
    fn remember(&self, agent: &RuntimeAgent) {
        let mut instances = self.instances.lock();
        instances.retain(|known| known.upgrade().is_some());
        if !instances
            .iter()
            .filter_map(|known| known.upgrade())
            .any(|known| known.same_runtime(agent))
        {
            instances.push(agent.weak_task_handle());
        }
    }

    // Reuses prepared source semantics while keeping object construction separate from task execution.
    fn builder(&self, state: Option<ConfiguredTaskState>) -> Result<AgentBuilder> {
        let mut options = self.scope.task_options();
        options.restore_host = Some(Arc::new(self.clone()));
        options.ceilings = self.ceilings.clone();
        let mut builder =
            AgentBuilder::from_prepared_spec_with_base_dir(self.prepared.clone(), &self.base_dir)
                .task_host_scope(self.scope.clone())?
                .automatic_task_options(options.clone());
        let component_state = if let Some(factory) = &self.components {
            let saved = state.as_ref().and_then(|state| state.components.as_ref());
            if let Some(saved) = saved {
                factory.validate(&self.prepared, saved)?;
            }
            let (configured, installed) = factory.install(&self.prepared, builder, saved, None)?;
            builder = configured;
            Some(installed)
        } else {
            builder = builder.auto_configure_llms()?.auto_configure_features()?;
            None
        };
        builder.verify_configured_host(&options, &self.scope, &self.prepared)?;
        builder = builder.configured_task_source(ConfiguredTaskSource {
            prepared: self.prepared.clone(),
            base_dir: self.base_dir.clone(),
            state,
            process_commands: self.components.is_none(),
            components: self.components.clone(),
            component_state,
        });
        if let Some(handler) = &self.approval {
            builder = builder.approval_handler(handler.clone());
        }
        if self.components.is_none()
            && (self.prepared.spec().memory.is_compacting()
                || self.prepared.spec().memory.token_budget.is_some())
        {
            builder = builder.hooks(Arc::new(ai_agents_hooks::LoggingHooks::with_prefix(
                "[Memory]",
            )));
        }
        Ok(builder)
    }

    /// Builds an initial configured runtime with this host installed for acknowledged persistence pauses.
    /// Persistence admission checks durability and clock evidence; disabled or volatile work does not acquire those prerequisites.
    pub async fn build(self: &Arc<Self>) -> Result<RuntimeAgent> {
        let agent = self
            .builder(None)?
            .auto_configure_mcp()
            .await?
            .auto_configure_spawner()
            .await?
            .build_task_host()
            .await?;
        agent.set_question_handler(self.questions.clone());
        if let Some(actor) = self.scope.actor_id() {
            agent.set_actor_id(actor)?;
        }
        self.remember(&agent);
        Ok(agent)
    }

    // A fresh MCP process or session may reset opaque server state; only an installed application binding can certify its reconstruction contract.
    fn validate_resource_contracts(&self) -> Result<()> {
        if self.components.as_ref().map(|factory| factory.identity()) != self.component_identity {
            return Err(AgentError::Config(
                "configured component factory identity changed".into(),
            ));
        }
        let has_mcp = std::iter::once(&self.prepared)
            .chain(self.declared.values().map(|(prepared, _)| prepared))
            .any(|prepared| {
                prepared
                    .spec()
                    .tools
                    .iter()
                    .flatten()
                    .any(|tool| tool.to_mcp_config().is_some())
            });
        if has_mcp && self.components.is_none() {
            return Err(AgentError::Config(
                "durable MCP hosting requires an installed component state binding".into(),
            ));
        }
        Ok(())
    }

    // The configured database and namespace select the lease; untrusted run IDs are never interpreted as filesystem paths.
    fn acquire_work_lease(&self, run_id: &str) -> Result<super::restore_lease::TaskFileLease> {
        let crate::spec::StorageConfig::Sqlite(storage) = &self.prepared.spec().storage else {
            return Err(AgentError::Config(
                "configured restoration requires SQLite task storage".into(),
            ));
        };
        super::restore_lease::TaskFileLease::acquire(
            std::path::Path::new(&storage.path),
            self.scope.identity(),
            run_id,
        )
    }

    /// Reconciles only an explicitly reviewed, interrupted preparation transaction whose local worker lease can be acquired.
    /// Unknown task effects, missing start evidence and external preparation without cleanup proof remain protected.
    pub async fn recover_prepared(
        self: &Arc<Self>,
        run_id: String,
        revision: u64,
        disposition: TaskRecoveryDisposition,
        reason: String,
    ) -> Result<TaskRunSummary> {
        if !matches!(
            disposition,
            TaskRecoveryDisposition::ResumePrepared | TaskRecoveryDisposition::CancelPrepared
        ) {
            return Err(AgentError::Config(
                "configured recovery accepts only prepared resume or cancellation".into(),
            ));
        }
        let store = self.scope.open_store().await?;
        let source = store
            .load(&run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if source.revision != revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let lease = self.acquire_work_lease(&run_id)?;
        let proof = Arc::new(ConfiguredRecoveryProof {
            host: self.clone(),
            source: serde_json::to_value(&source)?,
            _lease: lease,
        });
        let decision = TaskRecoveryDecision {
            expected_revision: revision,
            reason,
            stopped_at: None,
            interrupted_started_at: None,
            disposition,
            effects: BTreeMap::new(),
            children: BTreeMap::new(),
        };
        AutonomyRunner::reconcile_recovery(store, &run_id, decision, proof).await
    }

    /// Reconstructs only a cancellation cursor and uses the existing typed cleanup; it cannot resume pending business work.
    pub async fn cancel_prepared(
        self: &Arc<Self>,
        run_id: String,
        revision: u64,
    ) -> Result<TaskRunResult> {
        let store = self.scope.open_store().await?;
        AutonomyRunner::restore_cancelled(
            TaskRestoreRequest {
                run_id,
                expected_revision: revision,
                config_identity: self.scope.identity().into(),
                config: self.prepared.spec().autonomy.clone(),
                ceilings: self.ceilings.clone(),
            },
            store,
            self.clone(),
        )
        .await
    }

    /// Restores one exact authorized stored revision and registers its actual runner with the current foreground host, if any.
    /// The returned runner retains the exact restored runtime; keep it alive until the task is settled or safely parked.
    pub async fn restore(
        self: &Arc<Self>,
        run_id: String,
        revision: u64,
    ) -> Result<(Arc<AutonomyRunner>, TaskRunResult)> {
        let store = self.scope.open_store().await?;
        let (runner, result) = AutonomyRunner::restore_paused(
            TaskRestoreRequest {
                run_id,
                expected_revision: revision,
                config_identity: self.scope.identity().into(),
                config: self.prepared.spec().autonomy.clone(),
                ceilings: self.ceilings.clone(),
            },
            store,
            self.clone(),
        )
        .await?;
        let runner = runner.into_configured_host()?;
        if let Some(session) = super::host_session::current_session() {
            session.observe_result(runner.clone(), &result).await?;
        }
        Ok((runner, result))
    }

    /// Inspects startup candidates without preparing providers, claiming ownership or consuming saved decisions.
    pub async fn startup_candidate(&self) -> Result<Option<TaskRunSummary>> {
        let crate::spec::StorageConfig::Sqlite(storage) = &self.prepared.spec().storage else {
            return Err(AgentError::Config(
                "configured task restoration requires SQLite task storage".into(),
            ));
        };
        match std::fs::metadata(&storage.path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
            Ok(metadata) if !metadata.is_file() => {
                return Err(AgentError::Config("task storage must be a file".into()));
            }
            Ok(_) => {}
        }
        let store = self.scope.open_store().await?;
        // An owned or recovery-required startup task is not equivalent to having no saved work.
        for row in store.list().await? {
            if row.owned
                || row.status == TaskRunStatus::RecoveryRequired
                || (row.cancel_requested && !row.status.is_terminal())
            {
                let snapshot = store
                    .inspect_candidate(&row.key.run_id)
                    .await?
                    .ok_or(TaskRunStorageError::NotFound)?;
                if snapshot.revision != row.revision {
                    return Err(TaskRunStorageError::Conflict.into());
                }
                let payload: TaskCheckpointPayload =
                    serde_json::from_value(snapshot.payload.clone())?;
                if payload.config_identity != self.scope.identity() {
                    continue;
                }
                payload.validate(&snapshot, self.scope.identity())?;
                if payload.settings.persistence.as_ref().is_some_and(|policy| {
                    policy.enabled != Some(false) && policy.resume_on_start == Some(true)
                }) {
                    return Err(AgentError::Config(format!(
                        "task {} is owned or needs recovery; inspect it before startup",
                        row.key.run_id
                    )));
                }
            }
        }
        let selected = AutonomyRunner::select_startup(
            TaskStartupRequest {
                config: self.prepared.spec().autonomy.clone(),
                ceilings: self.ceilings.clone(),
                config_identity: self.scope.identity().into(),
                profile: None,
            },
            store.as_ref(),
            self,
        )
        .await?;
        let Some(selected) = selected else {
            return Ok(None);
        };
        let snapshot = store
            .load(&selected.run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != selected.expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        Ok(Some(snapshot.summary()))
    }

    /// Uses the same unique-candidate selection and exact-revision reconstruction as an explicit configured resume.
    pub async fn restore_on_start(
        self: &Arc<Self>,
    ) -> Result<Option<(Arc<AutonomyRunner>, TaskRunResult)>> {
        let Some(candidate) = self.startup_candidate().await? else {
            return Ok(None);
        };
        self.restore(candidate.key.run_id, candidate.revision)
            .await
            .map(Some)
    }

    // Fingerprints describe only host-prepared configuration; executable authority remains the installed factory and actual objects.
    fn binding_configuration(
        &self,
        prepared: &PreparedAgentSpec,
        base_dir: &std::path::Path,
    ) -> Result<Value> {
        let definition = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&json!({
                "spec": prepared.spec().identity_value()?, "policy": prepared.task_iteration_policy(),
            }))?)
        );
        Ok(json!({"host": self.configuration, "definition": definition, "base_dir": base_dir}))
    }

    // Resolve the inherited provider declarations from the same spawner binding used during reconstruction.
    fn restore_provider_spec(
        &self,
        manifest: &TaskRestoreManifest,
        prepared: PreparedAgentSpec,
        membership: Option<&(String, String)>,
    ) -> Result<PreparedAgentSpec> {
        let Some((spawner, _)) = membership else {
            return Ok(prepared);
        };
        if manifest
            .spawners
            .get(spawner)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?["shared"]
            .as_bool()
            != Some(true)
        {
            return Ok(prepared);
        }
        let (owner, _) = manifest
            .nodes
            .iter()
            .find(|(_, node)| node.spawner.as_ref() == Some(spawner))
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        self.source(manifest, owner).map(|(spec, _, _)| spec)
    }

    // An old tool batch does not record the selected alias or discarded thinking blocks.
    // Refuse unsigned continuation when any configured candidate could have produced a signed exchange.
    fn validate_anthropic_continuation(
        spec: &crate::spec::AgentSpec,
        runtime: &TaskRuntimeCheckpoint,
    ) -> Result<()> {
        let TaskContinuation::Suspended {
            batch: Some(batch), ..
        } = &runtime.continuation
        else {
            return Ok(());
        };
        let needs_signature = |config: &crate::spec::LLMConfig| {
            let Ok(provider) = config
                .provider
                .parse::<ai_agents_llm::providers::ProviderType>()
            else {
                return false;
            };
            provider.requires_signed_default_anthropic_history(
                &config.model,
                &config.to_provider_config(provider),
            )
        };
        let requires_signature = if spec.llms.is_empty() {
            spec.llm.as_config().is_some_and(needs_signature)
        } else {
            spec.llms.values().any(needs_signature)
        };
        if requires_signature {
            for message in &batch.messages {
                if message.role == ai_agents_core::Role::Assistant
                    && let Some(calls) =
                        ai_agents_core::decode_native_tool_call_markers(&message.content)?
                    && calls.provider_state().is_none()
                {
                    return Err(AgentError::Config(
                        "cannot restore unsigned legacy tool history for an Anthropic model that may require signed thinking blocks".into()
                    ));
                }
            }
        }
        Ok(())
    }

    // Resolves recorded child declarations through the saved topology, then checks any current declarative source against them.
    fn source(&self, manifest: &TaskRestoreManifest, id: &str) -> Result<NodeSource> {
        if id == manifest.root {
            return Ok((self.prepared.clone(), self.base_dir.clone(), None));
        }
        let mut found: Option<NodeSource> = None;
        for node in manifest.nodes.values() {
            let Some(registry_key) = &node.registry else {
                continue;
            };
            let registry = manifest
                .registries
                .get(registry_key)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            for declaration in node.runtime.snapshot.spawned_agents.iter().flatten() {
                let Some(entry) = registry
                    .entries
                    .get(&declaration.id)
                    .filter(|entry| entry.runtime == id)
                else {
                    continue;
                };
                let saved = PreparedAgentSpec::from_persistence_yaml(&declaration.spec_yaml)?;
                let directory =
                    if let Some((current, directory)) = self.declared.get(&declaration.id) {
                        if current.spec().identity_value()? != saved.spec().identity_value()?
                            || current.task_iteration_policy() != saved.task_iteration_policy()
                        {
                            return Err(AgentError::Config(
                                "declared child configuration changed".into(),
                            ));
                        }
                        directory.clone()
                    } else {
                        self.working_dir.clone()
                    };
                let spawner = entry.spawner.clone().ok_or_else(|| {
                    AgentError::Config(
                        "custom child requires an installed restoration factory".into(),
                    )
                })?;
                let candidate = (saved, directory, Some((spawner, declaration.id.clone())));
                if let Some((previous, base, _)) = &found
                    && self.binding_configuration(previous, base)?
                        != self.binding_configuration(&candidate.0, &candidate.1)?
                {
                    return Err(AgentError::Config(
                        "child aliases disagree on construction".into(),
                    ));
                }
                found = Some(candidate);
            }
        }
        found.ok_or_else(|| {
            AgentError::Config("restored child has no configured declaration".into())
        })
    }
}

#[async_trait]
impl TaskRestoreHost for ConfiguredTaskRestoreHost {
    /// Actual selected persistence requires a surviving journal and fresh OS evidence before semantic work starts.
    fn preflight_persistence(&self, process_durable: Option<bool>) -> Result<()> {
        if !matches!(
            self.prepared.spec().storage,
            crate::spec::StorageConfig::Sqlite(_)
        ) || process_durable == Some(false)
        {
            return Err(AgentError::Config(
                "configured persistence requires file-backed durable SQLite task storage".into(),
            ));
        }
        self.sample_clock(None)?;
        self.validate_resource_contracts()
    }

    /// Excludes other configured work on this actual run through the complete capture, restore or review transaction.
    fn work_lease(
        &self,
        snapshot: &TaskRunSnapshot,
    ) -> Result<Option<Box<dyn TaskRestoreWorkLease>>> {
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(snapshot, self.scope.identity())?;
        if snapshot.key.agent_id != self.prepared.spec().name
            || snapshot.actor_id.as_deref() != self.scope.actor_id()
        {
            return Err(AgentError::Config(
                "task lease scope differs from configured host".into(),
            ));
        }
        Ok(Some(Box::new(
            self.acquire_work_lease(&snapshot.key.run_id)?,
        )))
    }
    /// Reopens the configured SQLite namespace and checks the actual saved receipt, rather than inferring durability from a capability bit.
    async fn checkpoint_survives_exit(&self, snapshot: &TaskRunSnapshot) -> Result<bool> {
        let stored = self
            .scope
            .open_store()
            .await?
            .load(&snapshot.key.run_id)
            .await?;
        Ok(stored.map(serde_json::to_value).transpose()? == Some(serde_json::to_value(snapshot)?))
    }
    /// Identifies the installed factory implementation rather than a user-selected component label.
    fn identity(&self) -> (String, u32) {
        ("configured-yaml-host".into(), 2)
    }

    /// Each reconstruction transaction keeps the explicitly configured positive work bound.
    fn reconstruction_budget(&self) -> Duration {
        self.budget
    }

    /// Admission uses the later possible UTC so uncertainty never creates additional remaining time.
    fn now(&self) -> Result<chrono::DateTime<chrono::Utc>> {
        Ok(self.sample_clock(None)?.latest)
    }

    /// Capture retains actual synchronization evidence rather than treating a readable wall clock as proof.
    fn clock_evidence(&self) -> Result<Option<TaskUtcEvidence>> {
        self.sample_clock(None).map(Some)
    }

    /// Captures only first-party construction receipts and exact mutable framework state.
    fn runtime_binding(
        &self,
        runtime: &RuntimeAgent,
        _declaration: Option<&crate::spec::AgentSpec>,
    ) -> Result<TaskRestoreBinding> {
        let recipe = runtime.configured_task_recipe()?;
        Ok(TaskRestoreBinding {
            id: "configured-yaml-runtime".into(),
            contract_version: 1,
            configuration: self
                .binding_configuration(&recipe.source.prepared, &recipe.source.base_dir)?,
            state: serde_json::to_value(runtime.configured_task_state()?)?,
        })
    }

    /// Uses actual captured provider handles to identify the native configured factory, including shared child registries.
    fn provider_binding(
        &self,
        providers: &ai_agents_llm::LLMRegistry,
    ) -> Result<TaskRestoreBinding> {
        let instances = self.instances.lock();
        let instance = instances
            .iter()
            .filter_map(|known| known.upgrade())
            .find(|known| known.task_restore_providers().same_bindings(providers))
            .ok_or_else(|| {
                AgentError::Config(
                    "provider registry lacks a configured construction receipt".into(),
                )
            })?;
        let recipe = instance.configured_task_recipe()?;
        Ok(TaskRestoreBinding {
            id: "configured-yaml-providers".into(),
            contract_version: 1,
            configuration: self
                .binding_configuration(&recipe.source.prepared, &recipe.source.base_dir)?,
            state: Value::Null,
        })
    }

    /// Discovers the actual source graph and validates private construction receipts before recording provider identities.
    fn capture_graph(&self, root: Arc<RuntimeAgent>) -> Result<TaskRestoreGraph> {
        let graph = TaskRestoreGraph::from_root(root)?;
        for runtime in graph.runtimes.values() {
            runtime.configured_task_recipe()?;
            self.remember(runtime);
        }
        Ok(graph)
    }

    /// Pure preflight compares current host configuration and certified time before ownership is claimed.
    fn validate_manifest(&self, manifest: &TaskRestoreManifest) -> Result<()> {
        self.validate_resource_contracts()?;
        if manifest.version != 3
            || manifest.host != self.identity()
            || manifest.root != self.prepared.spec().name
        {
            return Err(AgentError::Config(
                "configured restoration recipe differs".into(),
            ));
        }
        for (id, node) in &manifest.nodes {
            let (prepared, base_dir, membership) = self.source(manifest, id)?;
            if node.binding.id != "configured-yaml-runtime"
                || node.binding.contract_version != 1
                || node.binding.configuration != self.binding_configuration(&prepared, &base_dir)?
            {
                return Err(AgentError::Config(
                    "configured runtime definition differs".into(),
                ));
            }
            let providers =
                self.restore_provider_spec(manifest, prepared.clone(), membership.as_ref())?;
            Self::validate_anthropic_continuation(providers.spec(), &node.runtime)?;
            let state: ConfiguredTaskState = serde_json::from_value(node.binding.state.clone())?;
            match (&self.components, &state.components) {
                (Some(factory), Some(binding)) => {
                    if factory.identity() != (binding.id.clone(), binding.contract_version) {
                        return Err(AgentError::Config(
                            "component implementation identity differs".into(),
                        ));
                    }
                    factory.validate(&prepared, binding)?;
                }
                (None, None) => {}
                _ => {
                    return Err(AgentError::Config(
                        "component restoration binding is missing or unexpected".into(),
                    ));
                }
            }
        }
        let previous = manifest.clock_evidence.as_ref().ok_or_else(|| {
            AgentError::Config("configured restoration lacks OS clock evidence".into())
        })?;
        self.sample_clock(Some(previous))?;
        Ok(())
    }

    /// Prepares actual configured resources under the engine's existing charged reconstruction transaction.
    /// Children use the same spawner constructor but no auto-spawn, capacity allocation or registry callback is replayed.
    async fn prepare_graph(
        &self,
        manifest: &TaskRestoreManifest,
    ) -> Result<Option<TaskRestoreGraph>> {
        self.validate_manifest(manifest)?;
        let state = serde_json::from_value(manifest.nodes[&manifest.root].binding.state.clone())?;
        let agent = self
            .builder(Some(state))?
            .auto_configure_mcp()
            .await?
            .prepare_restore_spawner()
            .await?
            .build_task_host()
            .await?;
        agent.set_question_handler(self.questions.clone());
        if let Some(actor) = self.scope.actor_id() {
            agent.set_actor_id(actor)?;
        }
        self.remember(&agent);
        let mut graph = TaskRestoreGraph::from_root(Arc::new(agent))?;
        for (id, node) in &manifest.nodes {
            if id == &manifest.root {
                continue;
            }
            let (prepared, directory, membership) = self.source(manifest, id)?;
            let (spawner_key, entry) = membership.ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            let spawner = graph
                .spawners
                .get(&spawner_key)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            let state = serde_json::from_value(node.binding.state.clone())?;
            let declaration = prepared.spec().clone();
            let child = Arc::new(
                spawner
                    .construct_child(&entry, prepared, Some(directory), Some(state))
                    .await?,
            );
            if child.info().id != *id {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            self.remember(&child);
            graph
                .providers
                .insert(format!("runtime:{id}"), child.task_restore_providers());
            graph.declarations.insert(id.clone(), declaration);
            graph.runtimes.insert(id.clone(), child);
        }
        for (operation, composition) in &manifest.compositions {
            if let Some(key) = &composition.providers {
                let (runtime, _) = manifest
                    .tools
                    .get(operation)
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                let providers = graph
                    .runtimes
                    .get(runtime)
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                    .task_restore_providers();
                graph.providers.insert(key.clone(), providers);
            }
        }
        Ok(Some(graph))
    }

    /// This host prepares resources explicitly; callers must enter through the framework reconstruction transaction.
    async fn stage(&self, _manifest: &TaskRestoreManifest) -> Result<TaskRestoreGraph> {
        Err(AgentError::Config(
            "configured restoration requires admitted resource preparation".into(),
        ))
    }
}

// Only the configured preparer's lease and an unchanged exact snapshot can back this recovery proof.
struct ConfiguredRecoveryProof {
    host: Arc<ConfiguredTaskRestoreHost>,
    source: Value,
    _lease: super::restore_lease::TaskFileLease,
}

impl TaskRecoveryHost for ConfiguredRecoveryProof {
    /// Validates local preparation exclusion and rejects semantic execution or uncertified component cleanup.
    fn validate(&self, snapshot: &TaskRunSnapshot, decision: &TaskRecoveryDecision) -> Result<()> {
        if serde_json::to_value(snapshot)? != self.source
            || decision.expected_revision != snapshot.revision
            || !decision.effects.is_empty()
            || !decision.children.is_empty()
        {
            return Err(AgentError::Config(
                "recovery proof does not match the observed preparation".into(),
            ));
        }
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(snapshot, self.host.scope.identity())?;
        let work = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.restore_work")
            .ok_or(TaskRunStorageError::NotResumable)?;
        if work.state["phase"] != "running"
            || !matches!(
                work.state["kind"].as_str(),
                Some("capture" | "restore" | "review")
            )
            || payload.clocks.active_interval_started_at.is_some()
        {
            return Err(AgentError::Config("interrupted execution or missing preparation start needs separate recovery evidence".into()));
        }
        let adapter = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.restore")
            .ok_or(TaskRunStorageError::NotResumable)?;
        let manifest: TaskRestoreManifest = serde_json::from_value(adapter.state.clone())?;
        self.host.validate_manifest(&manifest)?;
        if let Some(components) = &self.host.components {
            components.validate_interrupted_preparation(snapshot)?;
        }
        Ok(())
    }

    /// Interrupted work is charged through a certified upper UTC bound rather than an assumed process-death timestamp.
    fn now(&self) -> Result<chrono::DateTime<chrono::Utc>> {
        self.host.now()
    }

    /// Recovery uses the same explicit host-wide work ceiling as the preparation it reconciles.
    fn recovery_budget(&self) -> Duration {
        self.host.budget
    }
}
