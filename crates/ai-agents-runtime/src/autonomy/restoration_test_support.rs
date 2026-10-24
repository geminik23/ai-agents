//! Existing production fixtures rebuild their real runtimes and stateful providers without retaining the source runtime or owner.

pub(super) struct MemoryGate {
    target: String,
    phase: &'static str,
    pub armed: std::sync::atomic::AtomicBool,
    pub reached: tokio::sync::Semaphore,
}
impl MemoryGate {
    // Each runtime receives separate real memory; only the selected awaited component boundary is held for custody tests.
    pub fn new(target: &str, phase: &'static str) -> Arc<Self> {
        Arc::new(Self {
            target: target.into(),
            phase,
            armed: std::sync::atomic::AtomicBool::new(false),
            reached: tokio::sync::Semaphore::new(0),
        })
    }
    // Every reconstruction receives independent memory, while the host-controlled barrier selects one exact component await.
    pub fn memory(self: &Arc<Self>, id: String) -> Arc<dyn ai_agents_memory::Memory> {
        Arc::new(GatedMemory {
            inner: ai_agents_memory::InMemoryStore::new(1000),
            gate: self.clone(),
            id,
        })
    }
    // A one-shot semaphore witnesses the selected boundary before cancellation, without using timing to infer execution.
    async fn hold(&self, id: &str, phase: &str) {
        if self.target == id
            && self.phase == phase
            && self.armed.swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            self.reached.add_permits(1);
            std::future::pending::<()>().await;
        }
    }
}
struct GatedMemory {
    inner: ai_agents_memory::InMemoryStore,
    gate: Arc<MemoryGate>,
    id: String,
}
#[async_trait]
impl ai_agents_memory::Memory for GatedMemory {
    // Exact delegated provenance and no hidden provider calls preserve task admission through the test barrier.
    fn supports_task_admission(&self) -> bool {
        true
    }
}
#[async_trait]
impl ai_agents_core::Memory for GatedMemory {
    // Forward exact provenance without introducing hidden semantic work.
    async fn add_message(&self, message: ai_agents_core::ChatMessage) -> Result<()> {
        self.inner.add_message(message).await
    }
    // Reads use the real memory projection, not an alternate test history.
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ai_agents_core::ChatMessage>> {
        self.inner.get_messages(limit).await
    }
    // Reset remains the delegate's normal operation and never releases task ownership.
    async fn clear(&self) -> Result<()> {
        self.inner.clear().await
    }
    // Memory admission sees the delegate's actual retained history size.
    fn len(&self) -> usize {
        self.inner.len()
    }
    // Hold the read-only capture boundary before returning the exact snapshot.
    async fn snapshot(&self) -> Result<ai_agents_core::MemorySnapshot> {
        self.gate.hold(&self.id, "snapshot").await;
        self.inner.snapshot().await
    }
    // Hold only staged component installation, allowing tests to cancel after owners have been reserved.
    async fn restore(&self, snapshot: ai_agents_core::MemorySnapshot) -> Result<()> {
        self.gate.hold(&self.id, "restore").await;
        self.inner.restore(snapshot).await
    }
    // Native-history retention continues through the production memory implementation.
    async fn evict_oldest(&self, count: usize) -> Result<Vec<ai_agents_core::ChatMessage>> {
        self.inner.evict_oldest(count).await
    }
}

use super::runner_tests::RecordingProvider;
use super::*;
use crate::RuntimeAgent;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{Arc, Weak},
};

pub(super) struct FreshRuntime {
    pub agent: Arc<RuntimeAgent>,
    pub providers: Vec<Arc<RecordingProvider>>,
    children: Vec<(
        Arc<RuntimeAgent>,
        Vec<Arc<RecordingProvider>>,
        crate::spec::AgentSpec,
    )>,
}
impl FreshRuntime {
    // Factories reuse the owning production fixture and provide only host construction and opaque provider state.
    pub fn new(agent: Arc<RuntimeAgent>, providers: Vec<Arc<RecordingProvider>>) -> Self {
        Self {
            agent,
            providers,
            children: vec![],
        }
    }
    // Explicit child construction supplies declarations but does not replay registry membership or lifecycle hooks.
    pub fn with_child(
        mut self,
        agent: Arc<RuntimeAgent>,
        providers: Vec<Arc<RecordingProvider>>,
        spec: crate::spec::AgentSpec,
    ) -> Self {
        self.children.push((agent, providers, spec));
        self
    }
    // Actual runtime/provider pairs keep hidden queues bound to their owning node rather than their display aliases.
    fn nodes(&self) -> impl Iterator<Item = (&Arc<RuntimeAgent>, &Vec<Arc<RecordingProvider>>)> {
        std::iter::once((&self.agent, &self.providers)).chain(
            self.children
                .iter()
                .map(|(runtime, providers, _)| (runtime, providers)),
        )
    }
    // Empty staged registries are wired to real tools; only the framework publishes the saved membership.
    fn graph(&self) -> Result<TaskRestoreGraph> {
        let mut graph = TaskRestoreGraph::from_root(self.agent.clone())?;
        for (runtime, _, spec) in &self.children {
            let id = runtime.info().id;
            graph.runtimes.insert(id.clone(), runtime.clone());
            graph
                .providers
                .insert(format!("runtime:{id}"), runtime.task_restore_providers());
            graph.declarations.insert(id, spec.clone());
        }
        Ok(graph)
    }
}
type Factory = dyn Fn() -> Pin<Box<dyn Future<Output = FreshRuntime> + Send>> + Send + Sync;
type Providers = BTreeMap<usize, Vec<Weak<RecordingProvider>>>;

/// Test integration reconstructs established fixtures; the framework still owns topology, state, clocks and execution admission.
pub(super) struct RebuildHost {
    recipe: &'static str,
    factory: Box<Factory>,
    nodes: std::collections::BTreeSet<String>,
    providers: parking_lot::Mutex<Providers>,
    pub staged: parking_lot::Mutex<Weak<RuntimeAgent>>,
    pub clock: parking_lot::Mutex<Option<chrono::DateTime<chrono::Utc>>>,
    epoch: chrono::DateTime<chrono::Utc>,
    started: std::time::Instant,
}
impl RebuildHost {
    // Weak indexes witness actual implementations; one certified local UTC anchor and monotonic elapsed time keep positive fixtures independent of later OS clock rollback.
    pub fn new(recipe: &'static str, source: &FreshRuntime, factory: Box<Factory>) -> Arc<Self> {
        let started = std::time::Instant::now();
        let epoch = chrono::Utc::now();
        let host = Arc::new(Self {
            recipe,
            factory,
            nodes: source
                .nodes()
                .map(|(runtime, _)| runtime.info().id)
                .collect(),
            providers: Default::default(),
            staged: Default::default(),
            clock: Default::default(),
            epoch,
            started,
        });
        host.register(source);
        host
    }

    // The stable recipe covers builder/extension behavior while saved queues cover the custom provider's hidden mutable behavior.
    fn register(&self, fresh: &FreshRuntime) {
        for (runtime, providers) in fresh.nodes() {
            self.providers.lock().insert(
                Arc::as_ptr(runtime) as usize,
                providers.iter().map(Arc::downgrade).collect(),
            );
        }
    }
}
#[async_trait]
impl TaskRestoreHost for RebuildHost {
    // This identity versions installed fixture code, never a model-selected executable capability.
    fn identity(&self) -> (String, u32) {
        (self.recipe.into(), 1)
    }
    // Capture, reconstruction and review each retain the same positive bounded work contract.
    fn reconstruction_budget(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }
    // This same-process fixture certifies continuity from its fixed UTC epoch, not repeated unchecked OS samples.
    // Deliberate overrides still exercise rollback rejection; no work budget or original request deadline is widened.
    fn now(&self) -> Result<chrono::DateTime<chrono::Utc>> {
        if let Some(clock) = *self.clock.lock() {
            return Ok(clock);
        }
        let elapsed = chrono::Duration::from_std(self.started.elapsed())
            .map_err(|_| AgentError::Config("fixture clock elapsed overflow".into()))?;
        self.epoch
            .checked_add_signed(elapsed)
            .ok_or_else(|| AgentError::Config("fixture clock projection overflow".into()))
    }
    // Behavioral queue state is opaque host data; invocation counters are not execution state and are not restored.
    fn runtime_binding(
        &self,
        runtime: &RuntimeAgent,
        _: Option<&crate::spec::AgentSpec>,
    ) -> Result<TaskRestoreBinding> {
        let queues = self
            .providers
            .lock()
            .get(&(runtime as *const RuntimeAgent as usize))
            .ok_or_else(|| AgentError::Config("unregistered fixture runtime".into()))?
            .iter()
            .map(|provider| {
                provider
                    .upgrade()
                    .ok_or_else(|| AgentError::Config("fixture provider was dropped".into()))
                    .map(|provider| provider.restoration_state())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TaskRestoreBinding {
            id: self.recipe.into(),
            contract_version: 1,
            configuration: json!({"recipe":self.recipe,"provider_contract":"stateful-queue-v1"}),
            state: json!(queues),
        })
    }
    // The recipe and per-runtime queue binding jointly cover the actual fixture providers, not merely provider names.
    fn provider_binding(&self, _: &ai_agents_llm::LLMRegistry) -> Result<TaskRestoreBinding> {
        // The recipe installs the actual queue providers; runtime_binding covers their state and the framework verifies installed handles.
        Ok(TaskRestoreBinding {
            id: self.recipe.into(),
            contract_version: 1,
            configuration: json!({"recipe":self.recipe,"providers":"recording-v1"}),
            state: Value::Null,
        })
    }
    // Reject an incompatible immutable recipe or node inventory before the framework attempts a claim.
    fn validate_manifest(&self, manifest: &TaskRestoreManifest) -> Result<()> {
        if manifest.host != self.identity()
            || !manifest.nodes.keys().eq(self.nodes.iter())
            || manifest.nodes.values().any(|node| {
                node.binding.id != self.recipe
                    || node.binding.contract_version != 1
                    || node.binding.configuration
                        != json!({"recipe":self.recipe,"provider_contract":"stateful-queue-v1"})
            })
        {
            return Err(AgentError::Config(
                "incompatible fixture reconstruction recipe".into(),
            ));
        }
        Ok(())
    }
    // Rebuild using the owning fixture, install saved queues only into fresh providers, and leave execution to existing drivers.
    async fn stage(&self, manifest: &TaskRestoreManifest) -> Result<TaskRestoreGraph> {
        let fresh = (self.factory)().await;
        for (runtime, providers) in fresh.nodes() {
            let queues: Vec<Value> =
                serde_json::from_value(manifest.nodes[&runtime.info().id].binding.state.clone())?;
            if queues.len() != providers.len() {
                return Err(AgentError::Config(
                    "provider queue inventory differs".into(),
                ));
            }
            for (provider, queue) in providers.iter().zip(queues) {
                provider.install_restoration_state(queue)?;
            }
        }
        self.register(&fresh);
        *self.staged.lock() = Arc::downgrade(&fresh.agent);
        fresh.graph()
    }
}

// Explicit reconstruction never consumes the pending response; acceptance must approve restoration and answer separately.
pub(super) async fn restore(
    runner: AutonomyRunner,
    store: Arc<dyn TaskRunStore>,
    run: &TaskRun,
    config: AutonomyConfig,
    identity: &str,
    host: Arc<dyn TaskRestoreHost>,
) -> (AutonomyRunner, TaskRunResult) {
    let prepared = runner
        .checkpoint_for_restore(&run.key.run_id, run.revision, host.as_ref())
        .await
        .unwrap();
    drop(runner);
    let (restored, paused) = AutonomyRunner::restore_paused(
        TaskRestoreRequest {
            run_id: prepared.run.key.run_id,
            expected_revision: prepared.run.revision,
            config_identity: identity.into(),
            config,
            ceilings: Default::default(),
        },
        store,
        host,
    )
    .await
    .unwrap();
    let review = restored
        .restoration_review(&paused.run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    let reviewed = restored
        .approve_restoration(
            &paused.run.key.run_id,
            paused.run.revision,
            &review.id,
            ai_agents_hitl::ApprovalResult::Approved,
        )
        .await
        .unwrap();
    (restored, reviewed)
}
