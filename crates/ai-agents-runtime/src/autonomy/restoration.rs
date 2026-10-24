//! Explicit host-authorized reconstruction of a safe paused task, separate from ordinary session restore and response consumption.

use super::*;
use crate::{
    RuntimeAgent,
    spawner::{AgentRegistry, AgentSpawner},
    spec::AgentSpec,
};
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

#[path = "restoration_bindings.rs"]
mod bindings;
#[path = "restoration_driver.rs"]
mod driver;

pub(crate) const RESTORE_ADAPTER: &str = "runtime.restore";
pub(crate) const RESTORE_REVIEW: &str = "runtime.restore_review";
pub(crate) const RESTORE_WORK: &str = "runtime.restore_work";

/// An installed host's versioned, non-secret configuration and opaque component-state binding.
/// The host must cover actual provider/model/tool/adapter configuration and hidden service state, not just display names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreBinding {
    pub id: String,
    pub contract_version: u32,
    pub configuration: Value,
    pub state: Value,
}

/// Actual staged objects supplied by trusted Rust integration, never deserialized as executable authority.
/// Registries must be empty for staging; the framework installs their validated membership without replaying lifecycle hooks.
#[derive(Default)]
pub struct TaskRestoreGraph {
    pub runtimes: BTreeMap<String, Arc<RuntimeAgent>>,
    pub registries: BTreeMap<String, Arc<AgentRegistry>>,
    pub spawners: BTreeMap<String, Arc<AgentSpawner>>,
    pub providers: BTreeMap<String, Arc<ai_agents_llm::LLMRegistry>>,
    pub declarations: BTreeMap<String, AgentSpec>,
}

/// Host-installed reconstruction and compatibility authority; model-selected IDs cannot install this service.
/// Capture/validation perform no semantic work or hidden I/O. Staging constructs unpublished objects without providers, tools or lifecycle-hook effects.
/// Binding implementations and configured native-history destinations must be checked by the host before returning staged objects.
/// Common behavior remains declarative; this service binds trusted host implementations and hidden state, not task objectives or domain completion logic.
/// Immutable configuration must distinguish different implementations even when their display names match; mutable state may advance only through admitted work.
#[async_trait]
pub trait TaskRestoreHost: Send + Sync {
    /// Stable host contract identity, distinct from the task profile's compatibility identity.
    fn identity(&self) -> (String, u32);
    /// Host-selected maximum active work for each capture/reconstruction/review transaction.
    /// The whole bound is charged conservatively before publication, including possible storage/ACK waits; it never widens task limits.
    fn reconstruction_budget(&self) -> std::time::Duration;
    /// Covers the actual runtime's prepared implementations and opaque service state; secrets stay in host configuration.
    fn runtime_binding(
        &self,
        runtime: &RuntimeAgent,
        declaration: Option<&AgentSpec>,
    ) -> Result<TaskRestoreBinding>;
    /// Covers an actual captured auxiliary provider registry, including model/request-class configuration and hidden state.
    fn provider_binding(
        &self,
        providers: &ai_agents_llm::LLMRegistry,
    ) -> Result<TaskRestoreBinding>;
    /// Resolves supported immutable configuration references and native-history compatibility without constructing providers or mutating topology.
    fn validate_manifest(&self, manifest: &TaskRestoreManifest) -> Result<()>;
    /// Supplies the actual source topology; common root/spawner catalogues are discovered without custom code.
    fn capture_graph(&self, root: Arc<RuntimeAgent>) -> Result<TaskRestoreGraph> {
        TaskRestoreGraph::from_root(root)
    }
    /// Constructs fresh unpublished runtimes and empty registries with the recorded alias relationships and host bindings.
    async fn stage(&self, manifest: &TaskRestoreManifest) -> Result<TaskRestoreGraph>;
    /// The installed host certifies a trustworthy UTC epoch/continuity for stored bounds across processes and returns an error when it cannot.
    /// Sampling an unchecked OS clock alone is not a continuity proof; backward anchors are rejected and original monotonic cuts are preserved.
    /// Returned UTC must share the backend timestamp domain; projecting an independent old epoch cannot certify continuity with later backend writes.
    fn now(&self) -> Result<DateTime<Utc>>;
}

/// One authoritative current runtime image, distinct from immutable historical per-operation snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreNode {
    pub binding: TaskRestoreBinding,
    pub configuration_digest: String,
    pub runtime: TaskRuntimeCheckpoint,
    pub session: Value,
    pub registry: Option<String>,
    pub spawner: Option<String>,
    pub generations: TaskRestoreGenerations,
    pub participant_actor: Option<TaskRestoreParticipant>,
    pub skill: Option<ai_agents_skills::SkillDefinition>,
}

/// Participation is encoded independently of optional actor identity, so actorless members cannot become unstarted nodes during JSON round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreParticipant {
    pub actor_id: Option<String>,
}

/// Generation values identify the old binding for validation only; reconstruction binds to fresh actual generations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreGenerations {
    pub state: Option<u64>,
    pub control: u64,
    pub policy: u64,
    pub tools: u64,
    pub epoch: u64,
}

/// Registry membership records alias relationships without granting permission to execute a child.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreRegistry {
    pub send_with_context: bool,
    pub entries: BTreeMap<String, TaskRestoreEntry>,
}

/// A managed entry keeps its original capacity owner; an untracked host entry does not mint a reservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreEntry {
    pub spawned_at: DateTime<Utc>,
    pub runtime: String,
    pub spawner: Option<String>,
}

/// The same registry membership may back many distinct incoming operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreTarget {
    pub registry: String,
    pub entry: String,
    pub runtime: String,
}

/// Captured composition implementations can use registries/providers other than the root's own handles.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreComposition {
    pub registry: String,
    pub providers: Option<String>,
}

/// Original response binding data is audit input; only fresh reconstruction review enables its new live guards.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreResponse {
    pub runtime: String,
    pub request: ai_agents_core::ToolExecutionRequest,
    pub expires_at: Option<DateTime<Utc>>,
    pub applied: bool,
    pub timeout_disposition: bool,
}

/// Versioned immutable topology and exact state bound to one explicitly prepared safe checkpoint.
/// This is sensitive recovery data, not a listing/event payload or an arbitrary Rust-object serializer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestoreManifest {
    pub version: u32,
    pub host: (String, u32),
    pub revision: u64,
    pub payload_digest: String,
    pub captured_at: DateTime<Utc>,
    pub whole_expiry: Option<DateTime<Utc>>,
    pub deadlines: BTreeMap<String, DateTime<Utc>>,
    pub validation_bound: Option<(EvidenceIdentity, DateTime<Utc>)>,
    pub root: String,
    pub nodes: BTreeMap<String, TaskRestoreNode>,
    pub registries: BTreeMap<String, TaskRestoreRegistry>,
    pub spawners: BTreeMap<String, Value>,
    pub providers: BTreeMap<String, TaskRestoreBinding>,
    pub targets: BTreeMap<String, TaskRestoreTarget>,
    pub compositions: BTreeMap<String, TaskRestoreComposition>,
    pub message_contexts: BTreeMap<String, Value>,
    pub tools: BTreeMap<String, (String, String)>,
    pub responses: BTreeMap<String, TaskRestoreResponse>,
    pub operation_ordinals: BTreeMap<String, u64>,
    pub skill_finishing: Vec<String>,
    pub active_frames: Vec<String>,
}

/// Explicit reconstruction input; response data is deliberately absent and must be supplied separately after fresh review.
pub struct TaskRestoreRequest {
    pub run_id: String,
    pub expected_revision: u64,
    pub config_identity: String,
    pub config: AutonomyConfig,
    pub ceilings: AutonomyHostCeilings,
}

/// A fresh, exact restoration review is separate from answering the task's original pending question/approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRestorationReview {
    pub id: String,
    pub approved: bool,
    pub action_digest: String,
}

/// Opaque framework-created mutation proof; custom stores may forward it but cannot construct arbitrary reconstruction successors.
pub struct TaskReconstructionCheckpoint {
    pub(crate) previous: TaskRunSnapshot,
    pub(crate) mutation: TaskRunMutation,
}

/// Canonical JSON fingerprints bind exact recovery data without duplicating sensitive configuration in diagnostics.
pub(crate) fn restore_digest(value: &impl Serialize) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

/// A prepared image becomes stale after any task progress; a new explicit checkpoint is required before another restart.
fn payload_digest(payload: &TaskCheckpointPayload) -> Result<String> {
    let mut value = serde_json::to_value(payload)?;
    value["adapters"]
        .as_array_mut()
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?
        .retain(|adapter| adapter["id"] != RESTORE_ADAPTER && adapter["id"] != RESTORE_REVIEW);
    restore_digest(&value)
}

// A review binds the entire prepared host/configuration manifest and executable action set, excluding only its own transport metadata.
fn review_digest(payload: &TaskCheckpointPayload) -> Result<String> {
    let mut value = serde_json::to_value(payload)?;
    value["adapters"]
        .as_array_mut()
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?
        .retain(|adapter| adapter["id"] != RESTORE_REVIEW && adapter["id"] != RESTORE_WORK);
    // Charge-only clock updates and capture timestamps do not change the operation/configuration being reviewed.
    value["clocks"]
        .as_object_mut()
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?
        .remove("active_millis");
    for adapter in value["adapters"]
        .as_array_mut()
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?
    {
        if adapter["id"] == RESTORE_ADAPTER {
            let state = adapter["state"]
                .as_object_mut()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            state.remove("revision");
            state.remove("payload_digest");
            if let Some(nodes) = state.get_mut("nodes").and_then(Value::as_object_mut) {
                for node in nodes.values_mut() {
                    if let Some(snapshot) = node["runtime"]["snapshot"].as_object_mut() {
                        snapshot.remove("timestamp");
                    }
                }
            }
        }
    }
    restore_digest(&value)
}

/// Requires a truly safe acknowledged pause; claims, cancellation receipts and unknown effects are not quiescence.
/// Backward UTC is a separate clock-contract failure, never fresh allowance or ordinary expiry.
fn safe_pause(
    snapshot: &TaskRunSnapshot,
    payload: &TaskCheckpointPayload,
    now: DateTime<Utc>,
) -> Result<()> {
    if payload
        .adapters
        .iter()
        .find(|adapter| adapter.id == RESTORE_WORK)
        .is_some_and(|adapter| adapter.state["phase"] != "paused")
    {
        return Err(AgentError::Config(
            "reconstruction work requires explicit clock/effect reconciliation".into(),
        ));
    }
    if snapshot.status != TaskRunStatus::Paused
        || snapshot.owner_token.is_some()
        || snapshot.cancel_requested
        || payload.clocks.active_interval_started_at.is_some()
        || payload.reservations.iter().any(|reservation| {
            matches!(
                reservation.state,
                TaskEffectState::Dispatched | TaskEffectState::Uncertain
            )
        })
        || payload
            .children
            .iter()
            .any(|child| child.result.is_none() && child.pending.is_none())
    {
        return Err(TaskRunStorageError::NotResumable.into());
    }
    // Metadata high-water marks reveal backward UTC; this is not exhaustion of a valid original allowance.
    if now < snapshot.updated_at {
        return Err(TaskRunStorageError::ClockDiscontinuity.into());
    }
    if payload
        .clocks
        .expires_at
        .is_some_and(|expiry| now >= expiry)
    {
        return Err(TaskRunStorageError::AdmissionExpired.into());
    }
    Ok(())
}

impl TaskRestoreGraph {
    /// Discovers common declared/generated topology through actual runtime handles; custom tool catalogues can extend this graph.
    pub fn from_root(root: Arc<RuntimeAgent>) -> Result<Self> {
        let mut graph = Self::default();
        graph.runtimes.insert(root.info().id, root);
        graph.expand()?;
        Ok(graph)
    }

    // Stable manifest keys group actual source objects; pointer values never leave this capture process.
    fn expand(&mut self) -> Result<()> {
        loop {
            let before = (self.runtimes.len(), self.registries.len());
            for (id, runtime) in self.runtimes.clone() {
                if let Some(registry) = runtime.spawner_registry()
                    && !self
                        .registries
                        .values()
                        .any(|old| Arc::ptr_eq(old, registry))
                {
                    self.registries
                        .insert(format!("registry:{id}"), registry.clone());
                }
                if let Some(spawner) = runtime.spawner()
                    && !self.spawners.values().any(|old| Arc::ptr_eq(old, spawner))
                {
                    self.spawners
                        .insert(format!("spawner:{id}"), spawner.clone());
                }
                self.providers
                    .entry(format!("runtime:{id}"))
                    .or_insert_with(|| runtime.task_restore_providers());
            }
            for (id, spawner) in &self.spawners {
                if let Some(providers) = spawner.reconstruction_providers() {
                    self.providers
                        .entry(format!("spawner:{id}"))
                        .or_insert_with(|| Arc::new(providers));
                }
            }
            for registry in self.registries.values() {
                for entry in registry.reconstruction_entries() {
                    let id = entry.agent.info().id;
                    if self
                        .runtimes
                        .get(&id)
                        .is_some_and(|old| !Arc::ptr_eq(old, &entry.agent))
                    {
                        return Err(AgentError::Config(
                            "different runtime objects share a restore identity".into(),
                        ));
                    }
                    self.runtimes.insert(id.clone(), entry.agent.clone());
                    if let Some(old) = self.declarations.get(&id)
                        && serde_json::to_value(old)? != serde_json::to_value(&entry.spec)?
                    {
                        return Err(AgentError::Config("registry aliases with different retained declarations require an explicit host migration".into()));
                    }
                    self.declarations
                        .entry(id)
                        .or_insert_with(|| entry.spec.clone());
                }
            }
            if self.runtimes.len() + self.registries.len() > MAX_TASK_CHECKPOINT_RECORDS {
                return Err(TaskRunStorageError::CheckpointTooLarge.into());
            }
            if before == (self.runtimes.len(), self.registries.len()) {
                return Ok(());
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredRestoreContext {
    requested_name: String,
    canonical_id: String,
    display_name: String,
    provider_id: Option<String>,
    registry_version: u64,
    policy_version: u64,
    runtime_control_version: u64,
    call_id: String,
    source: ai_agents_core::ToolCallSource,
    actor: ai_agents_core::ToolActorContext,
    started_at: DateTime<Utc>,
    deadline: Option<DateTime<Utc>>,
    permission: ai_agents_core::ToolPolicyDecisionRecord,
    approval: Option<ai_agents_core::ToolApprovalRecord>,
    classification: ai_agents_core::ToolCallClassification,
    safety: ai_agents_core::ToolSafetyMetadata,
    limits: ai_agents_core::ToolExecutionLimits,
}

// Context persistence omits raw custom policy settings; current host policy supplies those again after compatibility validation.
fn capture_message_context(context: &ai_agents_core::ToolExecutionContext) -> Value {
    json!({"requested_name":context.requested_name,"canonical_id":context.canonical_id,"display_name":context.display_name,
        "provider_id":context.provider_id,"registry_version":context.registry_version,"policy_version":context.policy_version,
        "runtime_control_version":context.runtime_control_version,"call_id":context.call_id,"source":context.source,
        "actor":context.actor,"started_at":context.started_at,"deadline":context.deadline,"permission":context.permission,
        "approval":context.approval,"classification":context.classification,"safety":context.safety,"limits":context.limits})
}

// Converts only an actual source monotonic bound, then intersects prior absolute cuts so repeated reconstruction cannot extend it.
fn capture_deadline(deadline: Instant, host: &dyn TaskRestoreHost) -> Result<DateTime<Utc>> {
    let now = host.now()?;
    let remaining = chrono::Duration::from_std(deadline.saturating_duration_since(Instant::now()))
        .map_err(|_| AgentError::Config("restore clock projection overflow".into()))?;
    now.checked_add_signed(remaining)
        .ok_or_else(|| AgentError::Config("restore clock projection overflow".into()))
}

// A capture contains every actual catalogue and current runtime, including unstarted entries and tool-specific auxiliary registries.
// The installed host and its UTC history remain immutable authorities; a rollback cannot be recaptured as an ordinary expiry.
async fn capture_manifest(
    runner: &AutonomyRunner,
    paused: &PausedTask,
    payload: &TaskCheckpointPayload,
    host: &dyn TaskRestoreHost,
    revision: u64,
    check_response_guards: bool,
) -> Result<TaskRestoreManifest> {
    paused.owner.check()?;
    // A writable store cannot replace the exact executable cursor retained by the actual source owner.
    let acknowledged = if let Some(controller) = &paused.controller {
        controller_checkpoint(payload)?.is_some_and(|stored| {
            serde_json::to_value(&stored).ok() == serde_json::to_value(controller).ok()
        })
    } else if let Some(group) = &paused.group {
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.group")
            .is_some_and(|adapter| serde_json::to_value(group).ok() == Some(adapter.state.clone()))
    } else if let Some(batch) = &paused.batch {
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.tool_batch")
            .is_some_and(|adapter| serde_json::to_value(batch).ok() == Some(adapter.state.clone()))
    } else {
        false
    };
    if !acknowledged {
        return Err(AgentError::Config(
            "restore preparation cursor differs from live acknowledged state".into(),
        ));
    }
    let captured_at = host.now()?;
    let prior = payload
        .adapters
        .iter()
        .find(|adapter| adapter.id == RESTORE_ADAPTER)
        .map(|adapter| serde_json::from_value::<TaskRestoreManifest>(adapter.state.clone()))
        .transpose()?;
    if let Some(prior) = &prior {
        if prior.host != host.identity() {
            return Err(AgentError::Config(
                "restore host continuity could not be established".into(),
            ));
        }
        // A rollback is a clock-contract failure, not an expired work budget or permission to recapture under a new epoch.
        if captured_at < prior.captured_at {
            return Err(TaskRunStorageError::ClockDiscontinuity.into());
        }
    }
    let mut deadlines = BTreeMap::new();
    for (id, deadline) in &paused.approval_deadlines {
        deadlines.insert(id.clone(), capture_deadline(*deadline, host)?);
    }
    let source_responses: Vec<_> = paused
        .owner
        .responses
        .lock()
        .iter()
        .map(|(id, response)| (id.clone(), response.clone()))
        .collect();
    for (id, response) in &source_responses {
        if let Some(deadline) = response.deadline {
            let cut = capture_deadline(deadline, host)?;
            deadlines
                .entry(id.clone())
                .and_modify(|old| *old = (*old).min(cut))
                .or_insert(cut);
        }
    }
    if let Some(prior) = &prior {
        for (id, cut) in &mut deadlines {
            if let Some(previous) = prior.deadlines.get(id) {
                *cut = (*cut).min(*previous);
            }
        }
    }
    let mut whole_expiry = paused
        .whole_expiry
        .map(|deadline| capture_deadline(deadline, host))
        .transpose()?;
    if let Some(original) = payload.clocks.expires_at {
        whole_expiry = Some(whole_expiry.map_or(original, |cut| cut.min(original)));
    }
    if let Some(previous) = prior.as_ref().and_then(|prior| prior.whole_expiry) {
        whole_expiry = Some(whole_expiry.map_or(previous, |cut| cut.min(previous)));
    }
    let mut validation_bound = paused
        .controller
        .as_ref()
        .and_then(|controller| controller.driver.as_ref())
        .map(|driver| {
            driver
                .reconstruction_expiry(host.now()?)
                .map(|cut| cut.map(|cut| (driver.identity.clone(), cut)))
        })
        .transpose()?
        .flatten();
    if let (Some((identity, cut)), Some((old, previous))) = (
        &mut validation_bound,
        prior
            .as_ref()
            .and_then(|prior| prior.validation_bound.as_ref()),
    ) && identity == old
    {
        *cut = (*cut).min(*previous);
    }
    if paused.participants.unsettled() || paused.owner.work.unsettled() {
        return Err(TaskRunStorageError::NotResumable.into());
    }
    let mut graph = host.capture_graph(runner.agent.clone())?;
    if graph
        .runtimes
        .get(&runner.agent.info().id)
        .is_none_or(|root| !Arc::ptr_eq(root, &runner.agent))
    {
        return Err(AgentError::Config(
            "restore source graph does not contain the actual root".into(),
        ));
    }
    let live = paused.targets.reconstruction_bindings()?;
    for (identity, registry) in &live.registries {
        if !graph
            .registries
            .values()
            .any(|old| Arc::ptr_eq(old, registry))
        {
            graph.registries.insert(
                format!("captured:{}", graph.registries.len()),
                registry.clone(),
            );
        }
        if *identity != registry.reconstruction_identity() {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for (id, (registry, providers)) in &live.compositions {
        if !graph
            .registries
            .values()
            .any(|old| Arc::ptr_eq(old, registry))
        {
            graph
                .registries
                .insert(format!("composition:{id}"), registry.clone());
        }
        if let Some(providers) = providers {
            graph
                .providers
                .insert(format!("composition:{id}"), providers.clone());
        }
    }
    graph.expand()?;
    driver::validate_source_views(&graph)?;
    let registry_key = |identity| {
        graph
            .registries
            .iter()
            .find(|(_, registry)| registry.reconstruction_identity() == identity)
            .map(|(key, _)| key.clone())
            .ok_or_else(|| {
                AgentError::Config("captured registry is absent from source graph".into())
            })
    };
    let mut nodes = BTreeMap::new();
    for (id, runtime) in &graph.runtimes {
        let actual_owner = runtime.captured_task_owner();
        let expected = if Arc::ptr_eq(runtime, &runner.agent) {
            Some(&paused.owner)
        } else {
            actual_owner
                .as_ref()
                .filter(|owner| paused.participants.owns(owner))
        };
        if actual_owner.is_some() && expected.is_none() {
            return Err(TaskRunStorageError::Owned.into());
        }
        let mut continuation = if Arc::ptr_eq(runtime, &runner.agent) {
            payload.runtime.continuation.clone()
        } else {
            TaskContinuation::BetweenTurns
        };
        let pending: Vec<_> = payload
            .children
            .iter()
            .filter(|child| child.runtime.snapshot.agent_id == *id && child.pending.is_some())
            .collect();
        if pending.len() > 1 {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        if let Some(child) = pending.first() {
            continuation = child.runtime.continuation.clone();
        }
        let (checkpoint, session, generations) = runtime
            .capture_task_reconstruction(continuation, expected)
            .await?;
        let binding = host.runtime_binding(runtime, graph.declarations.get(id))?;
        if binding.id.trim().is_empty() || binding.contract_version == 0 {
            return Err(AgentError::Config(
                "invalid host reconstruction binding".into(),
            ));
        }
        nodes.insert(
            id.clone(),
            TaskRestoreNode {
                binding,
                configuration_digest: runtime.task_restore_configuration()?,
                runtime: checkpoint,
                session,
                generations,
                registry: runtime
                    .spawner_registry()
                    .map(|registry| registry_key(registry.reconstruction_identity()))
                    .transpose()?,
                spawner: runtime
                    .spawner()
                    .map(|spawner| {
                        graph
                            .spawners
                            .iter()
                            .find(|(_, candidate)| Arc::ptr_eq(candidate, spawner))
                            .map(|(key, _)| key.clone())
                            .ok_or_else(|| AgentError::Config("source spawner is absent".into()))
                    })
                    .transpose()?,
                participant_actor: expected
                    .filter(|_| !Arc::ptr_eq(runtime, &runner.agent))
                    .map(|owner| TaskRestoreParticipant {
                        actor_id: owner.actor_id.clone(),
                    }),
                skill: expected.and_then(|owner| owner.skill.read().clone()),
            },
        );
    }
    let mut registries = BTreeMap::new();
    let mut occupied = BTreeMap::<String, u64>::new();
    for (key, registry) in &graph.registries {
        let mut entries = BTreeMap::new();
        for entry in registry.reconstruction_entries() {
            let spawner = entry
                .reconstruction_capacity()
                .map(|identity| {
                    graph
                        .spawners
                        .iter()
                        .find(|(_, spawner)| spawner.reconstruction_identity() == identity)
                        .map(|(key, _)| key.clone())
                        .ok_or_else(|| {
                            AgentError::Config(
                                "managed source entry lacks a captured spawner".into(),
                            )
                        })
                })
                .transpose()?;
            if let Some(key) = &spawner {
                *occupied.entry(key.clone()).or_default() += 1;
            }
            entries.insert(
                entry.id.clone(),
                TaskRestoreEntry {
                    spawned_at: entry.spawned_at,
                    runtime: entry.agent.info().id,
                    spawner,
                },
            );
        }
        registries.insert(
            key.clone(),
            TaskRestoreRegistry {
                send_with_context: registry.reconstruction_send_context(),
                entries,
            },
        );
    }
    let mut spawners = BTreeMap::new();
    for (key, spawner) in &graph.spawners {
        let state = spawner.reconstruction_state()?;
        if state["occupied"].as_u64() != Some(*occupied.get(key).unwrap_or(&0)) {
            return Err(AgentError::Config(
                "source capacity has unacknowledged or foreign reservations".into(),
            ));
        }
        spawners.insert(key.clone(), state);
    }
    let mut targets = BTreeMap::new();
    for (operation, (identity, entry, runtime)) in live.targets {
        targets.insert(
            operation,
            TaskRestoreTarget {
                registry: registry_key(identity)?,
                entry,
                runtime: runtime.info().id,
            },
        );
    }
    let mut compositions = BTreeMap::new();
    for (id, (registry, providers)) in live.compositions {
        compositions.insert(
            id.clone(),
            TaskRestoreComposition {
                registry: registry_key(registry.reconstruction_identity())?,
                providers: providers.map(|_| format!("composition:{id}")),
            },
        );
    }
    let mut tools = BTreeMap::new();
    let frames: BTreeMap<_, DelegateFrame> = payload
        .adapters
        .iter()
        .filter(|adapter| adapter.adapter == "runtime.delegate")
        .map(|adapter| {
            Ok((
                adapter.id.clone(),
                serde_json::from_value(adapter.state.clone())?,
            ))
        })
        .collect::<Result<_>>()?;
    for (id, tool) in live.tools {
        let frame = frames
            .values()
            .find(|frame| frame.id == id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let state: super::super::message::MessageState =
            serde_json::from_value(frame.cursor.clone())?;
        let runtime = graph
            .runtimes
            .get(&frame.runtime_id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        if !Arc::ptr_eq(
            &runtime.task_restore_tool(&state.record.requested_name)?,
            &tool,
        ) {
            return Err(AgentError::Config(
                "captured tool is not installed on its source runtime".into(),
            ));
        }
        if !compositions.contains_key(&id) {
            let catalogues: std::collections::BTreeSet<_> = frame
                .slots()
                .iter()
                .filter_map(|slot| {
                    targets
                        .get(&slot.operation)
                        .map(|target| target.registry.clone())
                })
                .collect();
            if catalogues.len() == 1 {
                compositions.insert(
                    id.clone(),
                    TaskRestoreComposition {
                        registry: catalogues.into_iter().next().unwrap(),
                        providers: None,
                    },
                );
            }
        }
        tools.insert(id, (frame.runtime_id.clone(), state.record.requested_name));
    }
    let mut responses = BTreeMap::new();
    for (id, response) in source_responses {
        if check_response_guards && let Some(review) = &response.review {
            review.validate()?;
        }
        responses.insert(
            id.clone(),
            TaskRestoreResponse {
                runtime: response.runtime_id.clone(),
                request: response.request.clone(),
                expires_at: response.expires_at,
                applied: response.applied.load(std::sync::atomic::Ordering::Acquire),
                timeout_disposition: response
                    .timeout_disposition
                    .load(std::sync::atomic::Ordering::Acquire),
            },
        );
    }
    let skill_finishing = {
        let guards = paused.owner.skill_finishing.lock();
        for guard in guards.values() {
            guard.validate()?;
        }
        guards.keys().cloned().collect()
    };
    let mut providers = BTreeMap::new();
    for (id, provider) in &graph.providers {
        providers.insert(id.clone(), host.provider_binding(provider)?);
    }
    let message_contexts = live
        .contexts
        .iter()
        .filter(|(id, _)| tools.contains_key(*id))
        .map(|(id, context)| (id.clone(), capture_message_context(context)))
        .collect();
    let manifest = TaskRestoreManifest {
        version: 1,
        host: host.identity(),
        revision,
        payload_digest: payload_digest(payload)?,
        captured_at,
        whole_expiry,
        deadlines,
        validation_bound,
        root: runner.agent.info().id,
        nodes,
        registries,
        spawners,
        providers,
        targets,
        compositions,
        tools,
        message_contexts,
        responses,
        operation_ordinals: paused.participants.capture_ordinals()?,
        skill_finishing,
        active_frames: {
            let mut ids: Vec<_> = paused
                .group
                .as_ref()
                .map(|group| {
                    group
                        .frames
                        .values()
                        .map(|frame| frame.id.clone())
                        .collect()
                })
                .unwrap_or_default();
            if let Some(value) = paused
                .controller
                .as_ref()
                .and_then(|controller| controller.driver.as_ref())
                .and_then(|driver| driver.parked.as_ref())
                .and_then(|parked| parked.group.as_ref())
            {
                let group: TaskGroupState = serde_json::from_value(value.clone())?;
                ids.extend(group.frames.values().map(|frame| frame.id.clone()));
            }
            ids.sort();
            ids.dedup();
            ids
        },
    };
    Ok(manifest)
}

impl TaskReconstructionCheckpoint {
    /// Stores must forward this exact operation and its live guard, never translate it into an unguarded checkpoint.
    pub fn mutation(&self) -> &TaskRunMutation {
        &self.mutation
    }

    /// Opaque construction binds the deterministic transformation to one acknowledged claimed predecessor.
    pub fn validate(&self, actual: &TaskRunSnapshot) -> Result<()> {
        if serde_json::to_value(actual)? != serde_json::to_value(&self.previous)? {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let TaskRunMutation::FinalCheckpoint {
            status: TaskRunStatus::Paused,
            release: true,
            payload,
            evidence: Some(guard),
            ..
        } = &self.mutation
        else {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        };
        guard.validate()?;
        let old: TaskCheckpointPayload = serde_json::from_value(actual.payload.clone())?;
        let next: TaskCheckpointPayload = serde_json::from_value(payload.clone())?;
        for (left, right) in [
            (
                serde_json::to_value(&old.counters)?,
                serde_json::to_value(&next.counters)?,
            ),
            (
                serde_json::to_value(&old.reservations)?,
                serde_json::to_value(&next.reservations)?,
            ),
            (
                serde_json::to_value(&old.evidence)?,
                serde_json::to_value(&next.evidence)?,
            ),
            (
                serde_json::to_value(&old.progress)?,
                serde_json::to_value(&next.progress)?,
            ),
            (
                serde_json::to_value(&old.todos)?,
                serde_json::to_value(&next.todos)?,
            ),
            (
                serde_json::to_value(&old.settings)?,
                serde_json::to_value(&next.settings)?,
            ),
            (
                serde_json::to_value(&old.limits)?,
                serde_json::to_value(&next.limits)?,
            ),
        ] {
            if left != right {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
        }
        if old.objective != next.objective
            || old.objective_revision != next.objective_revision
            || old.config_identity != next.config_identity
            || old.profile != next.profile
            || old.children.len() != next.children.len()
            || old.clocks.expires_at != next.clocks.expires_at
            || next.clocks.active_millis < old.clocks.active_millis
            || next.clocks.interrupted_interval_millis != old.clocks.interrupted_interval_millis
            || next.clocks.active_interval_started_at.is_some()
            || old
                .consumed_request_ids
                .iter()
                .any(|id| !next.consumed_request_ids.contains(id))
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        for (old, next) in old.children.iter().zip(&next.children) {
            if old.child_id != next.child_id
                || old.config_identity != next.config_identity
                || old.result != next.result
                || serde_json::to_value(&old.runtime.snapshot)?
                    != serde_json::to_value(&next.runtime.snapshot)?
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
        }
        Ok(())
    }
}

struct RestorePublication {
    guards: Vec<Arc<dyn TaskFinalAdmission>>,
}
impl std::fmt::Debug for RestorePublication {
    // Host configuration and recovery payloads are never included in guard diagnostics.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RestorePublication")
    }
}
impl TaskFinalAdmission for RestorePublication {
    // Every actual staged owner and generation must still match at the backend's commit boundary.
    fn validate(&self) -> Result<()> {
        for guard in &self.guards {
            guard.validate()?;
        }
        Ok(())
    }
}

pub(super) struct LiveRestoreReview {
    pub(super) view: TaskRestorationReview,
    pub(super) approved: Arc<std::sync::atomic::AtomicBool>,
    pub(super) guards: Vec<Arc<dyn TaskFinalAdmission>>,
    pub(super) budget: std::time::Duration,
    pub(super) host: Arc<dyn TaskRestoreHost>,
}

// Recreates only execution bookkeeping after an acknowledged claim; it performs no provider/tool or driver work.
fn claimed_execution(
    runner: &AutonomyRunner,
    claimed: &TaskRunSnapshot,
    paused: &PausedTask,
) -> Result<Arc<RunExecution>> {
    let payload: TaskCheckpointPayload = serde_json::from_value(claimed.payload.clone())?;
    let controller = controller_checkpoint(&payload)?;
    let lifecycle = controller
        .as_ref()
        .map(|state| state.lifecycle.clone())
        .unwrap_or(LifecycleState::new(
            &payload.settings,
            paused.owner.skill.read().is_some(),
        )?);
    let mut execution = RunExecution::new(
        runner.store.clone(),
        claimed,
        lifecycle,
        paused.owner.clone(),
        Instant::now(),
    )?;
    let live = Arc::get_mut(&mut execution).ok_or(TaskRunStorageError::Conflict)?;
    live.participants = paused.participants.clone();
    live.targets = paused.targets.clone();
    *live.approval_deadlines.lock() = paused.approval_deadlines.clone();
    if let Some(group) = &paused.group {
        *live.delegate_frames.lock() = group.frames.clone();
        for leaf in &group.parked {
            live.parked_batches
                .lock()
                .insert(leaf.runtime_id.clone(), leaf.batch.clone());
        }
    }
    if let Some(batch) = &paused.batch {
        live.parked_batches
            .lock()
            .insert(runner.agent.info().id, batch.clone());
    }
    *live.controller.lock() = controller;
    if let Some(expiry) = paused.whole_expiry {
        live.expiry_projection = Some(
            live.expiry_projection
                .map_or(expiry, |current| current.min(expiry)),
        );
    }
    if let Some(todo) = &paused.todo {
        execution.install_todos(runner.agent.todo_store(), todo.checkpoint()?.binding);
    }
    Ok(execution)
}

// A host-selected bound is charged conservatively so construction and backend publication waits cannot return free active allowance.
// Safely paused human downtime is excluded; this is an explicit reconstruction-work ceiling, not a claim of measured elapsed time.
fn charge_reconstruction(
    payload: &mut TaskCheckpointPayload,
    budget: std::time::Duration,
) -> Result<()> {
    let elapsed = u64::try_from(budget.as_millis())
        .map_err(|_| AgentError::Config("reconstruction elapsed overflow".into()))?;
    if elapsed == 0 {
        return Err(AgentError::Config(
            "reconstruction requires a positive bounded work budget".into(),
        ));
    }
    payload.clocks.active_millis = payload
        .clocks
        .active_millis
        .checked_add(elapsed)
        .ok_or_else(|| AgentError::Config("reconstruction clock overflow".into()))?;
    let consumed = payload
        .clocks
        .active_millis
        .checked_add(payload.clocks.interrupted_interval_millis)
        .ok_or(TaskRunStorageError::AdmissionExpired)?;
    if consumed >= payload.limits.max_active_time_seconds.saturating_mul(1000) {
        return Err(TaskRunStorageError::AdmissionExpired.into());
    }
    Ok(())
}

// The work marker is recovery bookkeeping, not progress/evidence or permission to invoke any pending operation.
fn mark_work(
    payload: &mut TaskCheckpointPayload,
    kind: &str,
    phase: &str,
    budget: std::time::Duration,
    started_at: DateTime<Utc>,
) -> Result<()> {
    let adapter = TaskAdapterCheckpoint {
        id: RESTORE_WORK.into(),
        adapter: RESTORE_WORK.into(),
        contract_version: 1,
        config: json!({"contract":"bounded_reconstruction_v1"}),
        state: json!({"kind":kind,"phase":phase,"budget_millis":u64::try_from(budget.as_millis()).map_err(|_|AgentError::Config("reconstruction budget overflow".into()))?,"started_at":started_at}),
    };
    if let Some(old) = payload
        .adapters
        .iter_mut()
        .find(|old| old.id == RESTORE_WORK)
    {
        if old.config != adapter.config {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        *old = adapter;
    } else {
        payload.adapters.push(adapter);
    }
    Ok(())
}

// Precharges the bounded work in a claimed Running journal before staged components or approval publication can be attempted.
// An interrupted attempt remains owned and marked for explicit reconciliation; it cannot resume as an uncharged safe pause.
async fn begin_work(
    store: &Arc<dyn TaskRunStore>,
    claimed: &TaskRunSnapshot,
    ready: &TaskCheckpointPayload,
    kind: &str,
    budget: std::time::Duration,
    started_at: DateTime<Utc>,
    deadline: Instant,
) -> Result<TaskRunSnapshot> {
    if Instant::now() >= deadline {
        return Err(TaskRunStorageError::AdmissionExpired.into());
    }
    let mut running = ready.clone();
    mark_work(&mut running, kind, "running", budget, started_at)?;
    let mutation = TaskRunMutation::Checkpoint {
        expected_revision: claimed.revision,
        owner_token: claimed
            .owner_token
            .clone()
            .ok_or(TaskRunStorageError::Owned)?,
        status: TaskRunStatus::Running,
        payload: serde_json::to_value(running)?,
        release: false,
    };
    let saved = store.mutate(&claimed.key.run_id, &mutation).await?;
    super::super::store::verify_checkpoint_receipt(claimed, &mutation, &saved)?;
    if Instant::now() >= deadline {
        return Err(TaskRunStorageError::AdmissionExpired.into());
    }
    Ok(saved)
}

// One fixed monotonic deadline spans preflight, claim, staging and publication; later phases cannot renew it.
fn reconstruction_deadline(
    started: Instant,
    budget: std::time::Duration,
    whole: Option<Instant>,
) -> Result<Instant> {
    let deadline = started
        .checked_add(budget)
        .ok_or(TaskRunStorageError::AdmissionExpired)?;
    let deadline = whole.map_or(deadline, |whole| whole.min(deadline));
    if budget.is_zero() || Instant::now() >= deadline {
        return Err(TaskRunStorageError::AdmissionExpired.into());
    }
    Ok(deadline)
}

// Predictable adapter/configuration/byte-cap failures are rejected before the live pause is claimed.
fn prepare_capture(
    snapshot: &TaskRunSnapshot,
    payload: &mut TaskCheckpointPayload,
    mut manifest: TaskRestoreManifest,
    budget: std::time::Duration,
    started_at: DateTime<Utc>,
) -> Result<()> {
    charge_reconstruction(payload, budget)?;
    mark_work(payload, "capture", "paused", budget, started_at)?;
    manifest.payload_digest = payload_digest(payload)?;
    let adapter = TaskAdapterCheckpoint {
        id: RESTORE_ADAPTER.into(),
        adapter: RESTORE_ADAPTER.into(),
        contract_version: 1,
        config: json!({"host":manifest.host}),
        state: serde_json::to_value(&manifest)?,
    };
    if let Some(old) = payload
        .adapters
        .iter_mut()
        .find(|adapter| adapter.id == RESTORE_ADAPTER)
    {
        if old.config != adapter.config {
            return Err(AgentError::Config(
                "restore host binding cannot change within a run".into(),
            ));
        }
        *old = adapter;
    } else {
        payload.adapters.push(adapter);
    }
    let mut proposed = snapshot.clone();
    proposed.revision = manifest.revision;
    proposed.payload = serde_json::to_value(&*payload)?;
    payload.validate(&proposed, &payload.config_identity)?;
    proposed.status = TaskRunStatus::Running;
    proposed.owner_token = Some("00000000-0000-0000-0000-000000000000".into());
    payload.validate(&proposed, &payload.config_identity)?;
    Ok(())
}

// Preclaim capture is read-only custody: errors or future cancellation must return the exact retained pause without abandoning its owner.
struct PausePreparation<'a> {
    retained: &'a parking_lot::Mutex<BTreeMap<String, PausedTask>>,
    run_id: &'a str,
    paused: Option<PausedTask>,
}
impl Drop for PausePreparation<'_> {
    // The resume gate still excludes competing local control; disarming before claim separates safe rollback from ambiguous writes.
    fn drop(&mut self) {
        if let Some(paused) = self.paused.take() {
            self.retained.lock().insert(self.run_id.into(), paused);
        }
    }
}

impl AutonomyRunner {
    /// Prepares one explicit durable restart point from an acknowledged safe live pause.
    /// Current node state, host bindings and complete topology are captured without replay; subsequent progress requires a fresh preparation.
    /// Preclaim capture cancellation preserves the original live pause; cancellation after a claim starts retains ambiguous custody.
    pub async fn checkpoint_for_restore(
        &self,
        run_id: &str,
        expected_revision: u64,
        host: &dyn TaskRestoreHost,
    ) -> Result<TaskRunResult> {
        let _gate = self.resume_gate.lock().await;
        let started = Instant::now();
        let budget = host.reconstruction_budget();
        let started_at = host.now()?;
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;
        safe_pause(&snapshot, &payload, host.now()?)?;
        if snapshot.revision != expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let mut preparation = PausePreparation {
            retained: &self.paused,
            run_id,
            paused: Some(
                self.paused
                    .lock()
                    .remove(run_id)
                    .ok_or(TaskRunStorageError::NotResumable)?,
            ),
        };
        let paused = preparation
            .paused
            .as_ref()
            .ok_or(TaskRunStorageError::NotResumable)?;
        let next_revision = snapshot
            .revision
            .checked_add(3)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let manifest = capture_manifest(self, paused, &payload, host, next_revision, true).await?;
        host.validate_manifest(&manifest)?;
        if paused
            .whole_expiry
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        prepare_capture(&snapshot, &mut payload, manifest, budget, started_at)?;
        let deadline = reconstruction_deadline(started, budget, paused.whole_expiry)?;
        let paused = preparation
            .paused
            .take()
            .ok_or(TaskRunStorageError::NotResumable)?;
        let mut cleanup = OwnedTurnCleanup::new(paused.owner.clone());
        let token = uuid::Uuid::new_v4().to_string();
        let claimed = match self
            .store
            .claim_paused(run_id, snapshot.revision, token.clone())
            .await?
        {
            Some(claimed) => claimed,
            None => {
                cleanup.finish();
                self.paused.lock().insert(run_id.into(), paused);
                return Err(TaskRunStorageError::Conflict.into());
            }
        };
        super::super::store::verify_checkpoint_receipt(
            &snapshot,
            &TaskRunMutation::Claim {
                expected_revision: snapshot.revision,
                owner_token: token.clone(),
            },
            &claimed,
        )?;
        let claimed = begin_work(
            &self.store,
            &claimed,
            &payload,
            "capture",
            budget,
            started_at,
            deadline,
        )
        .await?;
        let execution = claimed_execution(self, &claimed, &paused)?;
        *self.active.lock() = Some(execution.clone());

        let mutation = TaskRunMutation::FinalCheckpoint {
            expected_revision: claimed.revision,
            owner_token: token,
            status: TaskRunStatus::Paused,
            payload: serde_json::to_value(&payload)?,
            release: true,
            deadline,
            expires_at: payload.clocks.expires_at,
            evidence: None,
        };
        let saved = self.store.mutate(run_id, &mutation).await?;
        super::super::store::verify_checkpoint_receipt(&claimed, &mutation, &saved)?;
        execution.acknowledge_snapshot(&saved);
        self.paused.lock().insert(run_id.into(), paused);
        cleanup.finish();
        Ok(TaskRunResult {
            run: TaskRun::from_checkpoint(&saved, &self.config_identity)?,
            final_response: None,
        })
    }
}
