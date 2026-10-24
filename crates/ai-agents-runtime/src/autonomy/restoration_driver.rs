//! Claimed reconstruction installs the existing typed continuation drivers without invoking them.

use super::*;

use std::sync::atomic::{AtomicBool, Ordering};

// Stored limits cannot exceed the current authenticated host even when raw profile settings still compare equal.
fn validate_current_host(
    request: &TaskRestoreRequest,
    payload: &TaskCheckpointPayload,
) -> Result<()> {
    let skill = payload
        .adapters
        .iter()
        .find(|adapter| adapter.id == "runtime.entry")
        .map(|adapter| {
            serde_json::from_value::<ai_agents_skills::SkillDefinition>(adapter.config.clone())
        })
        .transpose()?;
    let profile = resolve_profile(
        &request.config,
        if skill.is_some() {
            AutonomyScope::Skill
        } else {
            AutonomyScope::Task
        },
        payload.profile.as_deref(),
        skill.as_ref().and_then(|skill| skill.autonomy.as_ref()),
        None,
        Some(&payload.objective),
        &request.ceilings,
    )?;
    if profile.settings != payload.settings || !profile.enabled {
        return Err(AgentError::Config(
            "restored profile differs from current configuration".into(),
        ));
    }
    let limits = &payload.limits;
    let ceilings = &request.ceilings;
    for (old, current) in [
        (
            u64::from(limits.max_turns),
            ceilings.max_turns.map(u64::from),
        ),
        (
            limits.max_active_time_seconds,
            ceilings.max_active_time_seconds,
        ),
        (
            u64::from(limits.max_llm_calls),
            ceilings.max_llm_calls.map(u64::from),
        ),
        (
            u64::from(limits.max_tool_calls),
            ceilings.max_tool_calls.map(u64::from),
        ),
        (
            u64::from(limits.max_command_calls),
            ceilings.max_command_calls.map(u64::from),
        ),
    ] {
        if current.is_some_and(|current| old > current) {
            return Err(AgentError::Config("stored task allowance exceeds current host ceiling; explicit narrowing is required".into()));
        }
    }
    for (old, current) in [
        (limits.max_wall_time_seconds, ceilings.max_wall_time_seconds),
        (
            limits.max_micro_usd,
            ceilings.max_cost_usd.as_ref().map(|cost| cost.micro_usd()),
        ),
        (
            limits.max_declared_write_paths.map(u64::from),
            ceilings.max_declared_write_paths.map(u64::from),
        ),
    ] {
        if current.is_some_and(|current| old.is_none_or(|old| old > current)) {
            return Err(AgentError::Config(
                "stored optional allowance exceeds current host ceiling".into(),
            ));
        }
    }
    Ok(())
}

// Capture timestamps and equivalent YAML formatting are not semantic snapshot differences; memory/native content stays exact.
fn snapshot_identity(snapshot: &ai_agents_core::AgentSnapshot) -> Result<Value> {
    let mut value = serde_json::to_value(snapshot)?;
    value
        .as_object_mut()
        .ok_or(TaskRunStorageError::InvalidCheckpoint)?
        .remove("timestamp");
    if let Some(entries) = value["spawned_agents"].as_array_mut() {
        for entry in entries {
            if let Some(text) = entry["spec_yaml"].as_str() {
                entry["spec_yaml"] = serde_json::to_value(AgentSpec::from_yaml_strict(text)?)?;
            }
        }
    }
    Ok(value)
}

// Shape/reference validation precedes any mutable staging, claim or component restore.
fn validate_manifest(
    manifest: &TaskRestoreManifest,
    snapshot: &TaskRunSnapshot,
    payload: &TaskCheckpointPayload,
    host: &dyn TaskRestoreHost,
) -> Result<()> {
    if manifest.version != 1
        || manifest.host != host.identity()
        || manifest.host.0.trim().is_empty()
        || manifest.host.1 == 0
        || manifest.revision != snapshot.revision
        || manifest.payload_digest != payload_digest(payload)?
        || manifest.root != snapshot.key.agent_id
        || !manifest.nodes.contains_key(&manifest.root)
        || manifest.nodes.len() + manifest.registries.len() > MAX_TASK_CHECKPOINT_RECORDS
    {
        return Err(AgentError::Config(
            "restore manifest is stale or incompatible".into(),
        ));
    }
    for (id, node) in &manifest.nodes {
        node.runtime.validate()?;
        let authoritative = if id == &manifest.root {
            Some(&payload.runtime)
        } else {
            payload
                .children
                .iter()
                .find(|child| child.runtime.snapshot.agent_id == *id && child.pending.is_some())
                .map(|child| &child.runtime)
        };
        if let Some(authoritative) = authoritative {
            if snapshot_identity(&node.runtime.snapshot)?
                != snapshot_identity(&authoritative.snapshot)?
                || node.runtime.native_exchanges != authoritative.native_exchanges
            {
                return Err(AgentError::Config(
                    "current node image differs from its acknowledged pending checkpoint".into(),
                ));
            }
        }
        if node.runtime.snapshot.agent_id != *id
            || node.binding.id.trim().is_empty()
            || node.binding.contract_version == 0
            || node
                .registry
                .as_ref()
                .is_some_and(|key| !manifest.registries.contains_key(key))
            || node
                .spawner
                .as_ref()
                .is_some_and(|key| !manifest.spawners.contains_key(key))
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for registry in manifest.registries.values() {
        for entry in registry.entries.values() {
            if !manifest.nodes.contains_key(&entry.runtime)
                || entry
                    .spawner
                    .as_ref()
                    .is_some_and(|key| !manifest.spawners.contains_key(key))
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
        }
    }
    for target in manifest.targets.values() {
        if manifest
            .registries
            .get(&target.registry)
            .and_then(|registry| registry.entries.get(&target.entry))
            .is_none_or(|entry| entry.runtime != target.runtime)
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for composition in manifest.compositions.values() {
        if !manifest.registries.contains_key(&composition.registry)
            || composition
                .providers
                .as_ref()
                .is_some_and(|key| !manifest.providers.contains_key(key))
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for response in manifest.responses.values() {
        if !manifest.nodes.contains_key(&response.runtime) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    let mut frames = BTreeMap::new();
    for adapter in payload
        .adapters
        .iter()
        .filter(|adapter| adapter.adapter == "runtime.delegate")
    {
        let frame: DelegateFrame = serde_json::from_value(adapter.state.clone())?;
        if !manifest.nodes.contains_key(&frame.runtime_id)
            || frames.insert(frame.id.clone(), frame).is_some()
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for (id, (runtime, tool)) in &manifest.tools {
        let frame = frames
            .get(id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        if !manifest.nodes.contains_key(runtime)
            || frame.runtime_id != *runtime
            || frame.dispatch != crate::autonomy::composition::CompositionDispatch::ToolMessage
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let state: crate::autonomy::message::MessageState =
            serde_json::from_value(frame.cursor.clone())?;
        if state.record.requested_name != *tool {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for (id, context) in &manifest.message_contexts {
        let (_, tool) = manifest
            .tools
            .get(id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let context: StoredRestoreContext = serde_json::from_value(context.clone())?;
        let frame = frames
            .get(id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let state: crate::autonomy::message::MessageState =
            serde_json::from_value(frame.cursor.clone())?;
        if context.requested_name != *tool
            || context.call_id != state.record.call_id
            || context.canonical_id != state.record.canonical_id
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for id in manifest.compositions.keys() {
        if !manifest.tools.contains_key(id) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    for runtime in &manifest.skill_finishing {
        let node = manifest
            .nodes
            .get(runtime)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        if runtime != &manifest.root && node.participant_actor.is_none() {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    let mut active = Vec::new();
    if payload.pause_reason.as_deref() == Some("child_interaction") {
        let group: TaskGroupState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.group")
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                .state
                .clone(),
        )?;
        group.validate_checkpoint(payload)?;
        for frame in group.frames.values() {
            if let Some(old) = frames.get(&frame.id) {
                if serde_json::to_value(old)? != serde_json::to_value(frame)? {
                    return Err(AgentError::Config(
                        "group and delegate frame copies differ".into(),
                    ));
                }
            }
            frames.insert(frame.id.clone(), frame.clone());
        }
        active.extend(group.frames.values().map(|frame| frame.id.clone()));
    }
    if let Some(controller) = controller_checkpoint(payload)? {
        if let Some(value) = controller
            .driver
            .and_then(|driver| driver.parked)
            .and_then(|parked| parked.group)
        {
            let group: TaskGroupState = serde_json::from_value(value)?;
            group.validate_checkpoint(payload)?;
            for frame in group.frames.values() {
                if let Some(old) = frames.get(&frame.id) {
                    if serde_json::to_value(old)? != serde_json::to_value(frame)? {
                        return Err(AgentError::Config(
                            "observation group and delegate frame copies differ".into(),
                        ));
                    }
                }
                frames.insert(frame.id.clone(), frame.clone());
            }
            active.extend(group.frames.values().map(|frame| frame.id.clone()));
        }
    }
    active.sort();
    active.dedup();
    if active != manifest.active_frames {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    for id in &active {
        let frame = frames
            .get(id)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        for slot in frame.slots() {
            if !manifest.nodes.contains_key(&slot.runtime_id)
                || manifest.targets.get(&slot.operation).is_none_or(|target| {
                    target.runtime != slot.runtime_id || target.entry != slot.registry_id
                })
            {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
        }
    }
    let mut occupancy = BTreeMap::<String, u64>::new();
    for registry in manifest.registries.values() {
        for entry in registry.entries.values() {
            if let Some(owner) = &entry.spawner {
                *occupancy.entry(owner.clone()).or_default() += 1;
            }
        }
    }
    for (key, state) in &manifest.spawners {
        if state["version"] != 1
            || state["occupied"].as_u64() != Some(*occupancy.get(key).unwrap_or(&0))
            || state["counter"]
                .as_u64()
                .is_none_or(|counter| counter == 0 || counter > u64::from(u32::MAX))
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    host.validate_manifest(manifest)
}

// Mutable registry/capacity objects have one graph key; sharing is represented by repeated references to that key.
// Runtime provider views must resolve to their actual installed handles; hidden implementation/service aliases remain explicit host obligations.
pub(super) fn validate_source_views(graph: &TaskRestoreGraph) -> Result<()> {
    let mut registries = std::collections::BTreeSet::new();
    for registry in graph.registries.values() {
        if !registries.insert(registry.reconstruction_identity()) {
            return Err(AgentError::Config(
                "duplicate restore registry keys must be canonicalized".into(),
            ));
        }
    }
    let mut spawners = std::collections::BTreeSet::new();
    for spawner in graph.spawners.values() {
        if !spawners.insert(spawner.reconstruction_identity()) {
            return Err(AgentError::Config(
                "duplicate restore spawner keys must be canonicalized".into(),
            ));
        }
    }
    for (id, runtime) in &graph.runtimes {
        let view = graph
            .providers
            .get(&format!("runtime:{id}"))
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let actual = runtime.task_restore_providers();
        let mut aliases = view.aliases();
        aliases.sort();
        let mut installed = actual.aliases();
        installed.sort();
        if aliases != installed {
            return Err(AgentError::Config(
                "restored provider view differs from installed runtime".into(),
            ));
        }
        for alias in aliases {
            if !Arc::ptr_eq(&view.get(&alias)?, &actual.get(&alias)?) {
                return Err(AgentError::Config(
                    "restore provider view contains foreign handles".into(),
                ));
            }
        }
    }
    Ok(())
}

// The factory must produce new objects with the exact declared alias/configuration relationships, not merely matching labels.
fn validate_graph(
    graph: &TaskRestoreGraph,
    manifest: &TaskRestoreManifest,
    host: &dyn TaskRestoreHost,
) -> Result<()> {
    validate_source_views(graph)?;
    if graph.runtimes.keys().ne(manifest.nodes.keys())
        || graph.registries.keys().ne(manifest.registries.keys())
        || graph.spawners.keys().ne(manifest.spawners.keys())
        || graph.providers.keys().ne(manifest.providers.keys())
    {
        return Err(AgentError::Config(
            "staged restoration graph inventory differs".into(),
        ));
    }
    for (id, runtime) in &graph.runtimes {
        let node = &manifest.nodes[id];
        runtime.preflight_standalone_autonomy()?;
        if runtime.info().id != *id
            || runtime.captured_task_owner().is_some()
            || runtime.task_restore_configuration()? != node.configuration_digest
            || host.runtime_binding(runtime, graph.declarations.get(id))? != node.binding
        {
            return Err(AgentError::Config(
                "staged runtime binding differs from checkpoint".into(),
            ));
        }
        if runtime.spawner_registry().map(Arc::as_ptr)
            != node
                .registry
                .as_ref()
                .map(|key| Arc::as_ptr(&graph.registries[key]))
            || runtime.spawner().map(Arc::as_ptr)
                != node
                    .spawner
                    .as_ref()
                    .map(|key| Arc::as_ptr(&graph.spawners[key]))
        {
            return Err(AgentError::Config(
                "staged runtime topology handles differ".into(),
            ));
        }
    }
    for (id, spawner) in &graph.spawners {
        if spawner.reconstruction_state()?["configuration"]
            != manifest.spawners[id]["configuration"]
        {
            return Err(AgentError::Config(
                "staged spawner configuration differs".into(),
            ));
        }
    }
    for (id, providers) in &graph.providers {
        if host.provider_binding(providers)? != manifest.providers[id] {
            return Err(AgentError::Config("staged provider binding differs".into()));
        }
    }
    for (id, registry) in &graph.registries {
        if registry.reconstruction_send_context() != manifest.registries[id].send_with_context {
            return Err(AgentError::Config(
                "restored sender context policy differs".into(),
            ));
        }
        if registry.count() != 0 {
            return Err(AgentError::Config(
                "staged registry must be unpublished and empty".into(),
            ));
        }
    }
    Ok(())
}

// Each configured capacity owner is restored once; registration is silent and cannot run generation/removal hooks.
fn install_registries(graph: &TaskRestoreGraph, manifest: &TaskRestoreManifest) -> Result<()> {
    let mut reservations = BTreeMap::new();
    for (id, spawner) in &graph.spawners {
        reservations.insert(
            id.clone(),
            spawner.reconstruct_capacity(&manifest.spawners[id])?,
        );
    }
    for (id, registry) in &graph.registries {
        let mut entries = Vec::new();
        for (entry_id, entry) in &manifest.registries[id].entries {
            let declaration = graph
                .declarations
                .get(&entry.runtime)
                .cloned()
                .ok_or_else(|| {
                    AgentError::Config("restored registry declaration is absent".into())
                })?;
            declaration.validate()?;
            let reservation = entry
                .spawner
                .as_ref()
                .map(|key| {
                    reservations
                        .get_mut(key)
                        .and_then(Vec::pop)
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)
                })
                .transpose()?;
            entries.push(crate::spawner::SpawnedAgent::reconstructed(
                entry_id.clone(),
                graph.runtimes[&entry.runtime].clone(),
                declaration,
                reservation,
                entry.spawned_at,
            ));
        }
        registry.reconstruct_entries(entries)?;
    }
    if reservations.values().any(|entries| !entries.is_empty()) {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    Ok(())
}

struct HostBindingGuard {
    host: Arc<dyn TaskRestoreHost>,
    manifest: TaskRestoreManifest,
    nodes: BTreeMap<String, (std::sync::Weak<RuntimeAgent>, Option<AgentSpec>)>,
    providers: BTreeMap<String, Arc<ai_agents_llm::LLMRegistry>>,
    compare_state: bool,
}
impl std::fmt::Debug for HostBindingGuard {
    // Do not expose prepared secrets, prompt data or opaque host state in error paths.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HostBindingGuard")
    }
}
impl TaskFinalAdmission for HostBindingGuard {
    // Publication/review checks the initial component state; later admitted work may advance it but cannot silently replace immutable bindings.
    fn validate(&self) -> Result<()> {
        self.host.validate_manifest(&self.manifest)?;
        for (id, (runtime, declaration)) in &self.nodes {
            let runtime = runtime
                .upgrade()
                .ok_or_else(|| AgentError::Config("restored runtime binding was dropped".into()))?;
            let node = &self.manifest.nodes[id];
            let current = self.host.runtime_binding(&runtime, declaration.as_ref())?;
            if runtime.task_restore_configuration()? != node.configuration_digest
                || current.id != node.binding.id
                || current.contract_version != node.binding.contract_version
                || current.configuration != node.binding.configuration
                || self.compare_state && current.state != node.binding.state
            {
                return Err(AgentError::Config(
                    "restored immutable host binding changed".into(),
                ));
            }
        }
        for (id, providers) in &self.providers {
            let current = self.host.provider_binding(providers)?;
            let expected = &self.manifest.providers[id];
            if current.id != expected.id
                || current.contract_version != expected.contract_version
                || current.configuration != expected.configuration
                || self.compare_state && current.state != expected.state
            {
                return Err(AgentError::Config(
                    "restored provider binding changed".into(),
                ));
            }
        }
        Ok(())
    }
}

// Weak runtime references avoid a runtime -> owner -> response-guard -> runtime cycle.
fn host_binding_guard(
    host: Arc<dyn TaskRestoreHost>,
    manifest: &TaskRestoreManifest,
    graph: &TaskRestoreGraph,
    compare_state: bool,
    runtime: Option<&str>,
) -> Arc<dyn TaskFinalAdmission> {
    let nodes = graph
        .runtimes
        .iter()
        .filter(|(id, _)| runtime.is_none_or(|runtime| runtime == id.as_str()))
        .map(|(id, node)| {
            (
                id.clone(),
                (Arc::downgrade(node), graph.declarations.get(id).cloned()),
            )
        })
        .collect();
    let providers = graph
        .providers
        .iter()
        .filter(|(id, _)| {
            runtime.is_none_or(|runtime| {
                *id == &format!("runtime:{runtime}")
                    || manifest.compositions.iter().any(|(frame, composition)| {
                        composition.providers.as_ref() == Some(*id)
                            && manifest
                                .tools
                                .get(frame)
                                .is_some_and(|(owner, _)| owner == runtime)
                    })
            })
        })
        .map(|(id, providers)| (id.clone(), providers.clone()))
        .collect();
    Arc::new(HostBindingGuard {
        host,
        manifest: manifest.clone(),
        nodes,
        providers,
        compare_state,
    })
}

// Exact source requests are rebound to actual current implementations; an unapproved restoration cannot consume any historical receipt.
fn install_targets(
    graph: &TaskRestoreGraph,
    manifest: &TaskRestoreManifest,
    owner: &Arc<RunOwner>,
    owners: &BTreeMap<String, Arc<RunOwner>>,
    requests: &BTreeMap<String, String>,
    reviewed: Arc<AtomicBool>,
    host: Arc<dyn TaskRestoreHost>,
) -> Result<()> {
    for registry in graph.registries.values() {
        owner.targets.remember_registry(registry.clone())?;
    }
    let mut targets = Vec::new();
    for (operation, target) in &manifest.targets {
        let registry = &graph.registries[&target.registry];
        let pinned = registry.pin_task_target(&target.entry, &target.runtime)?;
        if !Arc::ptr_eq(&pinned.agent, &graph.runtimes[&target.runtime]) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        targets.push((operation.clone(), pinned));
    }
    owner.targets.install(targets)?;
    for (frame, (runtime, tool)) in &manifest.tools {
        owner
            .targets
            .bind_tool(frame, graph.runtimes[runtime].task_restore_tool(tool)?)?;
    }
    for (frame, composition) in &manifest.compositions {
        owner.targets.bind_composition(
            frame,
            graph.registries[&composition.registry].clone(),
            composition
                .providers
                .as_ref()
                .map(|key| graph.providers[key].clone()),
        )?;
    }
    for (frame, context) in &manifest.message_contexts {
        let runtime = &manifest
            .tools
            .get(frame)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .0;
        owner.targets.bind_message_context(
            frame,
            graph.runtimes[runtime].reconstruct_message_context(context)?,
        )?;
    }
    let clock = sample_clock(host.as_ref())?;
    for (id, response) in &manifest.responses {
        let runtime = &graph.runtimes[&response.runtime];
        let bound_owner = &owners[&response.runtime];
        let bound = match (response.expires_at, manifest.deadlines.get(id).copied()) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        owner.responses.lock().insert(
            requests.get(id).unwrap_or(id).clone(),
            Arc::new(crate::autonomy::response::RetainedResponse {
                runtime_id: response.runtime.clone(),
                request: response.request.clone(),
                expires_at: response.expires_at,
                deadline: deadline_projection(bound, clock)?,
                review: Some(Arc::new(RestorePublication {
                    guards: vec![
                        runtime
                            .task_reconstruction_guard(bound_owner.clone(), Some(reviewed.clone())),
                        host_binding_guard(
                            host.clone(),
                            manifest,
                            graph,
                            false,
                            Some(&response.runtime),
                        ),
                    ],
                })),
                timeout_disposition: AtomicBool::new(response.timeout_disposition),
                applied: AtomicBool::new(response.applied),
            }),
        );
    }
    for runtime in &manifest.skill_finishing {
        owner.skill_finishing.lock().insert(
            runtime.clone(),
            graph.runtimes[runtime].task_reconstruction_guard(owners[runtime].clone(), None),
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct RecoveryClock {
    utc: DateTime<Utc>,
    mono: Instant,
}

// Pair the wall sample with an earlier monotonic anchor; processing many bounds cannot renew later entries by loop time.
fn sample_clock(host: &dyn TaskRestoreHost) -> Result<RecoveryClock> {
    let mono = Instant::now();
    let utc = host.now()?;
    Ok(RecoveryClock { utc, mono })
}

// All original expiry values are projected without renewal; safely paused downtime is not charged as active work.
fn deadline_projection(
    expiry: Option<DateTime<Utc>>,
    now: RecoveryClock,
) -> Result<Option<Instant>> {
    expiry
        .map(|expiry| {
            now.mono
                .checked_add((expiry - now.utc).to_std().unwrap_or_default())
                .ok_or_else(|| TaskRunStorageError::AdmissionExpired.into())
        })
        .transpose()
}

// Extracts exact acknowledged paused positions and privately rehydrates only the selected validation attempt.
fn restored_pause(
    runner: &AutonomyRunner,
    payload: &TaskCheckpointPayload,
    owner: Arc<RunOwner>,
    participants: Arc<Participants>,
    review: LiveRestoreReview,
    manifest: &TaskRestoreManifest,
    now: RecoveryClock,
) -> Result<PausedTask> {
    let mut controller = controller_checkpoint(payload)?;
    if let Some(controller) = &mut controller {
        if let Some(driver) = &controller.driver {
            let profile = runner.retained_profile(payload)?;
            let bound = runner
                .agent
                .autonomy_extensions()
                .bind_for_agent(&runner.agent, &profile)?;
            let check = controller.driver_check(&bound, &payload.objective)?;
            controller.driver = Some(Box::new(ValidationDriverState::restore_reconstructed(
                serde_json::to_value(driver)?,
                &check,
                &controller.scope,
                runner.agent.autonomy_extensions(),
                now.utc,
                manifest
                    .validation_bound
                    .as_ref()
                    .filter(|(identity, _)| *identity == driver.identity)
                    .map(|(_, bound)| *bound),
                now.mono,
            )?));
        }
    }
    let group =
        if controller.is_none() && payload.pause_reason.as_deref() == Some("child_interaction") {
            Some(serde_json::from_value::<TaskGroupState>(
                payload
                    .adapters
                    .iter()
                    .find(|adapter| adapter.id == "runtime.group")
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?
                    .state
                    .clone(),
            )?)
        } else {
            None
        };
    let batch = if controller.is_none() && group.is_none() {
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.tool_batch")
            .map(|adapter| serde_json::from_value::<TaskBatchState>(adapter.state.clone()))
            .transpose()?
    } else {
        None
    };
    if let Some(group) = &group {
        group.validate_checkpoint(payload)?;
        runner.agent.task_group_child(group, &owner.targets)?;
    }
    if let Some(batch) = &batch {
        batch.validate_checkpoint(payload)?;
        if let Some(location) = &batch.location {
            runner.agent.validate_task_location(location)?;
        }
        if let Some(interaction) = &batch.interaction {
            runner.agent.validate_task_interaction(interaction)?;
        }
    }
    let mut deadlines = BTreeMap::new();
    let mut batches = Vec::new();
    if let Some(batch) = &batch {
        batches.push(batch.clone());
    }
    if let Some(group) = &group {
        batches.push(group.batch.clone());
        for leaf in &group.parked {
            batches.push(leaf.batch.clone());
        }
        for frame in group.frames.values() {
            if let Some(deadline) = deadline_projection(frame.expires_at, now)? {
                deadlines.insert(format!("composition:{}", frame.id), deadline);
            }
        }
    }
    if let Some(controller) = &controller {
        if let Some(review) = &controller.review {
            if let Some(deadline) = deadline_projection(review.expires_at, now)? {
                deadlines.insert(review.id.clone(), deadline);
            }
        }
        if let Some(value) = controller
            .driver
            .as_ref()
            .and_then(|driver| driver.parked.as_ref())
            .and_then(|parked| parked.group.as_ref())
        {
            let group: TaskGroupState = serde_json::from_value(value.clone())?;
            batches.push(group.batch.clone());
            for leaf in &group.parked {
                batches.push(leaf.batch.clone());
            }
            for frame in group.frames.values() {
                if let Some(deadline) = deadline_projection(frame.expires_at, now)? {
                    deadlines.insert(format!("composition:{}", frame.id), deadline);
                }
            }
        }
    }
    for batch in &batches {
        for approval in &batch.approvals {
            if let Some(deadline) = deadline_projection(approval.expires_at, now)? {
                deadlines.insert(approval.request_id.clone(), deadline);
            }
        }
    }
    if let Some(group) = &group {
        if let Some(deadline) = deadlines.get(&group.child_request.id).copied() {
            deadlines.insert(group.request_id.clone(), deadline);
        }
    }
    let todo = payload
        .todos
        .clone()
        .map(|checkpoint| {
            RunTodoAdapter::restore(
                runner.agent.todo_store(),
                &controller
                    .as_ref()
                    .map(|controller| controller.scope.key.clone())
                    .unwrap_or(TaskRunKey {
                        agent_id: runner.agent.info().id,
                        run_id: owner.run_id.clone(),
                    }),
                checkpoint,
            )
        })
        .transpose()?;
    if let Some(todo) = &todo {
        todo.set_open_limit(
            payload
                .settings
                .progress
                .as_ref()
                .and_then(|progress| progress.max_open_todos),
        )?;
    }
    Ok(PausedTask {
        owner: owner.clone(),
        participants,
        targets: owner.targets.clone(),
        todo,
        batch,
        group,
        controller,
        whole_expiry: deadline_projection(payload.clocks.expires_at, now)?,
        approval_deadlines: deadlines,
        restoration: Some(review),
    })
}

impl AutonomyRunner {
    /// Serialized review flags cannot replace the newly installed live review witness before task execution.
    pub(crate) fn ensure_restoration_review(
        &self,
        payload: &TaskCheckpointPayload,
        run_id: &str,
    ) -> Result<()> {
        if let Some(adapter) = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == RESTORE_REVIEW)
        {
            let view: TaskRestorationReview = serde_json::from_value(adapter.state.clone())?;
            let paused = self.paused.lock();
            let live = paused
                .get(run_id)
                .and_then(|paused| paused.restoration.as_ref())
                .ok_or(TaskRunStorageError::NotResumable)?;
            if !view.approved || !live.approved.load(Ordering::Acquire) || live.view.id != view.id {
                return Err(AgentError::Config(
                    "restoration requires fresh explicit host review before resume".into(),
                ));
            }
        }
        Ok(())
    }

    /// Reconstructs only an explicit, prepared safe pause and returns a fresh revision without executing or consuming a response.
    /// Claim/staging/ACK errors leave quarantine rather than automatically restoring the old claim or adopting an abandoned owner.
    pub async fn restore_paused(
        request: TaskRestoreRequest,
        store: Arc<dyn TaskRunStore>,
        host: Arc<dyn TaskRestoreHost>,
    ) -> Result<(Self, TaskRunResult)> {
        if !store.supports_reconstruction() {
            return Err(AgentError::Config(
                "task store lacks reconstruction admission".into(),
            ));
        }
        let started = Instant::now();
        let started_at = host.now()?;
        let snapshot = store
            .load(&request.run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &request.config_identity)?;
        safe_pause(&snapshot, &payload, host.now()?)?;
        if snapshot.revision != request.expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let manifest: TaskRestoreManifest = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == RESTORE_ADAPTER)
                .ok_or_else(|| {
                    AgentError::Config("checkpoint lacks an explicit restore manifest".into())
                })?
                .state
                .clone(),
        )?;
        validate_manifest(&manifest, &snapshot, &payload, host.as_ref())
            .map_err(|error| AgentError::Config(format!("restoration preflight: {error}")))?;
        validate_current_host(&request, &payload)?;
        let budget = host.reconstruction_budget();
        let expires = match (payload.clocks.expires_at, manifest.whole_expiry) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        let whole = deadline_projection(expires, sample_clock(host.as_ref())?)?;
        let deadline = reconstruction_deadline(started, budget, whole)?;
        charge_reconstruction(&mut payload, budget)?;
        mark_work(&mut payload, "restore", "paused", budget, started_at)?;
        let token = uuid::Uuid::new_v4().to_string();
        let claimed = store
            .claim_paused(&request.run_id, snapshot.revision, token.clone())
            .await?
            .ok_or(TaskRunStorageError::Conflict)?;
        super::super::super::store::verify_checkpoint_receipt(
            &snapshot,
            &TaskRunMutation::Claim {
                expected_revision: snapshot.revision,
                owner_token: token.clone(),
            },
            &claimed,
        )?;
        let claimed = begin_work(
            &store, &claimed, &payload, "restore", budget, started_at, deadline,
        )
        .await?;
        let graph = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            host.stage(&manifest),
        )
        .await
        .map_err(|_| TaskRunStorageError::AdmissionExpired)??;
        if Instant::now() >= deadline {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        validate_graph(&graph, &manifest, host.as_ref())?;
        if Instant::now() >= deadline {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        let root = graph.runtimes[&manifest.root].clone();
        let runner = Self::try_new(
            root,
            request.config,
            request.ceilings,
            store.clone(),
            request.config_identity,
        )?;
        let mut owners = BTreeMap::new();
        let owner = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            runner
                .agent
                .reserve_reconstructed_owner(request.run_id.clone(), snapshot.actor_id.clone()),
        )
        .await
        .map_err(|_| TaskRunStorageError::AdmissionExpired)??;
        let mut cleanup = OwnedTurnCleanup::new(owner.clone());
        owners.insert(manifest.root.clone(), owner.clone());
        let participants = Arc::new(Participants::default());
        for (id, runtime) in &graph.runtimes {
            if *id != manifest.root {
                let node = &manifest.nodes[id];
                let child = tokio::time::timeout_at(
                    tokio::time::Instant::from_std(deadline),
                    runtime.reserve_reconstructed_owner(
                        request.run_id.clone(),
                        node.participant_actor
                            .as_ref()
                            .and_then(|participant| participant.actor_id.clone()),
                    ),
                )
                .await
                .map_err(|_| TaskRunStorageError::AdmissionExpired)??;
                runtime.install_reconstructed_owner(child.clone(), &participants)?;
                owners.insert(id.clone(), child);
            }
        }
        for (id, runtime) in &graph.runtimes {
            let node = &manifest.nodes[id];
            tokio::time::timeout_at(
                tokio::time::Instant::from_std(deadline),
                runtime.apply_task_reconstruction(&owners[id], &node.runtime, &node.session),
            )
            .await
            .map_err(|_| TaskRunStorageError::AdmissionExpired)??;
            if Instant::now() >= deadline {
                return Err(TaskRunStorageError::AdmissionExpired.into());
            }
            if let Some(skill) = &node.skill {
                if !runtime.skills().iter().any(|installed| {
                    serde_json::to_value(installed).ok() == serde_json::to_value(skill).ok()
                }) {
                    return Err(AgentError::Config(
                        "restored admitted skill is not installed".into(),
                    ));
                }
                *owners[id].skill.write() = Some(skill.clone());
            }
        }
        install_registries(&graph, &manifest)?;
        participants.restore_ordinals(manifest.operation_ordinals.clone())?;
        let mut rebinding = super::bindings::Rebinding {
            manifest: &manifest,
            graph: &graph,
            revision: claimed.revision + 1,
            requests: BTreeMap::new(),
        };
        rebinding.payload(&mut payload).map_err(|error| {
            AgentError::Config(format!("restoration cursor rebinding: {error}"))
        })?;
        let requests = rebinding.requests;
        let approved = Arc::new(AtomicBool::new(false));
        install_targets(
            &graph,
            &manifest,
            &owner,
            &owners,
            &requests,
            approved.clone(),
            host.clone(),
        )?;

        let action_digest = review_digest(&payload)?;
        let view = TaskRestorationReview {
            id: uuid::Uuid::new_v4().to_string(),
            approved: false,
            action_digest,
        };
        if let Some(adapter) = payload
            .adapters
            .iter_mut()
            .find(|adapter| adapter.id == RESTORE_REVIEW)
        {
            adapter.state = serde_json::to_value(&view)?;
        } else {
            payload.adapters.push(TaskAdapterCheckpoint {
                id: RESTORE_REVIEW.into(),
                adapter: RESTORE_REVIEW.into(),
                contract_version: 1,
                config: json!({"kind":"fresh_restoration_review"}),
                state: serde_json::to_value(&view)?,
            });
        }
        let mut all_guards: Vec<_> = graph
            .runtimes
            .iter()
            .map(|(id, runtime)| runtime.task_reconstruction_guard(owners[id].clone(), None))
            .collect();
        let mut review_guards: Vec<_> = graph
            .runtimes
            .iter()
            .filter(|(id, _)| {
                **id == manifest.root || manifest.nodes[*id].participant_actor.is_some()
            })
            .map(|(id, runtime)| runtime.task_reconstruction_guard(owners[id].clone(), None))
            .collect();
        all_guards.push(host_binding_guard(
            host.clone(),
            &manifest,
            &graph,
            true,
            None,
        ));
        review_guards.push(host_binding_guard(
            host.clone(),
            &manifest,
            &graph,
            true,
            None,
        ));
        let live_review = LiveRestoreReview {
            host: host.clone(),
            budget,
            view,
            approved,
            guards: review_guards,
        };
        let clock = sample_clock(host.as_ref())?;
        let mut paused = restored_pause(
            &runner,
            &payload,
            owner.clone(),
            participants.clone(),
            live_review,
            &manifest,
            clock,
        )
        .map_err(|error| AgentError::Config(format!("restoration paused validation: {error}")))?;
        if let Some(whole) = whole {
            paused.whole_expiry = Some(
                paused
                    .whole_expiry
                    .map_or(whole, |current| current.min(whole)),
            );
        }
        for (old, cut) in &manifest.deadlines {
            if let Some(deadline) = deadline_projection(Some(*cut), clock)? {
                let id = requests.get(old).unwrap_or(old).clone();
                paused
                    .approval_deadlines
                    .entry(id)
                    .and_modify(|current| *current = (*current).min(deadline))
                    .or_insert(deadline);
            }
        }
        // The committed restored pause is itself a complete safe restart point, including after a lost final ACK.
        // Capture data under claim custody without treating still-pending fresh review guards as execution approval.
        let mut refreshed = capture_manifest(
            &runner,
            &paused,
            &payload,
            host.as_ref(),
            claimed.revision + 1,
            false,
        )
        .await?;
        for (id, node) in &mut refreshed.nodes {
            node.participant_actor = manifest.nodes[id].participant_actor.clone();
        }
        host.validate_manifest(&refreshed)?;
        payload
            .adapters
            .iter_mut()
            .find(|adapter| adapter.id == RESTORE_ADAPTER)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .state = serde_json::to_value(&refreshed)?;
        let digest = review_digest(&payload)?;
        let live = paused
            .restoration
            .as_mut()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        live.view.action_digest = digest;
        payload
            .adapters
            .iter_mut()
            .find(|adapter| adapter.id == RESTORE_REVIEW)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .state = serde_json::to_value(&live.view)?;
        let execution = claimed_execution(&runner, &claimed, &paused)?;
        *runner.active.lock() = Some(execution.clone());
        let final_guard: Arc<dyn TaskFinalAdmission> =
            Arc::new(RestorePublication { guards: all_guards });
        safe_claim_time(&snapshot, &payload, host.now()?)?;
        let checkpoint = TaskReconstructionCheckpoint {
            previous: claimed.clone(),
            mutation: TaskRunMutation::FinalCheckpoint {
                expected_revision: claimed.revision,
                owner_token: token,
                status: TaskRunStatus::Paused,
                payload: serde_json::to_value(&payload)?,
                release: true,
                deadline,
                expires_at: payload.clocks.expires_at,
                evidence: Some(final_guard),
            },
        };
        let saved = store
            .checkpoint_reconstruction(&request.run_id, &checkpoint)
            .await
            .map_err(|error| {
                AgentError::Config(format!("restoration checkpoint publication: {error}"))
            })?;
        super::super::super::store::verify_checkpoint_receipt(
            &claimed,
            &checkpoint.mutation,
            &saved,
        )?;
        execution.acknowledge_snapshot(&saved);
        for (id, runtime) in &graph.runtimes {
            if *id != manifest.root && manifest.nodes[id].participant_actor.is_none() {
                runtime.release_reconstruction_temporary(&owners[id])?;
                participants.retire_reconstructed_member(&owners[id])?;
            }
        }
        runner.paused.lock().insert(request.run_id, paused);
        cleanup.finish();
        let result = TaskRunResult {
            run: TaskRun::from_checkpoint(&saved, &runner.config_identity)?,
            final_response: None,
        };
        Ok((runner, result))
    }

    /// Returns a bounded fresh-review identity; full checkpoint data remains available only through authorized host storage access.
    pub async fn restoration_review(&self, run_id: &str) -> Result<Option<TaskRestorationReview>> {
        let paused = self.paused.lock();
        Ok(paused
            .get(run_id)
            .ok_or(TaskRunStorageError::NotResumable)?
            .restoration
            .as_ref()
            .map(|review| review.view.clone()))
    }

    /// Fresh restoration approval covers exactly the reconstructed ongoing operations; pending tool/question replies remain separate.
    /// Modification and timeout are not blanket approvals, and a policy/generation change invalidates the review before a write.
    pub async fn approve_restoration(
        &self,
        run_id: &str,
        expected_revision: u64,
        review_id: &str,
        decision: ai_agents_hitl::ApprovalResult,
    ) -> Result<TaskRunResult> {
        let gate = self.resume_gate.lock().await;
        let started = Instant::now();
        let clock_host = self
            .paused
            .lock()
            .get(run_id)
            .and_then(|paused| paused.restoration.as_ref())
            .map(|review| review.host.clone())
            .ok_or(TaskRunStorageError::NotResumable)?;
        let started_at = clock_host.now()?;
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        safe_pause(&snapshot, &payload, Utc::now())?;
        if snapshot.revision != expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let (owner, flag, guards, budget, whole, host) = {
            let paused = self.paused.lock();
            let paused = paused
                .get(run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            let review = paused
                .restoration
                .as_ref()
                .ok_or(TaskRunStorageError::NotResumable)?;
            if review.view.id != review_id || review.view.approved {
                return Err(TaskRunStorageError::NotResumable.into());
            }
            (
                paused.owner.clone(),
                review.approved.clone(),
                review.guards.clone(),
                review.budget,
                paused.whole_expiry,
                review.host.clone(),
            )
        };
        if matches!(decision, ai_agents_hitl::ApprovalResult::Rejected { .. }) {
            drop(gate);
            return self.cancel_paused(run_id, expected_revision).await;
        }
        if !matches!(decision, ai_agents_hitl::ApprovalResult::Approved { .. }) {
            return Err(AgentError::Config(
                "restoration requires explicit approval or rejection".into(),
            ));
        }
        owner.check()?;
        for guard in &guards {
            guard.validate()?;
        }
        let action_digest = review_digest(&payload)?;
        let adapter = payload
            .adapters
            .iter_mut()
            .find(|adapter| adapter.id == RESTORE_REVIEW)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let mut view: TaskRestorationReview = serde_json::from_value(adapter.state.clone())?;
        if view.action_digest != action_digest {
            return Err(AgentError::Config(
                "restoration review action set changed".into(),
            ));
        }
        if view.id != review_id || view.approved {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        view.approved = true;
        adapter.state = serde_json::to_value(&view)?;
        let deadline = reconstruction_deadline(started, budget, whole)?;
        charge_reconstruction(&mut payload, budget)?;
        mark_work(&mut payload, "review", "paused", budget, started_at)?;
        let mut cleanup = OwnedTurnCleanup::new(owner);
        let claimed = self.claim_retained(&snapshot, &mut cleanup).await?;
        let claimed = begin_work(
            &self.store,
            &claimed,
            &payload,
            "review",
            budget,
            started_at,
            deadline,
        )
        .await?;
        let mut paused = self
            .paused
            .lock()
            .remove(run_id)
            .ok_or(TaskRunStorageError::Conflict)?;
        let refreshed = capture_manifest(
            self,
            &paused,
            &payload,
            host.as_ref(),
            claimed.revision + 1,
            false,
        )
        .await?;
        host.validate_manifest(&refreshed)?;
        payload
            .adapters
            .iter_mut()
            .find(|adapter| adapter.id == RESTORE_ADAPTER)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .state = serde_json::to_value(refreshed)?;
        let execution = claimed_execution(self, &claimed, &paused)?;
        *self.active.lock() = Some(execution.clone());
        let mutation = TaskRunMutation::FinalCheckpoint {
            expected_revision: claimed.revision,
            owner_token: claimed
                .owner_token
                .clone()
                .ok_or(TaskRunStorageError::Owned)?,
            status: TaskRunStatus::Paused,
            payload: serde_json::to_value(payload)?,
            release: true,
            deadline,
            expires_at: execution.expiry(),
            evidence: Some(Arc::new(RestorePublication { guards })),
        };
        let saved = self.store.mutate(run_id, &mutation).await?;
        super::super::super::store::verify_checkpoint_receipt(&claimed, &mutation, &saved)?;
        execution.acknowledge_snapshot(&saved);
        flag.store(true, Ordering::Release);
        paused
            .restoration
            .as_mut()
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .view = view;
        self.paused.lock().insert(run_id.into(), paused);
        cleanup.finish();
        Ok(TaskRunResult {
            run: TaskRun::from_checkpoint(&saved, &self.config_identity)?,
            final_response: None,
        })
    }
}

// Time samples are checked again after every potentially awaited staging boundary before publication.
fn safe_claim_time(
    snapshot: &TaskRunSnapshot,
    payload: &TaskCheckpointPayload,
    now: DateTime<Utc>,
) -> Result<()> {
    if now < snapshot.updated_at
        || payload
            .clocks
            .expires_at
            .is_some_and(|expiry| now >= expiry)
    {
        return Err(TaskRunStorageError::AdmissionExpired.into());
    }
    Ok(())
}
