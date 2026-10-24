//! Executable reconstruction uses fresh objects, real stores and isolated effect witnesses, never retained owner Arcs.

#[path = "restoration_custody_tests.rs"]
mod custody_tests;
#[path = "restoration_topology_tests.rs"]
mod topology_tests;
use super::*;
use crate::{
    AgentBuilder, RuntimeAgent,
    spawner::{AgentRegistry, AgentSpawner},
    spec::AgentSpec,
};
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
    Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

// Opaque fixture replay state stays in exact runtime history; the fake endpoint program is fixed and needs no network.
fn fixture_native(messages: &[ChatMessage], mut calls: Vec<ai_agents_core::ToolCall>) -> String {
    use sha2::{Digest, Sha256};
    let identity = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(messages).unwrap())
    );
    for call in &mut calls {
        call.id = format!("{}:{}", call.id, &identity[..16]);
    }
    let state = ai_agents_core::NativeProviderState::new(
        format!("restore-{}", &identity[..16]),
        "fixture",
        "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://restore.invalid", "fixture-model-v1")
            .unwrap(),
        json!({"signature":"opaque-restoration-fixture"}),
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| ai_agents_core::NativeCallBinding::new(&call.id, index).unwrap())
            .collect(),
    )
    .unwrap();
    ai_agents_core::encode_native_tool_call_markers(&calls, Some(&state)).unwrap()
}

struct RestoreProvider {
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl LLMProvider for RestoreProvider {
    // This provider is request-state driven, so behavioral state is in exact native history rather than a hidden response queue.
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let start = messages
            .iter()
            .rposition(|message| matches!(message.role, ai_agents_core::Role::User))
            .unwrap_or(0);
        let replied = messages[start..]
            .iter()
            .any(|message| matches!(message.role, ai_agents_core::Role::Function));
        let systems = messages
            .iter()
            .filter(|message| matches!(message.role, ai_agents_core::Role::System))
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let has_result = |name: &str| {
            messages[start..].iter().any(|message| {
                matches!(message.role, ai_agents_core::Role::Function)
                    && message.name.as_deref() == Some(name)
            })
        };
        let model_call = |id: &str, tool: &str, arguments: Value| {
            fixture_native(
                messages,
                vec![ai_agents_core::ToolCall {
                    id: id.into(),
                    name: tool.into(),
                    arguments,
                }],
            )
        };
        let content = if systems.contains("restore-skill-root")
            || systems.contains("restore-state-root")
        {
            "done: saved work".into()
        } else if messages.len() == 1
            && messages[0]
                .content
                .starts_with("Generate a valid YAML agent specification.")
        {
            "name: Worker\nsystem_prompt: restore-generated-leaf\ntools: [ask_user]\n".into()
        } else if systems.contains("restore-generation-root") {
            if !has_result("spawn_agent") {
                model_call(
                    "generate",
                    "spawn_agent",
                    json!({"name":"Worker","description":"fixture evidence"}),
                )
            } else if !has_result("pipeline_process") {
                let mut target = "worker".to_string();
                for message in &messages[start..] {
                    if let Some(results) =
                        ai_agents_core::decode_native_tool_result_markers(&message.content)
                            .unwrap_or(None)
                    {
                        for result in results {
                            if result.tool() == "spawn_agent" {
                                let value = result
                                    .output()
                                    .as_str()
                                    .and_then(|text| serde_json::from_str::<Value>(text).ok())
                                    .unwrap_or_else(|| result.output().clone());
                                if let Some(id) = value["id"].as_str() {
                                    target = id.into();
                                }
                            }
                        }
                    }
                }
                model_call(
                    "compose",
                    "pipeline_process",
                    json!({"input":"evidence","stages":[target]}),
                )
            } else {
                "done: generated evidence".into()
            }
        } else if systems.contains("restore-model-pipeline") && !replied {
            model_call(
                "compose",
                "pipeline_process",
                json!({"input":"evidence","stages":["worker","peer"]}),
            )
        } else if systems.contains("restore-model-concurrent") && !replied {
            model_call(
                "compose",
                "concurrent_ask",
                json!({"question":"evidence","agents":["worker","peer"],"aggregation":"all"}),
            )
        } else if systems.contains("restore-model-send") && !replied {
            model_call(
                "send",
                "send_agent_message",
                json!({"to":"worker","message":"evidence"}),
            )
        } else if systems.contains("restore-model-route") {
            let count = messages[start..]
                .iter()
                .filter(|message| {
                    matches!(message.role, ai_agents_core::Role::Function)
                        && message.name.as_deref() == Some("route_to_agent")
                })
                .count();
            if count < 2 {
                model_call(
                    if count == 0 {
                        "route-first"
                    } else {
                        "route-second"
                    },
                    "route_to_agent",
                    json!({"input":"evidence","candidates":["worker","peer"],"method":"round_robin"}),
                )
            } else {
                "done: routed evidence".into()
            }
        } else if replied {
            "done: verified evidence".to_string()
        } else {
            let mut calls = vec![ai_agents_core::ToolCall {
                id: "question".into(),
                name: "ask_user".into(),
                arguments: json!({"question":"Confirm","options":["yes"],"allow_other":false}),
            }];
            if !systems.contains("restore-generated-leaf") {
                calls.push(ai_agents_core::ToolCall {
                    id: "effect".into(),
                    name: "restore_probe".into(),
                    arguments: json!({}),
                });
            }
            fixture_native(messages, calls)
        };
        Ok(LLMResponse::new(content, FinishReason::Stop))
    }
    // Blocking fixtures must not silently use a different execution surface.
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Err(LLMError::Config(
            "unexpected restoration fixture stream".into(),
        ))
    }
    // Stable versioned provider behavior is covered by the installed test host contract, not this name alone.
    fn provider_name(&self) -> &str {
        "restore-request-state-v1"
    }
    // Exercise the shared legacy completion wrapper rather than optional native network adapters.
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}
struct RestoreProbe {
    effects: Arc<AtomicUsize>,
    events: Arc<parking_lot::Mutex<Vec<String>>>,
    runtime: String,
    file: Option<std::path::PathBuf>,
}
#[async_trait]
impl Tool for RestoreProbe {
    // The canonical capability is installed by the host and separately granted by YAML.
    fn id(&self) -> &str {
        "restore_probe"
    }
    // Presentation is deliberately separate from the execution witness.
    fn name(&self) -> &str {
        "Restore probe"
    }
    // This fixture performs only an isolated in-memory observation effect.
    fn description(&self) -> &str {
        "Record an isolated fixture effect"
    }
    // No model arguments can choose a filesystem or network destination.
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    // Logical fixture output is safe to run concurrently with the independent question.
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    // Only a real executor body increments the witness; a tool-shaped response does not.
    async fn execute(&self, _: Value, context: ToolExecutionContext) -> ToolResult {
        if let Some(path) = &self.file {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap();
            writeln!(file, "{}:{}", self.runtime, context.call_id).unwrap();
            file.sync_all().unwrap();
        }
        self.events.lock().push(self.runtime.clone());
        self.effects.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("effect committed")
    }
}
pub(super) struct RestoreHost {
    pub(super) root: AgentSpec,
    pub(super) calls: Arc<AtomicUsize>,
    pub(super) effects: Arc<AtomicUsize>,
    events: Arc<parking_lot::Mutex<Vec<String>>>,
    effect_file: Option<std::path::PathBuf>,
    routes: parking_lot::Mutex<
        std::collections::BTreeMap<
            usize,
            std::sync::Weak<crate::orchestration::tools::RouteToAgentTool>,
        >,
    >,
    registrations: Arc<AtomicUsize>,
    pub(super) stages: AtomicUsize,
    pub(super) binding_version: AtomicUsize,
    pub(super) staged:
        parking_lot::Mutex<std::collections::BTreeMap<String, std::sync::Weak<RuntimeAgent>>>,
    memory_gate: Option<Arc<super::restoration_test_support::MemoryGate>>,
    pub(super) budget_millis: std::sync::atomic::AtomicU64,
    pub(super) stage_hold: std::sync::atomic::AtomicBool,
    pub(super) stage_reached: tokio::sync::Semaphore,
    pub(super) stage_release: tokio::sync::Semaphore,
    clock: parking_lot::Mutex<Option<chrono::DateTime<chrono::Utc>>>,
    clock_failure: std::sync::atomic::AtomicBool,
    contract: &'static str,
}
struct RegistrationWitness(Arc<AtomicUsize>);
#[async_trait]
impl crate::spawner::RegistryHooks for RegistrationWitness {
    // Only real source registration, not silent reconstruction, produces the business-lifecycle witness.
    async fn on_agent_spawned(&self, _: &str, _: &AgentSpec) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl RestoreHost {
    // Shared endpoint code is stateless with respect to requests; all execution accounting remains in the framework wrapper.
    fn provider_registry(&self) -> ai_agents_llm::LLMRegistry {
        let provider = Arc::new(RestoreProvider {
            calls: self.calls.clone(),
        });
        let mut registry = ai_agents_llm::LLMRegistry::new();
        registry.register("default", provider.clone());
        registry.register("router", provider);
        registry
    }
    // Registry hooks have an explicit isolated witness and must never be replayed by restoration.
    fn registry(&self, context: bool) -> Arc<AgentRegistry> {
        Arc::new(
            AgentRegistry::new()
                .with_send_context(context)
                .with_hooks(Arc::new(RegistrationWitness(self.registrations.clone()))),
        )
    }
    // Source and reconstructed builders use the same grants and managed wrappers but never share a runtime or owner.
    pub(super) fn runtime(&self, spec: AgentSpec) -> Arc<RuntimeAgent> {
        self.runtime_with_handles(spec, None, None, None, None)
    }

    // Runtime and tool-specific registry handles are distinct inputs, matching the framework's captured-composition contract.
    fn runtime_with_handles(
        &self,
        spec: AgentSpec,
        runtime_registry: Option<Arc<AgentRegistry>>,
        spawner: Option<Arc<AgentSpawner>>,
        tool_registry: Option<Arc<AgentRegistry>>,
        state: Option<&Value>,
    ) -> Arc<RuntimeAgent> {
        let id = spec.name.clone();
        let tools: Vec<String> = spec
            .tools
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| entry.name().to_string())
                    .collect()
            })
            .unwrap_or_default();
        let mut builder = AgentBuilder::from_spec(spec).llm_registry(self.provider_registry());
        if let Some(gate) = &self.memory_gate {
            builder = builder.memory(gate.memory(id.clone()));
        }
        if tools.iter().any(|tool| tool == "restore_probe") {
            builder = builder.tool(Arc::new(RestoreProbe {
                effects: self.effects.clone(),
                events: self.events.clone(),
                runtime: id.clone(),
                file: self.effect_file.clone(),
            }));
        }
        builder = builder.auto_configure_features().unwrap().tool(Arc::new(
            ai_agents_tools::builtin::AskUserTool::new(Arc::new(parking_lot::RwLock::new(None))),
        ));
        builder.prepare_routing(false).unwrap();
        builder.wrap_llm_registry_for_observability().unwrap();
        builder.prepare_routing(true).unwrap();
        let prepared = builder.prepared_registry().unwrap().clone();
        let spawner = spawner.or_else(|| {
            runtime_registry.as_ref().map(|_| {
                let mut spawner = AgentSpawner::new()
                    .with_shared_llms(prepared.clone())
                    .with_max_agents(3);
                if tools.iter().any(|tool| tool == "spawn_agent") {
                    spawner = spawner.with_name_prefix("generated_").unwrap();
                }
                Arc::new(spawner)
            })
        });
        let registry = tool_registry.or_else(|| runtime_registry.clone());
        let mut route = None;
        if let Some(registry) = &registry {
            for tool in &tools {
                builder = match tool.as_str() {
                    "pipeline_process" => builder.tool(Arc::new(
                        crate::orchestration::tools::PipelineProcessTool::new(registry.clone()),
                    )),
                    "concurrent_ask" => builder.tool(Arc::new(
                        crate::orchestration::tools::ConcurrentAskTool::new(
                            registry.clone(),
                            Arc::new(prepared.clone()),
                        ),
                    )),
                    "group_discussion" => builder.tool(Arc::new(
                        crate::orchestration::tools::GroupDiscussionTool::new(
                            registry.clone(),
                            Arc::new(prepared.clone()),
                        ),
                    )),
                    "handoff_conversation" => builder.tool(Arc::new(
                        crate::orchestration::tools::HandoffConversationTool::new(
                            registry.clone(),
                            Arc::new(prepared.clone()),
                        ),
                    )),
                    "route_to_agent" => {
                        let position = state
                            .and_then(|state| state["round_robin"].as_u64())
                            .map(|position| usize::try_from(position).unwrap())
                            .unwrap_or(0);
                        let tool = Arc::new(
                            crate::orchestration::tools::RouteToAgentTool::new(
                                registry.clone(),
                                Arc::new(prepared.clone()),
                            )
                            .with_round_robin_position(position),
                        );
                        route = Some(Arc::downgrade(&tool));
                        builder.tool(tool)
                    }
                    "send_agent_message" => builder.tool(Arc::new(
                        crate::spawner::SendMessageTool::new(registry.clone(), id.clone()),
                    )),
                    "spawn_agent" => {
                        let mut generation = ai_agents_llm::LLMRegistry::new();
                        generation.register("default", prepared.get("default").unwrap());
                        builder.tool(Arc::new(crate::spawner::GenerateAgentTool::new(
                            spawner.as_ref().unwrap().clone(),
                            registry.clone(),
                            Arc::new(generation),
                        )))
                    }
                    _ => builder,
                };
            }
        }
        let mut runtime = builder.build().unwrap();
        if let (Some(registry), Some(spawner)) = (runtime_registry, spawner) {
            runtime = runtime.with_spawner_handles(spawner, registry);
        }
        let runtime = Arc::new(runtime);
        if let Some(route) = route {
            self.routes
                .lock()
                .insert(Arc::as_ptr(&runtime) as usize, route);
        }
        runtime
    }
}
#[async_trait]
impl TaskRestoreHost for RestoreHost {
    // Identity refers to installed code and its complete test program, not user-provided checkpoint labels.
    fn identity(&self) -> (String, u32) {
        (self.contract.into(), 1)
    }
    // The controlled fixture explicitly trusts the common system clock; rollback/failure fixtures override this host contract.
    fn now(&self) -> ai_agents_core::Result<chrono::DateTime<chrono::Utc>> {
        if self.clock_failure.load(Ordering::SeqCst) {
            return Err(ai_agents_core::AgentError::Config(
                "clock continuity is not certified".into(),
            ));
        }
        Ok(self.clock.lock().unwrap_or_else(chrono::Utc::now))
    }

    // This explicit ceiling includes all construction and backend publication time; it is charged rather than silently refunded.
    fn reconstruction_budget(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.budget_millis.load(Ordering::SeqCst))
    }
    // Configuration and opaque service behavior are explicit; invocation counters are telemetry, not provider behavioral state.
    fn runtime_binding(
        &self,
        runtime: &RuntimeAgent,
        declaration: Option<&AgentSpec>,
    ) -> ai_agents_core::Result<TaskRestoreBinding> {
        Ok(TaskRestoreBinding {
            id: self.contract.into(),
            contract_version: 1,
            configuration: json!({"program":"request-state-v1","spec":declaration.unwrap_or(&self.root),"effect_scope":self.effect_file,"binding_version":self.binding_version.load(Ordering::SeqCst)}),
            state: self
                .routes
                .lock()
                .get(&(runtime as *const RuntimeAgent as usize))
                .and_then(std::sync::Weak::upgrade)
                .map(|route| json!({"round_robin":route.round_robin_position()}))
                .unwrap_or(Value::Null),
        })
    }
    // The deterministic endpoint program has no hidden service state or network credentials.
    fn provider_binding(
        &self,
        _: &ai_agents_llm::LLMRegistry,
    ) -> ai_agents_core::Result<TaskRestoreBinding> {
        Ok(TaskRestoreBinding {
            id: self.contract.into(),
            contract_version: 1,
            configuration: json!({"backend":"offline-request-state-v1"}),
            state: Value::Null,
        })
    }
    // Pure preflight rejects host substitution before claim, object construction or invocation.
    fn validate_manifest(&self, manifest: &TaskRestoreManifest) -> ai_agents_core::Result<()> {
        if manifest.host != self.identity()
            || manifest.nodes.get(&manifest.root).is_none_or(|node| {
                node.binding.configuration["spec"] != serde_json::to_value(&self.root).unwrap()
            })
            || manifest.nodes.values().any(|node| {
                node.binding.id != self.contract
                    || node.binding.configuration["program"] != "request-state-v1"
            })
        {
            return Err(ai_agents_core::AgentError::Config(
                "test host binding mismatch".into(),
            ));
        }
        for node in manifest.nodes.values() {
            if node.binding.configuration["effect_scope"]
                != serde_json::to_value(&self.effect_file)?
            {
                return Err(ai_agents_core::AgentError::Config(
                    "effect binding differs".into(),
                ));
            }
            for exchange in
                ai_agents_core::inspect_native_history(&node.runtime.snapshot.memory.messages)?
                    .exchanges()
            {
                if exchange.state().provider() != "fixture"
                    || exchange.state().api() != "native-tools"
                    || exchange.state().target().endpoint() != "https://restore.invalid"
                    || exchange.state().target().model() != "fixture-model-v1"
                {
                    return Err(ai_agents_core::AgentError::Config(
                        "native fixture destination differs".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    // All objects are newly built; empty registries are wired before framework-owned silent membership installation.
    async fn stage(
        &self,
        manifest: &TaskRestoreManifest,
    ) -> ai_agents_core::Result<TaskRestoreGraph> {
        self.stages.fetch_add(1, Ordering::SeqCst);
        self.stage_reached.add_permits(1);
        if self.stage_hold.load(Ordering::SeqCst) {
            self.stage_release.acquire().await.unwrap().forget();
        }
        let mut graph = TaskRestoreGraph::default();
        for (key, registry) in &manifest.registries {
            graph
                .registries
                .insert(key.clone(), self.registry(registry.send_with_context));
        }
        for (key, state) in &manifest.spawners {
            let mut spawner = AgentSpawner::new();
            if state["shared"] == true {
                spawner = if state["observed"] == true {
                    let mut builder = AgentBuilder::from_spec(self.root.clone())
                        .llm_registry(self.provider_registry());
                    builder.prepare_routing(false)?;
                    builder.wrap_llm_registry_for_observability()?;
                    builder.prepare_routing(true)?;
                    spawner.with_shared_observed_llms(
                        builder
                            .prepared_registry()
                            .ok_or_else(|| {
                                ai_agents_core::AgentError::Config(
                                    "fixture provider registry was not prepared".into(),
                                )
                            })?
                            .clone(),
                    )
                } else {
                    spawner.with_shared_llms(self.provider_registry())
                };
            }
            if let Some(maximum) = state["maximum"].as_u64() {
                spawner = spawner.with_max_agents(usize::try_from(maximum).unwrap());
            }
            if let Some(prefix) = state["prefix"].as_str() {
                spawner = spawner.with_name_prefix(prefix)?;
            }
            graph.spawners.insert(key.clone(), Arc::new(spawner));
        }
        for (id, node) in &manifest.nodes {
            let spec: AgentSpec =
                serde_json::from_value(node.binding.configuration["spec"].clone())?;
            let registry = node
                .registry
                .as_ref()
                .map(|key| graph.registries[key].clone());
            let spawner = node.spawner.as_ref().map(|key| graph.spawners[key].clone());
            let tool_registry = manifest
                .tools
                .iter()
                .find(|(_, (runtime, _))| runtime == id)
                .and_then(|(frame, _)| manifest.compositions.get(frame))
                .map(|binding| graph.registries[&binding.registry].clone());
            let runtime = self.runtime_with_handles(
                spec.clone(),
                registry,
                spawner,
                tool_registry,
                Some(&node.binding.state),
            );
            graph.runtimes.insert(id.clone(), runtime);
            if *id != manifest.root {
                graph.declarations.insert(id.clone(), spec);
            }
        }
        for key in manifest.providers.keys() {
            if let Some(spawner) = key
                .strip_prefix("spawner:")
                .and_then(|key| graph.spawners.get(key))
            {
                graph.providers.insert(
                    key.clone(),
                    Arc::new(spawner.reconstruction_providers().unwrap()),
                );
                continue;
            }
            let runtime = key
                .strip_prefix("runtime:")
                .and_then(|id| graph.runtimes.get(id))
                .unwrap_or(&graph.runtimes[&manifest.root]);
            graph
                .providers
                .insert(key.clone(), runtime.task_restore_providers());
        }
        *self.staged.lock() = graph
            .runtimes
            .iter()
            .map(|(id, runtime)| (id.clone(), Arc::downgrade(runtime)))
            .collect();
        Ok(graph)
    }
}

// Every test has independent provider/effect state and a concrete explicit task configuration.
pub(super) fn host() -> Arc<RestoreHost> {
    Arc::new(RestoreHost {
        root: AgentSpec::from_yaml_strict(
            "name: RestoreRoot\nsystem_prompt: evidence\ntools: [ask_user, restore_probe]\n",
        )
        .unwrap(),
        calls: Arc::new(AtomicUsize::new(0)),
        effects: Arc::new(AtomicUsize::new(0)),
        events: Default::default(),
        effect_file: None,
        routes: Default::default(),
        registrations: Arc::new(AtomicUsize::new(0)),
        stages: AtomicUsize::new(0),
        binding_version: AtomicUsize::new(1),
        staged: Default::default(),
        memory_gate: None,
        budget_millis: std::sync::atomic::AtomicU64::new(5000),
        stage_hold: std::sync::atomic::AtomicBool::new(false),
        stage_reached: tokio::sync::Semaphore::new(0),
        stage_release: tokio::sync::Semaphore::new(0),
        clock: Default::default(),
        clock_failure: std::sync::atomic::AtomicBool::new(false),
        contract: "restore-test.v1",
    })
}
// Exact per-run config remains disabled in YAML; restoration does not enable automatic entry or startup selection.
pub(super) fn config() -> AutonomyConfig {
    let mut config = super::runner_tests::config(12);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: Some(InteractionAction::PauseRun),
    });
    config
}
// A response is always bound to the newly advertised request and acknowledged revision.
async fn answer(runner: &AutonomyRunner, run: &TaskRun) -> ai_agents_core::Result<TaskRunResult> {
    runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::UserAnswer {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
}

// Explicit review is independent of tool answers and must not itself dispatch work.
async fn approve(runner: &AutonomyRunner, run: &TaskRun) -> TaskRunResult {
    let review = runner
        .restoration_review(&run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    runner
        .approve_restoration(
            &run.key.run_id,
            run.revision,
            &review.id,
            ai_agents_hitl::ApprovalResult::Approved,
        )
        .await
        .unwrap()
}

// Source composition uses real registry membership; stage reconstruction creates those entries silently from the manifest.
async fn declared_source(host: &RestoreHost, pattern: &str) -> Arc<RuntimeAgent> {
    let registry = host.registry(true);
    let nested = pattern.contains("middle");
    let workers = if nested {
        host.registry(true)
    } else {
        registry.clone()
    };
    for (id, name) in [("worker", "Worker"), ("peer", "Peer")] {
        let spec = AgentSpec::from_yaml_strict(&format!(
            "name: {name}\nsystem_prompt: restore-leaf\ntools: [ask_user, restore_probe]\n"
        ))
        .unwrap();
        let runtime = Arc::try_unwrap(host.runtime(spec.clone()))
            .unwrap_or_else(|_| panic!("source already shared"));
        workers
            .register(crate::spawner::SpawnedAgent::from_runtime(
                id.into(),
                runtime,
                spec,
            ))
            .await
            .unwrap();
    }
    if nested {
        let spec=AgentSpec::from_yaml_strict("name: Middle\nsystem_prompt: middle\nstates:\n  initial: work\n  states:\n    work:\n      concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all\n").unwrap();
        let runtime = Arc::try_unwrap(host.runtime(spec.clone()))
            .unwrap_or_else(|_| panic!("source already shared"))
            .with_spawner_handles(Arc::new(AgentSpawner::new()), workers);
        registry
            .register(crate::spawner::SpawnedAgent::from_runtime(
                "middle".into(),
                runtime,
                spec,
            ))
            .await
            .unwrap();
    }
    let _ = pattern;
    Arc::new(
        Arc::try_unwrap(host.runtime(host.root.clone()))
            .unwrap_or_else(|_| panic!("root already shared"))
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
    )
}

// Nested/repeated topology is reconstructed once per actual node, not once per historical invocation snapshot.
#[tokio::test]
async fn restore_paused_declared_nested_and_repeated_operations_preserve_topology() {
    for pattern in [
        "delegate: worker",
        "concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all",
        "pipeline:\n        stages: [worker, worker]",
        "delegate: middle",
    ] {
        for &sqlite in super::continuation_test_support::backends() {
            let mut host = host();
            Arc::get_mut(&mut host).unwrap().root=AgentSpec::from_yaml_strict(&format!("name: RestoreRoot\nsystem_prompt: root\nstates:\n  initial: work\n  states:\n    work:\n      {pattern}\n")).unwrap();
            let root = Box::pin(declared_source(&host, pattern)).await;
            let old = Arc::downgrade(&root);
            let store =
                super::continuation_test_support::store(root.info().id, "restore.v1", sqlite).await;
            let runner = AutonomyRunner::try_new(
                root.clone(),
                config(),
                Default::default(),
                store.clone(),
                "restore.v1".into(),
            )
            .unwrap();
            let mut source = runner.run("verify evidence", None).await.unwrap();
            assert_eq!(source.run.status, TaskRunStatus::Paused, "{pattern}");
            if pattern.starts_with("pipeline") {
                source = answer(&runner, &source.run).await.unwrap();
                assert_eq!(
                    source.run.status,
                    TaskRunStatus::Paused,
                    "second invocation must be a distinct question"
                );
            }
            let effects = host.effects.load(Ordering::SeqCst);
            let before = host.calls.load(Ordering::SeqCst);
            let prepared = runner
                .checkpoint_for_restore(&source.run.key.run_id, source.run.revision, host.as_ref())
                .await
                .unwrap();
            let historical = store
                .load(&prepared.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload["children"]
                .clone();
            drop(runner);
            drop(root);
            assert!(old.upgrade().is_none());
            let request = TaskRestoreRequest {
                run_id: prepared.run.key.run_id.clone(),
                expected_revision: prepared.run.revision,
                config_identity: "restore.v1".into(),
                config: config(),
                ceilings: Default::default(),
            };
            let (restored, result) =
                AutonomyRunner::restore_paused(request, store.clone(), host.clone())
                    .await
                    .unwrap();
            assert_eq!(
                host.calls.load(Ordering::SeqCst),
                before,
                "restore dispatched a provider"
            );
            assert_eq!(host.effects.load(Ordering::SeqCst), effects);
            let mut result = approve(&restored, &result.run).await;
            let mut count = 0;
            while result.run.status == TaskRunStatus::Paused {
                count += 1;
                assert!(count <= 2);
                result = answer(&restored, &result.run).await.unwrap();
            }
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{pattern}: {:?}",
                result.run.stop_reason
            );
            let expected = if pattern == "delegate: worker" { 1 } else { 2 };
            assert_eq!(
                host.effects.load(Ordering::SeqCst),
                expected,
                "{pattern}: completed effects replayed or unstarted work skipped"
            );
            {
                let events = host.events.lock();
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| event.as_str() == "Worker")
                        .count(),
                    if pattern.starts_with("pipeline") {
                        2
                    } else {
                        1
                    }
                );
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| event.as_str() == "Peer")
                        .count(),
                    usize::from(pattern.contains("concurrent") || pattern.contains("middle"))
                );
            }
            let saved = store.load(&result.run.key.run_id).await.unwrap().unwrap();
            for child in historical
                .as_array()
                .unwrap()
                .iter()
                .filter(|child| !child["result"].is_null())
            {
                assert_eq!(
                    saved.payload["children"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|next| next["child_id"] == child["child_id"])
                        .unwrap(),
                    child
                );
            }
        }
    }
}

// Existing model tools use restored exact contexts and captured handles, including an external registry not attached to the root runtime.
#[tokio::test]
async fn restore_paused_model_composition_and_round_robin_state() {
    for (tag, tool) in [
        ("pipeline", "pipeline_process"),
        ("concurrent", "concurrent_ask"),
        ("route", "route_to_agent"),
        ("send", "send_agent_message"),
    ] {
        for &sqlite in super::continuation_test_support::backends() {
            let mut host = host();
            Arc::get_mut(&mut host).unwrap().root = AgentSpec::from_yaml_strict(&format!(
                "name: RestoreRoot\nsystem_prompt: restore-model-{tag}\ntools: [{tool}]\n"
            ))
            .unwrap();
            let registry = host.registry(false);
            for (id, name) in [("worker", "Worker"), ("peer", "Peer")] {
                let spec = AgentSpec::from_yaml_strict(&format!(
                    "name: {name}\nsystem_prompt: restore-leaf\ntools: [ask_user, restore_probe]\n"
                ))
                .unwrap();
                let runtime = Arc::try_unwrap(host.runtime(spec.clone()))
                    .unwrap_or_else(|_| panic!("source unexpectedly shared"));
                registry
                    .register(crate::spawner::SpawnedAgent::from_runtime(
                        id.into(),
                        runtime,
                        spec,
                    ))
                    .await
                    .unwrap();
            }
            let attached = (tag != "send").then(|| registry.clone());
            let root = host.runtime_with_handles(
                host.root.clone(),
                attached,
                None,
                Some(registry.clone()),
                None,
            );
            let old = Arc::downgrade(&root);
            let store =
                super::continuation_test_support::store(root.info().id, "restore.v1", sqlite).await;
            let runner = AutonomyRunner::try_new(
                root.clone(),
                config(),
                Default::default(),
                store.clone(),
                "restore.v1".into(),
            )
            .unwrap();
            let paused = runner.run("verify model composition", None).await.unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{tag}: {:?}",
                paused.run.stop_reason
            );
            let prepared = runner
                .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
                .await
                .unwrap();
            let before = host.calls.load(Ordering::SeqCst);
            let effects = host.effects.load(Ordering::SeqCst);
            let registrations = host.registrations.load(Ordering::SeqCst);
            drop(runner);
            drop(root);
            drop(registry);
            assert!(old.upgrade().is_none());
            let request = TaskRestoreRequest {
                run_id: prepared.run.key.run_id.clone(),
                expected_revision: prepared.run.revision,
                config_identity: "restore.v1".into(),
                config: config(),
                ceilings: Default::default(),
            };
            let (restored, paused) =
                AutonomyRunner::restore_paused(request, store.clone(), host.clone())
                    .await
                    .unwrap();
            assert_eq!(
                host.calls.load(Ordering::SeqCst),
                before,
                "{tag}: restore dispatched"
            );
            assert_eq!(host.effects.load(Ordering::SeqCst), effects);
            assert_eq!(
                host.registrations.load(Ordering::SeqCst),
                registrations,
                "registration hooks replayed"
            );
            let mut result = approve(&restored, &paused.run).await;
            let mut count = 0;
            while result.run.status == TaskRunStatus::Paused {
                count += 1;
                assert!(count <= 2);
                result = answer(&restored, &result.run).await.unwrap();
            }
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{tag}: {:?}",
                result.run.stop_reason
            );
            assert_eq!(
                host.effects.load(Ordering::SeqCst),
                if tag == "send" { 1 } else { 2 },
                "{tag}"
            );
            assert_eq!(host.registrations.load(Ordering::SeqCst), registrations);
            if tag == "route" {
                assert_eq!(
                    count, 2,
                    "second selection must advance to the peer rather than reset to worker"
                );
                let events = host.events.lock();
                assert_eq!(events.as_slice(), ["Worker", "Peer"]);
            }
        }
    }
}

// Actual generation is complete before the checkpoint; reconstruction restores tracked capacity and future naming without invoking generation or hooks.
#[tokio::test]
async fn restore_paused_generated_child_preserves_capacity_naming_and_generation_charge() {
    for &sqlite in super::continuation_test_support::backends() {
        let mut host = host();
        Arc::get_mut(&mut host).unwrap().root=AgentSpec::from_yaml_strict("name: RestoreRoot\nsystem_prompt: restore-generation-root\ntools: [spawn_agent, pipeline_process]\n").unwrap();
        let registry = host.registry(true);
        let root = host.runtime_with_handles(
            host.root.clone(),
            Some(registry.clone()),
            None,
            Some(registry.clone()),
            None,
        );
        let old = Arc::downgrade(&root);
        let store =
            super::continuation_test_support::store(root.info().id, "restore.v1", sqlite).await;
        let runner = AutonomyRunner::try_new(
            root.clone(),
            config(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let paused = runner.run("generate evidence", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert!(registry.contains("generated_001"));
        assert_eq!(root.spawner().unwrap().spawned_count(), 1);
        let before = host.calls.load(Ordering::SeqCst);
        assert_eq!(host.registrations.load(Ordering::SeqCst), 1);
        let prepared = runner
            .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
            .await
            .unwrap();
        drop(runner);
        drop(root);
        drop(registry);
        assert!(old.upgrade().is_none());
        let request = TaskRestoreRequest {
            run_id: prepared.run.key.run_id.clone(),
            expected_revision: prepared.run.revision,
            config_identity: "restore.v1".into(),
            config: config(),
            ceilings: Default::default(),
        };
        let (restored, paused) =
            AutonomyRunner::restore_paused(request, store.clone(), host.clone())
                .await
                .unwrap();
        assert_eq!(host.calls.load(Ordering::SeqCst), before);
        assert_eq!(host.registrations.load(Ordering::SeqCst), 1);
        let reviewed = approve(&restored, &paused.run).await;
        let done = answer(&restored, &reviewed.run).await.unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(host.registrations.load(Ordering::SeqCst), 1);
        let current = restored.runtime();
        let spawner = current.spawner().unwrap();
        assert_eq!(spawner.spawned_count(), 1);
        let next = spawner
            .spawn_from_yaml(
                "name: Next\nsystem_prompt: restore-generated-leaf\ntools: [ask_user]\n",
            )
            .await
            .unwrap();
        assert_eq!(next.id, "generated_002");
        assert_eq!(spawner.spawned_count(), 2);
        drop(next);
        assert_eq!(spawner.spawned_count(), 1);
        assert!(
            current
                .spawner_registry()
                .unwrap()
                .remove("generated_001")
                .await
                .is_some()
        );
        assert_eq!(spawner.spawned_count(), 0);
    }
}

// Negative cases start from a real acknowledged image, then destroy all original runtime/owner objects.
async fn prepared_fixture(
    sqlite: bool,
) -> (Arc<RestoreHost>, Arc<ScopedTaskRunStore>, TaskRunResult) {
    let host = host();
    let root = host.runtime(host.root.clone());
    let store = super::continuation_test_support::store(root.info().id, "restore.v1", sqlite).await;
    let runner = AutonomyRunner::try_new(
        root,
        config(),
        Default::default(),
        store.clone(),
        "restore.v1".into(),
    )
    .unwrap();
    let paused = runner.run("verify evidence", None).await.unwrap();
    let prepared = runner
        .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
        .await
        .unwrap();
    drop(runner);
    (host, store, prepared)
}

// Current host ceilings and trustworthy recovery time are checked before claim, not inferred from unchanged raw profile settings.
#[tokio::test]
async fn restore_rejects_lower_host_ceilings_and_untrusted_clock_without_claim() {
    for &sqlite in super::continuation_test_support::backends() {
        let (host, store, prepared) = prepared_fixture(sqlite).await;
        let before = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        for ceilings in [
            AutonomyHostCeilings {
                max_llm_calls: Some(1),
                ..Default::default()
            },
            AutonomyHostCeilings {
                max_tool_calls: Some(1),
                ..Default::default()
            },
            AutonomyHostCeilings {
                max_active_time_seconds: Some(1),
                ..Default::default()
            },
        ] {
            let request = TaskRestoreRequest {
                run_id: prepared.run.key.run_id.clone(),
                expected_revision: prepared.run.revision,
                config_identity: "restore.v1".into(),
                config: config(),
                ceilings,
            };
            assert!(
                AutonomyRunner::restore_paused(request, store.clone(), host.clone())
                    .await
                    .is_err()
            );
            assert_eq!(
                serde_json::to_value(store.load(&prepared.run.key.run_id).await.unwrap().unwrap())
                    .unwrap(),
                serde_json::to_value(&before).unwrap()
            );
        }
        for mode in ["backward", "uncertified"] {
            *host.clock.lock() = if mode == "backward" {
                Some(before.updated_at - chrono::Duration::seconds(1))
            } else {
                None
            };
            host.clock_failure
                .store(mode == "uncertified", Ordering::SeqCst);
            let request = TaskRestoreRequest {
                run_id: prepared.run.key.run_id.clone(),
                expected_revision: prepared.run.revision,
                config_identity: "restore.v1".into(),
                config: config(),
                ceilings: Default::default(),
            };
            assert!(
                AutonomyRunner::restore_paused(request, store.clone(), host.clone())
                    .await
                    .is_err()
            );
        }
        assert_eq!(host.stages.load(Ordering::SeqCst), 0);
        assert_eq!(host.calls.load(Ordering::SeqCst), 1);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        assert_eq!(
            store
                .load(&prepared.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            before.revision
        );
    }
}

// Staging time and publication share one fixed deadline, while a failed claimed attempt retains its prepaid time and recovery marker.
#[tokio::test]
async fn restore_stage_overrun_cannot_refill_active_allowance_or_publish_pause() {
    let (host, store, prepared) = prepared_fixture(false).await;
    host.budget_millis.store(1000, Ordering::SeqCst);
    host.stage_hold.store(true, Ordering::SeqCst);
    let before = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
    let request = TaskRestoreRequest {
        run_id: prepared.run.key.run_id.clone(),
        expected_revision: prepared.run.revision,
        config_identity: "restore.v1".into(),
        config: config(),
        ceilings: Default::default(),
    };
    let start_host = host.clone();
    let start_store = store.clone();
    let mut work = tokio::spawn(async move {
        AutonomyRunner::restore_paused(request, start_store, start_host).await
    });
    tokio::select! {
        result=&mut work=>panic!("returned before actual stage boundary: {:?}",result.err()),
        permit=host.stage_reached.acquire()=>permit.unwrap().forget(),
        _=tokio::time::sleep(std::time::Duration::from_secs(10))=>panic!("stage boundary was not reached"),
    }
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), work)
        .await
        .unwrap()
        .unwrap();
    assert!(result.is_err());
    let saved = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(saved.status, TaskRunStatus::Running);
    assert!(saved.owner_token.is_some());
    assert!(
        saved.payload["clocks"]["active_millis"].as_u64().unwrap()
            >= before.payload["clocks"]["active_millis"].as_u64().unwrap() + 1000
    );
    assert!(
        saved.payload["adapters"]
            .as_array()
            .unwrap()
            .iter()
            .any(|adapter| adapter["id"] == "runtime.restore_work"
                && adapter["state"]["phase"] == "running")
    );
    assert_eq!(host.calls.load(Ordering::SeqCst), 1);
    assert_eq!(host.effects.load(Ordering::SeqCst), 1);
}

// Missing framework-owned references are typed preflight errors, never panicking indexing after component mutation.
#[tokio::test]
async fn restore_corrupt_manifest_references_reject_before_claim() {
    for kind in [
        "tool",
        "finishing",
        "context",
        "active_frame",
        "snapshot",
        "provider",
    ] {
        let (host, store, prepared) = prepared_fixture(false).await;
        let claimed = store
            .claim_paused(
                &prepared.run.key.run_id,
                prepared.run.revision,
                "fixture-corruption".into(),
            )
            .await
            .unwrap()
            .unwrap();
        let mut payload: TaskCheckpointPayload =
            serde_json::from_value(claimed.payload.clone()).unwrap();
        let adapter = payload
            .adapters
            .iter_mut()
            .find(|adapter| adapter.id == "runtime.restore")
            .unwrap();
        let manifest = &mut adapter.state;
        manifest["revision"] = json!(claimed.revision + 1);
        match kind {
            "tool" => manifest["tools"]["missing-frame"] = json!(["missing-runtime", "echo"]),
            "finishing" => manifest["skill_finishing"] = json!(["missing-runtime"]),
            "context" => manifest["message_contexts"]["missing-frame"] = json!({}),
            "active_frame" => manifest["active_frames"] = json!(["missing-frame"]),
            "provider" => {
                let root = manifest["root"].as_str().unwrap().to_string();
                manifest["providers"]
                    .as_object_mut()
                    .unwrap()
                    .remove(&format!("runtime:{root}"));
            }
            _ => {
                manifest["nodes"]["RestoreRoot"]["runtime"]["snapshot"]["context"]["corrupt"] =
                    json!(true)
            }
        }
        let mutation = TaskRunMutation::FinalCheckpoint {
            expected_revision: claimed.revision,
            owner_token: claimed.owner_token.unwrap(),
            status: TaskRunStatus::Paused,
            payload: serde_json::to_value(payload).unwrap(),
            release: true,
            deadline: std::time::Instant::now() + std::time::Duration::from_secs(30),
            expires_at: None,
            evidence: None,
        };
        let saved = store
            .mutate(&prepared.run.key.run_id, &mutation)
            .await
            .unwrap();
        let request = TaskRestoreRequest {
            run_id: prepared.run.key.run_id.clone(),
            expected_revision: saved.revision,
            config_identity: "restore.v1".into(),
            config: config(),
            ceilings: Default::default(),
        };
        assert!(
            AutonomyRunner::restore_paused(request, store.clone(), host.clone())
                .await
                .is_err(),
            "{kind}"
        );
        assert_eq!(host.stages.load(Ordering::SeqCst), 0, "{kind}");
        let after = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(after.revision, saved.revision);
        assert_eq!(after.status, TaskRunStatus::Paused);
    }
}

// A predictable image-size failure cannot abandon the existing safe continuation or consume a new revision.
#[tokio::test]
async fn restore_oversized_preparation_preserves_usable_live_pause() {
    let host = host();
    let root = host.runtime(host.root.clone());
    root.set_context("large_fixture", json!("x".repeat(2_095_000)))
        .unwrap();
    let store = super::continuation_test_support::store(root.info().id, "restore.v1", false).await;
    let runner = AutonomyRunner::try_new(
        root,
        config(),
        Default::default(),
        store.clone(),
        "restore.v1".into(),
    )
    .unwrap();
    let paused = runner.run("verify bounded image", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let before = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert!(
        runner
            .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
            .await
            .is_err()
    );
    let after = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.status, TaskRunStatus::Paused);
    assert_eq!(
        runner
            .cancel_paused(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap()
            .run
            .status,
        TaskRunStatus::Cancelled
    );
}

// Separate processes share only an isolated SQLite file and an external effect witness, never runtime/host/provider objects.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn restore_process_parent_reopens_sqlite_without_retained_objects() {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/restoration-tests")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&directory).unwrap();
    for phase in ["producer", "consumer"] {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "autonomy::restoration_tests::restore_process_worker",
                "--nocapture",
            ])
            .env("AI_AGENTS_RESTORE_PHASE", phase)
            .env("AI_AGENTS_RESTORE_DIRECTORY", &directory)
            .kill_on_drop(true);
        let output = tokio::time::timeout(std::time::Duration::from_secs(120), command.output())
            .await
            .unwrap()
            .unwrap();
        assert!(
            output.status.success(),
            "{phase}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(
        std::fs::read_to_string(directory.join("effects.log"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    crate::remove_sqlite_test_directory(&directory)
        .await
        .unwrap();
}

// This exact libtest entry is invoked by the parent in producer and consumer processes; normal suite discovery performs no work.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn restore_process_worker() {
    let Ok(phase) = std::env::var("AI_AGENTS_RESTORE_PHASE") else {
        return;
    };
    let directory =
        std::path::PathBuf::from(std::env::var_os("AI_AGENTS_RESTORE_DIRECTORY").unwrap());
    let mut host = host();
    Arc::get_mut(&mut host).unwrap().effect_file = Some(directory.join("effects.log"));
    let storage = Arc::new(
        ai_agents_storage::SqliteStorage::new(directory.join("tasks.sqlite").to_str().unwrap())
            .await
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::new(
            storage.clone(),
            host.root.name.clone(),
            None,
            "restore.v1".into(),
        )
        .unwrap(),
    );
    if phase == "producer" {
        let root = host.runtime(host.root.clone());
        let runner = AutonomyRunner::try_new(
            root,
            config(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let paused = runner.run("verify durable effect", None).await.unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        let prepared = runner
            .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
            .await
            .unwrap();
        std::fs::write(
            directory.join("request.json"),
            serde_json::to_vec(
                &json!({"id":prepared.run.key.run_id,"revision":prepared.run.revision}),
            )
            .unwrap(),
        )
        .unwrap();
        drop(runner);
    } else {
        assert_eq!(phase, "consumer");
        let saved: Value =
            serde_json::from_slice(&std::fs::read(directory.join("request.json")).unwrap())
                .unwrap();
        let request = TaskRestoreRequest {
            run_id: saved["id"].as_str().unwrap().into(),
            expected_revision: saved["revision"].as_u64().unwrap(),
            config_identity: "restore.v1".into(),
            config: config(),
            ceilings: Default::default(),
        };
        let (runner, paused) = AutonomyRunner::restore_paused(request, store.clone(), host.clone())
            .await
            .unwrap();
        assert_eq!(host.calls.load(Ordering::SeqCst), 0);
        assert_eq!(host.effects.load(Ordering::SeqCst), 0);
        let reviewed = approve(&runner, &paused.run).await;
        assert_eq!(host.calls.load(Ordering::SeqCst), 0);
        let done = answer(&runner, &reviewed.run).await.unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(done.run.counters.llm_attempts, 2);
        assert_eq!(host.calls.load(Ordering::SeqCst), 1);
        assert_eq!(host.effects.load(Ordering::SeqCst), 0);
        assert_eq!(
            std::fs::read_to_string(directory.join("effects.log"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        drop(runner);
    }
    drop(store);
    storage.close().await;
}

// Membership and actor identity must survive JSON independently; null actor data does not mean an unstarted node.
#[test]
fn restore_participant_actorless_marker_round_trips() {
    let member = Some(TaskRestoreParticipant { actor_id: None });
    let encoded = serde_json::to_value(&member).unwrap();
    assert!(encoded.is_object());
    let decoded: Option<TaskRestoreParticipant> = serde_json::from_value(encoded).unwrap();
    assert!(decoded.is_some_and(|member| member.actor_id.is_none()));
}

// Durable state plus current host configuration, not a retained Arc, revives exact pending work without repeating the completed effect.
#[tokio::test]
async fn restore_paused_new_runtime_requires_fresh_review_and_preserves_completed_effects() {
    for &sqlite in super::continuation_test_support::backends() {
        let host = host();
        let root = host.runtime(host.root.clone());
        let old = Arc::downgrade(&root);
        let store =
            super::continuation_test_support::store(root.info().id, "restore.v1", sqlite).await;
        let runner = AutonomyRunner::try_new(
            root.clone(),
            config(),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        let first = runner.run("verify evidence", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        let old_request = first.run.pending.as_ref().unwrap().id.clone();
        let prepared = runner
            .checkpoint_for_restore(&first.run.key.run_id, first.run.revision, host.as_ref())
            .await
            .unwrap();
        drop(runner);
        drop(root);
        assert!(
            old.upgrade().is_none(),
            "reconstruction must not retain the old root"
        );
        let before = host.calls.load(Ordering::SeqCst);
        let request = TaskRestoreRequest {
            run_id: prepared.run.key.run_id.clone(),
            expected_revision: prepared.run.revision,
            config_identity: "restore.v1".into(),
            config: config(),
            ceilings: Default::default(),
        };
        let (restored, result) =
            AutonomyRunner::restore_paused(request, store.clone(), host.clone())
                .await
                .unwrap();
        assert_eq!(host.calls.load(Ordering::SeqCst), before);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        assert_eq!(result.run.status, TaskRunStatus::Paused);
        assert_ne!(result.run.pending.as_ref().unwrap().id, old_request);
        assert!(
            answer(&restored, &result.run).await.is_err(),
            "restoration must not reuse serialized approval authority"
        );
        let review = restored
            .restoration_review(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap();
        let reviewed = restored
            .approve_restoration(
                &result.run.key.run_id,
                result.run.revision,
                &review.id,
                ai_agents_hitl::ApprovalResult::Approved,
            )
            .await
            .unwrap();
        assert_eq!(host.calls.load(Ordering::SeqCst), before);
        assert!(
            restored
                .resume(
                    &reviewed.run.key.run_id,
                    reviewed.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: old_request,
                        answer: json!({"answered":true,"selected":["yes"]})
                    }
                )
                .await
                .is_err()
        );
        let done = answer(&restored, &reviewed.run).await.unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(done.run.counters.llm_attempts, 2);
        assert_eq!(host.calls.load(Ordering::SeqCst), before + 1);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
    }
}
