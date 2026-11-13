//! Generated participants use the same managed admission and exact composition cursors as declared agents.

use super::co_completion_tests::{Script, answer, question, register, runner, runner_with_store};
use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner};
use crate::{AgentBuilder, RuntimeAgent};
use async_trait::async_trait;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

// Production native calls pass through tool grants, managed accounting and executor settlement.
fn call(id: &str, tool: &str, arguments: serde_json::Value) -> String {
    ai_agents_core::encode_native_tool_call_markers(
        &[ai_agents_core::ToolCall {
            id: id.into(),
            name: tool.into(),
            arguments,
        }],
        None,
    )
    .unwrap()
}

struct SpawnHook {
    calls: AtomicUsize,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
    block: bool,
}
impl SpawnHook {
    // A hook acknowledges registration has committed; blocking it must not be confused with an unregistered child.
    fn new(block: bool) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
            block,
        })
    }
}
#[async_trait]
impl crate::spawner::RegistryHooks for SpawnHook {
    // The exact committed registry hook is the interruption boundary, not a timer or an armed flag.
    async fn on_agent_spawned(&self, _: &str, _: &crate::spec::AgentSpec) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        if self.block {
            self.release.acquire().await.unwrap().forget();
        }
    }
}

// Build against the same prepared provider handles used by the root; no duplicate wrapper or new allowance is installed.
fn generated_parent(
    provider: Arc<Script>,
    registry: Arc<AgentRegistry>,
    capacity: usize,
    composition: Option<(&str, Arc<Script>)>,
) -> (Arc<RuntimeAgent>, Arc<AgentSpawner>) {
    let tools = composition
        .as_ref()
        .map_or("spawn_agent".to_string(), |(tool, _)| {
            format!("spawn_agent, {tool}")
        });
    let yaml = format!("name: GeneratedReport\nsystem_prompt: report\ntools: [{tools}]\n");
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register("default", provider);
    let mut builder = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm_registry(llms)
        .auto_configure_features()
        .unwrap();
    builder.prepare_routing(false).unwrap();
    builder.wrap_llm_registry_for_observability().unwrap();
    builder.prepare_routing(true).unwrap();
    let llms = builder.prepared_registry().unwrap().clone();
    let spawner = Arc::new(
        AgentSpawner::new()
            .with_shared_llms(llms.clone())
            .with_max_agents(capacity),
    );
    builder = builder.tool(Arc::new(crate::spawner::GenerateAgentTool::new(
        spawner.clone(),
        registry.clone(),
        Arc::new(llms.clone()),
    )));
    if let Some((tool, auxiliary)) = composition {
        let mut captured = llms;
        captured.register("router", auxiliary);
        let captured = Arc::new(captured);
        builder = match tool {
            "concurrent_ask" => builder.tool(Arc::new(
                crate::orchestration::tools::ConcurrentAskTool::new(registry.clone(), captured),
            )),
            "pipeline_process" => builder.tool(Arc::new(
                crate::orchestration::tools::PipelineProcessTool::new(registry.clone()),
            )),
            "route_to_agent" => builder.tool(Arc::new(
                crate::orchestration::tools::RouteToAgentTool::new(registry.clone(), captured),
            )),
            _ => panic!("unsupported fixture tool"),
        };
    }
    (
        Arc::new(
            builder
                .build()
                .unwrap()
                .with_spawner_handles(spawner.clone(), registry),
        ),
        spawner,
    )
}

// Failed generation, repair, construction capacity and registration cannot refund already consumed provider attempts.
#[tokio::test]
async fn co_completion_generated_failures_preserve_charges_capacity_and_existing_instance() {
    for kind in ["provider", "repair", "capacity", "duplicate", "repaired"] {
        let spawn = call(
            "create",
            "spawn_agent",
            json!({"name":"Worker","description":"domain worker"}),
        );
        let definition = "name: Worker\nsystem_prompt: report worker\n";
        let responses = match kind {
            "provider" => vec![
                Some(spawn.as_str()),
                None,
                Some("done: generation failure recorded"),
            ],
            "repair" => vec![
                Some(spawn.as_str()),
                Some("not valid yaml"),
                Some("still invalid yaml"),
                Some("done: repair failure recorded"),
            ],
            // The existing generator offers one repair after construction failure, including exhausted managed capacity.
            "capacity" => vec![
                Some(spawn.as_str()),
                Some(definition),
                Some(definition),
                Some("done: capacity failure recorded"),
            ],
            "repaired" => vec![
                Some(spawn.as_str()),
                Some("not valid yaml"),
                Some(definition),
                Some("done: registered after repair"),
            ],
            _ => vec![
                Some(spawn.as_str()),
                Some(definition),
                Some("done: creation outcome recorded"),
            ],
        };
        let provider = Script::new(&responses, 0);
        let hook = SpawnHook::new(false);
        let registry = Arc::new(AgentRegistry::new().with_hooks(hook.clone()));
        let existing = if kind == "duplicate" {
            Some(
                register(
                    &registry,
                    "worker",
                    AgentBuilder::from_yaml(definition)
                        .unwrap()
                        .llm(Script::new(&[], 0))
                        .build()
                        .unwrap(),
                    definition,
                )
                .await,
            )
        } else {
            None
        };
        let (root, spawner) = generated_parent(
            provider.clone(),
            registry.clone(),
            if kind == "capacity" { 0 } else { 1 },
            None,
        );
        let (runner, _) = runner(root.clone());
        let done = runner.run("create a report worker", None).await.unwrap();
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            responses.len(),
            "{kind}: {:?}",
            done.run.stop_reason
        );
        assert_eq!(
            done.run.counters.llm_attempts as usize,
            responses.len(),
            "{kind}"
        );
        let records: Vec<_> = done
            .run
            .evidence
            .tool_calls
            .iter()
            .filter(|call| call.record.canonical_id == "spawn_agent")
            .collect();
        assert_eq!(records.len(), 1, "{kind}");
        assert!(records[0].record.executed, "{kind}");
        assert_eq!(
            records[0].record.success,
            kind == "repaired",
            "{kind}: {}",
            records[0].record.output
        );
        assert_eq!(
            spawner.spawned_count(),
            u32::from(kind == "repaired"),
            "{kind}"
        );
        assert_eq!(
            hook.calls.load(Ordering::SeqCst),
            usize::from(kind == "repaired" || kind == "duplicate")
        );
        if let Some(existing) = existing {
            assert!(Arc::ptr_eq(&existing, &registry.get("worker").unwrap()));
        }
        if registry.contains("worker") {
            assert!(registry.remove("worker").await.is_some());
        }
        assert_eq!(spawner.spawned_count(), 0);
        if kind != "capacity" {
            let available = spawner
                .spawn_from_yaml("name: Available\nsystem_prompt: available\n")
                .await
                .unwrap();
            assert_eq!(spawner.spawned_count(), 1);
            drop(available);
            assert_eq!(spawner.spawned_count(), 0);
        }
        assert!(root.set_context("released", json!(true)).is_ok());
    }
}

// Drop during generation or after registration commit preserves uncertain effects and refuses an automatic fresh run.
#[tokio::test]
async fn co_completion_generated_interruption_keeps_effects_and_blocks_replay() {
    for after_registration in [false, true] {
        let spawn = call(
            "create",
            "spawn_agent",
            json!({"name":"Worker","description":"domain worker"}),
        );
        let provider = Script::new(
            &[
                Some(&spawn),
                Some("name: Worker\nsystem_prompt: worker\n"),
                Some("done"),
            ],
            if after_registration { 0 } else { 2 },
        );
        let hook = SpawnHook::new(after_registration);
        let registry = Arc::new(AgentRegistry::new().with_hooks(hook.clone()));
        let (root, spawner) = generated_parent(provider.clone(), registry.clone(), 1, None);
        let (runner, store) = runner(root.clone());
        let mut work = Box::pin(runner.run("generate", None));
        let reached = if after_registration {
            &hook.entered
        } else {
            &provider.entered
        };
        tokio::select! {
            result = &mut work => panic!("returned before selected boundary: {:?}", result.err()),
            permit = reached.acquire() => permit.unwrap().forget(),
        }
        let committed = registry.get("worker");
        assert_eq!(committed.is_some(), after_registration);
        assert_eq!(spawner.spawned_count(), u32::from(after_registration));
        drop(work);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert!(
            root.set_context("must_stay_protected", json!(true))
                .is_err()
        );
        assert!(runner.run("must not replay", None).await.is_err());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        let runs = store.list().await.unwrap();
        assert_eq!(runs.len(), 1);
        let saved = store.load(&runs[0].key.run_id).await.unwrap().unwrap();
        assert!(saved.owner_token.is_some());
        assert_eq!(saved.payload["counters"]["llm_attempts"], 2);
        if let Some(committed) = committed {
            assert!(Arc::ptr_eq(&committed, &registry.get("worker").unwrap()));
            assert_eq!(hook.calls.load(Ordering::SeqCst), 1);
            assert_eq!(spawner.spawned_count(), 1);
        } else {
            assert_eq!(registry.count(), 0);
            assert_eq!(hook.calls.load(Ordering::SeqCst), 0);
        }
    }
}

// Generated instances participate in existing pipeline/concurrent/route algorithms without replaying registration on child questions.
#[tokio::test]
async fn co_completion_generated_composition_reuses_actual_child_and_shared_allowance() {
    for tool in ["pipeline_process", "concurrent_ask", "route_to_agent"] {
        for cancel in [false, true] {
            let spawn = call(
                "create",
                "spawn_agent",
                json!({"name":"Worker","description":"report worker"}),
            );
            let args = match tool {
                "pipeline_process" => json!({"input":"report","stages":["worker","peer"]}),
                "concurrent_ask" => {
                    json!({"question":"report","agents":["worker","peer"],"aggregation":"llm_synthesis"})
                }
                _ => json!({"input":"report","candidates":["peer","worker"],"method":"llm"}),
            };
            let compose = call("compose", tool, args);
            let q = question();
            let provider = Script::new(
                &[
                    Some(&spawn),
                    Some("name: Worker\nsystem_prompt: report evidence\ntools: [ask_user]\n"),
                    Some(&compose),
                    Some(&q),
                    Some("verified worker report"),
                    Some("done: final report"),
                ],
                0,
            );
            let hook = SpawnHook::new(false);
            let registry = Arc::new(AgentRegistry::new().with_hooks(hook.clone()));
            let peer = Script::new(&[Some(&q), Some("verified peer report")], 0);
            let peer_yaml = "name: Peer\nsystem_prompt: peer\ntools: [ask_user]\n";
            register(
                &registry,
                "peer",
                AgentBuilder::from_yaml(peer_yaml)
                    .unwrap()
                    .llm(peer.clone())
                    .auto_configure_features()
                    .unwrap()
                    .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                        Arc::new(parking_lot::RwLock::new(None)),
                    )))
                    .build()
                    .unwrap(),
                peer_yaml,
            )
            .await;
            let auxiliary = Script::new(
                &[Some(if tool == "route_to_agent" {
                    "worker"
                } else {
                    "synthesized report"
                })],
                0,
            );
            let (root, spawner) = generated_parent(
                provider.clone(),
                registry.clone(),
                1,
                Some((tool, auxiliary.clone())),
            );
            let (runner, store) = runner(root.clone());
            let mut result = runner.run("produce a verified report", None).await.unwrap();
            assert_eq!(
                result.run.status,
                TaskRunStatus::Paused,
                "{tool}: {:?}",
                result.run.stop_reason
            );
            let child = registry.get("worker").unwrap();
            assert!(registry.remove("worker").await.is_none());
            let mut pauses = 0;
            while result.run.status == TaskRunStatus::Paused {
                pauses += 1;
                assert!(pauses <= 2, "{tool}");
                assert!(Arc::ptr_eq(&child, &registry.get("worker").unwrap()));
                assert_eq!(spawner.spawned_count(), 1);
                assert_eq!(hook.calls.load(Ordering::SeqCst), 2);
                result = if cancel {
                    runner
                        .cancel_paused(&result.run.key.run_id, result.run.revision)
                        .await
                        .unwrap()
                } else {
                    answer(&runner, &result.run).await
                };
            }
            assert_eq!(
                result.run.status,
                if cancel {
                    TaskRunStatus::Cancelled
                } else {
                    TaskRunStatus::Completed
                },
                "{tool}: {:?}",
                result.run.stop_reason
            );
            if !cancel {
                assert_eq!(pauses, if tool == "route_to_agent" { 1 } else { 2 });
                assert_eq!(provider.calls.load(Ordering::SeqCst), 6);
                assert_eq!(
                    peer.calls.load(Ordering::SeqCst),
                    if tool == "route_to_agent" { 0 } else { 2 }
                );
                assert_eq!(
                    auxiliary.calls.load(Ordering::SeqCst),
                    usize::from(tool != "pipeline_process")
                );
                assert_eq!(
                    result.run.counters.llm_attempts as usize,
                    provider.calls.load(Ordering::SeqCst)
                        + peer.calls.load(Ordering::SeqCst)
                        + auxiliary.calls.load(Ordering::SeqCst)
                );
            }
            let saved = store.load(&result.run.key.run_id).await.unwrap().unwrap();
            let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
            assert_eq!(
                payload
                    .evidence
                    .tool_calls
                    .iter()
                    .filter(|record| record.record.canonical_id == "spawn_agent")
                    .count(),
                1
            );
            assert!(
                payload
                    .children
                    .iter()
                    .all(|child| child.result.is_some() && child.pending.is_none())
            );
            assert!(root.set_context("released", json!(true)).is_ok());
            assert!(registry.remove("worker").await.is_some());
            assert_eq!(spawner.spawned_count(), 0);
            assert!(registry.remove("peer").await.is_some());
        }
    }
}

// Generated child results are not permission to release owners until the parent's exact terminal write is acknowledged.
#[tokio::test]
async fn co_completion_generated_terminal_ack_faults_preserve_child_and_registration() {
    use super::continuation_test_support::{ReceiptStore, backends, store};
    for &sqlite in backends() {
        for mode in ["positive", "reject", "lost", "drop"] {
            let spawn = call(
                "create",
                "spawn_agent",
                json!({"name":"Worker","description":"report worker"}),
            );
            let compose = call(
                "compose",
                "pipeline_process",
                json!({"input":"report","stages":["worker"]}),
            );
            let q = question();
            let provider = Script::new(
                &[
                    Some(&spawn),
                    Some("name: Worker\nsystem_prompt: evidence\ntools: [ask_user]\n"),
                    Some(&compose),
                    Some(&q),
                    Some("verified report"),
                    Some("done"),
                ],
                0,
            );
            let registry = Arc::new(AgentRegistry::new());
            let (root, spawner) = generated_parent(
                provider.clone(),
                registry.clone(),
                1,
                Some(("pipeline_process", Script::new(&[], 0))),
            );
            let backend = store(root.info().id, "co.v1", sqlite).await;
            let fault = ReceiptStore::new(backend.clone(), mode, TaskRunStatus::Completed);
            let runner = runner_with_store(root.clone(), fault.clone());
            let paused = runner.run("report", None).await.unwrap();
            assert_eq!(paused.run.status, TaskRunStatus::Paused);
            fault.armed.store(mode != "positive", Ordering::SeqCst);
            let mut work = Box::pin(runner.resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            ));
            if mode == "drop" {
                tokio::select! {
                    result = &mut work => panic!("returned before terminal receipt: {:?}", result.err()),
                    permit = fault.reached.acquire() => permit.unwrap().forget(),
                    _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => panic!("terminal write not reached"),
                }
                drop(work);
            } else if mode == "positive" {
                assert_eq!(work.await.unwrap().run.status, TaskRunStatus::Completed);
            } else {
                assert!(work.await.is_err(), "{mode}/{sqlite}");
                fault.reached.try_acquire().unwrap().forget();
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 6);
            assert_eq!(spawner.spawned_count(), 1);
            let saved = backend.load(&paused.run.key.run_id).await.unwrap().unwrap();
            let payload: TaskCheckpointPayload =
                serde_json::from_value(saved.payload.clone()).unwrap();
            assert_eq!(payload.children.len(), 1);
            assert!(payload.children[0].result.is_some() && payload.children[0].pending.is_none());
            let child = registry.get("worker").unwrap();
            if mode == "positive" {
                assert!(child.set_context("released", json!(true)).is_ok());
                assert!(root.set_context("released", json!(true)).is_ok());
                assert!(registry.remove("worker").await.is_some());
                assert_eq!(spawner.spawned_count(), 0);
            } else {
                fault.assert_hit();
                assert!(child.set_context("must_stay_owned", json!(true)).is_err());
                assert_eq!(saved.status == TaskRunStatus::Completed, mode != "reject");
                assert!(
                    root.set_context("must_stay_protected", json!(true))
                        .is_err()
                );
                assert!(registry.remove("worker").await.is_none());
                assert!(
                    runner
                        .resume(
                            &paused.run.key.run_id,
                            saved.revision,
                            TaskResumeInput::UserAnswer {
                                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                                answer: json!({"answered":true,"selected":["yes"]}),
                            }
                        )
                        .await
                        .is_err()
                );
                assert_eq!(provider.calls.load(Ordering::SeqCst), 6);
            }
        }
    }
}
