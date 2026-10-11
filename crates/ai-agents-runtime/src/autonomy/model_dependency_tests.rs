use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use crate::{AgentBuilder, RuntimeAgent};
use async_trait::async_trait;
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};

struct DelegateGate {
    entered: Arc<tokio::sync::Semaphore>,
    release: Arc<tokio::sync::Semaphore>,
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for DelegateGate {
    async fn on_delegate_start(&self, _: &str, _: &str) {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
    }
}

// Both parents are admitted before the chosen predecessor parks, using real declared and model dispatch boundaries.
async fn fixture(
    source: &str,
    destination: &str,
) -> (
    Arc<AutonomyRunner>,
    Arc<ScopedTaskRunStore>,
    Arc<RuntimeAgent>,
    Arc<AgentRegistry>,
    Arc<super::runner_tests::RecordingProvider>,
    Vec<Arc<super::runner_tests::RecordingProvider>>,
    Vec<Arc<tokio::sync::Semaphore>>,
    Arc<tokio::sync::Semaphore>,
) {
    let workers = Arc::new(AgentRegistry::new());
    let questions: Vec<_> = (0..2)
        .map(|index| {
            ai_agents_core::encode_native_tool_call_markers(
                &[ai_agents_core::ToolCall {
                    id: format!("q{index}"),
                    name: "ask_user".into(),
                    arguments: json!({"question":"choose","options":["yes"],"allow_other":false}),
                }],
                None,
            )
            .unwrap()
        })
        .collect();
    let worker = super::runner_tests::RecordingProvider::new(&[
        &questions[0],
        "done",
        &questions[1],
        "done",
    ]);
    let yaml = "name: SharedWorker\nsystem_prompt: worker\ntools: [ask_user]\n";
    let runtime = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(worker.clone())
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )))
        .build()
        .unwrap();
    workers
        .register(SpawnedAgent::from_runtime(
            "worker".into(),
            runtime,
            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
        ))
        .await
        .unwrap();
    let parents = Arc::new(AgentRegistry::new());
    let entered = Arc::new(tokio::sync::Semaphore::new(0));
    let mut releases = Vec::new();
    let mut providers = Vec::new();
    for (id, kind) in [("first", source), ("second", destination)] {
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let mut provider = super::runner_tests::RecordingProvider::new(&["unused"]);
        let (yaml, builder) = if kind == "declare" {
            let yaml = format!(
                "name: {id}\nsystem_prompt: parent\nstates:\n  initial: active\n  states:\n    active:\n      delegate: worker\n"
            );
            let builder = AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .hooks(Arc::new(DelegateGate {
                    entered: entered.clone(),
                    release: release.clone(),
                }));
            (yaml, builder)
        } else {
            let tool = if kind == "group" {
                "group_discussion"
            } else {
                "send_agent_message"
            };
            let args = if kind == "group" {
                json!({"participants":["worker"],"topic":"objective","max_rounds":1})
            } else {
                json!({"to":"worker","message":"objective"})
            };
            let marker = ai_agents_core::encode_native_tool_call_markers(
                &[ai_agents_core::ToolCall {
                    id: format!("{id}-call"),
                    name: tool.into(),
                    arguments: args,
                }],
                None,
            )
            .unwrap();
            provider = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
            let raw = Arc::get_mut(&mut provider).unwrap();
            raw.entered = Some(entered.clone());
            raw.release = Some(release.clone());
            let mut llms = ai_agents_llm::LLMRegistry::new();
            llms.register("default", provider.clone());
            llms.register(
                "router",
                super::runner_tests::RecordingProvider::new(&["synthesis"]),
            );
            let yaml = format!("name: {id}\nsystem_prompt: parent\ntools: [{tool}]\n");
            let builder = AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm_registry(llms.clone())
                .auto_configure_features()
                .unwrap();
            let builder = if kind == "group" {
                builder.tool(Arc::new(
                    crate::orchestration::tools::GroupDiscussionTool::new(
                        workers.clone(),
                        Arc::new(llms),
                    ),
                ))
            } else {
                builder.tool(Arc::new(crate::spawner::SendMessageTool::new(
                    workers.clone(),
                    id,
                )))
            };
            (yaml, builder)
        };
        parents
            .register(SpawnedAgent::from_runtime(
                id.into(),
                builder
                    .build()
                    .unwrap()
                    .with_spawner_handles(Arc::new(AgentSpawner::new()), workers.clone()),
                crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
            ))
            .await
            .unwrap();
        releases.push(release);
        providers.push(provider);
    }
    let yaml = "name: MixedRoot\nsystem_prompt: root\nstates:\n  initial: active\n  states:\n    active:\n      concurrent:\n        agents: [first, second]\n        aggregation:\n          strategy: all\n";
    let root = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&["unused"]))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), parents),
    );
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(root.info().id, None, "mixed.v1".into()).unwrap());
    let mut config = super::runner_tests::config(16);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: Some(InteractionAction::PauseRun),
    });
    let runner = Arc::new(
        AutonomyRunner::try_new(
            root.clone(),
            config,
            Default::default(),
            store.clone(),
            "mixed.v1".into(),
        )
        .unwrap(),
    );
    (
        runner, store, root, workers, worker, providers, releases, entered,
    )
}

// Gate releases choose the actual predecessor, not write ordinals or arbitrary sleeps.
async fn contract(source: &str, destination: &str, cancel: bool) {
    let (runner, store, root, registry, worker, parents, releases, entered) =
        Box::pin(fixture(source, destination)).await;
    let start = runner.clone();
    let task = tokio::spawn(async move { start.run("objective", None).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.acquire_many(2))
        .await
        .unwrap()
        .unwrap()
        .forget();
    releases[0].add_permits(8);
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if let Some(run) = store.list().await.unwrap().first() {
                let snapshot = store.load(&run.key.run_id).await.unwrap().unwrap();
                if snapshot.payload["children"]
                    .as_array()
                    .is_some_and(|children| {
                        children.iter().any(|child| {
                            child["runtime"]["snapshot"]["agent_id"] == "SharedWorker"
                                && child["pending"].is_object()
                        })
                    })
                {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    releases[1].add_permits(8);
    let first = tokio::time::timeout(std::time::Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        first.run.status,
        TaskRunStatus::Paused,
        "{source}->{destination}: {:?}",
        first.run.stop_reason
    );
    assert_eq!(worker.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&first.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    let group: TaskGroupState = serde_json::from_value(
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    assert_eq!(group.parked.len(), 1);
    assert!(
        payload.evidence.tool_calls.iter().all(|call| !call
            .record
            .output
            .contains("parked runtime")
            && !call.record.output.contains("parent dependency")),
        "{source}->{destination}"
    );
    if cancel {
        let done = runner
            .cancel_paused(&first.run.key.run_id, first.run.revision)
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Cancelled);
        assert_eq!(worker.calls.load(Ordering::SeqCst), 1);
    } else {
        let mut result = first;
        let mut count = 0;
        while result.run.status == TaskRunStatus::Paused {
            result = runner
                .resume(
                    &result.run.key.run_id,
                    result.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: result.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .await
                .unwrap();
            count += 1;
            assert!(count <= 2);
        }
        assert_eq!(
            count,
            2,
            "{source}->{destination}: records {:?}",
            result
                .run
                .evidence
                .tool_calls
                .iter()
                .map(|r| &r.record.output)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{source}->{destination}: {:?}",
            result.run.stop_reason
        );
        assert_eq!(worker.calls.load(Ordering::SeqCst), 4);
        for (index, kind) in [source, destination].into_iter().enumerate() {
            assert_eq!(
                parents[index].calls.load(Ordering::SeqCst),
                if kind == "declare" { 0 } else { 2 }
            );
        }
    }
    assert!(root.set_context("released", json!(true)).is_ok());
    assert!(registry.remove("worker").await.is_some());
}

#[tokio::test]
async fn model_dependency_declared_predecessor_and_successor_share_exact_leaf() {
    for (source, destination) in [("declare", "send"), ("send", "declare")] {
        Box::pin(contract(source, destination, false)).await;
    }
}

#[tokio::test]
async fn model_dependency_group_dynamic_call_is_a_valid_predecessor() {
    Box::pin(contract("group", "send", false)).await;
}

#[tokio::test]
async fn model_dependency_mixed_cancel_does_not_start_successor() {
    for (source, destination) in [("declare", "send"), ("send", "declare"), ("group", "send")] {
        Box::pin(contract(source, destination, true)).await;
    }
}

// Root batches validate both declared edges and dynamic group operations, including a predecessor blocked on a different slot.
async fn root_batch(kind: &str, cancel: bool) {
    let registry = Arc::new(AgentRegistry::new());
    let mut providers = Vec::new();
    for id in ["worker", "blocker"] {
        let questions: Vec<_> = (0..2).map(|index| ai_agents_core::encode_native_tool_call_markers(&[ai_agents_core::ToolCall {
            id:format!("{id}-q{index}"), name:"ask_user".into(), arguments:json!({"question":"choose","options":["yes"],"allow_other":false}),
        }],None).unwrap()).collect();
        let provider = super::runner_tests::RecordingProvider::new(&[
            &questions[0],
            "done",
            &questions[1],
            "done",
        ]);
        let yaml = format!("name: {id}\nsystem_prompt: child\ntools: [ask_user]\n");
        let agent = AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap();
        registry
            .register(SpawnedAgent::from_runtime(
                id.into(),
                agent,
                crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
            ))
            .await
            .unwrap();
        providers.push(provider);
    }
    let declared = if kind == "pipeline" {
        "pipeline:\n        stages: [blocker, worker]"
    } else {
        "delegate: worker"
    };
    let yaml = format!(
        "name: Middle\nsystem_prompt: middle\nstates:\n  initial: active\n  states:\n    active:\n      {declared}\n"
    );
    let middle = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(super::runner_tests::RecordingProvider::new(&["unused"]))
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap()
        .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone());
    registry
        .register(SpawnedAgent::from_runtime(
            "middle".into(),
            middle,
            crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
        ))
        .await
        .unwrap();
    let first = if kind == "group" {
        ai_agents_core::ToolCall {
            id: "group".into(),
            name: "group_discussion".into(),
            arguments: json!({"participants":["worker"],"topic":"objective","max_rounds":1}),
        }
    } else {
        ai_agents_core::ToolCall {
            id: "middle".into(),
            name: "send_agent_message".into(),
            arguments: json!({"to":"middle","message":"objective"}),
        }
    };
    let calls = if kind == "multi" {
        let yaml = "name: Other\nsystem_prompt: other\nstates:\n  initial: active\n  states:\n    active:\n      delegate: blocker\n";
        let other = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&["unused"]))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone());
        registry
            .register(SpawnedAgent::from_runtime(
                "other".into(),
                other,
                crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
            ))
            .await
            .unwrap();
        vec![
            first,
            ai_agents_core::ToolCall {
                id: "other".into(),
                name: "send_agent_message".into(),
                arguments: json!({"to":"other","message":"objective"}),
            },
            ai_agents_core::ToolCall {
                id: "successor".into(),
                name: "concurrent_ask".into(),
                arguments: json!({"agents":["worker","blocker"],"question":"next","aggregation":"all"}),
            },
        ]
    } else {
        vec![
            first,
            ai_agents_core::ToolCall {
                id: "successor".into(),
                name: "send_agent_message".into(),
                arguments: json!({"to":"worker","message":"next"}),
            },
        ]
    };
    let request = ai_agents_core::encode_native_tool_call_markers(&calls, None).unwrap();
    let main = super::runner_tests::RecordingProvider::new(&[&request, "done"]);
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register("default", main.clone());
    llms.register(
        "router",
        super::runner_tests::RecordingProvider::new(&["summary"]),
    );
    let yaml = "name: BatchRoot\nsystem_prompt: root\ntools: [send_agent_message, group_discussion, concurrent_ask]\n";
    let root = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_registry(llms.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(crate::spawner::SendMessageTool::new(
                registry.clone(),
                "root",
            )))
            .tool(Arc::new(
                crate::orchestration::tools::ConcurrentAskTool::new(
                    registry.clone(),
                    Arc::new(llms.clone()),
                ),
            ))
            .tool(Arc::new(
                crate::orchestration::tools::GroupDiscussionTool::new(
                    registry.clone(),
                    Arc::new(llms),
                ),
            ))
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone()),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(root.info().id, None, "root-batch.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(20);
    config.defaults.max_tool_calls = Some(10);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        root.clone(),
        config,
        Default::default(),
        store.clone(),
        "root-batch.v1".into(),
    )
    .unwrap();
    let mut result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        runner.run("objective", None),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Paused,
        "{kind}: {:?}",
        result.run.stop_reason
    );
    let snapshot = store.load(&result.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let group: TaskGroupState = serde_json::from_value(
        payload
            .adapters
            .iter()
            .find(|a| a.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    assert_eq!(group.parked.len(), if kind == "multi" { 2 } else { 1 });
    if kind == "pipeline" {
        assert_eq!(providers[0].calls.load(Ordering::SeqCst), 0);
    }
    if cancel {
        result = runner
            .cancel_paused(&result.run.key.run_id, result.run.revision)
            .await
            .unwrap();
        assert_eq!(result.run.status, TaskRunStatus::Cancelled);
    } else {
        let mut count = 0;
        while result.run.status == TaskRunStatus::Paused {
            let resumed = runner
                .resume(
                    &result.run.key.run_id,
                    result.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: result.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .await;
            if let Err(error) = resumed {
                let saved = store.load(&result.run.key.run_id).await.unwrap().unwrap();
                panic!(
                    "{kind} at answer {count}: {error:?}, counters={}, children={}, records={}",
                    saved.payload["counters"],
                    saved.payload["children"],
                    saved.payload["evidence"]["tool_calls"]
                );
            }
            result = resumed.unwrap();
            count += 1;
            assert!(count <= 4, "{kind}");
        }
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{kind}: {:?}",
            result.run.stop_reason
        );
        assert_eq!(
            count,
            if kind == "pipeline" {
                3
            } else if kind == "multi" {
                4
            } else {
                2
            }
        );
        if kind == "multi" {
            assert_eq!(providers[1].calls.load(Ordering::SeqCst), 4);
        }
        assert_eq!(providers[0].calls.load(Ordering::SeqCst), 4);
        assert!(
            result
                .run
                .evidence
                .tool_calls
                .iter()
                .all(|call| call.record.success),
            "{kind}: {:?}",
            result
                .run
                .evidence
                .tool_calls
                .iter()
                .map(|r| &r.record.output)
                .collect::<Vec<_>>()
        );
    }
    assert!(root.set_context("released", json!(true)).is_ok());
    if kind == "multi" {
        assert!(registry.remove("other").await.is_some());
    }
    for id in ["middle", "worker", "blocker"] {
        assert!(registry.remove(id).await.is_some());
    }
}

#[tokio::test]
async fn model_dependency_root_batch_accepts_declared_and_dynamic_call_predecessors() {
    for kind in ["declare", "group"] {
        Box::pin(root_batch(kind, false)).await;
    }
}

#[tokio::test]
async fn model_dependency_unstarted_target_propagates_another_predecessor_slot_pause() {
    for cancel in [false, true] {
        Box::pin(root_batch("pipeline", cancel)).await;
    }
}

#[tokio::test]
async fn model_dependency_multi_target_waits_for_all_declared_predecessors() {
    for cancel in [false, true] {
        Box::pin(root_batch("multi", cancel)).await;
    }
}

// A pipeline's later target is not a held runtime gate, but waiting for its ancestor dispatch would still be circular.
#[tokio::test]
async fn model_dependency_ancestor_catalogue_wait_rejects_before_target_dispatch() {
    let registry = Arc::new(AgentRegistry::new());
    let request = ai_agents_core::encode_native_tool_call_markers(
        &[ai_agents_core::ToolCall {
            id: "circular".into(),
            name: "send_agent_message".into(),
            arguments: json!({"to":"later","message":"should not run"}),
        }],
        None,
    )
    .unwrap();
    let main = super::runner_tests::RecordingProvider::new(&[&request, "done"]);
    let later = super::runner_tests::RecordingProvider::new(&["done"]);
    let yaml = "name: First\nsystem_prompt: first\ntools: [send_agent_message]\n";
    let first = AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(main.clone())
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(crate::spawner::SendMessageTool::new(
            registry.clone(),
            "first",
        )))
        .build()
        .unwrap()
        .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone());
    registry
        .register(SpawnedAgent::from_runtime(
            "first".into(),
            first,
            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
        ))
        .await
        .unwrap();
    let agent = AgentBuilder::new()
        .system_prompt("later")
        .llm(later.clone())
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "later".into(),
            agent,
            crate::spec::AgentSpec::default(),
        ))
        .await
        .unwrap();
    let yaml = "name: Root\nsystem_prompt: root\nstates:\n  initial: active\n  states:\n    active:\n      pipeline:\n        stages: [first, later]\n";
    let root = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&["unused"]))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(root.info().id, None, "ancestor.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        root,
        super::runner_tests::config(6),
        Default::default(),
        store,
        "ancestor.v1".into(),
    )
    .unwrap();
    let done = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        runner.run("objective", None),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(later.calls.load(Ordering::SeqCst), 1);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .any(|call| !call.record.success
                && call.record.output.contains("held ancestor dispatch"))
    );
}
