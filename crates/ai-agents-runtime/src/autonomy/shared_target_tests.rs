use super::*;
use crate::AgentBuilder;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};

// Both distinct parents are admitted before the first parks, so this exercises actual shared-target dependency custody.
#[tokio::test]
async fn shared_target_cross_parent_messages_defer_new_history_and_deduplicate_leaf_response() {
    for active in [false, true] {
        Box::pin(shared_target_contract(active, "complete", 2)).await;
    }
}

#[tokio::test]
async fn shared_target_cross_parent_cancel_and_stop_close_each_history_once() {
    for mode in ["cancel_first", "cancel_second", "stop"] {
        Box::pin(shared_target_contract(false, mode, 2)).await;
    }
}

#[tokio::test]
async fn shared_target_three_parents_serialize_successors_after_one_predecessor() {
    Box::pin(shared_target_contract(false, "complete", 3)).await;
}

#[tokio::test]
async fn shared_target_held_ancestor_is_rejected_before_dependency_wait() {
    let (runner, main, providers) = ancestor_fixture().await;
    let done = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        runner.run("objective", None),
    )
    .await
    .unwrap()
    .unwrap();
    assert_ne!(done.run.status, TaskRunStatus::RecoveryRequired);
    assert!(
        providers
            .iter()
            .all(|provider| provider.calls.load(Ordering::SeqCst) == 2)
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .any(|capture| !capture.record.success
                && capture.record.output.contains("held ancestor"))
    );
}

type AncestorFixture = (
    AutonomyRunner,
    Arc<super::runner_tests::RecordingProvider>,
    Vec<Arc<super::runner_tests::RecordingProvider>>,
);

// Fixture construction is erased so its unused temporaries do not remain on the nested execution test stack.
fn ancestor_fixture() -> std::pin::Pin<Box<dyn std::future::Future<Output = AncestorFixture> + Send>>
{
    Box::pin(async {
        let registry = Arc::new(AgentRegistry::new());
        let mut providers = Vec::new();
        for (name, target) in [("a", "b"), ("b", "a")] {
            let marker = ai_agents_core::encode_native_tool_call_markers(
                &[ai_agents_core::ToolCall {
                    id: format!("{name}-call"),
                    name: "send_agent_message".into(),
                    arguments: json!({"to":target,"message":"objective"}),
                }],
                None,
            )
            .unwrap();
            let provider = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
            let yaml = format!("name: {name}\nsystem_prompt: child\ntools: [send_agent_message]\n");
            let agent = AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
                    registry.clone(),
                    name,
                )))
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone());
            registry
                .register(SpawnedAgent::from_runtime(
                    name.into(),
                    agent,
                    crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
                ))
                .await
                .unwrap();
            providers.push(provider);
        }
        let marker = ai_agents_core::encode_native_tool_call_markers(
            &[ai_agents_core::ToolCall {
                id: "root-call".into(),
                name: "send_agent_message".into(),
                arguments: json!({"to":"a","message":"objective"}),
            }],
            None,
        )
        .unwrap();
        let main = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
        let yaml = "name: CycleRoot\nsystem_prompt: root\ntools: [send_agent_message]\n";
        let root = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(main.clone())
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
                    registry.clone(),
                    "root",
                )))
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
        );
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(root.info().id, None, "cycle.v1".into()).unwrap(),
        );
        let runner = AutonomyRunner::try_new(
            root,
            super::runner_tests::config(10),
            Default::default(),
            store,
            "cycle.v1".into(),
        )
        .unwrap();
        (runner, main, providers)
    })
}

#[tokio::test]
async fn shared_target_nested_dependency_revisits_runtime_without_execution_cycle() {
    let (runner, main, children) = nested_shared_fixture().await;
    let first = runner.run("objective", None).await.unwrap();
    assert_eq!(
        first.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        first.run.stop_reason
    );
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 1);
    let second = runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        second.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        second.run.stop_reason
    );
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 3);
    let done = runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(children[0].calls.load(Ordering::SeqCst), 4);
    assert_eq!(children[1].calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.counters.tool_attempts, 5);
}

// A model batch reaches a shared leaf directly and through a child, so frame ancestry must differ from runtime ancestry.
fn nested_shared_fixture()
-> std::pin::Pin<Box<dyn std::future::Future<Output = AncestorFixture> + Send>> {
    Box::pin(async {
        let registry = Arc::new(AgentRegistry::new());
        let question = |id: &str| {
            ai_agents_core::encode_native_tool_call_markers(
                &[ai_agents_core::ToolCall {
                    id: id.into(),
                    name: "ask_user".into(),
                    arguments: json!({"question":"Choose","options":["yes"],"allow_other":false}),
                }],
                None,
            )
            .unwrap()
        };
        let worker = super::runner_tests::RecordingProvider::new(&[
            &question("q1"),
            "done",
            &question("q2"),
            "done",
        ]);
        let yaml = "name: NestedWorker\nsystem_prompt: worker\ntools: [ask_user]\n";
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(worker.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap();
        registry
            .register(SpawnedAgent::from_runtime(
                "worker".into(),
                agent,
                crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
            ))
            .await
            .unwrap();
        let send = |id: &str, to: &str| ai_agents_core::ToolCall {
            id: id.into(),
            name: "send_agent_message".into(),
            arguments: json!({"to":to,"message":"objective"}),
        };
        let marker = ai_agents_core::encode_native_tool_call_markers(
            &[send("middle-worker", "worker")],
            None,
        )
        .unwrap();
        let middle = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
        let yaml = "name: NestedMiddle\nsystem_prompt: middle\ntools: [send_agent_message]\n";
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(middle.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
                registry.clone(),
                "middle",
            )))
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone());
        registry
            .register(SpawnedAgent::from_runtime(
                "middle".into(),
                agent,
                crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
            ))
            .await
            .unwrap();
        let marker = ai_agents_core::encode_native_tool_call_markers(
            &[send("root-worker", "worker"), send("root-middle", "middle")],
            None,
        )
        .unwrap();
        let main = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
        let yaml = "name: NestedRoot\nsystem_prompt: root\ntools: [send_agent_message]\n";
        let root = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(main.clone())
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
                    registry.clone(),
                    "root",
                )))
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
        );
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(root.info().id, None, "nested-shared.v1".into()).unwrap(),
        );
        let mut config = super::runner_tests::config(8);
        config.defaults.hitl = Some(AutonomyHitlConfig {
            on_approval_required: Some(InteractionAction::PauseRun),
            on_user_question: Some(InteractionAction::PauseRun),
        });
        let runner = AutonomyRunner::try_new(
            root,
            config,
            Default::default(),
            store,
            "nested-shared.v1".into(),
        )
        .unwrap();
        (runner, main, vec![worker, middle])
    })
}

// Gated provider execution distinguishes a genuinely active predecessor from an already acknowledged safe park.
async fn shared_target_contract(active: bool, mode: &str, parent_count: usize) {
    let workers = Arc::new(AgentRegistry::new());
    let question = |id: &str| {
        ai_agents_core::encode_native_tool_call_markers(
            &[ai_agents_core::ToolCall {
                id: id.into(),
                name: "ask_user".into(),
                arguments: json!({"question":"Choose","options":["yes"],"allow_other":false}),
            }],
            None,
        )
        .unwrap()
    };
    let mut worker_provider = super::runner_tests::RecordingProvider::new(&[
        &question("q1"),
        "done",
        &question("q2"),
        "done",
        &question("q3"),
        "done",
    ]);
    let worker_entered = Arc::new(tokio::sync::Semaphore::new(0));
    let worker_release = Arc::new(tokio::sync::Semaphore::new(0));
    if active {
        let live = Arc::get_mut(&mut worker_provider).unwrap();
        live.entered = Some(worker_entered.clone());
        live.release = Some(worker_release.clone());
    }
    let worker_yaml = "name: SharedWorker\nsystem_prompt: worker\ntools: [ask_user]\n";
    let worker = AgentBuilder::from_yaml(worker_yaml)
        .unwrap()
        .llm(worker_provider.clone())
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
            worker,
            crate::spec::AgentSpec::from_yaml_strict(worker_yaml).unwrap(),
        ))
        .await
        .unwrap();
    let registered = workers.get("worker").unwrap();
    let root_registry = Arc::new(AgentRegistry::new());
    let entered = Arc::new(tokio::sync::Semaphore::new(0));
    let mut releases = Vec::new();
    let mut parents = Vec::new();
    for id in ["parent_a", "parent_b", "parent_c"]
        .into_iter()
        .take(parent_count)
    {
        let marker = ai_agents_core::encode_native_tool_call_markers(
            &[ai_agents_core::ToolCall {
                id: format!("{id}-message"),
                name: "send_agent_message".into(),
                arguments: json!({"to":"worker","message":"objective"}),
            }],
            None,
        )
        .unwrap();
        let mut provider = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let live = Arc::get_mut(&mut provider).unwrap();
        live.entered = Some(entered.clone());
        live.release = Some(release.clone());
        releases.push(release);
        let yaml = format!("name: {id}\nsystem_prompt: parent\ntools: [send_agent_message]\n");
        let parent = AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
                workers.clone(),
                id,
            )))
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), workers.clone());
        root_registry
            .register(SpawnedAgent::from_runtime(
                id.into(),
                parent,
                crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
            ))
            .await
            .unwrap();
        parents.push(provider);
    }
    let root_yaml = format!(
        "name: Root\nsystem_prompt: root\nstates:\n  initial: active\n  states:\n    active:\n      concurrent:\n        agents: [{}]\n        aggregation:\n          strategy: all\n",
        ["parent_a", "parent_b", "parent_c"][..parent_count].join(", ")
    );
    let root = Arc::new(
        AgentBuilder::from_yaml(&root_yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&["unused"]))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), root_registry),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(root.info().id, None, "shared-target.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(12);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = Arc::new(
        AutonomyRunner::try_new(
            root.clone(),
            config,
            Default::default(),
            store.clone(),
            "shared-target.v1".into(),
        )
        .unwrap(),
    );
    let starting = runner.clone();
    let task = tokio::spawn(async move { starting.run("objective", None).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        entered
            .acquire_many(parent_count as u32)
            .await
            .unwrap()
            .forget();
    })
    .await
    .unwrap();
    releases[0].add_permits(4);
    if active {
        tokio::time::timeout(std::time::Duration::from_secs(5), worker_entered.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        releases[1].add_permits(4);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let runs = store.list().await.unwrap();
                if let Some(run) = runs.first() {
                    let snapshot = store.load(&run.key.run_id).await.unwrap().unwrap();
                    if snapshot.payload["adapters"]
                        .as_array()
                        .is_some_and(|adapters| {
                            adapters.iter().any(|adapter| {
                                adapter["state"]["runtime_id"] == "parent_b"
                                    && adapter["state"]["cursor"]["waiting_for"].is_string()
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
        worker_release.add_permits(6);
    }
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let runs = store.list().await.unwrap();
            if let Some(run) = runs.first() {
                let snapshot = store.load(&run.key.run_id).await.unwrap().unwrap();
                let payload: TaskCheckpointPayload =
                    serde_json::from_value(snapshot.payload).unwrap();
                if payload.children.iter().any(|child| {
                    child.runtime.snapshot.agent_id == "SharedWorker" && child.pending.is_some()
                }) {
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    for release in &releases[1..] {
        release.add_permits(4);
    }
    let first = task.await.unwrap().unwrap();
    assert_eq!(
        first.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        first.run.stop_reason
    );
    assert_eq!(worker_provider.calls.load(Ordering::SeqCst), 1);
    if matches!(mode, "cancel_first" | "stop") {
        let stopped = if mode == "stop" {
            runner
                .stop_paused(
                    &first.run.key.run_id,
                    first.run.revision,
                    "Host stopped shared worker",
                )
                .await
                .unwrap()
        } else {
            runner
                .cancel_paused(&first.run.key.run_id, first.run.revision)
                .await
                .unwrap()
        };
        assert_eq!(
            stopped.run.status,
            if mode == "stop" {
                TaskRunStatus::Incomplete
            } else {
                TaskRunStatus::Cancelled
            }
        );
        assert_eq!(worker_provider.calls.load(Ordering::SeqCst), 1);
        assert!(root.set_context("released", json!(true)).is_ok());
        assert!(registered.set_context("released", json!(true)).is_ok());
        return;
    }
    let second = runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        second.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        second.run.stop_reason
    );
    assert_eq!(worker_provider.calls.load(Ordering::SeqCst), 3);
    if mode == "cancel_second" {
        let stopped = runner
            .cancel_paused(&second.run.key.run_id, second.run.revision)
            .await
            .unwrap();
        assert_eq!(stopped.run.status, TaskRunStatus::Cancelled);
        assert_eq!(worker_provider.calls.load(Ordering::SeqCst), 3);
        assert!(root.set_context("released", json!(true)).is_ok());
        assert!(registered.set_context("released", json!(true)).is_ok());
        return;
    }
    let mut done = runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    if parent_count == 3 {
        assert_eq!(
            done.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(worker_provider.calls.load(Ordering::SeqCst), 5);
        done = runner
            .resume(
                &done.run.key.run_id,
                done.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: done.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(
        worker_provider.calls.load(Ordering::SeqCst),
        parent_count * 2
    );
    assert!(
        parents
            .iter()
            .all(|parent| parent.calls.load(Ordering::SeqCst) == 2)
    );
    assert_eq!(done.run.counters.tool_attempts, parent_count as u64 * 2);
    assert!(root.set_context("released", json!(true)).is_ok());
    assert!(registered.set_context("released", json!(true)).is_ok());
}
