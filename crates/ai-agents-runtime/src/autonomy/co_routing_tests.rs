//! Selector and aggregation waits preserve composition authority without changing ordinary selection or failure semantics.

use super::co_completion_tests::{Script, answer, question, register, runner};
use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner};
use crate::{AgentBuilder, RuntimeAgent};
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};

// Gated LLM selection is actual auxiliary work in the shared ledger, not an executable child permit.
#[tokio::test]
async fn co_completion_gated_selector_preserves_selected_instance_and_cancellation() {
    for cancel in [false, true] {
        let registry = Arc::new(AgentRegistry::new());
        let q = question();
        let selected = Script::new(&[Some(&q), Some("verified ticket")], 0);
        let unused = Script::new(&[Some("must not execute")], 0);
        let yaml = "name: SupportWorker\nsystem_prompt: ticket evidence\ntools: [ask_user]\n";
        let worker = register(
            &registry,
            "worker",
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(selected.clone())
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                    Arc::new(parking_lot::RwLock::new(None)),
                )))
                .build()
                .unwrap(),
            yaml,
        )
        .await;
        register(
            &registry,
            "other",
            AgentBuilder::new()
                .system_prompt("unused")
                .llm(unused.clone())
                .build()
                .unwrap(),
            "name: Other\nsystem_prompt: unused\n",
        )
        .await;
        let route = ai_agents_core::encode_native_tool_call_markers(&[ai_agents_core::ToolCall { id: "route".into(), name: "route_to_agent".into(),
            arguments: json!({"input":"resolve ticket","candidates":["other","worker"],"method":"llm"}) }], None).unwrap();
        let main = Script::new(&[Some(&route), Some("done: resolved ticket")], 0);
        let selector = Script::new(&[Some("worker")], 1);
        let mut llms = ai_agents_llm::LLMRegistry::new();
        llms.register("default", main.clone());
        llms.register("router", selector.clone());
        let root = Arc::new(
            AgentBuilder::from_yaml(
                "name: Support\nsystem_prompt: support\ntools: [route_to_agent]\n",
            )
            .unwrap()
            .llm_registry(llms.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(
                crate::orchestration::tools::RouteToAgentTool::new(
                    registry.clone(),
                    Arc::new(llms),
                ),
            ))
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone()),
        );
        let (runner, store) = runner(root.clone());
        let mut work = Box::pin(runner.run("resolve ticket", None));
        tokio::select! {
            result = &mut work => panic!("returned before selector: {:?}", result.err()),
            permit = selector.entered.acquire() => permit.unwrap().forget(),
        }
        assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
        assert_eq!(unused.calls.load(Ordering::SeqCst), 0);
        if cancel {
            let run = store.list().await.unwrap().remove(0);
            let receipt = runner.request_cancel(&run.key.run_id).await.unwrap();
            assert_ne!(receipt.status, TaskRunStatus::Completed);
            assert!(root.set_context("still_active", json!(true)).is_err());
        }
        selector.release.add_permits(1);
        let mut result = work.await.unwrap();
        if cancel {
            assert_ne!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
            assert_eq!(main.calls.load(Ordering::SeqCst), 1);
        } else {
            assert_eq!(result.run.status, TaskRunStatus::Paused);
            assert!(registry.remove("worker").await.is_none());
            assert!(Arc::ptr_eq(&worker, &registry.get("worker").unwrap()));
            result = answer(&runner, &result.run).await;
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                result.run.stop_reason
            );
            assert_eq!(selected.calls.load(Ordering::SeqCst), 2);
            assert_eq!(main.calls.load(Ordering::SeqCst), 2);
            assert_eq!(result.run.counters.llm_attempts, 5);
            let record = result
                .run
                .evidence
                .tool_calls
                .iter()
                .find(|call| call.record.canonical_id == "route_to_agent")
                .unwrap();
            let output: serde_json::Value = serde_json::from_str(&record.record.output).unwrap();
            assert_eq!(output["selected_agent"], "worker");
            assert!(record.record.executed && record.record.success);
        }
        assert_eq!(selector.calls.load(Ordering::SeqCst), 1);
        assert_eq!(unused.calls.load(Ordering::SeqCst), 0);
    }
}

struct AggregateFixture {
    runner: Arc<AutonomyRunner>,
    store: Arc<ScopedTaskRunStore>,
    root: Arc<RuntimeAgent>,
    registry: Arc<AgentRegistry>,
    first: Arc<Script>,
    fast: Arc<Script>,
    failed: Arc<Script>,
    auxiliary: Arc<Script>,
}

// The first declared result is slower because it asks a question; a later result and an optional failure already settle before resume.
async fn aggregate_fixture(strategy: &str, policy: &str, interrupt: bool) -> AggregateFixture {
    let registry = Arc::new(AgentRegistry::new());
    let q = question();
    let first = Script::new(&[Some(&q), Some("done: first verified evidence")], 0);
    let fast = Script::new(&[Some("done: later verified evidence")], 0);
    let failed = Script::new(&[None], 0);
    for (id, provider) in [
        ("first", first.clone()),
        ("fast", fast.clone()),
        ("failed", failed.clone()),
    ] {
        let yaml = format!("name: {id}\nsystem_prompt: research evidence\ntools: [ask_user]\n");
        register(
            &registry,
            id,
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider)
                .auto_configure_features()
                .unwrap()
                .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                    Arc::new(parking_lot::RwLock::new(None)),
                )))
                .build()
                .unwrap(),
            &yaml,
        )
        .await;
    }
    let auxiliary = Script::new(
        &[
            Some("done: verified synthesis"),
            Some("done: verified synthesis"),
        ],
        usize::from(interrupt),
    );
    let yaml = format!(
        "name: Research\nsystem_prompt: research\nstates:\n  initial: collect\n  states:\n    collect:\n      concurrent:\n        agents: [first, fast, failed]\n        on_partial_failure: {}\n{}        aggregation:\n          strategy: {strategy}\n          synthesizer_llm: router\n",
        if policy == "abort" {
            "abort"
        } else {
            "proceed_with_available"
        },
        if policy == "minimum" {
            "        min_required: 3\n"
        } else {
            ""
        }
    );
    let root = Arc::new(
        AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(Script::new(&[], 0))
            .llm_alias("router", auxiliary.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone()),
    );
    let (runner, store) = runner(root.clone());
    AggregateFixture {
        runner,
        store,
        root,
        registry,
        first,
        fast,
        failed,
        auxiliary,
    }
}

// Aggregation consumes ordered successful outcomes only; required failure and unmet minima cannot be hidden by synthesis or votes.
#[tokio::test]
async fn co_completion_resumed_aggregation_preserves_order_failure_policy_and_shared_calls() {
    for strategy in ["all", "first_wins", "llm_synthesis", "voting"] {
        for policy in ["optional", "abort", "minimum"] {
            let f = Box::pin(aggregate_fixture(strategy, policy, false)).await;
            let mut result = f
                .runner
                .run("synthesize verified research", None)
                .await
                .unwrap();
            if result.run.status == TaskRunStatus::Paused {
                assert_eq!(f.auxiliary.calls.load(Ordering::SeqCst), 0);
                result = answer(&f.runner, &result.run).await;
            }
            if policy == "optional" {
                assert_eq!(
                    result.run.status,
                    TaskRunStatus::Completed,
                    "{strategy}: {:?}",
                    result.run.stop_reason
                );
                assert_eq!(f.first.calls.load(Ordering::SeqCst), 2);
                assert_eq!(f.fast.calls.load(Ordering::SeqCst), 1);
                assert_eq!(f.failed.calls.load(Ordering::SeqCst), 1);
                let expected_aux = match strategy {
                    "llm_synthesis" => 1,
                    "voting" => 2,
                    _ => 0,
                };
                assert_eq!(
                    f.auxiliary.calls.load(Ordering::SeqCst),
                    expected_aux,
                    "{strategy}"
                );
                assert_eq!(result.run.counters.llm_attempts as usize, 4 + expected_aux);
                if strategy == "first_wins" {
                    assert!(
                        result
                            .final_response
                            .as_ref()
                            .unwrap()
                            .content
                            .contains("first verified evidence")
                    );
                }
                if strategy == "llm_synthesis" {
                    let inputs = f.auxiliary.inputs.lock();
                    let prompt = inputs[0]
                        .iter()
                        .map(|message| message.content.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    assert!(
                        prompt.find("first verified evidence").unwrap()
                            < prompt.find("later verified evidence").unwrap()
                    );
                    assert!(!prompt.contains("injected settled provider failure"));
                }
                assert!(f.root.set_context("released", json!(true)).is_ok());
                for id in ["first", "fast", "failed"] {
                    assert!(f.registry.remove(id).await.is_some());
                }
            } else {
                assert_ne!(
                    result.run.status,
                    TaskRunStatus::Completed,
                    "{strategy}/{policy}"
                );
                assert_eq!(
                    f.auxiliary.calls.load(Ordering::SeqCst),
                    0,
                    "{strategy}/{policy}"
                );
            }
        }
    }
}

// Completed children stay reserved while actual aggregation is in flight; dropping the parent is not terminal acknowledgement.
#[tokio::test]
async fn co_completion_dropped_resumed_aggregation_keeps_completed_children_protected() {
    let f = Box::pin(aggregate_fixture("llm_synthesis", "optional", true)).await;
    let paused = f.runner.run("research", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let mut work = Box::pin(answer(&f.runner, &paused.run));
    tokio::select! {
        _ = &mut work => panic!("returned before aggregation release"),
        permit = f.auxiliary.entered.acquire() => permit.unwrap().forget(),
    }
    let saved = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert!(
        saved.payload["children"]
            .as_array()
            .unwrap()
            .iter()
            .all(|child| child["result"].is_object() && child["pending"].is_null())
    );
    for id in ["first", "fast", "failed"] {
        assert!(f.registry.remove(id).await.is_none());
    }
    drop(work);
    assert!(
        f.root
            .set_context("must_stay_protected", json!(true))
            .is_err()
    );
    assert!(f.runner.run("do not replay", None).await.is_err());
    assert_eq!(f.first.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.fast.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.auxiliary.calls.load(Ordering::SeqCst), 1);
    for id in ["first", "fast", "failed"] {
        assert!(f.registry.remove(id).await.is_none());
    }
}
