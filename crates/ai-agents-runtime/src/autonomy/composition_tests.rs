use super::*;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
use crate::{Agent, RuntimeAgent};
use ai_agents_core::{Tool, ToolCall, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CounterTool {
    id: &'static str,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for CounterTool {
    fn id(&self) -> &str {
        self.id
    }
    fn name(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        "Count actual shared invocations"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("executed")
    }
}

struct NoApproval;
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for NoApproval {
    async fn request_approval(
        &self,
        _: ai_agents_hitl::ApprovalRequest,
    ) -> ai_agents_hitl::ApprovalResult {
        panic!("group pause polled an ordinary approval handler")
    }
}

type GroupFixture = (
    Arc<RuntimeAgent>,
    Arc<RuntimeAgent>,
    AutonomyRunner,
    Arc<ScopedTaskRunStore>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<super::runner_tests::RecordingProvider>,
);

// A production state dispatch owns a real registered child and a signed batch whose later result may already be complete.
async fn fixture(question: bool) -> GroupFixture {
    Box::pin(fixture_pattern(question, "delegate: worker")).await
}

// Real state configurations share the same child fixtures, exposing orchestration boundaries rather than substitute executors.
async fn fixture_pattern(question: bool, pattern: &str) -> GroupFixture {
    Box::pin(fixture_pattern_gated(question, pattern, false)).await
}

// Shared semaphores admit both provider requests before either can park, making multi-leaf cleanup tests deterministic.
async fn fixture_pattern_gated(question: bool, pattern: &str, gated: bool) -> GroupFixture {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let pending = if question {
        ToolCall {
            id: "pending".into(),
            name: "ask_user".into(),
            arguments: json!({"question":"Select", "options":["yes"], "allow_other":false}),
        }
    } else {
        ToolCall {
            id: "pending".into(),
            name: "first".into(),
            arguments: json!({}),
        }
    };
    let calls = vec![
        pending,
        ToolCall {
            id: "completed".into(),
            name: "second".into(),
            arguments: json!({}),
        },
    ];
    let native = ai_agents_core::NativeProviderState::new(
        "child-exchange",
        "fixture",
        "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://fixture.invalid", "fixture-model")
            .unwrap(),
        json!({"signature":"exact-child-state"}),
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| ai_agents_core::NativeCallBinding::new(&call.id, index).unwrap())
            .collect(),
    )
    .unwrap();
    let marker = ai_agents_core::encode_native_tool_call_markers(&calls, Some(&native)).unwrap();
    let second_marker = ai_agents_core::encode_native_tool_call_markers(&calls, None).unwrap();
    let mut provider = if pattern.contains("middle, middle") {
        super::runner_tests::RecordingProvider::new(&[&marker, "done", &second_marker, "done"])
    } else {
        super::runner_tests::RecordingProvider::new(&[&marker, "done"])
    };
    let entered = Arc::new(tokio::sync::Semaphore::new(0));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    if gated {
        let live = Arc::get_mut(&mut provider).unwrap();
        live.entered = Some(entered.clone());
        live.release = Some(release.clone());
    }
    let yaml = if question {
        "name: Worker\nsystem_prompt: child\ntools: [ask_user, second]\n"
    } else {
        "name: Worker\nsystem_prompt: child\ntools: [first, second]\nhitl:\n  tools:\n    first:\n      require_approval: true\n"
    };
    let builder = crate::AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(provider.clone())
        .tool(Arc::new(CounterTool {
            id: "second",
            calls: second.clone(),
        }))
        .auto_configure_features()
        .unwrap()
        .approval_handler(Arc::new(NoApproval));
    let builder = if question {
        builder.tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )))
    } else {
        builder.tool(Arc::new(CounterTool {
            id: "first",
            calls: first.clone(),
        }))
    };
    let child = builder.build().unwrap();
    let registry = Arc::new(AgentRegistry::new());
    Box::pin(registry.register(SpawnedAgent::from_runtime(
        "worker".into(),
        child,
        crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
    )))
    .await
    .unwrap();
    let child = registry.get("worker").unwrap();
    let peer_yaml = "name: Peer\nsystem_prompt: peer\ntools: [ask_user]\n";
    let peer_call = ToolCall {
        id: "peer-question".into(),
        name: "ask_user".into(),
        arguments: json!({"question":"Peer choice","options":["yes"],"allow_other":false}),
    };
    let peer_marker = ai_agents_core::encode_native_tool_call_markers(&[peer_call], None).unwrap();
    let mut peer_provider = if pattern.contains("middle, middle") {
        super::runner_tests::RecordingProvider::new(&[
            &peer_marker,
            "peer done",
            &peer_marker,
            "peer done",
        ])
    } else {
        super::runner_tests::RecordingProvider::new(&[&peer_marker, "peer done"])
    };
    if gated {
        let live = Arc::get_mut(&mut peer_provider).unwrap();
        live.entered = Some(entered);
        live.release = Some(release);
    }
    let peer = crate::AgentBuilder::from_yaml(peer_yaml)
        .unwrap()
        .llm(peer_provider)
        .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
            Arc::new(parking_lot::RwLock::new(None)),
        )))
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "peer".into(),
            peer,
            crate::spec::AgentSpec::from_yaml_strict(peer_yaml).unwrap(),
        ))
        .await
        .unwrap();
    let registry = if pattern.contains("middle") {
        let middle_yaml = "name: Middle\nsystem_prompt: middle\nstates:\n  initial: active\n  states:\n    active:\n      concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all\n";
        let middle = crate::AgentBuilder::from_yaml(middle_yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&[
                "unused middle",
            ]))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry);
        let outer = Arc::new(AgentRegistry::new());
        Box::pin(outer.register(SpawnedAgent::from_runtime(
            "middle".into(),
            middle,
            crate::spec::AgentSpec::from_yaml_strict(middle_yaml).unwrap(),
        )))
        .await
        .unwrap();
        outer
    } else {
        registry
    };
    let parent_yaml = format!(
        "name: GroupParent\nsystem_prompt: parent\nstates:\n  initial: active\n  states:\n    active:\n      {pattern}\n"
    );
    let parent = Arc::new(
        crate::AgentBuilder::from_yaml(&parent_yaml)
            .unwrap()
            .llm(super::runner_tests::RecordingProvider::new(&[
                "unused parent",
            ]))
            .llm_alias(
                "router",
                super::runner_tests::RecordingProvider::new(&[
                    r#"{"action":"peer","confidence":1.0,"reason":"next"}"#,
                    r#"{"action":"stay","confidence":1.0,"reason":"done"}"#,
                ]),
            )
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
    );
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(parent.info().id, None, "group.v1".into()).unwrap());
    let mut config = super::runner_tests::config(8);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        parent.clone(),
        config,
        Default::default(),
        store.clone(),
        "group.v1".into(),
    )
    .unwrap();
    (parent, child, runner, store, first, second, provider)
}

// An already expired parent dispatch denies provider admission rather than executing a first call before its timeout is observed.
#[tokio::test]
async fn zero_composition_timeout_does_not_admit_child_provider() {
    let pattern = "concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all\n        timeout_ms: 0";
    let (_, _, runner, _, first, completed, provider) = fixture_pattern(false, pattern).await;
    let finished = runner.run("objective", None).await.unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Incomplete);
    assert_eq!(
        finished.run.stop_reason.as_deref(),
        Some("composition_timeout")
    );
    assert_eq!(finished.run.counters.llm_attempts, 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(completed.load(Ordering::SeqCst), 0);
}

// The owning public entry consumes cursor permission once, preventing aggregation callbacks from reusing its cached sibling results.
#[tokio::test]
async fn composition_dispatch_permission_is_single_use_and_restores_after_callback() {
    composition::scope_composition("parent".into(), async {
        assert!(composition::enter_composition_dispatch());
        assert!(!composition::enter_composition_dispatch());
        composition::scope_dispatch_authority(false, async {
            assert!(composition::current_composition().is_none());
            assert!(!composition::enter_composition_dispatch());
        })
        .await;
        assert!(composition::current_composition().is_some());
        assert!(!composition::enter_composition_dispatch());
    })
    .await;
}

// Dynamic group-chat catalogue slots never replace the actual pending intermediate invocation during cancellation.
#[tokio::test]
async fn nested_group_chat_cancellation_uses_dynamic_incoming_operation() {
    let pattern = "group_chat:\n        participants:\n          - id: middle\n        max_rounds: 1\n        termination:\n          method: max_rounds";
    let (_, _, runner, store, first, _, provider) = fixture_pattern(false, pattern).await;
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let finished = runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Cancelled);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&finished.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.result.is_some() && child.pending.is_none())
    );
}

// An earlier settled invocation of the same intermediate runtime remains immutable when a later invocation is cancelled.
#[tokio::test]
async fn repeated_nested_pipeline_cancellation_preserves_completed_invocation() {
    let pattern = "pipeline:\n        stages: [middle, middle]";
    let (_, _, runner, store, _, _, _) = fixture_pattern(true, pattern).await;
    let mut paused = runner.run("objective", None).await.unwrap();
    let run_id = paused.run.key.run_id.clone();
    for _ in 0..2 {
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        paused = runner
            .resume(
                &run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: paused.run.pending.unwrap().id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
    }
    let before: TaskCheckpointPayload =
        serde_json::from_value(store.load(&run_id).await.unwrap().unwrap().payload).unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "children: {:?}",
        before
            .children
            .iter()
            .map(|child| (&child.child_id, &child.result))
            .collect::<Vec<_>>()
    );
    let completed = before
        .children
        .iter()
        .filter(|child| child.result.is_some())
        .map(|child| (child.child_id.clone(), serde_json::to_value(child).unwrap()))
        .collect::<Vec<_>>();
    assert!(!completed.is_empty());
    let finished = runner
        .cancel_paused(&run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Cancelled);
    let after: TaskCheckpointPayload =
        serde_json::from_value(store.load(&run_id).await.unwrap().unwrap().payload).unwrap();
    for (operation, original) in completed {
        let child = after
            .children
            .iter()
            .find(|child| child.child_id == operation)
            .unwrap();
        assert_eq!(serde_json::to_value(child).unwrap(), original);
    }
    assert!(
        after
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
}

// Each conversation style resumes its own exact position while repeated legitimate turns receive distinct operations.
#[tokio::test]
async fn group_chat_styles_resume_saved_conversation_positions() {
    for (settings, worker_calls, attempts) in [
        (
            "style: brainstorm\n        max_rounds: 1\n        termination:\n          method: max_rounds",
            2,
            4,
        ),
        (
            "style: consensus\n        max_rounds: 1\n        termination:\n          method: max_rounds",
            2,
            5,
        ),
        (
            "style: brainstorm\n        max_rounds: 1\n        manager:\n          agent: peer\n          method: llm_directed\n        termination:\n          method: manager_decides",
            2,
            7,
        ),
        (
            "style: maker_checker\n        maker_checker:\n          max_iterations: 1\n          acceptance_criteria: complete",
            2,
            4,
        ),
        (
            "style: debate\n        debate:\n          rounds: 1\n          synthesizer: worker",
            3,
            5,
        ),
        (
            "style: brainstorm\n        max_rounds: 1\n        manager:\n          method: llm_directed\n        termination:\n          method: max_rounds",
            2,
            6,
        ),
    ] {
        let pattern = format!(
            "group_chat:\n        participants:\n          - id: worker\n          - id: peer\n        {settings}"
        );
        let (_, _, runner, store, _, completed, provider) =
            Box::pin(fixture_pattern(true, &pattern)).await;
        let mut result = runner.run("objective", None).await.unwrap();
        let run_id = result.run.key.run_id.clone();
        let mut pauses = 0;
        for _ in 0..4 {
            if result.run.status != TaskRunStatus::Paused {
                break;
            }
            pauses += 1;
            result = runner
                .resume(
                    &run_id,
                    result.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: result.run.pending.unwrap().id,
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .await
                .unwrap();
        }
        assert_eq!(result.run.status, TaskRunStatus::Completed, "{settings}");
        assert_eq!(pauses, 2, "{settings}");
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            worker_calls,
            "{settings}"
        );
        assert_eq!(completed.load(Ordering::SeqCst), 1, "{settings}");
        assert_eq!(result.run.counters.llm_attempts, attempts, "{settings}");
        let payload: TaskCheckpointPayload =
            serde_json::from_value(store.load(&run_id).await.unwrap().unwrap().payload).unwrap();
        assert!(
            payload
                .children
                .iter()
                .all(|child| child.result.is_some() && child.pending.is_none())
        );
    }
}

// The saved handoff location preserves the earlier child and routing decision when the next specialist parks.
#[tokio::test]
async fn handoff_questions_resume_saved_chain_without_repeating_decisions() {
    let pattern = "handoff:\n        initial_agent: worker\n        available_agents: [worker, peer]\n        max_handoffs: 2";
    let (_, _, runner, store, _, completed, provider) = fixture_pattern(true, pattern).await;
    let mut result = runner.run("objective", None).await.unwrap();
    let run_id = result.run.key.run_id.clone();
    let mut pauses = 0;
    for _ in 0..4 {
        if result.run.status != TaskRunStatus::Paused {
            break;
        }
        pauses += 1;
        result = runner
            .resume(
                &run_id,
                result.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: result.run.pending.unwrap().id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(pauses, 2);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(completed.load(Ordering::SeqCst), 1);
    assert_eq!(result.run.counters.llm_attempts, 6);
    let payload: TaskCheckpointPayload =
        serde_json::from_value(store.load(&run_id).await.unwrap().unwrap().payload).unwrap();
    assert_eq!(payload.children.len(), 2);
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.result.is_some() && child.pending.is_none())
    );
}

// SQL-backed continuation executes the same multi-leaf CAS contract, including rejection of stale responses.
#[tokio::test]
async fn sqlite_concurrent_continuation_rejects_stale_receipts() {
    let pattern = "concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all";
    let (parent, _, _, _, _, completed, provider) = fixture_pattern(true, pattern).await;
    let store = Arc::new(
        ScopedTaskRunStore::new(
            Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
            parent.info().id,
            None,
            "group.v1".into(),
        )
        .unwrap(),
    );
    let mut config = super::runner_tests::config(8);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner =
        AutonomyRunner::try_new(parent, config, Default::default(), store, "group.v1".into())
            .unwrap();
    let first = runner.run("objective", None).await.unwrap();
    let run_id = first.run.key.run_id.clone();
    let old_revision = first.run.revision;
    let old_id = first.run.pending.as_ref().unwrap().id.clone();
    let mut result = runner
        .resume(
            &run_id,
            old_revision,
            TaskResumeInput::UserAnswer {
                request_id: old_id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert!(
        runner
            .resume(
                &run_id,
                old_revision,
                TaskResumeInput::UserAnswer {
                    request_id: old_id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                }
            )
            .await
            .is_err()
    );
    for _ in 0..3 {
        if result.run.status != TaskRunStatus::Paused {
            break;
        }
        result = runner
            .resume(
                &run_id,
                result.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: result.run.pending.unwrap().id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(completed.load(Ordering::SeqCst), 1);
}

// Original composition expiry wins over a valid answer; cleanup closes native history without another provider or pending tool call.
#[tokio::test]
async fn expired_pipeline_resume_closes_parked_history_without_invocation() {
    let pattern = "pipeline:\n        stages: [worker, peer]\n        timeout_ms: 5000";
    let (_, child, runner, store, first, _, provider) = fixture_pattern(false, pattern).await;
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&paused.run.key.run_id)
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
    assert_eq!(
        group.frame.cursor["pipeline"]["expiry"],
        serde_json::to_value(group.frame.expires_at).unwrap()
    );
    let remaining = (group.frame.expires_at.unwrap() - chrono::Utc::now())
        .to_std()
        .unwrap_or_default();
    tokio::time::sleep(remaining + std::time::Duration::from_millis(10)).await;
    let finished = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Incomplete);
    assert_eq!(
        finished.run.stop_reason.as_deref(),
        Some("composition_timeout")
    );
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&finished.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
    assert!(child.chat("ordinary").await.is_ok());
}

// A nested parent's exact dispatch is retained while leaf answers remain bound to the coordinating request.
#[tokio::test]
async fn nested_delegated_concurrent_questions_resume_without_parent_replay() {
    let (_, _, runner, store, _, completed, provider) =
        fixture_pattern(true, "delegate: middle").await;
    let mut result = runner.run("objective", None).await.unwrap();
    let run_id = result.run.key.run_id.clone();
    for _ in 0..4 {
        if result.run.status != TaskRunStatus::Paused {
            break;
        }
        result = runner
            .resume(
                &run_id,
                result.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: result.run.pending.unwrap().id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(completed.load(Ordering::SeqCst), 1);
    assert_eq!(result.run.counters.llm_attempts, 4);
    let payload: TaskCheckpointPayload =
        serde_json::from_value(store.load(&run_id).await.unwrap().unwrap().payload).unwrap();
    assert_eq!(payload.children.len(), 3);
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.result.is_some() && child.pending.is_none())
    );
}

// Nested cancellation closes leaves and intermediate frames before releasing the shared participant owners.
#[tokio::test]
async fn nested_group_cancellation_retires_intermediate_parent() {
    let (parent, child, runner, store, first, _, provider) =
        fixture_pattern(false, "delegate: middle").await;

    let paused = runner.run("objective", None).await.unwrap();

    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let cancelled = runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&cancelled.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.result.is_some() && child.pending.is_none())
    );
    assert!(parent.chat("ordinary").await.is_ok());
    assert!(child.chat("ordinary").await.is_ok());
}

// A child inherits only barrier custody, never permission to read or overwrite its parent's saved orchestration cursor.
#[tokio::test]
async fn child_scope_does_not_expose_parent_dispatch_cursor() {
    composition::scope_composition("parent".into(), async {
        let scope = composition::current_composition().unwrap();
        let slot = composition::CompositionChild {
            registry_id: "child".into(),
            runtime_id: "child".into(),
            operation: "operation".into(),
        };
        composition::run_composition_child(Some(scope), Some(slot), None, async {
            assert!(composition::current_composition().is_none());
            assert_eq!(
                composition::current_child_invocation("child"),
                Some("operation".into())
            );
            Ok(ai_agents_core::AgentResponse::new("done"))
        })
        .await
        .unwrap();
        assert!(composition::current_composition().is_some());
    })
    .await;
}

// Closing admission prevents a not-started sibling from being polled without setting permanent task cancellation.
#[tokio::test]
async fn closed_group_barrier_does_not_poll_not_started_sibling() {
    let polls = Arc::new(AtomicUsize::new(0));
    composition::scope_composition("parent".into(), async {
        let scope = composition::current_composition().unwrap();
        composition::close_composition_admission();
        let slot = composition::CompositionChild {
            registry_id: "child".into(),
            runtime_id: "child".into(),
            operation: "operation".into(),
        };
        let result = composition::run_composition_child(Some(scope), Some(slot), None, async {
            polls.fetch_add(1, Ordering::SeqCst);
            Ok(ai_agents_core::AgentResponse::new("done"))
        })
        .await;
        assert!(matches!(
            result,
            Err(ai_agents_core::AgentError::TaskSuspended(_))
        ));
    })
    .await;
    assert_eq!(polls.load(Ordering::SeqCst), 0);
}

// Both parked leaves are acknowledged before a rejection stops the parent; the other leaf is closed without invocation.
#[tokio::test]
async fn concurrent_rejection_retires_other_parked_leaf() {
    let pattern = "concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all";
    let (parent, child, runner, store, first, _, provider) =
        fixture_pattern_gated(false, pattern, true).await;
    let run = runner.run("objective", None);
    tokio::pin!(run);
    tokio::select! {
        _ = provider.entered.as_ref().unwrap().acquire_many(2) => {},
        result = &mut run => panic!("group completed before both providers entered: {result:?}"),
    }
    provider.release.as_ref().unwrap().add_permits(100);
    let paused = run.await.unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&paused.run.key.run_id)
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
    assert_eq!(group.parked.len(), 2);
    let finished = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::Rejected {
                    reason: Some("no".into()),
                },
            },
        )
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Failed);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&finished.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(payload.pending.is_none());
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
    assert!(parent.chat("ordinary").await.is_ok());
    assert!(child.chat("ordinary").await.is_ok());
}

// Both concurrent leaves may park or remain unstarted; each response consumes only the selected request without restarting its siblings.
#[tokio::test]
async fn concurrent_questions_preserve_siblings_and_single_parent_commit() {
    let pattern = "concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all\n        on_partial_failure: abort";
    let (_, _, runner, store, _, completed, provider) = fixture_pattern(true, pattern).await;
    let mut result = runner.run("objective", None).await.unwrap();
    let run_id = result.run.key.run_id.clone();
    let mut receipts = Vec::new();
    for _ in 0..4 {
        if result.run.status != TaskRunStatus::Paused {
            break;
        }
        let request_id = result.run.pending.as_ref().unwrap().id.clone();
        assert!(!receipts.contains(&request_id));
        receipts.push(request_id.clone());
        result = runner
            .resume(
                &run_id,
                result.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(receipts.len(), 2);
    assert_eq!(completed.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.run.counters.llm_attempts, 4);
    let payload: TaskCheckpointPayload =
        serde_json::from_value(store.load(&run_id).await.unwrap().unwrap().payload).unwrap();
    assert_eq!(payload.children.len(), 2);
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.result.is_some() && child.pending.is_none())
    );
    assert_eq!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::User)
            .count(),
        1
    );
    assert_eq!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::Assistant)
            .count(),
        1
    );
}

// A completed earlier stage stays in the saved cursor and the later stage receives the exact rendered input after resume.
#[tokio::test]
async fn pipeline_questions_resume_stage_cursor_without_replaying_previous_stage() {
    let pattern = "pipeline:\n        stages:\n          - worker\n          - id: peer\n            input: 'Previous: {{ previous_output }}; Original: {{ original_input }}'";
    let (_, _, runner, store, _, completed, provider) = fixture_pattern(true, pattern).await;
    let first = runner.run("objective", None).await.unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let second = runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.unwrap().id,
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    let snapshot = store.load(&second.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
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
    assert_eq!(group.frame.cursor["pipeline"]["next_stage"], 1);
    assert_eq!(
        group.frame.cursor["pipeline"]["prepared_input"],
        "Previous: done; Original: objective"
    );
    let finished = runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: second.run.pending.unwrap().id,
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Completed);
    assert_eq!(completed.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(finished.run.counters.llm_attempts, 4);
}

// Cancelling a concurrent parent closes every parked native exchange without polling another provider or pending tool.
#[tokio::test]
async fn concurrent_cancellation_retires_all_parked_leaves() {
    let pattern = "concurrent:\n        agents: [worker, peer]\n        aggregation:\n          strategy: all";
    let (_, child, runner, store, first, _, provider) = fixture_pattern(false, pattern).await;
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let cancelled = runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&cancelled.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
    assert!(child.chat("ordinary").await.is_ok());
}

// Parent and child remain reserved across the pause, then resume the child batch without rerouting either input.
#[tokio::test]
async fn delegated_approval_resumes_child_and_finalizes_parent_once() {
    let (parent, child, runner, store, first, second, provider) = fixture(false).await;
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert!(parent.chat("foreign").await.is_err());
    assert!(child.chat("foreign").await.is_err());
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert!(payload.children[0].pending.is_some());
    assert!(snapshot.owner_token.is_none());
    let finished = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Completed);
    assert_eq!(finished.run.counters.turns, 1);
    assert_eq!(finished.run.counters.llm_attempts, 2);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&finished.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
    assert_eq!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::User)
            .count(),
        1
    );
    assert_eq!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::Assistant)
            .count(),
        1
    );
    assert!(child.chat("ordinary").await.is_ok());
}

// A root question response binds to the leaf child request, not to another run or an approval receipt.
#[tokio::test]
async fn delegated_question_resumes_the_exact_child_action() {
    let (_, _, runner, _, _, second, provider) = fixture(true).await;
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let pending = paused.run.pending.unwrap();
    assert!(matches!(pending.kind, TaskPendingKind::UserQuestion));
    assert!(
        runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::Approval {
                    request_id: pending.id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved
                }
            )
            .await
            .is_err()
    );
    let finished = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: pending.id,
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(finished.run.status, TaskRunStatus::Completed);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

// Cancellation acknowledges every parked child and root identity before either runtime is released.
#[tokio::test]
async fn delegated_pause_cancellation_never_invokes_pending_child() {
    let (parent, child, runner, store, first, second, provider) = fixture(false).await;
    let paused = runner.run("objective", None).await.unwrap();
    let cancelled = runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&cancelled.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(payload.pending.is_none());
    assert!(
        payload
            .children
            .iter()
            .all(|child| child.pending.is_none() && child.result.is_some())
    );
    assert!(parent.chat("ordinary").await.is_ok());
    assert!(child.chat("ordinary").await.is_ok());
}
