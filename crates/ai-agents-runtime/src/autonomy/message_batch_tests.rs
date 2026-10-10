use super::*;
use crate::{
    Agent, RuntimeAgent,
    spawner::{AgentRegistry, AgentSpawner, SpawnedAgent},
};
use ai_agents_core::{Tool, ToolCall, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Count(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Count {
    fn id(&self) -> &str {
        "count"
    }
    fn name(&self) -> &str {
        "count"
    }
    fn description(&self) -> &str {
        "Count actual effects"
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
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("counted")
    }
}
struct NoApproval;
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for NoApproval {
    async fn request_approval(
        &self,
        _: ai_agents_hitl::ApprovalRequest,
    ) -> ai_agents_hitl::ApprovalResult {
        panic!("task interaction must park")
    }
}
struct Sent(Arc<AtomicUsize>);
#[async_trait]
impl crate::spawner::RegistryHooks for Sent {
    async fn on_message_sent(&self, _: &str, _: &str, _: &str) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
struct ResponseGate {
    enabled: std::sync::atomic::AtomicBool,
    entered: tokio::sync::Semaphore,
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for ResponseGate {
    async fn on_response(&self, _: &ai_agents_core::AgentResponse) {
        if self.enabled.load(Ordering::SeqCst) {
            self.entered.add_permits(1);
            std::future::pending::<()>().await;
        }
    }
}
struct Fixture {
    parent: Arc<RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    children: Vec<Arc<RuntimeAgent>>,
    effects: Vec<Arc<AtomicUsize>>,
    providers: Vec<Arc<super::runner_tests::RecordingProvider>>,
    main: Arc<super::runner_tests::RecordingProvider>,
    sent: Arc<AtomicUsize>,
    tail: Arc<AtomicUsize>,
    response_gate: Arc<ResponseGate>,
}

// Production send/route tools retain real registries and signed batches; construction is isolated from recursive polling.
async fn fixture(question: bool, route: bool) -> Fixture {
    Box::pin(fixture_repeated(question, route, false)).await
}

// Repeated recipients use one actual gate but distinct signed exchanges and incoming operation identities.
async fn fixture_repeated(question: bool, route: bool, repeated: bool) -> Fixture {
    Box::pin(fixture_options(question, route, repeated, false, false)).await
}

// Optional parent approvals and child re-pauses exercise the same cursor rather than separate fixture executors.
async fn fixture_options(
    question: bool,
    route: bool,
    repeated: bool,
    parent_approval: bool,
    re_pause: bool,
) -> Fixture {
    Box::pin(fixture_with_targets(
        question,
        route,
        if repeated { &["a", "a"] } else { &["a", "b"] },
        parent_approval,
        re_pause,
    ))
    .await
}

// Incoming calls can interleave repeated targets with independent pending children without changing declaration order.
async fn fixture_with_targets(
    question: bool,
    route: bool,
    targets: &[&str],
    parent_approval: bool,
    re_pause: bool,
) -> Fixture {
    let response_gate = Arc::new(ResponseGate {
        enabled: Default::default(),
        entered: tokio::sync::Semaphore::new(0),
    });
    let sent = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(AgentRegistry::new().with_hooks(Arc::new(Sent(sent.clone()))));
    let mut effects = Vec::new();
    let mut providers = Vec::new();
    let mut children = Vec::new();
    for id in ["a", "b"] {
        let effect = Arc::new(AtomicUsize::new(0));
        let call = ToolCall {
            id: format!("{id}-pending"),
            name: if question { "ask_user" } else { "count" }.into(),
            arguments: if question {
                json!({"question":"Select", "options":["yes"], "allow_other":false})
            } else {
                json!({})
            },
        };
        let native = ai_agents_core::NativeProviderState::new(
            format!("{id}-exchange"),
            "fixture",
            "native-tools",
            ai_agents_core::NativeProviderTarget::new("https://fixture.invalid", "fixture-model")
                .unwrap(),
            json!({"signature":id}),
            vec![ai_agents_core::NativeCallBinding::new(&call.id, 0).unwrap()],
        )
        .unwrap();
        let marker = ai_agents_core::encode_native_tool_call_markers(
            std::slice::from_ref(&call),
            Some(&native),
        )
        .unwrap();
        let mut repeat_call = call;
        repeat_call.id = format!("{id}-repeat-pending");
        let repeat_native = ai_agents_core::NativeProviderState::new(
            format!("{id}-repeat-exchange"),
            "fixture",
            "native-tools",
            ai_agents_core::NativeProviderTarget::new("https://fixture.invalid", "fixture-model")
                .unwrap(),
            json!({"signature":"repeat"}),
            vec![ai_agents_core::NativeCallBinding::new(&repeat_call.id, 0).unwrap()],
        )
        .unwrap();
        let repeat_marker =
            ai_agents_core::encode_native_tool_call_markers(&[repeat_call], Some(&repeat_native))
                .unwrap();
        let provider = if re_pause {
            super::runner_tests::RecordingProvider::new(&[&marker, &repeat_marker, "child done"])
        } else {
            let mut third: Value = serde_json::from_str(&repeat_marker).unwrap();
            third["id"] = json!(format!("{id}-third-pending"));
            third["_ai_agents_provider_state"]["exchange_id"] =
                json!(format!("{id}-third-exchange"));
            third["_ai_agents_provider_state"]["bindings"][0]["call_id"] = third["id"].clone();
            let third_marker = third.to_string();
            super::runner_tests::RecordingProvider::new(&[
                &marker,
                "child done",
                &repeat_marker,
                "child done",
                &third_marker,
                "child done",
            ])
        };
        let yaml = if question {
            "name: Worker\nsystem_prompt: child\ntools: [ask_user]\n"
        } else {
            "name: Worker\nsystem_prompt: child\ntools: [count]\nhitl:\n  tools:\n    count:\n      require_approval: true\n"
        };
        let yaml = yaml.replace("name: Worker", &format!("name: Worker-{id}"));
        let builder = crate::AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(NoApproval))
            .hooks(response_gate.clone());
        let builder = if question {
            builder.tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
        } else {
            builder.tool(Arc::new(Count(effect.clone())))
        };
        registry
            .register(SpawnedAgent::from_runtime(
                id.into(),
                builder.build().unwrap(),
                crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
            ))
            .await
            .unwrap();
        children.push(registry.get(id).unwrap());
        effects.push(effect);
        providers.push(provider);
    }
    let tail = Arc::new(AtomicUsize::new(0));
    let mut calls: Vec<_> = targets
        .iter()
        .enumerate()
        .map(|(index, id)| ToolCall {
            id: format!("parent-{}", ["a", "b", "c", "d"][index]),
            name: if route {
                "route_to_agent"
            } else {
                "send_agent_message"
            }
            .into(),
            arguments: if route {
                json!({"input":"child work", "candidates":[id], "method":"round_robin"})
            } else {
                json!({"to":id, "message":"child work"})
            },
        })
        .collect();
    calls.push(ToolCall {
        id: "tail".into(),
        name: "count".into(),
        arguments: json!({}),
    });
    let native = ai_agents_core::NativeProviderState::new(
        "parent-batch",
        "fixture",
        "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://fixture.invalid", "fixture-model")
            .unwrap(),
        json!({"signature":"parent"}),
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| ai_agents_core::NativeCallBinding::new(&call.id, index).unwrap())
            .collect(),
    )
    .unwrap();
    let marker = ai_agents_core::encode_native_tool_call_markers(&calls, Some(&native)).unwrap();
    let main = super::runner_tests::RecordingProvider::new(&[&marker, "parent done"]);
    let yaml = if parent_approval {
        "name: Parent\nsystem_prompt: parent\ntools: [send_agent_message, route_to_agent, count]\nhitl:\n  tools:\n    send_agent_message:\n      require_approval: true\n"
    } else {
        "name: Parent\nsystem_prompt: parent\ntools: [send_agent_message, route_to_agent, count]\n"
    };
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register("default", main.clone());
    llms.register(
        "router",
        super::runner_tests::RecordingProvider::new(&["a", "b"]),
    );
    let parent = Arc::new(
        crate::AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_registry(llms.clone())
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(NoApproval))
            .tool(Arc::new(Count(tail.clone())))
            .tool(Arc::new(crate::spawner::SendMessageTool::new(
                registry.clone(),
                "parent",
            )))
            .tool(Arc::new(
                crate::orchestration::tools::RouteToAgentTool::new(
                    registry.clone(),
                    Arc::new(llms),
                ),
            ))
            .build()
            .unwrap()
            .with_spawner_handles(Arc::new(AgentSpawner::new()), registry),
    );
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(parent.info().id, None, "batch.v1".into()).unwrap());
    let mut config =
        super::runner_tests::config(targets.len() as u32 * if re_pause { 3 } else { 2 } + 2);
    config.defaults.max_tool_calls = Some(targets.len() as u32 * if re_pause { 3 } else { 2 } + 1);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        parent.clone(),
        config,
        Default::default(),
        store.clone(),
        "batch.v1".into(),
    )
    .unwrap();
    Fixture {
        parent,
        runner,
        store,
        children,
        effects,
        providers,
        main,
        sent,
        tail,
        response_gate,
    }
}

// Inspect the exact persisted projection, not a UI dump that might conceal missing siblings.
async fn payload(fixture: &Fixture, run: &TaskRun) -> TaskCheckpointPayload {
    serde_json::from_value(
        fixture
            .store
            .load(&run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap()
}

// Sequential responses bind only the coordinating request, never a sibling's raw leaf ID.
async fn answer(fixture: &Fixture, run: &TaskRun, question: bool) -> TaskRunResult {
    let id = run.pending.as_ref().unwrap().id.clone();
    let result = fixture
        .runner
        .resume(
            &run.key.run_id,
            run.revision,
            if question {
                TaskResumeInput::UserAnswer {
                    request_id: id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                }
            } else {
                TaskResumeInput::Approval {
                    request_id: id,
                    result: ai_agents_hitl::ApprovalResult::Approved,
                }
            },
        )
        .await;
    result.unwrap()
}

// Native result markers are encoded messages, not necessarily Role::Tool entries.
fn assert_parent_results(payload: &TaskCheckpointPayload) {
    let history =
        ai_agents_core::inspect_native_history(&payload.runtime.snapshot.memory.messages).unwrap();
    let exchange = history
        .exchanges()
        .iter()
        .find(|exchange| exchange.state().exchange_id() == "parent-batch")
        .unwrap();
    assert_eq!(exchange.result_ids(), &["parent-a", "parent-b", "tail"]);
    assert!(exchange.is_complete());
}

// Multiple children park simultaneously; native history advances only through the contiguous completed prefix.
#[tokio::test]
async fn model_batch_multiple_children_resume_without_replay() {
    for question in [false, true] {
        for route in [false, true] {
            let f = Box::pin(fixture(question, route)).await;
            let paused = f.runner.run("objective", None).await.unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                paused.run.stop_reason
            );
            let initial = payload(&f, &paused.run).await;
            let group: TaskGroupState = serde_json::from_value(
                initial
                    .adapters
                    .iter()
                    .find(|a| a.id == "runtime.group")
                    .unwrap()
                    .state
                    .clone(),
            )
            .unwrap();
            assert_eq!(group.parked.len(), 2);
            assert_eq!(
                initial
                    .reservations
                    .iter()
                    .filter(|r| r.state == TaskEffectState::Suspended)
                    .count(),
                2
            );
            let batch: TaskBatchState = serde_json::from_value(group.frame.cursor).unwrap();
            assert_eq!(batch.appended, 0);
            assert!(batch.results[2].is_some());
            assert_eq!(f.tail.load(Ordering::SeqCst), 1);
            let next = answer(&f, &paused.run, question).await;
            assert_eq!(
                next.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                next.run.stop_reason
            );
            assert!(next.run.revision > paused.run.revision);
            assert_ne!(
                next.run.pending.as_ref().unwrap().id,
                paused.run.pending.as_ref().unwrap().id
            );
            let partial = payload(&f, &next.run).await;
            let group: TaskGroupState = serde_json::from_value(
                partial
                    .adapters
                    .iter()
                    .find(|a| a.id == "runtime.group")
                    .unwrap()
                    .state
                    .clone(),
            )
            .unwrap();
            assert_eq!(group.parked.len(), 1);
            let batch: TaskBatchState = serde_json::from_value(group.frame.cursor).unwrap();
            assert_eq!(batch.appended, 1);
            assert!(batch.results[0].is_some() && batch.results[2].is_some());
            assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
            let done = answer(&f, &next.run, question).await;
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(done.run.counters.tool_attempts, 5);
            assert_eq!(done.run.counters.llm_attempts, 6);
            assert_eq!(f.sent.load(Ordering::SeqCst), if route { 0 } else { 2 });
            assert_eq!(f.tail.load(Ordering::SeqCst), 1);
            for provider in &f.providers {
                assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            }
            for effect in &f.effects {
                assert_eq!(effect.load(Ordering::SeqCst), usize::from(!question));
            }
            let final_payload = payload(&f, &done.run).await;
            assert!(
                ai_agents_core::inspect_native_history(
                    &final_payload.runtime.snapshot.memory.messages
                )
                .unwrap()
                .exchanges()
                .iter()
                .all(|e| e.is_complete())
            );
            assert_parent_results(&final_payload);
        }
    }
}

// Known-safe cancellation closes all child and parent histories without invoking any pending effects.
#[tokio::test]
async fn model_batch_cancel_closes_all_invocations_once() {
    let f = Box::pin(fixture(false, false)).await;
    let paused = f.runner.run("objective", None).await.unwrap();
    let cancelled = f
        .runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    for effect in &f.effects {
        assert_eq!(effect.load(Ordering::SeqCst), 0);
    }
    for child in &f.children {
        assert!(child.chat("after cancellation").await.is_ok());
    }
    let p = payload(&f, &cancelled.run).await;
    assert_eq!(
        p.runtime
            .snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message
                .content
                .contains("message stopped after child suspension"))
            .count(),
        2
    );
    for id in ["parent-a", "parent-b"] {
        assert_eq!(
            p.evidence
                .tool_calls
                .iter()
                .filter(|e| e.record.call_id == id && e.record.cancelled)
                .count(),
            1
        );
    }
    assert_parent_results(&p);
}

// Cancellation after a response preserves the already appended sibling result rather than closing an obsolete batch projection.
#[tokio::test]
async fn model_batch_cancel_after_partial_resume_preserves_prefix() {
    let f = Box::pin(fixture(false, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let next = answer(&f, &first.run, false).await;
    let cancelled = f
        .runner
        .cancel_paused(&next.run.key.run_id, next.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(f.effects[0].load(Ordering::SeqCst), 1);
    assert_eq!(f.effects[1].load(Ordering::SeqCst), 0);
    let p = payload(&f, &cancelled.run).await;
    assert_parent_results(&p);
    assert_eq!(
        p.evidence
            .tool_calls
            .iter()
            .filter(|e| e.record.call_id == "parent-a" && e.record.success)
            .count(),
        1
    );
}

// Leaf identities, stale revisions and invalid answers cannot claim the root or alter sibling state.
#[tokio::test]
async fn model_batch_wrong_response_and_tampered_sibling_reject_before_claim() {
    let f = Box::pin(fixture(true, false)).await;
    let paused = f.runner.run("objective", None).await.unwrap();
    let p = payload(&f, &paused.run).await;
    let mut group: TaskGroupState = serde_json::from_value(
        p.adapters
            .iter()
            .find(|a| a.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    let wrong = group.parked[1].request.id.clone();
    assert!(
        f.runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: wrong,
                    answer: json!({"answered":true,"selected":["yes"]})
                }
            )
            .await
            .is_err()
    );
    let ids: Vec<String> =
        serde_json::from_value(group.frame.definition["messages"].clone()).unwrap();
    group.frames.get_mut(&ids[1]).unwrap().cursor["attempt"] = json!("foreign-attempt");
    assert!(group.validate_checkpoint(&p).is_err());
    assert_eq!(
        f.store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        paused.run.revision
    );
    f.runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
}

// Revocation after one response cleans the remaining sibling without repeating the completed invocation.
#[tokio::test]
async fn model_batch_revocation_after_partial_resume_cleans_remaining_work() {
    let f = Box::pin(fixture(false, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let next = answer(&f, &first.run, false).await;
    f.parent.runtime_control().set_tool_scope(Vec::new());
    let stopped = answer(&f, &next.run, false).await;
    assert_eq!(
        stopped.run.status,
        TaskRunStatus::Failed,
        "{:?}",
        stopped.run.stop_reason
    );
    assert_eq!(f.effects[0].load(Ordering::SeqCst), 1);
    assert_eq!(f.effects[1].load(Ordering::SeqCst), 0);
    let p = payload(&f, &stopped.run).await;
    assert_parent_results(&p);
}

// An independent leaf response cannot launch another deferred call while its target still has a parked incoming operation.
#[tokio::test]
async fn model_batch_interleaved_repeated_target_preserves_pending_ownership() {
    let f = Box::pin(fixture_with_targets(
        false,
        false,
        &["a", "b", "a", "a"],
        false,
        false,
    ))
    .await;
    let mut result = f.runner.run("objective", None).await.unwrap();
    result = answer(&f, &result.run, false).await;
    assert_eq!(result.run.status, TaskRunStatus::Paused);
    assert_eq!(f.providers[0].calls.load(Ordering::SeqCst), 3);
    result = answer(&f, &result.run, false).await;
    assert_eq!(result.run.status, TaskRunStatus::Paused);
    assert_eq!(f.providers[0].calls.load(Ordering::SeqCst), 3);
    result = answer(&f, &result.run, false).await;
    assert_eq!(result.run.status, TaskRunStatus::Paused);
    assert_eq!(f.providers[0].calls.load(Ordering::SeqCst), 5);
    result = answer(&f, &result.run, false).await;
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(result.run.counters.tool_attempts, 9);
    assert_eq!(result.run.counters.llm_attempts, 10);
    assert_eq!(f.effects[0].load(Ordering::SeqCst), 3);
    assert_eq!(f.effects[1].load(Ordering::SeqCst), 1);
    assert_eq!(f.sent.load(Ordering::SeqCst), 4);
    let p = payload(&f, &result.run).await;
    let history =
        ai_agents_core::inspect_native_history(&p.runtime.snapshot.memory.messages).unwrap();
    assert_eq!(
        history.exchanges()[0].result_ids(),
        &["parent-a", "parent-b", "parent-c", "parent-d", "tail"]
    );
}

// Re-pausing the selected child keeps the other sibling parked and never resets the parent admission.
#[tokio::test]
async fn model_batch_child_repause_preserves_other_leaf_and_attempts() {
    let f = Box::pin(fixture_options(false, false, false, false, true)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let next = answer(&f, &first.run, false).await;
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    let p = payload(&f, &next.run).await;
    let group: TaskGroupState = serde_json::from_value(
        p.adapters
            .iter()
            .find(|a| a.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    assert_eq!(group.parked.len(), 2);
    assert_ne!(
        next.run.pending.as_ref().unwrap().id,
        first.run.pending.as_ref().unwrap().id
    );
    let next = answer(&f, &next.run, false).await;
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    let next = answer(&f, &next.run, false).await;
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    let done = answer(&f, &next.run, false).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(done.run.counters.tool_attempts, 7);
    assert_eq!(done.run.counters.llm_attempts, 8);
    assert_eq!(f.sent.load(Ordering::SeqCst), 2);
    for effect in &f.effects {
        assert_eq!(effect.load(Ordering::SeqCst), 2);
    }
}

// Modified parent arguments remain separate from the original native calls through child and local-approval pauses.
#[tokio::test]
async fn model_batch_modified_parent_arguments_and_local_approval_resume() {
    let f = Box::pin(fixture_options(false, false, false, true, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let child_pause = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::Approval {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([("to".into(), json!("b"))]),
                },
            },
        )
        .await
        .unwrap();
    assert_eq!(child_pause.run.status, TaskRunStatus::Paused);
    let local = answer(&f, &child_pause.run, false).await;
    assert_eq!(local.run.status, TaskRunStatus::Paused);
    let child_pause = answer(&f, &local.run, false).await;
    assert_eq!(child_pause.run.status, TaskRunStatus::Paused);
    let done = answer(&f, &child_pause.run, false).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(f.effects[0].load(Ordering::SeqCst), 0);
    assert_eq!(f.effects[1].load(Ordering::SeqCst), 2);
    let p = payload(&f, &done.run).await;
    let record = &p
        .evidence
        .tool_calls
        .iter()
        .find(|e| e.record.call_id == "parent-a")
        .unwrap()
        .record;
    assert_eq!(record.arguments["to"], "a");
    assert_eq!(record.executed_arguments["to"], "b");
    assert_parent_results(&p);
}

// A child rejection still closes uninvoked local approvals after the message coordinator has retired.
#[tokio::test]
async fn model_batch_child_rejection_closes_remaining_local_approval() {
    let f = Box::pin(fixture_options(false, false, false, true, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let child_pause = answer(&f, &first.run, false).await;
    let stopped = f
        .runner
        .resume(
            &first.run.key.run_id,
            child_pause.run.revision,
            TaskResumeInput::Approval {
                request_id: child_pause.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Rejected {
                    reason: Some("stop child".into()),
                },
            },
        )
        .await
        .unwrap();
    assert_eq!(
        stopped.run.status,
        TaskRunStatus::Failed,
        "{:?}",
        stopped.run.stop_reason
    );
    assert_eq!(f.sent.load(Ordering::SeqCst), 1);
    assert_eq!(f.providers[1].calls.load(Ordering::SeqCst), 0);
    assert_parent_results(&payload(&f, &stopped.run).await);
}

// A declared parent retains the multi-message model batch without replaying its input or completed sibling.
#[tokio::test]
async fn model_batch_nested_under_declared_delegate_resumes_all_calls() {
    for reject in [false, true] {
        let f = Box::pin(fixture_options(false, false, false, reject, false)).await;
        let Fixture {
            parent,
            runner,
            main,
            sent,
            effects,
            ..
        } = f;
        drop(runner);
        let middle = Arc::try_unwrap(parent).unwrap_or_else(|_| panic!("fixture retained parent"));
        let registry = Arc::new(AgentRegistry::new());
        registry.register(SpawnedAgent::from_runtime("middle".into(),middle,
        crate::spec::AgentSpec::from_yaml_strict("name: Parent\nsystem_prompt: parent\ntools: [send_agent_message, route_to_agent, count]\n").unwrap())).await.unwrap();
        let outer=Arc::new(crate::AgentBuilder::from_yaml("name: Outer\nsystem_prompt: outer\nstates:\n  initial: active\n  states:\n    active:\n      delegate: middle\n").unwrap()
        .llm(super::runner_tests::RecordingProvider::new(&["unused outer"]))
        .auto_configure_features().unwrap().build().unwrap().with_spawner_handles(Arc::new(AgentSpawner::new()),registry));
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(outer.info().id, None, "nested-batch.v1".into()).unwrap(),
        );
        let mut config = super::runner_tests::config(6);
        config.defaults.hitl = Some(AutonomyHitlConfig {
            on_approval_required: Some(InteractionAction::PauseRun),
            on_user_question: Some(InteractionAction::PauseRun),
        });
        let runner = AutonomyRunner::try_new(
            outer,
            config,
            Default::default(),
            store,
            "nested-batch.v1".into(),
        )
        .unwrap();
        let mut result = runner.run("objective", None).await.unwrap();
        for response_index in 0..2 {
            assert_eq!(
                result.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                result.run.stop_reason
            );
            result = runner
                .resume(
                    &result.run.key.run_id,
                    result.run.revision,
                    TaskResumeInput::Approval {
                        request_id: result.run.pending.as_ref().unwrap().id.clone(),
                        result: if reject && response_index == 1 {
                            ai_agents_hitl::ApprovalResult::Rejected {
                                reason: Some("stop nested child".into()),
                            }
                        } else {
                            ai_agents_hitl::ApprovalResult::Approved
                        },
                    },
                )
                .await
                .unwrap();
        }
        assert_eq!(
            result.run.status,
            if reject {
                TaskRunStatus::Failed
            } else {
                TaskRunStatus::Completed
            },
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(
            main.calls.load(Ordering::SeqCst),
            if reject { 1 } else { 2 }
        );
        assert_eq!(sent.load(Ordering::SeqCst), if reject { 1 } else { 2 });
        for effect in effects {
            assert_eq!(effect.load(Ordering::SeqCst), usize::from(!reject));
        }
    }
}

// Dropping a resumed callback leaves its dispatched effect and other sibling protected rather than acknowledging cleanup.
#[tokio::test]
async fn model_batch_dropped_resume_retains_all_runtime_and_registry_protection() {
    let f = Box::pin(fixture(false, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    f.response_gate.enabled.store(true, Ordering::SeqCst);
    let mut resumed = Box::pin(f.runner.resume(
        &first.run.key.run_id,
        first.run.revision,
        TaskResumeInput::Approval {
            request_id: first.run.pending.as_ref().unwrap().id.clone(),
            result: ai_agents_hitl::ApprovalResult::Approved,
        },
    ));
    tokio::select! {
        outcome=&mut resumed => panic!("response hook did not hold execution: {outcome:?}"),
        entered=f.response_gate.entered.acquire() => entered.unwrap().forget(),
        _=tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("resume did not reach callback"),
    }
    drop(resumed);
    assert!(f.parent.chat("cannot steal parent").await.is_err());
    for child in &f.children {
        assert!(child.chat("cannot steal child").await.is_err());
    }
    for id in ["a", "b"] {
        assert!(
            f.parent
                .spawner_registry()
                .unwrap()
                .remove(id)
                .await
                .is_none()
        );
    }
    let p = payload(&f, &first.run).await;
    assert!(p.reservations.iter().any(|r| matches!(
        r.state,
        TaskEffectState::Dispatched | TaskEffectState::Uncertain
    )));
    assert!(
        p.reservations
            .iter()
            .any(|r| r.state == TaskEffectState::Suspended)
    );
}

// The original parent deadline closes every known-safe leaf without granting another child invocation.
#[tokio::test]
async fn model_batch_expired_original_deadline_closes_all_histories() {
    let f = Box::pin(fixture(false, false)).await;
    f.parent
        .runtime_control()
        .set_tool_security(ai_agents_tools::ToolSecurityConfig {
            default_timeout_ms: 5000,
            ..Default::default()
        });
    let first = f.runner.run("objective", None).await.unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let p = payload(&f, &first.run).await;
    let group: TaskGroupState = serde_json::from_value(
        p.adapters
            .iter()
            .find(|a| a.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    let remaining = (group.frame.expires_at.unwrap() - chrono::Utc::now())
        .to_std()
        .unwrap_or_default();
    tokio::time::sleep(remaining + std::time::Duration::from_millis(10)).await;
    let stopped = answer(&f, &first.run, false).await;
    assert_eq!(
        stopped.run.status,
        TaskRunStatus::Incomplete,
        "{:?}",
        stopped.run.stop_reason
    );
    assert_eq!(
        stopped.run.stop_reason.as_deref(),
        Some("composition_timeout")
    );
    for effect in &f.effects {
        assert_eq!(effect.load(Ordering::SeqCst), 0);
    }
    assert_parent_results(&payload(&f, &stopped.run).await);
}

// A cancellation receipt interrupts an unacknowledged callback and retains the other safely parked sibling.
#[tokio::test]
async fn model_batch_cancel_during_resumed_callback_requires_recovery() {
    let f = Box::pin(fixture(false, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    f.response_gate.enabled.store(true, Ordering::SeqCst);
    let mut resumed = Box::pin(f.runner.resume(
        &first.run.key.run_id,
        first.run.revision,
        TaskResumeInput::Approval {
            request_id: first.run.pending.as_ref().unwrap().id.clone(),
            result: ai_agents_hitl::ApprovalResult::Approved,
        },
    ));
    tokio::select! {
        outcome=&mut resumed => panic!("callback did not block: {outcome:?}"),
        entered=f.response_gate.entered.acquire() => entered.unwrap().forget(),
        _=tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("callback not reached"),
    }
    f.runner
        .request_cancel(&first.run.key.run_id)
        .await
        .unwrap();
    let stopped = resumed.await.unwrap();
    assert_eq!(stopped.run.status, TaskRunStatus::RecoveryRequired);
    assert!(f.parent.chat("cannot steal owner").await.is_err());
    for child in &f.children {
        assert!(child.chat("cannot steal child").await.is_err());
    }
}

struct FailSecondSuspension {
    inner: Arc<ScopedTaskRunStore>,
    fired: std::sync::atomic::AtomicBool,
}
#[async_trait]
impl TaskRunStore for FailSecondSuspension {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(id, revision).await
    }
    async fn mutate(
        &self,
        id: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        if let TaskRunMutation::Checkpoint { payload, .. } = mutation
            && payload["reservations"]
                .as_array()
                .is_some_and(|reservations| {
                    reservations
                        .iter()
                        .filter(|r| r["state"] == "suspended")
                        .count()
                        >= 2
                })
            && !self.fired.swap(true, Ordering::SeqCst)
        {
            return Err(ai_agents_core::AgentError::Persistence(
                "second suspension acknowledgement failed".into(),
            ));
        }
        self.inner.mutate(id, mutation).await
    }
}

// Failure to acknowledge the second park must retain both live bindings, not publish a partially safe pause.
#[tokio::test]
async fn model_batch_failed_second_suspension_retains_protection() {
    let f = Box::pin(fixture(false, false)).await;
    let faulty = Arc::new(FailSecondSuspension {
        inner: f.store.clone(),
        fired: Default::default(),
    });
    let mut config = super::runner_tests::config(6);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        f.parent.clone(),
        config,
        Default::default(),
        faulty.clone(),
        "batch.v1".into(),
    )
    .unwrap();
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::RecoveryRequired);
    assert!(faulty.fired.load(Ordering::SeqCst));
    assert!(f.parent.chat("cannot steal owner").await.is_err());
    for child in &f.children {
        assert!(child.chat("cannot steal child").await.is_err());
    }
    for id in ["a", "b"] {
        assert!(
            f.parent
                .spawner_registry()
                .unwrap()
                .remove(id)
                .await
                .is_none()
        );
    }
}

// A child rejection closes deferred attempts even when no original leaf remains pending.
#[tokio::test]
async fn model_batch_rejection_closes_siblings_and_deferred_history() {
    for repeated in [false, true] {
        let f = Box::pin(fixture_repeated(false, false, repeated)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        let stopped = f
            .runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Rejected {
                        reason: Some("stop".into()),
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(
            stopped.run.status,
            TaskRunStatus::Failed,
            "{:?}",
            stopped.run.stop_reason
        );
        for effect in &f.effects {
            assert_eq!(effect.load(Ordering::SeqCst), 0);
        }
        let p = payload(&f, &stopped.run).await;
        assert_parent_results(&p);
        assert!(
            p.reservations
                .iter()
                .all(|r| r.state == TaskEffectState::Completed)
        );
        assert_eq!(f.sent.load(Ordering::SeqCst), 2);
    }
}

// SQLite retains multi-call cursors and CAS rejects a consumed coordinating response before another invocation.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn model_batch_sqlite_repause_rejects_stale_response() {
    let mut f = Box::pin(fixture(true, false)).await;
    let store = Arc::new(
        ScopedTaskRunStore::new(
            Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
            f.parent.info().id,
            None,
            "batch.v1".into(),
        )
        .unwrap(),
    );
    let mut config = super::runner_tests::config(6);
    config.defaults.max_tool_calls = Some(5);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    f.runner = AutonomyRunner::try_new(
        f.parent.clone(),
        config,
        Default::default(),
        store.clone(),
        "batch.v1".into(),
    )
    .unwrap();
    f.store = store;
    let first = f.runner.run("objective", None).await.unwrap();
    let next = answer(&f, &first.run, true).await;
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    for revision in [first.run.revision, next.run.revision] {
        assert!(
            f.runner
                .resume(
                    &first.run.key.run_id,
                    revision,
                    TaskResumeInput::UserAnswer {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    }
                )
                .await
                .is_err()
        );
    }
    assert_eq!(
        f.store
            .load(&first.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        next.run.revision
    );
    let done = answer(&f, &next.run, true).await;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_parent_results(&payload(&f, &done.run).await);
}

// Serialized omission cannot detach an outstanding child and its consumed reservation from the coordinator.
#[tokio::test]
async fn model_batch_orphaned_pending_invocation_is_invalid() {
    let f = Box::pin(fixture(false, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let p = payload(&f, &first.run).await;
    let mut group: TaskGroupState = serde_json::from_value(
        p.adapters
            .iter()
            .find(|a| a.id == "runtime.group")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    let mut ids: Vec<String> =
        serde_json::from_value(group.frame.definition["messages"].clone()).unwrap();
    ids.pop();
    group.frame.definition["messages"] = json!(ids);
    group.frame.children.pop();
    group
        .frames
        .insert(group.frame.runtime_id.clone(), group.frame.clone());
    group.parked.pop();
    assert!(group.validate_checkpoint(&p).is_err());
    f.runner
        .cancel_paused(&first.run.key.run_id, first.run.revision)
        .await
        .unwrap();
}

// A repeated target's second incoming operation waits without corrupting the first signed child history.
#[tokio::test]
async fn model_batch_repeated_runtime_operations_resume_without_redelivery() {
    for question in [false, true] {
        for route in [false, true] {
            let f = Box::pin(fixture_repeated(question, route, true)).await;
            let first = f.runner.run("objective", None).await.unwrap();
            assert_eq!(
                first.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                first.run.stop_reason
            );
            let p = payload(&f, &first.run).await;
            let group: TaskGroupState = serde_json::from_value(
                p.adapters
                    .iter()
                    .find(|a| a.id == "runtime.group")
                    .unwrap()
                    .state
                    .clone(),
            )
            .unwrap();
            assert_eq!(group.parked.len(), 1);
            assert_eq!(group.frame.children.len(), 2);
            assert_ne!(
                group.frame.children[0].operation,
                group.frame.children[1].operation
            );
            assert_eq!(f.providers[0].calls.load(Ordering::SeqCst), 1);
            let next = answer(&f, &first.run, question).await;
            assert_eq!(
                next.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                next.run.stop_reason
            );
            assert_eq!(f.providers[0].calls.load(Ordering::SeqCst), 3);
            assert_eq!(f.providers[1].calls.load(Ordering::SeqCst), 0);
            let done = answer(&f, &next.run, question).await;
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(f.providers[0].calls.load(Ordering::SeqCst), 4);
            assert_eq!(
                f.effects[0].load(Ordering::SeqCst),
                if question { 0 } else { 2 }
            );
            assert_eq!(f.sent.load(Ordering::SeqCst), if route { 0 } else { 2 });
            assert_eq!(done.run.counters.tool_attempts, 5);
            let p = payload(&f, &done.run).await;
            assert_eq!(p.children.len(), 2);
            assert!(
                p.children
                    .iter()
                    .all(|child| child.result.is_some() && child.pending.is_none())
            );
            assert_parent_results(&p);
        }
    }
}

// Cancelling a deferred same-runtime invocation settles its consumed parent attempt without starting another child.
#[tokio::test]
async fn model_batch_repeated_runtime_cancel_never_starts_deferred_child() {
    let f = Box::pin(fixture_repeated(false, false, true)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let cancelled = f
        .runner
        .cancel_paused(&first.run.key.run_id, first.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(f.providers[0].calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.effects[0].load(Ordering::SeqCst), 0);
    assert_eq!(cancelled.run.counters.tool_attempts, 3);
    let p = payload(&f, &cancelled.run).await;
    assert_eq!(p.children.len(), 1);
    assert_parent_results(&p);
}
