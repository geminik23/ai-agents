use super::super::continuation_test_support::{self as support, ReceiptStore};
use super::*;
use ai_agents_core::Result;
use ai_agents_hitl::ApprovalResult;

fn configured_runner(agent: Arc<RuntimeAgent>, store: Arc<dyn TaskRunStore>) -> AutonomyRunner {
    runner_with_timeout(agent, store, false)
}
fn runner_with_timeout(
    agent: Arc<RuntimeAgent>,
    store: Arc<dyn TaskRunStore>,
    timeout: bool,
) -> AutonomyRunner {
    let mut config = super::super::runner_tests::config(20);
    if timeout {
        config.defaults.max_active_time_seconds = Some(1);
    }
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store,
        "locations.v1".into(),
    )
    .unwrap()
}
async fn modified(f: &Fixture, run: &TaskRun, value: &str) -> Result<TaskRunResult> {
    f.runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::Approval {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([("value".into(), json!(value))]),
                },
            },
        )
        .await
}
async fn saved(f: &Fixture, run: &TaskRun) -> TaskCheckpointPayload {
    serde_json::from_value(
        f.store
            .load(&run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap()
}

#[tokio::test]
async fn skill_acceptance_repeated_modified_steps_then_controller_do_not_replay_initial_segment() {
    for sqlite in support::backends() {
        let mut f = Box::pin(skill_fixture_options(false, true, false)).await;
        f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
        f.runner = configured_runner(f.agent.clone(), f.store.clone());
        let first = f.runner.run("objective", None).await.unwrap();
        let second = modified(&f, &first.run, "first modification")
            .await
            .unwrap();
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        assert_ne!(
            first.run.pending.as_ref().unwrap().id,
            second.run.pending.as_ref().unwrap().id
        );
        assert!(modified(&f, &first.run, "stale").await.is_err());
        assert_eq!(f.second.load(Ordering::SeqCst), 1);
        let done = modified(&f, &second.run, "second modification")
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 2);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 3);
        assert_eq!(f.responses.load(Ordering::SeqCst), 2);
        assert_eq!(done.run.counters.turns, 2);
        assert_eq!(done.run.counters.tool_attempts, 3);
        let rendered = &f.main.captured_messages()[1];
        let text = rendered
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("first modification")
                && text.contains("second modification")
                && text.contains("prefix")
        );
        assert!(!text.contains("{{ steps"));
        let p = saved(&f, &done.run).await;
        let args: Vec<_> = p
            .evidence
            .tool_calls
            .iter()
            .filter(|r| r.record.canonical_id == "second")
            .map(|r| r.record.executed_arguments["value"].clone())
            .collect();
        assert_eq!(
            args,
            vec![json!("first modification"), json!("second modification")]
        );
        assert_eq!(
            p.runtime
                .snapshot
                .memory
                .messages
                .iter()
                .filter(|m| m.role == ai_agents_core::Role::User)
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn skill_acceptance_repeated_questions_keep_distinct_ids_and_completed_prefix() {
    let f = Box::pin(skill_fixture_options(true, true, false)).await;
    let mut current = f.runner.run("objective", None).await.unwrap();
    let first_id = current.run.pending.as_ref().unwrap().id.clone();
    for _ in 0..2 {
        current = f
            .runner
            .resume(
                &current.run.key.run_id,
                current.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: current.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
        if current.run.status == TaskRunStatus::Paused {
            assert_ne!(current.run.pending.as_ref().unwrap().id, first_id);
        }
    }
    assert_eq!(current.run.status, TaskRunStatus::Completed);
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 3);
    assert_eq!(current.run.counters.tool_attempts, 3);
}

#[tokio::test]
async fn skill_acceptance_original_approval_deadline_stops_without_later_prompt() {
    let f = Box::pin(skill_fixture_options(false, false, true)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let response = |result| TaskResumeInput::Approval {
        request_id: first.run.pending.as_ref().unwrap().id.clone(),
        result,
    };
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                response(ApprovalResult::Timeout)
            )
            .await
            .is_err()
    );
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                response(ApprovalResult::Approved)
            )
            .await
            .is_err()
    );
    let done = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            response(ApprovalResult::Timeout),
        )
        .await
        .unwrap();
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.second.load(Ordering::SeqCst), 0);
    assert_eq!(f.responses.load(Ordering::SeqCst), 1);
    assert!(f.agent.set_context("released", json!(true)).is_ok());
}

#[tokio::test]
async fn skill_acceptance_repause_and_terminal_receipt_failures_keep_owner_and_no_replay() {
    for sqlite in support::backends() {
        for terminal in [false, true] {
            for mode in ["positive", "reject", "lost", "drop"] {
                let mut f = Box::pin(skill_fixture_options(false, !terminal, false)).await;
                f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
                let fault = ReceiptStore::new(
                    f.store.clone(),
                    mode,
                    if terminal {
                        TaskRunStatus::Completed
                    } else {
                        TaskRunStatus::Paused
                    },
                );
                f.runner = configured_runner(f.agent.clone(), fault.clone());
                let first = f.runner.run("objective", None).await.unwrap();
                fault.armed.store(true, Ordering::SeqCst);
                let mut resume = Box::pin(modified(&f, &first.run, "approved"));
                if mode == "positive" {
                    let mut done = resume.await.unwrap();
                    fault.assert_final_write(&first.run.key.run_id).await;
                    if !terminal {
                        done = modified(&f, &done.run, "next").await.unwrap();
                    }
                    assert_eq!(done.run.status, TaskRunStatus::Completed);
                    assert!(f.agent.set_context("released", json!(true)).is_ok());
                } else {
                    if mode == "drop" {
                        tokio::select! {
                            result = &mut resume => panic!("receipt did not block: {result:?}"),
                            permit = fault.reached.acquire() => permit.unwrap().forget(),
                            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("receipt not reached"),
                        }
                        drop(resume);
                    } else {
                        assert!(resume.await.is_err());
                    }
                    fault.assert_final_write(&first.run.key.run_id).await;
                    let before = f.second.load(Ordering::SeqCst);
                    assert!(modified(&f, &first.run, "replay").await.is_err());
                    assert_eq!(f.second.load(Ordering::SeqCst), before);
                    assert!(f.agent.set_context("blocked", json!(true)).is_err());
                    let snapshot = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
                    assert!(
                        f.runner
                            .resume(
                                &first.run.key.run_id,
                                snapshot.revision,
                                TaskResumeInput::Approval {
                                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                                    result: ApprovalResult::Approved
                                }
                            )
                            .await
                            .is_err()
                    );
                    assert_eq!(f.first.load(Ordering::SeqCst), 1);
                    assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(f.second.load(Ordering::SeqCst), 1);
                    assert_eq!(
                        f.main.calls.load(Ordering::SeqCst),
                        if terminal { 2 } else { 1 }
                    );
                    assert_eq!(f.responses.load(Ordering::SeqCst), usize::from(terminal));
                }
            }
        }
    }
}

#[tokio::test]
async fn skill_acceptance_failure_or_drop_in_final_prompt_cannot_replay_settled_steps() {
    for mode in ["error", "drop", "timeout"] {
        let first = Arc::new(AtomicUsize::new(0));
        let finalized = Arc::new(AtomicUsize::new(0));
        let main = support::FaultProvider::new(
            &["prefix", "unused"],
            1,
            if mode == "timeout" { "drop" } else { mode },
        );
        let yaml = "name: SkillFailure\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ntools: [first]\nhitl:\n  tools:\n    first:\n      require_approval: true\nskills:\n  - id: script\n    description: work\n    trigger: work\n    steps:\n      - prompt: prefix\n      - tool: first\n        args: {}\n      - prompt: finish\n";
        let mut llms = ai_agents_llm::LLMRegistry::new();
        llms.register("default", main.clone());
        llms.register(
            "router",
            super::super::runner_tests::RecordingProvider::new(&["script"]),
        );
        let agent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm_registry(llms)
                .auto_configure_features()
                .unwrap()
                .approval_handler(Arc::new(NoApproval))
                .hooks(Arc::new(Responses(finalized.clone())))
                .tool(Arc::new(Probe {
                    id: "first",
                    calls: first.clone(),
                    fail: false,
                }))
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "locations.v1", false).await;
        let runner = runner_with_timeout(agent.clone(), store.clone(), mode == "timeout");
        let paused = runner.run("objective", None).await.unwrap();
        let mut resume = Box::pin(runner.resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Approved,
            },
        ));
        if mode == "drop" {
            tokio::select! {
                result = &mut resume => panic!("provider not held: {result:?}"),
                permit = main.reached.acquire() => permit.unwrap().forget(),
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("provider not reached"),
            }
            drop(resume);
        } else {
            assert!(
                tokio::time::timeout(std::time::Duration::from_secs(5), resume)
                    .await
                    .unwrap()
                    .is_err()
            );
        }
        let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        let p: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        if mode != "error" {
            assert!(p.reservations.iter().any(|r| matches!(
                r.state,
                TaskEffectState::Dispatched | TaskEffectState::Uncertain
            )));
        }
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(main.calls.load(Ordering::SeqCst), 2);
        assert_eq!(finalized.load(Ordering::SeqCst), 0);
        assert!(agent.set_context("blocked", json!(true)).is_err());
        assert!(
            runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Approval {
                        request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                        result: ApprovalResult::Approved
                    }
                )
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn skill_acceptance_signed_child_repause_preserves_script_cursor_and_history() {
    use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
    for cancel in [false, true] {
        let registry = Arc::new(AgentRegistry::new());
        let questions: Vec<_> = (0..2)
            .map(|index| {
                let call = ToolCall {
                    id: format!("question-{index}"),
                    name: "ask_user".into(),
                    arguments: json!({"question":"Select","options":["yes"],"allow_other":false}),
                };
                let state = ai_agents_core::NativeProviderState::new(
                    format!("signed-{index}"),
                    "fixture",
                    "native-tools",
                    ai_agents_core::NativeProviderTarget::new(
                        "https://fixture.invalid",
                        "fixture-model",
                    )
                    .unwrap(),
                    json!({"signature":index}),
                    vec![ai_agents_core::NativeCallBinding::new(&call.id, 0).unwrap()],
                )
                .unwrap();
                ai_agents_core::encode_native_tool_call_markers(&[call], Some(&state)).unwrap()
            })
            .collect();
        let child_main = super::super::runner_tests::RecordingProvider::new(&[
            &questions[0],
            &questions[1],
            "child done",
        ]);
        let child_yaml = "name: Child\nsystem_prompt: test\ntools: [ask_user]\n";
        let child = AgentBuilder::from_yaml(child_yaml)
            .unwrap()
            .llm(child_main.clone())
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
                child,
                crate::spec::AgentSpec::from_yaml_strict(child_yaml).unwrap(),
            ))
            .await
            .unwrap();
        let main =
            super::super::runner_tests::RecordingProvider::new(&["prefix", "working", "done"]);
        let router = super::super::runner_tests::RecordingProvider::new(&["script"]);
        let finalized = Arc::new(AtomicUsize::new(0));
        let mut llms = ai_agents_llm::LLMRegistry::new();
        llms.register("default", main.clone());
        llms.register("router", router.clone());
        let yaml = "name: ScriptParent\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ntools: [send_agent_message]\nskills:\n  - id: script\n    description: work\n    trigger: work\n    steps:\n      - prompt: prefix\n      - tool: send_agent_message\n        args: {to: worker, message: '{{ steps[0].result }}'}\n      - prompt: '{{ steps[1].result }}'\n";
        let parent = Arc::new(
            AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm_registry(llms)
                .auto_configure_features()
                .unwrap()
                .hooks(Arc::new(Responses(finalized.clone())))
                .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
                    registry.clone(),
                    "parent",
                )))
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone()),
        );
        let store = support::store(parent.info().id, "locations.v1", false).await;
        let runner = configured_runner(parent.clone(), store.clone());
        let first = runner.run("objective", None).await.unwrap();
        assert_eq!(
            first.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            first.run.stop_reason
        );
        assert!(registry.remove("worker").await.is_none());
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
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        let done = if cancel {
            runner
                .cancel_paused(&second.run.key.run_id, second.run.revision)
                .await
                .unwrap()
        } else {
            runner
                .resume(
                    &second.run.key.run_id,
                    second.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: second.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                )
                .await
                .unwrap()
        };
        assert_eq!(
            done.run.status,
            if cancel {
                TaskRunStatus::Cancelled
            } else {
                TaskRunStatus::Completed
            },
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(
            main.calls.load(Ordering::SeqCst),
            if cancel { 1 } else { 3 }
        );
        assert_eq!(router.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            child_main.calls.load(Ordering::SeqCst),
            if cancel { 2 } else { 3 }
        );
        assert_eq!(finalized.load(Ordering::SeqCst), if cancel { 0 } else { 2 });
        let p: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&done.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(p.children.len(), 1);
        for child in &p.children {
            let inspection =
                ai_agents_core::inspect_native_history(&child.runtime.snapshot.memory.messages)
                    .unwrap();
            assert_eq!(inspection.exchanges().len(), 2);
            for (index, exchange) in inspection.exchanges().iter().enumerate() {
                assert!(exchange.is_complete());
                assert_eq!(exchange.state().exchange_id(), format!("signed-{index}"));
                assert_eq!(
                    exchange.state().model_content(),
                    &json!({"signature":index})
                );
                assert_eq!(exchange.result_ids(), &[format!("question-{index}")]);
                assert_eq!(exchange.state().bindings().len(), 1);
            }
        }
        assert!(registry.remove("worker").await.is_some());
        assert!(parent.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn skill_acceptance_question_timeout_has_no_extra_prompt_or_tool_attempt() {
    for sqlite in support::backends() {
        let mut f = Box::pin(skill_fixture_options(true, false, true)).await;
        f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
        f.runner = configured_runner(f.agent.clone(), f.store.clone());
        let first = f.runner.run("objective", None).await.unwrap();
        let timeout = || TaskResumeInput::QuestionTimeout {
            request_id: first.run.pending.as_ref().unwrap().id.clone(),
        };
        assert!(
            f.runner
                .resume(&first.run.key.run_id, first.run.revision, timeout())
                .await
                .is_err()
        );
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let done = f
            .runner
            .resume(&first.run.key.run_id, first.run.revision, timeout())
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Incomplete);
        assert_eq!(done.run.stop_reason.as_deref(), Some("question_timeout"));
        assert_eq!(
            done.run.counters.tool_attempts,
            first.run.counters.tool_attempts
        );
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.responses.load(Ordering::SeqCst), 1);
        assert!(f.agent.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn skill_acceptance_initial_pause_receipt_failures_protect_completed_prefix() {
    for sqlite in support::backends() {
        for mode in ["reject", "lost", "drop"] {
            let mut f = Box::pin(skill_fixture(false)).await;
            f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
            let fault = ReceiptStore::new(f.store.clone(), mode, TaskRunStatus::Paused);
            fault.armed.store(true, Ordering::SeqCst);
            f.runner = configured_runner(f.agent.clone(), fault.clone());
            let mut run = Box::pin(f.runner.run("objective", None));
            if mode == "drop" {
                tokio::select! {
                    result = &mut run => panic!("receipt not held: {result:?}"),
                    permit = fault.reached.acquire() => permit.unwrap().forget(),
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("receipt not reached"),
                }
                drop(run);
            } else {
                assert!(run.await.is_err());
            }
            assert_eq!(f.first.load(Ordering::SeqCst), 1);
            fault.assert_hit();
            let snapshot = f.store.list().await.unwrap().pop().unwrap();
            fault.assert_final_write(&snapshot.key.run_id).await;
            assert_eq!(f.second.load(Ordering::SeqCst), 0);
            assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
            assert_eq!(f.responses.load(Ordering::SeqCst), 0);
            assert!(f.agent.set_context("blocked", json!(true)).is_err());
            assert!(f.runner.run("replay", None).await.is_err());
        }
    }
}

#[tokio::test]
async fn skill_acceptance_drop_during_response_hook_does_not_finalize_or_release_twice() {
    let mut f = Box::pin(skill_fixture(false)).await;
    drop(f.runner);
    let gate = support::ResponseGate::new();
    let runtime = Arc::try_unwrap(f.agent).unwrap_or_else(|_| panic!("fixture retained runtime"));
    f.agent = Arc::new(runtime.with_hooks(gate.clone()));
    f.runner = configured_runner(f.agent.clone(), f.store.clone());
    let first = f.runner.run("objective", None).await.unwrap();
    let mut resume = Box::pin(modified(&f, &first.run, "approved"));
    tokio::select! {
        result = &mut resume => panic!("hook not held: {result:?}"),
        permit = gate.reached.acquire() => permit.unwrap().forget(),
        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("hook not reached"),
    }
    drop(resume);
    assert_eq!(f.second.load(Ordering::SeqCst), 1);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
    assert!(modified(&f, &first.run, "replay").await.is_err());
    assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
    assert!(f.agent.set_context("blocked", json!(true)).is_err());
}

#[tokio::test]
async fn skill_acceptance_consumed_step_and_final_prompt_settlement_failures_do_not_replay() {
    for sqlite in support::backends() {
        for consume in [false, true] {
            for mode in ["positive", "reject", "lost", "drop"] {
                let mut f = Box::pin(skill_fixture(false)).await;
                f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
                let fault = ReceiptStore::checkpoint(f.store.clone(), mode, json!({"completed":2}));
                f.runner = configured_runner(f.agent.clone(), fault.clone());
                let first = f.runner.run("objective", None).await.unwrap();
                if consume {
                    *fault.checkpoint.lock() =
                        Some(json!({"consume": first.run.pending.as_ref().unwrap().id}));
                }
                fault.armed.store(true, Ordering::SeqCst);
                let mut resume = Box::pin(modified(&f, &first.run, "approved"));
                if mode == "positive" {
                    let done = resume.await.unwrap();
                    fault.assert_hit();
                    assert_eq!(done.run.status, TaskRunStatus::Completed);
                    assert!(f.agent.set_context("released", json!(true)).is_ok());
                } else {
                    if mode == "drop" {
                        tokio::select! {
                            result = &mut resume => panic!("checkpoint not held: {result:?}"),
                            permit = fault.reached.acquire() => permit.unwrap().forget(),
                            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("checkpoint not reached"),
                        }
                        drop(resume);
                    } else {
                        assert!(resume.await.is_err());
                    }
                    fault.assert_hit();
                    let p = saved(&f, &first.run).await;
                    assert_eq!(
                        p.consumed_request_ids
                            .contains(&first.run.pending.as_ref().unwrap().id),
                        !consume || mode != "reject"
                    );
                    assert_eq!(f.second.load(Ordering::SeqCst), 1);
                    assert_eq!(
                        f.main.calls.load(Ordering::SeqCst),
                        if consume { 1 } else { 2 }
                    );
                    assert_eq!(f.responses.load(Ordering::SeqCst), 0);
                    assert!(f.agent.set_context("blocked", json!(true)).is_err());
                    assert!(modified(&f, &first.run, "replay").await.is_err());
                    assert_eq!(f.second.load(Ordering::SeqCst), 1);
                }
            }
        }
    }
}
