use super::*;
use std::collections::HashMap;

#[tokio::test]
async fn location_skill_modified_receipt_preserves_completed_template_prefix() {
    let f = Box::pin(skill_fixture(false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    let done = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::Approval {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Modified {
                    changes: HashMap::from([("value".into(), json!("modified"))]),
                },
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
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.second.load(Ordering::SeqCst), 1);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.responses.load(Ordering::SeqCst), 1);
    assert!(
        done.run.evidence.tool_calls.iter().any(|capture| capture
            .record
            .executed_arguments
            .get("value")
            == Some(&json!("modified")))
    );
}

#[tokio::test]
async fn location_skill_wrong_kind_or_scope_revocation_cannot_invoke_pending_step() {
    let f = Box::pin(skill_fixture(false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true})
                }
            )
            .await
            .is_err()
    );
    assert_eq!(
        f.store
            .load(&first.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        first.run.revision
    );
    f.agent.runtime_control().set_tool_scope(Vec::new());
    let done = approve(&f, &first.run).await;
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.second.load(Ordering::SeqCst), 0);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.responses.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn location_nested_skill_entry_action_keeps_exit_commit_and_finalization_once() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let responses = Arc::new(AtomicUsize::new(0));
    let main = super::super::runner_tests::RecordingProvider::new(&["done", "entry done"]);
    let router = super::super::runner_tests::RecordingProvider::new(&["none"]);
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register("default", main.clone());
    llms.register("router", router);
    let yaml = "name: NestedAction\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ntools: [first, second]\nhitl:\n  tools:\n    second:\n      require_approval: true\nskills:\n  - id: entry_script\n    description: Entry action only\n    trigger: Not ordinary routing\n    steps:\n      - tool: second\n        args: {value: nested}\n      - prompt: 'Finish entry {{ steps[0].result }}'\nstates:\n  initial: active\n  states:\n    active:\n      on_exit:\n        - tool: first\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready:\n      regenerate_on_enter: false\n      on_enter:\n        - skill: entry_script\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_registry(llms)
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(NoApproval))
            .hooks(Arc::new(Responses(responses.clone())))
            .tool(Arc::new(Probe {
                id: "first",
                calls: first.clone(),
                fail: false,
            }))
            .tool(Arc::new(Probe {
                id: "second",
                calls: second.clone(),
                fail: false,
            }))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "locations.v1".into()).unwrap(),
    );
    let runner = runner(agent.clone(), store);
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 0);
    assert_eq!(agent.current_state().as_deref(), Some("ready"));
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
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
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(responses.load(Ordering::SeqCst), 1);
    assert_eq!(agent.state_history().len(), 1);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn location_fallback_question_resumes_actual_hop_without_repeating_failed_original() {
    let calls = Arc::new(AtomicUsize::new(0));
    let question = json!({"question":"Select","options":["yes"],"allow_other":false});
    let marker = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "failed".into(),
            name: "first".into(),
            arguments: question,
        }],
        None,
    )
    .unwrap();
    let main = super::super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
    let yaml = "name: FallbackQuestion\nsystem_prompt: test\ntools: [first, ask_user]\nerror_recovery:\n  tools:\n    first:\n      max_retries: 0\n      on_failure:\n        action: fallback\n        fallback_tool: ask_user\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(main.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(Probe {
                id: "first",
                calls: calls.clone(),
                fail: true,
            }))
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "locations.v1".into()).unwrap(),
    );
    let runner = runner(agent, store);
    let first = runner.run("objective", None).await.unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let done = runner
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
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(done.run.counters.tool_attempts, 2);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
}
