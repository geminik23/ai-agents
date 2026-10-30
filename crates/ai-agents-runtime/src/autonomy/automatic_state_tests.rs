//! State scope endings retain transition action custody, required decisions and exact-limit completion.

use super::*;
use crate::autonomy::continuation_test_support as support;

// Exiting or already committed state changes cannot waive a pending action question or replay its completed effect prefix.
#[tokio::test]
async fn auto_state_exit_and_entry_questions_preserve_commit_phase_and_one_finalization() {
    for &sqlite in support::backends() {
        for entering in [false, true] {
            let actions = "        - tool: effect\n          args: {phase: transition}\n        - tool: ask_user\n          args: {question: 'Continue the transition?', options: [yes]}\n";
            let mut yaml = STATE.replace("        completion: {response_contains: done}", "        completion: {state: resolved}\n        hitl: {on_user_question: pause_run, on_approval_required: pause_run}");
            if entering {
                yaml += &format!(
                    "      transitions: [{{to: resolved, guard: 'true'}}]\n    resolved:\n      on_enter:\n{actions}"
                );
            } else {
                yaml += &format!(
                    "      on_exit:\n{actions}      transitions: [{{to: resolved, guard: 'true'}}]\n    resolved: {{}}\n"
                );
            }
            yaml += "tools: [effect, ask_user]\n";
            let store = support::store("auto".into(), "automatic.v1", sqlite).await;
            let effects = Arc::new(AtomicUsize::new(0));
            let responses = Arc::new(AtomicUsize::new(0));
            let main = RecordingProvider::new(&["candidate"]);
            let runtime = AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(main.clone())
                .llm_alias("router", RecordingProvider::new(&[]))
                .tool(Arc::new(Effect(effects.clone())))
                .tool(Arc::new(ai_agents_tools::AskUserTool::new(
                    Default::default(),
                )))
                .hooks(Arc::new(Responses(responses.clone())))
                .automatic_task_options(AutomaticTaskOptions {
                    store: Some(store),
                    config_identity: Some("automatic.v1".into()),
                    ..Default::default()
                })
                .build()
                .unwrap();
            runtime.chat("resolve support ticket").await.unwrap();
            let paused = runtime.automatic_task_result().unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                paused.run.stop_reason
            );
            assert_eq!(
                runtime.current_state().as_deref(),
                Some(if entering { "resolved" } else { "work" })
            );
            assert_eq!(effects.load(Ordering::SeqCst), 1);
            assert_eq!(responses.load(Ordering::SeqCst), 0);
            let result = runtime
                .automatic_task_operation(TaskStreamOperation::Resume {
                    run_id: paused.run.key.run_id,
                    revision: paused.run.revision,
                    input: TaskResumeInput::UserAnswer {
                        request_id: paused.run.pending.unwrap().id,
                        answer: json!({"answered":true,"selected":["yes"]}),
                    },
                })
                .await
                .unwrap();
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                result.run.stop_reason
            );
            assert_eq!(runtime.current_state().as_deref(), Some("resolved"));
            assert_eq!(effects.load(Ordering::SeqCst), 1);
            assert_eq!(main.calls.load(Ordering::SeqCst), 1);
            assert_eq!(responses.load(Ordering::SeqCst), 1);
            assert_eq!(result.run.counters.tool_attempts, 2);
            assert_eq!(runtime.state_history().len(), 1);
        }
    }
}

// The automatic event adapter owns the actual controller future, not a detached worker hidden behind progress events.
#[tokio::test]
async fn auto_event_stream_drop_after_admission_retains_uncertain_owner() {
    let main = RecordingProvider::gated();
    let runtime = agent(STATE, main.clone(), RecordingProvider::new(&[]));
    let mut stream = runtime.chat_stream_events("work").await.unwrap();
    loop {
        match stream.next().await {
            Some(AgentStreamEvent::Task(event)) if event.kind == TaskRunEventKind::TaskStarted => {
                break;
            }
            Some(AgentStreamEvent::Chunk(crate::StreamChunk::Error { message })) => {
                panic!("{message}")
            }
            Some(_) => {}
            None => panic!("admitted task produced no start event"),
        }
    }
    drop(stream);
    assert!(runtime.chat("unrelated input").await.is_err());
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
}

// Required stages remain mandatory even when the initial answer passes, and the final admitted slot may establish completion.
#[tokio::test]
async fn auto_last_slot_completion_and_required_lifecycle_are_not_early_finalized() {
    for lifecycle in [false, true] {
        let yaml = if lifecycle {
            STATE.replace(
                "        max_llm_calls: 4",
                "        max_llm_calls: 2\n        lifecycle: [{id: work}, {id: review}]",
            )
        } else {
            STATE.replace("        max_llm_calls: 4", "        max_llm_calls: 1")
        };
        let main = RecordingProvider::new(&["done", "done"]);
        let runtime = agent(&yaml, main.clone(), RecordingProvider::new(&[]));
        runtime
            .chat("analysis without tools or todos")
            .await
            .unwrap();
        let result = runtime.automatic_task_result().unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(
            result.run.counters.llm_attempts,
            if lifecycle { 2 } else { 1 }
        );
        assert_eq!(
            main.calls.load(Ordering::SeqCst),
            if lifecycle { 2 } else { 1 }
        );
    }
}
