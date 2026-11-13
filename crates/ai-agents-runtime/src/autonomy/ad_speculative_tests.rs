use super::*;
use crate::autonomy::continuation_test_support as support;

// A reviewed speculative route retains the already processed input and settled alternatives on both approval and rejection.
#[tokio::test]
async fn ad_speculative_review_memory_sqlite_keeps_selection_and_processed_input() {
    for &sqlite in support::backends() {
        for approved in [false, true] {
            let yaml = format!("{}runtime:\n  optimization:\n    enabled: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\n    speculative_state_transitions: true\nprocess:\n  input:\n    - type: transform\n      config:\n        llm: router\n        prompt: Prepare input\n", STATE.replace("          guard: 'true'", "          when: route to ready\n          timing: parallel\n          requires_response: false").replace("hitl:\n", "hitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n"));
            let mut f = fixture(
                &yaml,
                &["settled old draft done", "ready done"],
                &["processed once", "1"],
            );
            f.store = support::store(f.agent.info().id, "interaction.v1", sqlite).await;
            let mut config = super::super::runner_tests::config(16);
            config.defaults.hitl = Some(AutonomyHitlConfig {
                on_approval_required: Some(InteractionAction::PauseRun),
                on_user_question: Some(InteractionAction::PauseRun),
            });
            f.runner = AutonomyRunner::try_new(
                f.agent.clone(),
                config,
                Default::default(),
                f.store.clone(),
                "interaction.v1".into(),
            )
            .unwrap();
            let paused = f.runner.run("objective", None).await.unwrap();
            assert_eq!(paused.run.status, TaskRunStatus::Paused);
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
            assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
            assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
            let result = respond(
                &f,
                &paused.run,
                if approved {
                    ApprovalResult::Approved
                } else {
                    ApprovalResult::Rejected {
                        reason: Some("keep current route".into()),
                    }
                },
            )
            .await;
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{sqlite}/{approved}: {:?}",
                result.run.stop_reason
            );
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
            assert_eq!(
                f.main.calls.load(Ordering::SeqCst),
                if approved { 2 } else { 1 }
            );
            assert_eq!(
                f.agent.current_state().as_deref(),
                Some(if approved { "ready" } else { "active" })
            );
            assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
        }
    }
}

// Transition review and a later exit-action question are separate pauses; neither replays selection, completed actions, or input processing.
#[tokio::test]
async fn ad_speculative_review_then_exit_question_preserves_exact_action_cursor() {
    for &sqlite in support::backends() {
        let yaml = format!("{}runtime:\n  optimization:\n    enabled: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\n    speculative_state_transitions: true\n", STATE
            .replace("tools: [probe]", "tools: [probe,ask_user]")
            .replace("hitl:\n", "hitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n")
            .replace("    active:\n", "    active:\n      on_exit:\n        - tool: probe\n        - tool: ask_user\n          args: {question: Choice, options: [yes], allow_other: false}\n")
            .replace("          guard: 'true'", "          when: route to ready\n          timing: parallel\n          requires_response: false"));
        let mut f = fixture(&yaml, &["discarded draft", "done"], &["1"]);
        f.store = support::store(f.agent.info().id, "interaction.v1", sqlite).await;
        let mut config = super::super::runner_tests::config(16);
        config.defaults.hitl = Some(AutonomyHitlConfig {
            on_approval_required: Some(InteractionAction::PauseRun),
            on_user_question: Some(InteractionAction::PauseRun),
        });
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            config,
            Default::default(),
            f.store.clone(),
            "interaction.v1".into(),
        )
        .unwrap();
        let reviewed = f.runner.run("objective", None).await.unwrap();
        assert_eq!(reviewed.run.status, TaskRunStatus::Paused);
        let question = respond(&f, &reviewed.run, ApprovalResult::Approved).await;
        assert_eq!(question.run.status, TaskRunStatus::Paused);
        assert_eq!(f.effects.load(Ordering::SeqCst), 1);
        assert_eq!(f.agent.current_state().as_deref(), Some("active"));
        let result = f
            .runner
            .resume(
                &question.run.key.run_id,
                question.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: question.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(f.effects.load(Ordering::SeqCst), 1);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
        assert_eq!(f.agent.current_state().as_deref(), Some("ready"));
    }
}
