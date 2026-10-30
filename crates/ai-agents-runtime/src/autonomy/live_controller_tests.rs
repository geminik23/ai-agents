use super::super::continuation_test_support as support;
use super::*;
use ai_agents_hitl::ApprovalResult;

#[path = "live_controller_admission_tests.rs"]
mod admission_tests;
#[path = "live_controller_edge_tests.rs"]
mod edge_tests;

// The runner always owns the selected profile; no fixture enables automatic task entry on the agent builder.
fn live_runner(
    agent: Arc<RuntimeAgent>,
    store: Arc<dyn TaskRunStore>,
    config: AutonomyConfig,
    identity: &str,
) -> AutonomyRunner {
    AutonomyRunner::try_new(agent, config, Default::default(), store, identity.into()).unwrap()
}

// Host-visible controller answers are text data, not tool question objects or host-control commands.
fn answer(run: &TaskRun) -> TaskResumeInput {
    TaskResumeInput::UserAnswer {
        request_id: run.pending.as_ref().unwrap().id.clone(),
        answer: json!("Use the available evidence and continue"),
    }
}

#[tokio::test]
async fn live_skill_explicit_scope_survives_repause_and_hands_off_without_routing_or_replay() {
    for sqlite in support::backends() {
        let mut f = Box::pin(skill_fixture_options(false, true, false)).await;
        f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
        f.runner = runner(f.agent.clone(), f.store.clone());
        let first = f
            .runner
            .run_skill("report objective", "script", None)
            .await
            .unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 0);
        assert!(
            f.runner
                .modify_objective_paused(
                    &first.run.key.run_id,
                    first.run.revision,
                    "new",
                    "host revision"
                )
                .await
                .is_err()
        );
        let second = approve(&f, &first.run).await;
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        let done = approve(&f, &second.run).await;
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 2);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 3);
        assert_eq!(done.run.counters.turns, 2);
        let p: TaskCheckpointPayload = serde_json::from_value(
            f.store
                .load(&done.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert!(
            p.adapters
                .iter()
                .find(|a| a.id == "runtime.controller")
                .unwrap()
                .state["lifecycle"]["skill_segment_completed"]
                .as_bool()
                .unwrap()
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
        assert_eq!(f.responses.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn live_controller_questions_repause_once_per_observation_and_reject_wrong_responses_before_claim()
 {
    for sqlite in support::backends() {
        let provider = super::super::runner_tests::RecordingProvider::new(&[
            "working",
            "still working",
            "done",
        ]);
        let (agent, _, _) = super::super::runner_tests::fixture(provider.clone(), 4);
        let store = support::store(agent.info().id, "prepared-v1", *sqlite).await;
        let mut config = super::super::runner_tests::config(4);
        config.defaults.premature_finish =
            Some(serde_yaml::from_str("action: ask_user\non_unavailable: fail").unwrap());
        let runner = live_runner(agent.clone(), store.clone(), config, "prepared-v1");
        let first = runner.run("tool-free analysis", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        assert!(
            runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::Continue
                )
                .await
                .is_err()
        );
        assert!(
            runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::Approval {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        result: ApprovalResult::Approved
                    }
                )
                .await
                .is_err()
        );
        assert!(
            runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!({"action":"revise_limits"})
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(
            store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
        let second = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                answer(&first.run),
            )
            .await
            .unwrap();
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        assert_ne!(
            first.run.pending.as_ref().unwrap().id,
            second.run.pending.as_ref().unwrap().id
        );
        assert!(
            runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    answer(&first.run)
                )
                .await
                .is_err()
        );
        let done = runner
            .resume(
                &second.run.key.run_id,
                second.run.revision,
                answer(&second.run),
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
        assert_eq!(done.run.counters.turns, 3);
        assert_eq!(done.run.counters.tool_attempts, 0);
        assert_eq!(done.run.counters.continuations, 0);
        assert!(
            provider.captured_messages()[1]
                .iter()
                .any(|m| m.content.contains("host_answer"))
        );
        assert!(agent.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn live_controller_capacity_wait_requires_explicit_revision_and_preserves_pending_and_usage()
{
    for sqlite in support::backends() {
        for action in ["ask_user", "pause_run"] {
            let provider = super::super::runner_tests::RecordingProvider::new(&["working", "done"]);
            let (agent, _, _) = super::super::runner_tests::fixture(provider.clone(), 1);
            let store = support::store(agent.info().id, "prepared-v1", *sqlite).await;
            let mut config = super::super::runner_tests::config(1);
            config.defaults.on_limit = Some(
                serde_yaml::from_str(&format!("action: {action}\non_unavailable: fail")).unwrap(),
            );
            let runner = live_runner(agent.clone(), store.clone(), config, "prepared-v1");
            let paused = runner.run("support resolution", None).await.unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{action}: {:?}",
                paused.run.stop_reason
            );
            let input = if action == "pause_run" {
                TaskResumeInput::Continue
            } else {
                answer(&paused.run)
            };
            assert!(
                runner
                    .resume(&paused.run.key.run_id, paused.run.revision, input)
                    .await
                    .is_err()
            );
            assert_eq!(
                store
                    .load(&paused.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                paused.run.revision
            );
            let payload: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&paused.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            let mut limits = payload.limits;
            limits.max_llm_calls = 2;
            let revised = runner
                .revise_limits_paused(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    limits,
                    "allow another attempt",
                )
                .await
                .unwrap();
            assert_eq!(
                revised.run.pending.as_ref().unwrap().id,
                paused.run.pending.as_ref().unwrap().id
            );
            assert_eq!(revised.run.counters.llm_attempts, 1);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let input = if action == "pause_run" {
                TaskResumeInput::Continue
            } else {
                answer(&revised.run)
            };
            let done = runner
                .resume(&revised.run.key.run_id, revised.run.revision, input)
                .await
                .unwrap();
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(done.run.counters.llm_attempts, 2);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        }
    }
}

#[tokio::test]
async fn live_controller_stagnation_question_suppresses_premature_finish_and_completion_wins() {
    for first_done in [false, true] {
        let provider = super::super::runner_tests::RecordingProvider::new(if first_done {
            &["done"]
        } else {
            &["working", "done"]
        });
        let (agent, _, store) = super::super::runner_tests::fixture(provider.clone(), 4);
        let mut config = super::super::runner_tests::config(4);
        config.defaults.progress = Some(serde_yaml::from_str("stagnation: {max_cycles_without_progress: 1, action: ask_user, on_unavailable: fail}").unwrap());
        config.defaults.premature_finish = Some(serde_yaml::from_str("action: fail").unwrap());
        let runner = live_runner(agent, store, config, "prepared-v1");
        let result = runner.run("research", None).await.unwrap();
        if first_done {
            assert_eq!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        } else {
            assert_eq!(result.run.status, TaskRunStatus::Paused);
            assert_eq!(
                result.run.pending.as_ref().unwrap().reviewed_action["proposal"]["reason"],
                "no_progress"
            );
            let done = runner
                .resume(
                    &result.run.key.run_id,
                    result.run.revision,
                    answer(&result.run),
                )
                .await
                .unwrap();
            assert_eq!(done.run.status, TaskRunStatus::Completed);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        }
    }
}

#[tokio::test]
async fn live_controller_unavailable_and_host_stop_do_not_approve_or_complete_unmet_gates() {
    for available in [false, true] {
        let provider = super::super::runner_tests::RecordingProvider::new(&["working"]);
        let (agent, _, store) = super::super::runner_tests::fixture(provider.clone(), 4);
        let mut config = super::super::runner_tests::config(4);
        config.defaults.premature_finish =
            Some(serde_yaml::from_str("action: ask_user\non_unavailable: fail").unwrap());
        let runner = live_runner(agent.clone(), store, config, "prepared-v1")
            .with_interaction_delivery(available);
        let result = runner.run("support", None).await.unwrap();
        let done = if available {
            assert_eq!(result.run.status, TaskRunStatus::Paused);
            runner
                .controller_unavailable_paused(
                    &result.run.key.run_id,
                    result.run.revision,
                    &result.run.pending.as_ref().unwrap().id,
                    "host confirmed channel loss",
                )
                .await
                .unwrap()
        } else {
            result
        };
        assert_eq!(done.run.status, TaskRunStatus::Failed);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(agent.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn live_skill_capacity_preserves_the_original_prompt_prefix_and_logical_turn() {
    for sqlite in support::backends() {
        let mut f = Box::pin(skill_fixture(false)).await;
        f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
        let mut config = super::super::runner_tests::config(1);
        config.defaults.hitl = Some(
            serde_yaml::from_str("on_approval_required: pause_run\non_user_question: pause_run")
                .unwrap(),
        );
        config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
        f.runner = live_runner(f.agent.clone(), f.store.clone(), config, "locations.v1");
        let approval = f.runner.run_skill("report", "script", None).await.unwrap();
        let outcome = f
            .runner
            .resume(
                &approval.run.key.run_id,
                approval.run.revision,
                TaskResumeInput::Approval {
                    request_id: approval.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Approved,
                },
            )
            .await;
        assert!(outcome.is_ok(), "{outcome:?}");
        let paused = outcome.unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        assert!(matches!(
            paused.run.pending.as_ref().unwrap().kind,
            TaskPendingKind::Limit
        ));
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 1);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            f.store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let mut limits = payload.limits;
        limits.max_llm_calls = 2;
        let revised = f
            .runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                paused.run.revision,
                limits,
                "complete report prompt",
            )
            .await
            .unwrap();
        let done = f
            .runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 1);
        assert_eq!(done.run.counters.turns, 1);
        assert_eq!(f.responses.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn live_skill_final_tool_admission_reparks_after_approval_without_transferring_authority() {
    for sqlite in support::backends() {
        let mut f = Box::pin(skill_fixture(false)).await;
        f.store = support::store(f.agent.info().id, "locations.v1", *sqlite).await;
        let mut config = super::super::runner_tests::config(4);
        config.defaults.max_tool_calls = Some(1);
        config.defaults.hitl =
            Some(serde_yaml::from_str("on_approval_required: pause_run").unwrap());
        config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
        f.runner = live_runner(f.agent.clone(), f.store.clone(), config, "locations.v1");
        let first = f.runner.run_skill("report", "script", None).await.unwrap();
        let paused = approve(&f, &first.run).await;
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        assert!(matches!(
            paused.run.pending.as_ref().unwrap().kind,
            TaskPendingKind::Limit
        ));
        assert_eq!(f.second.load(Ordering::SeqCst), 0);
        assert!(
            f.runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    TaskResumeInput::Continue
                )
                .await
                .is_err()
        );
        let payload: TaskCheckpointPayload = serde_json::from_value(
            f.store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let mut limits = payload.limits;
        limits.max_tool_calls = 2;
        let revised = f
            .runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                paused.run.revision,
                limits,
                "admit remaining tool",
            )
            .await
            .unwrap();
        let done = f
            .runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
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
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 0);
        assert_eq!(done.run.counters.turns, 1);
        assert_eq!(done.run.counters.tool_attempts, 2);
    }
}

#[tokio::test]
async fn live_controller_required_judge_resumes_only_validation_after_explicit_limit_revision() {
    for sqlite in support::backends() {
        let provider = super::super::runner_tests::RecordingProvider::new(&[
            "done",
            "{\"overall_score\":1.0}",
        ]);
        let (agent, _, _) = super::super::runner_tests::fixture(provider.clone(), 1);
        let store = support::store(agent.info().id, "prepared-v1", *sqlite).await;
        let mut config = super::super::runner_tests::config(1);
        config.defaults.validation = Some(serde_yaml::from_str("checks:\n  - id: quality\n    adapter: builtin.judge\n    timeout_seconds: 60\n    config: {criteria: [assess report quality], pass_threshold: 0.7}").unwrap());
        config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
        let runner = live_runner(agent, store.clone(), config, "prepared-v1");
        let paused = runner.run("research report", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        let payload: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let mut limits = payload.limits;
        limits.max_llm_calls = 2;
        let revised = runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                paused.run.revision,
                limits,
                "allow required judge",
            )
            .await
            .unwrap();
        let done = runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(done.run.counters.turns, 1);
    }
}

#[tokio::test]
async fn live_controller_question_timeout_and_single_winner_stop_preserve_owner_release() {
    let provider = super::super::runner_tests::RecordingProvider::new(&["working"]);
    let yaml = "name: support\nsystem_prompt: help\nhitl: {default_timeout_seconds: 1, on_timeout: reject}\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(NoApproval))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "support.v1".into()).unwrap(),
    );
    let mut config = super::super::runner_tests::config(4);
    config.defaults.premature_finish = Some(serde_yaml::from_str("action: ask_user").unwrap());
    let runner = live_runner(agent.clone(), store.clone(), config, "support.v1");
    let paused = runner.run("support", None).await.unwrap();
    assert!(
        runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::QuestionTimeout {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone()
                }
            )
            .await
            .is_err()
    );
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    assert!(
        runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                answer(&paused.run)
            )
            .await
            .is_err()
    );
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::QuestionTimeout {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Incomplete);
    assert_eq!(done.run.stop_reason.as_deref(), Some("question_timeout"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(agent.set_context("released", json!(true)).is_ok());

    for sqlite in support::backends() {
        let provider = super::super::runner_tests::RecordingProvider::new(&["working", "done"]);
        let (agent, _, _) = super::super::runner_tests::fixture(provider.clone(), 4);
        let store = support::store(agent.info().id, "prepared-v1", *sqlite).await;
        let mut config = super::super::runner_tests::config(4);
        config.defaults.premature_finish = Some(serde_yaml::from_str("action: ask_user").unwrap());
        let runner = live_runner(agent.clone(), store, config, "prepared-v1");
        let paused = runner.run("support", None).await.unwrap();
        let (resumed, stopped) = tokio::join!(
            runner.resume(
                &paused.run.key.run_id,
                paused.run.revision,
                answer(&paused.run)
            ),
            runner.stop_paused(&paused.run.key.run_id, paused.run.revision, "stop")
        );
        assert_eq!(
            usize::from(resumed.is_ok()) + usize::from(stopped.is_ok()),
            1
        );
        assert!(agent.set_context("released", json!(true)).is_ok());
        assert!(provider.calls.load(Ordering::SeqCst) <= 2);
    }
}

// Inject failure only after the selected ACK boundary is reached; scheduler latency before that point is not an interrupted acknowledgement.
#[tokio::test]
async fn live_controller_pause_and_answer_ack_faults_retain_protection_without_dispatch() {
    for sqlite in support::backends() {
        for mode in ["reject", "lost", "drop"] {
            for consumption in [false, true] {
                let provider =
                    super::super::runner_tests::RecordingProvider::new(&["working", "done"]);
                let (agent, _, _) = super::super::runner_tests::fixture(provider.clone(), 4);
                let store = support::store(agent.info().id, "prepared-v1", *sqlite).await;
                let mut config = super::super::runner_tests::config(4);
                config.defaults.premature_finish =
                    Some(serde_yaml::from_str("action: ask_user").unwrap());
                let fault = support::ReceiptStore::new(store.clone(), mode, TaskRunStatus::Paused);
                let runner = live_runner(agent.clone(), fault.clone(), config, "prepared-v1");
                let run = if consumption {
                    let paused = runner.run("analysis", None).await.unwrap();
                    *fault.checkpoint.lock() =
                        Some(json!({"consume":paused.run.pending.as_ref().unwrap().id}));
                    Some(paused.run)
                } else {
                    None
                };
                fault.armed.store(true, Ordering::SeqCst);
                let mut work = Box::pin(async {
                    if let Some(run) = run {
                        runner
                            .resume(&run.key.run_id, run.revision, answer(&run))
                            .await
                    } else {
                        runner.run("analysis", None).await
                    }
                });
                let returned = tokio::select! {
                    result = &mut work => {
                        fault.reached.try_acquire().expect("operation failed before the injected ACK fault").forget();
                        Some(result)
                    },
                    permit = fault.reached.acquire() => { permit.unwrap().forget(); None },
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("selected ACK boundary was not reached"),
                };
                fault.assert_hit();
                if mode == "drop" {
                    assert!(returned.is_none(), "dropped ACK must remain pending");
                    drop(work);
                } else {
                    let outcome = if let Some(result) = returned {
                        result
                    } else {
                        tokio::time::timeout(std::time::Duration::from_secs(5), &mut work)
                            .await
                            .expect("injected ACK error handling did not finish")
                    };
                    assert!(outcome.is_err());
                }
                fault.assert_hit();
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                assert!(agent.set_context("protected", json!(true)).is_err());
            }
        }
    }
}
