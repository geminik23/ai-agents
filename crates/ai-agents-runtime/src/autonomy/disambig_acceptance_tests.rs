use super::*;

use super::super::continuation_test_support::{self as support, ReceiptStore};
use ai_agents_core::Result;

const CLEAR: &str = r#"{"is_ambiguous":false,"confidence":1.0}"#;
const UNCLEAR: &str = r#"{"status":"unclear"}"#;
const CONFIRM: &str = r#"{"question":"Confirm target one?"}"#;

fn confirmation_yaml(action: &str, hitl: bool) -> String {
    let yaml = CLARIFICATION.replace("  clarification:\n    llm: router\n", &format!("  clarification:\n    llm: router\n    max_attempts: 2\n    on_max_attempts: {action}\n"));
    format!(
        "{yaml}{}states:\n  initial: active\n  states:\n    active:\n      disambiguation:\n        require_confirmation: true\n    other: {{}}\n",
        if hitl {
            "hitl:\n  default_timeout_seconds: 1\n  message_language:\n    strategy: explicit\n    explicit: en\n"
        } else {
            ""
        }
    )
}
fn skill_yaml(action: &str) -> String {
    format!("{}process:\n  input:\n    - type: extract\n      config:\n        llm: router\n        schema:\n          tag:\n            type: string\n            required: true\n        store_in_context: extracted\n  output:\n    - type: format\n      config:\n        template: '{{{{ response }}}} : {{{{ context.extracted }}}}'\nskills:\n  - id: script\n    description: work\n    trigger: work\n    disambiguation:\n      enabled: true\n      required_clarity: [target]\n    steps:\n      - prompt: 'Finish {{{{ input }}}}'\n", CLARIFICATION.replace("  clarification:\n    llm: router\n", &format!("  clarification:\n    llm: router\n    max_attempts: 2\n    on_max_attempts: {action}\n")))
}
fn runner(agent: Arc<crate::RuntimeAgent>, store: Arc<dyn TaskRunStore>) -> AutonomyRunner {
    runner_with_timeout(agent, store, false)
}
fn runner_with_timeout(
    agent: Arc<crate::RuntimeAgent>,
    store: Arc<dyn TaskRunStore>,
    timeout: bool,
) -> AutonomyRunner {
    let mut config = super::super::runner_tests::config(24);
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
        "interaction.v1".into(),
    )
    .unwrap()
}
async fn answer(f: &Fixture, run: &TaskRun, text: &str) -> Result<TaskRunResult> {
    f.runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::UserAnswer {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                answer: json!(text),
            },
        )
        .await
}
async fn payload(f: &Fixture, run: &TaskRun) -> TaskCheckpointPayload {
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
async fn disambig_acceptance_original_question_deadline_settles_root_confirmation_and_skill_recheck()
 {
    for kind in ["root", "confirmation", "skill"] {
        let yaml = match kind {
            "confirmation" => confirmation_yaml("apologize_and_stop", true),
            "skill" => format!(
                "{}hitl:\n  default_timeout_seconds: 1\n",
                skill_yaml("apologize_and_stop")
            ),
            _ => format!("{CLARIFICATION}hitl:\n  default_timeout_seconds: 1\n"),
        };
        let replies = match kind {
            "confirmation" => vec![AMBIGUOUS, QUESTION, ANSWER, CONFIRM],
            "skill" => vec![
                CLEAR,
                r#"{"tag":"original"}"#,
                "script",
                AMBIGUOUS,
                QUESTION,
                ANSWER,
                AMBIGUOUS,
                QUESTION,
            ],
            _ => vec![AMBIGUOUS, QUESTION],
        };
        let f = fixture(&yaml, &["done"], &replies);
        let first = f.runner.run("objective", None).await.unwrap();
        let paused = if kind == "root" {
            first
        } else {
            answer(&f, &first.run, "target one").await.unwrap()
        };
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        let p = payload(&f, &paused.run).await;
        let batch = p
            .adapters
            .iter()
            .find(|a| a.id == "runtime.tool_batch")
            .unwrap();
        let expiry: chrono::DateTime<chrono::Utc> =
            serde_json::from_value(batch.state["approvals"][0]["expires_at"].clone())
                .expect("real clarification deadline required");
        let timeout = || TaskResumeInput::QuestionTimeout {
            request_id: paused.run.pending.as_ref().unwrap().id.clone(),
        };
        assert!(
            f.runner
                .resume(&paused.run.key.run_id, paused.run.revision, timeout())
                .await
                .is_err()
        );
        let before = f.router.calls.load(Ordering::SeqCst);
        let remaining = (expiry - chrono::Utc::now()).to_std().unwrap_or_default();
        tokio::time::sleep(remaining + std::time::Duration::from_millis(20)).await;
        assert!(answer(&f, &paused.run, "late answer").await.is_err());
        let done = f
            .runner
            .resume(&paused.run.key.run_id, paused.run.revision, timeout())
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Incomplete);
        assert_eq!(done.run.stop_reason.as_deref(), Some("question_timeout"));
        assert_eq!(f.router.calls.load(Ordering::SeqCst), before);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
        assert!(
            !f.agent
                .disambiguation_manager()
                .unwrap()
                .has_pending_clarification()
                .await
        );
        assert!(f.agent.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn disambig_acceptance_confirmation_exhaustion_never_bypasses_required_confirmation() {
    for (action, hitl) in [
        ("apologize_and_stop", false),
        ("proceed_with_best_guess", false),
        ("escalate", false),
        ("escalate", true),
    ] {
        for approved in [false, true] {
            let f = fixture(
                &confirmation_yaml(action, hitl),
                &["done"],
                &[
                    AMBIGUOUS,
                    QUESTION,
                    ANSWER,
                    CONFIRM,
                    UNCLEAR,
                    "done apology",
                ],
            );
            let first = f.runner.run("original objective", None).await.unwrap();
            let confirmation = answer(&f, &first.run, "target one").await.unwrap();
            let mut done = answer(&f, &confirmation.run, "maybe").await.unwrap();
            if action == "escalate" && hitl {
                assert_eq!(done.run.status, TaskRunStatus::Paused);
                assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
                done = respond(
                    &f,
                    &done.run,
                    if approved {
                        ApprovalResult::Approved
                    } else {
                        ApprovalResult::Rejected { reason: None }
                    },
                )
                .await;
                assert_eq!(
                    done.run.status,
                    if approved {
                        TaskRunStatus::Completed
                    } else {
                        TaskRunStatus::Failed
                    }
                );
                assert_eq!(f.main.calls.load(Ordering::SeqCst), usize::from(approved));
            } else {
                assert_eq!(
                    done.run.status,
                    TaskRunStatus::Incomplete,
                    "action={action}"
                );
                assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
            }
            assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
            assert!(
                !f.agent
                    .disambiguation_manager()
                    .unwrap()
                    .has_pending_clarification()
                    .await
            );
            assert!(f.agent.set_context("released", json!(true)).is_ok());
            let p = payload(&f, &done.run).await;
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
}

#[tokio::test]
async fn disambig_acceptance_skill_recheck_repause_exhaustion_preserves_original_processing() {
    for (action, review) in [
        ("apologize_and_stop", false),
        ("escalate", false),
        ("escalate", true),
    ] {
        for approved in [false, true] {
            let yaml = format!(
                "{}{}",
                skill_yaml(action),
                if review {
                    "hitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n"
                } else {
                    ""
                }
            );
            let f = fixture(
                &yaml,
                &["done"],
                &[
                    CLEAR,
                    r#"{"tag":"original"}"#,
                    "script",
                    AMBIGUOUS,
                    QUESTION,
                    ANSWER,
                    AMBIGUOUS,
                    QUESTION,
                    UNCLEAR,
                    "done apology",
                ],
            );
            let first = f.runner.run("original objective", None).await.unwrap();
            let second = answer(&f, &first.run, "target one").await.unwrap();
            assert_eq!(second.run.status, TaskRunStatus::Paused);
            assert_ne!(
                second.run.pending.as_ref().unwrap().id,
                first.run.pending.as_ref().unwrap().id
            );
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 8);
            assert!(answer(&f, &first.run, "stale").await.is_err());
            let mut done = answer(&f, &second.run, "maybe").await.unwrap();
            if review {
                assert_eq!(done.run.status, TaskRunStatus::Paused);
                done = respond(
                    &f,
                    &done.run,
                    if approved {
                        ApprovalResult::Approved
                    } else {
                        ApprovalResult::Rejected { reason: None }
                    },
                )
                .await;
                assert_eq!(
                    done.run.status,
                    if approved {
                        TaskRunStatus::Completed
                    } else {
                        TaskRunStatus::Failed
                    }
                );
                if approved {
                    assert!(
                        done.final_response
                            .as_ref()
                            .unwrap()
                            .content
                            .contains("original")
                    );
                }
            } else {
                assert_eq!(done.run.status, TaskRunStatus::Incomplete);
            }
            assert_eq!(
                f.main.calls.load(Ordering::SeqCst),
                usize::from(review && approved)
            );
            assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
            assert_eq!(
                f.router.calls.load(Ordering::SeqCst),
                if action == "escalate" && review {
                    9
                } else {
                    10
                }
            );
            assert!(
                !f.agent
                    .disambiguation_manager()
                    .unwrap()
                    .has_pending_clarification()
                    .await
            );
        }
    }
}

#[tokio::test]
async fn disambig_acceptance_replacement_or_generation_rejects_before_claim_and_preserves_new_exchange()
 {
    for kind in ["replacement", "generation", "cleared"] {
        let f = fixture(
            &confirmation_yaml("apologize_and_stop", false),
            &["done"],
            &[AMBIGUOUS, QUESTION, AMBIGUOUS, QUESTION],
        );
        let first = f.runner.run("objective", None).await.unwrap();
        let manager = f.agent.disambiguation_manager().unwrap();
        let old = manager.pending_checkpoint().await.unwrap().unwrap();
        match kind {
            "replacement" => {
                manager.clear_pending_checkpoint(&old).await.unwrap();
                manager
                    .process_input(
                        "new exchange",
                        &ai_agents_disambiguation::DisambiguationContext::default(),
                    )
                    .await
                    .unwrap();
            }
            "generation" => {
                let machine = f.agent.state_machine().unwrap();
                machine.transition_to("other", "trusted host").unwrap();
                machine.transition_to("active", "trusted host").unwrap();
            }
            _ => {
                manager.clear_pending_checkpoint(&old).await.unwrap();
            }
        }
        let live = manager.pending_checkpoint().await.unwrap();
        let before = f.router.calls.load(Ordering::SeqCst);
        assert!(answer(&f, &first.run, "old answer").await.is_err());
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
        assert_eq!(f.router.calls.load(Ordering::SeqCst), before);
        let stopped = f
            .runner
            .cancel_paused(&first.run.key.run_id, first.run.revision)
            .await
            .unwrap();
        assert_eq!(stopped.run.status, TaskRunStatus::Cancelled);
        if kind == "replacement" {
            assert_eq!(manager.pending_checkpoint().await.unwrap(), live);
        } else {
            assert!(!manager.has_pending_clarification().await);
        }
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn disambig_acceptance_repause_and_terminal_receipt_failures_preserve_semantics_and_owner() {
    for sqlite in support::backends() {
        for terminal in [false, true] {
            for mode in ["positive", "reject", "lost", "drop"] {
                let mut f = fixture(
                    &confirmation_yaml("apologize_and_stop", false),
                    &["done"],
                    &[
                        AMBIGUOUS,
                        QUESTION,
                        ANSWER,
                        CONFIRM,
                        r#"{"status":"confirmed"}"#,
                    ],
                );
                f.store = support::store(f.agent.info().id, "interaction.v1", *sqlite).await;
                let fault = ReceiptStore::new(
                    f.store.clone(),
                    mode,
                    if terminal {
                        TaskRunStatus::Completed
                    } else {
                        TaskRunStatus::Paused
                    },
                );
                f.runner = runner(f.agent.clone(), fault.clone());
                let mut paused = f.runner.run("objective", None).await.unwrap();
                if terminal {
                    paused = answer(&f, &paused.run, "target one").await.unwrap();
                }
                fault.armed.store(true, Ordering::SeqCst);
                let mut resume = Box::pin(answer(
                    &f,
                    &paused.run,
                    if terminal { "yes" } else { "target one" },
                ));
                if mode == "positive" {
                    let mut done = resume.await.unwrap();
                    fault.assert_final_write(&paused.run.key.run_id).await;
                    if !terminal {
                        done = answer(&f, &done.run, "yes").await.unwrap();
                    }
                    assert_eq!(done.run.status, TaskRunStatus::Completed);
                    assert_eq!(f.router.calls.load(Ordering::SeqCst), 5);
                    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
                    assert!(f.agent.set_context("released", json!(true)).is_ok());
                } else {
                    if mode == "drop" {
                        tokio::select! {
                            outcome = &mut resume => panic!("receipt did not hold: {outcome:?}"),
                            permit = fault.reached.acquire() => permit.unwrap().forget(),
                            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("receipt not reached"),
                        }
                        drop(resume);
                    } else {
                        assert!(resume.await.is_err());
                    }
                    fault.assert_final_write(&paused.run.key.run_id).await;
                    let count = f.router.calls.load(Ordering::SeqCst);
                    assert!(answer(&f, &paused.run, "replay").await.is_err());
                    assert_eq!(f.router.calls.load(Ordering::SeqCst), count);
                    assert_eq!(count, if terminal { 5 } else { 4 });
                    assert_eq!(f.main.calls.load(Ordering::SeqCst), usize::from(terminal));
                    assert_eq!(f.finalized.load(Ordering::SeqCst), usize::from(terminal));
                    assert!(f.agent.set_context("blocked", json!(true)).is_err());
                    let snapshot = f.store.load(&paused.run.key.run_id).await.unwrap().unwrap();
                    assert!(
                        f.runner
                            .resume(
                                &paused.run.key.run_id,
                                snapshot.revision,
                                TaskResumeInput::UserAnswer {
                                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                                    answer: json!("replay")
                                }
                            )
                            .await
                            .is_err()
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn disambig_acceptance_provider_failure_or_drop_refuses_semantic_replay() {
    for confirmation in [false, true] {
        for fault_at in [2, 3] {
            for mode in ["error", "drop", "timeout"] {
                let yaml = if confirmation {
                    confirmation_yaml("apologize_and_stop", false)
                } else {
                    CLARIFICATION.replace("  clarification:\n    llm: router\n", "  clarification:\n    llm: router\n    max_attempts: 3\n    on_max_attempts: apologize_and_stop\n")
                };
                let replies = if confirmation {
                    vec![AMBIGUOUS, QUESTION, ANSWER, CONFIRM]
                } else {
                    vec![AMBIGUOUS, QUESTION, UNCLEAR, QUESTION]
                };
                let provider = support::FaultProvider::new(
                    &replies,
                    fault_at,
                    if mode == "timeout" { "drop" } else { mode },
                );
                let main = super::super::runner_tests::RecordingProvider::new(&["done"]);
                let finalized = Arc::new(AtomicUsize::new(0));
                let mut llms = ai_agents_llm::LLMRegistry::new();
                llms.register("router", provider.clone());
                llms.register("default", main.clone());
                let agent = Arc::new(
                    AgentBuilder::from_yaml(&yaml)
                        .unwrap()
                        .llm_registry(llms)
                        .auto_configure_features()
                        .unwrap()
                        .hooks(Arc::new(Hooks(finalized.clone())))
                        .build()
                        .unwrap(),
                );
                let store = support::store(agent.info().id, "interaction.v1", false).await;
                let runner = runner_with_timeout(agent.clone(), store.clone(), mode == "timeout");
                let first = runner.run("objective", None).await.unwrap();
                let input = || TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!("target one"),
                };
                let mut resume =
                    Box::pin(runner.resume(&first.run.key.run_id, first.run.revision, input()));
                if mode == "drop" {
                    tokio::select! {
                        outcome = &mut resume => panic!("provider did not hold: {outcome:?}"),
                        permit = provider.reached.acquire() => permit.unwrap().forget(),
                        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("provider not reached"),
                    }
                    drop(resume);
                } else {
                    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), resume)
                        .await
                        .unwrap();
                    assert!(
                        outcome.is_err(),
                        "confirmation={confirmation}, at={fault_at}, calls={}, outcome={outcome:?}",
                        provider.calls.load(Ordering::SeqCst)
                    );
                }
                assert_eq!(provider.calls.load(Ordering::SeqCst), fault_at + 1);
                assert_eq!(main.calls.load(Ordering::SeqCst), 0);
                assert_eq!(finalized.load(Ordering::SeqCst), 0);
                assert!(agent.set_context("blocked", json!(true)).is_err());
                assert!(
                    runner
                        .resume(&first.run.key.run_id, first.run.revision, input())
                        .await
                        .is_err()
                );
                let current = store.load(&first.run.key.run_id).await.unwrap().unwrap();
                assert!(
                    runner
                        .resume(&first.run.key.run_id, current.revision, input())
                        .await
                        .is_err()
                );
                assert_eq!(provider.calls.load(Ordering::SeqCst), fault_at + 1);
                let p: TaskCheckpointPayload = serde_json::from_value(current.payload).unwrap();
                if mode != "error" {
                    assert!(p.reservations.iter().any(|r| matches!(
                        r.state,
                        TaskEffectState::Dispatched | TaskEffectState::Uncertain
                    )));
                }
            }
        }
    }
}

#[tokio::test]
async fn disambig_acceptance_initial_pause_receipt_failure_has_no_semantic_replay() {
    for sqlite in support::backends() {
        for mode in ["reject", "lost", "drop"] {
            let mut f = fixture(CLARIFICATION, &["done"], &[AMBIGUOUS, QUESTION]);
            f.store = support::store(f.agent.info().id, "interaction.v1", *sqlite).await;
            let fault = ReceiptStore::new(f.store.clone(), mode, TaskRunStatus::Paused);
            fault.armed.store(true, Ordering::SeqCst);
            f.runner = runner(f.agent.clone(), fault.clone());
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
            fault.assert_hit();
            let snapshot = f.store.list().await.unwrap().pop().unwrap();
            fault.assert_final_write(&snapshot.key.run_id).await;
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
            assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
            assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
            assert!(f.agent.set_context("blocked", json!(true)).is_err());
            assert!(f.runner.run("replay", None).await.is_err());
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
        }
    }
}

#[tokio::test]
async fn disambig_acceptance_drop_during_finalization_protects_resolved_exchange() {
    let mut f = fixture(CLARIFICATION, &["done"], &[AMBIGUOUS, QUESTION, ANSWER]);
    drop(f.runner);
    let gate = support::ResponseGate::new();
    let runtime = Arc::try_unwrap(f.agent).unwrap_or_else(|_| panic!("fixture retained runtime"));
    f.agent = Arc::new(runtime.with_hooks(gate.clone()));
    f.runner = runner(f.agent.clone(), f.store.clone());
    let first = f.runner.run("objective", None).await.unwrap();
    let mut resume = Box::pin(answer(&f, &first.run, "target one"));
    tokio::select! {
        result = &mut resume => panic!("hook not held: {result:?}"),
        permit = gate.reached.acquire() => permit.unwrap().forget(),
        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("hook not reached"),
    }
    drop(resume);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 3);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
    assert!(answer(&f, &first.run, "replay").await.is_err());
    assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
    assert!(f.agent.set_context("blocked", json!(true)).is_err());
}

#[tokio::test]
async fn disambig_acceptance_inflight_generation_change_cannot_clear_replaced_exchange() {
    let yaml = confirmation_yaml("apologize_and_stop", false)
        .replace("require_confirmation: true", "require_confirmation: false");
    let provider = support::FaultProvider::new(
        &[AMBIGUOUS, QUESTION, ANSWER, AMBIGUOUS, QUESTION],
        2,
        "gate",
    );
    let main = super::super::runner_tests::RecordingProvider::new(&["done"]);
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register("default", main.clone());
    llms.register("router", provider.clone());
    let agent = Arc::new(
        AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm_registry(llms)
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let store = support::store(agent.info().id, "interaction.v1", false).await;
    let runner = runner(agent.clone(), store);
    let first = runner.run("objective", None).await.unwrap();
    let mut resume = Box::pin(runner.resume(
        &first.run.key.run_id,
        first.run.revision,
        TaskResumeInput::UserAnswer {
            request_id: first.run.pending.as_ref().unwrap().id.clone(),
            answer: json!("target one"),
        },
    ));
    tokio::select! {
        result = &mut resume => panic!("parser not held: {result:?}"),
        permit = provider.reached.acquire() => permit.unwrap().forget(),
        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("parser not reached"),
    }
    let manager = agent.disambiguation_manager().unwrap();
    manager.clear_pending().await;
    manager
        .process_input(
            "new exchange",
            &ai_agents_disambiguation::DisambiguationContext::default(),
        )
        .await
        .unwrap();
    let new_exchange = manager.pending_checkpoint().await.unwrap();
    agent
        .state_machine()
        .unwrap()
        .transition_to("other", "trusted host")
        .unwrap();
    provider.release.add_permits(1);
    assert!(resume.await.is_err());
    assert_eq!(manager.pending_checkpoint().await.unwrap(), new_exchange);
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    assert!(agent.set_context("blocked", json!(true)).is_err());
}

#[tokio::test]
async fn disambig_acceptance_initial_skill_and_recheck_invalidation_preserve_newer_exchange() {
    for recheck in [false, true] {
        let yaml = format!(
            "{}states:\n  initial: active\n  states:\n    active: {{}}\n    other: {{}}\n",
            skill_yaml("apologize_and_stop")
        );
        let replies = if recheck {
            vec![
                CLEAR,
                r#"{"tag":"original"}"#,
                "script",
                AMBIGUOUS,
                QUESTION,
                ANSWER,
                CLEAR,
                AMBIGUOUS,
                QUESTION,
            ]
        } else {
            vec![
                CLEAR,
                r#"{"tag":"original"}"#,
                "script",
                CLEAR,
                AMBIGUOUS,
                QUESTION,
            ]
        };
        let fault_at = if recheck { 6 } else { 3 };
        let provider = support::FaultProvider::new(&replies, fault_at, "gate");
        let main = super::super::runner_tests::RecordingProvider::new(&["done"]);
        let mut llms = ai_agents_llm::LLMRegistry::new();
        llms.register("default", main.clone());
        llms.register("router", provider.clone());
        let agent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm_registry(llms)
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "interaction.v1", false).await;
        let runner = runner(agent.clone(), store);
        let paused = if recheck {
            Some(runner.run("objective", None).await.unwrap())
        } else {
            None
        };
        let mut work = Box::pin(async {
            if let Some(first) = &paused {
                runner
                    .resume(
                        &first.run.key.run_id,
                        first.run.revision,
                        TaskResumeInput::UserAnswer {
                            request_id: first.run.pending.as_ref().unwrap().id.clone(),
                            answer: json!("target one"),
                        },
                    )
                    .await
            } else {
                runner.run("objective", None).await
            }
        });
        tokio::select! {
            result = &mut work => panic!("semantic work not gated: {result:?}"),
            permit = provider.reached.acquire() => permit.unwrap().forget(),
            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("semantic work not reached"),
        }
        let manager = agent.disambiguation_manager().unwrap();
        manager.clear_pending().await;
        manager
            .process_input(
                "new exchange",
                &ai_agents_disambiguation::DisambiguationContext::default(),
            )
            .await
            .unwrap();
        let replacement = manager.pending_checkpoint().await.unwrap();
        agent
            .state_machine()
            .unwrap()
            .transition_to("other", "trusted host")
            .unwrap();
        provider.release.add_permits(1);
        let outcome = work.await;
        assert!(
            outcome
                .as_ref()
                .is_ok_and(|r| r.run.status != TaskRunStatus::Completed)
                || outcome.is_err()
        );
        assert_eq!(manager.pending_checkpoint().await.unwrap(), replacement);
        assert_eq!(main.calls.load(Ordering::SeqCst), 0);
        if recheck {
            assert!(agent.set_context("blocked", json!(true)).is_err());
        }
    }
}

#[tokio::test]
async fn disambig_acceptance_unclear_confirmation_repause_preserves_resolution_and_rejects_old_id()
{
    let yaml = confirmation_yaml("apologize_and_stop", false)
        .replace("max_attempts: 2", "max_attempts: 3");
    let f = fixture(
        &yaml,
        &["done"],
        &[
            AMBIGUOUS,
            QUESTION,
            ANSWER,
            CONFIRM,
            UNCLEAR,
            r#"{"status":"confirmed"}"#,
        ],
    );
    let first = f.runner.run("objective", None).await.unwrap();
    let confirmed = answer(&f, &first.run, "target one").await.unwrap();
    let second = answer(&f, &confirmed.run, "maybe").await.unwrap();
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 5);
    assert!(answer(&f, &confirmed.run, "stale").await.is_err());
    let checkpoint = f
        .agent
        .disambiguation_manager()
        .unwrap()
        .pending_checkpoint()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        checkpoint["phase"]["Confirmation"]["enriched_input"],
        "objective for target one"
    );
    assert_eq!(
        checkpoint["phase"]["Confirmation"]["resolved"]["target"],
        "one"
    );
    let done = answer(&f, &second.run, "yes").await.unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 6);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn disambig_acceptance_semantic_abandonment_switch_and_rejection_do_not_start_new_task_work()
{
    for skill in [false, true] {
        for confirmation in [false, true] {
            for status in if confirmation {
                &["abandoned", "switched", "rejected"][..]
            } else {
                &["abandoned", "switched"][..]
            } {
                let yaml = if skill {
                    format!(
                        "{}{}",
                        skill_yaml("apologize_and_stop"),
                        if confirmation {
                            "states:\n  initial: active\n  states:\n    active:\n      disambiguation:\n        require_confirmation: true\n"
                        } else {
                            ""
                        }
                    )
                } else if confirmation {
                    confirmation_yaml("apologize_and_stop", false)
                } else {
                    CLARIFICATION.into()
                };
                let parsed = format!("{{\"status\":\"{status}\"}}");
                let mut replies = if skill {
                    vec![
                        CLEAR,
                        r#"{"tag":"original"}"#,
                        "script",
                        AMBIGUOUS,
                        QUESTION,
                    ]
                } else {
                    vec![AMBIGUOUS, QUESTION]
                };
                if confirmation {
                    replies.extend([ANSWER, CONFIRM]);
                }
                replies.push(&parsed);
                let f = fixture(&yaml, &["done"], &replies);
                let mut first = f.runner.run("objective", None).await.unwrap();
                if confirmation {
                    first = answer(&f, &first.run, "target one").await.unwrap();
                }
                let done = answer(&f, &first.run, "new topic or no").await.unwrap();
                assert_eq!(
                    done.run.status,
                    TaskRunStatus::Incomplete,
                    "status={status}"
                );
                assert_eq!(done.run.stop_reason.as_deref(), Some("scope_ended"));
                assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
                assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
                assert!(
                    !f.agent
                        .disambiguation_manager()
                        .unwrap()
                        .has_pending_clarification()
                        .await
                );
            }
        }
    }
}

#[tokio::test]
async fn disambig_acceptance_best_guess_exhaustion_keeps_selected_skill_and_processed_context() {
    for skill in [false, true] {
        let yaml = if skill {
            skill_yaml("proceed_with_best_guess")
        } else {
            CLARIFICATION.replace("  clarification:\n    llm: router\n", "  clarification:\n    llm: router\n    max_attempts: 2\n    on_max_attempts: proceed_with_best_guess\n")
        };
        let replies = if skill {
            vec![
                CLEAR,
                r#"{"tag":"original"}"#,
                "script",
                AMBIGUOUS,
                QUESTION,
                UNCLEAR,
                CLEAR,
            ]
        } else {
            vec![AMBIGUOUS, QUESTION, UNCLEAR]
        };
        let f = fixture(&yaml, &["done"], &replies);
        let first = f.runner.run("objective", None).await.unwrap();
        let done = answer(&f, &first.run, "maybe").await.unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(
            f.router.calls.load(Ordering::SeqCst),
            if skill { 7 } else { 3 }
        );
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
        if skill {
            assert!(
                done.final_response
                    .as_ref()
                    .unwrap()
                    .content
                    .contains("original")
            );
        }
        let p = payload(&f, &done.run).await;
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
async fn disambig_acceptance_confirmation_parse_failure_or_drop_cannot_repeat_prior_resolution() {
    for mode in ["error", "drop", "malformed"] {
        let provider = support::FaultProvider::new(
            &[AMBIGUOUS, QUESTION, ANSWER, CONFIRM, "invalid JSON"],
            4,
            mode,
        );
        let mut llms = ai_agents_llm::LLMRegistry::new();
        llms.register(
            "default",
            super::super::runner_tests::RecordingProvider::new(&["done"]),
        );
        llms.register("router", provider.clone());
        let agent = Arc::new(
            AgentBuilder::from_yaml(&confirmation_yaml("apologize_and_stop", false))
                .unwrap()
                .llm_registry(llms)
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "interaction.v1", false).await;
        let runner = runner(agent.clone(), store);
        let first = runner.run("objective", None).await.unwrap();
        let second = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!("target one"),
                },
            )
            .await
            .unwrap();
        let mut resume = Box::pin(runner.resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("yes"),
            },
        ));
        if mode == "drop" {
            tokio::select! {
                result = &mut resume => panic!("parser not held: {result:?}"),
                permit = provider.reached.acquire() => permit.unwrap().forget(),
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("parser not reached"),
            }
            drop(resume);
        } else {
            assert!(resume.await.is_err());
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
        assert!(agent.set_context("blocked", json!(true)).is_err());
        assert!(
            runner
                .resume(
                    &second.run.key.run_id,
                    second.run.revision,
                    TaskResumeInput::UserAnswer {
                        request_id: second.run.pending.as_ref().unwrap().id.clone(),
                        answer: json!("yes")
                    }
                )
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn disambig_acceptance_consumption_and_provider_settlement_receipts_protect_semantic_progress()
 {
    for sqlite in support::backends() {
        for consume in [false, true] {
            for mode in ["positive", "reject", "lost", "drop"] {
                let mut f = fixture(
                    &confirmation_yaml("apologize_and_stop", false),
                    &["done"],
                    &[
                        AMBIGUOUS,
                        QUESTION,
                        ANSWER,
                        CONFIRM,
                        r#"{"status":"confirmed"}"#,
                    ],
                );
                f.store = support::store(f.agent.info().id, "interaction.v1", *sqlite).await;
                let fault = ReceiptStore::checkpoint(
                    f.store.clone(),
                    mode,
                    if consume {
                        json!({"consume":"unset"})
                    } else {
                        json!({"completed":2})
                    },
                );
                f.runner = runner(f.agent.clone(), fault.clone());
                let first = f.runner.run("objective", None).await.unwrap();
                let binding = if consume {
                    json!({"consume": first.run.pending.as_ref().unwrap().id})
                } else {
                    json!({"completed":2})
                };
                // Bind the selector after publication supplies the real request identity, without replacing the retained runner.
                *fault.checkpoint.lock() = Some(binding);
                fault.armed.store(true, Ordering::SeqCst);
                let mut resume = Box::pin(answer(&f, &first.run, "target one"));
                if mode == "positive" {
                    let second = resume.await.unwrap();
                    fault.assert_hit();
                    let done = answer(&f, &second.run, "yes").await.unwrap();
                    assert_eq!(done.run.status, TaskRunStatus::Completed);
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
                    let saved = payload(&f, &first.run).await;
                    assert_eq!(
                        saved
                            .consumed_request_ids
                            .contains(&first.run.pending.as_ref().unwrap().id),
                        !consume || mode != "reject"
                    );
                    assert_eq!(
                        f.router.calls.load(Ordering::SeqCst),
                        if consume { 2 } else { 3 }
                    );
                    assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
                    assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
                    assert!(f.agent.set_context("blocked", json!(true)).is_err());
                    assert!(answer(&f, &first.run, "replay").await.is_err());
                }
            }
        }
    }
}

#[tokio::test]
async fn disambig_acceptance_nested_confirmation_timeout_closes_only_original_leaf() {
    use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};
    for sqlite in support::backends() {
        let yaml = confirmation_yaml("apologize_and_stop", true);
        let child = fixture(&yaml, &["done"], &[AMBIGUOUS, QUESTION, ANSWER, CONFIRM]);
        drop(child.runner);
        let registry = Arc::new(AgentRegistry::new());
        let child_agent = Arc::try_unwrap(child.agent).unwrap_or_else(|_| panic!("child shared"));
        registry
            .register(SpawnedAgent::from_runtime(
                "worker".into(),
                child_agent,
                crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap(),
            ))
            .await
            .unwrap();
        let parent_yaml = "name: Parent\nsystem_prompt: test\nstates:\n  initial: active\n  states:\n    active:\n      delegate: worker\n";
        let main = super::super::runner_tests::RecordingProvider::new(&["unused"]);
        let parent = Arc::new(
            AgentBuilder::from_yaml(parent_yaml)
                .unwrap()
                .llm(main.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap()
                .with_spawner_handles(Arc::new(AgentSpawner::new()), registry.clone()),
        );
        let store = support::store(parent.info().id, "interaction.v1", *sqlite).await;
        let runner = runner(parent.clone(), store);
        let first = runner.run("objective", None).await.unwrap();
        let second = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!("target one"),
                },
            )
            .await
            .unwrap();
        assert_eq!(second.run.status, TaskRunStatus::Paused);
        assert!(registry.remove("worker").await.is_none());
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let done = runner
            .resume(
                &second.run.key.run_id,
                second.run.revision,
                TaskResumeInput::QuestionTimeout {
                    request_id: second.run.pending.as_ref().unwrap().id.clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Incomplete);
        assert_eq!(child.router.calls.load(Ordering::SeqCst), 4);
        assert_eq!(child.main.calls.load(Ordering::SeqCst), 0);
        assert_eq!(main.calls.load(Ordering::SeqCst), 0);
        assert!(registry.remove("worker").await.is_some());
        assert!(parent.set_context("released", json!(true)).is_ok());
    }
}
