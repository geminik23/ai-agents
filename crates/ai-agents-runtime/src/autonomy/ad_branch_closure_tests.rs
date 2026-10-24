use super::*;
use std::collections::BTreeMap;

const REVIEW: &str = "name: Reviewed\nsystem_prompt: test\nllm:\n  default: default\n  router: router\nruntime:\n  optimization:\n    enabled: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\n    speculative_state_transitions: true\nhitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n  states:\n    ready:\n      on_enter: require_approval\nstates:\n  initial: active\n  states:\n    active:\n      transitions:\n        - to: ready\n          when: ready request\n          timing: parallel\n          requires_response: false\n    ready:\n      regenerate_on_enter: false\ntools: []\n";

struct Output(std::path::PathBuf);
impl Drop for Output {
    // Observability exports are isolated per fixture and never touch the checkout or another test's reports.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// Actual observed managed providers share the real runner; the output guard owns any generated report files.
fn observed(main: Arc<Provider>, router: Arc<Provider>) -> (Arc<RuntimeAgent>, Output) {
    observed_yaml(main, router, REVIEW)
}

// Variant fixtures retain the same isolated observation configuration and real builder boundary.
fn observed_yaml(
    main: Arc<Provider>,
    router: Arc<Provider>,
    yaml: &str,
) -> (Arc<RuntimeAgent>, Output) {
    let output = Output(std::env::temp_dir().join(format!("ad-branches-{}", uuid::Uuid::new_v4())));
    let mut config = ai_agents_observability::ObservabilityConfig {
        enabled: true,
        ..Default::default()
    };
    config.export.write_report = false;
    config.export.write_raw_events = true;
    config.export.path = output.0.to_string_lossy().into_owned();
    let manager = ai_agents_observability::ObservabilityManager::new(config);
    let agent = Arc::new(
        crate::AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(main)
            .llm_alias("router", router)
            .observability(manager)
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    (agent, output)
}

// Every deferred original branch has one disposition; consumed losing calls remain reportable rather than refunded or hidden.
fn dispositions(agent: &RuntimeAgent) -> BTreeMap<String, (String, String)> {
    let mut branches = BTreeMap::new();
    for event in agent.observability().unwrap().raw_events() {
        if let (Some(id), Some(status), Some(behavior)) = (
            event.dimensions.get("runtime.branch_id"),
            event.dimensions.get("runtime.branch_status"),
            event.dimensions.get("runtime.commit_behavior"),
        ) {
            let disposition = (status.clone(), behavior.clone());
            if let Some(previous) = branches.insert(id.clone(), disposition.clone()) {
                assert_eq!(
                    previous, disposition,
                    "branch disposition changed after publication"
                );
            }
        }
    }
    branches
}

// Both completion orders and both review outcomes publish the actual selected branch rather than eagerly discarding reusable alternatives.
#[tokio::test]
async fn ad_review_branch_dispositions_follow_actual_adoption() {
    for draft_first in [false, true] {
        for approved in [false, true] {
            let main = Provider::new(&["old done", "ready done"], true);
            let router = Provider::new(&["1"], true);
            let (agent, _output) = observed(main.clone(), router.clone());
            let runner = reviewed_runner(agent.clone());
            let mut work = Box::pin(runner.run("objective", None));
            tokio::select! { result = &mut work => panic!("main did not dispatch: {:?}", result.err()), permit = main.entered.acquire() => permit.unwrap().forget() }
            tokio::select! { result = &mut work => panic!("router did not dispatch: {:?}", result.err()), permit = router.entered.acquire() => permit.unwrap().forget() }
            if draft_first {
                main.release.add_permits(1);
            } else {
                router.release.add_permits(1);
            }
            assert!(futures::poll!(&mut work).is_pending());
            if draft_first {
                router.release.add_permits(1);
            } else {
                main.release.add_permits(1);
            }
            let paused = work.await.unwrap();
            assert_eq!(paused.run.status, TaskRunStatus::Paused);
            assert!(
                dispositions(&agent).is_empty(),
                "review is not yet a commit/discard decision"
            );
            main.release.add_permits(1);
            let result = runner
                .resume(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    crate::autonomy::TaskResumeInput::Approval {
                        request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                        result: if approved {
                            ai_agents_hitl::ApprovalResult::Approved
                        } else {
                            ai_agents_hitl::ApprovalResult::Rejected {
                                reason: Some("stay".into()),
                            }
                        },
                    },
                )
                .await
                .unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(
                result.run.counters.llm_attempts,
                if approved { 3 } else { 2 }
            );
            let branches = dispositions(&agent);

            assert_eq!(
                branches.len(),
                2,
                "missing original branch observation: {branches:?}"
            );
            assert!(
                branches
                    .values()
                    .any(|(status, behavior)| status == "committed"
                        && behavior
                            == if approved {
                                "transition_decision"
                            } else {
                                "final_response"
                            })
            );
            assert!(
                branches
                    .values()
                    .any(|(status, behavior)| status == "discarded"
                        && behavior
                            == if approved {
                                "final_response"
                            } else {
                                "transition_decision"
                            })
            );
        }
    }
}

// Known-safe cancellation retires inert observations without applying the selected transition or executing another provider.
#[tokio::test]
async fn ad_review_cancel_retires_branch_receipts_without_execution() {
    let main = Provider::new(&["old done"], false);
    let router = Provider::new(&["1"], false);
    let (agent, _output) = observed(main.clone(), router.clone());
    let runner = reviewed_runner(agent.clone());
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let result = runner
        .stop_paused(
            &paused.run.key.run_id,
            paused.run.revision,
            "stop reviewed route",
        )
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Incomplete);
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(router.calls.load(Ordering::SeqCst), 1);
    let branches = dispositions(&agent);
    assert_eq!(branches.len(), 2);
    assert!(
        branches.values().all(|(status, _)| status == "cancelled"),
        "{branches:?}"
    );
    assert_eq!(agent.current_state().as_deref(), Some("active"));
}

struct LosingFailure {
    calls: AtomicUsize,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl LLMProvider for LosingFailure {
    // A genuinely dispatched transient failure arrives after local branch admission has closed.
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<ai_agents_core::LLMResponse, LLMError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
            Err(LLMError::RateLimit { retry_after: None })
        } else {
            Ok(ai_agents_core::LLMResponse::new("done", FinishReason::Stop))
        }
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        unreachable!()
    }
    fn provider_name(&self) -> &str {
        "losing transient"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

// Completed failure is not uninvoked work, and closing a losing branch prevents configured retries and provider fallback.
#[tokio::test]
async fn ad_closed_loser_failure_cannot_retry_fallback_or_replay_after_review() {
    for approved in [false, true] {
        let main = Arc::new(LosingFailure {
            calls: AtomicUsize::new(0),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        });
        let fallback = Provider::new(&["forbidden fallback"], false);
        let router = Provider::new(&["1"], false);
        let yaml = format!(
            "{REVIEW}error_recovery:\n  default:\n    max_retries: 2\n    backoff:\n      initial_ms: 0\n      max_ms: 0\n  llm:\n    on_failure:\n      action: fallback_llm\n      fallback_llm: rescue\n"
        );
        let agent = Arc::new(
            crate::AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(main.clone())
                .llm_alias("router", router.clone())
                .llm_alias("rescue", fallback.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        let runner = reviewed_runner(agent);
        let mut work = Box::pin(runner.run("objective", None));
        tokio::select! { result = &mut work => panic!("no actual loser: {:?}", result.err()), permit = main.entered.acquire() => permit.unwrap().forget() }
        assert!(futures::poll!(&mut work).is_pending());
        assert_eq!(router.calls.load(Ordering::SeqCst), 1);
        main.release.add_permits(1);
        let paused = work.await.unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        assert_eq!(main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 0);
        let result = runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                crate::autonomy::TaskResumeInput::Approval {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    result: if approved {
                        ai_agents_hitl::ApprovalResult::Approved
                    } else {
                        ai_agents_hitl::ApprovalResult::Rejected { reason: None }
                    },
                },
            )
            .await;
        if approved {
            let result = result.unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(result.run.counters.llm_attempts, 3);
        } else {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Rate limit exceeded")
            );
            assert_eq!(paused.run.counters.llm_attempts, 2);
        }
        assert_eq!(
            main.calls.load(Ordering::SeqCst),
            if approved { 2 } else { 1 }
        );
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 0);
    }
}

// An optional branch refused by local capacity before invocation remains unfinished regardless of which completed outcome the scheduler consumes first.
#[tokio::test]
async fn ad_uninvoked_skill_before_transition_is_not_a_completed_no_match() {
    let main = Provider::new(&["old draft", "script done"], true);
    let router = Provider::new(&["1", "selected"], true);
    let yaml = format!(
        "{}skills:\n  - id: selected\n    description: work\n    trigger: work\n    steps:\n      - prompt: selected script\n",
        REVIEW
            .replace(
                "max_parallel_runtime_tasks: 2",
                "max_parallel_runtime_tasks: 3"
            )
            .replace(
                "    speculative_state_transitions: true",
                "    speculative_state_transitions: true\n    speculative_skill_routing: true"
            )
    );
    let (agent, _output) = observed_yaml(main.clone(), router.clone(), &yaml);
    let runner = reviewed_runner(agent);
    let mut work = Box::pin(runner.run("objective", None));
    tokio::select! { result = &mut work => panic!("main not dispatched: {:?}", result.err()), permit = main.entered.acquire() => permit.unwrap().forget() }
    tokio::select! { result = &mut work => panic!("transition not dispatched: {:?}", result.err()), permit = router.entered.acquire() => permit.unwrap().forget() }
    assert!(futures::poll!(&mut work).is_pending());
    router.release.add_permits(1);
    assert!(futures::poll!(&mut work).is_pending());
    main.release.add_permits(1);
    let paused = work.await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(router.calls.load(Ordering::SeqCst), 1);
    main.release.add_permits(1);
    router.release.add_permits(1);
    let result = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            crate::autonomy::TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Rejected { reason: None },
            },
        )
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(router.calls.load(Ordering::SeqCst), 2);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.run.counters.llm_attempts, 4);
    assert!(
        main.requests
            .lock()
            .iter()
            .flatten()
            .any(|message| message.content.contains("selected script"))
    );
}

// Dropping while a dispatched loser drains retires observations only; actual effect reservations and the runtime owner remain protected.
#[tokio::test]
async fn ad_drop_during_speculative_drain_publishes_interrupted_observations() {
    let main = Provider::new(&["loser"], true);
    let router = Provider::new(&["1"], false);
    let (agent, _output) = observed(main.clone(), router);
    let runner = reviewed_runner(agent.clone());
    let mut work = Box::pin(runner.run("objective", None));
    tokio::select! { result = &mut work => panic!("no actual loser: {:?}", result.err()), permit = main.entered.acquire() => permit.unwrap().forget() }
    assert!(futures::poll!(&mut work).is_pending());
    drop(work);
    assert!(agent.chat("foreign").await.is_err());
    let branches = dispositions(&agent);
    assert_eq!(branches.len(), 2, "{branches:?}");
    assert!(branches.values().all(|(status, _)| status == "cancelled"));
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
}

// A dropped resumed review can interrupt an exit provider before any second pause; the live receipt guard must still retire its observations.
#[tokio::test]
async fn ad_drop_during_reviewed_exit_keeps_custody_and_retires_observations() {
    let yaml = REVIEW.replace(
        "    active:\n",
        "    active:\n      on_exit:\n        - prompt: exit work\n",
    );
    let main = Provider::new(&["old done", "exit done"], true);
    main.release.add_permits(1);
    let (agent, _output) = observed_yaml(main.clone(), Provider::new(&["1"], false), &yaml);
    let runner = reviewed_runner(agent.clone());
    let review = runner.run("objective", None).await.unwrap();
    main.entered.acquire().await.unwrap().forget();
    let response = crate::autonomy::TaskResumeInput::Approval {
        request_id: review.run.pending.as_ref().unwrap().id.clone(),
        result: ai_agents_hitl::ApprovalResult::Approved,
    };
    let mut work = Box::pin(runner.resume(&review.run.key.run_id, review.run.revision, response));
    tokio::select! { result = &mut work => panic!("exit work did not wait: {:?}", result.err()), permit = main.entered.acquire() => permit.unwrap().forget() }
    drop(work);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(agent.current_state().as_deref(), Some("active"));
    assert!(agent.chat("foreign").await.is_err());
    let branches = dispositions(&agent);
    assert_eq!(branches.len(), 2);
    assert!(branches.values().all(|(status, _)| status == "cancelled"));
}

// Exit timeout cannot commit the transition; entry timeout cannot erase its prior commit or leave branch observations hidden.
#[tokio::test]
async fn ad_review_action_question_timeout_preserves_commit_and_observations() {
    for entering in [false, true] {
        let base = REVIEW
            .replace("hitl:\n", "hitl:\n  default_timeout_seconds: 20\n")
            .replace("tools: []", "tools: [ask_user]");
        let action = "      on_exit:\n        - tool: ask_user\n          args: {question: Choice, options: [yes], allow_other: false, timeout_seconds: 0}\n";
        let yaml = if entering {
            base.replace(
                "      regenerate_on_enter: false",
                &format!(
                    "      regenerate_on_enter: false\n{}",
                    action.replace("on_exit", "on_enter")
                ),
            )
        } else {
            base.replace("    active:\n", &format!("    active:\n{action}"))
        };
        let main = Provider::new(&["old done", "ready done"], false);
        let router = Provider::new(&["1"], false);
        let (agent, _output) = observed_yaml(main.clone(), router, &yaml);
        let runner = reviewed_runner(agent.clone());
        let review = runner.run("objective", None).await.unwrap();
        let question = runner
            .resume(
                &review.run.key.run_id,
                review.run.revision,
                crate::autonomy::TaskResumeInput::Approval {
                    request_id: review.run.pending.as_ref().unwrap().id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
                },
            )
            .await
            .unwrap();
        assert_eq!(question.run.status, TaskRunStatus::Paused);

        let result = runner
            .resume(
                &question.run.key.run_id,
                question.run.revision,
                crate::autonomy::TaskResumeInput::QuestionTimeout {
                    request_id: question.run.pending.as_ref().unwrap().id.clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(result.run.status, TaskRunStatus::Incomplete);
        assert_eq!(agent.state_history().len(), usize::from(entering));
        let branches = dispositions(&agent);
        assert_eq!(branches.len(), 2);
        assert!(
            branches
                .values()
                .any(|(status, behavior)| behavior == "transition_decision"
                    && status == if entering { "committed" } else { "cancelled" })
        );
        assert_eq!(main.calls.load(Ordering::SeqCst), 1);
    }
}

// Actual standalone speculative reasoning uses one shared ledger for the losing draft, mode judge and selected reasoning work.
#[tokio::test]
async fn ad_speculative_reasoning_modes_share_accounting() {
    for mode in ["none", "cot", "react", "plan_and_execute"] {
        let plan = r#"{"steps":[{"id":"one","description":"Verify","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
        let replies: Vec<&str> = if mode == "plan_and_execute" {
            vec!["discarded plain", plan, "thought done", "final done"]
        } else {
            vec!["plain done", "reasoned done"]
        };
        let main = Provider::new(&replies, false);
        let router = Provider::new(&[mode], false);
        let yaml = "name: Reasoning\nsystem_prompt: test\nllm:\n  default: default\n  router: router\nreasoning:\n  mode: auto\n  judge_llm: router\n  planning:\n    planner_llm: default\n    reflection:\n      enabled: false\nruntime:\n  optimization:\n    enabled: true\n    speculative_reasoning_auto: true\n    max_speculative_llm_calls_per_turn: 2\n    max_parallel_runtime_tasks: 2\ntools: []\n";
        let agent = Arc::new(
            crate::AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(main.clone())
                .llm_alias("router", router.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        let (runner, _) = runner(agent, 8);
        let result = runner.run("analyze", None).await.unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{mode}: {:?}",
            result.run.stop_reason
        );
        assert_eq!(router.calls.load(Ordering::SeqCst), 1, "{mode}");
        assert_eq!(
            result.run.counters.llm_attempts,
            (main.calls.load(Ordering::SeqCst) + 1) as u64
        );
        assert_eq!(result.run.counters.tool_attempts, 0);
        assert!(main.calls.load(Ordering::SeqCst) >= if mode == "none" { 1 } else { 2 });
    }
}

// The new reviewed cursor cannot turn lost provider/pause acknowledgements into completion or permission to replay work.
#[tokio::test]
async fn ad_speculative_settlement_and_pause_ack_faults_keep_custody() {
    use crate::autonomy::continuation_test_support as support;
    for &sqlite in support::backends() {
        for pause in [false, true] {
            for mode in ["reject", "lost"] {
                let main = Provider::new(&["old done", "ready done"], false);
                let router = Provider::new(&["1"], false);
                let (agent, _output) = observed(main.clone(), router.clone());
                let store = support::store(agent.info().id, "ad-fault", sqlite).await;
                let fault = if pause {
                    support::ReceiptStore::new(store, mode, TaskRunStatus::Paused)
                } else {
                    support::ReceiptStore::checkpoint(
                        store,
                        mode,
                        serde_json::json!({"completed":0}),
                    )
                };
                fault.armed.store(true, Ordering::SeqCst);
                let config = AutonomyConfig {
                    defaults: AutonomyProfile {
                        enabled: Some(true),
                        max_turns: Some(2),
                        max_llm_calls: Some(8),
                        completion: Some(CompletionGate::ResponseContains("done".into())),
                        hitl: Some(crate::autonomy::AutonomyHitlConfig {
                            on_approval_required: Some(
                                crate::autonomy::InteractionAction::PauseRun,
                            ),
                            on_user_question: Some(crate::autonomy::InteractionAction::PauseRun),
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let runner = AutonomyRunner::try_new(
                    agent.clone(),
                    config,
                    Default::default(),
                    fault.clone(),
                    "ad-fault".into(),
                )
                .unwrap();
                let result = runner.run("objective", None).await;
                fault.assert_hit();
                assert!(
                    !result
                        .as_ref()
                        .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
                );
                let before =
                    main.calls.load(Ordering::SeqCst) + router.calls.load(Ordering::SeqCst);
                assert!(agent.chat("foreign").await.is_err());
                assert!(runner.run("replay", None).await.is_err());
                assert_eq!(
                    main.calls.load(Ordering::SeqCst) + router.calls.load(Ordering::SeqCst),
                    before
                );
            }
        }
    }
}

// A discarded draft may contain the entire completion marker; only the adopted response can satisfy the gate.
#[tokio::test]
async fn ad_discarded_draft_cannot_supply_completion_credit() {
    let main = Provider::new(&["LOSER_ONLY", "not completed"], false);
    let router = Provider::new(&["1"], false);
    let (agent, _output) = observed(main.clone(), router);
    let store =
        Arc::new(ScopedTaskRunStore::in_memory(agent.info().id, None, "ad-loser".into()).unwrap());
    let config = AutonomyConfig {
        defaults: AutonomyProfile {
            enabled: Some(true),
            max_turns: Some(1),
            max_llm_calls: Some(3),
            completion: Some(CompletionGate::ResponseContains("LOSER_ONLY".into())),
            hitl: Some(crate::autonomy::AutonomyHitlConfig {
                on_approval_required: Some(crate::autonomy::InteractionAction::PauseRun),
                on_user_question: Some(crate::autonomy::InteractionAction::PauseRun),
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        Default::default(),
        store,
        "ad-loser".into(),
    )
    .unwrap();
    let paused = runner.run("objective", None).await.unwrap();
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            crate::autonomy::TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::LimitReached);
    assert_eq!(done.run.counters.llm_attempts, 3);
    assert!(
        agent
            .memory
            .get_messages(None)
            .await
            .unwrap()
            .iter()
            .all(|message| !message.content.contains("LOSER_ONLY"))
    );
}
