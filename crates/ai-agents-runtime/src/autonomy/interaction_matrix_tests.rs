use super::*;
use crate::Agent;
use crate::spawner::{AgentRegistry, AgentSpawner, SpawnedAgent};

#[tokio::test]
async fn interaction_confirmation_repause_preserves_parsed_resolution() {
    let yaml = format!(
        "{CLARIFICATION}states:\n  initial: active\n  states:\n    active:\n      disambiguation:\n        require_confirmation: true\n"
    );
    let f = fixture(
        &yaml,
        &["done"],
        &[
            AMBIGUOUS,
            QUESTION,
            ANSWER,
            r#"{"question":"Confirm target one?"}"#,
            r#"{"status":"confirmed"}"#,
        ],
    );
    let first = f.runner.run("objective", None).await.unwrap();
    let second = f
        .runner
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
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 4);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
    assert_ne!(
        first.run.pending.as_ref().unwrap().id,
        second.run.pending.as_ref().unwrap().id
    );
    assert!(
        f.runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!("yes")
                }
            )
            .await
            .is_err()
    );
    let done = f
        .runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("yes"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 5);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    assert_eq!(done.run.counters.turns, 1);
}

// Exhaustion review must preserve the objective rather than reinterpret the final unclear answer as new work.
#[tokio::test]
async fn interaction_escalation_preserves_original_input_and_rejection_stops() {
    for approved in [false, true] {
        let yaml = format!("{}hitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n", CLARIFICATION.replace("  clarification:\n    llm: router\n", "  clarification:\n    llm: router\n    max_attempts: 2\n    on_max_attempts: escalate\n"));
        let f = fixture(
            &yaml,
            &["done"],
            &[AMBIGUOUS, QUESTION, r#"{"status":"unclear"}"#],
        );
        let first = f.runner.run("original objective", None).await.unwrap();
        let review = f
            .runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!("maybe"),
                },
            )
            .await
            .unwrap();
        assert_eq!(review.run.status, TaskRunStatus::Paused);
        assert!(matches!(
            review.run.pending.as_ref().unwrap().kind,
            TaskPendingKind::Approval
        ));
        let payload: TaskCheckpointPayload = serde_json::from_value(
            f.store
                .load(&review.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let batch: TaskBatchState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|a| a.id == "runtime.tool_batch")
                .unwrap()
                .state
                .clone(),
        )
        .unwrap();
        let super::super::interaction::TaskInteractionKind::Escalation { input, .. } =
            &batch.interaction.as_ref().unwrap().kind
        else {
            panic!("missing escalation cursor")
        };
        assert_eq!(input, "original objective");
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 3);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
        let done = respond(
            &f,
            &review.run,
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
        assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 3);
    }
}

// Apologize-and-stop exhausts the task even when its localized text happens to satisfy a completion predicate.
#[tokio::test]
async fn interaction_exhaustion_stops_task_before_completion_gate() {
    let yaml = CLARIFICATION.replace("  clarification:\n    llm: router\n", "  clarification:\n    llm: router\n    max_attempts: 2\n    on_max_attempts: apologize_and_stop\n");
    let f = fixture(
        &yaml,
        &["done apology"],
        &[
            AMBIGUOUS,
            QUESTION,
            r#"{"status":"unclear"}"#,
            "done apology",
        ],
    );
    let first = f.runner.run("original objective", None).await.unwrap();
    let done = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("maybe"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Incomplete);
    assert_eq!(
        done.run.stop_reason.as_deref(),
        Some("disambiguation_exhausted")
    );
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 4);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    assert_eq!(done.run.counters.turns, 1);
}

// Missing human escalation cannot turn a transfer message into evidence that the task objective completed.
#[tokio::test]
async fn interaction_unavailable_escalation_cannot_pass_completion_gate() {
    let yaml = CLARIFICATION.replace(
        "  clarification:\n    llm: router\n",
        "  clarification:\n    llm: router\n    max_attempts: 2\n    on_max_attempts: escalate\n",
    );
    let f = fixture(
        &yaml,
        &["done transfer"],
        &[
            AMBIGUOUS,
            QUESTION,
            r#"{"status":"unclear"}"#,
            "done transfer",
        ],
    );
    let first = f.runner.run("objective", None).await.unwrap();
    let done = f
        .runner
        .resume(
            &first.run.key.run_id,
            first.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: first.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("maybe"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Incomplete);
    assert_eq!(
        done.run.stop_reason.as_deref(),
        Some("interaction_unavailable")
    );
    assert_eq!(done.run.counters.turns, 1);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
}

// Host-approved escalation returns to the selected skill with its original processed context and no new routing.
#[tokio::test]
async fn interaction_skill_escalation_keeps_processed_context_and_selected_skill() {
    for repause in [false, true] {
        let yaml = format!("{}hitl:\n  message_language:\n    strategy: explicit\n    explicit: en\nprocess:\n  input:\n    - type: extract\n      config:\n        llm: router\n        schema:\n          tag:\n            type: string\n            required: true\n        store_in_context: extracted\n  output:\n    - type: format\n      config:\n        template: '{{{{ response }}}} : {{{{ context.extracted }}}}'\nskills:\n  - id: script\n    description: Finish task\n    trigger: Work requested\n    disambiguation:\n      enabled: true\n      required_clarity: [target]\n    steps:\n      - prompt: 'Finish {{{{ input }}}}'\n", CLARIFICATION.replace("  clarification:\n    llm: router\n", "  clarification:\n    llm: router\n    max_attempts: 2\n    on_max_attempts: escalate\n"));
        let yaml = if repause {
            yaml.replace(
                "hitl:\n",
                "hitl:\n  tools:\n    probe:\n      require_approval: true\n",
            )
            .replace(
                "    steps:\n      - prompt:",
                "    steps:\n      - tool: probe\n        args: {}\n      - prompt:",
            )
            .replace(
                "system_prompt: test\n",
                "system_prompt: test\ntools: [probe]\n",
            )
        } else {
            yaml
        };
        let f = fixture(
            &yaml,
            &["done"],
            &[
                r#"{"is_ambiguous":false,"confidence":1.0}"#,
                r#"{"tag":"original"}"#,
                "script",
                AMBIGUOUS,
                QUESTION,
                r#"{"status":"unclear"}"#,
            ],
        );
        let first = f.runner.run("original objective", None).await.unwrap();
        let review = f
            .runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!("maybe"),
                },
            )
            .await
            .unwrap();
        let mut done = respond(&f, &review.run, ApprovalResult::Approved).await;
        if repause {
            assert_eq!(done.run.status, TaskRunStatus::Paused);
            assert_eq!(f.router.calls.load(Ordering::SeqCst), 6);
            assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
            done = respond(&f, &done.run, ApprovalResult::Approved).await;
        }
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert!(done.final_response.unwrap().content.contains("original"));
        assert_eq!(f.effects.load(Ordering::SeqCst), usize::from(repause));
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 6);
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn interaction_task_entry_cannot_consume_ordinary_clarification() {
    let f = fixture(CLARIFICATION, &["done"], &[AMBIGUOUS, QUESTION, ANSWER]);
    let question = f.agent.chat("ordinary objective").await.unwrap();
    assert_eq!(question.content, "Which target?");
    assert!(
        f.runner
            .run("unrelated task objective", None)
            .await
            .is_err()
    );
    assert!(f.store.list().await.unwrap().is_empty());
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.agent.chat("target one").await.unwrap().content, "done");
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn interaction_missing_question_policy_stops_without_bypassing_clarity() {
    let mut f = fixture(CLARIFICATION, &["done"], &[AMBIGUOUS, QUESTION]);
    f.runner = AutonomyRunner::try_new(
        f.agent.clone(),
        super::super::runner_tests::config(16),
        Default::default(),
        f.store.clone(),
        "interaction.v1".into(),
    )
    .unwrap();
    let result = f.runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Incomplete);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn interaction_post_transition_regeneration_keeps_one_finalization_and_prior_work() {
    let yaml = STATE.replace("regenerate_on_enter: false", "regenerate_on_enter: true");
    let marker = ai_agents_core::encode_native_tool_call_markers(
        &[ai_agents_core::ToolCall {
            id: "regenerated".into(),
            name: "probe".into(),
            arguments: json!({}),
        }],
        None,
    )
    .unwrap();
    let f = fixture(&yaml, &["done draft", &marker, "done final"], &[]);
    let paused = f.runner.run("objective", None).await.unwrap();
    let done = respond(&f, &paused.run, ApprovalResult::Approved).await;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.final_response.unwrap().content, "done final");
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 3);
    assert_eq!(f.effects.load(Ordering::SeqCst), 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    assert_eq!(f.agent.state_history().len(), 1);
}

#[tokio::test]
async fn interaction_skill_clarification_retains_processed_context_without_rerouting() {
    let yaml = format!(
        "{CLARIFICATION}process:\n  input:\n    - type: extract\n      config:\n        llm: router\n        schema:\n          tag:\n            type: string\n            required: true\n        store_in_context: extracted\n  output:\n    - type: format\n      config:\n        template: '{{{{ response }}}} : {{{{ context.extracted }}}}'\nskills:\n  - id: script\n    description: Finish task\n    trigger: Work requested\n    disambiguation:\n      enabled: true\n      required_clarity: [target]\n    steps:\n      - prompt: 'Finish {{{{ input }}}}'\n"
    );
    let clear = r#"{"is_ambiguous":false,"confidence":1.0}"#;
    let f = fixture(
        &yaml,
        &["done"],
        &[
            clear,
            r#"{"tag":"original"}"#,
            "script",
            AMBIGUOUS,
            QUESTION,
            ANSWER,
            clear,
        ],
    );
    let paused = f.runner.run("objective", None).await.unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 5);
    let done = f
        .runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                answer: json!("target one"),
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
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 7);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    assert!(done.final_response.unwrap().content.contains("original"));
    let payload: TaskCheckpointPayload = serde_json::from_value(
        f.store
            .load(&done.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
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
}

#[tokio::test]
async fn interaction_pre_response_approval_reuses_processed_input_before_model_work() {
    for approved in [false, true] {
        let yaml = format!("{}runtime:\n  optimization:\n    enabled: true\n    pre_response_deterministic_transitions: true\nprocess:\n  input:\n    - type: transform\n      config:\n        llm: router\n        prompt: Prepare input\n", STATE.replace("          guard: 'true'", "          guard: 'true'\n          timing: pre_response\n          requires_response: false").replace("hitl:\n", "hitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n"));
        let f = fixture(&yaml, &["done"], &["processed once"]);
        let paused = f.runner.run("objective", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
        let done = respond(
            &f,
            &paused.run,
            if approved {
                ApprovalResult::Approved
            } else {
                ApprovalResult::Rejected { reason: None }
            },
        )
        .await;
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.router.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
        assert_eq!(
            f.agent.current_state().as_deref(),
            Some(if approved { "ready" } else { "active" })
        );
    }
}

struct NestedChild {
    main: Arc<super::super::runner_tests::RecordingProvider>,
    finalized: Arc<AtomicUsize>,
}

// Registered child fixtures share the real controller ledger and retained catalogue, including non-tool leaf positions.
async fn nested_fixture(
    state_child: bool,
    review_parent: bool,
) -> (
    NestedChild,
    Arc<crate::RuntimeAgent>,
    AutonomyRunner,
    Arc<ScopedTaskRunStore>,
    Arc<AtomicUsize>,
) {
    let child_yaml = if state_child { STATE } else { CLARIFICATION };
    let child = fixture(
        child_yaml,
        &["done"],
        if state_child {
            &[]
        } else {
            &[AMBIGUOUS, QUESTION, ANSWER]
        },
    );
    drop(child.runner);
    let runtime = Arc::try_unwrap(child.agent)
        .unwrap_or_else(|_| panic!("fixture unexpectedly shared child runtime"));
    let registry = Arc::new(AgentRegistry::new());
    registry
        .register(SpawnedAgent::from_runtime(
            "worker".into(),
            runtime,
            crate::spec::AgentSpec::from_yaml_strict(child_yaml).unwrap(),
        ))
        .await
        .unwrap();

    let parent_yaml = if review_parent {
        STATE.replace(
            "      transitions:\n",
            "      delegate: worker\n      transitions:\n",
        )
    } else {
        "name: Parent\nsystem_prompt: parent\nstates:\n  initial: active\n  states:\n    active:\n      delegate: worker\n".into()
    };
    let parent = fixture(&parent_yaml, &["unused"], &[]);
    drop(parent.runner);
    let runtime = Arc::try_unwrap(parent.agent)
        .unwrap_or_else(|_| panic!("fixture unexpectedly shared parent runtime"))
        .with_spawner_handles(Arc::new(AgentSpawner::new()), registry);
    let parent_agent = Arc::new(runtime);
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(parent_agent.info().id, None, "nested.interaction.v1".into())
            .unwrap(),
    );
    let mut config = super::super::runner_tests::config(16);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        parent_agent.clone(),
        config,
        Default::default(),
        store.clone(),
        "nested.interaction.v1".into(),
    )
    .unwrap();
    let finalized = parent.finalized.clone();
    let child_stub = NestedChild {
        main: child.main,
        finalized: child.finalized,
    };
    (child_stub, parent_agent, runner, store, finalized)
}

#[tokio::test]
async fn interaction_nested_non_tool_leaf_resumes_through_exact_parent() {
    for state_child in [false, true] {
        let (child, parent, runner, _, finalized) =
            Box::pin(nested_fixture(state_child, false)).await;
        let paused = runner.run("objective", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        let request_id = paused.run.pending.as_ref().unwrap().id.clone();
        let input = if state_child {
            TaskResumeInput::Approval {
                request_id,
                result: ApprovalResult::Approved,
            }
        } else {
            TaskResumeInput::UserAnswer {
                request_id,
                answer: json!("target one"),
            }
        };
        let done = runner
            .resume(&paused.run.key.run_id, paused.run.revision, input)
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(child.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(child.finalized.load(Ordering::SeqCst), 1);
        assert_eq!(finalized.load(Ordering::SeqCst), 1);
        assert!(parent.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn interaction_parent_review_after_child_repause_keeps_completed_child() {
    let (child, _, runner, _, finalized) = Box::pin(nested_fixture(false, true)).await;
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
    assert_eq!(child.main.calls.load(Ordering::SeqCst), 1);
    let done = runner
        .resume(
            &second.run.key.run_id,
            second.run.revision,
            TaskResumeInput::Approval {
                request_id: second.run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(child.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(child.finalized.load(Ordering::SeqCst), 1);
    assert_eq!(finalized.load(Ordering::SeqCst), 1);
    assert_eq!(
        done.final_response.unwrap().metadata.unwrap()["orchestration"]["type"],
        json!("delegate")
    );
}
