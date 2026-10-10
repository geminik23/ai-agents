use super::*;
use crate::Agent;
use ai_agents_core::{Tool, ToolCall, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct ChildProbe {
    child: Arc<crate::RuntimeAgent>,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for ChildProbe {
    fn id(&self) -> &str {
        "child_probe"
    }
    fn name(&self) -> &str {
        "child_probe"
    }
    fn description(&self) -> &str {
        "Invoke a read-only task participant"
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
        match self.child.chat("child objective").await {
            Ok(response) => ToolResult::ok(response.content),
            Err(error) => ToolResult::error(error.to_string()),
        }
    }
}

// Completed child ownership and operation ordinals survive a coordinating pause and a later legitimate child call.
#[tokio::test]
async fn foreground_pause_retains_completed_child_owners_and_operation_identity() {
    let child_provider = super::runner_tests::RecordingProvider::new(&["child done", "child done"]);
    let child = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("child")
            .llm(child_provider.clone())
            .build()
            .unwrap(),
    );
    let first = ai_agents_core::encode_native_tool_call_markers(
        &[
            ToolCall {
                id: "child-first".into(),
                name: "child_probe".into(),
                arguments: json!({}),
            },
            ToolCall {
                id: "approve".into(),
                name: "approved".into(),
                arguments: json!({}),
            },
        ],
        None,
    )
    .unwrap();
    let later = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "child-later".into(),
            name: "child_probe".into(),
            arguments: json!({}),
        }],
        None,
    )
    .unwrap();
    let provider = super::runner_tests::RecordingProvider::new(&[&first, &later, "done"]);
    let child_calls = Arc::new(AtomicUsize::new(0));
    let approved_calls = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(crate::AgentBuilder::from_yaml("name: ChildPause\nsystem_prompt: parent\ntools: [child_probe, approved]\nhitl:\n  tools:\n    approved:\n      require_approval: true\n").unwrap()
        .llm(provider.clone())
        .tool(Arc::new(ChildProbe { child: child.clone(), calls: child_calls.clone() }))
        .tool(Arc::new(Probe { id: "approved", calls: approved_calls.clone() }))
        .auto_configure_features().unwrap().approval_handler(Arc::new(UnexpectedApproval)).build().unwrap());
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "child-pause.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(5);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: None,
    });
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        Default::default(),
        store.clone(),
        "child-pause.v1".into(),
    )
    .unwrap();
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(child_calls.load(Ordering::SeqCst), 1);
    assert_eq!(child_provider.calls.load(Ordering::SeqCst), 1);
    assert!(child.chat("foreign").await.is_err());
    let completed = runner
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
    assert_eq!(completed.run.status, TaskRunStatus::Completed);
    assert_eq!(completed.run.counters.llm_attempts, 5);
    assert_eq!(child_calls.load(Ordering::SeqCst), 2);
    assert_eq!(child_provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(approved_calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&completed.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(payload.children.len(), 2);
    assert_ne!(payload.children[0].child_id, payload.children[1].child_id);
    assert!(payload.children.iter().all(|child| child.result.is_some()));
    assert!(child.chat("ordinary").await.is_ok());
}

struct Probe {
    id: &'static str,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for Probe {
    fn id(&self) -> &str {
        self.id
    }
    fn name(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        "Count actual invocation"
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

// The handler must never be called before an acknowledged pause is returned to the host.
struct UnexpectedApproval;
#[async_trait]
impl ai_agents_hitl::ApprovalHandler for UnexpectedApproval {
    async fn request_approval(
        &self,
        _: ai_agents_hitl::ApprovalRequest,
    ) -> ai_agents_hitl::ApprovalResult {
        panic!("task suspension invoked the ordinary approval handler")
    }
}

// Task questions must use the acknowledged bridge instead of an ordinary awaited handler.
struct UnexpectedQuestion;
#[async_trait]
impl ai_agents_tools::QuestionHandler for UnexpectedQuestion {
    async fn ask_question(
        &self,
        _: ai_agents_tools::QuestionRequest,
    ) -> ai_agents_tools::QuestionResponse {
        panic!("task suspension invoked the ordinary question handler")
    }
}

// The later completed sibling exposes any accidental suffix replay during question resume.
fn question_fixture(args: Value) -> Fixture {
    question_fixture_options(args, false)
}

// Approval modifications must remain bound to the final question without rewriting the committed model call.
fn question_fixture_options(args: Value, approved: bool) -> Fixture {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let calls = vec![
        ToolCall {
            id: "question-call".into(),
            name: "ask_user".into(),
            arguments: args,
        },
        ToolCall {
            id: "sibling-call".into(),
            name: "second".into(),
            arguments: json!({}),
        },
    ];
    let marker = ai_agents_core::encode_native_tool_call_markers(&calls, None).unwrap();
    let provider = super::runner_tests::RecordingProvider::new(&[&marker, "done"]);
    let handler: Arc<dyn ai_agents_tools::QuestionHandler> = Arc::new(UnexpectedQuestion);
    let builder = if approved {
        crate::AgentBuilder::from_yaml("name: QuestionApproval\nsystem_prompt: test\ntools: [ask_user, second]\nhitl:\n  tools:\n    ask_user:\n      require_approval: true\n").unwrap().auto_configure_features().unwrap().approval_handler(Arc::new(UnexpectedApproval))
    } else {
        crate::AgentBuilder::new().system_prompt("test")
    };
    let agent = Arc::new(
        builder
            .llm(provider.clone())
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(Some(handler))),
            )))
            .tool(Arc::new(Probe {
                id: "second",
                calls: second.clone(),
            }))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "question-test.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(3);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: if approved {
            Some(InteractionAction::PauseRun)
        } else {
            None
        },
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        Default::default(),
        store.clone(),
        "question-test.v1".into(),
    )
    .unwrap();
    (agent, runner, store, first, second, provider)
}

// Parking is a non-invocation and must leave the one allowed security rate slot for answer execution.
#[tokio::test]
async fn parked_question_does_not_consume_security_rate_admission() {
    let (agent, runner, store, _, _, _) =
        question_fixture(json!({"question":"Select", "options":["yes"]}));
    let mut security = ai_agents_tools::ToolSecurityConfig {
        enabled: true,
        fail_closed: true,
        ..Default::default()
    };
    security.tools.insert(
        "ask_user".into(),
        ai_agents_tools::ToolPolicyConfig {
            rate_limit: Some(1),
            ..Default::default()
        },
    );
    agent.runtime_control().set_tool_security(security);
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let completed = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.unwrap().id,
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&completed.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    let question = payload
        .evidence
        .tool_calls
        .iter()
        .find(|record| record.record.canonical_id == "ask_user")
        .unwrap();
    assert!(question.record.executed && question.record.success);
}

// The answer is reviewed against modified arguments while shared approval replay starts from the unchanged original call.
#[tokio::test]
async fn approval_modified_question_resumes_from_its_parked_argument_shape() {
    let (_, runner, store, _, second, provider) = question_fixture_options(
        json!({"question":"Original", "options":["old"], "allow_other":false}),
        true,
    );
    let approval = runner.run("objective", None).await.unwrap();
    let question = runner
        .resume(
            &approval.run.key.run_id,
            approval.run.revision,
            TaskResumeInput::Approval {
                request_id: approval.run.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([
                        ("question".into(), json!("Revised")),
                        ("options".into(), json!(["new"])),
                    ]),
                },
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        question.run.pending.as_ref().unwrap().kind,
        TaskPendingKind::UserQuestion
    ));
    let result = runner
        .resume(
            &question.run.key.run_id,
            question.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: question.run.pending.unwrap().id,
                answer: json!({"answered":true,"selected":["new"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .any(|message| message.role == ai_agents_core::Role::Function
                && message.content.contains("new"))
    );
}

// A wrong answer or response kind must leave both durable revision and already completed results untouched.
#[tokio::test]
async fn question_pause_validates_shape_before_claim_and_reuses_completed_sibling() {
    let (agent, runner, store, _, second, provider) = question_fixture(
        json!({"question":"Select", "options":["yes", "no"], "allow_other":false}),
    );
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(paused.run.counters.tool_attempts, 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    let pending = paused.run.pending.unwrap();
    assert!(matches!(pending.kind, TaskPendingKind::UserQuestion));
    assert!(agent.chat("unrelated").await.is_err());
    for input in [
        TaskResumeInput::Approval {
            request_id: pending.id.clone(),
            result: ai_agents_hitl::ApprovalResult::Approved,
        },
        TaskResumeInput::UserAnswer {
            request_id: pending.id.clone(),
            answer: json!({"answered":true,"selected":["invented"]}),
        },
        TaskResumeInput::UserAnswer {
            request_id: pending.id.clone(),
            answer: json!({"answered":true,"selected":["yes","no"]}),
        },
        TaskResumeInput::UserAnswer {
            request_id: pending.id.clone(),
            answer: json!({"answered":true,"other_text":"free"}),
        },
    ] {
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
    }
    let completed = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: pending.id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(completed.run.status, TaskRunStatus::Completed);
    assert_eq!(completed.run.counters.tool_attempts, 2);
    assert_eq!(completed.run.counters.turns, 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&completed.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        payload
            .runtime
            .snapshot
            .memory
            .messages
            .iter()
            .any(|message| message.content.contains("yes")
                && message.role == ai_agents_core::Role::Function),
        "{:?}",
        payload.runtime.snapshot.memory.messages
    );
    assert!(
        runner
            .resume(
                &completed.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: pending.id,
                    answer: json!({"answered":true,"selected":["yes"]})
                }
            )
            .await
            .is_err()
    );
}

// Cancellation retires the question and closes native history without polling its handler or another provider.
#[tokio::test]
async fn parked_question_can_be_cancelled_without_host_invocation() {
    let (agent, runner, _, _, second, provider) =
        question_fixture(json!({"question":"Select", "options":["yes"]}));
    let paused = runner.run("objective", None).await.unwrap();
    let cancelled = runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(agent.chat("ordinary").await.is_ok());
}

// A timer receipt is accepted only after the original expiry and never polls a pending implementation.
#[tokio::test]
async fn expired_question_timeout_closes_history_without_an_extra_call() {
    let (_, runner, _, _, second, provider) =
        question_fixture(json!({"question":"Select", "options":["yes"], "timeout_seconds":0}));
    let paused = runner.run("objective", None).await.unwrap();
    let stopped = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::QuestionTimeout {
                request_id: paused.run.pending.unwrap().id,
            },
        )
        .await
        .unwrap();
    assert_eq!(stopped.run.status, TaskRunStatus::Incomplete);
    assert_eq!(stopped.run.stop_reason.as_deref(), Some("question_timeout"));
    assert_eq!(stopped.run.counters.tool_attempts, 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// An early timer receipt cannot replace a valid question response or advance storage revision.
#[tokio::test]
async fn early_question_timeout_does_not_claim() {
    let (_, runner, store, _, _, _) =
        question_fixture(json!({"question":"Select", "options":["yes"], "timeout_seconds":600}));
    let paused = runner.run("objective", None).await.unwrap();
    assert!(
        runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::QuestionTimeout {
                    request_id: paused.run.pending.unwrap().id
                }
            )
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
}

// The original question timeout cannot be extended by a later answer receipt.
#[tokio::test]
async fn late_question_answer_cannot_claim_or_invoke() {
    let (_, runner, store, _, _, provider) =
        question_fixture(json!({"question":"Select", "options":["yes"], "timeout_seconds":0}));
    let paused = runner.run("objective", None).await.unwrap();
    assert!(matches!(
        runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: paused.run.pending.unwrap().id,
                    answer: json!({"answered":true,"selected":["yes"]})
                }
            )
            .await,
        Err(ai_agents_core::AgentError::HITLTimeout)
    ));
    assert_eq!(
        store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        paused.run.revision
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

type Fixture = (
    Arc<crate::RuntimeAgent>,
    AutonomyRunner,
    Arc<ScopedTaskRunStore>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<super::runner_tests::RecordingProvider>,
);

// Native call IDs are fixed inputs so out-of-order sibling results can be tested without model parsing.
fn fixture(approved: bool, two_calls: bool) -> Fixture {
    fixture_with_conditions(approved, two_calls, false)
}

// Conditional approval exercises multiple gates on one logical tool action.
fn fixture_with_conditions(approved: bool, two_calls: bool, conditions: bool) -> Fixture {
    fixture_script(approved, two_calls, conditions, false)
}

// Separate native batches inside one root turn must retire their cursor before another pause.
fn fixture_script(approved: bool, two_calls: bool, conditions: bool, repeated: bool) -> Fixture {
    fixture_options(approved, two_calls, conditions, repeated, false)
}

// Zero request timeout exercises expiry deterministically without wall-clock sleep assertions.
fn fixture_options(
    approved: bool,
    two_calls: bool,
    conditions: bool,
    repeated: bool,
    expired: bool,
) -> Fixture {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let calls = if two_calls {
        vec![
            ToolCall {
                id: "a".into(),
                name: "first".into(),
                arguments: json!({}),
            },
            ToolCall {
                id: "b".into(),
                name: "second".into(),
                arguments: json!({}),
            },
        ]
    } else {
        vec![ToolCall {
            id: "a".into(),
            name: "first".into(),
            arguments: json!({}),
        }]
    };
    let state = ai_agents_core::NativeProviderState::new(
        "root-first",
        "fixture",
        "native-tools",
        ai_agents_core::NativeProviderTarget::new("https://fixture.invalid", "fixture-model")
            .unwrap(),
        json!({"signature":"first-exact-state"}),
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| ai_agents_core::NativeCallBinding::new(&call.id, index).unwrap())
            .collect(),
    )
    .unwrap();
    let marker = ai_agents_core::encode_native_tool_call_markers(&calls, Some(&state)).unwrap();
    let second_marker = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "later".into(),
            name: "first".into(),
            arguments: json!({}),
        }],
        Some(
            &ai_agents_core::NativeProviderState::new(
                "root-second",
                "fixture",
                "native-tools",
                ai_agents_core::NativeProviderTarget::new(
                    "https://fixture.invalid",
                    "fixture-model",
                )
                .unwrap(),
                json!({"signature":"second-exact-state"}),
                vec![ai_agents_core::NativeCallBinding::new("later", 0).unwrap()],
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let provider = if repeated {
        super::runner_tests::RecordingProvider::new(&[&marker, &second_marker, "done"])
    } else {
        super::runner_tests::RecordingProvider::new(&[&marker, "done"])
    };
    let yaml = if approved {
        "name: SuspensionTest\nsystem_prompt: test\ntools: [first, second]\nhitl:\n  tools:\n    first:\n      require_approval: true\n    second:\n      require_approval: true\n"
    } else {
        "name: SuspensionTest\nsystem_prompt: test\ntools: [first, second]\nhitl:\n  tools:\n    first:\n      require_approval: true\n"
    };
    let yaml = if conditions {
        format!(
            "{yaml}  conditions:\n    - name: high_amount\n      when: amount > 1000\n      require_approval: true\n"
        )
    } else {
        yaml.into()
    };
    let yaml = if expired {
        yaml.replace("hitl:\n", "hitl:\n  default_timeout_seconds: 0\n")
    } else {
        yaml
    };
    let agent = Arc::new(
        crate::AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .tool(Arc::new(Probe {
                id: "first",
                calls: first.clone(),
            }))
            .tool(Arc::new(Probe {
                id: "second",
                calls: second.clone(),
            }))
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(UnexpectedApproval))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "pause-test.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(3);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: None,
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        Default::default(),
        store.clone(),
        "pause-test.v1".into(),
    )
    .unwrap();
    (agent, runner, store, first, second, provider)
}

// A safe pause releases the durable claim but excludes ordinary chat until the exact turn finishes.
#[tokio::test]
async fn approval_pause_resumes_exact_native_batch_without_repeating_user_or_assistant_commit() {
    let (agent, runner, store, first, _, provider) = fixture(false, false);
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert!(paused.final_response.is_none());
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert!(agent.chat("unrelated").await.is_err());
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert!(snapshot.owner_token.is_none());
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let request_id = payload.pending.unwrap().id;
    let completed = runner
        .resume(
            &paused.run.key.run_id,
            snapshot.revision,
            TaskResumeInput::Approval {
                request_id: request_id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(completed.run.status, TaskRunStatus::Completed);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(completed.run.counters.turns, 1);
    let saved = store
        .load(&completed.run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
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
            .filter(|message| message.role == ai_agents_core::Role::Assistant
                && ai_agents_core::decode_native_tool_call_markers(&message.content)
                    .unwrap()
                    .is_some())
            .count(),
        1
    );
    assert_eq!(payload.consumed_request_ids, vec![request_id.clone()]);
    assert!(
        runner
            .resume(
                &completed.run.key.run_id,
                snapshot.revision,
                TaskResumeInput::Approval {
                    request_id,
                    result: ai_agents_hitl::ApprovalResult::Approved
                }
            )
            .await
            .is_err()
    );
    assert!(agent.chat("ordinary").await.is_ok());
}

// A late approval cannot claim ownership; explicit timeout resolution follows the configured rejection policy.
#[tokio::test]
async fn expired_approval_is_not_invoked_and_can_be_settled_without_a_new_model_call() {
    let (_, runner, store, first, _, provider) = fixture_options(false, false, false, false, true);
    let paused = runner.run("objective", None).await.unwrap();
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let request_id = paused.run.pending.unwrap().id;
    assert!(matches!(
        runner
            .resume(
                &paused.run.key.run_id,
                snapshot.revision,
                TaskResumeInput::Approval {
                    request_id: request_id.clone(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
                }
            )
            .await,
        Err(ai_agents_core::AgentError::HITLTimeout)
    ));
    assert_eq!(
        store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        snapshot.revision
    );
    let result = runner
        .resume(
            &paused.run.key.run_id,
            snapshot.revision,
            TaskResumeInput::Approval {
                request_id,
                result: ai_agents_hitl::ApprovalResult::Timeout,
            },
        )
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Failed);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Competing receipts are serialized and the losing resumer cannot execute or replace checkpoint data.
#[tokio::test]
async fn competing_and_mismatched_receipts_do_not_duplicate_dispatch() {
    let (_, runner, store, first, _, provider) = fixture(false, false);
    let paused = runner.run("objective", None).await.unwrap();
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert!(
        runner
            .resume(
                &paused.run.key.run_id,
                snapshot.revision,
                TaskResumeInput::Approval {
                    request_id: "wrong-request".into(),
                    result: ai_agents_hitl::ApprovalResult::Approved,
                }
            )
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
        snapshot.revision
    );
    let request_id = payload.pending.unwrap().id;
    let input = TaskResumeInput::Approval {
        request_id,
        result: ai_agents_hitl::ApprovalResult::Approved,
    };
    let (first_result, second_result) = tokio::join!(
        runner.resume(&paused.run.key.run_id, snapshot.revision, input.clone()),
        runner.resume(&paused.run.key.run_id, snapshot.revision, input)
    );
    assert!(first_result.is_ok() ^ second_result.is_ok());
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

// The same turn can create another native batch without losing completed effects or resetting its iteration count.
#[tokio::test]
async fn consecutive_native_batches_can_pause_in_the_same_turn() {
    let (_, runner, store, first, _, provider) = fixture_script(false, false, false, true);
    let mut paused = runner.run("objective", None).await.unwrap();
    for batch_index in 0..2 {
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        let batch: TaskBatchState = serde_json::from_value(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.tool_batch")
                .unwrap()
                .state
                .clone(),
        )
        .unwrap();
        assert_eq!(
            batch.loop_state.as_ref().unwrap().native_exchanges.len(),
            batch_index + 1
        );
        paused = runner
            .resume(
                &paused.run.key.run_id,
                snapshot.revision,
                TaskResumeInput::Approval {
                    request_id: payload.pending.unwrap().id,
                    result: ai_agents_hitl::ApprovalResult::Approved,
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(paused.run.status, TaskRunStatus::Completed);
    assert_eq!(paused.run.counters.turns, 1);
    assert_eq!(first.load(Ordering::SeqCst), 2);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
}

// Corruption is rejected before the storage claim, not merely by final admission after runtime mutation.
#[tokio::test]
async fn checkpoint_calls_and_reviewed_action_must_match_before_claim() {
    let (_, runner, store, _, _, _) = fixture(false, false);
    let paused = runner.run("objective", None).await.unwrap();
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let state: TaskBatchState = serde_json::from_value(
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.tool_batch")
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    let mut changed = state.clone();
    changed.calls[0].arguments = json!({"changed":true});
    assert!(changed.validate_checkpoint(&payload).is_err());
    let mut changed = state;
    changed.approvals[0].trigger = json!({"changed":true});
    assert!(changed.validate_checkpoint(&payload).is_err());
    assert_eq!(
        store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        snapshot.revision
    );
}

// A later sibling may already finish while the first waits; resume must append its result without reinvocation.
#[tokio::test]
async fn out_of_order_completed_sibling_is_not_replayed_on_resume() {
    let (_, runner, store, first, second, _) = fixture(false, true);
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let completed = runner
        .resume(
            &paused.run.key.run_id,
            snapshot.revision,
            TaskResumeInput::Approval {
                request_id: payload.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(completed.run.status, TaskRunStatus::Completed);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
}

// Each parked action receives its own single-use receipt and advances the same ordered native cursor.
#[tokio::test]
async fn multiple_approvals_repause_without_budget_or_result_reset() {
    let (_, runner, store, first, second, provider) = fixture(true, true);
    let mut paused = runner.run("objective", None).await.unwrap();
    for _ in 0..2 {
        let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        paused = runner
            .resume(
                &paused.run.key.run_id,
                snapshot.revision,
                TaskResumeInput::Approval {
                    request_id: payload.pending.unwrap().id,
                    result: ai_agents_hitl::ApprovalResult::Approved,
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(paused.run.status, TaskRunStatus::Completed);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

// Rejection closes all unresolved sibling history without permitting later approval or model work.
#[tokio::test]
async fn rejecting_one_call_cancels_all_parked_siblings() {
    let (_, runner, store, first, second, provider) = fixture(true, true);
    let paused = runner.run("objective", None).await.unwrap();
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let result = runner
        .resume(
            &paused.run.key.run_id,
            snapshot.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::rejected(None),
            },
        )
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Failed);
    assert!(result.run.pending.is_none());
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(second.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Rejection closes history and must not trigger another model decision or implementation.
#[tokio::test]
async fn rejected_receipt_does_not_continue_the_model_loop() {
    let (_, runner, store, first, _, provider) = fixture(false, false);
    let paused = runner.run("objective", None).await.unwrap();
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    let result = runner
        .resume(
            &paused.run.key.run_id,
            snapshot.revision,
            TaskResumeInput::Approval {
                request_id: payload.pending.unwrap().id,
                result: ai_agents_hitl::ApprovalResult::rejected(None),
            },
        )
        .await
        .unwrap();
    assert_ne!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Passed gates and modified arguments survive a later condition pause without losing other pending siblings.
#[tokio::test]
async fn conditional_reapproval_preserves_other_siblings_and_prior_gate_receipts() {
    let (_, runner, store, first, second, _) = fixture_with_conditions(true, true, true);
    let mut paused = runner.run("objective", None).await.unwrap();
    for round in 0..3 {
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        let result = if round == 0 {
            ai_agents_hitl::ApprovalResult::Modified {
                changes: std::collections::HashMap::from([("amount".into(), json!(2000))]),
            }
        } else {
            ai_agents_hitl::ApprovalResult::Approved
        };
        paused = runner
            .resume(
                &paused.run.key.run_id,
                snapshot.revision,
                TaskResumeInput::Approval {
                    request_id: payload.pending.unwrap().id,
                    result,
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(paused.run.status, TaskRunStatus::Completed);
    assert_eq!(first.load(Ordering::SeqCst), 1);
    assert_eq!(second.load(Ordering::SeqCst), 1);
}

// A cancelled parked batch produces complete native history and releases ownership only after CAS acknowledgement.
#[tokio::test]
async fn paused_cancellation_does_not_invoke_and_releases_the_runtime() {
    let (agent, runner, store, first, _, provider) = fixture(false, false);
    let paused = runner.run("objective", None).await.unwrap();
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    let cancelled = runner
        .cancel_paused(&paused.run.key.run_id, snapshot.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.run.status, TaskRunStatus::Cancelled);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(agent.chat("ordinary").await.is_ok());
}
