use super::*;

#[path = "disambig_acceptance_tests.rs"]
mod disambig_acceptance_tests;
#[path = "interaction_failure_tests.rs"]
mod failure_tests;
#[path = "interaction_matrix_tests.rs"]
mod matrix_tests;
use crate::AgentBuilder;
use ai_agents_core::{Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use ai_agents_hitl::{ApprovalHandler, ApprovalRequest, ApprovalResult};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct NoApproval;
#[async_trait]
impl ApprovalHandler for NoApproval {
    async fn request_approval(&self, _: ApprovalRequest) -> ApprovalResult {
        panic!("non-tool interaction must pause before the handler")
    }
}
struct Probe(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Probe {
    fn id(&self) -> &str {
        "probe"
    }
    fn name(&self) -> &str {
        "probe"
    }
    fn description(&self) -> &str {
        "Count admitted effects"
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
        ToolResult::ok("effect")
    }
}
struct Hooks(Arc<AtomicUsize>);
#[async_trait]
impl ai_agents_hooks::AgentHooks for Hooks {
    async fn on_response(&self, _: &ai_agents_core::AgentResponse) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
struct Fixture {
    agent: Arc<crate::RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    main: Arc<super::runner_tests::RecordingProvider>,
    router: Arc<super::runner_tests::RecordingProvider>,
    effects: Arc<AtomicUsize>,
    finalized: Arc<AtomicUsize>,
}
// Fixtures use production YAML, frozen aliases and the actual runtime, not a fake interaction executor.
fn fixture(yaml: &str, main_replies: &[&str], router_replies: &[&str]) -> Fixture {
    let main = super::runner_tests::RecordingProvider::new(main_replies);
    let router = super::runner_tests::RecordingProvider::new(router_replies);
    let mut registry = ai_agents_llm::LLMRegistry::new();
    registry.register("default", main.clone());
    registry.register("router", router.clone());
    let effects = Arc::new(AtomicUsize::new(0));
    let finalized = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_registry(registry)
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(Probe(effects.clone())))
            .approval_handler(Arc::new(NoApproval))
            .hooks(Arc::new(Hooks(finalized.clone())))
            .build()
            .unwrap(),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "interaction.v1".into()).unwrap(),
    );
    let mut config = super::runner_tests::config(16);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        Default::default(),
        store.clone(),
        "interaction.v1".into(),
    )
    .unwrap();
    Fixture {
        agent,
        runner,
        store,
        main,
        router,
        effects,
        finalized,
    }
}
const STATE: &str = "name: Review\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ntools: [probe]\nhitl:\n  states:\n    ready:\n      on_enter: require_approval\n      approval_message: Approve transition\nstates:\n  initial: active\n  states:\n    active:\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready:\n      regenerate_on_enter: false\n";

async fn respond(f: &Fixture, paused: &TaskRun, result: ApprovalResult) -> TaskRunResult {
    f.runner
        .resume(
            &paused.key.run_id,
            paused.revision,
            TaskResumeInput::Approval {
                request_id: paused.pending.as_ref().unwrap().id.clone(),
                result,
            },
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn interaction_transition_approval_and_rejection_preserve_post_text_position() {
    for approved in [false, true] {
        let f = fixture(STATE, &["done"], &[]);
        let paused = f.runner.run("objective", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
        assert_eq!(f.agent.current_state().as_deref(), Some("active"));
        let done = respond(
            &f,
            &paused.run,
            if approved {
                ApprovalResult::Approved
            } else {
                ApprovalResult::Rejected {
                    reason: Some("no".into()),
                }
            },
        )
        .await;
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(
            f.agent.current_state().as_deref(),
            Some(if approved { "ready" } else { "active" })
        );
        assert_eq!(f.agent.state_history().len(), usize::from(approved));
        assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn interaction_rejected_pre_tool_transition_keeps_original_native_calls() {
    let marker = ai_agents_core::encode_native_tool_call_markers(
        &[ai_agents_core::ToolCall {
            id: "original".into(),
            name: "probe".into(),
            arguments: json!({}),
        }],
        None,
    )
    .unwrap();
    let f = fixture(STATE, &[&marker, "done"], &[]);
    let first = f.runner.run("objective", None).await.unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    assert_eq!(f.effects.load(Ordering::SeqCst), 0);
    let second = respond(&f, &first.run, ApprovalResult::Rejected { reason: None }).await;
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    assert_eq!(f.effects.load(Ordering::SeqCst), 1);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
    let done = respond(&f, &second.run, ApprovalResult::Rejected { reason: None }).await;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.effects.load(Ordering::SeqCst), 1);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 2);
    assert_eq!(done.run.counters.tool_attempts, 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        f.store
            .load(&done.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert!(
        ai_agents_core::inspect_native_history(&payload.runtime.snapshot.memory.messages)
            .unwrap()
            .exchanges()
            .iter()
            .all(|exchange| exchange.is_complete())
    );
}

const AMBIGUOUS: &str = r#"{"is_ambiguous":true,"confidence":0.1,"what_is_unclear":["target"],"reasoning":"missing","ambiguity_type":"missing_target"}"#;
const QUESTION: &str = r#"{"question":"Which target?","style":"open","clarifying":["target"]}"#;
const ANSWER: &str = r#"{"status":"answered","enriched_input":"objective for target one","resolved":{"target":"one"}}"#;
const CLARIFICATION: &str = "name: Clarify\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ndisambiguation:\n  enabled: true\n  detection:\n    llm: router\n  clarification:\n    llm: router\n";

#[tokio::test]
async fn interaction_clarification_resumes_without_repeating_detection_or_root_commit() {
    let f = fixture(CLARIFICATION, &["done"], &[AMBIGUOUS, QUESTION, ANSWER]);
    let paused = f.runner.run("objective", None).await.unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 0);
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
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 3);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.finalized.load(Ordering::SeqCst), 1);
    assert_eq!(done.run.counters.turns, 1);
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
async fn interaction_clarification_cancel_and_invalid_response_do_not_parse_answer() {
    let f = fixture(CLARIFICATION, &["done"], &[AMBIGUOUS, QUESTION]);
    let paused = f.runner.run("objective", None).await.unwrap();
    assert!(
        f.runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                    answer: json!({"text":"wrong shape"})
                }
            )
            .await
            .is_err()
    );
    assert_eq!(
        f.store
            .load(&paused.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .revision,
        paused.run.revision
    );
    let done = f
        .runner
        .cancel_paused(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Cancelled);
    assert_eq!(f.router.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.main.calls.load(Ordering::SeqCst), 0);
}
