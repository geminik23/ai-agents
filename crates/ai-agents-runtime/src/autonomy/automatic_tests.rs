//! Automatic entry fixtures assert real runtime ownership and effects rather than inferring execution from task-shaped text.

#[path = "automatic_acceptance_tests.rs"]
mod acceptance;
#[path = "automatic_compatibility_tests.rs"]
mod compatibility;
#[path = "automatic_failure_tests.rs"]
mod failure_acceptance;
#[path = "automatic_race_tests.rs"]
mod races;
#[path = "automatic_state_tests.rs"]
mod state_acceptance;

use super::runner_tests::RecordingProvider;
use super::*;
use crate::{Agent, AgentBuilder, AgentStreamEvent, RuntimeAgent};
use ai_agents_core::{AgentResponse, Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Responses(Arc<AtomicUsize>);
#[async_trait]
impl ai_agents_hooks::AgentHooks for Responses {
    // The fixture owns only local counters and has no hidden provider or detached work.
    fn task_providers(&self) -> Option<Vec<Arc<dyn ai_agents_core::LLMProvider>>> {
        Some(vec![])
    }
    // Witness completed logical-turn callbacks separately from task status and display delivery.
    async fn on_response(&self, _: &AgentResponse) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct Effect(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Effect {
    // Keep one fixture capability identity stable across retained execution.
    fn id(&self) -> &str {
        "effect"
    }
    // Model-facing naming must not create another granted capability.
    fn name(&self) -> &str {
        "effect"
    }
    // Advertise isolation rather than imply a real external operation.
    fn description(&self) -> &str {
        "Records an isolated fixture effect"
    }
    // Arguments are structural fixture data, not semantic completion decisions.
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    // Only the local invocation witness changes; the fixture cannot mutate workspace or host resources.
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            ..Default::default()
        }
    }
    // Count actual body invocation so requests and approvals alone cannot satisfy effect assertions.
    async fn execute(&self, args: Value, _: ToolExecutionContext) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok(args.to_string())
    }
}

// Hosts install ordinary providers once; automatic declarations do not call them during construction.
fn agent(yaml: &str, main: Arc<RecordingProvider>, router: Arc<RecordingProvider>) -> RuntimeAgent {
    AgentBuilder::from_yaml(yaml)
        .unwrap()
        .llm(main)
        .llm_alias("router", router)
        .build()
        .unwrap()
}

const STATE: &str = "name: auto\nsystem_prompt: Helpful\nllm: {default: default, router: router}\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        max_turns: 4\n        max_llm_calls: 4\n        completion: {response_contains: done}\n";

const SKILL: &str = "name: auto\nsystem_prompt: Helpful\nllm: {default: default, router: router}\nskills:\n  - id: work\n    description: Research\n    trigger: Research the request\n    autonomy:\n      enabled: true\n      mode: skill_until_complete\n      max_turns: 4\n      max_llm_calls: 4\n      completion: {response_contains: done}\n    steps:\n      - prompt: 'Research {{ user_input }}'\n";

#[tokio::test]
// Construction is inert, and completed task turns retain one input commit and their own finalization callbacks.
async fn auto_state_entry_is_input_triggered_and_completes_without_reprocessing() {
    let main = RecordingProvider::new(&["draft", "done"]);
    let router = RecordingProvider::new(&[]);
    let responses = Arc::new(AtomicUsize::new(0));
    let agent = AgentBuilder::from_yaml(STATE)
        .unwrap()
        .llm(main.clone())
        .llm_alias("router", router.clone())
        .hooks(Arc::new(Responses(responses.clone())))
        .build()
        .unwrap();
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    assert!(agent.automatic_task_runner().is_none());
    let response = agent.chat("사용자 요청").await.unwrap();
    let result = agent.automatic_task_result().unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(response.content, "done");
    assert_eq!(
        response.metadata.unwrap()["task_run"]["status"],
        "completed"
    );
    assert_eq!(result.run.counters.llm_attempts, 2);
    assert_eq!(result.run.counters.turns, 2);
    assert_eq!(responses.load(Ordering::SeqCst), 2);
    let snapshot = agent.save_state().await.unwrap();
    assert_eq!(
        snapshot
            .memory
            .messages
            .iter()
            .filter(|message| message.role == ai_agents_core::Role::User)
            .count(),
        1
    );
    assert_eq!(router.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
// Prior tool results must reach normal continuation without replaying selection, script effects or response metadata.
async fn auto_skill_routes_once_executes_prefix_once_and_continues_same_run() {
    let main = RecordingProvider::new(&["draft", "done"]);
    let router = RecordingProvider::new(&["work"]);
    let effects = Arc::new(AtomicUsize::new(0));
    let yaml = SKILL.replace(
        "    steps:\n",
        "    steps:\n      - tool: effect\n        args: {record: fixture-only-earlier-step-value}\n",
    ) + "tools: [effect]\n";
    let agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(main.clone())
        .llm_alias("router", router.clone())
        .tool(Arc::new(Effect(effects.clone())))
        .build()
        .unwrap();
    let response = agent.chat("find sources").await.unwrap();
    let result = agent.automatic_task_result().unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(response.content, "done");
    assert_eq!(router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(effects.load(Ordering::SeqCst), 1);
    assert_eq!(result.run.counters.tool_attempts, 1);
    assert_eq!(result.run.counters.llm_attempts, 2);
    assert!(
        main.captured_messages()[0]
            .iter()
            .any(|message| message.content.contains("find sources"))
    );
    let requests = main.captured_messages();
    let controller: Value = requests[1]
        .iter()
        .find_map(|message| {
            message
                .content
                .strip_prefix("Autonomy controller instruction (not a user statement):\n")
        })
        .map(|data| serde_json::from_str(data).unwrap())
        .unwrap();
    assert_eq!(
        controller["skill_results"][0]["steps"][0]["result"]["record"],
        "fixture-only-earlier-step-value"
    );
    assert_eq!(
        controller["skill_results"][0]["completion_authority"],
        false
    );
    assert!(
        controller["skill_results"][0]["steps"][0]
            .get("metadata")
            .is_none()
    );
}

#[tokio::test]
// A scope exit can satisfy declared gates but cannot manufacture proof through unrelated destination work.
async fn auto_state_exit_uses_post_exit_gates_and_does_not_execute_destination_work() {
    for success in [false, true] {
        let yaml = STATE.replace(
            "completion: {response_contains: done}",
            if success {
                "completion: {state: resolved}"
            } else {
                "completion: {state: absent}"
            },
        ) + "      transitions:\n        - to: resolved\n          guard: 'true'\n    resolved:\n      prompt: Unrelated destination work\n";
        let main = RecordingProvider::new(&["candidate", "must not execute"]);
        let agent = agent(&yaml, main.clone(), RecordingProvider::new(&[]));
        agent.chat("resolve ticket").await.unwrap();
        let result = agent.automatic_task_result().unwrap();
        assert_eq!(
            result.run.status,
            if success {
                TaskRunStatus::Completed
            } else {
                TaskRunStatus::Incomplete
            },
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(agent.current_state().as_deref(), Some("resolved"));
        assert!(agent.chat("ordinary follow-up").await.is_ok());
    }
}

#[tokio::test]
// Only the exact retained response may resume the reserved runtime; stale delivery cannot spend another call.
async fn auto_pause_has_real_host_resume_and_preserves_original_owner_and_counters() {
    let yaml = STATE.replace(
        "        completion:",
        "        premature_finish: {action: ask_user, on_unavailable: stop}\n        completion:",
    );
    let main = RecordingProvider::new(&["draft", "done"]);
    let agent = agent(&yaml, main.clone(), RecordingProvider::new(&[]));
    agent.chat("work").await.unwrap();
    let paused = agent.automatic_task_result().unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert!(agent.chat("unrelated").await.is_err());
    let pending = paused.run.pending.as_ref().unwrap();
    assert!(
        agent
            .automatic_task_operation(TaskStreamOperation::Resume {
                run_id: paused.run.key.run_id.clone(),
                revision: paused.run.revision + 1,
                input: TaskResumeInput::UserAnswer {
                    request_id: pending.id.clone(),
                    answer: json!("continue")
                }
            })
            .await
            .is_err()
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
    let done = agent
        .automatic_task_operation(TaskStreamOperation::Resume {
            run_id: paused.run.key.run_id.clone(),
            revision: paused.run.revision,
            input: TaskResumeInput::UserAnswer {
                request_id: pending.id.clone(),
                answer: json!("continue"),
            },
        })
        .await
        .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(done.run.key, paused.run.key);
    assert_eq!(done.run.counters.llm_attempts, 2);
}

#[tokio::test]
// Event delivery is lazy and default-safe, while the acknowledged task outcome precedes the dispatch Final.
async fn auto_stream_is_lazy_and_has_same_authoritative_task_result() {
    let main = RecordingProvider::new(&["draft", "done"]);
    let agent = agent(STATE, main.clone(), RecordingProvider::new(&[]));
    let stream = agent.chat_stream_events("not polled").await.unwrap();
    drop(stream);
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    assert!(agent.automatic_task_runner().is_none());
    let mut stream = agent.chat_stream_events("report").await.unwrap();
    let mut task_events = Vec::new();
    let mut final_response = None;
    while let Some(event) = stream.next().await {
        match event {
            AgentStreamEvent::Final(response) => {
                assert!(final_response.is_none());
                final_response = Some(response);
            }
            AgentStreamEvent::Chunk(crate::StreamChunk::Error { message }) => panic!("{message}"),
            AgentStreamEvent::Task(event) => {
                assert!(final_response.is_none());
                task_events.push(event);
            }
            _ => {}
        }
    }
    assert_eq!(
        task_events.first().unwrap().kind,
        TaskRunEventKind::TaskStarted
    );
    assert_eq!(
        task_events.last().unwrap().kind,
        TaskRunEventKind::TaskCompleted
    );
    assert!(
        task_events
            .iter()
            .any(|event| event.kind == TaskRunEventKind::CompletionGateChecked)
    );
    assert!(
        !serde_json::to_string(&task_events)
            .unwrap()
            .contains("report")
    );
    assert_eq!(final_response.unwrap().content, "done");
    assert_eq!(
        agent.automatic_task_result().unwrap().run.status,
        TaskRunStatus::Completed
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
// Nested autonomous skill configuration must not replace the coordinating state's completion contract.
async fn auto_state_owns_selected_autonomous_skill_without_replacing_outer_gate() {
    let state = "states:\n  initial: support\n  states:\n    support:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        max_turns: 3\n        completion: {response_contains: done}\n";
    let yaml = SKILL.replace(
        "completion: {response_contains: done}",
        "completion: {response_contains: unreachable}",
    ) + state;
    let router = RecordingProvider::new(&["work"]);
    let main = RecordingProvider::new(&["draft", "done"]);
    let agent = agent(&yaml, main.clone(), router.clone());
    agent.chat("research ticket").await.unwrap();
    let result = agent.automatic_task_result().unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        result.run.stop_reason
    );
    assert_eq!(result.run.counters.llm_attempts, 3);
    assert_eq!(router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
// Templates resolve accepted language-neutral input and actual selected metadata before controller work.
async fn auto_objective_templates_bind_user_state_and_selected_skill_once() {
    let yaml = SKILL.replace("      mode: skill_until_complete", "      mode: skill_until_complete\n      objective_template: '{{ user_input }} / {{ skill.id }} / {{ skill.trigger }}'");
    let agent = agent(
        &yaml,
        RecordingProvider::new(&["done"]),
        RecordingProvider::new(&["work"]),
    );
    agent.chat("조사").await.unwrap();
    let result = agent.automatic_task_result().unwrap();
    assert_eq!(result.run.objective, "조사 / work / Research the request");
    assert_eq!(result.run.status, TaskRunStatus::Completed);
}

#[tokio::test]
// Explicit disable preserves ordinary execution rather than installing unnecessary task requirements.
async fn auto_disabled_scopes_remain_ordinary_and_do_not_require_task_resources() {
    let yaml = STATE.replace("enabled: true", "enabled: false");
    let main = RecordingProvider::new(&["ordinary"]);
    let agent = agent(&yaml, main.clone(), RecordingProvider::new(&[]));
    assert_eq!(agent.chat("hello").await.unwrap().content, "ordinary");
    assert!(agent.automatic_task_result().is_none());
}
