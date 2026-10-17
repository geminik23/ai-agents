use super::continuation_test_support::{self as support, ReceiptStore};
use super::*;
use crate::{AgentBuilder, RuntimeAgent};
use ai_agents_core::{Tool, ToolExecutionContext, ToolResult, ToolSafetyMetadata};
use ai_agents_hitl::ApprovalResult;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Probe {
    id: &'static str,
    calls: Arc<AtomicUsize>,
    arguments: Arc<parking_lot::Mutex<Vec<Value>>>,
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
        "State action invocation probe"
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
    async fn execute(&self, args: Value, _: ToolExecutionContext) -> ToolResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.arguments.lock().push(args.clone());
        ToolResult::ok(args.to_string())
    }
}
#[derive(Default)]
struct Hooks {
    responses: AtomicUsize,
    transitions: AtomicUsize,
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for Hooks {
    async fn on_response(&self, _: &ai_agents_core::AgentResponse) {
        self.responses.fetch_add(1, Ordering::SeqCst);
    }
    async fn on_state_transition(&self, _: Option<&str>, _: &str, _: &str) {
        self.transitions.fetch_add(1, Ordering::SeqCst);
    }
}
struct Fixture {
    agent: Arc<RuntimeAgent>,
    runner: AutonomyRunner,
    store: Arc<ScopedTaskRunStore>,
    first: Arc<AtomicUsize>,
    second: Arc<AtomicUsize>,
    arguments: Arc<parking_lot::Mutex<Vec<Value>>>,
    hooks: Arc<Hooks>,
}
fn runner(agent: Arc<RuntimeAgent>, store: Arc<dyn TaskRunStore>) -> AutonomyRunner {
    let mut config = super::runner_tests::config(16);
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_user_question: Some(InteractionAction::PauseRun),
        on_approval_required: Some(InteractionAction::PauseRun),
    });
    AutonomyRunner::try_new(agent, config, Default::default(), store, "state.v1".into()).unwrap()
}
async fn from_yaml(
    yaml: &str,
    main: Arc<dyn ai_agents_core::LLMProvider>,
    router: Arc<dyn ai_agents_core::LLMProvider>,
    sqlite: bool,
) -> Fixture {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let arguments = Arc::new(parking_lot::Mutex::new(vec![]));
    let hooks = Arc::new(Hooks::default());
    let mut llms = ai_agents_llm::LLMRegistry::new();
    llms.register("default", main);
    llms.register("router", router);
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm_registry(llms)
            .auto_configure_features()
            .unwrap()
            .hooks(hooks.clone())
            .tool(Arc::new(Probe {
                id: "first",
                calls: first.clone(),
                arguments: arguments.clone(),
            }))
            .tool(Arc::new(Probe {
                id: "second",
                calls: second.clone(),
                arguments: arguments.clone(),
            }))
            .tool(Arc::new(ai_agents_tools::builtin::AskUserTool::new(
                Arc::new(parking_lot::RwLock::new(None)),
            )))
            .build()
            .unwrap(),
    );
    let store = support::store(agent.info().id, "state.v1", sqlite).await;
    let runner = runner(agent.clone(), store.clone());
    Fixture {
        agent,
        runner,
        store,
        first,
        second,
        arguments,
        hooks,
    }
}
// The same logical transition parks before commit or after entry, so cleanup must preserve the actual phase.
async fn fixture(entering: bool, question: bool, reentry: bool, sqlite: bool) -> Fixture {
    let action = if question {
        "        - tool: ask_user\n          args: {question: Choice, options: [yes], allow_other: false, timeout_seconds: 1}\n"
    } else {
        "        - tool: second\n          args: {value: pending}\n"
    };
    let target = if reentry { "active" } else { "ready" };
    let actions = format!("        - tool: first\n{action}");
    let yaml = format!(
        "name: StateFlow\nsystem_prompt: test\ntools: [first,second,ask_user]\nhitl:\n  default_timeout_seconds: 1\n  on_timeout: reject\n  tools:\n    second:\n      require_approval: true\nstates:\n  initial: active\n  states:\n    active:\n{}      transitions:\n        - to: {target}\n          guard: 'true'\n{}",
        if !entering && !reentry {
            format!("      on_exit:\n{actions}")
        } else {
            String::new()
        },
        if reentry {
            format!(
                "      on_enter:\n        - set_context: {{wrong_entry: true}}\n      on_reenter:\n{actions}"
            )
        } else if entering {
            format!("    ready:\n      regenerate_on_enter: false\n      on_enter:\n{actions}")
        } else {
            "    ready:\n      regenerate_on_enter: false\n".into()
        }
    );
    Box::pin(from_yaml(
        &yaml,
        super::runner_tests::RecordingProvider::new(&["done"]),
        super::runner_tests::RecordingProvider::new(&[]),
        sqlite,
    ))
    .await
}
async fn approve(f: &Fixture, run: &TaskRun) -> ai_agents_core::Result<TaskRunResult> {
    f.runner
        .resume(
            &run.key.run_id,
            run.revision,
            TaskResumeInput::Approval {
                request_id: run.pending.as_ref().unwrap().id.clone(),
                result: ApprovalResult::Modified {
                    changes: std::collections::HashMap::from([("value".into(), json!("modified"))]),
                },
            },
        )
        .await
}

#[tokio::test]
async fn state_acceptance_reentry_and_entry_exit_resume_preserve_commit_and_hooks() {
    for sqlite in support::backends() {
        for (entering, reentry) in [(false, false), (true, false), (true, true)] {
            let f = Box::pin(fixture(entering, false, reentry, *sqlite)).await;
            let first = f.runner.run("objective", None).await.unwrap();
            assert_eq!(first.run.status, TaskRunStatus::Paused);
            assert_eq!(f.first.load(Ordering::SeqCst), 1);
            assert_eq!(f.agent.state_history().len(), usize::from(entering));
            let done = approve(&f, &first.run).await.unwrap();
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(f.first.load(Ordering::SeqCst), 1);
            assert_eq!(f.second.load(Ordering::SeqCst), 1);
            assert_eq!(f.agent.state_history().len(), 1);
            assert_eq!(f.hooks.transitions.load(Ordering::SeqCst), 1);
            assert_eq!(f.hooks.responses.load(Ordering::SeqCst), 1);
            assert!(!f.agent.get_context().contains_key("wrong_entry"));
            assert_eq!(f.arguments.lock().last().unwrap()["value"], "modified");
        }
    }
}

#[tokio::test]
async fn state_acceptance_exit_and_entry_question_timeout_does_not_commit_again() {
    for entering in [false, true] {
        let f = Box::pin(fixture(entering, true, false, false)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let done = f
            .runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::QuestionTimeout {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Incomplete);
        assert_eq!(f.agent.state_history().len(), usize::from(entering));
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 0);
        assert!(f.agent.set_context("released", json!(true)).is_ok());
    }
}

#[tokio::test]
async fn state_acceptance_stop_cancel_and_limit_revision_preserve_actual_commit_phase() {
    for sqlite in support::backends() {
        for entering in [false, true] {
            for control in ["stop", "cancel", "limits"] {
                let f = Box::pin(fixture(entering, false, false, *sqlite)).await;
                let first = f.runner.run("objective", None).await.unwrap();
                let done = match control {
                    "stop" => f
                        .runner
                        .stop_paused(
                            &first.run.key.run_id,
                            first.run.revision,
                            "host stopped state",
                        )
                        .await
                        .unwrap(),
                    "cancel" => f
                        .runner
                        .cancel_paused(&first.run.key.run_id, first.run.revision)
                        .await
                        .unwrap(),
                    _ => {
                        let mut limits = first.run.limits.clone();
                        limits.max_tool_calls += 1;
                        let revised = f
                            .runner
                            .revise_limits_paused(
                                &first.run.key.run_id,
                                first.run.revision,
                                limits,
                                "capacity",
                            )
                            .await
                            .unwrap();
                        assert_eq!(
                            revised.run.pending.as_ref().unwrap().id,
                            first.run.pending.as_ref().unwrap().id
                        );
                        assert_eq!(
                            serde_json::to_value(&revised.run.counters).unwrap(),
                            serde_json::to_value(&first.run.counters).unwrap()
                        );
                        assert!(approve(&f, &first.run).await.is_err());
                        approve(&f, &revised.run).await.unwrap()
                    }
                };
                assert_eq!(
                    done.run.status,
                    match control {
                        "stop" => TaskRunStatus::Incomplete,
                        "cancel" => TaskRunStatus::Cancelled,
                        _ => TaskRunStatus::Completed,
                    }
                );
                assert_eq!(f.first.load(Ordering::SeqCst), 1);
                assert_eq!(
                    f.second.load(Ordering::SeqCst),
                    usize::from(control == "limits")
                );
                assert_eq!(
                    f.agent.state_history().len(),
                    if control == "limits" {
                        1
                    } else {
                        usize::from(entering)
                    }
                );
                assert!(f.agent.set_context("released", json!(true)).is_ok());
            }
        }
    }
}

#[tokio::test]
async fn state_acceptance_generation_change_before_resume_rejects_without_new_action() {
    for entering in [false, true] {
        let f = Box::pin(fixture(entering, false, false, false)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        let machine = f.agent.state_machine().unwrap();
        machine
            .transition_to(if entering { "active" } else { "ready" }, "trusted host")
            .unwrap();
        assert!(approve(&f, &first.run).await.is_err());
        assert_eq!(f.second.load(Ordering::SeqCst), 0);
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            first.run.revision
        );
        f.runner
            .cancel_paused(&first.run.key.run_id, first.run.revision)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn state_acceptance_scope_revocation_preserves_denied_record_without_invocation() {
    let f = Box::pin(fixture(false, false, false, false)).await;
    let first = f.runner.run("objective", None).await.unwrap();
    f.agent.runtime_control().set_tool_scope(Vec::new());
    let done = approve(&f, &first.run).await.unwrap();
    assert_eq!(f.second.load(Ordering::SeqCst), 0);
    let denied = done
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|r| r.record.canonical_id == "second")
        .unwrap();
    assert!(!denied.record.executed && !denied.record.success);
    // State actions retain the existing tolerate-failure semantics; a denied action is not fabricated as successful work.
    assert_eq!(f.agent.state_history().len(), 1);
}

#[tokio::test]
async fn state_acceptance_terminal_receipt_failure_protects_committed_transition() {
    for sqlite in support::backends() {
        for mode in ["positive", "reject", "lost", "drop"] {
            let mut f = Box::pin(fixture(true, false, false, *sqlite)).await;
            let fault = ReceiptStore::new(f.store.clone(), mode, TaskRunStatus::Completed);
            f.runner = runner(f.agent.clone(), fault.clone());
            let first = f.runner.run("objective", None).await.unwrap();
            fault.armed.store(true, Ordering::SeqCst);
            let mut resume = Box::pin(approve(&f, &first.run));
            if mode == "positive" {
                assert_eq!(resume.await.unwrap().run.status, TaskRunStatus::Completed);
            } else {
                if mode == "drop" {
                    tokio::select! {result=&mut resume=>panic!("receipt not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("receipt not reached") }
                    drop(resume);
                } else {
                    assert!(resume.await.is_err());
                }
                assert!(f.agent.set_context("blocked", json!(true)).is_err());
                assert!(approve(&f, &first.run).await.is_err());
            }
            fault.assert_final_write(&first.run.key.run_id).await;
            assert_eq!(f.agent.state_history().len(), 1);
            assert_eq!(f.first.load(Ordering::SeqCst), 1);
            assert_eq!(f.second.load(Ordering::SeqCst), 1);
            assert_eq!(f.hooks.responses.load(Ordering::SeqCst), 1);
        }
    }
}

#[tokio::test]
async fn state_acceptance_generation_change_at_prompt_checkpoint_prevents_provider_dispatch() {
    let main = super::runner_tests::RecordingProvider::new(&["done", "must not dispatch"]);
    let yaml = "name: PromptCheckpoint\nsystem_prompt: test\ntools: [second]\nstates:\n  initial: active\n  states:\n    active:\n      on_exit:\n        - prompt: Prepare\n          store_as: old_result\n        - tool: second\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready: {}\n    other: {}\n";
    let mut f = Box::pin(from_yaml(
        yaml,
        main.clone(),
        super::runner_tests::RecordingProvider::new(&[]),
        false,
    ))
    .await;
    let fault = ReceiptStore::checkpoint(f.store.clone(), "hold", json!({"prompt":true}));
    fault.armed.store(true, Ordering::SeqCst);
    f.runner = runner(f.agent.clone(), fault.clone());
    let mut run = Box::pin(f.runner.run("objective", None));
    tokio::select! {result=&mut run=>panic!("checkpoint not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("checkpoint not reached") }
    f.agent
        .state_machine()
        .unwrap()
        .transition_to("other", "trusted host")
        .unwrap();
    fault.release.add_permits(1);
    let outcome = run.await;
    assert!(
        !outcome
            .as_ref()
            .is_ok_and(|r| r.run.status == TaskRunStatus::Completed)
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.second.load(Ordering::SeqCst), 0);
    assert!(!f.agent.get_context().contains_key("old_result"));
    fault.assert_hit();
}

#[tokio::test]
async fn state_acceptance_generation_change_during_prompt_cannot_run_next_old_action() {
    let provider = support::FaultProvider::new(&["done", "state prompt result"], 1, "gate");
    let yaml = "name: PromptRace\nsystem_prompt: test\ntools: [second]\nstates:\n  initial: active\n  states:\n    active:\n      on_exit:\n        - prompt: Prepare state\n          store_as: old_result\n        - tool: second\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready: {}\n    other: {}\n";
    let f = Box::pin(from_yaml(
        yaml,
        provider.clone(),
        super::runner_tests::RecordingProvider::new(&[]),
        false,
    ))
    .await;
    let mut work = Box::pin(f.runner.run("objective", None));
    tokio::select! {result=&mut work=>panic!("provider not held: {result:?}"),permit=provider.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("provider not reached") }
    f.agent
        .state_machine()
        .unwrap()
        .transition_to("other", "trusted host")
        .unwrap();
    provider.release.add_permits(1);
    let outcome = work.await;
    assert!(
        !outcome
            .as_ref()
            .is_ok_and(|r| r.run.status == TaskRunStatus::Completed)
    );
    assert_eq!(
        f.second.load(Ordering::SeqCst),
        0,
        "old action must not run in a new generation"
    );
    assert!(!f.agent.get_context().contains_key("old_result"));
}

#[tokio::test]
async fn state_acceptance_previously_visited_target_uses_reentry_actions() {
    let yaml = "name: Visited\nsystem_prompt: test\ntools: [first,second]\nhitl:\n  tools:\n    second:\n      require_approval: true\nstates:\n  initial: active\n  states:\n    active:\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready:\n      regenerate_on_enter: false\n      on_enter:\n        - set_context: {wrong_entry: true}\n      on_reenter:\n        - tool: first\n        - tool: second\n";
    let f = Box::pin(from_yaml(
        yaml,
        super::runner_tests::RecordingProvider::new(&["done"]),
        super::runner_tests::RecordingProvider::new(&[]),
        false,
    ))
    .await;
    let machine = f.agent.state_machine().unwrap();
    machine.transition_to("ready", "history").unwrap();
    machine.transition_to("active", "history").unwrap();
    let first = f.runner.run("objective", None).await.unwrap();
    let done = approve(&f, &first.run).await.unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.agent.state_history().len(), 3);
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.hooks.transitions.load(Ordering::SeqCst), 1);
    assert!(!f.agent.get_context().contains_key("wrong_entry"));
}

#[tokio::test]
async fn state_acceptance_initial_pause_receipt_failure_preserves_exit_or_entry_position() {
    for sqlite in support::backends() {
        for entering in [false, true] {
            for mode in ["reject", "lost", "drop"] {
                let mut f = Box::pin(fixture(entering, false, false, *sqlite)).await;
                let fault = ReceiptStore::new(f.store.clone(), mode, TaskRunStatus::Paused);
                fault.armed.store(true, Ordering::SeqCst);
                f.runner = runner(f.agent.clone(), fault.clone());
                let mut run = Box::pin(f.runner.run("objective", None));
                if mode == "drop" {
                    tokio::select! {result=&mut run=>panic!("receipt not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("receipt not reached") }
                    drop(run);
                } else {
                    assert!(run.await.is_err());
                }
                let saved = f.store.list().await.unwrap().pop().unwrap();
                fault.assert_final_write(&saved.key.run_id).await;
                assert_eq!(f.first.load(Ordering::SeqCst), 1);
                assert_eq!(f.second.load(Ordering::SeqCst), 0);
                assert_eq!(f.agent.state_history().len(), usize::from(entering));
                assert!(f.agent.set_context("blocked", json!(true)).is_err());
            }
        }
    }
}

struct HookGate {
    reached: tokio::sync::Semaphore,
    calls: AtomicUsize,
    point: &'static str,
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for HookGate {
    async fn on_state_transition(&self, _: Option<&str>, _: &str, _: &str) {
        if self.point == "transition" {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.reached.add_permits(1);
            std::future::pending::<()>().await;
        }
    }
    async fn on_response(&self, _: &ai_agents_core::AgentResponse) {
        if self.point == "response" {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.reached.add_permits(1);
            std::future::pending::<()>().await;
        }
    }
}

#[tokio::test]
async fn state_acceptance_interrupted_transition_or_response_hook_cannot_repeat_commit() {
    for point in ["transition", "response"] {
        let mut f = Box::pin(fixture(false, false, false, false)).await;
        drop(f.runner);
        let gate = Arc::new(HookGate {
            reached: tokio::sync::Semaphore::new(0),
            calls: AtomicUsize::new(0),
            point,
        });
        let runtime =
            Arc::try_unwrap(f.agent).unwrap_or_else(|_| panic!("fixture retained runtime"));
        f.agent = Arc::new(runtime.with_hooks(gate.clone()));
        f.runner = runner(f.agent.clone(), f.store.clone());
        let first = f.runner.run("objective", None).await.unwrap();
        let mut resume = Box::pin(approve(&f, &first.run));
        tokio::select! {result=&mut resume=>panic!("hook not held: {result:?}"),permit=gate.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("hook not reached") }
        drop(resume);
        assert_eq!(f.agent.state_history().len(), 1);
        assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 1);
        assert!(approve(&f, &first.run).await.is_err());
        assert!(f.agent.set_context("blocked", json!(true)).is_err());
    }
}

#[tokio::test]
async fn state_acceptance_provider_failure_deadline_or_drop_prevents_later_actions() {
    for mode in ["error", "timeout", "drop"] {
        let provider = support::FaultProvider::new(
            &["done", "unused"],
            1,
            if mode == "timeout" { "drop" } else { mode },
        );
        let yaml = "name: StateProviderFailure\nsystem_prompt: test\ntools: [first,second]\nstates:\n  initial: active\n  states:\n    active:\n      on_exit:\n        - tool: first\n        - prompt: Prepare state\n          store_as: old_result\n        - tool: second\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready: {}\n";
        let mut f = Box::pin(from_yaml(
            yaml,
            provider.clone(),
            super::runner_tests::RecordingProvider::new(&[]),
            false,
        ))
        .await;
        if mode == "timeout" {
            let mut config = super::runner_tests::config(16);
            config.defaults.max_active_time_seconds = Some(1);
            f.runner = AutonomyRunner::try_new(
                f.agent.clone(),
                config,
                Default::default(),
                f.store.clone(),
                "state.v1".into(),
            )
            .unwrap();
        }
        let mut run = Box::pin(f.runner.run("objective", None));
        if mode == "drop" {
            tokio::select! {result=&mut run=>panic!("provider not held: {result:?}"),permit=provider.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("provider not reached") }
            drop(run);
        } else {
            let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), run)
                .await
                .unwrap();
            assert!(
                !outcome
                    .as_ref()
                    .is_ok_and(|r| r.run.status == TaskRunStatus::Completed)
            );
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.second.load(Ordering::SeqCst), 0);
        assert_eq!(f.agent.state_history().len(), 0);
        assert!(!f.agent.get_context().contains_key("old_result"));
        if mode != "error" {
            assert!(f.agent.set_context("blocked", json!(true)).is_err());
        }
    }
}

#[tokio::test]
async fn state_acceptance_pre_response_extractor_stages_once_and_commits_only_with_transition() {
    for reject in [false, true] {
        let yaml = "name: Staged\nsystem_prompt: test\nllm:\n  default: default\n  router: router\ntools: [first,second]\nruntime:\n  optimization:\n    enabled: true\n    pre_response_deterministic_transitions: true\nhitl:\n  message_language:\n    strategy: explicit\n    explicit: en\n  tools:\n    second:\n      require_approval: true\n  states:\n    ready:\n      on_enter: require_approval\n      approval_message: Approve\nstates:\n  initial: active\n  states:\n    active:\n      extract:\n        - key: topic\n          description: Topic\n      on_exit:\n        - tool: first\n        - tool: second\n      transitions:\n        - to: ready\n          guard:\n            context:\n              topic:\n                eq: Ada\n          timing: pre_response\n          run_extractors: true\n    ready:\n      regenerate_on_enter: false\n      on_enter:\n        - tool: first\n          args: {topic: '{{ context.topic }}'}\n";
        let main = super::runner_tests::RecordingProvider::new(&["done"]);
        let router = super::runner_tests::RecordingProvider::new(&["Ada", "__NONE__"]);
        let f = Box::pin(from_yaml(yaml, main.clone(), router.clone(), false)).await;
        let first = f.runner.run("objective", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        assert_eq!(router.calls.load(Ordering::SeqCst), 1);
        assert!(!f.agent.get_context().contains_key("topic"));
        assert_eq!(f.first.load(Ordering::SeqCst), 0);
        let next = f
            .runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.as_ref().unwrap().id.clone(),
                    result: if reject {
                        ApprovalResult::Rejected { reason: None }
                    } else {
                        ApprovalResult::Approved
                    },
                },
            )
            .await
            .unwrap();
        let done = if reject {
            next
        } else {
            assert_eq!(next.run.status, TaskRunStatus::Paused);
            assert!(!f.agent.get_context().contains_key("topic"));
            assert_eq!(f.agent.state_history().len(), 0);
            approve(&f, &next.run).await.unwrap()
        };
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        // Rejection returns to the ordinary post-response extraction stage; that is new counted work, not replay of the staged transition candidate.
        assert_eq!(
            router.calls.load(Ordering::SeqCst),
            if reject { 2 } else { 1 }
        );
        assert_eq!(main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.agent.state_history().len(), usize::from(!reject));
        if !reject {
            assert_eq!(f.agent.get_context()["topic"], "Ada");
            assert_eq!(f.arguments.lock().last().unwrap()["topic"], "Ada");
        } else {
            assert!(!f.agent.get_context().contains_key("topic"));
        }
    }
}

#[tokio::test]
async fn state_acceptance_state_turn_timeout_resumes_ready_context_without_repeating_transition() {
    for entering in [false, true] {
        let yaml = format!(
            "name: TurnTimeout\nsystem_prompt: test\ntools: [first,second]\nhitl:\n  tools:\n    second:\n      require_approval: true\nstates:\n  initial: active\n  states:\n    active:\n      max_turns: 1\n      timeout_to: ready\n{}    ready:\n      regenerate_on_enter: false\n{}",
            if entering {
                String::new()
            } else {
                "      on_exit:\n        - tool: first\n        - tool: second\n".into()
            },
            if entering {
                "      on_enter:\n        - tool: first\n        - tool: second\n"
            } else {
                ""
            }
        );
        let main = super::runner_tests::RecordingProvider::new(&["done"]);
        let f = Box::pin(from_yaml(
            &yaml,
            main.clone(),
            super::runner_tests::RecordingProvider::new(&[]),
            false,
        ))
        .await;
        f.agent.state_machine().unwrap().increment_turn();
        let first = f.runner.run("objective", None).await.unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        assert_eq!(main.calls.load(Ordering::SeqCst), 0);
        let done = approve(&f, &first.run).await.unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(main.calls.load(Ordering::SeqCst), 1);
        assert_eq!(f.agent.state_history().len(), 1);
        assert_eq!(f.first.load(Ordering::SeqCst), 1);
        assert_eq!(f.hooks.transitions.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn state_acceptance_empty_reentry_list_falls_back_to_enter_without_repeated_prefix() {
    let yaml = "name: ReentryFallback\nsystem_prompt: test\ntools: [first,second]\nhitl:\n  tools:\n    second:\n      require_approval: true\nstates:\n  initial: active\n  states:\n    active:\n      regenerate_on_enter: false\n      on_enter:\n        - tool: first\n        - tool: second\n      on_reenter: []\n      transitions:\n        - to: active\n          guard: 'true'\n";
    let f = Box::pin(from_yaml(
        yaml,
        super::runner_tests::RecordingProvider::new(&["done"]),
        super::runner_tests::RecordingProvider::new(&[]),
        false,
    ))
    .await;
    let first = f.runner.run("objective", None).await.unwrap();
    let done = approve(&f, &first.run).await.unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.first.load(Ordering::SeqCst), 1);
    assert_eq!(f.second.load(Ordering::SeqCst), 1);
    assert_eq!(f.agent.state_history().len(), 1);
}

#[tokio::test]
async fn state_acceptance_pending_action_admission_and_settlement_receipts_keep_phase_custody() {
    for sqlite in support::backends() {
        for entering in [false, true] {
            for phase in ["admit", "settle", "consume"] {
                for mode in ["positive", "reject", "lost", "drop"] {
                    let mut f = Box::pin(fixture(entering, false, false, *sqlite)).await;
                    let fault = ReceiptStore::checkpoint(
                        f.store.clone(),
                        mode,
                        json!({"tool":"second","phase":phase}),
                    );
                    f.runner = runner(f.agent.clone(), fault.clone());
                    let first = f.runner.run("objective", None).await.unwrap();
                    if phase == "consume" {
                        *fault.checkpoint.lock() =
                            Some(json!({"consume":first.run.pending.as_ref().unwrap().id}));
                    }
                    fault.armed.store(true, Ordering::SeqCst);
                    let mut resume = Box::pin(approve(&f, &first.run));
                    if mode == "positive" {
                        assert_eq!(resume.await.unwrap().run.status, TaskRunStatus::Completed);
                        fault.assert_hit();
                    } else {
                        if mode == "drop" {
                            tokio::select! {result=&mut resume=>panic!("checkpoint not held: {result:?}"),permit=fault.reached.acquire()=>permit.unwrap().forget(),_=tokio::time::sleep(std::time::Duration::from_secs(5))=>panic!("checkpoint not reached") }
                            drop(resume);
                        } else {
                            assert!(resume.await.is_err());
                        }
                        fault.assert_hit();
                        assert_eq!(
                            f.second.load(Ordering::SeqCst),
                            usize::from(phase != "admit")
                        );
                        assert_eq!(f.agent.state_history().len(), usize::from(entering));
                        assert!(f.agent.set_context("blocked", json!(true)).is_err());
                        assert!(approve(&f, &first.run).await.is_err());
                    }
                    assert_eq!(f.first.load(Ordering::SeqCst), 1);
                }
            }
        }
    }
}
