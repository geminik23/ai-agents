//! Deterministic acceptance of actual task counters and source-aware model loop limits.

use ai_agents_core::{
    FinishReason, LLMFeature, LLMResponse, Tool, ToolCall, ToolChoice, ToolExecutionContext,
    ToolResult, ToolSafetyMetadata,
};
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_runtime::{Agent, AgentBuilder, autonomy::*};
use ai_agents_tools::{CommandRequest, CommandResponse, CommandRunner};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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
        "Count actual isolated observations"
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
        ToolResult::ok("observed")
    }
}

// Nontrivial bounded output exercises real retained evidence growth without accessing external files or services.
struct PayloadProbe(Arc<AtomicUsize>);
#[async_trait]
impl Tool for PayloadProbe {
    fn id(&self) -> &str {
        "probe"
    }
    fn name(&self) -> &str {
        "probe"
    }
    fn description(&self) -> &str {
        "Return bounded isolated observation data"
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
        ToolResult::ok(format!("observation:{}", "x".repeat(1024)))
    }
}

struct CommandProbe(Arc<AtomicUsize>);
#[async_trait]
impl CommandRunner for CommandProbe {
    async fn run_command(&self, _: CommandRequest, _: ToolExecutionContext) -> CommandResponse {
        self.0.fetch_add(1, Ordering::SeqCst);
        CommandResponse {
            success: true,
            exit_code: Some(0),
            termination: "exited".into(),
            ..Default::default()
        }
    }
}

fn call(id: String, tool: &str) -> ToolCall {
    ToolCall {
        id,
        name: tool.into(),
        arguments: if tool == "command" {
            json!({"command":"original", "cwd":"."})
        } else {
            json!({})
        },
    }
}

fn mock(replies: Vec<LLMResponse>) -> Arc<MockLLMProvider> {
    let mut provider = MockLLMProvider::new("deterministic-task");
    provider.set_tool_choice(Some(ToolChoice::Auto));
    provider.set_feature_support(LLMFeature::FunctionCalling, true);
    for reply in replies {
        provider.add_response(reply);
    }
    Arc::new(provider)
}

fn tool_response(calls: Vec<ToolCall>) -> LLMResponse {
    LLMResponse::new("", FinishReason::ToolCall)
        .with_tool_calls(calls)
        .unwrap()
}

fn task_config() -> AutonomyConfig {
    serde_yaml::from_str("enabled: true\ncompletion: {response_contains: done}\n").unwrap()
}

fn runner(
    agent: Arc<ai_agents_runtime::RuntimeAgent>,
    host: AutonomyHostCeilings,
) -> (AutonomyRunner, Arc<ScopedTaskRunStore>) {
    runner_with_config(agent, task_config(), host)
}

fn runner_with_config(
    agent: Arc<ai_agents_runtime::RuntimeAgent>,
    config: AutonomyConfig,
    host: AutonomyHostCeilings,
) -> (AutonomyRunner, Arc<ScopedTaskRunStore>) {
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "execution-defaults.v1".into())
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        config,
        host,
        store.clone(),
        "execution-defaults.v1".into(),
    )
    .unwrap();
    (runner, store)
}

// The four previous cumulative ceilings are crossed through real model, executor, command-runner and controller admission.
#[tokio::test]
async fn default_task_exceeds_previous_cumulative_call_and_turn_limits() {
    let mut replies = Vec::new();
    let mut commands = 0usize;
    for cycle in 0..31 {
        let mut calls = Vec::new();
        for item in 0..4 {
            let tool = if commands < 11 {
                commands += 1;
                "command"
            } else {
                "probe"
            };
            calls.push(call(format!("cycle-{cycle}-call-{item}"), tool));
        }
        replies.push(tool_response(calls));
        replies.push(LLMResponse::new(
            if cycle == 30 { "done" } else { "working" },
            FinishReason::Stop,
        ));
    }
    let provider = mock(replies);
    let observations = Arc::new(AtomicUsize::new(0));
    let commands = Arc::new(AtomicUsize::new(0));
    let yaml = "name: long-task\nsystem_prompt: test\ntools: [probe, command]\ntool_security:\n  enabled: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      allowed_commands: [{argv: [original]}]\n      working_dirs: [.]\n";
    let agent = Arc::new(
        AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(PayloadProbe(observations.clone())))
            .build()
            .unwrap(),
    );
    agent.set_command_runner(Arc::new(CommandProbe(commands.clone())));
    let (runner, store) = runner(agent, AutonomyHostCeilings::default());
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(600),
        runner.run("objective", None),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        result.run.status,
        TaskRunStatus::Completed,
        "reason={:?}; counters={:?}; provider={}; probe={}; commands={}; last_evidence={:?}; native_history_error={:?}",
        result.run.stop_reason,
        result.run.counters,
        provider.call_count(),
        observations.load(Ordering::SeqCst),
        commands.load(Ordering::SeqCst),
        result.run.evidence.records.last(),
        ai_agents_core::inspect_native_history(&provider.last_call().unwrap().messages).err()
    );
    assert_eq!(result.run.counters.turns, 31);
    assert_eq!(result.run.counters.llm_attempts, 62);
    assert_eq!(result.run.counters.tool_attempts, 124);
    assert_eq!(result.run.counters.command_attempts, 11);
    assert_eq!(provider.call_count(), 62);
    assert_eq!(observations.load(Ordering::SeqCst), 113);
    assert_eq!(commands.load(Ordering::SeqCst), 11);
    assert_eq!(
        result
            .run
            .evidence
            .tool_calls
            .iter()
            .filter(|item| item.record.executed)
            .count(),
        124
    );
    assert_eq!(result.run.limits.max_turns, None);
    assert_eq!(result.run.limits.max_llm_calls, None);
    assert_eq!(result.run.limits.max_tool_calls, None);
    assert_eq!(result.run.limits.max_command_calls, None);
    assert_eq!(result.run.limits.max_active_time_seconds, None);
    let snapshot = store.load(&result.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(
        snapshot.payload["controller_state"]["remaining_llm_calls"],
        "unlimited"
    );
    let checkpoint_bytes = serde_json::to_vec(&snapshot).unwrap().len();
    assert!(checkpoint_bytes < ai_agents_core::autonomy::MAX_TASK_CHECKPOINT_BYTES);
    snapshot.validate().unwrap();
    let final_input = provider.last_call().unwrap().messages;
    let estimated_tokens: u32 = final_input
        .iter()
        .map(|message| (message.content.len() as f32 / 4.0).ceil() as u32)
        .sum();
    assert!(estimated_tokens <= 128_000);
    assert_eq!(result.run.evidence.tool_calls.len(), 124);
    assert!(
        result
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|entry| entry.record.output.len() >= 1024)
    );
    let reservation_count = snapshot.payload["reservations"].as_array().unwrap().len();
    assert!(reservation_count >= 124);
    let history = ai_agents_core::inspect_native_history(&final_input).unwrap();
    assert!(!history.has_incomplete_unsigned());
    assert!(
        final_input
            .iter()
            .any(|message| message.content.contains("objective"))
    );
    eprintln!(
        "long-task checkpoint_bytes={checkpoint_bytes}, final_request_messages={}, estimated_tokens={estimated_tokens}, evidence_records={}, reservations={reservation_count}",
        final_input.len(),
        result.run.evidence.records.len()
    );
}

// Finite task requests and independent host ceilings stop before an additional actual invocation.
#[tokio::test]
async fn finite_task_and_host_caps_stop_at_real_admission_boundaries() {
    for resource in ["turn", "llm", "tool", "command"] {
        for host_bound in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let commands = Arc::new(AtomicUsize::new(0));
            let tool = if resource == "command" {
                "command"
            } else {
                "probe"
            };
            let replies = if resource == "tool" || resource == "command" {
                vec![
                    tool_response((0..3).map(|n| call(format!("limit-{n}"), tool)).collect()),
                    LLMResponse::new("done", FinishReason::Stop),
                ]
            } else {
                vec![
                    LLMResponse::new("working", FinishReason::Stop),
                    LLMResponse::new("working", FinishReason::Stop),
                    LLMResponse::new("done", FinishReason::Stop),
                ]
            };
            let provider = mock(replies);
            let yaml = "name: limits\nsystem_prompt: test\ntools: [probe, command]\ntool_security:\n  enabled: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      allowed_commands: [{argv: [original]}]\n      working_dirs: [.]\n";
            let agent = Arc::new(
                AgentBuilder::from_yaml(yaml)
                    .unwrap()
                    .llm(provider.clone())
                    .auto_configure_features()
                    .unwrap()
                    .tool(Arc::new(Probe(calls.clone())))
                    .build()
                    .unwrap(),
            );
            agent.set_command_runner(Arc::new(CommandProbe(commands.clone())));
            let mut config = task_config();
            let mut host = AutonomyHostCeilings::default();
            match (resource, host_bound) {
                ("turn", false) => config.defaults.max_turns = Some(AutonomyLimit::Finite(2)),
                ("llm", false) => config.defaults.max_llm_calls = Some(AutonomyLimit::Finite(2)),
                ("tool", false) => config.defaults.max_tool_calls = Some(AutonomyLimit::Finite(2)),
                ("command", false) => {
                    config.defaults.max_command_calls = Some(AutonomyLimit::Finite(2))
                }
                ("turn", true) => host.max_turns = Some(2),
                ("llm", true) => host.max_llm_calls = Some(2),
                ("tool", true) => host.max_tool_calls = Some(2),
                ("command", true) => host.max_command_calls = Some(2),
                _ => unreachable!(),
            }
            let (runner, _) = runner_with_config(agent, config, host);
            let result = runner.run("objective", None).await.unwrap();
            assert_eq!(
                result.run.status,
                TaskRunStatus::LimitReached,
                "resource={resource} host={host_bound} reason={:?}",
                result.run.stop_reason
            );
            match resource {
                "turn" | "llm" => {
                    assert_eq!(provider.call_count(), 2);
                    assert_eq!(result.run.counters.turns, 2);
                }
                "tool" => {
                    assert_eq!(calls.load(Ordering::SeqCst), 2);
                    assert_eq!(result.run.counters.tool_attempts, 2);
                }
                "command" => {
                    assert_eq!(commands.load(Ordering::SeqCst), 2);
                    assert_eq!(result.run.counters.command_attempts, 2);
                    assert_eq!(result.run.counters.tool_attempts, 2);
                }
                _ => unreachable!(),
            }
        }
    }
}

// A parent invocation deadline remains binding through a task pause even when all aggregate budgets are absent.
#[tokio::test]
async fn inherited_invocation_deadline_survives_unlimited_pause_and_allows_cleanup() {
    let question = ToolCall {
        id: "question-bound".into(),
        name: "ask_user".into(),
        arguments: json!({"question":"Continue?","options":["yes"],"allow_other":false}),
    };
    let provider = mock(vec![
        tool_response(vec![question]),
        LLMResponse::new("must not run", FinishReason::Stop),
    ]);
    let agent = Arc::new(
        AgentBuilder::from_yaml("name: inherited-bound\nsystem_prompt: test\ntools: [ask_user]\n")
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let mut config = task_config();
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let (runner, store) = runner_with_config(agent, config, Default::default());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let bounds = ai_agents_llm::InvocationBounds::new(
        std::time::SystemTime::now() + std::time::Duration::from_secs(60),
        deadline,
    );
    let paused =
        ai_agents_llm::scope_invocation_bounds(Some(bounds), runner.run("objective", None))
            .await
            .unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert!(!snapshot.payload["clocks"]["inherited_expires_at"].is_null());
    assert!(snapshot.payload["controller_state"]["remaining_wall_seconds"].is_number());
    let mut limits = paused.run.limits.clone();
    limits.max_wall_time_seconds = Some(20);
    let revised = runner
        .revise_limits_paused(
            &paused.run.key.run_id,
            paused.run.revision,
            limits,
            "test original bound",
        )
        .await
        .unwrap();
    let revised_snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(
        revised_snapshot.payload["clocks"]["expires_at"],
        snapshot.payload["clocks"]["inherited_expires_at"]
    );
    tokio::time::sleep_until(deadline.into()).await;
    let pending = revised.run.pending.as_ref().unwrap();
    let resumed = runner
        .resume(
            &revised.run.key.run_id,
            revised.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: pending.id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await;
    assert!(resumed.is_err());
    assert_eq!(provider.call_count(), 1);
    let retained = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(retained.status, TaskRunStatus::Paused);
    assert_eq!(retained.revision, revised.run.revision);
    let cleanup = runner
        .cancel_task(&revised.run.key.run_id, revised.run.revision)
        .await
        .unwrap();
    assert_eq!(cleanup.disposition, CancellationDisposition::Acknowledged);
}

// An unbounded pause resumes the exact pending question without replaying its model call or resetting usage.
#[tokio::test]
async fn unlimited_pause_resume_preserves_counters_and_pending_cursor() {
    let question = ToolCall {
        id: "question-1".into(),
        name: "ask_user".into(),
        arguments: json!({"question":"Continue?","options":["yes"],"allow_other":false}),
    };
    let provider = mock(vec![
        tool_response(vec![question]),
        LLMResponse::new("done", FinishReason::Stop),
    ]);
    let agent = Arc::new(
        AgentBuilder::from_yaml("name: pause-unlimited\nsystem_prompt: test\ntools: [ask_user]\n")
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    let mut config = task_config();
    config.defaults.hitl = Some(AutonomyHitlConfig {
        on_approval_required: Some(InteractionAction::PauseRun),
        on_user_question: Some(InteractionAction::PauseRun),
    });
    let (runner, store) = runner_with_config(agent, config, Default::default());
    let paused = runner.run("objective", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(paused.run.counters.turns, 1);
    assert_eq!(paused.run.counters.llm_attempts, 1);
    assert_eq!(paused.run.limits.max_llm_calls, None);
    let snapshot = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
    assert_eq!(snapshot.payload["version"], TASK_PAYLOAD_VERSION);
    let interaction = runner
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(interaction.prompt, TaskHostPrompt::Question(_)));
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: interaction.request_id,
                answer: json!({"answered":true,"selected":["yes"]}),
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
    assert_eq!(done.run.counters.turns, 1);
    assert_eq!(done.run.counters.llm_attempts, 2);
    assert_eq!(provider.call_count(), 2);
    assert_eq!(done.run.limits.max_llm_calls, None);
}

// Typed and source-aware namespaces must not alias, and late numeric overrides cannot change a bound declaration.
#[test]
fn source_aware_host_scope_rejects_late_limit_overrides() {
    let prepared = ai_agents_runtime::spec::PreparedAgentSpec::from_yaml_strict(
        "name: scoped\nsystem_prompt: test\nmax_iterations: 10\n",
    )
    .unwrap();
    let scope =
        TaskHostScope::from_prepared_spec(&prepared, std::path::Path::new("."), None).unwrap();
    let legacy =
        TaskHostScope::from_spec(prepared.spec(), std::path::Path::new("."), None).unwrap();
    assert_ne!(scope.identity(), legacy.identity());
    let mut provider = MockLLMProvider::new("scope");
    provider.set_response("done");
    let builder = AgentBuilder::from_prepared_spec(prepared)
        .llm(Arc::new(provider))
        .task_host_scope(scope)
        .unwrap()
        .max_iterations(100);
    assert!(builder.build().is_err());
}

// Spawned YAML children retain source presence, while typed children conservatively keep their materialized limits.
#[tokio::test]
async fn spawned_child_task_distinguishes_omitted_explicit_and_typed_iterations() {
    for mode in ["omitted", "explicit", "typed"] {
        let mut replies = (0..11)
            .map(|index| {
                tool_response(vec![ToolCall {
                    id: format!("child-{index}"),
                    name: "echo".into(),
                    arguments: json!({"message":"isolated observation"}),
                }])
            })
            .collect::<Vec<_>>();
        replies.push(LLMResponse::new("done", FinishReason::Stop));
        let provider = mock(replies);
        let mut registry = ai_agents_llm::LLMRegistry::new();
        registry.register("default", ai_agents_llm::managed_provider(provider.clone()));
        registry.set_default("default");
        let spawner = ai_agents_runtime::spawner::AgentSpawner::new().with_shared_llms(registry);
        let yaml = format!(
            "name: child\nsystem_prompt: test\ntools: [echo]\n{}",
            if mode == "explicit" {
                "max_iterations: 10\n"
            } else {
                ""
            }
        );
        let child = if mode == "typed" {
            let spec = ai_agents_runtime::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
            spawner.spawn_from_spec(spec).await.unwrap()
        } else {
            spawner.spawn_from_yaml(&yaml).await.unwrap()
        };
        let (runner, _) = runner(child.agent.clone(), AutonomyHostCeilings::default());
        let result = runner.run("objective", None).await.unwrap();
        let expected = if mode == "omitted" { 12 } else { 10 };
        assert_eq!(provider.call_count(), expected, "{mode}");
        assert_eq!(result.run.counters.llm_attempts, expected as u64, "{mode}");
        assert_eq!(
            result.run.status == TaskRunStatus::Completed,
            mode == "omitted",
            "{mode}"
        );
        assert_eq!(
            result
                .run
                .evidence
                .tool_calls
                .iter()
                .filter(|entry| entry.record.executed)
                .count(),
            if mode == "omitted" { 11 } else { 10 },
            "{mode}"
        );
    }
}

// Full state redispatch keeps the original response's explicit budget instead of issuing a fresh one.
#[tokio::test]
async fn task_full_state_redispatch_preserves_explicit_loop_limit() {
    for explicit in [false, true] {
        let provider = mock(vec![
            LLMResponse::new("first", FinishReason::Stop),
            LLMResponse::new("done", FinishReason::Stop),
        ]);
        let yaml = format!(
            "name: redispatch\nsystem_prompt: test\n{}states:\n  initial: first\n  states:\n    first:\n      transitions:\n        - to: second\n          guard: 'true'\n    second:\n      reasoning: {{mode: cot}}\n",
            if explicit { "max_iterations: 1\n" } else { "" }
        );
        let agent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        let (runner, _) = runner(agent, AutonomyHostCeilings::default());
        let result = runner.run("objective", None).await.unwrap();
        if explicit {
            assert_ne!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(provider.call_count(), 1);
        } else {
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                result.run.stop_reason
            );
            assert_eq!(provider.call_count(), 2);
        }
    }
}

// The reasoning struct's materialized default of five is not a task cap unless it was explicitly supplied.
#[tokio::test]
async fn task_reasoning_omission_and_explicit_five_remain_distinct() {
    for explicit in [false, true] {
        let mut replies = (0..6)
            .map(|n| tool_response(vec![call(format!("reason-{n}"), "probe")]))
            .collect::<Vec<_>>();
        replies.push(LLMResponse::new("done", FinishReason::Stop));
        let provider = mock(replies);
        let calls = Arc::new(AtomicUsize::new(0));
        let yaml = format!(
            "name: reasoning-task\nsystem_prompt: test\ntools: [probe]\nreasoning: {}\n",
            if explicit {
                "{mode: cot, max_iterations: 5}"
            } else {
                "{mode: cot}"
            }
        );
        let agent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider.clone())
                .tool(Arc::new(Probe(calls.clone())))
                .build()
                .unwrap(),
        );
        let (runner, _) = runner(agent, AutonomyHostCeilings::default());
        let result = runner.run("objective", None).await.unwrap();
        if explicit {
            assert_ne!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(provider.call_count(), 5);
            assert_eq!(calls.load(Ordering::SeqCst), 5);
        } else {
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                result.run.stop_reason
            );
            assert_eq!(provider.call_count(), 7);
            assert_eq!(calls.load(Ordering::SeqCst), 6);
        }
    }
}

// Source-aware task policy must not relax ordinary streaming's default or explicit loop limits.
#[tokio::test]
async fn ordinary_streaming_retains_default_and_explicit_iteration_limits() {
    for (setting, expected) in [("", 10), ("max_iterations: 3\n", 3)] {
        let mut mock = MockLLMProvider::new("ordinary-stream");
        for _ in 0..12 {
            mock.add_response(LLMResponse::new(
                r#"{"tool":"probe","arguments":{}}"#,
                FinishReason::Stop,
            ));
        }
        let provider = Arc::new(mock);
        let effects = Arc::new(AtomicUsize::new(0));
        let yaml = format!("name: ordinary-stream\nsystem_prompt: test\ntools: [probe]\n{setting}");
        let agent = AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .tool(Arc::new(Probe(effects.clone())))
            .build()
            .unwrap();
        let mut stream = agent.chat_stream("objective").await.unwrap();
        let mut error = None;
        while let Some(chunk) = stream.next().await {
            if let ai_agents_runtime::StreamChunk::Error { message } = chunk {
                error = Some(message);
            }
        }
        assert!(error.is_some_and(|message| message.contains("Max iterations")));
        assert_eq!(provider.call_count(), expected);
        assert_eq!(effects.load(Ordering::SeqCst), expected);
    }
}

// The same response loop counts the final answer: omission can pass ten, explicit ten cannot call an eleventh time.
#[tokio::test]
async fn task_loop_omission_and_explicit_ten_differ_without_changing_ordinary_chat() {
    for (explicit, task) in [(false, true), (true, true), (false, false)] {
        let mut replies = (0..12)
            .map(|n| tool_response(vec![call(format!("call-{n}"), "probe")]))
            .collect::<Vec<_>>();
        replies.push(LLMResponse::new("done", FinishReason::Stop));
        let provider = mock(replies);
        let calls = Arc::new(AtomicUsize::new(0));
        let yaml = format!(
            "name: loop-task\nsystem_prompt: test\ntools: [probe]\n{}",
            if explicit { "max_iterations: 10\n" } else { "" }
        );
        let agent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider.clone())
                .tool(Arc::new(Probe(calls.clone())))
                .build()
                .unwrap(),
        );
        if task {
            let (runner, _) = runner(agent, AutonomyHostCeilings::default());
            let result = runner.run("objective", None).await.unwrap();
            if explicit {
                assert_ne!(result.run.status, TaskRunStatus::Completed);
                assert_eq!(provider.call_count(), 10);
                assert_eq!(calls.load(Ordering::SeqCst), 10);
            } else {
                assert_eq!(
                    result.run.status,
                    TaskRunStatus::Completed,
                    "{:?}",
                    result.run.stop_reason
                );
                assert_eq!(result.run.counters.turns, 1);
                assert_eq!(result.run.counters.llm_attempts, 13);
                assert_eq!(calls.load(Ordering::SeqCst), 12);
            }
        } else {
            assert!(agent.chat("objective").await.is_err());
            assert_eq!(provider.call_count(), 10);
            assert_eq!(calls.load(Ordering::SeqCst), 10);
        }
    }
}
