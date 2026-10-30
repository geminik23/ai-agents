use ai_agents_core::{
    FinishReason, LLMResponse, ToolCall, ToolCallSource, ToolExecutionContext, ToolExecutionRecord,
    ToolExecutionRequest, ToolInvoker,
};
use ai_agents_hooks::AgentHooks;
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_runtime::{Agent, AgentBuilder, RuntimeAgent, autonomy::*};
use ai_agents_tools::{
    CommandRequest, CommandResponse, CommandRunner, FileEditTool, ToolPolicyConfig,
    ToolSecurityConfig, ToolSecurityEngine,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct OutputCommandProbe {
    exit_code: i32,
    reported_success: bool,
    timed_out: bool,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl CommandRunner for OutputCommandProbe {
    // The shared executor still owns admission and evidence; this host supplies only the command result.
    async fn run_command(&self, _: CommandRequest, _: ToolExecutionContext) -> CommandResponse {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let stdout = format!("START {} END", "한글\\\"".repeat(500));
        let stderr = format!("{} FINAL-ERROR", "warning ".repeat(500));
        CommandResponse {
            success: self.reported_success,
            exit_code: Some(self.exit_code),
            termination: "exited".into(),
            timed_out: self.timed_out,
            combined_output: format!("{stdout}\n{stderr}"),
            stdout,
            stderr,
            ..Default::default()
        }
    }
}

// The configured builtin command still passes through the production registry, policy and shared executor.
fn command_agent(cap: usize, exit_code: i32, calls: Arc<AtomicUsize>) -> Arc<RuntimeAgent> {
    let yaml = format!(
        "name: OutputEvidence\nsystem_prompt: test\ntools: [command]\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      max_output_chars: {cap}\n      allowed_commands: [{{argv: [original]}}]\n      working_dirs: [.]\n"
    );
    let mut provider = MockLLMProvider::new("output-evidence");
    provider.set_response("done");
    let agent = Arc::new(
        AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(Arc::new(provider))
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(AutonomyExtensions::builtins())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(Arc::new(OutputCommandProbe {
        exit_code,
        reported_success: exit_code == 0,
        timed_out: false,
        calls,
    }));
    agent
}

// This ignored helper is launched only by the isolated process-backed command test below.
#[test]
#[ignore = "subprocess helper"]
fn command_output_helper() {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    write!(stdout, "START {} END", "가".repeat(2_000)).unwrap();
    write!(stderr, "{} FINAL-ERROR-🧪", "warning ".repeat(1_000)).unwrap();
    stdout.flush().unwrap();
    stderr.flush().unwrap();
    std::process::exit(1);
}

// A real isolated process, builtin command, executor, validator and direct gate share one bounded exit result.
#[tokio::test]
async fn process_output_reaches_the_real_executor_and_completion_gates() {
    let executable = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let argv = vec![
        executable.clone(),
        "command_output_helper".into(),
        "--exact".into(),
        "--ignored".into(),
        "--nocapture".into(),
    ];
    let command = format!("\"{executable}\" command_output_helper --exact --ignored --nocapture");
    let yaml = format!(
        "name: ProcessEvidence\nsystem_prompt: test\ntools: [command]\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    command:\n      allow_without_confirmation: true\n      max_output_chars: 512\n      allowed_commands: [{{argv: {}}}]\n      working_dirs: [.]\n",
        serde_json::to_string(&argv).unwrap()
    );
    let mut provider = MockLLMProvider::new("process-evidence");
    provider.set_response("done");
    let agent = Arc::new(
        AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(Arc::new(provider))
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(AutonomyExtensions::builtins())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(Arc::new(ai_agents_tools::ProcessCommandRunner));
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "process.v1".into()).unwrap(),
    );
    let quoted_command = serde_json::to_string(&command).unwrap();
    let profile = format!(
        "enabled: true\nmax_turns: 4\nmax_llm_calls: 1\nvalidation:\n  checks: [{{id: check, adapter: builtin.command, config: {{command: {quoted_command}, expected_exit: 1}}}}]\ncompletion: {{all: [{{validator_passed: check}}, {{command_exit: {{command: {quoted_command}, code: 1}}}}]}}\n"
    );
    let runner = AutonomyRunner::try_new(
        agent,
        AutonomyConfig {
            defaults: serde_yaml::from_str(&profile).unwrap(),
            ..Default::default()
        },
        Default::default(),
        store,
        "process.v1".into(),
    )
    .unwrap();
    let result = runner.run("objective", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    let record = &result
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|evidence| evidence.record.canonical_id == "command")
        .expect("actual process command record")
        .record;
    assert!(record.executed);
    assert!(!record.output_truncated);
    let output: Value = serde_json::from_str(&record.output).unwrap();
    assert_eq!(output["exit_code"], 1);
    assert_eq!(output["success"], false);
    assert_eq!(output["truncated"], true);
    assert!(
        output["combined_output"]
            .as_str()
            .unwrap()
            .contains("FINAL-ERROR-🧪")
    );
}

// Validator and direct completion gates consume the same actual bounded record without conflating their predicates.
#[tokio::test]
async fn command_validation_uses_the_actual_bounded_record() {
    for (exit_code, expected_exit, gate_code, timed_out, reported_success, should_complete) in [
        (0, 0, 0, false, true, true),
        (1, 1, 1, false, false, true),
        (1, 0, 1, false, false, false),
        (0, 0, 1, false, true, false),
        (0, 0, 0, true, true, false),
        (0, 0, 0, false, false, false),
        (1, 1, 1, false, true, false),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let agent = command_agent(512, exit_code, calls.clone());
        agent.set_command_runner(Arc::new(OutputCommandProbe {
            exit_code,
            reported_success,
            timed_out,
            calls: calls.clone(),
        }));
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "output.v1".into()).unwrap(),
        );
        let profile = format!(
            "enabled: true\nmax_turns: 4\nmax_llm_calls: 1\nvalidation:\n  checks: [{{id: check, adapter: builtin.command, config: {{command: original, expected_exit: {expected_exit}}}}}]\ncompletion: {{all: [{{validator_passed: check}}, {{command_exit: {{command: original, code: {gate_code}}}}}]}}\n"
        );
        let config = AutonomyConfig {
            defaults: serde_yaml::from_str(&profile).unwrap(),
            ..Default::default()
        };
        let runner =
            AutonomyRunner::try_new(agent, config, Default::default(), store, "output.v1".into())
                .unwrap();
        let result = runner.run("objective", None).await.unwrap();
        assert_eq!(
            result.run.status == TaskRunStatus::Completed,
            should_complete
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let record = &result
            .run
            .evidence
            .tool_calls
            .iter()
            .find(|evidence| evidence.record.canonical_id == "command")
            .expect("actual command record")
            .record;
        assert!(record.executed);
        assert!(record.success, "tool success is distinct from process exit");
        assert!(!record.output_truncated);
        assert!(record.output.chars().count() <= 512);
        let output: Value = serde_json::from_str(&record.output).unwrap();
        assert_eq!(output["exit_code"], exit_code);
        assert_eq!(output["truncated"], true);
        assert!(
            output["combined_output"]
                .as_str()
                .unwrap()
                .contains("FINAL-ERROR")
        );
    }
}

struct TimeoutCommandProbe(Arc<AtomicUsize>);

#[async_trait]
impl CommandRunner for TimeoutCommandProbe {
    // A returned timeout is distinct from an exited command even when the tool itself executed successfully.
    async fn run_command(&self, _: CommandRequest, _: ToolExecutionContext) -> CommandResponse {
        self.0.fetch_add(1, Ordering::SeqCst);
        CommandResponse {
            success: false,
            termination: "timeout".into(),
            timed_out: true,
            stderr: "command timed out".into(),
            combined_output: "command timed out".into(),
            ..Default::default()
        }
    }
}

// Denied requests never invoke the host, and a returned timeout cannot prove command completion.
#[tokio::test]
async fn denied_and_timed_out_commands_never_prove_completion() {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = command_agent(512, 0, calls.clone());
    let denied = agent
        .invoke_tool(ToolExecutionRequest::new(
            "denied-command",
            "command",
            json!({"command":"not-allowed","cwd":"."}),
            ToolCallSource::Manual,
        ))
        .await
        .unwrap();
    assert!(!denied.executed);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    agent.set_command_runner(Arc::new(TimeoutCommandProbe(calls.clone())));
    let profile: AutonomyProfile = serde_yaml::from_str(
        "enabled: true\nmax_turns: 4\nmax_llm_calls: 1\nvalidation:\n  checks: [{id: check, adapter: builtin.command, config: {command: original}}]\ncompletion: {all: [{validator_passed: check}, {command_exit: {command: original, code: 0}}]}\n",
    )
    .unwrap();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "timeout.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        AutonomyConfig {
            defaults: profile,
            ..Default::default()
        },
        Default::default(),
        store,
        "timeout.v1".into(),
    )
    .unwrap();
    let result = runner.run("objective", None).await.unwrap();
    assert_ne!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let record = &result
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|evidence| evidence.record.canonical_id == "command")
        .expect("actual timed-out command record")
        .record;
    assert!(record.executed);
    let output: Value = serde_json::from_str(&record.output).unwrap();
    assert_eq!(output["termination"], "timeout");
    assert_eq!(output["timed_out"], true);
}

// A tiny cap cannot carry a full command envelope and must not be mistaken for malformed host data or a passing result.
#[tokio::test]
async fn tiny_command_envelope_and_malformed_host_output_are_distinct() {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = command_agent(16, 0, calls.clone());
    let arguments = json!({"command":"original","cwd":"."});
    let actual = agent
        .invoke_tool(ToolExecutionRequest::new(
            "command-call",
            "command",
            arguments.clone(),
            ToolCallSource::Manual,
        ))
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(actual.executed);
    assert!(actual.success);
    assert!(actual.output_truncated);
    assert!(actual.output.chars().count() <= 16);

    let profile: AutonomyProfile = serde_json::from_value(json!({
        "validation":{"checks":[{"id":"check","adapter":"builtin.command","config":{"command":"original"}}]}
    }))
    .unwrap();
    let bound = agent
        .autonomy_extensions()
        .bind(
            &profile,
            &ValidationCapabilities {
                tools: BTreeSet::from(["command".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let check = &bound.checks[0];
    let scope = EvaluationScope {
        key: TaskRunKey {
            agent_id: agent.info().id,
            run_id: "run".into(),
        },
        objective_revision: 0,
        cycle: 1,
        stage: None,
        mutation_generation: 0,
        target_revisions: BTreeMap::new(),
        target_generations: BTreeMap::new(),
        validation_bindings: BTreeMap::new(),
        validation_attempts: BTreeMap::new(),
        validation_dependencies: BTreeMap::new(),
    };
    let identity = EvidenceIdentity {
        key: scope.key.clone(),
        objective_revision: 0,
        cycle: 1,
        sequence: 1,
        stage: None,
        attempt: Some("attempt".into()),
        config_identity: check.config_identity.clone(),
        target: None,
        target_revision: None,
        mutation_generation: 0,
        validation_dependencies: BTreeMap::new(),
    };
    let evidence = EvaluationEvidence::default();
    let previous = json!({});
    let config = json!({"command":"original"});
    let request = ValidationObservationRequest::Tool {
        id: "observe".into(),
        tool: "command".into(),
        arguments,
        host_binding: None,
    };
    for truncated in [true, false] {
        let mut record = actual.clone();
        if !truncated {
            record.output = "{\"exit_code\":".into();
            record.output_truncated = false;
        }
        let observations = BTreeMap::from([(
            "observe".into(),
            CompletedObservation {
                identity: Some(identity.clone()),
                request: request.clone(),
                request_identity: "request".into(),
                effective_identity: "request".into(),
                result: ObservationResult::Tool {
                    record: Box::new(record),
                },
            },
        )]);
        let input = ValidationInput {
            scope: &scope,
            identity: &identity,
            config: &config,
            evidence: &evidence,
            previous: &previous,
            observations: &observations,
        };
        let error = check
            .adapter
            .evaluate(&input)
            .err()
            .expect("invalid command output must not pass")
            .to_string();
        if truncated {
            assert!(error.contains("output limit"), "{error}");
            assert!(error.contains("validity is unknown"), "{error}");
        } else {
            assert!(error.contains("malformed command observation"), "{error}");
        }
    }
}

struct TestWorkspace(std::path::PathBuf);

impl TestWorkspace {
    // A fresh directory keeps actual file effects separate from the developer's workspace.
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("ai-agents-edit-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestWorkspace {
    // Cleanup is limited to this test's uniquely owned directory.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct RecordHooks(parking_lot::Mutex<Vec<ToolExecutionRecord>>);

#[async_trait]
impl AgentHooks for RecordHooks {
    // The hook records the authoritative shared-executor result, not a model-suggested tool call.
    async fn on_tool_execution_record(&self, record: &ToolExecutionRecord) {
        self.0.lock().push(record.clone());
    }
}

// When an output cap cannot carry a complete receipt, the record and actual file state remain distinct from partial model text.
#[tokio::test]
async fn tiny_edit_limits_never_hide_actual_execution_in_the_record() {
    for (cap, missing, expected_truncation, applied) in [
        (16, true, true, false),
        (32, true, false, false),
        (130, true, false, false),
        (16, false, true, false),
        (48, false, false, true),
    ] {
        let workspace = TestWorkspace::new();
        let path = workspace.0.join("edit.txt");
        std::fs::write(&path, "a\n").unwrap();
        let mut provider = MockLLMProvider::new("small-edit-output");
        provider.set_response("done");
        let mut security = ToolSecurityConfig {
            enabled: true,
            fail_closed: true,
            ..Default::default()
        };
        security.tools.insert(
            "file_edit".into(),
            ToolPolicyConfig {
                write_paths: vec![workspace.0.to_string_lossy().into_owned()],
                allow_without_confirmation: true,
                max_output_chars: Some(cap),
                ..Default::default()
            },
        );
        let agent = AgentBuilder::new()
            .system_prompt("test")
            .llm(Arc::new(provider))
            .tool(Arc::new(FileEditTool::new()))
            .tool_security(ToolSecurityEngine::try_new(security).unwrap())
            .build()
            .unwrap();
        let record = agent
            .invoke_tool(ToolExecutionRequest::new(
                "small-edit",
                "file_edit",
                json!({"path":path.to_string_lossy(),"old_text":if missing {"absent"} else {"a"},"new_text":"aa","dry_run":false}),
                ToolCallSource::Manual,
            ))
            .await
            .unwrap();
        assert!(record.executed);
        assert_eq!(record.success, applied);
        assert_eq!(record.output_truncated, expected_truncation);
        assert!(record.output.chars().count() <= cap);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            if applied { "aa\n" } else { "a\n" }
        );
        if !missing && !applied {
            assert!(record.output.starts_with("edit not applied"));
        }
        if matches!(cap, 32 | 130) && missing {
            let output: Value = serde_json::from_str(&record.output).unwrap();
            assert_eq!(output["truncated"], true);
        }
        if cap == 48 && !missing {
            let output: Value = serde_json::from_str(&record.output).unwrap();
            assert_eq!(output["mutation_performed"], true);
            assert_eq!(output["truncated"], true);
        }
    }
}

#[derive(Clone)]
struct CapturedLogs(Arc<parking_lot::Mutex<Vec<String>>>);

struct LogFields<'a>(&'a mut Vec<String>);

impl tracing::field::Visit for LogFields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push(format!("{}={value:?}", field.name()));
    }
}

impl tracing::Subscriber for CapturedLogs {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        event.record(&mut LogFields(&mut self.0.lock()));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

// Failure details remain available to an authorized model but never enter retry or state-action logs.
#[tokio::test(flavor = "current_thread")]
async fn edit_context_is_not_written_to_retry_or_state_action_logs() {
    use ai_agents_recovery::{
        ErrorRecoveryConfig, RecoveryManager, ToolRecoveryConfig, ToolRetryConfig,
    };
    let logs = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let _guard = tracing::subscriber::set_default(CapturedLogs(logs.clone()));
    let workspace = TestWorkspace::new();
    let path = workspace.0.join("edit.txt");
    let before = "Status: private-fixture-value\n";
    std::fs::write(&path, before).unwrap();
    let mut security = ToolSecurityConfig {
        enabled: true,
        fail_closed: true,
        ..Default::default()
    };
    security.tools.insert(
        "file_edit".into(),
        ToolPolicyConfig {
            read_paths: vec![workspace.0.to_string_lossy().into_owned()],
            write_paths: vec![workspace.0.to_string_lossy().into_owned()],
            allow_without_confirmation: true,
            ..Default::default()
        },
    );
    let mut mock = MockLLMProvider::new("retry-privacy");
    mock.set_response("done");
    let agent = AgentBuilder::new()
        .system_prompt("test")
        .llm(Arc::new(mock))
        .tool(Arc::new(FileEditTool::new()))
        .tool_security(ToolSecurityEngine::try_new(security).unwrap())
        .hooks(Arc::new(ai_agents_hooks::LoggingHooks::new()))
        .recovery_manager(RecoveryManager::new(ErrorRecoveryConfig {
            tools: ToolRecoveryConfig {
                per_tool: HashMap::from([(
                    "file_edit".into(),
                    ToolRetryConfig {
                        max_retries: 1,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            },
            ..Default::default()
        }))
        .build()
        .unwrap();
    let record = agent
        .invoke_tool(ToolExecutionRequest::new(
            "edit-retry",
            "file_edit",
            json!({"path":path.to_string_lossy(),"old_text":"Status: ready","new_text":"Status: private-input-fixture","dry_run":true}),
            ToolCallSource::Manual,
        ))
        .await
        .unwrap();
    assert!(record.executed);
    assert!(!record.success);
    assert!(record.output.contains("private-fixture-value"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);

    let root = serde_json::to_string(&workspace.0.to_string_lossy()).unwrap();
    let file = serde_json::to_string(&path.to_string_lossy()).unwrap();
    let yaml = format!(
        "name: StateEdit\nsystem_prompt: test\ntools: [file_edit]\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_edit:\n      read_paths: [{root}]\n      write_paths: [{root}]\n      allow_without_confirmation: true\nstates:\n  initial: active\n  states:\n    active:\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready:\n      regenerate_on_enter: false\n      on_enter:\n        - tool: file_edit\n          args: {{path: {file}, old_text: 'Status: ready', new_text: 'Status: private-input-fixture', dry_run: true}}\n"
    );
    let mut mock = MockLLMProvider::new("state-action-privacy");
    mock.set_response("done");
    let state_agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(Arc::new(mock))
        .auto_configure_features()
        .unwrap()
        .hooks(Arc::new(ai_agents_hooks::LoggingHooks::new()))
        .build()
        .unwrap();
    state_agent.chat("go").await.unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    let logging = ai_agents_hooks::LoggingHooks::new();
    logging.on_message_received("private-fixture-value").await;
    logging
        .on_response(&ai_agents_core::AgentResponse::new("private-fixture-value"))
        .await;
    logging
        .on_group_chat_round(1, "worker", "private-fixture-value")
        .await;
    let logs = logs.lock().join("\n");
    assert!(logs.contains("Response received"), "{logs}");
    assert!(logs.contains("Group chat round"), "{logs}");
    assert!(logs.contains("Retrying failed tool call"), "{logs}");
    assert!(logs.contains("State action: tool failed"), "{logs}");
    assert!(!logs.contains("private-fixture-value"), "{logs}");
    assert!(!logs.contains("private-input-fixture"), "{logs}");
}

// Fallbacks reuse approved arguments and state-action skills can wrap tool failures, so neither path may log source text.
#[tokio::test(flavor = "current_thread")]
async fn edit_context_is_not_logged_through_fallback_or_skill() {
    let logs = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let _guard = tracing::subscriber::set_default(CapturedLogs(logs.clone()));
    let workspace = TestWorkspace::new();
    let path = workspace.0.join("edit.txt");
    let before = "Status: private-fixture-value\n";
    std::fs::write(&path, before).unwrap();
    let root = serde_json::to_string(&workspace.0.to_string_lossy()).unwrap();
    let file = serde_json::to_string(&path.to_string_lossy()).unwrap();
    let policy = format!(
        "tool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_edit:\n      read_paths: [{root}]\n      write_paths: [{root}]\n      allow_without_confirmation: true\n"
    );
    let args = json!({"path":path.to_string_lossy(),"old_text":"Status: ready","new_text":"Status: private-input-fixture","dry_run":true});
    let records = Arc::new(RecordHooks::default());
    let hooks = ai_agents_hooks::CompositeHooks::new()
        .add(Arc::new(ai_agents_hooks::LoggingHooks::new()))
        .add(records.clone());
    let fallback_yaml = format!(
        "name: EditFallback\nsystem_prompt: test\ntools: [file_edit, echo]\n{policy}error_recovery:\n  tools:\n    file_edit:\n      max_retries: 0\n      on_failure: {{action: fallback, fallback_tool: echo}}\n"
    );
    let mut provider = MockLLMProvider::new("fallback-privacy");
    provider.set_response("done");
    let fallback = AgentBuilder::from_yaml(&fallback_yaml)
        .unwrap()
        .llm(Arc::new(provider))
        .auto_configure_features()
        .unwrap()
        .hooks(Arc::new(hooks))
        .build()
        .unwrap();
    let _ = fallback
        .invoke_tool(ToolExecutionRequest::new(
            "fallback-edit",
            "file_edit",
            args.clone(),
            ToolCallSource::Manual,
        ))
        .await
        .unwrap();
    let records = records.0.lock().clone();
    assert!(
        records
            .iter()
            .any(|record| record.canonical_id == "file_edit" && record.executed)
    );
    assert!(records.iter().any(|record| record.requested_name == "echo"));
    assert!(
        records
            .iter()
            .any(|record| record.output.contains("private-fixture-value"))
    );

    let records = Arc::new(RecordHooks::default());
    let hooks = ai_agents_hooks::CompositeHooks::new()
        .add(Arc::new(ai_agents_hooks::LoggingHooks::new()))
        .add(records.clone());
    let skill_yaml = format!(
        "name: SkillState\nsystem_prompt: test\ntools: [file_edit]\n{policy}skills:\n  - id: edit_skill\n    description: Inspect an edit\n    trigger: Never choose automatically\n    steps:\n      - tool: file_edit\n        args: {{path: {file}, old_text: 'Status: ready', new_text: 'Status: private-input-fixture', dry_run: true}}\nstates:\n  initial: active\n  states:\n    active:\n      transitions:\n        - to: ready\n          guard: 'true'\n    ready:\n      regenerate_on_enter: false\n      on_enter:\n        - skill: edit_skill\n"
    );
    let mut provider = MockLLMProvider::new("skill-privacy");
    provider.set_response("done");
    let state_agent = AgentBuilder::from_yaml(&skill_yaml)
        .unwrap()
        .llm(Arc::new(provider))
        .auto_configure_features()
        .unwrap()
        .hooks(Arc::new(hooks))
        .build()
        .unwrap();
    state_agent.chat("go").await.unwrap();
    let records = records.0.lock();
    assert!(
        records
            .iter()
            .any(|record| record.canonical_id == "file_edit" && record.executed)
    );
    assert!(
        records
            .iter()
            .any(|record| record.output.contains("private-fixture-value"))
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    let logs = logs.lock().join("\n");
    assert!(logs.contains("State action: skill failed"), "{logs}");
    assert!(!logs.contains("private-fixture-value"), "{logs}");
    assert!(!logs.contains("private-input-fixture"), "{logs}");
}

// Pipeline errors remain available to the caller without copying skill-carried file context into warning logs.
#[tokio::test(flavor = "current_thread")]
async fn pipeline_skill_failure_keeps_context_out_of_logs() {
    let logs = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let _guard = tracing::subscriber::set_default(CapturedLogs(logs.clone()));
    let workspace = TestWorkspace::new();
    let path = workspace.0.join("edit.txt");
    std::fs::write(&path, "Status: private-fixture-value\n").unwrap();
    let root = serde_json::to_string(&workspace.0.to_string_lossy()).unwrap();
    let file = serde_json::to_string(&path.to_string_lossy()).unwrap();
    let yaml = format!(
        "name: SkillChild\nsystem_prompt: test\ntools: [file_edit]\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_edit:\n      read_paths: [{root}]\n      write_paths: [{root}]\n      allow_without_confirmation: true\nskills:\n  - id: edit_skill\n    description: Attempt an edit\n    trigger: Edit the status\n    steps:\n      - tool: file_edit\n        args: {{path: {file}, old_text: 'Status: ready', new_text: 'Status: private-input-fixture', dry_run: true}}\n"
    );
    let mut provider = MockLLMProvider::new("pipeline-privacy");
    provider.set_response("edit_skill");
    let agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(Arc::new(provider))
        .auto_configure_features()
        .unwrap()
        .hooks(Arc::new(ai_agents_hooks::LoggingHooks::new()))
        .build()
        .unwrap();
    let registry = ai_agents_runtime::spawner::AgentRegistry::new();
    registry
        .register(ai_agents_runtime::spawner::SpawnedAgent::from_runtime(
            "worker".into(),
            agent,
            Default::default(),
        ))
        .await
        .unwrap();
    let error = ai_agents_runtime::orchestration::pipeline(
        &registry,
        "Edit the status",
        &[ai_agents_runtime::orchestration::PipelineStage::id(
            "worker",
        )],
        None,
        None,
        None,
    )
    .await
    .expect_err("the failing skill must reach the caller");
    assert!(error.to_string().contains("private-fixture-value"));

    let registry = ai_agents_runtime::spawner::AgentRegistry::new();
    for (id, reply) in [("first", "private-fixture-value"), ("second", "done")] {
        let mut provider = MockLLMProvider::new(id);
        provider.set_response(reply);
        let agent = AgentBuilder::new()
            .system_prompt("test")
            .llm(Arc::new(provider))
            .hooks(Arc::new(ai_agents_hooks::LoggingHooks::new()))
            .build()
            .unwrap();
        registry
            .register(ai_agents_runtime::spawner::SpawnedAgent::from_runtime(
                id.into(),
                agent,
                Default::default(),
            ))
            .await
            .unwrap();
    }
    let completed = ai_agents_runtime::orchestration::pipeline(
        &registry,
        "go",
        &[
            ai_agents_runtime::orchestration::PipelineStage::id("first"),
            ai_agents_runtime::orchestration::PipelineStage::id("second"),
        ],
        None,
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(completed.response.content, "done");
    let logs = logs.lock().join("\n");
    assert!(logs.contains("Pipeline stage failed"), "{logs}");
    assert!(!logs.contains("private-fixture-value"), "{logs}");
    assert!(!logs.contains("private-input-fixture"), "{logs}");
}

// Failed edit hints must survive the shared executor and reach both legacy and native provider requests without changing the file.
#[tokio::test]
async fn failed_edit_context_reaches_the_model() {
    for native in [false, true] {
        let workspace = TestWorkspace::new();
        let path = workspace.0.join("edit.txt");
        let line = format!("Status: draft {}", "\\\"".repeat(80));
        let before = format!("unrelated confidential fixture\n{line}\n{line}\n{line}\n");
        std::fs::write(&path, &before).unwrap();
        let arguments = json!({
            "path": path.to_string_lossy(),
            "old_text": "Status: ready",
            "new_text": "Status: reviewed",
            "dry_run": true
        });
        let mut mock = MockLLMProvider::new("edit-feedback");
        if native {
            mock.set_tool_choice(Some(ai_agents_core::ToolChoice::Auto));
            mock.add_response(
                LLMResponse::new("", FinishReason::ToolCall)
                    .with_tool_calls(vec![ToolCall {
                        id: "edit-call".into(),
                        name: "file_edit".into(),
                        arguments: arguments.clone(),
                    }])
                    .unwrap(),
            );
            mock.add_response(LLMResponse::new("done", FinishReason::Stop));
        } else {
            mock.set_responses(
                vec![
                    json!({"tool":"file_edit","arguments":arguments}).to_string(),
                    "done".into(),
                ],
                false,
            );
        }
        let observed = mock.clone();
        let hooks = Arc::new(RecordHooks::default());
        let mut security = ToolSecurityConfig {
            enabled: true,
            fail_closed: true,
            ..Default::default()
        };
        security.tools.insert(
            "file_edit".into(),
            ToolPolicyConfig {
                read_paths: vec![workspace.0.to_string_lossy().into_owned()],
                write_paths: vec![workspace.0.to_string_lossy().into_owned()],
                allow_without_confirmation: true,
                max_output_chars: Some(512),
                ..Default::default()
            },
        );
        let agent = AgentBuilder::new()
            .system_prompt("Edit the requested file.")
            .llm(Arc::new(mock))
            .tool(Arc::new(FileEditTool::new()))
            .tool_security(ToolSecurityEngine::try_new(security).unwrap())
            .hooks(hooks.clone())
            .build()
            .unwrap();

        assert_eq!(
            agent.chat("Update the status").await.unwrap().content,
            "done"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
        let records = hooks.0.lock();
        let record = records
            .iter()
            .find(|record| record.canonical_id == "file_edit")
            .unwrap();
        assert!(record.executed);
        assert!(!record.success);
        assert!(!record.output_truncated);
        let calls = observed.call_history();
        assert_eq!(calls.len(), 2);
        let model_input = calls[1]
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            model_input.contains("Status: draft"),
            "native={native}: {model_input}"
        );
        assert!(
            model_input.contains("re-read"),
            "native={native}: {model_input}"
        );
        assert!(!model_input.contains("unrelated confidential fixture"));
        let message = calls[1]
            .messages
            .iter()
            .find(|message| message.name.as_deref() == Some("file_edit"))
            .expect("model-visible tool result");
        let payload = if native {
            let results = ai_agents_core::decode_native_tool_result_markers(&message.content)
                .unwrap()
                .expect("native result marker");
            results[0].output().as_str().unwrap().to_string()
        } else {
            message.content.clone()
        };
        assert!(
            payload.chars().count() <= 512,
            "native={native}: {}",
            payload.chars().count()
        );
    }
}
