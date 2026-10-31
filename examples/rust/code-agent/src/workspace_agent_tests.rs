use super::*;
use ai_agents::ApprovalResult;
use ai_agents::autonomy::{
    AutonomyHostCeilings, AutonomyRunner, ScopedTaskRunStore, TaskHostPrompt, TaskResumeInput,
    TaskRunResult, TaskRunStatus,
};
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::tool_security::{SecurityCheckResult, ToolSecurityEngine};
use ai_agents::tools::ToolCallSource;
use std::collections::VecDeque;
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

struct Sandbox {
    parent: PathBuf,
    root: PathBuf,
    source: PathBuf,
    outside: PathBuf,
    checker: PathBuf,
}

impl Sandbox {
    // These tests own a fresh directory and use a separate checker with host-owned Rust assertions.
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from("/tmp").join(format!(
            "code-agent-workspace-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&parent).unwrap();
        let root = parent.join("workspace");
        let checker_root = parent.join("checker");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&checker_root).unwrap();
        let source = root.join("source.rs");
        let outside = root.join("protected.rs");
        fs::write(&source, "pub fn value() -> i32 { 1 }\n").unwrap();
        fs::write(&outside, "user data stays unchanged\n").unwrap();
        let harness = checker_root.join("acceptance.rs");
        fs::write(&harness, format!("#[path = {:?}] mod candidate;\n#[test] fn expected_value() {{ assert_eq!(candidate::value(), 2); }}\n", source)).unwrap();
        let checker = checker_root.join("check.sh");
        let rustc = fs::canonicalize(env!("CARGO"))
            .unwrap()
            .parent()
            .unwrap()
            .join("rustc");
        let test_binary = checker_root.join("acceptance-test");
        fs::write(&checker, format!("#!/bin/sh\nset -eu\nPATH=/usr/bin:/bin\nexport PATH\n'{}' --edition 2024 --test '{}' -o '{}'\nexec '{}' --test-threads=1\n", rustc.display(), harness.display(), test_binary.display(), test_binary.display())).unwrap();
        fs::set_permissions(&checker, fs::Permissions::from_mode(0o700)).unwrap();
        Self {
            parent,
            root,
            source,
            outside,
            checker,
        }
    }

    fn setup(&self) -> WorkspaceSetup {
        prepare(
            &self.root,
            Path::new("source.rs"),
            &self.checker,
            "openai",
            "offline-test",
            None,
        )
        .unwrap()
    }

    fn cleanup(self) {
        fs::remove_dir_all(self.parent).unwrap();
    }
}

struct Provider {
    responses: Mutex<VecDeque<String>>,
    requests: Mutex<Vec<Vec<ChatMessage>>>,
}

impl Provider {
    fn new(responses: VecDeque<String>) -> Self {
        Self {
            responses: Mutex::new(responses),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl LLMProvider for Provider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.requests.lock().unwrap().push(messages.to_vec());
        let next = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LLMError::Config("mock responses exhausted".into()))?;
        Ok(LLMResponse::new(next, FinishReason::Stop))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Err(LLMError::Config("unexpected stream".into()))
    }
    fn provider_name(&self) -> &str {
        "workspace-test"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

fn scripted_provider(setup: &WorkspaceSetup, edit_path: &Path) -> Arc<Provider> {
    let responses = VecDeque::from([
        json!({"tool":"file_read","arguments":{"path":setup.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":edit_path,"old_text":"{ 1 }","new_text":"{ 1 + 1 }","dry_run":false}}).to_string(),
        json!({"tool":"command","arguments":{"command":setup.command,"cwd":setup.root}}).to_string(),
        "The edit is ready for validation.".to_string(),
    ]);
    Arc::new(Provider::new(responses))
}

fn runner(setup: &WorkspaceSetup, edit_path: &Path) -> AutonomyRunner {
    runner_with_provider(setup, scripted_provider(setup, edit_path))
}

fn runner_with_provider(setup: &WorkspaceSetup, provider: Arc<Provider>) -> AutonomyRunner {
    let agent = Arc::new(
        setup
            .builder()
            .llm(provider)
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(setup.extensions().unwrap())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(setup.checker_runner());
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "workspace-test.v1".into()).unwrap(),
    );
    AutonomyRunner::try_new(
        agent,
        setup.spec.autonomy.clone(),
        AutonomyHostCeilings::default(),
        store,
        "workspace-test.v1".into(),
    )
    .unwrap()
}

// Each response uses the current retained request and revision, never a reconstructed approval.
async fn respond(
    runner: &AutonomyRunner,
    result: TaskRunResult,
    decision: ApprovalResult,
) -> TaskRunResult {
    let interaction = runner
        .task_interaction(&result.run.key.run_id, result.run.revision)
        .await
        .unwrap()
        .expect("retained interaction");
    assert!(matches!(
        interaction.prompt,
        TaskHostPrompt::Approval { .. }
    ));
    runner
        .resume(
            &result.run.key.run_id,
            result.run.revision,
            TaskResumeInput::Approval {
                request_id: interaction.request_id,
                result: decision,
            },
        )
        .await
        .unwrap()
}

// Test-owned files are removed only after a paused task acknowledges cancellation.
async fn cancel_if_paused(runner: &AutonomyRunner, result: &TaskRunResult) {
    if result.run.status == TaskRunStatus::Paused {
        let cleanup = runner
            .cancel_task(&result.run.key.run_id, result.run.revision)
            .await
            .unwrap();
        assert_eq!(
            cleanup.disposition,
            ai_agents::CancellationDisposition::Acknowledged
        );
    } else {
        assert_ne!(result.run.status, TaskRunStatus::RecoveryRequired);
    }
}

#[tokio::test]
async fn workspace_configuration_is_narrow_and_provider_free() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    assert_eq!(setup.max_output_tokens(), 2000);
    assert!(setup.set_max_output_tokens(0).is_err());
    setup.set_max_output_tokens(4096).unwrap();
    assert_eq!(setup.max_output_tokens(), 4096);
    assert_eq!(setup.checker_timeout_ms(), 30000);
    assert!(setup.set_checker_timeout_ms(0).is_err());
    setup.set_checker_timeout_ms(45000).unwrap();
    assert_eq!(setup.checker_timeout_ms(), 45000);
    assert_eq!(
        setup
            .spec
            .autonomy
            .defaults
            .validation
            .as_ref()
            .unwrap()
            .checks
            .as_ref()
            .unwrap()[0]
            .config
            .as_ref()
            .unwrap()["timeout_ms"],
        45000
    );
    assert!(
        !setup
            .spec
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .any(|tool| tool.name() == "command")
    );
    let policy = &setup.spec.tool_security.tools["file_edit"];
    assert_eq!(policy.read_paths, vec![sandbox.source.to_str().unwrap()]);
    assert_eq!(policy.write_paths, vec![sandbox.source.to_str().unwrap()]);
    assert!(policy.require_read_before_write && policy.require_confirmation);
    assert!(!policy.allow_without_confirmation);
    let engine = ToolSecurityEngine::try_new(setup.spec.tool_security.clone()).unwrap();
    let denied = engine
        .check_tool_execution(
            "file_edit",
            &json!({"path":sandbox.outside,"old_text":"x","new_text":"y","dry_run":false}),
        )
        .await
        .unwrap();
    assert!(matches!(denied, SecurityCheckResult::Block { .. }));
    setup
        .builder()
        .auto_configure_features()
        .unwrap()
        .autonomy_extensions(setup.extensions().unwrap());
    assert!(
        prepare(
            &sandbox.root,
            Path::new("../protected.rs"),
            &sandbox.checker,
            "openai",
            "test",
            None
        )
        .is_err()
    );
    assert!(
        prepare(
            &sandbox.root,
            Path::new("source.rs"),
            &sandbox.source,
            "openai",
            "test",
            None
        )
        .is_err()
    );
    assert!(
        prepare(
            &sandbox.root,
            Path::new("source.rs"),
            &sandbox.checker,
            "openai-compatible",
            "test",
            None
        )
        .is_err()
    );
    fs::write(
        sandbox.root.join("instructions.md"),
        "Review this selected source.\n",
    )
    .unwrap();
    let with_instructions = prepare(
        &sandbox.root,
        Path::new("source.rs"),
        &sandbox.checker,
        "openai",
        "test",
        Some(Path::new("instructions.md")),
    )
    .unwrap();
    assert!(
        matches!(&with_instructions.spec.context["instruction_snapshot"], ContextSource::Runtime { default: Some(value), .. } if value.as_str() == Some("Review this selected source.\n"))
    );
    fs::set_permissions(&sandbox.source, fs::Permissions::from_mode(0o1700)).unwrap();
    assert!(
        prepare(
            &sandbox.root,
            Path::new("source.rs"),
            &sandbox.checker,
            "openai",
            "test",
            None
        )
        .is_err()
    );
    fs::set_permissions(&sandbox.source, fs::Permissions::from_mode(0o600)).unwrap();
    std::os::unix::fs::symlink(&sandbox.outside, sandbox.root.join("linked.rs")).unwrap();
    assert!(
        prepare(
            &sandbox.root,
            Path::new("linked.rs"),
            &sandbox.checker,
            "openai",
            "test",
            None
        )
        .is_err()
    );
    sandbox.cleanup();
}

#[tokio::test]
async fn changed_checker_is_refused_before_any_process_runs() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let checker = setup.checker_runner();
    assert!(checker.is_available());
    fs::write(&sandbox.checker, "#!/bin/sh\nexit 0\n").unwrap();
    assert!(!checker.is_available());
    let response = checker
        .run_command(
            CommandRequest {
                argv: setup.argv.clone(),
                cwd: Some(setup.root.to_string_lossy().into_owned()),
                ..Default::default()
            },
            ToolExecutionContext::test("command"),
        )
        .await;
    assert_eq!(response.termination, "unavailable");
    assert!(!response.success);
    assert!(!sandbox.parent.join("checker/acceptance-test").exists());
    sandbox.cleanup();
}

#[tokio::test]
async fn approved_edit_runs_only_the_separate_host_checker() {
    let sandbox = Sandbox::new();
    fs::set_permissions(&sandbox.source, fs::Permissions::from_mode(0o700)).unwrap();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    let second = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 }\n"
    );
    let done = respond(&runner, second, ApprovalResult::Approved).await;
    let records: Vec<_> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .map(|e| (&e.record.canonical_id, e.record.executed, &e.record.output))
        .collect();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "records: {records:?}"
    );
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "file_edit" && e.record.executed && e.record.success)
    );
    assert!(done.run.evidence.tool_calls.iter().any(|e| {
        e.record.canonical_id == "command"
            && e.record.executed
            && e.record.success
            && matches!(&e.record.source, ToolCallSource::Task)
    }));
    assert!(done.run.evidence.tool_calls.iter().any(|e| {
        e.record.canonical_id == "command"
            && !e.record.executed
            && matches!(&e.record.source, ToolCallSource::Model)
    }));
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "user data stays unchanged\n"
    );
    assert!(sandbox.parent.join("checker/acceptance-test").is_file());
    assert_eq!(
        fs::metadata(&sandbox.source).unwrap().permissions().mode() & 0o777,
        0o700
    );
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn failed_validation_allows_another_edit_and_fresh_checker_success() {
    let sandbox = Sandbox::new();
    let initial = std::process::Command::new(&sandbox.checker)
        .arg(&sandbox.root)
        .arg(&sandbox.source)
        .env_clear()
        .output()
        .unwrap();
    assert!(!initial.status.success());
    let setup = sandbox.setup();
    let responses = VecDeque::from([
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 1 + 2 }","dry_run":false}}).to_string(),
        "The first edit is ready for validation.".to_string(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 + 2 }","new_text":"{ 1 + 1 }","dry_run":false}}).to_string(),
        "The repaired edit is ready for validation.".to_string(),
    ]);
    let provider = Arc::new(Provider::new(responses));
    let runner = runner_with_provider(&setup, provider.clone());
    let first_edit = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first_edit.run.status, TaskRunStatus::Paused);
    let first_check = respond(&runner, first_edit, ApprovalResult::Approved).await;
    assert_eq!(first_check.run.status, TaskRunStatus::Paused);
    let guidance = respond(&runner, first_check, ApprovalResult::Approved).await;
    assert_eq!(guidance.run.status, TaskRunStatus::Paused);
    let question = runner
        .task_interaction(&guidance.run.key.run_id, guidance.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(question.prompt, TaskHostPrompt::Question(_)));
    let repair_edit = runner
        .resume(
            &guidance.run.key.run_id,
            guidance.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: question.request_id,
                answer: json!("The checker reported 3 instead of 2; repair the calculation."),
            },
        )
        .await
        .unwrap();
    assert_eq!(repair_edit.run.status, TaskRunStatus::Paused);
    assert!(
        provider
            .requests
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .any(|message| message
                .content
                .contains("The checker reported 3 instead of 2"))
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 2 }\n"
    );
    let second_check = respond(&runner, repair_edit, ApprovalResult::Approved).await;
    assert_eq!(second_check.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, second_check, ApprovalResult::Approved).await;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 }\n"
    );
    let edits = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|e| e.record.canonical_id == "file_edit" && e.record.executed && e.record.success)
        .count();
    let checks: Vec<_> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|e| {
            e.record.canonical_id == "command"
                && e.record.executed
                && matches!(&e.record.source, ToolCallSource::Task)
        })
        .collect();
    assert_eq!(edits, 2);
    assert_eq!(checks.len(), 2);
    let first: serde_json::Value = serde_json::from_str(&checks[0].record.output).unwrap();
    let second: serde_json::Value = serde_json::from_str(&checks[1].record.output).unwrap();
    assert_ne!(first["exit_code"], 0);
    assert_eq!(second["exit_code"], 0);
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "user data stays unchanged\n"
    );
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn changed_checker_timeout_cannot_prove_the_original_command_exit() {
    for changed in [false, true] {
        let sandbox = Sandbox::new();
        fs::write(&sandbox.source, "pub fn value() -> i32 { 2 }\n").unwrap();
        let mut setup = sandbox.setup();
        setup.set_checker_timeout_ms(45000).unwrap();
        setup.spec.tools =
            Some(serde_json::from_value(json!(["file_read", "file_edit", "command"])).unwrap());
        setup
            .spec
            .tool_security
            .tools
            .get_mut("command")
            .unwrap()
            .timeout_ms = Some(60000);
        let check = &mut setup
            .spec
            .autonomy
            .defaults
            .validation
            .as_mut()
            .unwrap()
            .checks
            .as_mut()
            .unwrap()[0];
        check.required = Some(false);
        check
            .config
            .as_mut()
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("host_binding");
        setup.spec.autonomy.defaults.completion = Some(
            serde_json::from_value(json!({
                "command_exit": {"command": setup.command, "code": 0}
            }))
            .unwrap(),
        );
        setup.spec.autonomy.defaults.premature_finish =
            Some(serde_json::from_value(json!({"action":"summarize_and_stop"})).unwrap());
        let provider = Arc::new(Provider::new(VecDeque::from([
            "Ready for the check.".to_string()
        ])));
        let runner = runner_with_provider(&setup, provider);
        let first = runner
            .run("Validate the selected function.", None)
            .await
            .unwrap();
        assert_eq!(first.run.status, TaskRunStatus::Paused);
        let decision = if changed {
            ApprovalResult::Modified {
                changes: std::collections::HashMap::from([("timeout_ms".into(), json!(60000))]),
            }
        } else {
            ApprovalResult::Approved
        };
        let done = respond(&runner, first, decision).await;
        assert_eq!(done.run.status == TaskRunStatus::Completed, !changed);
        assert!(done.run.evidence.tool_calls.iter().any(|e| {
            e.record.canonical_id == "command"
                && e.record.executed
                && e.record.executed_arguments["timeout_ms"] == if changed { 60000 } else { 45000 }
        }));
        drop(runner);
        sandbox.cleanup();
    }
}

#[tokio::test]
async fn configured_checker_timeout_reaches_the_fixed_approved_operation() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.set_checker_timeout_ms(45000).unwrap();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let checker_review = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(checker_review.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, checker_review, ApprovalResult::Approved).await;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    let command = done
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|e| {
            e.record.canonical_id == "command"
                && e.record.executed
                && matches!(&e.record.source, ToolCallSource::Task)
        })
        .unwrap();
    assert_eq!(command.record.executed_arguments["timeout_ms"], 45000);
    assert!(sandbox.parent.join("checker/acceptance-test").is_file());
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn configured_host_retains_the_real_task_and_checker_binding() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let agent = setup
        .builder()
        .llm(scripted_provider(&setup, &sandbox.source))
        .auto_configure_features()
        .unwrap()
        .autonomy_extensions(setup.extensions().unwrap())
        .task_interaction_delivery(true)
        .build_task_host()
        .await
        .unwrap();
    agent.set_command_runner(setup.checker_runner());
    let first = agent
        .run_task("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let runner = agent.automatic_task_runner().expect("retained live runner");
    let second = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, second, ApprovalResult::Approved).await;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "command"
                && e.record.executed
                && matches!(&e.record.source, ToolCallSource::Task))
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "user data stays unchanged\n"
    );
    drop(runner);
    drop(agent);
    sandbox.cleanup();
}

#[tokio::test]
async fn out_of_scope_edit_is_denied_before_approval_and_preserves_user_files() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.outside);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let done = if first.run.status == TaskRunStatus::Paused {
        respond(&runner, first, ApprovalResult::Approved).await
    } else {
        first
    };
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "file_edit" && !e.record.executed)
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "user data stays unchanged\n"
    );
    cancel_if_paused(&runner, &done).await;
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn modified_edit_approval_cannot_expand_the_selected_file_scope() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let changes = std::collections::HashMap::from([("path".to_string(), json!(sandbox.outside))]);
    let next = respond(&runner, first, ApprovalResult::Modified { changes }).await;
    let done = if next.run.status == TaskRunStatus::Paused {
        respond(&runner, next, ApprovalResult::Approved).await
    } else {
        next
    };
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "user data stays unchanged\n"
    );
    assert!(
        !done
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "file_edit" && e.record.executed)
    );
    cancel_if_paused(&runner, &done).await;
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn later_user_edit_is_not_overwritten_by_approval() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    fs::write(&sandbox.source, "pub fn value() -> i32 { 99 }\n").unwrap();
    let next = respond(&runner, first, ApprovalResult::Approved).await;
    let done = if next.run.status == TaskRunStatus::Paused {
        respond(&runner, next, ApprovalResult::Approved).await
    } else {
        next
    };
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 99 }\n"
    );
    assert!(
        !done
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "file_edit" && e.record.success)
    );
    cancel_if_paused(&runner, &done).await;
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn failing_checker_never_proves_completion() {
    let sandbox = Sandbox::new();
    fs::write(&sandbox.checker, "#!/bin/sh\nexit 1\n").unwrap();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let checker_review = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(checker_review.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, checker_review, ApprovalResult::Approved).await;
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "command" && e.record.executed)
    );
    cancel_if_paused(&runner, &done).await;
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn rejecting_checker_requests_guidance_without_retrying_it() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let provider = scripted_provider(&setup, &sandbox.source);
    let runner = runner_with_provider(&setup, provider.clone());
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let checker_review = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(checker_review.run.status, TaskRunStatus::Paused);
    let requests_before = provider.requests.lock().unwrap().len();
    let guidance = respond(
        &runner,
        checker_review,
        ApprovalResult::Rejected { reason: None },
    )
    .await;
    assert_eq!(guidance.run.status, TaskRunStatus::Paused);
    assert_eq!(provider.requests.lock().unwrap().len(), requests_before);
    let interaction = runner
        .task_interaction(&guidance.run.key.run_id, guidance.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(interaction.prompt, TaskHostPrompt::Question(_)));
    assert!(
        !guidance
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "command" && e.record.executed)
    );
    assert!(!sandbox.parent.join("checker/acceptance-test").exists());
    let cleanup = runner
        .cancel_task(&guidance.run.key.run_id, guidance.run.revision)
        .await
        .unwrap();
    assert_eq!(
        cleanup.disposition,
        ai_agents::CancellationDisposition::Acknowledged
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 }\n"
    );
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn approval_cannot_expand_the_fixed_checker_timeout() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.set_checker_timeout_ms(45000).unwrap();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let review = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(review.run.status, TaskRunStatus::Paused);
    let run_id = review.run.key.run_id.clone();
    let interaction = runner
        .task_interaction(&run_id, review.run.revision)
        .await
        .unwrap()
        .unwrap();
    let changes = std::collections::HashMap::from([("timeout_ms".to_string(), json!(60000))]);
    let result = runner
        .resume(
            &run_id,
            review.run.revision,
            TaskResumeInput::Approval {
                request_id: interaction.request_id,
                result: ApprovalResult::Modified { changes },
            },
        )
        .await;
    match result {
        Ok(done) => {
            assert_ne!(done.run.status, TaskRunStatus::Completed);
            assert!(
                !done
                    .run
                    .evidence
                    .tool_calls
                    .iter()
                    .any(|e| { e.record.canonical_id == "command" && e.record.executed })
            );
            cancel_if_paused(&runner, &done).await;
        }
        Err(_) => {
            let display = runner.task_display(&run_id).await.unwrap();
            assert_ne!(display.summary.status, TaskRunStatus::Completed);
            let cleanup = runner
                .cancel_task(&run_id, display.summary.revision)
                .await
                .unwrap();
            assert_eq!(
                cleanup.disposition,
                ai_agents::CancellationDisposition::Acknowledged
            );
        }
    }
    assert!(!sandbox.parent.join("checker/acceptance-test").exists());
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn approval_cannot_replace_the_fixed_checker_command() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let checker_review = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(checker_review.run.status, TaskRunStatus::Paused);
    let run_id = checker_review.run.key.run_id.clone();
    let revision = checker_review.run.revision;
    let interaction = runner
        .task_interaction(&run_id, revision)
        .await
        .unwrap()
        .unwrap();
    let changes = std::collections::HashMap::from([("command".to_string(), json!("/bin/true"))]);
    let result = runner
        .resume(
            &run_id,
            revision,
            TaskResumeInput::Approval {
                request_id: interaction.request_id,
                result: ApprovalResult::Modified { changes },
            },
        )
        .await;
    match result {
        Ok(done) => {
            assert_ne!(done.run.status, TaskRunStatus::Completed);
            assert!(
                !done
                    .run
                    .evidence
                    .tool_calls
                    .iter()
                    .any(|e| e.record.canonical_id == "command" && e.record.executed)
            );
            cancel_if_paused(&runner, &done).await;
        }
        Err(_) => {
            let display = runner.task_display(&run_id).await.unwrap();
            assert_ne!(display.summary.status, TaskRunStatus::Completed);
            let cleanup = runner
                .cancel_task(&run_id, display.summary.revision)
                .await
                .unwrap();
            assert_eq!(
                cleanup.disposition,
                ai_agents::CancellationDisposition::Acknowledged
            );
        }
    }
    assert!(!sandbox.parent.join("checker/acceptance-test").exists());
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn checker_changed_while_awaiting_approval_cannot_be_invoked() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let checker_review = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(checker_review.run.status, TaskRunStatus::Paused);
    let run_id = checker_review.run.key.run_id.clone();
    let revision = checker_review.run.revision;
    let interaction = runner
        .task_interaction(&run_id, revision)
        .await
        .unwrap()
        .unwrap();
    fs::write(&sandbox.checker, "#!/bin/sh\nexit 0\n").unwrap();
    let result = runner
        .resume(
            &run_id,
            revision,
            TaskResumeInput::Approval {
                request_id: interaction.request_id,
                result: ApprovalResult::Approved,
            },
        )
        .await;
    let error = result.expect_err("changed checker must not resume");
    assert!(
        error
            .to_string()
            .contains("required host operation unavailable")
    );
    assert!(!sandbox.parent.join("checker/acceptance-test").exists());
    let display = runner.task_display(&run_id).await.unwrap();
    assert_ne!(display.summary.status, TaskRunStatus::Completed);
    let cleanup = runner
        .cancel_task(&run_id, display.summary.revision)
        .await
        .unwrap();
    assert_eq!(
        cleanup.disposition,
        ai_agents::CancellationDisposition::Acknowledged
    );
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn rejecting_edit_keeps_source_and_never_reports_completion() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let runner = runner(&setup, &sandbox.source);
    let first = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let second = respond(&runner, first, ApprovalResult::Rejected { reason: None }).await;
    let done = if second.run.status == TaskRunStatus::Paused {
        respond(&runner, second, ApprovalResult::Approved).await
    } else {
        second
    };
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    assert!(
        !done
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "file_edit" && e.record.executed)
    );
    cancel_if_paused(&runner, &done).await;
    drop(runner);
    sandbox.cleanup();
}
