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

struct Provider(Mutex<VecDeque<String>>);

#[async_trait]
impl LLMProvider for Provider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let next = self
            .0
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
    Arc::new(Provider(Mutex::new(responses)))
}

fn runner(setup: &WorkspaceSetup, edit_path: &Path) -> AutonomyRunner {
    let agent = Arc::new(
        setup
            .builder()
            .llm(scripted_provider(setup, edit_path))
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

#[tokio::test]
async fn workspace_configuration_is_narrow_and_provider_free() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
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
    drop(runner);
    sandbox.cleanup();
}
