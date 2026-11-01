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
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_SANDBOX: AtomicU64 = AtomicU64::new(0);

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
            "code-agent-workspace-test-{}-{nonce}-{}",
            std::process::id(),
            NEXT_SANDBOX.fetch_add(1, Ordering::Relaxed)
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

struct FeedbackFailProvider {
    requests: Mutex<Vec<Vec<ChatMessage>>>,
}

#[async_trait]
impl LLMProvider for FeedbackFailProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.requests.lock().unwrap().push(messages.to_vec());
        if messages.iter().any(|message| {
            message
                .content
                .contains("Untrusted output from a failed authorized validation check")
        }) {
            Err(LLMError::RateLimit { retry_after: None })
        } else {
            Ok(LLMResponse::new("Ready for checking.", FinishReason::Stop))
        }
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
        "feedback-failure-test"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

struct HookMutation(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait]
impl ai_agents::tools::Tool for HookMutation {
    fn id(&self) -> &str {
        "hook_mutation"
    }
    fn name(&self) -> &str {
        "hook_mutation"
    }
    fn description(&self) -> &str {
        "Count an admitted fixture mutation"
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ai_agents::tools::ToolSafetyMetadata {
        ai_agents::tools::ToolSafetyMetadata {
            default_requires_approval: false,
            ..Default::default()
        }
    }
    async fn execute(
        &self,
        _: serde_json::Value,
        _: ToolExecutionContext,
    ) -> ai_agents::tools::ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ai_agents::tools::ToolResult::ok("changed")
    }
}

struct FeedbackMutationHook {
    runtime: Mutex<std::sync::Weak<ai_agents::RuntimeAgent>>,
}
#[async_trait]
impl ai_agents::AgentHooks for FeedbackMutationHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_llm_start(&self, messages: &[ChatMessage]) {
        if !messages.iter().any(|message| {
            message
                .content
                .contains("Untrusted output from a failed authorized validation check")
        }) {
            return;
        }
        let runtime = self.runtime.lock().unwrap().upgrade();
        if let Some(runtime) = runtime {
            use ai_agents::tools::ToolInvoker;
            let _ = runtime
                .invoke_tool(ai_agents::tools::ToolExecutionRequest::new(
                    "hook-change",
                    "hook_mutation",
                    json!({}),
                    ToolCallSource::Model,
                ))
                .await;
        }
    }
}

struct RejectEditApproveChecker;
#[async_trait]
impl ai_agents::ApprovalHandler for RejectEditApproveChecker {
    async fn request_approval(&self, request: ai_agents::ApprovalRequest) -> ApprovalResult {
        match request.trigger {
            ai_agents::ApprovalTrigger::Tool { name, .. } if name == "file_edit" => {
                ApprovalResult::Rejected { reason: None }
            }
            _ => ApprovalResult::Approved,
        }
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
    let repair_edit = respond(&runner, first_check, ApprovalResult::Approved).await;
    assert_eq!(repair_edit.run.status, TaskRunStatus::Paused);
    let interaction = runner
        .task_interaction(&repair_edit.run.key.run_id, repair_edit.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        interaction.prompt,
        TaskHostPrompt::Approval { .. }
    ));
    let requests = provider.requests.lock().unwrap();
    assert!(requests.iter().flatten().any(|message| {
        message.role == ai_agents::Role::User
            && message
                .content
                .contains("Untrusted output from a failed authorized validation check")
            && message.content.contains("left: 3")
    }));
    assert!(
        !requests
            .iter()
            .flatten()
            .any(|message| message.role == ai_agents::Role::System
                && message.content.contains("left: 3"))
    );
    drop(requests);
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

// Discovery is an explicit broader read-only grant and does not widen direct file edits.
#[tokio::test]
async fn opted_in_workspace_discovery_executes_with_root_scope_but_keeps_edit_scope() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.enable_discovery().unwrap();
    let policy = &setup.spec.tool_security.tools["file_edit"];
    assert_eq!(policy.write_paths, vec![sandbox.source.to_str().unwrap()]);
    let provider = Arc::new(Provider::new(VecDeque::from([
        json!({"tool":"grep","arguments":{"path":sandbox.root,"pattern":"user data stays unchanged"}}).to_string(),
        "Ready for checking.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let result = runner.run("Inspect the workspace.", None).await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Paused);
    assert!(
        result
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|entry| entry.record.canonical_id == "grep" && entry.record.executed)
    );
    assert!(
        provider
            .requests
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .any(|message| message.content.contains("user data stays unchanged"))
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "user data stays unchanged\n"
    );
    cancel_if_paused(&runner, &result).await;
    drop(runner);
    sandbox.cleanup();
}

// Two explicitly selected files share one real checker while the pre-existing index, unstaged file and untracked file remain untouched.
#[tokio::test]
async fn selected_multifile_repairs_keep_dirty_workspace_and_require_fresh_check() {
    let sandbox = Sandbox::new();
    let second = sandbox.root.join("second.rs");
    let untracked = sandbox.root.join("untracked.txt");
    fs::write(
        &sandbox.source,
        "pub fn value() -> i32 { 1 } // user note\n",
    )
    .unwrap();
    fs::write(&second, "pub fn value() -> i32 { 1 }\n").unwrap();
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&sandbox.root)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git fixture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    };
    git(&["init", "--quiet"]);
    git(&["add", "--", "source.rs", "second.rs", "protected.rs"]);
    let staged = ["source.rs", "second.rs", "protected.rs"]
        .map(|file| git(&["--no-pager", "show", &format!(":{file}")]));
    fs::write(&sandbox.outside, "unstaged user change\n").unwrap();
    fs::write(&untracked, "untracked user data\n").unwrap();
    let before = git(&["--no-optional-locks", "status", "--porcelain=v1"]);
    assert!(String::from_utf8_lossy(&before).contains("protected.rs"));
    assert!(String::from_utf8_lossy(&before).contains("untracked.txt"));

    let checker_root = sandbox.parent.join("checker");
    let harness = checker_root.join("multi_acceptance.rs");
    fs::write(&harness, format!("#[path = {:?}] mod first;\n#[path = {:?}] mod second;\n#[test] fn expected_values() {{ assert_eq!(first::value(), 2); assert_eq!(second::value(), 3); }}\n", sandbox.source, second)).unwrap();
    let rustc = fs::canonicalize(env!("CARGO"))
        .unwrap()
        .parent()
        .unwrap()
        .join("rustc");
    let binary = checker_root.join("multi-acceptance");
    fs::write(&sandbox.checker, format!("#!/bin/sh\nset -eu\nPATH=/usr/bin:/bin\nexport PATH\n'{}' --edition 2024 --test '{}' -o '{}'\nexec '{}' --test-threads=1\n", rustc.display(), harness.display(), binary.display(), binary.display())).unwrap();
    let setup = prepare_files(
        &sandbox.root,
        &[PathBuf::from("source.rs"), PathBuf::from("second.rs")],
        &sandbox.checker,
        "openai",
        "offline-test",
        None,
    )
    .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 1 + 1 }","dry_run":false}}).to_string(),
        json!({"tool":"file_read","arguments":{"path":second}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":second,"old_text":"{ 1 }","new_text":"{ 1 + 3 }","dry_run":false}}).to_string(),
        "First edit ready.".into(),
        json!({"tool":"file_read","arguments":{"path":second}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":second,"old_text":"{ 1 + 3 }","new_text":"{ 1 + 2 }","dry_run":false}}).to_string(),
        "Repair ready.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let first = runner
        .run("Fix both selected functions.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let second_edit = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(second_edit.run.status, TaskRunStatus::Paused);
    let first_check = respond(&runner, second_edit, ApprovalResult::Approved).await;
    assert_eq!(first_check.run.status, TaskRunStatus::Paused);
    let repair = respond(&runner, first_check, ApprovalResult::Approved).await;
    assert_eq!(repair.run.status, TaskRunStatus::Paused);
    let final_check = respond(&runner, repair, ApprovalResult::Approved).await;
    assert_eq!(final_check.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, final_check, ApprovalResult::Approved).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 } // user note\n"
    );
    assert_eq!(
        fs::read_to_string(&second).unwrap(),
        "pub fn value() -> i32 { 1 + 2 }\n"
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "unstaged user change\n"
    );
    assert_eq!(
        fs::read_to_string(&untracked).unwrap(),
        "untracked user data\n"
    );
    for (index, file) in ["source.rs", "second.rs", "protected.rs"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            git(&["--no-pager", "show", &format!(":{file}")]),
            staged[index]
        );
    }
    assert_eq!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .filter(|entry| entry.record.canonical_id == "command" && entry.record.executed)
            .count(),
        2
    );
    assert_eq!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .filter(|entry| entry.record.canonical_id == "file_edit"
                && entry.record.executed
                && entry.record.success)
            .count(),
        3
    );
    assert!(
        provider
            .requests
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .any(|message| message.role == ai_agents::Role::User
                && message
                    .content
                    .contains("Untrusted output from a failed authorized validation check"))
    );
    drop(runner);
    sandbox.cleanup();
}

// The host limits both the untrusted provider projection and retries; repeated real failures fall back to a human question.
#[tokio::test]
async fn failed_checker_feedback_is_bounded_and_does_not_retry_indefinitely() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    let feedback = setup
        .spec
        .autonomy
        .defaults
        .premature_finish
        .as_mut()
        .unwrap()
        .validation_feedback
        .as_mut()
        .unwrap();
    feedback.max_output_chars = 12;
    feedback.max_retries = 1;
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for the first check.".into(),
        "Still not fixed.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let first_check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first_check.run.status, TaskRunStatus::Paused);
    let second_check = respond(&runner, first_check, ApprovalResult::Approved).await;
    assert_eq!(second_check.run.status, TaskRunStatus::Paused);
    let guidance = respond(&runner, second_check, ApprovalResult::Approved).await;
    assert_eq!(guidance.run.status, TaskRunStatus::Paused);
    let pending = runner
        .task_interaction(&guidance.run.key.run_id, guidance.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(pending.prompt, TaskHostPrompt::Question(_)));
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let projected = requests
        .iter()
        .flatten()
        .find(|message| {
            message.role == ai_agents::Role::User
                && message
                    .content
                    .starts_with("Untrusted output from a failed authorized validation check")
        })
        .expect("actual failed check output must reach the next model request");
    let (_, body) = projected.content.split_once('\n').unwrap();
    let body: serde_json::Value = serde_json::from_str(body).unwrap();
    assert!(body["text"].as_str().unwrap().chars().count() <= 12);
    assert_eq!(body["truncated"], true);
    drop(requests);
    let checks = guidance
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|record| record.record.canonical_id == "command" && record.record.executed)
        .count();
    assert_eq!(checks, 2);
    cancel_if_paused(&runner, &guidance).await;
    drop(runner);
    sandbox.cleanup();
}

// State-driven approval paths do not auto-continue from checker feedback; the host must obtain fresh guidance.
#[tokio::test]
async fn checker_feedback_with_state_machine_requires_guidance() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.spec.states =
        Some(serde_json::from_value(json!({"initial":"idle","states":{"idle":{}}})).unwrap());
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        "must not run".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let check = runner
        .run("Inspect the selected function.", None)
        .await
        .unwrap();
    assert_eq!(check.run.status, TaskRunStatus::Paused);
    let guidance = respond(&runner, check, ApprovalResult::Approved).await;
    assert_eq!(guidance.run.status, TaskRunStatus::Paused);
    let pending = runner
        .task_interaction(&guidance.run.key.run_id, guidance.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(pending.prompt, TaskHostPrompt::Question(_)));
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
    cancel_if_paused(&runner, &guidance).await;
    drop(runner);
    sandbox.cleanup();
}

// A feedback-bearing provider request is one-shot even if ordinary LLM retry and fallback are configured.
#[tokio::test]
async fn checker_feedback_never_retries_or_leaks_to_fallback_provider() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.spec.error_recovery.default.max_retries = 2;
    setup.spec.error_recovery.llm.on_failure =
        serde_json::from_value(json!({"action":"fallback_llm","fallback_llm":"fallback"})).unwrap();
    let primary = Arc::new(FeedbackFailProvider {
        requests: Mutex::new(Vec::new()),
    });
    let fallback = Arc::new(Provider::new(VecDeque::from(
        ["unexpected fallback".into()],
    )));
    let mut registry = ai_agents::LLMRegistry::new();
    registry.register("default", primary.clone());
    registry.register("fallback", fallback.clone());
    let agent = Arc::new(
        setup
            .builder()
            .llm_registry(registry)
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(setup.extensions().unwrap())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(setup.checker_runner());
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "feedback-once.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        setup.spec.autonomy.clone(),
        Default::default(),
        store,
        "feedback-once.v1".into(),
    )
    .unwrap();
    let check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(check.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, check, ApprovalResult::Approved).await;
    assert_ne!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(primary.requests.lock().unwrap().len(), 2);
    assert!(fallback.requests.lock().unwrap().is_empty());
    cancel_if_paused(&runner, &done).await;
    drop(runner);
    sandbox.cleanup();
}

// A real managed mutation during the pre-provider hook invalidates feedback before the model receives it.
#[tokio::test]
async fn checker_feedback_stops_before_dispatch_when_hook_mutates() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.spec.tools =
        Some(serde_json::from_value(json!(["file_read", "file_edit", "hook_mutation"])).unwrap());
    setup.spec.tool_security.tools.insert(
        "hook_mutation".into(),
        serde_json::from_value(json!({"allow_without_confirmation":true})).unwrap(),
    );
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        "must not run".into(),
    ])));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hook = Arc::new(FeedbackMutationHook {
        runtime: Mutex::new(std::sync::Weak::new()),
    });
    let agent = Arc::new(
        setup
            .builder()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(HookMutation(calls.clone())))
            .hooks(hook.clone())
            .autonomy_extensions(setup.extensions().unwrap())
            .build()
            .unwrap(),
    );
    *hook.runtime.lock().unwrap() = Arc::downgrade(&agent);
    agent.set_command_runner(setup.checker_runner());
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "feedback-stale.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        setup.spec.autonomy.clone(),
        Default::default(),
        store,
        "feedback-stale.v1".into(),
    )
    .unwrap();
    let check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(check.run.status, TaskRunStatus::Paused);
    let stopped = respond(&runner, check, ApprovalResult::Approved).await;
    assert_ne!(stopped.run.status, TaskRunStatus::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
    cancel_if_paused(&runner, &stopped).await;
    drop(runner);
    sandbox.cleanup();
}

// An inline human rejection is not converted into permission to try the same edit again after a failed check.
#[tokio::test]
async fn inline_edit_rejection_does_not_trigger_automatic_checker_feedback() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup
        .spec
        .autonomy
        .defaults
        .hitl
        .as_mut()
        .unwrap()
        .on_approval_required = None;
    let provider = scripted_provider(&setup, &sandbox.source);
    let agent = Arc::new(
        setup
            .builder()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(RejectEditApproveChecker))
            .autonomy_extensions(setup.extensions().unwrap())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(setup.checker_runner());
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "inline-reject.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        setup.spec.autonomy.clone(),
        Default::default(),
        store,
        "inline-reject.v1".into(),
    )
    .unwrap();
    let result = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_ne!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    cancel_if_paused(&runner, &result).await;
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
