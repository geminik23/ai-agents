use super::*;
use ai_agents::autonomy::{
    AutonomyHostCeilings, AutonomyRunner, ScopedTaskRunStore, TaskCheckpointPayload,
    TaskHostPrompt, TaskResumeInput, TaskRunMutation, TaskRunResult, TaskRunSnapshot,
    TaskRunStatus, TaskRunStore, TaskRunSummary,
};
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::tool_security::{SecurityCheckResult, ToolSecurityEngine};
use ai_agents::tools::ToolCallSource;
use ai_agents::{Agent, ApprovalResult};
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
        json!({"tool":"command","arguments":{"command":setup.checker.as_ref().unwrap().command,"cwd":setup.root}}).to_string(),
        "The edit is ready for validation.".to_string(),
    ]);
    Arc::new(Provider::new(responses))
}

fn runner(setup: &WorkspaceSetup, edit_path: &Path) -> AutonomyRunner {
    runner_with_provider(setup, scripted_provider(setup, edit_path))
}

fn runner_with_provider(setup: &WorkspaceSetup, provider: Arc<Provider>) -> AutonomyRunner {
    runner_with_extensions(setup, provider, setup.extensions().unwrap(), None)
}

// Exercise the same workspace builder and runner with an explicitly installed live host target.
fn runner_with_extensions(
    setup: &WorkspaceSetup,
    provider: Arc<Provider>,
    extensions: AutonomyExtensions,
    hooks: Option<Arc<dyn ai_agents::AgentHooks>>,
) -> AutonomyRunner {
    let mut builder = setup
        .builder()
        .unwrap()
        .llm(provider)
        .auto_configure_features()
        .unwrap()
        .autonomy_extensions(extensions);
    if let Some(hooks) = hooks {
        builder = builder.hooks(hooks);
    }
    let agent = Arc::new(builder.build().unwrap());
    if let Some(checker) = setup.checker_runner() {
        agent.set_command_runner(checker);
    }
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
    assert_eq!(setup.checker_timeout_ms(), Some(30000));
    assert!(setup.set_checker_timeout_ms(0).is_err());
    setup.set_checker_timeout_ms(45000).unwrap();
    assert_eq!(setup.checker_timeout_ms(), Some(45000));
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
        .unwrap()
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

// Read-only tasks use the same controller without installing mutation, command or checker authority.
#[tokio::test]
async fn automatic_instruction_mode_reaches_real_task_requests_and_updates() {
    let sandbox = Sandbox::new();
    fs::create_dir(sandbox.root.join("src")).unwrap();
    fs::create_dir(sandbox.root.join("other")).unwrap();
    fs::write(sandbox.root.join("src/code.rs"), "fn main() {}\n").unwrap();
    fs::write(sandbox.root.join("AGENTS.md"), "root-rule-v1").unwrap();
    fs::write(sandbox.root.join("src/AGENTS.md"), "nested-rule-v1").unwrap();
    fs::write(sandbox.root.join("other/AGENTS.md"), "outside-scope-secret").unwrap();
    let args = vec![
        sandbox.root.to_str().unwrap().to_string(),
        "src/code.rs".into(),
        "openai".into(),
        "offline-test".into(),
        "--auto-instructions".into(),
    ];
    let setup = crate::workspace_setup(&args, true).unwrap();
    assert!(setup.automatic_instructions_enabled());
    let provider = Arc::new(Provider::new(VecDeque::from([
        "first answer".into(),
        "second answer".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let first = runner.run("read", None).await.unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Completed);
    fs::remove_file(sandbox.root.join("AGENTS.md")).unwrap();
    fs::write(sandbox.root.join("src/AGENTS.md"), "nested-rule-v2").unwrap();
    let second = runner.run("read again", None).await.unwrap();
    assert_eq!(second.run.status, TaskRunStatus::Completed);
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let first_text = requests[0]
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let second_text = requests[1]
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(first_text.contains("root-rule-v1") && first_text.contains("nested-rule-v1"));
    assert!(!first_text.contains("outside-scope-secret"));
    assert!(second_text.contains("nested-rule-v2"));
    assert!(!second_text.contains("nested-rule-v1") && !second_text.contains("root-rule-v1"));
    assert!(!second_text.contains("outside-scope-secret"));
    drop(requests);
    drop(runner);
    let conflicting = vec![
        sandbox.root.to_str().unwrap().to_string(),
        "src/code.rs".into(),
        "openai".into(),
        "offline-test".into(),
        "src/AGENTS.md".into(),
        "--auto-instructions".into(),
    ];
    assert!(crate::workspace_setup(&conflicting, true).is_err());
    sandbox.cleanup();
}

#[tokio::test]
async fn read_only_workspace_answers_without_edit_or_checker() {
    let sandbox = Sandbox::new();
    let mut setup = prepare_read_only(
        &sandbox.root,
        &[PathBuf::from("source.rs")],
        "openai",
        "offline-test",
        None,
    )
    .unwrap();
    assert!(setup.checker_runner().is_none());
    assert_eq!(setup.checker_timeout_ms(), None);
    assert!(setup.set_checker_timeout_ms(30000).is_err());
    assert!(setup.spec.autonomy.defaults.validation.is_none());
    assert!(
        !setup
            .spec
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .any(|tool| matches!(
                tool.name(),
                "file_edit" | "file_write" | "patch" | "command"
            ))
    );
    setup.enable_discovery().unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        "The selected function returns one.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let done = runner
        .run("Explain the selected function.", None)
        .await
        .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .any(|entry| entry.record.canonical_id == "file_read" && entry.record.executed)
    );
    assert!(!done.run.evidence.tool_calls.iter().any(|entry| matches!(
        entry.record.canonical_id.as_str(),
        "file_edit" | "file_write" | "patch" | "command"
    ) && entry.record.executed));
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    assert!(!sandbox.parent.join("checker/acceptance-test").exists());
    drop(runner);
    sandbox.cleanup();
}

// A read-only objective or misleading model request cannot install edit authority.
#[tokio::test]
async fn read_only_workspace_denies_model_mutation() {
    let sandbox = Sandbox::new();
    let setup = prepare_read_only(
        &sandbox.root,
        &[PathBuf::from("source.rs")],
        "openai",
        "offline-test",
        None,
    )
    .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 2 }","dry_run":false}}).to_string(),
        "No edit was made.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider);
    let done = runner
        .run("Explain the selected function.", None)
        .await
        .unwrap();
    assert!(
        !done
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|entry| entry.record.canonical_id == "file_edit" && entry.record.executed)
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    assert!(!sandbox.parent.join("checker/acceptance-test").exists());
    cancel_if_paused(&runner, &done).await;
    drop(runner);
    sandbox.cleanup();
}

#[tokio::test]
async fn changed_checker_is_refused_before_any_process_runs() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let checker = setup.checker_runner().expect("checker installed");
    assert!(checker.is_available());
    fs::write(&sandbox.checker, "#!/bin/sh\nexit 0\n").unwrap();
    assert!(!checker.is_available());
    let response = checker
        .run_command(
            CommandRequest {
                argv: setup.checker.as_ref().unwrap().argv.clone(),
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
    {
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
    }
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

// Compaction and overflow must preserve actual failed-check feedback and the current controller and instructions.
#[tokio::test]
async fn long_context_retains_actual_checker_feedback_and_latest_edit() {
    let sandbox = Sandbox::new();
    fs::write(
        sandbox.root.join("AGENTS.md"),
        "HOST-RULE: preserve the protected file",
    )
    .unwrap();
    let mut setup = sandbox.setup();
    setup.enable_automatic_instructions().unwrap();
    setup.spec.max_context_tokens = 2500;
    setup.spec.error_recovery.llm.on_context_overflow =
        serde_json::from_value(json!({"action":"truncate","keep_recent":4})).unwrap();
    setup.spec.memory.token_budget = Some(ai_agents::MemoryTokenBudget {
        total: 200_000,
        allocation: ai_agents::TokenAllocation {
            summary: 32,
            recent_messages: 10_000,
            facts: 0,
            relationships: 0,
        },
        ..Default::default()
    });
    let memory = Arc::new(ai_agents::CompactingMemory::new(
        Arc::new(ai_agents::NoopSummarizer),
        ai_agents::CompactingMemoryConfig {
            max_recent_messages: 1,
            compress_threshold: 3,
            summarize_batch_size: 2,
            max_summary_length: 2000,
        },
    ));
    let memory_handle: Arc<dyn ai_agents::Memory> = memory.clone();
    memory_handle
        .add_message(ChatMessage::user(
            "discardable earlier discussion ".repeat(500),
        ))
        .await
        .unwrap();
    for index in 0..4 {
        memory_handle
            .add_message(ChatMessage::user(format!("earlier turn {index}")))
            .await
            .unwrap();
    }
    let responses = VecDeque::from([
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 1 + 2 }","dry_run":false}}).to_string(),
        "The first edit is ready for validation.".to_string(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 + 2 }","new_text":"{ 1 + 1 }","dry_run":false}}).to_string(),
        "The repaired edit is ready for validation.".to_string(),
    ]);
    let provider = Arc::new(Provider::new(responses));
    let agent = Arc::new(
        setup
            .builder()
            .unwrap()
            .llm(provider.clone())
            .memory(memory.clone())
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(setup.extensions().unwrap())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(setup.checker_runner().unwrap());
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "workspace-context-test.v1".into())
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        setup.spec.autonomy.clone(),
        AutonomyHostCeilings::default(),
        store,
        "workspace-context-test.v1".into(),
    )
    .unwrap();

    let first_edit = runner
        .run("CURRENT-OBJECTIVE: fix the selected function", None)
        .await
        .unwrap();
    assert_eq!(
        first_edit.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        first_edit.run.stop_reason
    );
    let first_check = respond(&runner, first_edit, ApprovalResult::Approved).await;
    let repair_edit = respond(&runner, first_check, ApprovalResult::Approved).await;
    assert_eq!(
        repair_edit.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        repair_edit.run.stop_reason
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 2 }\n"
    );
    {
        let requests = provider.requests.lock().unwrap();
        assert!(
            !requests[0]
                .iter()
                .any(|message| message.content.contains("discardable earlier discussion"))
        );
        let repair = requests
            .iter()
            .find(|messages| {
                messages.iter().any(|message| {
                    message
                        .content
                        .contains("Untrusted output from a failed authorized validation check")
                })
            })
            .expect("actual failed checker must reach the next request");
        let text = repair
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("CURRENT-OBJECTIVE: fix the selected function"));
        assert!(text.contains("HOST-RULE: preserve the protected file"));
        assert!(text.contains("left: 3"));
        assert!(
            text.contains("1 + 2"),
            "latest modification missing from repair request: {text}"
        );
    }
    let second_check = respond(&runner, repair_edit, ApprovalResult::Approved).await;
    let done = respond(&runner, second_check, ApprovalResult::Approved).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 }\n"
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "user data stays unchanged\n"
    );
    let edits = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|e| e.record.canonical_id == "file_edit" && e.record.executed && e.record.success)
        .count();
    let checks = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|e| {
            e.record.canonical_id == "command"
                && e.record.executed
                && matches!(&e.record.source, ToolCallSource::Task)
        })
        .count();
    assert_eq!((edits, checks), (2, 2));
    assert!(
        !memory.compression_history().is_empty(),
        "the test must actually compact history"
    );
    drop(runner);
    sandbox.cleanup();
}

// The fixture host owns revision changes; it does not claim to observe arbitrary filesystem writes.
struct WorkspaceRevision(AtomicU64);
impl ai_agents::autonomy::HostRevisionSource for WorkspaceRevision {
    fn revision(&self) -> ai_agents::Result<String> {
        let value = self.0.load(Ordering::SeqCst);
        if value == u64::MAX {
            return Err(ai_agents::AgentError::Config(
                "fixture revision unavailable".into(),
            ));
        }
        Ok(value.to_string())
    }
}

// This isolated fixture observes its own edits; a real host must synchronize all relevant changes itself.
#[async_trait]
impl ai_agents::AgentHooks for WorkspaceRevision {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_tool_execution_record(&self, record: &ai_agents::tools::ToolExecutionRecord) {
        if matches!(record.canonical_id.as_str(), "file_edit" | "file_write") && record.executed {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
}

// A task can create and repair a selected new file while preserving the Git index and user work.
#[tokio::test]
async fn selected_new_file_and_repair_report_dirty_baseline() {
    use ai_agents_cli::CliTaskObserver;
    let sandbox = Sandbox::new();
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&sandbox.root)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
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
    git(&["add", "--", "source.rs", "protected.rs"]);
    git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "fixture baseline",
    ]);
    fs::write(
        &sandbox.source,
        "#[path = \"feature.rs\"] mod feature;\npub fn value() -> i32 { 1 } // staged user note\n",
    )
    .unwrap();
    git(&["add", "--", "source.rs"]);
    let staged = git(&["--no-pager", "show", ":source.rs"]);
    fs::write(&sandbox.source, "#[path = \"feature.rs\"] mod feature;\npub fn value() -> i32 { 1 } // staged user note + unstaged note\n").unwrap();
    fs::write(&sandbox.outside, "unselected user change\n").unwrap();
    fs::write(
        sandbox.root.join("user-untracked.txt"),
        "untracked user content\n",
    )
    .unwrap();
    let feature = sandbox.root.join("feature.rs");
    let mut setup = prepare_files_with_new(
        &sandbox.root,
        &[PathBuf::from("source.rs")],
        &[PathBuf::from("feature.rs")],
        &sandbox.checker,
        "openai",
        "offline-test",
        None,
    )
    .unwrap();
    setup.assume_exclusive_workspace().unwrap();
    let harness = sandbox.parent.join("checker/acceptance.rs");
    fs::write(&harness, format!("#[path = {:?}] mod candidate;\n#[test] fn expected_feature() {{ assert_eq!(candidate::value(), 4); }}\n", sandbox.source)).unwrap();
    let checker = setup.checker.as_ref().unwrap();
    // The checker fixture owns its harness and keeps the fixed executable unchanged.
    assert_eq!(checker.bytes, checker_snapshot(&sandbox.checker).unwrap());
    let revision = setup.guard.as_ref().unwrap().clone();
    let extensions = setup.extensions().unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 1 + feature::delta() }"}}).to_string(),
        json!({"tool":"file_write","arguments":{"path":feature,"content":"pub fn delta() -> i32 { 2 }\n"}}).to_string(),
        json!({"tool":"file_read","arguments":{"path":feature}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":feature,"old_text":"missing text","new_text":"replacement"}}).to_string(),
        "Ready for checking.".into(),
        json!({"tool":"file_read","arguments":{"path":feature}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":feature,"old_text":"{ 2 }","new_text":"{ 3 }"}}).to_string(),
        "Repair ready.".into(),
        "Second task ready.".into(),
    ])));
    let runner = runner_with_extensions(&setup, provider.clone(), extensions, None);
    let report = crate::workspace_report::WorkspaceReport::new(
        sandbox.root.clone(),
        setup.source_files().to_vec(),
    );
    report.before_task().await.unwrap();
    let first = runner
        .run("Add the feature and make the independent check pass.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let prompt = runner
        .task_interaction(&first.run.key.run_id, first.run.revision)
        .await
        .unwrap()
        .unwrap();
    match &prompt.prompt {
        TaskHostPrompt::Approval { trigger, .. } => {
            assert_eq!(trigger["name"], "file_edit");
            assert_eq!(trigger["args"]["path"], json!(sandbox.source));
            assert_eq!(trigger["args"]["new_text"], "{ 1 + feature::delta() }");
        }
        _ => panic!("expected the actual edit approval"),
    }
    let create = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(create.run.status, TaskRunStatus::Paused);
    let create_prompt = runner
        .task_interaction(&create.run.key.run_id, create.run.revision)
        .await
        .unwrap()
        .unwrap();
    match &create_prompt.prompt {
        TaskHostPrompt::Approval { trigger, .. } => {
            assert_eq!(trigger["name"], "file_write");
            assert_eq!(trigger["args"]["path"], json!(feature));
            assert_eq!(trigger["args"]["content"], "pub fn delta() -> i32 { 2 }\n");
        }
        _ => panic!("expected the actual create approval"),
    }
    let mismatch = respond(&runner, create, ApprovalResult::Approved).await;
    assert_eq!(mismatch.run.status, TaskRunStatus::Paused);
    let first_check = respond(&runner, mismatch, ApprovalResult::Approved).await;
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
    let exits: Vec<_> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "command" && entry.record.executed)
        .map(|entry| {
            serde_json::from_str::<serde_json::Value>(&entry.record.output).unwrap()["exit_code"]
                .as_i64()
                .unwrap()
        })
        .collect();
    assert_eq!(exits.len(), 2);
    assert_ne!(exits[0], 0);
    assert_eq!(exits[1], 0);
    // Three performed mutations advance the token; the mismatched edit changed nothing and keeps it.
    assert_eq!(revision.revision().unwrap(), "6");
    assert!(!revision.requires_reconciliation());
    assert_eq!(git(&["--no-pager", "show", ":source.rs"]), staged);
    assert!(
        fs::read_to_string(&sandbox.source)
            .unwrap()
            .contains("staged user note + unstaged note")
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "unselected user change\n"
    );
    assert_eq!(
        fs::read_to_string(sandbox.root.join("user-untracked.txt")).unwrap(),
        "untracked user content\n"
    );
    assert_eq!(
        fs::read_to_string(&feature).unwrap(),
        "pub fn delta() -> i32 { 3 }\n"
    );
    let text = report.after_task(Some(&done)).await.unwrap();
    assert!(
        text.contains("Before task: 1 staged, 2 unstaged, 1 untracked"),
        "{text}"
    );
    assert!(
        text.contains("Recorded actual mutation receipts: 3; executed commands: 2"),
        "{text}"
    );
    assert!(
        text.contains("Existing \"source.rs\": index 'M', worktree 'M'"),
        "{text}"
    );
    assert!(text.contains("Selected \"feature.rs\": changed bytes or existence; matching recorded task mutation: true"), "{text}");
    assert!(text.contains("do not attribute every change"));
    assert!(
        provider
            .requests
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .all(|message| !message.content.contains("untracked user content"))
    );
    report.before_task().await.unwrap();
    let second = runner
        .run("Verify the completed feature again.", None)
        .await
        .unwrap();
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    let second_done = respond(&runner, second, ApprovalResult::Approved).await;
    assert_eq!(second_done.run.status, TaskRunStatus::Completed);
    assert_ne!(done.run.key.run_id, second_done.run.key.run_id);
    let second_report = report.after_task(Some(&second_done)).await.unwrap();
    assert!(second_report.contains("unchanged bytes"), "{second_report}");
    assert!(
        second_report.contains("Recorded actual mutation receipts: 0; executed commands: 1"),
        "{second_report}"
    );
    report.before_task().await.unwrap();
    let unavailable = report.after_task(None).await.unwrap();
    assert!(unavailable.contains("effects and cleanup are unconfirmed"));
    drop(runner);
    sandbox.cleanup();
}

// The actual foreground REPL consumes the same workspace host, exact approvals and local report.
#[tokio::test]
async fn workspace_foreground_process_creates_repairs_and_reports() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    if std::env::var_os("CODE_AGENT_FOREGROUND_FIXTURE").is_some() {
        let root = PathBuf::from(std::env::var_os("CODE_AGENT_FIXTURE_ROOT").unwrap());
        let checker = PathBuf::from(std::env::var_os("CODE_AGENT_FIXTURE_CHECKER").unwrap());
        let source = root.join("source.rs");
        let feature = root.join("feature.rs");
        let mut setup = prepare_files_with_new(
            &root,
            &[PathBuf::from("source.rs")],
            &[PathBuf::from("feature.rs")],
            &checker,
            "openai",
            "offline-test",
            None,
        )
        .unwrap();
        setup.assume_exclusive_workspace().unwrap();
        let provider = Arc::new(Provider::new(VecDeque::from([
            json!({"tool":"file_read","arguments":{"path":source}}).to_string(),
            json!({"tool":"file_edit","arguments":{"path":source,"old_text":"{ 1 }","new_text":"{ 1 + feature::delta() }"}}).to_string(),
            json!({"tool":"file_write","arguments":{"path":feature,"content":"pub fn delta() -> i32 { 2 }\n"}}).to_string(),
            "Ready for checking.".into(),
            json!({"tool":"file_read","arguments":{"path":feature}}).to_string(),
            json!({"tool":"file_edit","arguments":{"path":feature,"old_text":"{ 2 }","new_text":"{ 3 }"}}).to_string(),
            "Repair ready.".into(),
        ])));
        let agent = setup
            .builder()
            .unwrap()
            .llm(provider)
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(setup.extensions().unwrap())
            .approval_handler(CliApprovalHandler::from_metadata(None))
            .task_interaction_delivery(true)
            .build_task_host()
            .await
            .unwrap();
        workspace_repl(&setup, agent).run().await.unwrap();
        return;
    }

    let sandbox = Sandbox::new();
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&sandbox.root)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "fixture Git failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    };
    git(&["init", "--quiet"]);
    git(&["add", "--", "source.rs", "protected.rs"]);
    git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "fixture baseline",
    ]);
    fs::write(
        &sandbox.source,
        "#[path = \"feature.rs\"] mod feature;\npub fn value() -> i32 { 1 } // staged\n",
    )
    .unwrap();
    git(&["add", "--", "source.rs"]);
    let staged = git(&["--no-pager", "show", ":source.rs"]);
    fs::write(&sandbox.source, "#[path = \"feature.rs\"] mod feature;\npub fn value() -> i32 { 1 } // staged and unstaged\n").unwrap();
    fs::write(&sandbox.outside, "unselected user work\n").unwrap();
    fs::write(sandbox.root.join("user.txt"), "untracked user work\n").unwrap();
    fs::write(sandbox.parent.join("checker/acceptance.rs"), format!("#[path = {:?}] mod candidate;\n#[test] fn feature() {{ assert_eq!(candidate::value(), 4); }}\n", sandbox.source)).unwrap();

    let mut child = tokio::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "workspace_agent::tests::workspace_foreground_process_creates_repairs_and_reports",
            "--nocapture",
        ])
        .env_clear()
        .env("CODE_AGENT_FOREGROUND_FIXTURE", "1")
        .env("CODE_AGENT_FIXTURE_ROOT", &sandbox.root)
        .env("CODE_AGENT_FIXTURE_CHECKER", &sandbox.checker)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let output = child.stdout.take().unwrap();
    let process = async {
        input
            .write_all(b"/task run Add the feature.\nyes\nyes\nyes\nyes\nyes\n")
            .await
            .unwrap();
        let mut lines = tokio::io::BufReader::new(output).lines();
        let mut captured = String::new();
        let mut quit_sent = false;
        while let Some(line) = lines.next_line().await.unwrap() {
            if line.contains("Workspace report for task") && !quit_sent {
                input.write_all(b"/quit\n").await.unwrap();
                quit_sent = true;
            }
            captured.push_str(&line);
            captured.push('\n');
        }
        let status = child.wait().await.unwrap();
        (status, captured, quit_sent)
    };
    let (status, output, quit_sent) =
        tokio::time::timeout(std::time::Duration::from_secs(90), process)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "foreground timed out; fixture retained at {}",
                    sandbox.parent.display()
                )
            });
    assert!(
        status.success(),
        "foreground failed; fixture retained at {}: {output}",
        sandbox.parent.display()
    );
    assert!(quit_sent, "no workspace report: {output}");
    assert!(
        output.contains("file_write") && output.contains("feature.rs"),
        "{output}"
    );
    assert!(
        output.contains("Before task: 1 staged, 2 unstaged, 1 untracked"),
        "{output}"
    );
    assert!(
        output.contains("Recorded actual mutation receipts: 3; executed commands: 2"),
        "{output}"
    );
    assert!(
        output.contains("matching recorded task mutation: true"),
        "{output}"
    );
    assert_eq!(git(&["--no-pager", "show", ":source.rs"]), staged);
    assert!(
        fs::read_to_string(&sandbox.source)
            .unwrap()
            .contains("staged and unstaged")
    );
    assert_eq!(
        fs::read_to_string(&sandbox.outside).unwrap(),
        "unselected user work\n"
    );
    assert_eq!(
        fs::read_to_string(sandbox.root.join("user.txt")).unwrap(),
        "untracked user work\n"
    );
    assert_eq!(
        fs::read_to_string(sandbox.root.join("feature.rs")).unwrap(),
        "pub fn delta() -> i32 { 3 }\n"
    );
    sandbox.cleanup();
}

// The same fixed checker can provide bounded failure feedback when its actual live revision is bound.
#[tokio::test]
async fn target_bound_checker_failure_allows_fresh_repair() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 1 + 1 }","dry_run":false}}).to_string(),
        "Repair ready.".into(),
    ])));
    let runner =
        runner_with_extensions(&setup, provider.clone(), extensions, Some(revision.clone()));
    let first_check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(first_check.run.status, TaskRunStatus::Paused);
    let edit = respond(&runner, first_check, ApprovalResult::Approved).await;
    assert_eq!(edit.run.status, TaskRunStatus::Paused);
    assert!(
        provider
            .requests
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .any(|message| {
                message.role == ai_agents::Role::User
                    && message
                        .content
                        .contains("Untrusted output from a failed authorized validation check")
            })
    );
    let final_check = respond(&runner, edit, ApprovalResult::Approved).await;
    assert_eq!(final_check.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, final_check, ApprovalResult::Approved).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(revision.0.load(Ordering::SeqCst), 2);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 }\n"
    );
    assert_eq!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .filter(|entry| entry.record.canonical_id == "command" && entry.record.executed)
            .count(),
        2
    );
    drop(runner);
    sandbox.cleanup();
}

// A declared multi-stage profile uses the same fixed checker binding and cannot reuse a Pass after an edit.
#[tokio::test]
async fn actual_checker_pass_then_edit_requires_fresh_validation() {
    let sandbox = Sandbox::new();
    fs::write(&sandbox.source, "pub fn value() -> i32 { 1 + 1 }\n").unwrap();
    let mut setup = sandbox.setup();
    setup.spec.autonomy.defaults.lifecycle = Some(
        serde_json::from_value(json!([
            {"id":"baseline"}, {"id":"repair"}
        ]))
        .unwrap(),
    );
    setup.spec.autonomy.defaults.premature_finish = None;
    setup
        .spec
        .autonomy
        .defaults
        .validation
        .as_mut()
        .unwrap()
        .checks
        .as_mut()
        .unwrap()[0]
        .schedule = Some(ai_agents::autonomy::ValidationSchedule::EachCycle);
    setup.spec.validate().unwrap();
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Baseline ready.".into(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 + 1 }","new_text":"{ 1 + 3 }","dry_run":false}}).to_string(),
        "Changed source ready.".into(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 + 3 }","new_text":"{ 2 }","dry_run":false}}).to_string(),
        "Repair ready.".into(),
    ])));
    let runner = runner_with_extensions(&setup, provider, extensions, Some(revision.clone()));
    let initial = runner
        .run(
            "Keep the function correct while applying the requested changes.",
            None,
        )
        .await
        .unwrap();
    assert_eq!(initial.run.status, TaskRunStatus::Paused);
    let broken_edit = respond(&runner, initial, ApprovalResult::Approved).await;
    assert_eq!(broken_edit.run.status, TaskRunStatus::Paused);
    let failed_check = respond(&runner, broken_edit, ApprovalResult::Approved).await;
    assert_eq!(failed_check.run.status, TaskRunStatus::Paused);
    let repair_edit = respond(&runner, failed_check, ApprovalResult::Approved).await;
    assert_eq!(repair_edit.run.status, TaskRunStatus::Paused);
    assert_ne!(repair_edit.run.status, TaskRunStatus::Completed);
    let final_check = respond(&runner, repair_edit, ApprovalResult::Approved).await;
    assert_eq!(final_check.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, final_check, ApprovalResult::Approved).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    let exits: Vec<_> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "command" && entry.record.executed)
        .map(|entry| {
            serde_json::from_str::<serde_json::Value>(&entry.record.output).unwrap()["exit_code"]
                .as_i64()
                .unwrap()
        })
        .collect();
    assert_eq!(exits.len(), 3);
    assert_eq!(exits[0], 0);
    assert_ne!(exits[1], 0);
    assert_eq!(exits[2], 0);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 2 }\n"
    );
    assert_eq!(revision.0.load(Ordering::SeqCst), 3);
    drop(runner);
    sandbox.cleanup();
}

struct SuccessfulCheckChangeHook {
    revision: Arc<WorkspaceRevision>,
    source: PathBuf,
}

// A cooperative host changes the source as the actual passing checker result is being published.
#[async_trait]
impl ai_agents::AgentHooks for SuccessfulCheckChangeHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_tool_execution_record(&self, record: &ai_agents::tools::ToolExecutionRecord) {
        if record.canonical_id == "command"
            && record.executed
            && serde_json::from_str::<serde_json::Value>(&record.output)
                .ok()
                .and_then(|output| output["exit_code"].as_i64())
                == Some(0)
        {
            fs::write(&self.source, "pub fn value() -> i32 { 99 }\n").unwrap();
            self.revision.0.fetch_add(1, Ordering::SeqCst);
        }
    }
}

// A real checker Pass cannot complete against a cooperative host revision that changed before publication.
#[tokio::test]
async fn actual_checker_pass_rejects_concurrent_host_change() {
    let sandbox = Sandbox::new();
    fs::write(&sandbox.source, "pub fn value() -> i32 { 2 }\n").unwrap();
    let setup = sandbox.setup();
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from(
        ["Ready for checking.".into()],
    )));
    let hook = Arc::new(SuccessfulCheckChangeHook {
        revision,
        source: sandbox.source.clone(),
    });
    let runner = runner_with_extensions(&setup, provider, extensions, Some(hook));
    let pending = runner
        .run("Verify the selected function.", None)
        .await
        .unwrap();
    assert_eq!(pending.run.status, TaskRunStatus::Paused);
    let run_id = pending.run.key.run_id.clone();
    let interaction = runner
        .task_interaction(&run_id, pending.run.revision)
        .await
        .unwrap()
        .unwrap();
    let result = runner
        .resume(
            &run_id,
            pending.run.revision,
            TaskResumeInput::Approval {
                request_id: interaction.request_id,
                result: ApprovalResult::Approved,
            },
        )
        .await;
    if let Ok(done) = &result {
        assert_ne!(done.run.status, TaskRunStatus::Completed);
    }
    let display = runner.task_display(&run_id).await.unwrap();
    assert_ne!(display.summary.status, TaskRunStatus::Completed);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 99 }\n"
    );
    if display.summary.status == TaskRunStatus::Paused {
        let cancelled = runner
            .cancel_task(&run_id, display.summary.revision)
            .await
            .unwrap();
        assert_eq!(
            cancelled.disposition,
            ai_agents::CancellationDisposition::Acknowledged
        );
    } else {
        assert!(
            !display.locally_retained,
            "fixture must retain files while cleanup is uncertain"
        );
    }
    drop(runner);
    sandbox.cleanup();
}

struct FeedbackTargetChangeHook {
    revision: Arc<WorkspaceRevision>,
    source: PathBuf,
}

// Simulate a cooperative external editor changing the checked input after feedback is prepared.
#[async_trait]
impl ai_agents::AgentHooks for FeedbackTargetChangeHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_llm_start(&self, messages: &[ChatMessage]) {
        if messages.iter().any(|message| {
            message
                .content
                .contains("Untrusted output from a failed authorized validation check")
        }) {
            fs::write(&self.source, "pub fn value() -> i32 { 99 }\n").unwrap();
            self.revision.0.fetch_add(1, Ordering::SeqCst);
        }
    }
}

// A changed host revision stops stale feedback before the managed provider can observe it.
#[tokio::test]
async fn target_bound_checker_feedback_stops_after_host_revision_changes() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        "must not run".into(),
    ])));
    let hook = Arc::new(FeedbackTargetChangeHook {
        revision,
        source: sandbox.source.clone(),
    });
    let runner = runner_with_extensions(&setup, provider.clone(), extensions, Some(hook));
    let check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(check.run.status, TaskRunStatus::Paused);
    let stopped = respond(&runner, check, ApprovalResult::Approved).await;
    assert_ne!(stopped.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 99 }\n"
    );
    cancel_if_paused(&runner, &stopped).await;
    drop(runner);
    sandbox.cleanup();
}

struct FeedbackCapacityHook {
    runtime: Mutex<std::sync::Weak<ai_agents::RuntimeAgent>>,
    calls: AtomicU64,
}

// This cooperative hook uses the runtime's actual managed provider and shares the task's allowance.
#[async_trait]
impl ai_agents::AgentHooks for FeedbackCapacityHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_llm_start(&self, messages: &[ChatMessage]) {
        if messages.iter().any(|message| {
            message
                .content
                .contains("Untrusted output from a failed authorized validation check")
        }) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let runtime = self.runtime.lock().unwrap().upgrade().unwrap();
            let provider = runtime.llm_registry().get("default").unwrap();
            let _ = provider.complete(messages, None).await;
        }
    }
}

// A changed target cannot consume a retained capacity response before the stale feedback is rejected.
#[tokio::test]
async fn target_bound_feedback_capacity_resume_preserves_stale_pause() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.spec.autonomy.defaults.max_llm_calls =
        Some(ai_agents::autonomy::AutonomyLimit::Finite(2));
    setup.spec.autonomy.defaults.on_limit =
        Some(serde_json::from_value(json!({"action":"pause_run"})).unwrap());
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        "hook result".into(),
        "must not run".into(),
    ])));
    let hook = Arc::new(FeedbackCapacityHook {
        runtime: Mutex::new(std::sync::Weak::new()),
        calls: AtomicU64::new(0),
    });
    let runner = runner_with_extensions(&setup, provider.clone(), extensions, Some(hook.clone()));
    *hook.runtime.lock().unwrap() = Arc::downgrade(runner.runtime());
    let check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(check.run.status, TaskRunStatus::Paused);
    let capacity = respond(&runner, check, ApprovalResult::Approved).await;
    assert_eq!(
        capacity.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        capacity.run.stop_reason
    );
    assert_eq!(hook.calls.load(Ordering::SeqCst), 1);
    let mut limits = capacity.run.limits.clone();
    limits.max_llm_calls = Some(3);
    let revised = runner
        .revise_limits_paused(
            &capacity.run.key.run_id,
            capacity.run.revision,
            limits,
            "allow feedback",
        )
        .await
        .unwrap();
    revision.0.fetch_add(1, Ordering::SeqCst);
    assert!(
        runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue
            )
            .await
            .is_err()
    );
    let display = runner.task_display(&revised.run.key.run_id).await.unwrap();
    assert_eq!(display.summary.status, TaskRunStatus::Paused);
    assert_eq!(display.summary.revision, revised.run.revision);
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    revision.0.store(u64::MAX, Ordering::SeqCst);
    assert!(
        runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue
            )
            .await
            .is_err()
    );
    let cancelled = runner
        .cancel_task(&revised.run.key.run_id, revised.run.revision)
        .await
        .unwrap();
    assert_eq!(
        cancelled.disposition,
        ai_agents::CancellationDisposition::Acknowledged
    );
    drop(runner);
    sandbox.cleanup();
}

// Once the feedback-bearing model call has completed, a later edit may legitimately advance the host revision.
#[tokio::test]
async fn feedback_capacity_after_edit_accepts_new_target_and_rechecks() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.spec.autonomy.defaults.max_llm_calls =
        Some(ai_agents::autonomy::AutonomyLimit::Finite(3));
    setup.spec.autonomy.defaults.on_limit =
        Some(serde_json::from_value(json!({"action":"pause_run"})).unwrap());
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 2 }","dry_run":false}}).to_string(),
        "Ready after the edit.".into(),
    ])));
    let runner = runner_with_extensions(&setup, provider, extensions, Some(revision.clone()));
    let first_check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let edit = respond(&runner, first_check, ApprovalResult::Approved).await;
    assert_eq!(edit.run.status, TaskRunStatus::Paused);
    let capacity = respond(&runner, edit, ApprovalResult::Approved).await;
    assert_eq!(capacity.run.status, TaskRunStatus::Paused);
    assert_eq!(revision.0.load(Ordering::SeqCst), 2);
    let mut limits = capacity.run.limits.clone();
    limits.max_llm_calls = Some(4);
    let revised = runner
        .revise_limits_paused(
            &capacity.run.key.run_id,
            capacity.run.revision,
            limits,
            "continue after edit",
        )
        .await
        .unwrap();
    let final_check = runner
        .resume(
            &revised.run.key.run_id,
            revised.run.revision,
            TaskResumeInput::Continue,
        )
        .await
        .unwrap();
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
        "pub fn value() -> i32 { 2 }\n"
    );
    drop(runner);
    sandbox.cleanup();
}

// Validator capacity after a completed feedback-driven edit must use the new check target, not the old feedback target.
#[tokio::test]
async fn feedback_then_edit_allows_fresh_validator_capacity_resume() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.spec.autonomy.defaults.max_command_calls =
        Some(ai_agents::autonomy::AutonomyLimit::Finite(1));
    setup.spec.autonomy.defaults.on_limit =
        Some(serde_json::from_value(json!({"action":"pause_run"})).unwrap());
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 2 }","dry_run":false}}).to_string(),
        "Ready after the edit.".into(),
    ])));
    let runner = runner_with_extensions(&setup, provider, extensions, Some(revision.clone()));
    let first_check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let edit = respond(&runner, first_check, ApprovalResult::Approved).await;
    let capacity = respond(&runner, edit, ApprovalResult::Approved).await;
    assert_eq!(
        capacity.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        capacity.run.stop_reason
    );
    assert_eq!(revision.0.load(Ordering::SeqCst), 2);
    let mut limits = capacity.run.limits.clone();
    limits.max_command_calls = Some(2);
    let revised = runner
        .revise_limits_paused(
            &capacity.run.key.run_id,
            capacity.run.revision,
            limits,
            "allow fresh checker",
        )
        .await
        .unwrap();
    let next = runner
        .resume(
            &revised.run.key.run_id,
            revised.run.revision,
            TaskResumeInput::Continue,
        )
        .await
        .unwrap();
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    let done = respond(&runner, next, ApprovalResult::Approved).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    drop(runner);
    sandbox.cleanup();
}

// Resuming a feedback provider after capacity must not repeat a completed on_llm_start hook.
#[tokio::test]
async fn feedback_capacity_resume_does_not_replay_hook() {
    for target_bound in [false, true] {
        let sandbox = Sandbox::new();
        let mut setup = sandbox.setup();
        setup.spec.autonomy.defaults.max_llm_calls =
            Some(ai_agents::autonomy::AutonomyLimit::Finite(2));
        setup.spec.autonomy.defaults.on_limit =
            Some(serde_json::from_value(json!({"action":"pause_run"})).unwrap());
        let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
        let extensions = if target_bound {
            setup
                .extensions_with_target(Some(("workspace", revision)))
                .unwrap()
        } else {
            setup.extensions().unwrap()
        };
        let provider = Arc::new(Provider::new(VecDeque::from([
            "Ready for checking.".into(),
            "hook result".into(),
            "provider result".into(),
        ])));
        let hook = Arc::new(FeedbackCapacityHook {
            runtime: Mutex::new(std::sync::Weak::new()),
            calls: AtomicU64::new(0),
        });
        let runner = runner_with_extensions(&setup, provider, extensions, Some(hook.clone()));
        *hook.runtime.lock().unwrap() = Arc::downgrade(runner.runtime());
        let check = runner
            .run("Fix the selected function.", None)
            .await
            .unwrap();
        let capacity = respond(&runner, check, ApprovalResult::Approved).await;
        assert_eq!(capacity.run.status, TaskRunStatus::Paused);
        assert_eq!(hook.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            capacity.run.pending.as_ref().unwrap().reviewed_action["proposal"]["limit"]["model"]["feedback_hook_completed"],
            true
        );
        let mut limits = capacity.run.limits.clone();
        limits.max_llm_calls = Some(4);
        let revised = runner
            .revise_limits_paused(
                &capacity.run.key.run_id,
                capacity.run.revision,
                limits,
                "allow feedback",
            )
            .await
            .unwrap();
        let resumed = runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        assert_eq!(
            hook.calls.load(Ordering::SeqCst),
            1,
            "completed feedback hook must not run again"
        );
        cancel_if_paused(&runner, &resumed).await;
        drop(runner);
        sandbox.cleanup();
    }
}

struct FeedbackCapacityAndUnavailableHook {
    runtime: Mutex<std::sync::Weak<ai_agents::RuntimeAgent>>,
    revision: Arc<WorkspaceRevision>,
    calls: AtomicU64,
}

// Finish a host callback, exhaust managed capacity, then lose its revision before the post-hook proof check.
#[async_trait]
impl ai_agents::AgentHooks for FeedbackCapacityAndUnavailableHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_llm_start(&self, messages: &[ChatMessage]) {
        if messages.iter().any(|message| {
            message
                .content
                .contains("Untrusted output from a failed authorized validation check")
        }) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let runtime = self.runtime.lock().unwrap().upgrade().unwrap();
            let provider = runtime.llm_registry().get("default").unwrap();
            let _ = provider.complete(messages, None).await;
            let _ = provider.complete(messages, None).await;
            self.revision.0.store(u64::MAX, Ordering::SeqCst);
        }
    }
}

// A simultaneous capacity stop and temporarily unavailable proof must not replay an already completed hook.
#[tokio::test]
async fn feedback_capacity_with_unavailable_target_preserves_completed_hook() {
    let sandbox = Sandbox::new();
    let mut setup = sandbox.setup();
    setup.spec.autonomy.defaults.max_llm_calls =
        Some(ai_agents::autonomy::AutonomyLimit::Finite(2));
    setup.spec.autonomy.defaults.on_limit =
        Some(serde_json::from_value(json!({"action":"pause_run"})).unwrap());
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(), "hook result".into(),
        json!({"tool":"file_read","arguments":{"path":sandbox.source}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":sandbox.source,"old_text":"{ 1 }","new_text":"{ 2 }","dry_run":false}}).to_string(),
    ])));
    let hook = Arc::new(FeedbackCapacityAndUnavailableHook {
        runtime: Mutex::new(std::sync::Weak::new()),
        revision: revision.clone(),
        calls: AtomicU64::new(0),
    });
    let runner = runner_with_extensions(&setup, provider, extensions, Some(hook.clone()));
    *hook.runtime.lock().unwrap() = Arc::downgrade(runner.runtime());
    let check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let capacity = respond(&runner, check, ApprovalResult::Approved).await;
    assert_eq!(
        capacity.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        capacity.run.stop_reason
    );
    assert_eq!(hook.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        capacity.run.pending.as_ref().unwrap().reviewed_action["proposal"]["limit"]["model"]["feedback_hook_completed"],
        true
    );
    revision.0.store(1, Ordering::SeqCst);
    let mut limits = capacity.run.limits.clone();
    limits.max_llm_calls = Some(4);
    let revised = runner
        .revise_limits_paused(
            &capacity.run.key.run_id,
            capacity.run.revision,
            limits,
            "continue after host recovery",
        )
        .await
        .unwrap();
    let resumed = runner
        .resume(
            &revised.run.key.run_id,
            revised.run.revision,
            TaskResumeInput::Continue,
        )
        .await
        .unwrap();
    assert_eq!(hook.calls.load(Ordering::SeqCst), 1);
    assert_eq!(resumed.run.status, TaskRunStatus::Paused);
    cancel_if_paused(&runner, &resumed).await;
    drop(runner);
    sandbox.cleanup();
}

struct FeedbackTargetUnavailableHook(Arc<WorkspaceRevision>);

// A lost host revision is not an empty or successful observation.
#[async_trait]
impl ai_agents::AgentHooks for FeedbackTargetUnavailableHook {
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_llm_start(&self, messages: &[ChatMessage]) {
        if messages.iter().any(|message| {
            message
                .content
                .contains("Untrusted output from a failed authorized validation check")
        }) {
            self.0.0.store(u64::MAX, Ordering::SeqCst);
        }
    }
}

// An unavailable source must stop before disclosing checker output to the provider.
#[tokio::test]
async fn target_bound_feedback_rejects_unavailable_revision() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(Provider::new(VecDeque::from([
        "Ready for checking.".into(),
        "must not run".into(),
    ])));
    let runner = runner_with_extensions(
        &setup,
        provider.clone(),
        extensions,
        Some(Arc::new(FeedbackTargetUnavailableHook(revision))),
    );
    let check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let stopped = respond(&runner, check, ApprovalResult::Approved).await;
    assert_ne!(stopped.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
    cancel_if_paused(&runner, &stopped).await;
    drop(runner);
    sandbox.cleanup();
}

struct FeedbackChangingProvider {
    revision: Arc<WorkspaceRevision>,
    requests: AtomicU64,
}

// Change the host revision while the admitted feedback request is in flight.
#[async_trait]
impl LLMProvider for FeedbackChangingProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        if messages.iter().any(|message| {
            message
                .content
                .contains("Untrusted output from a failed authorized validation check")
        }) {
            self.revision.0.fetch_add(1, Ordering::SeqCst);
            Ok(LLMResponse::new("late result", FinishReason::Stop))
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
        "feedback-changing-test"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

// The provider result is not admitted after its target changes during the call.
#[tokio::test]
async fn target_bound_feedback_rejects_late_provider_result() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let revision = Arc::new(WorkspaceRevision(AtomicU64::new(1)));
    let extensions = setup
        .extensions_with_target(Some(("workspace", revision.clone())))
        .unwrap();
    let provider = Arc::new(FeedbackChangingProvider {
        revision,
        requests: AtomicU64::new(0),
    });
    let agent = Arc::new(
        setup
            .builder()
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(extensions)
            .build()
            .unwrap(),
    );
    agent.set_command_runner(setup.checker_runner().expect("checker installed"));
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "feedback-late.v1".into()).unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        setup.spec.autonomy.clone(),
        Default::default(),
        store,
        "feedback-late.v1".into(),
    )
    .unwrap();
    let check = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let stopped = respond(&runner, check, ApprovalResult::Approved).await;
    assert_eq!(stopped.run.status, TaskRunStatus::RecoveryRequired);
    assert_eq!(provider.requests.load(Ordering::SeqCst), 2);
    assert!(
        !stopped
            .final_response
            .as_ref()
            .is_some_and(|response| response.content.contains("late result"))
    );
    // A returned provider call with rejected settlement is not proof of cleanup or safe fixture removal.
    drop(runner);
    assert!(sandbox.parent.exists());
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

// Selected tracked and untracked files retain existing user changes and the index while one real checker validates all three.
#[tokio::test]
async fn selected_multifile_repairs_keep_dirty_workspace_and_require_fresh_check() {
    let sandbox = Sandbox::new();
    let second = sandbox.root.join("second.rs");
    let third = sandbox.root.join("third.rs");
    let untracked = sandbox.root.join("untracked.txt");
    fs::write(&second, "pub fn value() -> i32 { 1 }\n").unwrap();
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&sandbox.root)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
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
    git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "fixture baseline",
    ]);
    fs::write(
        &sandbox.source,
        "pub fn value() -> i32 { 1 } // staged user note\n",
    )
    .unwrap();
    git(&["add", "--", "source.rs"]);
    let staged = ["source.rs", "second.rs", "protected.rs"]
        .map(|file| git(&["--no-pager", "show", &format!(":{file}")]));
    fs::write(
        &sandbox.source,
        "pub fn value() -> i32 { 1 } // staged user note + unstaged note\n",
    )
    .unwrap();
    fs::write(&sandbox.outside, "unstaged user change\n").unwrap();
    fs::write(
        &third,
        "pub fn value() -> i32 { 1 } // untracked user note\n",
    )
    .unwrap();
    fs::write(&untracked, "untracked user data\n").unwrap();
    let before = git(&["--no-optional-locks", "status", "--porcelain=v1"]);
    assert!(String::from_utf8_lossy(&before).contains("MM source.rs"));
    assert!(String::from_utf8_lossy(&before).contains("protected.rs"));
    assert!(String::from_utf8_lossy(&before).contains("?? third.rs"));
    assert!(String::from_utf8_lossy(&before).contains("untracked.txt"));

    let checker_root = sandbox.parent.join("checker");
    let harness = checker_root.join("multi_acceptance.rs");
    fs::write(&harness, format!("#[path = {:?}] mod first;\n#[path = {:?}] mod second;\n#[path = {:?}] mod third;\n#[test] fn expected_values() {{ assert_eq!(first::value(), 2); assert_eq!(second::value(), 3); assert_eq!(third::value(), 4); }}\n", sandbox.source, second, third)).unwrap();
    let rustc = fs::canonicalize(env!("CARGO"))
        .unwrap()
        .parent()
        .unwrap()
        .join("rustc");
    let binary = checker_root.join("multi-acceptance");
    fs::write(&sandbox.checker, format!("#!/bin/sh\nset -eu\nPATH=/usr/bin:/bin\nexport PATH\n'{}' --edition 2024 --test '{}' -o '{}'\nexec '{}' --test-threads=1\n", rustc.display(), harness.display(), binary.display(), binary.display())).unwrap();
    let setup = prepare_files(
        &sandbox.root,
        &[
            PathBuf::from("source.rs"),
            PathBuf::from("second.rs"),
            PathBuf::from("third.rs"),
        ],
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
        json!({"tool":"file_read","arguments":{"path":third}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":third,"old_text":"{ 1 }","new_text":"{ 1 + 3 }","dry_run":false}}).to_string(),
        "First edit ready.".into(),
        json!({"tool":"file_read","arguments":{"path":second}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":second,"old_text":"{ 1 + 3 }","new_text":"{ 1 + 2 }","dry_run":false}}).to_string(),
        "Repair ready.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let first = runner
        .run("Fix all three selected functions.", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let second_edit = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(second_edit.run.status, TaskRunStatus::Paused);
    let third_edit = respond(&runner, second_edit, ApprovalResult::Approved).await;
    assert_eq!(third_edit.run.status, TaskRunStatus::Paused);
    let first_check = respond(&runner, third_edit, ApprovalResult::Approved).await;
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
        "pub fn value() -> i32 { 1 + 1 } // staged user note + unstaged note\n"
    );
    assert_eq!(
        fs::read_to_string(&second).unwrap(),
        "pub fn value() -> i32 { 1 + 2 }\n"
    );
    assert_eq!(
        fs::read_to_string(&third).unwrap(),
        "pub fn value() -> i32 { 1 + 3 } // untracked user note\n"
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
        4
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
    {
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
    }
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
            .unwrap()
            .llm_registry(registry)
            .auto_configure_features()
            .unwrap()
            .autonomy_extensions(setup.extensions().unwrap())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(setup.checker_runner().expect("checker installed"));
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
            .unwrap()
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
    agent.set_command_runner(setup.checker_runner().expect("checker installed"));
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
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .approval_handler(Arc::new(RejectEditApproveChecker))
            .autonomy_extensions(setup.extensions().unwrap())
            .build()
            .unwrap(),
    );
    agent.set_command_runner(setup.checker_runner().expect("checker installed"));
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
                "command_exit": {"command": setup.checker.as_ref().unwrap().command, "code": 0}
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
        .unwrap()
        .llm(scripted_provider(&setup, &sandbox.source))
        .auto_configure_features()
        .unwrap()
        .autonomy_extensions(setup.extensions().unwrap())
        .task_interaction_delivery(true)
        .build_task_host()
        .await
        .unwrap();
    agent.set_command_runner(setup.checker_runner().expect("checker installed"));
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

// An exact answer resumes the retained objective without treating guidance as approval or replaying the prior edit.
#[tokio::test]
async fn workspace_question_resumes_without_replaying_edit_or_skipping_checker() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let provider = scripted_provider(&setup, &sandbox.source);
    provider
        .responses
        .lock()
        .unwrap()
        .push_back("Ready for the fixed check.".into());
    let runner = runner_with_provider(&setup, provider);
    let edit = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    let checker = respond(&runner, edit, ApprovalResult::Approved).await;
    let question = respond(&runner, checker, ApprovalResult::Rejected { reason: None }).await;
    assert_eq!(question.run.status, TaskRunStatus::Paused);
    let interaction = runner
        .task_interaction(&question.run.key.run_id, question.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(interaction.prompt, TaskHostPrompt::Question(_)));
    let next = runner
        .resume(
            &question.run.key.run_id,
            question.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: interaction.request_id,
                answer: json!("Please request the fixed checker again."),
            },
        )
        .await
        .unwrap();
    assert_eq!(next.run.status, TaskRunStatus::Paused);
    let next_interaction = runner
        .task_interaction(&next.run.key.run_id, next.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        next_interaction.prompt,
        TaskHostPrompt::Approval { .. }
    ));
    assert!(
        !next
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|entry| entry.record.canonical_id == "command" && entry.record.executed)
    );
    let done = respond(&runner, next, ApprovalResult::Approved).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
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
        1
    );
    assert_eq!(
        done.run
            .evidence
            .tool_calls
            .iter()
            .filter(|entry| entry.record.canonical_id == "command" && entry.record.executed)
            .count(),
        1
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 }\n"
    );
    drop(runner);
    sandbox.cleanup();
}

// Stale approval and question responses cannot consume a pending decision or replay the completed edit.
#[tokio::test]
async fn workspace_rejects_stale_responses_without_replaying_completed_work() {
    let sandbox = Sandbox::new();
    let setup = sandbox.setup();
    let provider = scripted_provider(&setup, &sandbox.source);
    let runner = runner_with_provider(&setup, provider.clone());
    let edit = runner
        .run("Fix the selected function.", None)
        .await
        .unwrap();
    assert_eq!(edit.run.status, TaskRunStatus::Paused);
    let edit_id = edit.run.key.run_id.clone();
    let edit_interaction = runner
        .task_interaction(&edit_id, edit.run.revision)
        .await
        .unwrap()
        .unwrap();
    let requests = provider.requests.lock().unwrap().len();
    assert!(
        runner
            .resume(
                &edit_id,
                edit.run.revision + 1,
                TaskResumeInput::Approval {
                    request_id: edit_interaction.request_id.clone(),
                    result: ApprovalResult::Approved,
                }
            )
            .await
            .is_err()
    );
    assert!(
        runner
            .resume(
                &edit_id,
                edit.run.revision,
                TaskResumeInput::Approval {
                    request_id: "wrong-request".into(),
                    result: ApprovalResult::Approved,
                }
            )
            .await
            .is_err()
    );
    assert_eq!(provider.requests.lock().unwrap().len(), requests);
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 }\n"
    );
    let checker = respond(&runner, edit, ApprovalResult::Approved).await;
    let question = respond(&runner, checker, ApprovalResult::Rejected { reason: None }).await;
    assert_eq!(question.run.status, TaskRunStatus::Paused);
    let interaction = runner
        .task_interaction(&edit_id, question.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(interaction.prompt, TaskHostPrompt::Question(_)));
    let requests = provider.requests.lock().unwrap().len();
    assert!(
        runner
            .resume(
                &edit_id,
                question.run.revision + 1,
                TaskResumeInput::UserAnswer {
                    request_id: interaction.request_id.clone(),
                    answer: json!("continue"),
                }
            )
            .await
            .is_err()
    );
    assert!(
        runner
            .resume(
                &edit_id,
                question.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: "wrong-request".into(),
                    answer: json!("continue"),
                }
            )
            .await
            .is_err()
    );
    assert_eq!(provider.requests.lock().unwrap().len(), requests);
    assert_eq!(
        question
            .run
            .evidence
            .tool_calls
            .iter()
            .filter(|entry| entry.record.canonical_id == "file_edit"
                && entry.record.executed
                && entry.record.success)
            .count(),
        1
    );
    assert!(
        !question
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|entry| entry.record.canonical_id == "command" && entry.record.executed)
    );
    assert_eq!(
        fs::read_to_string(&sandbox.source).unwrap(),
        "pub fn value() -> i32 { 1 + 1 }\n"
    );
    cancel_if_paused(&runner, &question).await;
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

// Root mode keeps the acceptance test inside the workspace, so protecting checker inputs is observable rather than assumed.
struct RootSandbox {
    parent: PathBuf,
    root: PathBuf,
    checker: PathBuf,
    acceptance: PathBuf,
}

impl RootSandbox {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from("/tmp").join(format!(
            "code-agent-root-test-{}-{nonce}-{}",
            std::process::id(),
            NEXT_SANDBOX.fetch_add(1, Ordering::Relaxed)
        ));
        let root = parent.join("workspace");
        let checker_root = parent.join("checker");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("tests")).unwrap();
        fs::create_dir(&checker_root).unwrap();
        for name in ["a.rs", "b.rs"] {
            fs::write(root.join("src").join(name), "pub fn value() -> i32 { 1 }\n").unwrap();
        }
        let acceptance = root.join("tests").join("acceptance.rs");
        fs::write(&acceptance, "#[path = \"../src/a.rs\"] mod a;\n#[path = \"../src/b.rs\"] mod b;\n#[path = \"../src/c.rs\"] mod c;\n#[test] fn expected_values() { assert_eq!(a::value(), 2); assert_eq!(b::value(), 3); assert_eq!(c::value(), 4); }\n").unwrap();
        let rustc = fs::canonicalize(env!("CARGO"))
            .unwrap()
            .parent()
            .unwrap()
            .join("rustc");
        let binary = checker_root.join("root-acceptance");
        let checker = checker_root.join("check.sh");
        // The fixed checker receives only the workspace root and compiles the protected test inside it.
        fs::write(&checker, format!("#!/bin/sh\nset -eu\nPATH=/usr/bin:/bin\nexport PATH\n'{}' --edition 2024 --test \"$1/tests/acceptance.rs\" -o '{}'\nexec '{}' --test-threads=1\n", rustc.display(), binary.display(), binary.display())).unwrap();
        fs::set_permissions(&checker, fs::Permissions::from_mode(0o700)).unwrap();
        Self {
            parent,
            root,
            checker,
            acceptance,
        }
    }

    fn setup(&self) -> WorkspaceSetup {
        let mut setup = prepare_root(
            &self.root,
            &[PathBuf::from("tests")],
            &self.checker,
            "openai",
            "offline-test",
            None,
        )
        .unwrap();
        setup.assume_exclusive_workspace().unwrap();
        setup
    }

    fn git(&self, args: &[&str]) -> Vec<u8> {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&self.root)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git fixture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    fn cleanup(self) {
        fs::remove_dir_all(self.parent).unwrap();
    }
}

// Approve every retained request and answer a guidance question, returning the order of tool approvals and questions.
async fn settle_prompts(
    runner: &AutonomyRunner,
    mut result: TaskRunResult,
    guidance: &str,
) -> (TaskRunResult, Vec<String>) {
    let mut prompts = Vec::new();
    while result.run.status == TaskRunStatus::Paused {
        let interaction = runner
            .task_interaction(&result.run.key.run_id, result.run.revision)
            .await
            .unwrap()
            .expect("retained interaction");
        assert!(prompts.len() <= 12, "unbounded prompts: {prompts:?}");
        result = match &interaction.prompt {
            TaskHostPrompt::Approval { trigger, .. } => {
                prompts.push(
                    trigger
                        .get("name")
                        .and_then(|name| name.as_str())
                        .unwrap_or("unnamed")
                        .to_string(),
                );
                respond(runner, result, ApprovalResult::Approved).await
            }
            TaskHostPrompt::Question(_) => {
                prompts.push("question".into());
                runner
                    .resume(
                        &result.run.key.run_id,
                        result.run.revision,
                        TaskResumeInput::UserAnswer {
                            request_id: interaction.request_id,
                            answer: json!(guidance),
                        },
                    )
                    .await
                    .unwrap()
            }
            other => panic!("unexpected prompt after {prompts:?}: {other:?}"),
        };
    }
    (result, prompts)
}

fn tool_record<'a>(
    result: &'a TaskRunResult,
    tool: &str,
    path: &Path,
) -> Vec<&'a ai_agents::tools::ToolExecutionRecord> {
    result
        .run
        .evidence
        .tool_calls
        .iter()
        .map(|entry| &entry.record)
        .filter(|record| {
            record.canonical_id == tool && record.arguments.get("path") == Some(&json!(path))
        })
        .collect()
}

#[tokio::test]
async fn root_mode_configuration_binds_the_root_and_protects_checker_inputs() {
    let sandbox = RootSandbox::new();
    let reject = |protected: &[&str], checker: &Path| {
        let protected: Vec<_> = protected.iter().map(PathBuf::from).collect();
        prepare_root(
            &sandbox.root,
            &protected,
            checker,
            "openai",
            "offline-test",
            None,
        )
        .err()
        .map(|error| error.to_string())
        .expect("configuration must be rejected")
    };
    assert!(reject(&[], &sandbox.checker).contains("at least one protected"));
    assert!(reject(&["tests", "tests"], &sandbox.checker).contains("repeated"));
    assert!(reject(&["../outside"], &sandbox.checker).contains("normal components"));
    assert!(!reject(&["missing"], &sandbox.checker).is_empty());
    std::os::unix::fs::symlink(sandbox.root.join("tests"), sandbox.root.join("alias")).unwrap();
    assert!(reject(&["alias"], &sandbox.checker).contains("symlinked"));
    let inside = sandbox.root.join("check.sh");
    fs::copy(&sandbox.checker, &inside).unwrap();
    assert!(reject(&["tests"], &inside).contains("outside the workspace"));

    let mut setup = sandbox.setup();
    let root = sandbox.root.to_str().unwrap();
    let protected = sandbox.root.join("tests");
    let policies = &setup.spec.tool_security.tools;
    for name in ["file_edit", "file_write"] {
        assert_eq!(policies[name].write_paths, vec![root]);
        assert_eq!(policies[name].paths.deny, vec![protected.to_str().unwrap()]);
        assert!(policies[name].require_confirmation);
    }
    assert!(!policies["file_write"].overwrite_existing);
    assert!(policies["file_edit"].require_read_before_write);
    for name in ["file_read", "file_list", "glob", "grep"] {
        assert_eq!(policies[name].read_paths, vec![root]);
    }
    assert_eq!(
        setup.checker.as_ref().unwrap().argv,
        vec![sandbox.checker.to_str().unwrap(), root]
    );
    assert!(setup.discovery_enabled());
    assert!(setup.cooperative_revision_enabled());
    assert_eq!(setup.source_file(), None);
    assert!(setup.enable_discovery().is_err());
    assert!(setup.enable_automatic_instructions().is_err());
    // The guard binds exactly one runtime, so a second builder cannot share its revision.
    let _builder = setup.builder().unwrap();
    assert!(setup.builder().is_err());
    sandbox.cleanup();
}

// The model finds files itself, recovers from an ordinary tool rejection, cannot weaken the protected test and needs a fresh real check after repair.
// Its denied attempt on the checker input turns the failed check into a human question instead of automatic feedback.
#[tokio::test]
async fn root_mode_discovers_repairs_files_and_keeps_user_work_and_checker_inputs() {
    use ai_agents_cli::CliTaskObserver;
    let sandbox = RootSandbox::new();
    let a = sandbox.root.join("src/a.rs");
    let b = sandbox.root.join("src/b.rs");
    let c = sandbox.root.join("src/c.rs");
    let scratch = sandbox.root.join("scratch.txt");
    sandbox.git(&["init", "--quiet"]);
    sandbox.git(&["add", "--", "src", "tests"]);
    sandbox.git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "fixture baseline",
    ]);
    fs::write(&a, "pub fn value() -> i32 { 1 } // staged note\n").unwrap();
    sandbox.git(&["add", "--", "src/a.rs"]);
    let staged = ["src/a.rs", "src/b.rs", "tests/acceptance.rs"]
        .map(|file| sandbox.git(&["--no-pager", "show", &format!(":{file}")]));
    fs::write(
        &a,
        "pub fn value() -> i32 { 1 } // staged note + unstaged note\n",
    )
    .unwrap();
    fs::write(&scratch, "untracked user data\n").unwrap();
    let acceptance = fs::read_to_string(&sandbox.acceptance).unwrap();

    let setup = sandbox.setup();
    let guard = setup.guard.as_ref().unwrap().clone();
    let provider = Arc::new(Provider::new(VecDeque::from([
        json!({"tool":"grep","arguments":{"path":sandbox.root,"pattern":"fn value"}}).to_string(),
        json!({"tool":"file_write","arguments":{"path":sandbox.acceptance,"content":"#[test] fn expected_values() {}\n","overwrite":true}}).to_string(),
        json!({"tool":"file_write","arguments":{"path":a,"content":"pub fn value() -> i32 { 2 }\n"}}).to_string(),
        json!({"tool":"file_read","arguments":{"path":a}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":a,"old_text":"{ 1 }","new_text":"{ 1 + 1 }"}}).to_string(),
        json!({"tool":"file_read","arguments":{"path":b}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":b,"old_text":"{ 1 }","new_text":"{ 1 + 1 }"}}).to_string(),
        json!({"tool":"file_write","arguments":{"path":c,"content":"pub fn value() -> i32 { 4 }\n"}}).to_string(),
        "Ready for checking.".into(),
        json!({"tool":"file_read","arguments":{"path":b}}).to_string(),
        json!({"tool":"file_edit","arguments":{"path":b,"old_text":"{ 1 + 1 }","new_text":"{ 1 + 2 }"}}).to_string(),
        "Repair ready.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider.clone());
    let report = crate::workspace_report::WorkspaceReport::new(sandbox.root.clone(), Vec::new());
    report.before_task().await.unwrap();
    let started = runner
        .run("Make the acceptance test pass.", None)
        .await
        .unwrap();
    let guidance = "Leave the acceptance test unchanged and fix the source so it returns 3.";
    let (done, prompts) = settle_prompts(&runner, started, guidance).await;
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?} after {prompts:?}",
        done.run.stop_reason
    );
    assert_eq!(
        prompts,
        [
            "file_write",
            "file_edit",
            "file_edit",
            "file_write",
            "command",
            "question",
            "file_edit",
            "command"
        ]
    );

    // Discovery ran, the protected input was denied before approval and the ordinary rejection settled without effect.
    assert!(
        tool_record(&done, "grep", &sandbox.root)
            .iter()
            .any(|record| record.executed && record.success)
    );
    let denied = tool_record(&done, "file_write", &sandbox.acceptance);
    assert!(!denied.is_empty() && denied.iter().all(|record| !record.executed));
    let rejected = tool_record(&done, "file_write", &a);
    assert_eq!(rejected.len(), 1);
    assert!(rejected[0].executed && !rejected[0].success);
    assert!(
        rejected[0].output.contains("overwrite must be true"),
        "{}",
        rejected[0].output
    );
    assert!(!rejected[0].metadata.contains_key("task_effect_uncertain"));

    // One real failing check and one fresh passing check, with the token advanced only by the four performed changes.
    let exits: Vec<_> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "command" && entry.record.executed)
        .map(|entry| {
            serde_json::from_str::<serde_json::Value>(&entry.record.output).unwrap()["exit_code"]
                .as_i64()
                .unwrap()
        })
        .collect();
    assert_eq!(exits.len(), 2);
    assert_ne!(exits[0], 0);
    assert_eq!(exits[1], 0);
    assert_eq!(guard.revision().as_deref(), Some("8"));
    assert!(!guard.requires_reconciliation());

    assert_eq!(
        fs::read_to_string(&a).unwrap(),
        "pub fn value() -> i32 { 1 + 1 } // staged note + unstaged note\n"
    );
    assert_eq!(
        fs::read_to_string(&b).unwrap(),
        "pub fn value() -> i32 { 1 + 2 }\n"
    );
    assert_eq!(
        fs::read_to_string(&c).unwrap(),
        "pub fn value() -> i32 { 4 }\n"
    );
    assert_eq!(fs::read_to_string(&sandbox.acceptance).unwrap(), acceptance);
    assert_eq!(
        fs::read_to_string(&scratch).unwrap(),
        "untracked user data\n"
    );
    for (index, file) in ["src/a.rs", "src/b.rs", "tests/acceptance.rs"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            sandbox.git(&["--no-pager", "show", &format!(":{file}")]),
            staged[index]
        );
    }
    let text = report.after_task(Some(&done)).await.unwrap();
    for target in ["src/a.rs", "src/b.rs", "src/c.rs"] {
        assert!(
            text.contains(&format!("Recorded mutation target {target:?}")),
            "{text}"
        );
    }
    assert!(!text.contains("tests/acceptance.rs\""), "{text}");
    let requests = provider.requests.lock().unwrap();
    assert!(requests.iter().flatten().all(|message| {
        !message
            .content
            .contains("Untrusted output from a failed authorized validation check")
    }));
    assert!(
        requests
            .iter()
            .flatten()
            .any(|message| message.content.contains(guidance))
    );
    drop(requests);
    drop(runner);
    sandbox.cleanup();
}

struct FailedTerminalStore {
    inner: Arc<ScopedTaskRunStore>,
    mode: &'static str,
    reached: tokio::sync::Semaphore,
}

#[async_trait]
impl TaskRunStore for FailedTerminalStore {
    // Forward setup and ordinary checkpoints without changing their guards or identity.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents::Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> ai_agents::Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    // Only the resumed execution-error terminal write is rejected, loses its receipt or is dropped after commit.
    async fn mutate(
        &self,
        id: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents::Result<TaskRunSnapshot> {
        let selected = matches!(
            mutation,
            TaskRunMutation::FinalCheckpoint {
                status: TaskRunStatus::Failed,
                ..
            }
        );
        if selected && self.mode == "reject" {
            self.reached.add_permits(1);
            return Err(ai_agents::AgentError::Persistence(
                "rejected terminal receipt".into(),
            ));
        }
        let saved = self.inner.mutate(id, mutation).await?;
        if selected {
            self.reached.add_permits(1);
            if self.mode == "drop" {
                std::future::pending::<()>().await;
            }
            return Err(ai_agents::AgentError::Persistence(
                "lost terminal receipt".into(),
            ));
        }
        Ok(saved)
    }
    async fn list(&self) -> ai_agents::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> ai_agents::Result<()> {
        self.inner.delete(id, revision).await
    }
}

// Failure settlement must keep live custody after a missing terminal ACK without replaying approvals or refunding completed work.
#[tokio::test]
async fn root_mode_resume_error_terminal_ack_failure_keeps_custody() {
    for mode in ["reject", "lost", "drop"] {
        let sandbox = RootSandbox::new();
        let c = sandbox.root.join("src/c.rs");
        let d = sandbox.root.join("src/d.rs");
        let setup = sandbox.setup();
        let provider = Arc::new(Provider::new(VecDeque::from([
            json!({"tool":"file_write","arguments":{"path":c,"content":"pub fn value() -> i32 { 4 }\n"}}).to_string(),
            json!({"tool":"file_write","arguments":{"path":d,"content":"pub fn other() {}\n"}}).to_string(),
            json!({"tool":"file_read","arguments":{"path":c}}).to_string(),
            "Ready for checking.".into(),
        ])));
        let prepared = runner_with_provider(&setup, provider.clone());
        let agent = prepared.runtime().clone();
        let store = Arc::new(FailedTerminalStore {
            inner: Arc::new(
                ScopedTaskRunStore::in_memory(agent.info().id, None, "workspace-test.v1".into())
                    .unwrap(),
            ),
            mode,
            reached: tokio::sync::Semaphore::new(0),
        });
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            setup.spec.autonomy.clone(),
            Default::default(),
            store.clone(),
            "workspace-test.v1".into(),
        )
        .unwrap();
        drop(prepared);
        let first = runner.run("Create the files.", None).await.unwrap();
        let first_request = first.run.pending.as_ref().unwrap().id.clone();
        let second = respond(&runner, first, ApprovalResult::Approved).await;
        let id = second.run.key.run_id.clone();
        let request_id = second.run.pending.as_ref().unwrap().id.clone();
        let mut resumed = Box::pin(runner.resume(
            &id,
            second.run.revision,
            TaskResumeInput::Approval {
                request_id: request_id.clone(),
                result: ApprovalResult::Approved,
            },
        ));
        let replacement = sandbox.root.join("replacement.rs");
        fs::write(&replacement, "user replacement\n").unwrap();
        fs::rename(replacement, &c).unwrap();
        if mode == "drop" {
            tokio::select! {
                result = &mut resumed => panic!("unacknowledged terminal returned: {result:?}"),
                permit = store.reached.acquire() => permit.unwrap().forget(),
            }
        } else {
            assert!(matches!(
                resumed.as_mut().await,
                Err(ai_agents::AgentError::Persistence(_))
            ));
            store.reached.acquire().await.unwrap().forget();
        }
        drop(resumed);
        let snapshot = store.load(&id).await.unwrap().unwrap();
        assert_eq!(
            snapshot.status,
            if mode == "reject" {
                TaskRunStatus::Running
            } else {
                TaskRunStatus::Failed
            }
        );
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        assert_eq!(
            payload.consumed_request_ids,
            vec![first_request, request_id.clone()]
        );
        assert!(payload.pending.is_none());
        assert_eq!(payload.counters.tool_attempts, 3);
        assert_eq!(payload.counters.llm_attempts, 4);
        assert_eq!(provider.requests.lock().unwrap().len(), 4);
        assert!(agent.chat("blocked").await.is_err());
        assert!(
            runner
                .resume(
                    &id,
                    second.run.revision,
                    TaskResumeInput::Approval {
                        request_id,
                        result: ApprovalResult::Approved
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(provider.requests.lock().unwrap().len(), 4);
        assert_eq!(fs::read_to_string(&c).unwrap(), "user replacement\n");
        assert_eq!(fs::read_to_string(&d).unwrap(), "pub fn other() {}\n");
        assert!(!sandbox.parent.join("checker/root-acceptance").exists());
        drop(runner);
        drop(agent);
        sandbox.cleanup();
    }
}

// An external replacement of an agent-created file makes the bound revision unavailable, so no later check can complete the task.
#[tokio::test]
async fn root_mode_external_replacement_prevents_completion() {
    let sandbox = RootSandbox::new();
    let c = sandbox.root.join("src/c.rs");
    let d = sandbox.root.join("src/d.rs");
    let setup = sandbox.setup();
    let guard = setup.guard.as_ref().unwrap().clone();
    let provider = Arc::new(Provider::new(VecDeque::from([
        json!({"tool":"file_write","arguments":{"path":c,"content":"pub fn value() -> i32 { 4 }\n"}}).to_string(),
        json!({"tool":"file_write","arguments":{"path":d,"content":"pub fn other() {}\n"}}).to_string(),
        json!({"tool":"file_read","arguments":{"path":c}}).to_string(),
        "Ready for checking.".into(),
    ])));
    let runner = runner_with_provider(&setup, provider);
    let first = runner.run("Create the file.", None).await.unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let second = respond(&runner, first, ApprovalResult::Approved).await;
    assert_eq!(second.run.status, TaskRunStatus::Paused);
    assert_eq!(guard.revision().as_deref(), Some("2"));

    // Another writer replaces the created file while the task waits for approval.
    let replacement = sandbox.root.join("replacement.rs");
    fs::write(&replacement, "pub fn value() -> i32 { 99 }\n").unwrap();
    fs::rename(&replacement, &c).unwrap();
    let run_id = second.run.key.run_id.clone();
    let interaction = runner
        .task_interaction(&run_id, second.run.revision)
        .await
        .unwrap()
        .unwrap();
    let resumed = runner
        .resume(
            &run_id,
            second.run.revision,
            TaskResumeInput::Approval {
                request_id: interaction.request_id,
                result: ApprovalResult::Approved,
            },
        )
        .await;
    let failed = resumed.unwrap();
    assert_eq!(failed.run.status, TaskRunStatus::Failed);
    assert_eq!(failed.run.stop_reason.as_deref(), Some("execution_error"));
    let display = runner.task_display(&run_id).await.unwrap();
    assert_eq!(display.summary.status, TaskRunStatus::Failed);
    assert!(!display.locally_retained);
    assert!(guard.requires_reconciliation());
    assert_eq!(guard.revision(), None);
    assert_eq!(
        fs::read_to_string(&c).unwrap(),
        "pub fn value() -> i32 { 99 }\n"
    );
    assert!(d.exists());
    // The checker writes its test binary when it runs, so its absence shows no check was attempted.
    assert!(!sandbox.parent.join("checker/root-acceptance").exists());
    // Settled effects can fail normally even while the guard still requires host reconciliation before reuse.
    assert!(
        runner
            .task_interaction(&run_id, failed.run.revision)
            .await
            .unwrap()
            .is_none()
    );
    drop(runner);
    sandbox.cleanup();
}
