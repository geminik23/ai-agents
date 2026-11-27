// Real-server acceptance uses guarded Code Agent edits and the configured restore host.
use super::*;
use ai_agents_core::{AgentError, LLMChunk, LLMConfig, LLMFeature, LLMResponse};
use ai_agents_core::{LLMError, LLMProvider};
use ai_agents_runtime::spec::PreparedAgentSpec;
use ai_agents_tools::{
    AskUserTool, DiagnosticsTool, ToolRegistry, WorkspaceGuard, WorkspaceScope,
    lsp::{LspDiagnostics, LspDiagnosticsConfig, LspDiagnosticsHost},
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::Ordering,
    time::Duration,
};

const ORIGINAL: &str = "pub fn clamp(value: i32, lower: i32, upper: i32) -> i32 {\n    let _: i32 = \"wrong\";\n    value.max(lower).max(upper)\n}\n";
const BROKEN_LINE: &str = "    let _: i32 = \"wrong\";\n    value.max(lower).max(upper)";
const FIX: &str = "    value.max(lower).min(upper)";
const PATH: &str = "src/lib.rs";

fn project(root: &Path) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"diagnostics-test-project\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[workspace]\n",
    )
    .unwrap();
    fs::write(
        root.join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"diagnostics-test-project\"\nversion = \"0.0.0\"\n",
    )
    .unwrap();
    fs::write(root.join(PATH), ORIGINAL).unwrap();
}

fn server(root: &Path, env: &str, document: &str) -> LspDiagnosticsConfig {
    let program =
        PathBuf::from(std::env::var_os(env).expect("set the absolute language server path"));
    LspDiagnosticsConfig {
        root: root.into(),
        program,
        args: vec![],
        env: ["PATH", "HOME", "RUSTUP_HOME", "CARGO_HOME"]
            .into_iter()
            .filter_map(|name| std::env::var(name).ok().map(|value| (name.into(), value)))
            .collect(),
        documents: vec![document.into()],
        inputs: vec!["Cargo.toml".into(), "Cargo.lock".into()],
        language_id: "rust".into(),
        initialization_options: Value::Null,
        timeout_ms: 20_000,
    }
}

#[tokio::test]
#[ignore = "requires rust-analyzer and Ruff executables"]
async fn live_servers_use_current_documents_through_the_same_tool() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    project(root);
    let runtime = agent(200, 32_000);
    let install = |config| {
        runtime.set_diagnostics_provider(Arc::new(
            LspDiagnostics::rust_analyzer(config, || Some("exclusive-test-inputs".into())).unwrap(),
        ))
    };
    let target = root.join(PATH);
    let absolute = target.to_str().unwrap();
    install(server(root, "AI_AGENTS_RUST_ANALYZER", PATH));
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Fail);
    let mut config = server(root, "AI_AGENTS_RUST_ANALYZER", PATH);
    config.inputs.push("rust-analyzer.toml".into());
    fs::write(
        root.join("rust-analyzer.toml"),
        "[diagnostics]\nenable = false\n",
    )
    .unwrap();
    install(config);
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    fs::write(
        root.join("rust-analyzer.toml"),
        "[diagnostics]\nenable = true\n",
    )
    .unwrap();
    let mut config = server(root, "AI_AGENTS_RUST_ANALYZER", PATH);
    config.inputs.push("rust-analyzer.toml".into());
    config.initialization_options = json!({"cargo":{"features":["selected"]}});
    fs::OpenOptions::new()
        .append(true)
        .open(root.join("Cargo.toml"))
        .unwrap()
        .write_all(b"\n[features]\nselected = []\n")
        .unwrap();
    fs::write(
        root.join(PATH),
        format!("#[cfg(feature = \"selected\")]\n{ORIGINAL}"),
    )
    .unwrap();
    install(config);
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Fail);
    fs::write(root.join(PATH), ORIGINAL.replace(BROKEN_LINE, FIX)).unwrap();
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Pass);
    let (_, scope, evidence) = observe(runtime.clone(), ".").await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);

    // A discovered project override must travel with its snapshot, even when it currently enables analysis.
    install(server(root, "AI_AGENTS_RUST_ANALYZER", PATH));
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    fs::remove_file(root.join("rust-analyzer.toml")).unwrap();
    let mut config = server(root, "AI_AGENTS_RUST_ANALYZER", "outside.rs");
    fs::write(root.join("outside.rs"), "").unwrap();
    config.inputs.push(PATH.into());
    install(config);
    let (_, scope, evidence) =
        observe(runtime.clone(), root.join("outside.rs").to_str().unwrap()).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);

    let mut config = server(root, "AI_AGENTS_RUST_ANALYZER", PATH);
    config.initialization_options = json!({"diagnostics":{"enable":false}});
    install(config);
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);

    let mut config = server(root, "AI_AGENTS_RUST_ANALYZER", PATH);
    config.program = root.join("not-installed");
    install(config);
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Unknown);

    let mut registration: Value = serde_json::from_str(include_str!("ruff.json")).unwrap();
    registration["root"] = json!(root);
    registration["program"] = json!(std::env::var("AI_AGENTS_RUFF").unwrap());
    let config = serde_json::from_value(registration).unwrap();
    let rust = LspDiagnostics::rust_analyzer(server(root, "AI_AGENTS_RUST_ANALYZER", PATH), || {
        Some("exclusive-test-inputs".into())
    })
    .unwrap();
    let ruff = LspDiagnostics::new(config, || Some("exclusive-test-inputs".into())).unwrap();
    assert!(LspDiagnosticsHost::new(vec![rust.clone(), rust.clone()]).is_err());
    runtime.set_diagnostics_provider(Arc::new(LspDiagnosticsHost::new(vec![rust, ruff]).unwrap()));
    let (_, scope, evidence) = observe(runtime.clone(), absolute).await;
    assert_outcomes(&scope, &evidence, GateOutcome::Pass);
    let python = root.join("diagnostic.py");
    for (source, settings, count) in [
        ("print(missing)\n", "[tool.ruff]\n", 1),
        ("print('defined')\n", "[tool.ruff]\n", 0),
        (
            "print(missing)\n",
            "[tool.ruff]\nexclude=['diagnostic.py']\n",
            0,
        ),
    ] {
        fs::write(&python, source).unwrap();
        fs::write(root.join("pyproject.toml"), settings).unwrap();
        let (state, scope, evidence) = observe(runtime.clone(), python.to_str().unwrap()).await;
        let ObservationResult::Tool { record } = &state.completed["observe"].result else {
            panic!("expected diagnostics record")
        };
        let output: Value = serde_json::from_str(&record.output).unwrap();
        assert_eq!(output["available"], true, "{output}");
        assert_eq!(output["count"], count, "{output}");
        if count == 1 {
            assert_eq!(output["diagnostics"][0]["code"], "F821");
        }
        assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    }
}

// A new host generation cannot inherit the previous guard's revision counter.
struct InstalledRevision {
    generation: uuid::Uuid,
    guard: Arc<WorkspaceGuard>,
}

impl HostRevisionSource for InstalledRevision {
    fn revision(&self) -> ai_agents_core::Result<String> {
        let revision = self
            .guard
            .revision()
            .ok_or_else(|| AgentError::Config("workspace revision unavailable".into()))?;
        Ok(format!("{}:{revision}", self.generation))
    }
}

struct Application {
    directory: PathBuf,
}

struct Installed {
    configuration: Value,
    cursor: Arc<AtomicU64>,
}

impl TaskRestoreComponentState for Installed {
    fn binding(&self) -> ai_agents_core::Result<TaskRestoreBinding> {
        Ok(TaskRestoreBinding {
            id: "lsp-acceptance".into(),
            contract_version: 1,
            configuration: self.configuration.clone(),
            state: json!(self.cursor.load(Ordering::SeqCst)),
        })
    }
}

impl Application {
    fn configuration(&self) -> Value {
        json!({
            "directory": self.directory,
            "analyzer": std::env::var("AI_AGENTS_RUST_ANALYZER").unwrap(),
        })
    }
}

impl TaskRestoreComponents for Application {
    fn identity(&self) -> (String, u32) {
        ("lsp-acceptance".into(), 1)
    }

    fn validate(
        &self,
        _: &PreparedAgentSpec,
        binding: &TaskRestoreBinding,
    ) -> ai_agents_core::Result<()> {
        if binding.configuration == self.configuration() && binding.state.as_u64().is_some() {
            Ok(())
        } else {
            Err(AgentError::Config(
                "diagnostics component binding changed".into(),
            ))
        }
    }

    fn install(
        &self,
        _: &PreparedAgentSpec,
        builder: AgentBuilder,
        saved: Option<&TaskRestoreBinding>,
        _: Option<&ai_agents_llm::LLMRegistry>,
    ) -> ai_agents_core::Result<(AgentBuilder, Arc<dyn TaskRestoreComponentState>)> {
        let root = self.directory.join("project");
        let target = root.join(PATH);
        let guard = WorkspaceGuard::new(
            &root,
            WorkspaceScope::Selected([target.clone()].into()),
            vec![],
        )
        .map_err(|error| AgentError::Config(error.to_string()))?;
        let mut tools = ToolRegistry::new();
        tools
            .register(Arc::new(DiagnosticsTool::new(
                tools.diagnostics_provider_slot(),
            )))
            .unwrap();
        tools
            .register(Arc::new(AskUserTool::new(tools.question_handler_slot())))
            .unwrap();
        guard
            .install(&mut tools)
            .map_err(|error| AgentError::Config(error.to_string()))?;
        let revision = Arc::new(InstalledRevision {
            generation: uuid::Uuid::new_v4(),
            guard,
        });
        let source = revision.clone();
        let diagnostics = LspDiagnostics::rust_analyzer(
            server(&root, "AI_AGENTS_RUST_ANALYZER", PATH),
            move || source.revision().ok(),
        )?;
        tools.set_diagnostics_provider(Arc::new(diagnostics));
        let mut extensions = AutonomyExtensions::builtins();
        extensions.register_validation_target("syntax", "selected-document", revision)?;
        let cursor = Arc::new(AtomicU64::new(
            saved
                .and_then(|binding| binding.state.as_u64())
                .unwrap_or_default(),
        ));
        let builder = builder
            .tools(tools)
            .llm(Arc::new(EditProgram {
                cursor: cursor.clone(),
                target,
            }))
            .auto_configure_features()?
            .autonomy_extensions(extensions);
        Ok((
            builder,
            Arc::new(Installed {
                configuration: self.configuration(),
                cursor,
            }),
        ))
    }
}

struct EditProgram {
    cursor: Arc<AtomicU64>,
    target: PathBuf,
}

#[async_trait]
impl LLMProvider for EditProgram {
    async fn complete(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        let content = match self.cursor.fetch_add(1, Ordering::SeqCst) {
            0 => json!({"tool":"file_read","arguments":{"path":self.target}}).to_string(),
            1 => json!({"tool":"file_edit","arguments":{"path":self.target,"old_text":BROKEN_LINE,"new_text":FIX}}).to_string(),
            2 => json!({"tool":"diagnostics","arguments":{"path":self.target}}).to_string(),
            3 => json!({"tool":"ask_user","arguments":{"question":"Confirm the edit","options":["yes"],"allow_other":false}}).to_string(),
            4 => "done".into(),
            _ => return Err(LLMError::Config("unexpected repeated or extra model work".into())),
        };
        Ok(LLMResponse::new(
            content,
            ai_agents_core::FinishReason::Stop,
        ))
    }

    async fn complete_stream(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        Err(LLMError::Config("unexpected stream".into()))
    }

    fn provider_name(&self) -> &str {
        "lsp-edit-program"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

fn host(directory: &Path) -> Arc<ConfiguredTaskRestoreHost> {
    let spec = PreparedAgentSpec::from_yaml_strict(
        &fs::read_to_string(directory.join("agent.yaml")).unwrap(),
    )
    .unwrap();
    Arc::new(
        ConfiguredTaskRestoreHost::new(spec, directory.into(), None, Duration::from_secs(20))
            .unwrap()
            .components(Arc::new(Application {
                directory: directory.into(),
            }))
            .unwrap(),
    )
}

#[tokio::test]
async fn process_worker() {
    let Some(mode) = std::env::var_os("AI_AGENTS_LSP_PROCESS_MODE") else {
        return;
    };
    let directory = std::env::current_dir().unwrap();
    let host = host(&directory);
    let receipt = directory.join("paused.json");
    if mode == "seed" {
        let result = host
            .build()
            .await
            .unwrap()
            .run_task("Fix the Rust document", None)
            .await
            .unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Paused,
            "{:?} {:?}",
            result.run.stop_reason,
            result.run.evidence.tool_calls
        );
        assert_eq!(
            fs::read_to_string(directory.join("project").join(PATH)).unwrap(),
            ORIGINAL.replace(BROKEN_LINE, FIX),
            "pending={:?}, calls={:?}",
            result.run.pending,
            result.run.evidence.tool_calls
        );
        let previous = result
            .run
            .evidence
            .tool_calls
            .iter()
            .find(|entry| entry.record.canonical_id == "diagnostics" && entry.record.executed)
            .expect("pre-restart diagnostics ran");
        let previous_output: Value = serde_json::from_str(&previous.record.output).unwrap();
        assert_eq!(previous_output["available"], true, "{previous_output}");
        assert_eq!(previous_output["count"], 0);
        fs::write(
            receipt,
            json!({
                "id":result.run.key.run_id,
                "revision":result.run.revision,
                "attempts":result.run.counters.llm_attempts,
                "diagnostic_call":previous.record.call_id,
                "expires_at":result.run.clocks.expires_at.expect("wall allowance"),
            })
            .to_string(),
        )
        .unwrap();
        return;
    }
    let saved: Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
    let id = saved["id"].as_str().unwrap();
    let restored = host
        .restore(id.into(), saved["revision"].as_u64().unwrap())
        .await;
    if mode == "resume-rebound" {
        let error = restored
            .err()
            .expect("different installed server must refuse restoration");
        assert!(error.to_string().contains("binding"), "{error}");
        return;
    }
    let (runner, paused) = restored.unwrap();
    let review = runner.restoration_review(id).await.unwrap().unwrap();
    let reviewed = runner
        .approve_restoration(
            id,
            paused.run.revision,
            &review.id,
            ai_agents_hitl::ApprovalResult::Approved,
        )
        .await
        .unwrap();
    let outcome = runner
        .resume(
            id,
            reviewed.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: reviewed.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.run.counters.llm_attempts,
        saved["attempts"].as_u64().unwrap() + 1
    );
    assert_eq!(json!(outcome.run.clocks.expires_at), saved["expires_at"]);
    let edits = outcome
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "file_edit" && entry.record.executed)
        .count();
    assert_eq!(edits, 1);
    let mut diagnostics = outcome
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "diagnostics" && entry.record.executed);
    assert_eq!(
        json!(diagnostics.next().unwrap().record.call_id),
        saved["diagnostic_call"]
    );
    let diagnostics_record = diagnostics
        .next()
        .expect("new diagnostics ran after restart");
    assert_ne!(
        json!(diagnostics_record.record.call_id),
        saved["diagnostic_call"]
    );
    assert!(diagnostics.next().is_none());
    let output: Value = serde_json::from_str(&diagnostics_record.record.output).unwrap();
    if mode == "resume-unavailable" {
        assert_eq!(output["available"], false, "{output}");
        assert_ne!(outcome.run.status, TaskRunStatus::Completed);
        return;
    }
    assert_eq!(output["available"], true, "{output}");
    if mode == "resume-current" {
        assert_eq!(
            outcome.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            outcome.run.stop_reason
        );
        assert_eq!(output["count"], 0);
    } else {
        assert_ne!(outcome.run.status, TaskRunStatus::Completed);
        assert!(
            output["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["severity"] == "error"),
            "{output}"
        );
    }
}

#[tokio::test]
#[ignore = "requires rust-analyzer and native Linux restoration clock evidence"]
async fn restart_with_live_rust_diagnostics_does_not_replay_the_edit() {
    for mode in [
        "resume-current",
        "resume-changed",
        "resume-rebound",
        "resume-unavailable",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let project_root = directory.path().join("project");
        project(&project_root);
        let root = directory.path();
        fs::write(
            root.join("agent.yaml"),
            format!(
                r#"
name: DiagnosticsProcess
system_prompt: Complete the declared work.
tools: [file_read, file_edit, ask_user, diagnostics]
tool_security:
  enabled: true
  tools:
    file_read: {{allow_without_confirmation: true}}
    file_edit: {{allow_without_confirmation: true, require_read_before_write: true, write_paths: ['{0}']}}
    diagnostics: {{allow_without_confirmation: true}}
autonomy:
  enabled: true
  restoration_timeout_ms: 20000
  max_llm_calls: 5
  max_wall_time_seconds: 120
  persistence: {{enabled: true, resume_on_start: true}}
  hitl: {{on_user_question: pause_run, on_approval_required: pause_run}}
  validation:
    checks: [{{id: syntax, adapter: builtin.diagnostics, config: {{path: '{0}'}}}}]
  completion: {{all: [{{response_contains: done}}, {{validator_passed: syntax}}]}}
storage: {{type: sqlite, path: '{1}'}}
"#,
                project_root.join(PATH).display(), root.join("tasks.sqlite").display()
            ),
        )
        .unwrap();
        let registered = root.join("language-server");
        std::os::unix::fs::symlink(
            std::env::var("AI_AGENTS_RUST_ANALYZER").unwrap(),
            &registered,
        )
        .unwrap();
        for step in ["seed", mode] {
            if step == "resume-unavailable" {
                fs::remove_file(&registered).unwrap();
            }
            if step == "resume-changed" {
                fs::write(project_root.join(PATH), ORIGINAL).unwrap();
            }
            let before = fs::metadata(project_root.join(PATH))
                .unwrap()
                .modified()
                .unwrap();
            let mut worker = tokio::process::Command::new(std::env::current_exe().unwrap());
            let output = tokio::time::timeout(
                Duration::from_secs(50),
                worker
                    .args(["--exact", "lsp_acceptance::process_worker", "--nocapture"])
                    .env("AI_AGENTS_LSP_PROCESS_MODE", step)
                    .env(
                        "AI_AGENTS_RUST_ANALYZER",
                        if step == "resume-rebound" {
                            root.join("different-server")
                        } else {
                            registered.clone()
                        },
                    )
                    .current_dir(root)
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .unwrap()
            .unwrap();
            assert!(
                output.status.success(),
                "{step}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if step != "seed" {
                assert_eq!(
                    fs::metadata(project_root.join(PATH))
                        .unwrap()
                        .modified()
                        .unwrap(),
                    before
                );
            }
        }
    }
}

async fn wait_until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("controlled server did not reach the expected boundary");
}

fn controlled(root: &Path, control: &Path, mode: &str, revision: Arc<AtomicU64>) -> LspDiagnostics {
    let config = LspDiagnosticsConfig {
        root: root.into(),
        program: std::env::var_os("AI_AGENTS_LSP_TEST_PYTHON")
            .expect("set test Python path")
            .into(),
        args: vec![
            "-c".into(),
            include_str!("server.py").into(),
            mode.into(),
            control.to_str().unwrap().into(),
        ],
        env: BTreeMap::new(),
        language_id: "fixture".into(),
        documents: vec![PATH.into()],
        inputs: vec!["Cargo.toml".into()],
        initialization_options: Value::Null,
        timeout_ms: 10_000,
    };
    let provider = LspDiagnostics::new(config, move || {
        Some(revision.load(Ordering::SeqCst).to_string())
    })
    .unwrap();
    assert!(
        !control.join("pid").exists(),
        "registration started a process"
    );
    provider
}

#[tokio::test]
#[ignore = "requires Python for the controllable stdio peer"]
async fn lsp_stale_partial_crashed_and_cancelled_sessions_never_supply_clean_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    project(root);
    let agent = agent(200, 32_000);
    let revision = Arc::new(AtomicU64::new(1));
    for mode in ["wait", "partial", "crash"] {
        let control = root.join(mode);
        fs::create_dir(&control).unwrap();
        agent.set_diagnostics_provider(Arc::new(controlled(
            root,
            &control,
            mode,
            revision.clone(),
        )));
        let current = agent.clone();
        let file = root.join(PATH);
        let observation =
            tokio::spawn(async move { observe(current, file.to_str().unwrap()).await });
        if mode == "wait" {
            wait_until(|| control.join("collecting").exists()).await;
            // The target path is unchanged, but a related input invalidates the held clean response.
            fs::OpenOptions::new()
                .append(true)
                .open(root.join("Cargo.toml"))
                .unwrap()
                .write_all(b"\n# new revision\n")
                .unwrap();
            revision.fetch_add(1, Ordering::SeqCst);
            fs::write(control.join("release"), "").unwrap();
        }
        let (state, scope, evidence) = observation.await.unwrap();
        let ObservationResult::Tool { record } = &state.completed["observe"].result else {
            panic!("missing diagnostics record")
        };
        assert_eq!(
            serde_json::from_str::<Value>(&record.output).unwrap()["available"],
            false
        );
        assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
    }
    let control = root.join("cancel");
    fs::create_dir(&control).unwrap();
    let provider = controlled(root, &control, "wait", revision);
    assert!(
        !provider
            .diagnostics(DiagnosticsRequest {
                path: Some(PATH.into()),
                ..Default::default()
            })
            .await
            .available
    );
    assert!(!control.join("pid").exists());
    let current = provider.clone();
    let file = root.join(PATH).to_str().unwrap().to_owned();
    let observation = tokio::spawn(async move {
        current
            .diagnostics(DiagnosticsRequest {
                path: Some(file),
                ..Default::default()
            })
            .await
    });
    wait_until(|| control.join("collecting").exists()).await;
    let pid = fs::read_to_string(control.join("pid")).unwrap();
    observation.abort();
    assert!(observation.await.unwrap_err().is_cancelled());
    wait_until(|| !Path::new("/proc").join(&pid).exists()).await;
    fs::write(control.join("release"), "").unwrap();
    assert!(
        provider
            .diagnostics(DiagnosticsRequest {
                path: Some(root.join(PATH).to_str().unwrap().into()),
                ..Default::default()
            })
            .await
            .available
    );
}

#[tokio::test]
#[ignore = "requires Python for the controllable stdio peer"]
async fn tool_policy_and_registered_document_share_one_path() {
    if let Some(other) = std::env::var_os("AI_AGENTS_OTHER_DOCUMENT_ROOT") {
        let registered_root = PathBuf::from(other);
        let cwd = std::env::current_dir().unwrap();
        let provider = controlled(
            &registered_root,
            &registered_root,
            "partial",
            Arc::new(AtomicU64::new(1)),
        );
        let yaml = format!(
            "name: PolicyPath\nsystem_prompt: test\ntools: [diagnostics]\ntool_security:\n  enabled: true\n  tools:\n    diagnostics:\n      read_paths: ['{}']\n",
            cwd.display()
        );
        let runtime = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(Arc::new(MockLLMProvider::new("path")))
                .auto_configure_features()
                .unwrap()
                .autonomy_extensions(AutonomyExtensions::builtins())
                .build()
                .unwrap(),
        );
        runtime.set_diagnostics_provider(Arc::new(provider));
        for (path, executed) in [
            (PATH.to_string(), true),
            (registered_root.join(PATH).display().to_string(), false),
        ] {
            let (state, scope, evidence) = observe(runtime.clone(), &path).await;
            let ObservationResult::Tool { record } = &state.completed["observe"].result else {
                panic!("missing policy record")
            };
            assert_eq!(record.executed, executed, "{}", record.output);
            assert_outcomes(&scope, &evidence, GateOutcome::Unknown);
            assert!(!registered_root.join("pid").exists());
        }
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let allowed = directory.path().join("allowed");
    let registered = directory.path().join("registered");
    for root in [&allowed, &registered] {
        project(root);
    }
    let output = tokio::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "lsp_acceptance::tool_policy_and_registered_document_share_one_path",
            "--ignored",
            "--nocapture",
        ])
        .current_dir(allowed)
        .env("AI_AGENTS_OTHER_DOCUMENT_ROOT", registered)
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
