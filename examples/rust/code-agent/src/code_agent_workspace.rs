// Shared preparation for a disposable workspace and a provider-free task configuration.
// Flow: validate sample and toolchain -> bind canonical paths and policy -> build or verify.
// Use this sample-only host to inspect task wiring, not to run arbitrary user projects.
// The caller supplies a new /tmp workspace and co-located trusted cargo/rustc paths on Linux/WSL.
// The verifier accepts only the reviewed sample sources and does not isolate arbitrary Rust code.

use ai_agents::AgentBuilder;
use ai_agents::context::ContextSource;
use ai_agents::spec::AgentSpec;
use ai_agents::tool_security::CommandRuleConfig;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const AGENT_YAML: &str = include_str!("../../../yaml/code-agent/agent.yaml");
const MANIFEST: &[u8] = include_bytes!("../sample-project/Cargo.toml");
const LOCK: &[u8] = include_bytes!("../sample-project/Cargo.lock");
const INSTRUCTIONS: &[u8] = include_bytes!("../sample-project/AGENTS.md");
const ORIGINAL_SOURCE: &[u8] = include_bytes!("../sample-project/src/lib.rs");
const ACCEPTANCE: &[u8] = include_bytes!("../sample-project/tests/acceptance.rs");
const MAX_SOURCE_BYTES: u64 = 64 * 1024;
const MAX_EVIDENCE_BYTES: usize = 1024 * 1024;
const VERIFICATION_TIMEOUT: Duration = Duration::from_secs(25);
static NEXT_SCRATCH: AtomicU64 = AtomicU64::new(0);

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct PreparedConfiguration {
    spec: AgentSpec,
    root: PathBuf,
    cargo: PathBuf,
    rustc: PathBuf,
}

impl PreparedConfiguration {
    // The caller supplies providers and runners later; this does not replace an existing eval registry.
    pub fn builder(&self) -> AgentBuilder {
        AgentBuilder::from_spec(self.spec.clone())
    }

    // The application configuration and independent fixture check use the same resolved bindings.
    pub fn verify_fixture(&self) -> Result<Verification> {
        verify(&self.root, &self.cargo, &self.rustc)
    }

    // The offline host consumes the same YAML task profile and canonical source binding.
    pub fn task_profile(&self) -> ai_agents::autonomy::AutonomyConfig {
        self.spec.autonomy.clone()
    }

    pub fn source_file(&self) -> PathBuf {
        self.root.join("src/lib.rs")
    }
}

pub struct Verification {
    pub passed: bool,
    pub scratch: PathBuf,
}

// Limit this example to newly created Linux/WSL temporary workspaces rather than user checkouts.
fn temp_root() -> Result<PathBuf> {
    if !cfg!(target_os = "linux") {
        return Err("sample project preparation currently supports Linux/WSL only".into());
    }
    Ok(fs::canonicalize("/tmp")?)
}

fn workspace_root(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err("workspace must be an absolute path".into());
    }
    let root = fs::canonicalize(path)?;
    let temporary = temp_root()?;
    if root == temporary || !root.starts_with(&temporary) || !root.is_dir() {
        return Err("workspace must be a disposable directory under /tmp".into());
    }
    Ok(root)
}

// The host never overwrites a pre-existing path, even if preparing a partial sample fails.
pub fn prepare(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err("new sample path must be absolute".into());
    }
    let parent = fs::canonicalize(path.parent().ok_or("sample has no parent")?)?;
    if !parent.starts_with(temp_root()?) {
        return Err("new sample must be created under /tmp".into());
    }
    let name = path.file_name().ok_or("sample has no directory name")?;
    let root = parent.join(name);
    fs::create_dir(&root)?;
    fs::create_dir(root.join("src"))?;
    fs::create_dir(root.join("tests"))?;
    for (name, bytes) in [
        ("Cargo.toml", MANIFEST),
        ("Cargo.lock", LOCK),
        ("AGENTS.md", INSTRUCTIONS),
        ("src/lib.rs", ORIGINAL_SOURCE),
        ("tests/acceptance.rs", ACCEPTANCE),
    ] {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(name))?;
        file.write_all(bytes)?;
    }
    check_workspace(&root)?;
    Ok(root)
}

// Exact tree and protected bytes detect accidental test, manifest, configuration, and symlink changes.
fn check_directory(path: &Path, expected: &[&str]) -> Result<()> {
    if !fs::symlink_metadata(path)?.file_type().is_dir() {
        return Err(format!("expected a real directory: {}", path.display()).into());
    }
    let mut actual = fs::read_dir(path)?
        .map(|entry| Ok(entry?.file_name()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut expected = expected
        .iter()
        .map(std::ffi::OsString::from)
        .collect::<Vec<_>>();
    actual.sort();
    expected.sort();
    if actual != expected {
        return Err(format!("sample directory entries changed: {}", path.display()).into());
    }
    Ok(())
}

fn read_regular(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > limit {
        return Err(format!("expected a bounded regular file: {}", path.display()).into());
    }
    Ok(fs::read(path)?)
}

// Cargo searches parent directories for config even when a manifest path is explicit.
fn reject_ambient_cargo_config(root: &Path) -> Result<()> {
    for ancestor in root.ancestors() {
        for name in [".cargo/config", ".cargo/config.toml"] {
            match fs::symlink_metadata(ancestor.join(name)) {
                Ok(_) => {
                    return Err(format!(
                        "ambient Cargo configuration is not supported: {}",
                        ancestor.display()
                    )
                    .into());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(())
}

fn check_workspace(path: &Path) -> Result<(PathBuf, Vec<u8>)> {
    let root = workspace_root(path)?;
    check_directory(
        &root,
        &["AGENTS.md", "Cargo.lock", "Cargo.toml", "src", "tests"],
    )?;
    check_directory(&root.join("src"), &["lib.rs"])?;
    check_directory(&root.join("tests"), &["acceptance.rs"])?;
    for (name, expected) in [
        ("Cargo.toml", MANIFEST),
        ("Cargo.lock", LOCK),
        ("AGENTS.md", INSTRUCTIONS),
        ("tests/acceptance.rs", ACCEPTANCE),
    ] {
        if read_regular(&root.join(name), MAX_SOURCE_BYTES)? != expected {
            return Err(format!("protected sample input changed: {name}").into());
        }
    }
    reject_ambient_cargo_config(&root)?;
    let source = read_regular(&root.join("src/lib.rs"), MAX_SOURCE_BYTES)?;
    Ok((root, source))
}

// Require explicitly supplied, co-located toolchain executables instead of inheriting shell paths.
fn toolchain(cargo: &Path, rustc: &Path) -> Result<(PathBuf, PathBuf)> {
    if !cargo.is_absolute() || !rustc.is_absolute() {
        return Err("cargo and rustc must be absolute trusted toolchain paths".into());
    }
    let cargo = fs::canonicalize(cargo)?;
    let rustc = fs::canonicalize(rustc)?;
    if !cargo.is_file()
        || !rustc.is_file()
        || cargo.parent() != rustc.parent()
        || cargo.file_name().and_then(|s| s.to_str()) != Some("cargo")
        || rustc.file_name().and_then(|s| s.to_str()) != Some("rustc")
    {
        return Err(
            "cargo and rustc must be co-located toolchain executables, not rustup proxies".into(),
        );
    }
    Ok((cargo, rustc))
}

// The builtin validator uses a command string, so reject paths its parser cannot represent safely.
pub(crate) fn command_word(path: &Path) -> Result<String> {
    let value = path.to_str().ok_or("command path must be UTF-8")?;
    if value.is_empty()
        || value
            .chars()
            .any(|c| c.is_whitespace() || "'\"\\;&|<>`$".contains(c))
    {
        return Err("command paths with whitespace or shell metacharacters are unsupported".into());
    }
    Ok(value.to_string())
}

// Bind the same explicit source scope and fixed acceptance recipe without constructing a provider.
pub fn prepare_configuration(
    root: &Path,
    host: &Path,
    cargo: &Path,
    rustc: &Path,
) -> Result<PreparedConfiguration> {
    let (root, _) = check_workspace(root)?;
    let host = fs::canonicalize(host)?;
    if !host.is_file() || host.starts_with(&root) {
        return Err("the trusted host executable must be outside the workspace".into());
    }
    let (cargo, rustc) = toolchain(cargo, rustc)?;
    let argv = vec![
        command_word(&host)?,
        "verify".into(),
        command_word(&root)?,
        command_word(&cargo)?,
        command_word(&rustc)?,
    ];
    let command = argv.join(" ");
    let mut spec = AgentSpec::from_yaml_strict(AGENT_YAML)?;
    let root_string = root
        .to_str()
        .ok_or("workspace path must be UTF-8")?
        .to_string();
    let source_string = root
        .join("src/lib.rs")
        .to_str()
        .ok_or("source path must be UTF-8")?
        .to_string();
    spec.tool_security
        .tools
        .get_mut("file_read")
        .ok_or("missing file_read policy")?
        .read_paths = vec![root_string.clone()];
    let edit = spec
        .tool_security
        .tools
        .get_mut("file_edit")
        .ok_or("missing file_edit policy")?;
    edit.read_paths = vec![root_string.clone()];
    edit.write_paths = vec![source_string.clone()];
    let policy = spec
        .tool_security
        .tools
        .get_mut("command")
        .ok_or("missing command policy")?;
    policy.allowed_commands = vec![CommandRuleConfig { argv }];
    policy.working_dirs = vec![root_string.clone()];
    let checks = spec
        .autonomy
        .defaults
        .validation
        .as_mut()
        .and_then(|v| v.checks.as_mut())
        .ok_or("missing acceptance validation")?;
    if checks.len() != 1 || checks[0].id != "acceptance" {
        return Err("unexpected acceptance validation configuration".into());
    }
    checks[0].config = Some(json!({"command": command, "cwd": root_string}));
    for (key, value) in [
        (
            "instruction_snapshot",
            String::from_utf8(INSTRUCTIONS.to_vec())?,
        ),
        ("workspace_root", root_string),
        ("source_file", source_string),
    ] {
        match spec.context.get_mut(key) {
            Some(ContextSource::Runtime { default, .. }) => {
                *default = Some(serde_json::Value::String(value));
            }
            _ => return Err(format!("missing runtime context: {key}").into()),
        }
    }
    spec.validate()?;
    Ok(PreparedConfiguration {
        spec,
        root,
        cargo,
        rustc,
    })
}

// Scratch is intentionally retained, including on timeout, because process termination is not proof of descendant cleanup.
fn create_scratch(root: &Path) -> Result<PathBuf> {
    let parent = root.parent().ok_or("workspace has no parent")?;
    for _ in 0..100 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let counter = NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed);
        let scratch = parent.join(format!(
            "code-agent-verify-{}-{now}-{counter}",
            std::process::id()
        ));
        match fs::create_dir(&scratch) {
            Ok(()) => return Ok(scratch),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err("could not allocate a unique verifier directory".into())
}

// The preparation verifier accepts only two reviewed fixture inputs, not arbitrary agent-authored code.
fn reviewed_source(source: &[u8]) -> Result<bool> {
    let original = std::str::from_utf8(ORIGINAL_SOURCE)?;
    let before = "value.max(lower).max(upper)";
    if original.matches(before).count() != 1 {
        return Err("sample reference edit is ambiguous".into());
    }
    let fixed = original.replace(before, "value.max(lower).min(upper)");
    Ok(source == ORIGINAL_SOURCE || source == fixed.as_bytes())
}

struct CapturedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

// Drain both pipes while retaining bounded evidence; detached readers never write unbounded files.
fn capture<R: Read + Send + 'static>(
    mut reader: R,
) -> mpsc::Receiver<std::io::Result<CapturedOutput>> {
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = (|| {
            let mut captured = CapturedOutput {
                bytes: Vec::new(),
                truncated: false,
            };
            let mut buffer = [0_u8; 8192];
            loop {
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                let available = MAX_EVIDENCE_BYTES.saturating_sub(captured.bytes.len());
                captured
                    .bytes
                    .extend_from_slice(&buffer[..count.min(available)]);
                captured.truncated |= count > available;
            }
            Ok(captured)
        })();
        let _ = sender.send(result);
    });
    receiver
}

// Verify reviewed cases with an isolated Cargo home and target; this is not an OS sandbox for compiled code.
pub fn verify(root: &Path, cargo: &Path, rustc: &Path) -> Result<Verification> {
    let (root, source_before) = check_workspace(root)?;
    if !reviewed_source(&source_before)? {
        return Err("source is not one of the reviewed preparation inputs; arbitrary agent edits are unsupported".into());
    }
    let (cargo, rustc) = toolchain(cargo, rustc)?;
    let scratch = create_scratch(&root)?;
    let cargo_home = scratch.join("cargo-home");
    fs::create_dir(&cargo_home)?;
    let mut child = Command::new(cargo)
        .args(["test", "--offline", "--locked", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["--test", "acceptance", "--", "--test-threads=1"])
        .current_dir(&root)
        .env_clear()
        .env("HOME", &scratch)
        .env("CARGO_HOME", &cargo_home)
        .env("CARGO_TARGET_DIR", scratch.join("target"))
        .env("RUSTC", rustc)
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = capture(child.stdout.take().ok_or("missing stdout pipe")?);
    let stderr = capture(child.stderr.take().ok_or("missing stderr pipe")?);
    let deadline = Instant::now() + VERIFICATION_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "acceptance timed out; cleanup is unconfirmed and evidence remains at {}",
                scratch.display()
            )
            .into());
        }
        thread::sleep(Duration::from_millis(25));
    };
    let remaining = || deadline.saturating_duration_since(Instant::now());
    let stdout = stdout.recv_timeout(remaining()).map_err(|_| {
        format!(
            "stdout cleanup is unconfirmed; evidence remains at {}",
            scratch.display()
        )
    })??;
    let stderr = stderr.recv_timeout(remaining()).map_err(|_| {
        format!(
            "stderr cleanup is unconfirmed; evidence remains at {}",
            scratch.display()
        )
    })??;
    fs::write(scratch.join("stdout.txt"), &stdout.bytes)?;
    fs::write(scratch.join("stderr.txt"), &stderr.bytes)?;
    let (_, source_after) = check_workspace(&root)?;
    if source_before != source_after {
        return Err(format!(
            "source changed during verification; evidence remains at {}",
            scratch.display()
        )
        .into());
    }
    let output = String::from_utf8(stdout.bytes)?;
    let expected = [
        "running 5 tests",
        "test clamps_above_upper_bound ... ok",
        "test clamps_below_lower_bound ... ok",
        "test preserves_value_inside_bounds ... ok",
        "test uses_supplied_negative_bounds ... ok",
        "test uses_supplied_positive_bounds ... ok",
        "test result: ok. 5 passed; 0 failed; 0 ignored",
    ];
    let passed = status.success()
        && !stdout.truncated
        && !stderr.truncated
        && expected.iter().all(|marker| output.contains(marker));
    Ok(Verification { passed, scratch })
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use ai_agents::tool_security::{SecurityCheckResult, ToolSecurityEngine};

    fn fixture() -> PathBuf {
        let root = temp_root().unwrap().join(format!(
            "code-agent-setup-test-{}-{}",
            std::process::id(),
            NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed)
        ));
        prepare(&root).unwrap()
    }

    fn executables() -> (PathBuf, PathBuf, PathBuf) {
        let cargo = fs::canonicalize(env!("CARGO")).unwrap();
        let rustc = cargo.parent().unwrap().join("rustc");
        (std::env::current_exe().unwrap(), cargo, rustc)
    }

    #[test]
    fn configuration_is_host_bound_without_provider_construction() {
        let root = fixture();
        let (host, cargo, rustc) = executables();
        let prepared = prepare_configuration(&root, &host, &cargo, &rustc).unwrap();
        assert_eq!(
            prepared.spec.tool_security.tools["file_edit"].write_paths,
            vec![root.join("src/lib.rs").to_str().unwrap()]
        );
        assert!(prepared.spec.tool_security.tools["file_edit"].require_read_before_write);
        assert_eq!(
            prepared.spec.tool_security.tools["command"].working_dirs,
            vec![root.to_str().unwrap()]
        );
        assert!(
            prepared.spec.tool_security.tools["command"]
                .env_passthrough
                .is_empty()
        );
        assert!(
            matches!(&prepared.spec.context["instruction_snapshot"], ContextSource::Runtime { default: Some(value), .. } if value.as_str() == std::str::from_utf8(INSTRUCTIONS).ok())
        );
        prepared.builder().auto_configure_features().unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn instruction_snapshot_and_paths_reach_the_actual_mock_provider_request() {
        use ai_agents::Agent;
        use ai_agents::llm::{
            ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider,
            LLMResponse,
        };
        use std::sync::{Arc, Mutex};

        struct Provider(Mutex<Vec<ChatMessage>>);
        #[async_trait::async_trait]
        impl LLMProvider for Provider {
            async fn complete(
                &self,
                messages: &[ChatMessage],
                _: Option<&LLMConfig>,
            ) -> std::result::Result<LLMResponse, LLMError> {
                *self.0.lock().unwrap() = messages.to_vec();
                Ok(LLMResponse::new("fixture response", FinishReason::Stop))
            }
            async fn complete_stream(
                &self,
                _: &[ChatMessage],
                _: Option<&LLMConfig>,
            ) -> std::result::Result<
                Box<
                    dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>>
                        + Unpin
                        + Send,
                >,
                LLMError,
            > {
                Err(LLMError::Config("unexpected stream".into()))
            }
            fn provider_name(&self) -> &str {
                "offline-fixture"
            }
            fn supports(&self, _: LLMFeature) -> bool {
                false
            }
        }
        let root = fixture();
        let (host, cargo, rustc) = executables();
        let prepared = prepare_configuration(&root, &host, &cargo, &rustc).unwrap();
        let provider = Arc::new(Provider(Mutex::new(Vec::new())));
        let agent = prepared
            .builder()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap();
        agent.chat("Inspect the fixture.").await.unwrap();
        let messages = provider.0.lock().unwrap();
        let prompt = messages
            .iter()
            .find(|message| message.role == ai_agents::llm::Role::System)
            .unwrap();
        assert!(prompt.content.contains(root.to_str().unwrap()));
        assert!(
            prompt
                .content
                .contains(root.join("src/lib.rs").to_str().unwrap())
        );
        assert!(
            prompt
                .content
                .contains(std::str::from_utf8(INSTRUCTIONS).unwrap())
        );
        drop(messages);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn effective_policy_rejects_protected_and_outside_edits() {
        let root = fixture();
        let (host, cargo, rustc) = executables();
        let prepared = prepare_configuration(&root, &host, &cargo, &rustc).unwrap();
        let engine = ToolSecurityEngine::try_new(prepared.spec.tool_security).unwrap();
        for path in [
            root.join("Cargo.toml"),
            root.join("tests/acceptance.rs"),
            root.join("AGENTS.md"),
            root.parent().unwrap().join("outside.rs"),
        ] {
            let result = engine
                .check_tool_execution(
                    "file_edit",
                    &json!({"path":path,"old_text":"a","new_text":"b","dry_run":false}),
                )
                .await
                .unwrap();
            assert!(
                matches!(result, SecurityCheckResult::Block { .. }),
                "unexpected policy result: {result:?}"
            );
        }
        let allowed = engine.check_tool_execution("file_edit", &json!({"path":root.join("src/lib.rs"),"old_text":"a","new_text":"b","dry_run":false})).await.unwrap();
        assert!(allowed.is_allowed());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn fixed_command_and_cwd_are_checked_by_the_security_engine() {
        let root = fixture();
        let (host, cargo, rustc) = executables();
        let prepared = prepare_configuration(&root, &host, &cargo, &rustc).unwrap();
        let config = prepared
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
            .unwrap();
        let command = config["command"].as_str().unwrap();
        let engine = ToolSecurityEngine::try_new(prepared.spec.tool_security.clone()).unwrap();
        let allowed = engine
            .check_tool_execution("command", &json!({"command":command,"cwd":root}))
            .await
            .unwrap();
        assert!(
            allowed.is_allowed(),
            "unexpected policy result: {allowed:?}"
        );
        let wrong = engine
            .check_tool_execution("command", &json!({"command":"cargo test","cwd":root}))
            .await
            .unwrap();
        assert!(matches!(wrong, SecurityCheckResult::Block { .. }));
        let wrong_cwd = engine
            .check_tool_execution("command", &json!({"command":command,"cwd":"/tmp"}))
            .await
            .unwrap();
        assert!(matches!(wrong_cwd, SecurityCheckResult::Block { .. }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn protected_changes_and_extra_entries_are_rejected() {
        let root = fixture();
        assert!(prepare(&root).is_err());
        assert_eq!(fs::read(root.join("src/lib.rs")).unwrap(), ORIGINAL_SOURCE);
        fs::write(root.join("tests/acceptance.rs"), "// no tests\n").unwrap();
        assert!(check_workspace(&root).is_err());
        fs::write(root.join("tests/acceptance.rs"), ACCEPTANCE).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        assert!(check_workspace(&root).is_err());
        fs::write(root.join("Cargo.toml"), MANIFEST).unwrap();
        fs::write(root.join("build.rs"), "fn main() {}\n").unwrap();
        assert!(check_workspace(&root).is_err());
        fs::remove_file(root.join("build.rs")).unwrap();
        check_workspace(&root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unreviewed_source_cannot_forge_a_preparation_pass() {
        let root = fixture();
        let (_, cargo, rustc) = executables();
        fs::write(root.join("src/lib.rs"), "pub fn clamp(_: i32, _: i32, _: i32) -> i32 { println!(\"test result: ok. 5 passed; 0 failed; 0 ignored\"); std::process::exit(0) }\n").unwrap();
        let error = match verify(&root, &cargo, &rustc) {
            Ok(_) => panic!("unreviewed source was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("reviewed preparation inputs"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn output_capture_is_bounded_and_detects_truncation() {
        let input = std::io::Cursor::new(vec![b'x'; MAX_EVIDENCE_BYTES + 1]);
        let output = capture(input).recv().unwrap().unwrap();
        assert_eq!(output.bytes.len(), MAX_EVIDENCE_BYTES);
        assert!(output.truncated);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_source_is_rejected() {
        let root = fixture();
        fs::remove_file(root.join("src/lib.rs")).unwrap();
        std::os::unix::fs::symlink(root.join("tests/acceptance.rs"), root.join("src/lib.rs"))
            .unwrap();
        assert!(check_workspace(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn actual_fixture_fails_then_reference_fix_passes_the_protected_cases() {
        let root = fixture();
        let (host, cargo, rustc) = executables();
        let prepared = prepare_configuration(&root, &host, &cargo, &rustc).unwrap();
        let initial = prepared.verify_fixture().unwrap();
        assert!(!initial.passed);
        let initial_output = fs::read_to_string(initial.scratch.join("stdout.txt")).unwrap();
        assert!(initial_output.contains("test preserves_value_inside_bounds ... FAILED"));
        let original = String::from_utf8(ORIGINAL_SOURCE.to_vec()).unwrap();
        let corrected =
            original.replace("value.max(lower).max(upper)", "value.max(lower).min(upper)");
        assert_ne!(original, corrected);
        fs::write(root.join("src/lib.rs"), corrected).unwrap();
        let fixed = prepared.verify_fixture().unwrap();
        assert!(fixed.passed);
        fs::remove_dir_all(initial.scratch).unwrap();
        fs::remove_dir_all(fixed.scratch).unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
