// Bind explicitly selected workspace files to an actual provider and a host-owned checker.
// Flow: validate explicit paths -> prepare a provider-free builder -> start a task with typed approvals.
// The checker is trusted host code; the caller must provide any required isolation and independent tests.

use crate::code_agent_workspace::command_word;
use crate::workspace_guard::WorkspaceGuard;
use crate::workspace_report::WorkspaceReport;
use ai_agents::AgentBuilder;
use ai_agents::autonomy::{AutonomyExtensions, HostRevisionSource, HostValidationBinding};
use ai_agents::context::ContextSource;
use ai_agents::spec::{AgentSpec, LLMConfigOrSelector, PreparedAgentSpec, TaskIterationPolicy};
use ai_agents::tool_security::CommandRuleConfig;
use ai_agents::tools::{
    CommandRequest, CommandResponse, CommandRunner, ProcessCommandRunner, ToolExecutionContext,
};
use ai_agents_cli::{CliApprovalHandler, CliRepl, CommandResult};
use async_trait::async_trait;
use serde_json::json;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

const YAML: &str = include_str!("../../../yaml/code-agent/workspace.yaml");
const READ_ONLY_YAML: &str = include_str!("../../../yaml/code-agent/workspace-read-only.yaml");
const CHECKER_LIMIT: u64 = 8 * 1024 * 1024;
const INSTRUCTIONS_LIMIT: u64 = 64 * 1024;

#[cfg(all(test, target_os = "linux"))]
#[path = "workspace_agent_tests.rs"]
mod tests;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct WorkspaceSetup {
    spec: AgentSpec,
    task_iteration_policy: TaskIterationPolicy,
    root: PathBuf,
    source: PathBuf,
    sources: Vec<PathBuf>,
    new_files: Vec<PathBuf>,
    guard: Option<Arc<WorkspaceGuard>>,
    cooperative_revision: bool,
    discovery_enabled: bool,
    checker: Option<CheckerBinding>,
}

struct CheckerBinding {
    path: PathBuf,
    bytes: Vec<u8>,
    argv: Vec<String>,
    command: String,
    timeout_ms: Option<u64>,
}

// Resolve each workspace component without following a symlink or allowing parent traversal.
fn selected_file(root: &Path, relative: &Path) -> Result<PathBuf> {
    if relative.as_os_str().is_empty() || relative.is_absolute() {
        return Err("selected files must use nonempty paths relative to the workspace".into());
    }
    let mut path = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return Err("selected file paths must contain only normal components".into());
        };
        path.push(part);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err("symlinked workspace paths are unsupported".into());
        }
    }
    if !fs::symlink_metadata(&path)?.file_type().is_file() {
        return Err("selected path must be a regular file".into());
    }
    Ok(path)
}

// A proposed new file must have existing non-symlinked parents and must not already exist at setup.
fn selected_new_file(root: &Path, relative: &Path) -> Result<PathBuf> {
    if relative.as_os_str().is_empty() || relative.is_absolute() {
        return Err("new files must use nonempty paths relative to the workspace".into());
    }
    let components: Vec<_> = relative.components().collect();
    let mut path = root.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err("new file paths must contain only normal components".into());
        };
        path.push(part);
        if index + 1 == components.len() {
            match fs::symlink_metadata(&path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
                Ok(_) => return Err("new file path already exists".into()),
            }
        } else {
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
                return Err("new file parent must be an existing non-symlinked directory".into());
            }
        }
    }
    Ok(path)
}

// Checker bytes are retained so a replaced host program is refused before execution.
fn checker_snapshot(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > CHECKER_LIMIT {
        return Err("checker must be a regular file no larger than 8 MiB".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err("checker must be executable".into());
        }
    }
    let bytes = fs::read(path)?;
    if bytes.len() as u64 > CHECKER_LIMIT {
        return Err("checker exceeds the size limit".into());
    }
    Ok(bytes)
}

// Only explicitly selected instruction bytes are shared with the provider; no workspace discovery runs.
fn instructions(root: &Path, relative: Option<&Path>) -> Result<String> {
    let Some(relative) = relative else {
        return Ok("No project instruction file was supplied by the host.".into());
    };
    let path = selected_file(root, relative)?;
    let metadata = fs::metadata(&path)?;
    if metadata.len() > INSTRUCTIONS_LIMIT {
        return Err("instruction snapshot exceeds 64 KiB".into());
    }
    let bytes = fs::read(path)?;
    if bytes.len() as u64 > INSTRUCTIONS_LIMIT {
        return Err("instruction snapshot exceeds 64 KiB".into());
    }
    Ok(String::from_utf8(bytes)?)
}

// Preserve the original single-file entry while using the same exact multi-file authority checks.
pub fn prepare(
    root: &Path,
    source: &Path,
    checker: &Path,
    provider: &str,
    model: &str,
    instruction_file: Option<&Path>,
) -> Result<WorkspaceSetup> {
    prepare_files(
        root,
        &[source.to_path_buf()],
        checker,
        provider,
        model,
        instruction_file,
    )
}

// Every selected editable source is explicitly host-authorized and the checker remains a fixed host operation.
pub fn prepare_files(
    root: &Path,
    sources: &[PathBuf],
    checker: &Path,
    provider: &str,
    model: &str,
    instruction_file: Option<&Path>,
) -> Result<WorkspaceSetup> {
    prepare_files_with_new(
        root,
        sources,
        &[],
        checker,
        provider,
        model,
        instruction_file,
    )
}

// New paths can be read and edited only after this host has acknowledged their actual creation.
pub fn prepare_files_with_new(
    root: &Path,
    sources: &[PathBuf],
    new_files: &[PathBuf],
    checker: &Path,
    provider: &str,
    model: &str,
    instruction_file: Option<&Path>,
) -> Result<WorkspaceSetup> {
    prepare_workspace(
        root,
        sources,
        new_files,
        Some(checker),
        provider,
        model,
        instruction_file,
    )
}

// A separate declaration removes edit and checker authority rather than inferring safety from the objective.
pub fn prepare_read_only(
    root: &Path,
    sources: &[PathBuf],
    provider: &str,
    model: &str,
    instruction_file: Option<&Path>,
) -> Result<WorkspaceSetup> {
    prepare_workspace(root, sources, &[], None, provider, model, instruction_file)
}

// Resolve both declarations through the same path and provider checks without giving the read-only mode a checker.
fn prepare_workspace(
    root: &Path,
    sources: &[PathBuf],
    new_files: &[PathBuf],
    checker: Option<&Path>,
    provider: &str,
    model: &str,
    instruction_file: Option<&Path>,
) -> Result<WorkspaceSetup> {
    if !root.is_absolute() || checker.is_some_and(|path| !path.is_absolute()) {
        return Err("workspace and checker must be absolute paths".into());
    }
    let root = fs::canonicalize(root)?;
    if !root.is_dir() {
        return Err("workspace must be a directory".into());
    }
    if sources.is_empty() {
        return Err("at least one source file must be selected".into());
    }
    if !new_files.is_empty() && !cfg!(unix) {
        return Err("new-file ownership is supported only on Unix".into());
    }
    let mut selected = Vec::with_capacity(sources.len());
    for source in sources {
        let source = selected_file(&root, source)?;
        if selected.contains(&source) {
            return Err("source files must not be selected more than once".into());
        }
        // Atomic replacement preserves ordinary Unix bits, not special privilege bits or extended metadata.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if checker.is_some() && fs::metadata(&source)?.permissions().mode() & 0o7000 != 0 {
                return Err("source files with special permission bits are unsupported".into());
            }
        }
        selected.push(source);
    }
    let source = selected[0].clone();
    let existing_count = selected.len();
    let mut selected_new = Vec::with_capacity(new_files.len());
    for relative in new_files {
        let path = selected_new_file(&root, relative)?;
        if selected.contains(&path) || selected_new.contains(&path) {
            return Err("source files must not be selected more than once".into());
        }
        selected_new.push(path);
    }
    selected.extend(selected_new.iter().cloned());
    let checker = checker
        .map(|path| {
            let path = fs::canonicalize(path)?;
            if path.starts_with(&root) {
                return Err::<_, Box<dyn std::error::Error>>(
                    "the trusted checker must be outside the workspace".into(),
                );
            }
            let bytes = checker_snapshot(&path)?;
            Ok((path, bytes))
        })
        .transpose()?;
    if provider.trim() != provider
        || provider.is_empty()
        || model.trim() != model
        || model.is_empty()
        || model.len() > 256
    {
        return Err("provider and model must be nonempty names".into());
    }
    let provider_type = ai_agents::ProviderType::from_str(provider)?;
    if provider_type == ai_agents::ProviderType::OpenAICompatible {
        return Err(
            "openai-compatible requires a base URL, which this example does not configure".into(),
        );
    }
    let snapshot = instructions(&root, instruction_file)?;
    let root_string = command_word(&root)?;
    let source_strings = selected
        .iter()
        .map(|source| command_word(source))
        .collect::<Result<Vec<_>>>()?;
    let source_string = source_strings[0].clone();
    let existing_strings = source_strings[..existing_count].to_vec();
    let new_strings = source_strings[existing_count..].to_vec();
    let checker = checker
        .map(|(path, bytes)| {
            let mut argv = vec![command_word(&path)?, root_string.clone()];
            argv.extend(source_strings.iter().cloned());
            let command = argv.join(" ");
            Ok::<_, Box<dyn std::error::Error>>(CheckerBinding {
                path,
                bytes,
                argv,
                command,
                timeout_ms: None,
            })
        })
        .transpose()?;
    let yaml = if checker.is_some() {
        YAML
    } else {
        READ_ONLY_YAML
    };
    let (mut spec, task_iteration_policy) = PreparedAgentSpec::from_yaml_strict(yaml)?.into_parts();
    match &mut spec.llm {
        LLMConfigOrSelector::Config(config) => {
            config.provider = provider.to_string();
            config.model = model.to_string();
        }
        _ => return Err("workspace example needs a direct provider configuration".into()),
    }
    spec.tool_security
        .tools
        .get_mut("file_read")
        .ok_or("missing file_read policy")?
        .read_paths = source_strings.clone();
    if let Some(checker) = &checker {
        let edit = spec
            .tool_security
            .tools
            .get_mut("file_edit")
            .ok_or("missing file_edit policy")?;
        edit.read_paths = source_strings.clone();
        edit.write_paths = source_strings.clone();
        if !selected_new.is_empty() {
            let create_paths = selected_new
                .iter()
                .map(|path| command_word(path))
                .collect::<Result<Vec<_>>>()?;
            spec.tools
                .as_mut()
                .ok_or("missing tool grants")?
                .push(serde_json::from_value(json!("file_write"))?);
            spec.tool_security.tools.insert(
                "file_write".into(),
                serde_json::from_value(json!({
                    "write_paths": create_paths,
                    "overwrite_existing": false,
                    "create_parent_dirs": false,
                    "require_confirmation": true
                }))?,
            );
        }
        match spec.context.get_mut("create_files") {
            Some(ContextSource::Runtime { default, .. }) => {
                *default = Some(json!(serde_json::to_string(&new_strings)?));
            }
            _ => return Err("missing create_files context".into()),
        }
        let policy = spec
            .tool_security
            .tools
            .get_mut("command")
            .ok_or("missing command policy")?;
        policy.allowed_commands = vec![CommandRuleConfig {
            argv: checker.argv.clone(),
        }];
        policy.working_dirs = vec![root_string.clone()];
        let checks = spec
            .autonomy
            .defaults
            .validation
            .as_mut()
            .and_then(|v| v.checks.as_mut())
            .ok_or("missing validation check")?;
        if checks.len() != 1 || checks[0].id != "acceptance" {
            return Err("unexpected validation configuration".into());
        }
        checks[0].config = Some(
            json!({"command":checker.command,"cwd":root_string,"host_binding":"workspace-checker"}),
        );
    }
    for (key, value) in [
        ("workspace_root", root_string),
        ("source_file", source_string),
        ("source_files", serde_json::to_string(&existing_strings)?),
        ("discovery_scope", "disabled".into()),
        ("instruction_snapshot", snapshot),
    ] {
        match spec.context.get_mut(key) {
            Some(ContextSource::Runtime { default, .. }) => {
                *default = Some(serde_json::Value::String(value))
            }
            _ => return Err(format!("missing runtime context: {key}").into()),
        }
    }
    spec.validate()?;
    let guard = (!selected_new.is_empty())
        .then(|| WorkspaceGuard::new(root.clone(), selected.clone(), selected_new.clone()));
    Ok(WorkspaceSetup {
        spec,
        task_iteration_policy,
        root,
        source,
        sources: selected,
        new_files: selected_new,
        guard,
        cooperative_revision: false,
        discovery_enabled: false,
        checker,
    })
}

impl WorkspaceSetup {
    // The caller is responsible for synchronizing every external checker input for the whole session.
    pub fn assume_exclusive_workspace(&mut self) -> Result<()> {
        if self.checker.is_none() {
            return Err("read-only mode has no validation target".into());
        }
        if self.guard.is_none() {
            self.guard = Some(WorkspaceGuard::new(
                self.root.clone(),
                self.sources.clone(),
                self.new_files.clone(),
            ));
        }
        self.cooperative_revision = true;
        Ok(())
    }

    // Explicit opt-in permits path discovery and content search under the root, but never widens file_read or write grants.
    pub fn enable_discovery(&mut self) -> Result<()> {
        let root = command_word(&self.root)?;
        let mut spec = self.spec.clone();
        let tools = spec.tools.as_mut().ok_or("missing tool grants")?;
        for name in ["file_list", "glob", "grep"] {
            if tools.iter().any(|tool| tool.name() == name) {
                continue;
            }
            tools.push(serde_json::from_value(json!(name))?);
            spec.tool_security.tools.insert(
                name.into(),
                serde_json::from_value(
                    json!({"read_paths":[root],"allow_without_confirmation":true}),
                )?,
            );
        }
        match spec.context.get_mut("discovery_scope") {
            Some(ContextSource::Runtime { default, .. }) => {
                *default = Some(json!(
                    "enabled for path listing and content search under the workspace root"
                ))
            }
            _ => return Err("missing discovery scope context".into()),
        }
        spec.validate()?;
        self.spec = spec;
        self.discovery_enabled = true;
        Ok(())
    }

    // The host must choose a positive output limit supported by its selected provider and model.
    pub fn set_max_output_tokens(&mut self, tokens: u32) -> Result<()> {
        if tokens == 0 {
            return Err("maximum output tokens must be positive".into());
        }
        match &mut self.spec.llm {
            LLMConfigOrSelector::Config(config) => config.max_tokens = tokens,
            _ => return Err("workspace example needs a direct provider configuration".into()),
        }
        self.spec.validate()?;
        Ok(())
    }

    // Keep the validation request, policy and fixed approval bound to the same host-selected timeout.
    pub fn set_checker_timeout_ms(&mut self, timeout: u64) -> Result<()> {
        if self.checker.is_none() {
            return Err("read-only mode has no checker".into());
        }
        if timeout == 0 || timeout > ai_agents::MAX_TOOL_TIMEOUT_MS {
            return Err(
                "checker timeout must be positive and within the tool timeout limit".into(),
            );
        }
        let policy = self
            .spec
            .tool_security
            .tools
            .get_mut("command")
            .ok_or("missing command policy")?;
        policy.timeout_ms = Some(timeout);
        let checks = self
            .spec
            .autonomy
            .defaults
            .validation
            .as_mut()
            .and_then(|validation| validation.checks.as_mut())
            .ok_or("missing validation check")?;
        let config = checks
            .first_mut()
            .and_then(|check| check.config.as_mut())
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("missing validation configuration")?;
        config.insert("timeout_ms".into(), json!(timeout));
        self.checker
            .as_mut()
            .expect("checker verified above")
            .timeout_ms = Some(timeout);
        self.spec.validate()?;
        Ok(())
    }

    pub fn checker_timeout_ms(&self) -> Option<u64> {
        self.checker.as_ref().map(|_| {
            self.spec
                .tool_security
                .tools
                .get("command")
                .and_then(|policy| policy.timeout_ms)
                .unwrap_or(self.spec.tool_security.default_timeout_ms)
        })
    }

    pub fn max_output_tokens(&self) -> u32 {
        match &self.spec.llm {
            LLMConfigOrSelector::Config(config) => config.max_tokens,
            _ => unreachable!("workspace setup requires a direct provider configuration"),
        }
    }

    pub fn builder(&self) -> Result<AgentBuilder> {
        let builder = AgentBuilder::from_prepared_spec(PreparedAgentSpec::new(
            self.spec.clone(),
            self.task_iteration_policy.clone(),
        ));
        Ok(match &self.guard {
            Some(guard) => builder.tools(guard.registry()?),
            None => builder,
        })
    }

    // A fixed host-only validation binding cannot become an ordinary model-callable command grant.
    pub fn extensions(&self) -> ai_agents::Result<AutonomyExtensions> {
        let target = if self.cooperative_revision {
            self.guard
                .as_ref()
                .map(|guard| ("workspace", guard.clone() as Arc<dyn HostRevisionSource>))
        } else {
            None
        };
        self.extensions_with_target(target)
    }

    // A cooperating embedding host may bind its own authoritative revision; the CLI does not invent one for ambient files.
    pub fn extensions_with_target(
        &self,
        target: Option<(&str, Arc<dyn HostRevisionSource>)>,
    ) -> ai_agents::Result<AutonomyExtensions> {
        let mut extensions = AutonomyExtensions::builtins();
        if let Some(checker) = &self.checker {
            let mut arguments = json!({"command":checker.command,"cwd":self.root});
            if let Some(timeout) = checker.timeout_ms {
                arguments["timeout_ms"] = json!(timeout);
            }
            extensions.register_host_validation(HostValidationBinding {
                id: "workspace-checker".into(),
                validator: "builtin.command".into(),
                version: 1,
                profile: None,
                scope: "task".into(),
                tool: "command".into(),
                arguments,
                require_approval: true,
            })?;
            if let Some((name, source)) = target {
                extensions.register_validation_target("acceptance", name, source)?;
            }
        } else if target.is_some() {
            return Err(ai_agents::AgentError::Config(
                "read-only mode has no validation target".into(),
            ));
        }
        Ok(extensions)
    }

    pub fn checker_runner(&self) -> Option<Arc<dyn CommandRunner>> {
        self.checker.as_ref().map(|checker| {
            Arc::new(CheckerRunner {
                checker: checker.path.clone(),
                expected: checker.bytes.clone(),
                argv: checker.argv.clone(),
                cwd: self.root.to_string_lossy().into_owned(),
                inner: ProcessCommandRunner,
            }) as Arc<dyn CommandRunner>
        })
    }

    pub fn source_file(&self) -> &Path {
        &self.source
    }

    pub fn source_files(&self) -> &[PathBuf] {
        &self.sources
    }

    pub fn discovery_enabled(&self) -> bool {
        self.discovery_enabled
    }

    pub fn cooperative_revision_enabled(&self) -> bool {
        self.cooperative_revision
    }
}

struct CheckerRunner {
    checker: PathBuf,
    expected: Vec<u8>,
    argv: Vec<String>,
    cwd: String,
    inner: ProcessCommandRunner,
}

// Recheck the selected host program before dispatch; the host must still prevent concurrent path replacement.
#[async_trait]
impl CommandRunner for CheckerRunner {
    fn is_available(&self) -> bool {
        checker_snapshot(&self.checker).is_ok_and(|bytes| bytes == self.expected)
    }

    async fn run_command(
        &self,
        request: CommandRequest,
        ctx: ToolExecutionContext,
    ) -> CommandResponse {
        if !self.is_available()
            || request.argv != self.argv
            || request.cwd.as_deref() != Some(&self.cwd)
            || !request.env.is_empty()
        {
            return CommandResponse {
                success: false,
                termination: "unavailable".into(),
                stderr: "trusted checker binding is unavailable or changed".into(),
                cwd: request.cwd,
                ..Default::default()
            };
        }
        self.inner.run_command(request, ctx).await
    }
}

// The existing CLI owns foreground controls; ordinary chat is blocked to avoid unverified edit claims.
pub async fn run(setup: WorkspaceSetup) -> Result<()> {
    println!("Primary source: {}", setup.source_file().display());
    for source in setup.source_files() {
        println!("Authorized source: {}", source.display());
    }
    println!("Workspace discovery enabled: {}", setup.discovery_enabled());
    println!("Maximum model output tokens: {}", setup.max_output_tokens());
    if let Some(timeout) = setup.checker_timeout_ms() {
        println!("Checker timeout: {timeout} ms");
    } else {
        println!("Read-only mode: no edit grant or checker is installed.");
    }
    println!(
        "Selected sources, instruction snapshot and task messages may be sent to the chosen provider and may incur costs."
    );
    if setup.checker.is_some() {
        println!(
            "The external checker runs with host permissions; you are responsible for its independence, isolation and cleanup."
        );
        println!(
            "Per-task workspace observations do not certify external revisions or attribute every change."
        );
        println!("Eligible failed-check output may also be sent to the chosen provider.");
        println!("Only /task commands are enabled; command execution and edits require approval.");
        println!(
            "An eligible failed checker can send bounded untrusted output to the provider; rejected or unavailable checks require guidance."
        );
    } else {
        println!(
            "Only /task commands are enabled; this read-only task cannot edit files or run a checker."
        );
    }
    if setup.discovery_enabled() {
        println!(
            "Workspace discovery may share file names and search matches outside the selected edit files."
        );
    }
    if setup.checker.is_some() {
        println!(
            "A feedback-bearing provider request is one-shot and does not use automatic retries or fallback."
        );
        println!(
            "Atomic edits preserve ordinary Unix mode bits, not ownership, ACLs or extended attributes."
        );
    }
    if setup.cooperative_revision {
        println!(
            "Cooperative revision: the caller must prevent external writes and keep checker inputs and dependencies immutable throughout this session; this host does not enforce those conditions."
        );
    }
    let agent = setup
        .builder()?
        .auto_configure_llms()?
        .auto_configure_features()?
        .autonomy_extensions(setup.extensions()?)
        .approval_handler(CliApprovalHandler::from_metadata(None))
        .task_interaction_delivery(true)
        .build_task_host()
        .await?;
    workspace_repl(&setup, agent).run().await?;
    Ok(())
}

// Both provider-backed and deterministic hosts use the same foreground driver and task observation boundary.
fn workspace_repl(setup: &WorkspaceSetup, agent: ai_agents::RuntimeAgent) -> CliRepl {
    if let Some(runner) = setup.checker_runner() {
        agent.set_command_runner(runner);
    }
    let observer = Arc::new(WorkspaceReport::new(
        setup.root.clone(),
        setup.sources.clone(),
    ));
    CliRepl::new(agent)
        .task_observer(observer)
        .welcome("Code Agent: explicitly selected workspace files")
        .hint("Start with /task run <objective>; approvals and cancellation stay in this session.")
        .on_command(|input, _| {
            let allowed = input.starts_with("/task run ")
                || input.starts_with("/task resume ")
                || input.starts_with("/task cancel ")
                || input.starts_with("/task stop ")
                || input.starts_with("/task show ")
                || input == "/task list"
                || input == "/help"
                || input == "?"
                || input == "/quit"
                || input == "/exit";
            if allowed {
                CommandResult::NotHandled
            } else {
                eprintln!("Use /task run <objective>, a task control, or /quit; ordinary chat is disabled.");
                CommandResult::Handled
            }
        })
}
