// Bind one selected workspace file to an actual provider and a host-owned checker.
// Flow: validate explicit paths -> prepare a provider-free builder -> start a task with typed approvals.
// The checker is trusted host code; the caller must provide any required isolation and independent tests.

use crate::code_agent_workspace::command_word;
use ai_agents::AgentBuilder;
use ai_agents::autonomy::{AutonomyExtensions, HostValidationBinding};
use ai_agents::context::ContextSource;
use ai_agents::spec::{AgentSpec, LLMConfigOrSelector};
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
const CHECKER_LIMIT: u64 = 8 * 1024 * 1024;
const INSTRUCTIONS_LIMIT: u64 = 64 * 1024;

#[cfg(all(test, target_os = "linux"))]
#[path = "workspace_agent_tests.rs"]
mod tests;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct WorkspaceSetup {
    spec: AgentSpec,
    root: PathBuf,
    source: PathBuf,
    checker: PathBuf,
    checker_bytes: Vec<u8>,
    argv: Vec<String>,
    command: String,
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

// Bind one selected file and one host-only checker without constructing a provider or starting work.
pub fn prepare(
    root: &Path,
    source: &Path,
    checker: &Path,
    provider: &str,
    model: &str,
    instruction_file: Option<&Path>,
) -> Result<WorkspaceSetup> {
    if !root.is_absolute() || !checker.is_absolute() {
        return Err("workspace and checker must be absolute paths".into());
    }
    let root = fs::canonicalize(root)?;
    if !root.is_dir() {
        return Err("workspace must be a directory".into());
    }
    let source = selected_file(&root, source)?;
    // Atomic replacement preserves ordinary Unix bits, not special privilege bits or extended metadata.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(&source)?.permissions().mode() & 0o7000 != 0 {
            return Err("source files with special permission bits are unsupported".into());
        }
    }
    let checker = fs::canonicalize(checker)?;
    if checker.starts_with(&root) {
        return Err("the trusted checker must be outside the workspace".into());
    }
    let checker_bytes = checker_snapshot(&checker)?;
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
    let argv = vec![
        command_word(&checker)?,
        command_word(&root)?,
        command_word(&source)?,
    ];
    let command = argv.join(" ");
    let root_string = command_word(&root)?;
    let source_string = command_word(&source)?;
    let mut spec = AgentSpec::from_yaml_strict(YAML)?;
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
        .read_paths = vec![source_string.clone()];
    let edit = spec
        .tool_security
        .tools
        .get_mut("file_edit")
        .ok_or("missing file_edit policy")?;
    edit.read_paths = vec![source_string.clone()];
    edit.write_paths = vec![source_string.clone()];
    let policy = spec
        .tool_security
        .tools
        .get_mut("command")
        .ok_or("missing command policy")?;
    policy.allowed_commands = vec![CommandRuleConfig { argv: argv.clone() }];
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
    checks[0].config =
        Some(json!({"command":command,"cwd":root_string,"host_binding":"workspace-checker"}));
    for (key, value) in [
        ("workspace_root", root_string),
        ("source_file", source_string),
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
    Ok(WorkspaceSetup {
        spec,
        root,
        source,
        checker,
        checker_bytes,
        argv,
        command,
    })
}

impl WorkspaceSetup {
    pub fn builder(&self) -> AgentBuilder {
        AgentBuilder::from_spec(self.spec.clone())
    }

    // A fixed host-only validation binding cannot become an ordinary model-callable command grant.
    pub fn extensions(&self) -> ai_agents::Result<AutonomyExtensions> {
        let mut extensions = AutonomyExtensions::builtins();
        extensions.register_host_validation(HostValidationBinding {
            id: "workspace-checker".into(),
            validator: "builtin.command".into(),
            version: 1,
            profile: None,
            scope: "task".into(),
            tool: "command".into(),
            arguments: json!({"command":self.command,"cwd":self.root}),
            require_approval: true,
        })?;
        Ok(extensions)
    }

    pub fn checker_runner(&self) -> Arc<dyn CommandRunner> {
        Arc::new(CheckerRunner {
            checker: self.checker.clone(),
            expected: self.checker_bytes.clone(),
            argv: self.argv.clone(),
            cwd: self.root.to_string_lossy().into_owned(),
            inner: ProcessCommandRunner,
        })
    }

    pub fn source_file(&self) -> &Path {
        &self.source
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
    println!("Selected source: {}", setup.source_file().display());
    println!(
        "The selected source, instruction snapshot, task messages and checker output may be sent to the chosen provider and may incur costs."
    );
    println!(
        "The external checker runs with host permissions; you are responsible for its independence, isolation and cleanup."
    );
    println!("Only /task commands are enabled; command execution and edits require approval.");
    println!(
        "Atomic edits preserve ordinary Unix mode bits, not ownership, ACLs or extended attributes."
    );
    let agent = setup
        .builder()
        .auto_configure_llms()?
        .auto_configure_features()?
        .autonomy_extensions(setup.extensions()?)
        .approval_handler(CliApprovalHandler::from_metadata(None))
        .task_interaction_delivery(true)
        .build_task_host()
        .await?;
    agent.set_command_runner(setup.checker_runner());
    CliRepl::new(agent)
        .welcome("Code Agent: one selected file, one host-owned checker")
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
        .run()
        .await?;
    Ok(())
}
