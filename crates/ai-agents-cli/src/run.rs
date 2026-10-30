use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_agents::{
    AgentBuilder, LoggingHooks, RuntimeAgent,
    spec::{AgentSpec, CliPromptStyle},
    tools::ProcessCommandRunner,
};
use anyhow::{Context, Result};

use crate::approval::CliApprovalHandler;
use crate::cli::{Cli, Command as CliCommand, RunArgs};
use crate::metadata::{CliOverrides, ResolvedCliMetadata};

use crate::repl::{CliRepl, CliReplConfig, PromptStyle, ReplMode};

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub agent_path: PathBuf,
    pub welcome: Option<String>,
    pub extra_hints: Vec<String>,
    pub stream: Option<bool>,
    pub show_tools: Option<bool>,
    pub show_state: Option<bool>,
    pub show_timing: Option<bool>,
    pub no_builtins: bool,
    pub contexts: Vec<String>,
    pub context_file: Option<PathBuf>,
    pub actor: Option<String>,
    pub plain: bool,
    pub theme: Option<String>,
}

impl RunOptions {
    pub fn from_run_args(args: &RunArgs) -> Self {
        Self {
            agent_path: args.agent.clone(),
            welcome: args.welcome.clone(),
            extra_hints: args.hints.clone(),
            stream: args.stream.then_some(true),
            show_tools: args.show_tools.then_some(true),
            show_state: args.show_state.then_some(true),
            show_timing: args.show_timing.then_some(true),
            no_builtins: args.no_builtins,
            contexts: args.contexts.clone(),
            context_file: args.context_file.clone(),
            actor: args.actor.clone(),
            plain: args.plain,
            theme: args.theme.clone(),
        }
    }

    fn to_overrides(&self) -> CliOverrides {
        CliOverrides {
            welcome: self.welcome.clone(),
            hints: if self.extra_hints.is_empty() {
                None
            } else {
                Some(self.extra_hints.clone())
            },
            show_tools: self.show_tools,
            show_state: self.show_state,
            show_timing: self.show_timing,
            streaming: self.stream,
            prompt_style: None,
            disable_builtin_commands: self.no_builtins.then_some(true),
            hitl: None,
            theme: self.theme.clone(),
        }
    }
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        CliCommand::Run(args) => run_agent(RunOptions::from_run_args(&args)).await,
        CliCommand::Task(args) => crate::task::run(args).await,
        CliCommand::Eval(args) => crate::eval::run_eval(args).await,
        CliCommand::Validate(args) => {
            crate::init_tracing();
            validate_agent(&args.agent).await
        }
    }
}

pub async fn run_agent(options: RunOptions) -> Result<()> {
    let spec = load_spec(&options.agent_path)?;
    let metadata = ResolvedCliMetadata::from_metadata_value(spec.metadata.as_ref())
        .merge_overrides(options.to_overrides());

    let is_tui = !options.plain && std::io::stdout().is_terminal();
    if !is_tui {
        // Plain mode: standard fmt subscriber to stdout.
        crate::init_tracing();
    }
    // TUI mode: no subscriber installed yet.
    // Tracing calls during build_agent() are silently dropped.
    // The TUI installs a channel-based subscriber inside run_tui().

    if !is_tui {
        let io = crate::session::FrontendSession::terminal(false, true);
        let namespace = ai_agents::TaskHostScope::from_spec(
            &spec,
            options.agent_path.parent().unwrap_or(Path::new(".")),
            options.actor.clone(),
        )?;
        let agent = build_frontend_agent(
            &options.agent_path,
            spec.clone(),
            io.bridge(),
            Some(namespace),
        )
        .await?;
        prepare_frontend(&agent, &options).await?;
        let config = resolve_cli_config(&spec, &metadata);
        return CliRepl::new(agent)
            .with_config(config)
            .run_with_io(io)
            .await;
    }

    let (io, input, output) = crate::session::FrontendSession::embedded();
    let namespace = ai_agents::TaskHostScope::from_spec(
        &spec,
        options.agent_path.parent().unwrap_or(Path::new(".")),
        options.actor.clone(),
    )?;
    let agent = build_frontend_agent(
        &options.agent_path,
        spec.clone(),
        io.bridge(),
        Some(namespace),
    )
    .await?;
    prepare_frontend(&agent, &options).await?;
    let config = resolve_cli_config(&spec, &metadata);
    crate::tui::run_prepared(agent, config, metadata.theme.clone(), io, input, output)
        .await
        .map_err(|error| anyhow::anyhow!("failed to run TUI session: {error:#}"))
}

/// Parse "key=value" where key supports dotted paths and value is auto-typed.
fn parse_context_pair(pair: &str) -> Result<(String, serde_json::Value)> {
    let (key, raw_value) = pair
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected key=value, got: {}", pair))?;
    let value = serde_json::from_str(raw_value)
        .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
    Ok((key.to_string(), value))
}

/// Inject context values from --context flags and --context-file into the agent.
pub(crate) fn inject_context(agent: &RuntimeAgent, options: &RunOptions) -> Result<()> {
    // Load context file first (--context overrides file values).
    if let Some(ref path) = options.context_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read context file: {}", path.display()))?;
        let obj: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse context file as JSON: {}", path.display()))?;
        if let serde_json::Value::Object(map) = obj {
            for (key, value) in map {
                agent
                    .set_context(&key, value)
                    .with_context(|| format!("failed to set context key: {}", key))?;
            }
        } else {
            anyhow::bail!("context file must contain a JSON object at the top level");
        }
    }

    // Then apply --context pairs (overrides file values on conflict).
    for pair in &options.contexts {
        let (key, value) = parse_context_pair(pair)?;
        agent
            .set_context(&key, value)
            .with_context(|| format!("failed to set context: {}", pair))?;
    }

    // Set actor_id from --actor flag.
    if let Some(ref actor_id) = options.actor {
        agent
            .set_actor_id(actor_id)
            .with_context(|| format!("failed to set actor_id: {}", actor_id))?;
    }

    Ok(())
}

pub async fn validate_agent(path: &Path) -> Result<()> {
    let spec = load_spec(path)?;
    println!(
        "Valid agent spec: {} v{} ({})",
        spec.name,
        spec.version,
        path.display()
    );
    Ok(())
}

pub fn load_spec(path: &Path) -> Result<AgentSpec> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read agent file: {}", path.display()))?;

    let spec = AgentSpec::from_yaml_strict(&content).map_err(|error| {
        anyhow::anyhow!(
            "failed to parse YAML agent file: {}: {error}",
            path.display()
        )
    })?;

    spec.validate()
        .with_context(|| format!("agent spec validation failed: {}", path.display()))?;

    Ok(spec)
}

pub async fn build_agent(path: &Path) -> Result<RuntimeAgent> {
    let spec = load_spec(path)?;
    build_agent_from_spec(path, spec).await
}

async fn build_agent_from_spec(path: &Path, spec: AgentSpec) -> Result<RuntimeAgent> {
    configured_builder(path, spec, None, None)
        .await?
        .build()
        .context("failed to build runtime agent")
}

// Both ordinary and task hosting finalize this same configured builder; neither path reinterprets YAML or repeats preparation.
async fn configured_builder(
    path: &Path,
    spec: AgentSpec,
    bridge: Option<Arc<crate::session::InputBridge>>,
    scope: Option<ai_agents::TaskHostScope>,
) -> Result<AgentBuilder> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let builder = AgentBuilder::from_spec_with_base_dir(spec.clone(), base_dir);
    let builder = if let Some(scope) = scope {
        builder.task_host_scope(scope)?
    } else {
        builder
    };
    let mut builder = builder
        .auto_configure_llms()
        .context("failed to auto-configure LLMs from environment")?
        .auto_configure_features()
        .context("failed to auto-configure agent features")?
        .auto_configure_mcp()
        .await
        .context("failed to auto-configure MCP tools")?
        .auto_configure_spawner()
        .await
        .context("failed to auto-configure spawner")?;

    // Attach LoggingHooks when the agent uses memory features that produce
    // hook events (compacting memory, token budgeting). Without this, budget
    // warnings and compression events are silently discarded by NoopHooks.
    if spec.memory.is_compacting() || spec.memory.token_budget.is_some() {
        builder = builder.hooks(Arc::new(LoggingHooks::with_prefix("[Memory]")));
    }

    // Attach CLI approval handler when the spec declares HITL config.
    if spec.hitl.is_some() {
        let hitl_meta = spec
            .metadata
            .as_ref()
            .and_then(|m| m.get("cli"))
            .and_then(|c| serde_json::from_value::<ai_agents::spec::CliMetadata>(c.clone()).ok())
            .and_then(|m| m.hitl);
        if let Some(bridge) = &bridge {
            bridge.set_show_context(
                hitl_meta
                    .as_ref()
                    .and_then(|meta| meta.show_context)
                    .unwrap_or(true),
            );
        }
        builder = if let Some(bridge) = bridge {
            bridge.set_auto_approval(
                match hitl_meta.as_ref().and_then(|meta| meta.style.as_ref()) {
                    Some(ai_agents::spec::CliHitlStyle::AutoApprove) => Some(true),
                    Some(ai_agents::spec::CliHitlStyle::AutoReject) => Some(false),
                    _ => None,
                },
            );
            builder.approval_handler(bridge)
        } else {
            builder.approval_handler(CliApprovalHandler::from_metadata(hitl_meta.as_ref()))
        };
    }

    Ok(builder)
}

/// Prepares the same ordinary storage/context/actor contract before foreground task admission.
pub(crate) async fn prepare_frontend(agent: &RuntimeAgent, options: &RunOptions) -> Result<()> {
    agent.set_command_runner(Arc::new(ProcessCommandRunner));
    agent
        .init_storage()
        .await
        .context("failed to initialize agent storage")?;
    inject_context(agent, options)
}

/// Reuses the production builder chain; the bridge changes ordinary input delivery, never task approval authority.
pub(crate) async fn build_frontend_agent(
    path: &Path,
    spec: AgentSpec,
    bridge: Arc<crate::session::InputBridge>,
    scope: Option<ai_agents::TaskHostScope>,
) -> Result<RuntimeAgent> {
    let builder = configured_builder(path, spec, Some(bridge.clone()), scope).await?;
    let agent = builder.build_task_host().await?;
    agent.set_question_handler(Some(bridge));
    Ok(agent)
}

pub fn resolve_cli_config(spec: &AgentSpec, metadata: &ResolvedCliMetadata) -> CliReplConfig {
    let show_state = metadata.show_state.unwrap_or(spec.states.is_some());
    let prompt = match metadata.prompt_style.clone() {
        Some(CliPromptStyle::Simple) => PromptStyle::Simple,
        Some(CliPromptStyle::WithState) => PromptStyle::WithState,
        None if show_state => PromptStyle::WithState,
        None => PromptStyle::Simple,
    };

    CliReplConfig {
        welcome: metadata.welcome.clone(),
        prompt,
        mode: if metadata.streaming.unwrap_or(false) {
            ReplMode::Streaming
        } else {
            ReplMode::Chat
        },
        show_tool_calls: metadata.show_tools.unwrap_or(false),
        show_state_transitions: show_state,
        show_timing: metadata.show_timing.unwrap_or(false),
        builtin_commands: !metadata.disable_builtin_commands.unwrap_or(false),
        hints: metadata.hints.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_spec_rejects_nested_unknown_path() {
        let path = std::env::temp_dir().join(format!(
            "ai-agents-cli-strict-spec-{}.yaml",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"
name: TestAgent
system_prompt: test
tool_security:
  enabeld: true
"#,
        )
        .unwrap();

        let error = load_spec(&path).unwrap_err();
        let _ = fs::remove_file(&path);
        let message = error.to_string();
        assert!(message.contains("'tool_security.enabeld'"), "{message}");
    }
}
