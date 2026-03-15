use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_agents::{
    AgentBuilder, LoggingHooks, RuntimeAgent,
    spec::{AgentSpec, CliPromptStyle},
};
use anyhow::{Context, Result};

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
        }
    }
}

pub async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        CliCommand::Run(args) => run_agent(RunOptions::from_run_args(&args)).await,
        CliCommand::Validate(args) => validate_agent(&args.agent).await,
    }
}

pub async fn run_agent(options: RunOptions) -> Result<()> {
    let spec = load_spec(&options.agent_path)?;
    let metadata = ResolvedCliMetadata::from_metadata_value(spec.metadata.as_ref())
        .merge_overrides(options.to_overrides());

    let agent = build_agent(&options.agent_path).await?;
    let config = resolve_cli_config(&spec, &metadata);

    CliRepl::new(agent)
        .with_config(config)
        .run()
        .await
        .context("failed to run interactive session")
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

    let spec: AgentSpec = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse YAML agent file: {}", path.display()))?;

    spec.validate()
        .with_context(|| format!("agent spec validation failed: {}", path.display()))?;

    Ok(spec)
}

pub async fn build_agent(path: &Path) -> Result<RuntimeAgent> {
    let spec = load_spec(path)?;

    let mut builder = AgentBuilder::from_yaml_file(path)
        .with_context(|| format!("failed to load YAML from {}", path.display()))?
        .auto_configure_llms()
        .context("failed to auto-configure LLMs from environment")?
        .auto_configure_features()
        .context("failed to auto-configure agent features")?
        .auto_configure_mcp()
        .await
        .context("failed to auto-configure MCP tools")?;

    // Attach LoggingHooks when the agent uses memory features that produce
    // hook events (compacting memory, token budgeting). Without this, budget
    // warnings and compression events are silently discarded by NoopHooks.
    if spec.memory.is_compacting() || spec.memory.token_budget.is_some() {
        builder = builder.hooks(Arc::new(LoggingHooks::with_prefix("[Memory]")));
    }

    builder.build().context("failed to build runtime agent")
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
