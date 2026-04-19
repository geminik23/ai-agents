use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "ai-agents-cli",
    version,
    about = "Run YAML-defined AI agents from the command line",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run an agent YAML file interactively
    Run(RunArgs),
    /// Validate an agent YAML file without starting the REPL
    Validate(ValidateArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct RunArgs {
    /// Path to the agent YAML file
    pub agent: PathBuf,

    /// Stream response tokens in real time
    #[arg(long, action = ArgAction::SetTrue)]
    pub stream: bool,

    /// Show tool calls used by the agent
    #[arg(long, action = ArgAction::SetTrue)]
    pub show_tools: bool,

    /// Show current state in the prompt and transitions in output
    #[arg(long, action = ArgAction::SetTrue)]
    pub show_state: bool,

    /// Show elapsed time for each response
    #[arg(long, action = ArgAction::SetTrue)]
    pub show_timing: bool,

    /// Disable built-in REPL commands such as help/reset/info/state/history
    #[arg(long, action = ArgAction::SetTrue)]
    pub no_builtins: bool,

    /// Override YAML metadata welcome message
    #[arg(long)]
    pub welcome: Option<String>,

    /// Add an extra startup hint (can be repeated)
    #[arg(long = "hint")]
    pub hints: Vec<String>,

    /// Inject a runtime context value as key=value (repeatable, supports dotted paths)
    #[arg(long = "context", value_name = "KEY=VALUE")]
    pub contexts: Vec<String>,

    /// Inject runtime context from a JSON file
    #[arg(long = "context-file", value_name = "PATH")]
    pub context_file: Option<PathBuf>,

    /// Force plain line REPL (skip TUI even on interactive TTY)
    #[arg(long, action = ArgAction::SetTrue)]
    pub plain: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct ValidateArgs {
    /// Path to the agent YAML file
    pub agent: PathBuf,
}
