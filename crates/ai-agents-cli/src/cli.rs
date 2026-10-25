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
    /// Run and manage development task-controller work (not production restart hosting)
    Task(TaskArgs),
    /// Evaluate an agent YAML against declarative scenarios
    Eval(EvalArgs),
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

    /// Set actor_id at startup (enables cross-session actor memory)
    #[arg(long, value_name = "ID")]
    pub actor: Option<String>,

    /// Force plain line REPL (skip TUI even on interactive TTY)
    #[arg(long, action = ArgAction::SetTrue)]
    pub plain: bool,

    /// Color theme (dark, one-dark, catppuccin-mocha, dracula, tokyo-night, vscode-dark, nord, gruvbox-dark, light, one-half-light, github-light)
    #[arg(long)]
    pub theme: Option<String>,
}

#[derive(Debug, Clone, Parser)]
pub struct TaskArgs {
    #[command(subcommand)]
    pub command: TaskCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum TaskCommand {
    /// Run an objective through the configured shared controller
    Run(TaskRunArgs),
    /// Request new-process resume; the default host rejects unsupported reconstruction
    Resume(TaskRevisionArgs),
    /// Record a cancellation request, not remote cleanup acknowledgement
    Cancel(TaskRevisionArgs),
    /// List task metadata without constructing providers or consuming pending decisions
    List(TaskAgentArgs),
    /// Show a bounded default-redacted task snapshot
    Show(TaskIdArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct TaskAgentArgs {
    pub agent: PathBuf,
    #[arg(long, value_name = "ID")]
    pub actor: Option<String>,
    /// Emit safe JSON output; exact interactive prompts remain on stderr
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct TaskRunArgs {
    #[command(flatten)]
    pub host: TaskAgentArgs,
    pub objective: String,
    #[arg(long)]
    pub profile: Option<String>,
    /// Do not service stdin prompts; unavailable policies still permit normal completion
    #[arg(long)]
    pub non_interactive: bool,
    #[arg(long = "context", value_name = "KEY=VALUE")]
    pub contexts: Vec<String>,
    #[arg(long)]
    pub context_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Parser)]
pub struct TaskIdArgs {
    #[command(flatten)]
    pub host: TaskAgentArgs,
    pub run_id: String,
}

#[derive(Debug, Clone, Parser)]
pub struct TaskRevisionArgs {
    #[command(flatten)]
    pub target: TaskIdArgs,
    /// Exact observed revision; required for cancellation, never auto-refreshed
    #[arg(long)]
    pub revision: Option<u64>,
}

#[derive(Debug, Clone, Parser)]
pub struct EvalArgs {
    /// Path to the agent YAML file. Overrides suite agent.
    #[arg(short, long)]
    pub agent: Option<PathBuf>,

    /// Scenario suite YAML or JSONL file
    #[arg(short, long)]
    pub scenarios: PathBuf,

    /// Output directory
    #[arg(short, long, default_value = "./eval_results")]
    pub output: PathBuf,

    /// Run only a scenario ID. Repeatable.
    #[arg(short = 'i', long = "id")]
    pub ids: Vec<String>,

    /// Run scenarios matching tags. Repeatable.
    #[arg(short, long = "tags")]
    pub tags: Vec<String>,

    /// Tag match mode: all or any
    #[arg(long, default_value = "any")]
    pub tag_mode: String,

    /// Run only scenarios for a language. Repeatable.
    #[arg(long = "language")]
    pub languages: Vec<String>,

    /// Override retry count
    #[arg(long)]
    pub retries: Option<u32>,

    /// Override timeout per turn in milliseconds
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Run up to N scenarios concurrently where isolation allows it
    #[arg(long)]
    pub parallel: Option<usize>,

    /// Stop after first failure
    #[arg(long, action = ArgAction::SetTrue)]
    pub fail_fast: bool,

    /// Write JUnit XML
    #[arg(long, action = ArgAction::SetTrue)]
    pub junit: bool,

    /// Print summary JSON to stdout
    #[arg(long, action = ArgAction::SetTrue)]
    pub json: bool,

    /// Enable default observability overlay if suite does not specify it
    #[arg(long, action = ArgAction::SetTrue)]
    pub observability: bool,

    /// Validate the suite and referenced agent without executing scenarios or providers
    #[arg(long, action = ArgAction::SetTrue)]
    pub dry_config_check: bool,

    /// Record real LLM responses to a cassette JSONL file
    #[arg(long, value_name = "FILE")]
    pub record: Option<PathBuf>,

    /// Replay LLM responses from a cassette JSONL file
    #[arg(long, value_name = "FILE")]
    pub replay: Option<PathBuf>,

    /// Force real LLM calls even when the suite config declares a fixture mode
    #[arg(long, action = ArgAction::SetTrue)]
    pub real_llm: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct ValidateArgs {
    /// Path to the agent YAML file
    pub agent: PathBuf,
}
