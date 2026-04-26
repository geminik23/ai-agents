pub mod approval;
pub mod cli;
pub mod metadata;
pub mod repl;
pub mod run;
pub mod tui;

pub use approval::CliApprovalHandler;
pub use cli::{Cli, Command, RunArgs, ValidateArgs};
pub use metadata::{CliOverrides, ResolvedCliMetadata};
pub use repl::{CliRepl, CliReplConfig, CommandResult, PromptStyle, ReplMode};
pub use run::{
    RunOptions, build_agent, load_spec, resolve_cli_config, run, run_agent, validate_agent,
};

pub fn init_tracing() {
    init_tracing_with_default("ai_agents=info,ai_agents_facts=warn");
}

pub fn init_tracing_with_default(default_filter: &str) {
    let builder = tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| default_filter.to_string()));

    let _ = builder.try_init();
}
