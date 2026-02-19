mod common;

pub use common::repl::{PromptStyle, Repl, ReplConfig, ReplMode};

pub fn init_tracing() {
    init_tracing_with_default("ai_agents=info");
}

pub fn init_tracing_with_default(default_filter: &str) {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| default_filter.to_string()))
        .init();
}
