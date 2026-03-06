use ai_agents::{ProviderType, Result, UnifiedLLMProvider};
use ai_agents_cli::{CliRepl as Repl, ReplMode, init_tracing};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-mini")?;

    let agent = ai_agents::AgentBuilder::new()
        .system_prompt("You are a helpful assistant. Keep responses clear and concise.")
        .llm(Arc::new(llm))
        .streaming(true)
        .build()?;

    Repl::new(agent)
        .welcome("=== Streaming Chat Agent ===")
        .hint("This example streams tokens as they are generated.")
        .hint("Try: 'Write a short poem about Rust.'")
        .hint("Try: 'Explain async/await simply.'")
        .with_config(ai_agents_cli::CliReplConfig {
            mode: ReplMode::Streaming,
            ..Default::default()
        })
        .run()
        .await
}
