use ai_agents::{AgentBuilder, ProviderType, Result, UnifiedLLMProvider};
use ai_agents_cli::{init_tracing, CliRepl as Repl};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-mini")?;

    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a friendly, curious, and helpful AI assistant with a casual and conversational style.",
        )
        .llm(Arc::new(llm))
        .build()?;

    Repl::new(agent)
        .welcome("=== Simple Chat Agent ===")
        .run()
        .await
}
