use ai_agents::{AgentBuilder, AgentSpec, ProviderType, Result, UnifiedLLMProvider};
use example_support::{Repl, init_tracing};
use std::sync::Arc;

fn create_tool_agent_spec() -> AgentSpec {
    AgentSpec {
        name: "ToolAgent".to_string(),
        version: "1.0.0".to_string(),
        description: Some("An agent that demonstrates tool usage".to_string()),
        system_prompt: "You are a helpful assistant with access to tools.".to_string(),
        max_iterations: 5,
        ..Default::default()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;

    let agent = AgentBuilder::from_spec(create_tool_agent_spec())
        .llm(Arc::new(llm))
        .auto_configure_features()?
        .build()?;

    Repl::new(agent)
        .welcome("=== Tool Agent Demo ===")
        .show_tool_calls()
        .hint("Try: 'calculate 2+2*3'")
        .hint("Try: 'What time is it?'")
        .hint("Try: 'echo Hello World'")
        .run()
        .await
}
