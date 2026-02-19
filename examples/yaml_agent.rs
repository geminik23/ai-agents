use ai_agents::{AgentBuilder, Result};
use example_support::{Repl, init_tracing};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_template("skill_agent")?
        .auto_configure_llms()?
        .build()?;

    Repl::new(agent)
        .welcome("=== YAML Skill Agent ===")
        .hint("Try: 'Hello!' (triggers greeting skill)")
        .hint("Try: 'Can you help me calculate something?'")
        .hint("Try: 'Tell me a joke' (normal chat, no skill)")
        .run()
        .await
}
