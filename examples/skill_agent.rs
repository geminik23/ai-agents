use ai_agents::{AgentBuilder, Result};
use example_support::{Repl, init_tracing};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_template("skill_agent")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    Repl::new(agent)
        .welcome("=== Skill Agent Demo ===")
        .show_tool_calls()
        .hint("Try: 'Hello!' (triggers greeting skill — inline)")
        .hint("Try: 'What is 15 * 7 + 3?' (triggers math_helper — external by name)")
        .hint("Try: 'What should I wear today?' (triggers weather_clothes — external by path)")
        .hint("Try: 'Tell me a joke' (normal chat, no skill match)")
        .run()
        .await
}
