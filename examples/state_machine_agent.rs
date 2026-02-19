use ai_agents::{AgentBuilder, Result};
use example_support::{Repl, init_tracing};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_template("support_state_machine")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    Repl::new(agent)
        .welcome("=== State Machine Agent Demo ===")
        .show_state()
        .hint("Try: 'I can't log in to my account'")
        .hint("Try: 'Where is my order?'")
        .hint("Try: 'I want a refund'")
        .hint("Try: 'What laptop do you recommend?'")
        .run()
        .await
}
