use ai_agents::{AgentBuilder, Result};
use ai_agents_cli::{CliRepl as Repl, init_tracing};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_yaml_file("../../yaml/basic/simple_chat.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    Repl::new(agent)
        .welcome("=== YAML Loader Example ===")
        .hint("This Rust example loads an agent from YAML.")
        .hint("Try: 'hello'")
        .hint("Try: 'what can you do?'")
        .run()
        .await
}
