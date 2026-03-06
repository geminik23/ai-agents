use ai_agents::{AgentBuilder, Result};
use ai_agents_cli::{CliRepl, init_tracing};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_yaml_file("agents/support_state_machine.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    CliRepl::new(agent)
        .show_state()
        .run()
        .await
}
