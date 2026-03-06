use ai_agents::{AgentBuilder, Result};
use ai_agents_cli::{CliRepl, CliReplConfig, init_tracing};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let agent = AgentBuilder::from_yaml_file("agents/skill_agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    CliRepl::new(agent)
        .with_config(CliReplConfig {
            welcome: Some("=== Skill Agent Demo ===".to_string()),
            show_tool_calls: true,
            hints: vec![
                "Try: 'Hello!' (triggers greeting skill — inline)".to_string(),
                "Try: 'What is 15 * 7 + 3?' (triggers math_helper — external by name)".to_string(),
                "Try: 'What should I wear today?' (triggers weather_clothes — external by path)".to_string(),
                "Try: 'Tell me a joke' (normal chat, no skill match)".to_string(),
            ],
            ..Default::default()
        })
        .run()
        .await
}
