use ai_agents::llm::providers::ProviderType;
use ai_agents::{
    Agent, AgentBuilder, LLMRegistry, SkillDefinition, SkillStep, UnifiedLLMProvider,
    create_builtin_registry,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Skill Agent Demo ===\n");

    let provider = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;
    let provider = Arc::new(provider);

    let mut llm_registry = LLMRegistry::new();
    llm_registry.register("default", provider.clone());
    llm_registry.register("router", provider.clone());
    llm_registry.set_default("default");
    llm_registry.set_router("router");

    let inline_skill = SkillDefinition {
        id: "greeting".to_string(),
        description: "Greet the user warmly".to_string(),
        trigger: "When user says hello, hi, or greets".to_string(),
        steps: vec![SkillStep::Prompt {
            prompt: r#"The user greeted you with: "{{ user_input }}"

Respond with a warm, friendly greeting. Be enthusiastic but not over the top."#
                .to_string(),
            llm: None,
        }],
    };

    let calc_skill = SkillDefinition {
        id: "calculator".to_string(),
        description: "Perform mathematical calculations".to_string(),
        trigger: "When user asks to calculate something or needs math help".to_string(),
        steps: vec![
            SkillStep::Tool {
                tool: "calculator".to_string(),
                args: Some(serde_json::json!({
                    "operation": "2 + 2"
                })),
                output_as: None,
            },
            SkillStep::Prompt {
                prompt: r#"User asked: {{ user_input }}

Calculator result: {{ steps[0].result }}

Explain the result to the user in a helpful way."#
                    .to_string(),
                llm: None,
            },
        ],
    };

    let agent = AgentBuilder::new()
        .system_prompt("You are a helpful assistant with specialized skills.")
        .llm_registry(llm_registry)
        .tools(create_builtin_registry())
        .skill(inline_skill)
        .skill(calc_skill)
        .build()?;

    println!("Agent created with {} skills\n", agent.skills().len());

    println!("--- Test 1: Greeting ---");
    let response = agent.chat("Hello there!").await?;
    println!("Response: {}\n", response.content);

    agent.reset().await?;

    println!("--- Test 2: Calculation ---");
    let response = agent.chat("Can you help me calculate something?").await?;
    println!("Response: {}\n", response.content);

    agent.reset().await?;

    println!("--- Test 3: Normal chat (no skill match) ---");
    let response = agent.chat("Tell me about the weather").await?;
    println!("Response: {}\n", response.content);

    println!("=== Demo Complete ===");
    Ok(())
}
