use ai_agents::{
    create_builtin_registry, Agent, AgentBuilder, LLMRegistry, ProviderType, SkillDefinition,
    SkillStep, UnifiedLLMProvider,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "ai_agents=info".to_string()))
        .init();

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
            // First: Extract the math expression from user input using LLM
            SkillStep::Prompt {
                prompt: r#"Extract ONLY the mathematical expression from this user message.
Output ONLY the math expression, nothing else. If no clear expression, output "2 + 2" as default.

User message: "{{ user_input }}"

Mathematical expression:"#
                    .to_string(),
                llm: None,
            },
            // Second: Call calculator with the extracted expression
            SkillStep::Tool {
                tool: "calculator".to_string(),
                args: Some(serde_json::json!({
                    "expression": "{{ steps[0].result }}"
                })),
                output_as: None,
            },
            // Third: Explain the result to the user
            SkillStep::Prompt {
                prompt: r#"User asked: {{ user_input }}

Calculator result: {{ steps[1].result }}

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
    let response = agent.chat("Can you help me calculate 15 * 7 + 3?").await?;
    println!("Response: {}\n", response.content);

    agent.reset().await?;

    println!("--- Test 3: Normal chat (no skill match) ---");
    let response = agent.chat("Tell me about the weather").await?;
    println!("Response: {}\n", response.content);

    println!("=== Demo Complete ===");
    Ok(())
}
