use ai_agents::{
    create_memory_from_config, Agent, AgentBuilder, ProviderType, TemplateLoader,
    UnifiedLLMProvider,
};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Chat Agent ===\n");

    let mut loader = TemplateLoader::new();
    loader.add_search_path("templates");
    loader.set_variable("agent_name", "ChatBot");
    loader.set_variable(
        "system_prompt",
        "You are a friendly, curious, and helpful AI assistant with a casual and conversational style.",
    );

    let spec = loader.load_and_parse("simple")?;

    println!("Loaded agent: {} v{}", spec.name, spec.version);
    if let Some(ref desc) = spec.description {
        println!("Description: {}", desc);
    }
    println!();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;

    let memory = create_memory_from_config(&spec.memory)?;
    let agent = AgentBuilder::from_spec(spec)
        .llm(Arc::new(llm))
        .memory(memory)
        .build()?;

    println!("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n");

    let stdin = io::stdin();
    let reader = stdin.lock();

    for line in reader.lines() {
        let input = line?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }

        if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
            println!("\nGoodbye!");
            break;
        }

        print!("You: {}\n", trimmed);
        io::stdout().flush()?;

        let rt = tokio::runtime::Runtime::new()?;
        match rt.block_on(agent.chat(trimmed)) {
            Ok(response) => {
                println!("Bot: {}\n", response.content);
            }
            Err(e) => {
                eprintln!("Error: {}\n", e);
            }
        }
    }

    Ok(())
}
