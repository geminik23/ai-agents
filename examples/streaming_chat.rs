use ai_agents::{
    AgentBuilder, ProviderType, StreamChunk, TemplateLoader, UnifiedLLMProvider,
    create_memory_from_config,
};
use futures::StreamExt;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "ai_agents=info".to_string()))
        .init();

    println!("=== Streaming Chat Agent ===\n");

    let mut loader = TemplateLoader::new();
    loader.add_search_path("templates");
    loader.set_variable("agent_name", "StreamingBot");
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

    let memory = create_memory_from_config(&spec.memory);
    let agent = AgentBuilder::from_spec(spec)
        .llm(Arc::new(llm))
        .memory(memory)
        .build()?;

    println!("Type your message and press Enter. Type 'quit' or 'exit' to stop.");
    println!("Responses will stream in real-time.\n");

    let stdin = io::stdin();

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }

        if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
            println!("\nGoodbye!");
            break;
        }

        print!("Bot: ");
        io::stdout().flush()?;

        match agent.chat_stream(trimmed).await {
            Ok(mut stream) => {
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        StreamChunk::Content { text } => {
                            print!("{}", text);
                            io::stdout().flush()?;
                        }
                        StreamChunk::ToolCallStart { name, .. } => {
                            print!("\n[Calling tool: {}...", name);
                            io::stdout().flush()?;
                        }
                        StreamChunk::ToolResult {
                            output, success, ..
                        } => {
                            if success {
                                print!(" ✓]\n[Result: {}]", output);
                            } else {
                                print!(" ✗]\n[Error: {}]", output);
                            }
                            io::stdout().flush()?;
                        }
                        StreamChunk::ToolCallEnd { .. } => {}
                        StreamChunk::StateTransition { from, to } => {
                            let from_str = from.as_deref().unwrap_or("none");
                            print!("\n[State: {} → {}]", from_str, to);
                            io::stdout().flush()?;
                        }
                        StreamChunk::Done {} => {
                            println!("\n");
                            break;
                        }
                        StreamChunk::Error { message } => {
                            eprintln!("\n[Stream Error: {}]\n", message);
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                eprintln!("\nError: {}\n", e);
            }
        }
    }

    Ok(())
}
