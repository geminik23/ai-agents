use std::io::{self, BufRead, Write};
use std::sync::Arc;

use ai_agents::{
    Agent, AgentBuilder, AgentSpec, ProviderType, Result, UnifiedLLMProvider,
    create_builtin_registry,
};

fn create_tool_agent_spec() -> AgentSpec {
    AgentSpec {
        name: "ToolAgent".to_string(),
        version: "1.0.0".to_string(),
        description: Some("An agent that demonstrates tool usage".to_string()),
        system_prompt: "You are a helpful assistant with access to tools.".to_string(),
        max_iterations: 5,
        ..Default::default()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging (use RUST_LOG=ai_agents=debug for verbose output)
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "ai_agents=info".to_string()))
        .init();

    println!(":::::Tool Agent Demo (gpt-4.1-nano):::::");
    println!();
    println!("Available commands:");
    println!("  - 'calculate <expr>' : Run a calculation (e.g., 'calculate 2+2*3')");
    println!("  - 'echo <message>'   : Echo back a message");
    println!("  - 'quit'             : Exit the demo");
    println!();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")
        .expect("Failed to create LLM provider. Make sure OPENAI_API_KEY is set.");

    let spec = create_tool_agent_spec();
    let tools = create_builtin_registry();

    let agent = AgentBuilder::from_spec(spec)
        .llm(Arc::new(llm))
        .tools(tools)
        .build()?;

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\n[You] > ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        match agent.chat(input).await {
            Ok(response) => {
                println!("\n[Agent] {}", response.content);
                if let Some(ref calls) = response.tool_calls {
                    if !calls.is_empty() {
                        println!(
                            "  (Used tools: {:?})",
                            calls.iter().map(|t| &t.name).collect::<Vec<_>>()
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("\n[Error] {}", e);
            }
        }
    }

    Ok(())
}
