use ai_agents::{Agent, AgentBuilder, Result};
use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "ai_agents=info".to_string()))
        .init();

    println!("=== State Machine Agent Demo ===\n");

    let agent = AgentBuilder::from_template("support_state_machine")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    println!("Agent: {}", agent.info().name);
    println!("Initial state: {:?}", agent.current_state());
    println!();
    println!("This agent has multiple branches:");
    println!("  - Technical support (login issues, errors)");
    println!("  - Order support (shipping, returns, refunds)");
    println!("  - Product inquiry (recommendations, specs)");
    println!("  - Escalation (for frustrated users)");
    println!();
    println!("Try conversations like:");
    println!("  - 'I can't log in to my account'");
    println!("  - 'Where is my order?'");
    println!("  - 'I want a refund'");
    println!("  - 'What laptop do you recommend?'");
    println!();
    println!("Commands: 'state' (show current), 'history' (show transitions), 'reset', 'quit'\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Show current state in prompt
        let state_name = agent.current_state().unwrap_or_else(|| "none".to_string());
        print!("[{}] You > ", state_name);
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input.to_lowercase().as_str() {
            "quit" | "exit" => {
                println!("Goodbye!");
                break;
            }
            "state" => {
                println!("Current state: {:?}", agent.current_state());
                continue;
            }
            "history" => {
                let history = agent.state_history();
                if history.is_empty() {
                    println!("No transitions yet.");
                } else {
                    println!("Transition history:");
                    for event in history {
                        println!("  {} -> {} ({})", event.from, event.to, event.reason);
                    }
                }
                continue;
            }
            "reset" => {
                agent.reset().await?;
                println!(
                    "Agent reset. Back to initial state: {:?}",
                    agent.current_state()
                );
                continue;
            }
            _ => {}
        }

        match agent.chat(input).await {
            Ok(response) => {
                let new_state = agent.current_state().unwrap_or_else(|| "none".to_string());
                println!("\n[{}] Agent: {}\n", new_state, response.content);
            }
            Err(e) => {
                eprintln!("\n[Error] {}\n", e);
            }
        }
    }

    Ok(())
}
