use ai_agents::{Agent, AgentBuilder, Result};
use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== YAML Agent Demo ===\n");
    println!("Loading agent from templates/skill_agent.yaml...\n");

    let agent = AgentBuilder::from_template("skill_agent")?
        .auto_configure_llms()?
        .build()?;

    println!("Agent loaded: {}", agent.info().name);
    println!("Skills: {}", agent.skills().len());
    println!();
    println!("Try these:");
    println!("  - 'Hello!' (triggers greeting skill)");
    println!("  - 'Can you help me calculate something?' (triggers calculator skill)");
    println!("  - 'Tell me a joke' (normal chat, no skill)");
    println!();
    println!("Type 'quit' to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("[You] > ");
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
                println!("\n[Agent] {}\n", response.content);
            }
            Err(e) => {
                eprintln!("\n[Error] {}\n", e);
            }
        }
    }

    Ok(())
}
