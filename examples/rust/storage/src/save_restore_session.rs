use ai_agents::{AgentBuilder, Result, SqliteStorage};
use ai_agents_cli::{CliRepl, CommandResult, init_tracing};
use std::sync::Arc;

/// Run an async block from inside the sync `on_command` callback.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(f))
}

fn handle_command(
    input: &str,
    agent: &ai_agents::RuntimeAgent,
    storage: &SqliteStorage,
) -> CommandResult {
    let (cmd, arg) = match input.split_once(' ') {
        Some((c, a)) => (c.to_lowercase(), a.trim().to_string()),
        None => (input.to_lowercase(), String::new()),
    };

    match cmd.as_str() {
        "/save" => {
            if arg.is_empty() {
                println!("Usage: /save <session_id>\n");
            } else {
                match block_on(agent.save_to(storage, &arg)) {
                    Ok(()) => println!("Session saved as \"{}\".\n", arg),
                    Err(e) => eprintln!("Error saving session: {}\n", e),
                }
            }
            CommandResult::Handled
        }
        "/load" => {
            if arg.is_empty() {
                println!("Usage: /load <session_id>\n");
            } else {
                match block_on(agent.load_from(storage, &arg)) {
                    Ok(true) => {
                        // Show what was restored so the user can see it worked.
                        match block_on(agent.save_state()) {
                            Ok(snapshot) => {
                                let msg_count = snapshot.memory.messages.len();
                                let has_summary = snapshot.memory.summary.is_some();
                                println!("Session \"{}\" restored.", arg);
                                println!(
                                    "  {} recent message(s){}",
                                    msg_count,
                                    if has_summary {
                                        " + conversation summary"
                                    } else {
                                        ""
                                    }
                                );
                                if let Some(ref state) = snapshot.state_machine {
                                    println!("  State: {}", state.current_state);
                                }
                                println!();
                            }
                            Err(_) => {
                                println!("Session \"{}\" restored.\n", arg);
                            }
                        }
                    }
                    Ok(false) => println!("Session \"{}\" not found.\n", arg),
                    Err(e) => eprintln!("Error loading session: {}\n", e),
                }
            }
            CommandResult::Handled
        }
        _ => CommandResult::NotHandled,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    // Create SQLite storage (the file is created automatically).
    let storage = Arc::new(SqliteStorage::new("./save_restore_sessions.db").await?);

    let agent = AgentBuilder::from_yaml_file("agents/save_restore_agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .build()?;

    let storage_cmd = storage.clone();

    CliRepl::new(agent)
        .welcome("=== Save & Restore Session Demo ===")
        .hint("Minimal session persistence — save a session, quit, restart, load it back.")
        .hint("")
        .hint("/save <id>  Save the current session")
        .hint("/load <id>  Restore a previously saved session")
        .hint("")
        .hint("Try: chat a bit, then /save mysession, quit, restart, /load mysession")
        .on_command(move |input, agent| handle_command(input, agent, &storage_cmd))
        .run()
        .await?;

    Ok(())
}
