use ai_agents::{AgentBuilder, AgentStorage, Result, SessionQuery, SqliteStorage, StorageConfig};
use example_support::{CommandResult, Repl, init_tracing};
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
        Some((c, a)) => (c.to_lowercase(), a.trim()),
        None => (input.to_lowercase(), ""),
    };

    match cmd.as_str() {
        "/save" => {
            if arg.is_empty() {
                println!("Usage: /save <session_id>\n");
            } else {
                match block_on(agent.save_to(storage, arg)) {
                    Ok(()) => println!("Session saved: {}\n", arg),
                    Err(e) => eprintln!("Error saving: {}\n", e),
                }
            }
        }
        "/load" => {
            if arg.is_empty() {
                println!("Usage: /load <session_id>\n");
            } else {
                match block_on(agent.load_from(storage, arg)) {
                    Ok(true) => println!("Session loaded: {}\n", arg),
                    Ok(false) => println!("Session not found: {}\n", arg),
                    Err(e) => eprintln!("Error loading: {}\n", e),
                }
            }
        }
        "/delete" => {
            if arg.is_empty() {
                println!("Usage: /delete <session_id>\n");
            } else {
                match block_on(storage.delete(arg)) {
                    Ok(()) => println!("Session deleted: {}\n", arg),
                    Err(e) => eprintln!("Error deleting: {}\n", e),
                }
            }
        }
        "/info" => {
            if arg.is_empty() {
                println!("Usage: /info <session_id>\n");
            } else {
                match block_on(storage.get_session_info(arg)) {
                    Ok(Some(info)) => {
                        println!("\n--- Session Info ---");
                        println!("  ID:       {}", info.session_id);
                        println!("  Agent:    {}", info.agent_id);
                        println!("  Messages: {}", info.message_count);
                        println!("  State:    {:?}", info.current_state);
                        println!("  Created:  {}", info.created_at);
                        println!("  Updated:  {}", info.updated_at);
                        println!("--------------------\n");
                    }
                    Ok(None) => println!("Session not found: {}\n", arg),
                    Err(e) => eprintln!("Error: {}\n", e),
                }
            }
        }
        "/search" => {
            if arg.is_empty() {
                println!("Usage: /search <agent_name>\n");
            } else {
                let query = SessionQuery {
                    agent_id: Some(arg.to_string()),
                    limit: Some(10),
                    ..Default::default()
                };
                match block_on(storage.search_sessions(&query)) {
                    Ok(sessions) if sessions.is_empty() => {
                        println!("No sessions found for agent: {}\n", arg);
                    }
                    Ok(sessions) => {
                        println!("\n--- Sessions for '{}' ({}) ---", arg, sessions.len());
                        for s in sessions {
                            println!(
                                "  - {} (msgs: {}, state: {:?})",
                                s.session_id, s.message_count, s.current_state
                            );
                        }
                        println!("---\n");
                    }
                    Err(e) => eprintln!("Error: {}\n", e),
                }
            }
        }
        "/list" => match block_on(storage.list_sessions()) {
            Ok(ids) if ids.is_empty() => println!("No sessions found.\n"),
            Ok(ids) => {
                println!("\n--- Sessions ({}) ---", ids.len());
                for id in ids {
                    println!("  - {}", id);
                }
                println!("---\n");
            }
            Err(e) => eprintln!("Error: {}\n", e),
        },
        "/storage" => {
            let config = StorageConfig::sqlite("./agent_sessions.db");
            println!("\n--- Storage Info ---");
            println!("  Type: {}", config.storage_type());
            println!("  Path: {:?}", config.get_path());
            if let Some(sc) = config.as_sqlite() {
                println!("  Table: {}", sc.table.as_deref().unwrap_or("(default)"));
            }
            println!("--------------------\n");
        }
        _ => return CommandResult::NotHandled,
    }
    CommandResult::Handled
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let storage = Arc::new(SqliteStorage::new("./agent_sessions.db").await?);

    let agent = AgentBuilder::from_template("sqlite_persistence")?
        .auto_configure_llms()?
        .build()?;

    let storage_cmd = storage.clone();

    Repl::new(agent)
        .welcome("=== SQLite Persistence Demo ===")
        .hint("/save <id>      - Save current session")
        .hint("/load <id>      - Load a saved session")
        .hint("/list           - List all sessions")
        .hint("/search <agent> - Search sessions by agent name")
        .hint("/delete <id>    - Delete a session")
        .hint("/info <id>      - Show session details")
        .hint("/storage        - Show storage configuration")
        .on_command(move |input, agent| handle_command(input, agent, &storage_cmd))
        .run()
        .await
}
