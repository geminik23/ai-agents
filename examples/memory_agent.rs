//! Memory Agent Example - Demonstrating CompactingMemory with Auto-Summarization
//!
//! This example shows how to:
//! - Configure CompactingMemory with token budgeting
//! - Use LLMSummarizer for automatic message compression
//! - Monitor memory events via hooks
//!
//! Run with: cargo run --example memory_agent

use ai_agents::{
    Agent, AgentBuilder, AgentHooks, AgentResponse, MemoryBudgetEvent, MemoryCompressEvent,
    MemoryEvictEvent, Result, RuntimeAgent,
};
use async_trait::async_trait;
use example_support::{CommandResult, Repl, init_tracing};
use parking_lot::Mutex;
use std::sync::Arc;

struct MemoryMonitorHooks {
    compress_count: Mutex<usize>,
    evict_count: Mutex<usize>,
    warnings: Mutex<Vec<String>>,
}

impl MemoryMonitorHooks {
    fn new() -> Self {
        Self {
            compress_count: Mutex::new(0),
            evict_count: Mutex::new(0),
            warnings: Mutex::new(Vec::new()),
        }
    }

    fn print_stats(&self) {
        println!("\n--- Memory Statistics ---");
        println!("  Compressions: {}", *self.compress_count.lock());
        println!("  Evictions: {}", *self.evict_count.lock());
        let warnings = self.warnings.lock();
        if !warnings.is_empty() {
            println!("  Warnings:");
            for w in warnings.iter() {
                println!("    - {}", w);
            }
        }
        println!("-------------------------\n");
    }
}

#[async_trait]
impl AgentHooks for MemoryMonitorHooks {
    async fn on_memory_compress(&self, event: &MemoryCompressEvent) {
        *self.compress_count.lock() += 1;
        println!(
            "\n[Memory] Compressed {} messages (ratio: {:.2})",
            event.messages_compressed, event.compression_ratio
        );
    }

    async fn on_memory_evict(&self, event: &MemoryEvictEvent) {
        *self.evict_count.lock() += 1;
        println!(
            "\n[Memory] Evicted {} messages (reason: {:?})",
            event.messages_evicted, event.reason
        );
    }

    async fn on_memory_budget_warning(&self, event: &MemoryBudgetEvent) {
        let msg = format!(
            "{}: {:.1}% ({}/{} tokens)",
            event.component, event.usage_percent, event.used_tokens, event.budget_tokens
        );
        self.warnings.lock().push(msg.clone());
        println!("\n[Warning] Memory budget: {}", msg);
    }

    async fn on_response(&self, response: &AgentResponse) {
        let preview = if response.content.len() > 100 {
            format!("{}...", &response.content[..100])
        } else {
            response.content.clone()
        };
        println!("[Agent] {}", preview);
    }
}

fn handle_command(
    input: &str,
    agent: &RuntimeAgent,
    hooks: &MemoryMonitorHooks,
) -> CommandResult {
    match input.to_lowercase().as_str() {
        "/stats" => {
            hooks.print_stats();
            CommandResult::Handled
        }
        "/history" | "/hist" => {
            let snapshot = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(agent.save_state())
            });
            match snapshot {
                Ok(snapshot) => {
                    println!("\n--- Conversation History ---");

                    if let Some(ref summary) = snapshot.memory.summary {
                        println!("\n[Summary of previous messages]");
                        println!("{}", summary);
                        println!();
                    }

                    if snapshot.memory.messages.is_empty() {
                        println!("(No messages in memory)");
                    } else {
                        println!("Recent messages ({}):", snapshot.memory.messages.len());
                        for (i, msg) in snapshot.memory.messages.iter().enumerate() {
                            let role = format!("{:?}", msg.role);
                            let content = if msg.content.len() > 80 {
                                format!("{}...", &msg.content[..80])
                            } else {
                                msg.content.clone()
                            };
                            println!("  {}. [{}] {}", i + 1, role, content);
                        }
                    }
                    println!("----------------------------\n");
                }
                Err(e) => {
                    eprintln!("Error getting history: {}\n", e);
                }
            }
            CommandResult::Handled
        }
        "/fill" => {
            println!("Adding test messages...");
            let topics = [
                "Tell me about the weather",
                "What's 2 + 2?",
                "Who wrote Romeo and Juliet?",
                "What is the capital of France?",
                "How do computers work?",
                "What is machine learning?",
                "Explain quantum physics",
                "What is the meaning of life?",
            ];
            tokio::task::block_in_place(|| {
                let handle = tokio::runtime::Handle::current();
                for topic in topics {
                    println!("  Asking: {}", topic);
                    let _ = handle.block_on(agent.chat(topic));
                }
            });
            println!("Done! Memory should have compressed.\n");
            hooks.print_stats();
            CommandResult::Handled
        }
        _ => CommandResult::NotHandled,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let hooks = Arc::new(MemoryMonitorHooks::new());

    let agent = AgentBuilder::from_template("memory_agent")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .hooks(hooks.clone())
        .build()?;

    let hooks_cmd = hooks.clone();
    let hooks_quit = hooks.clone();

    Repl::new(agent)
        .welcome("=== Memory Agent Demo ===\n\nCompactingMemory with auto-summarization.")
        .hint("Memory type: compacting | Compress threshold: 8 messages | Token budget: 4096")
        .hint("/stats   - Show memory statistics")
        .hint("/history - Show conversation history (messages + summary)")
        .hint("/fill    - Add test messages to trigger compression")
        .hint("Try having a long conversation to see compression in action!")
        .on_command(move |input, agent| handle_command(input, agent, &hooks_cmd))
        .on_quit(move |_agent| hooks_quit.print_stats())
        .run()
        .await
}
