// Stateful tool
//
// interior mutability with RwLock.
//
// Tool::execute takes &self, not &mut self. Any state that changes across calls must use interior mutability.
// This example uses parking_lot::RwLock<HashMap>.
//
// Key points:
// - RwLock pattern for mutable state behind &self
// - Multi-operation tool (add, get, list, delete)
// - State that persists across turns within a session (resets on restart)
//
// For state that survives restarts, combine with the storage system
// (see rust/storage/).
//
// Run: cd examples/rust/custom-tools && cargo run --bin stateful-tool

use ai_agents::{AgentBuilder, ProviderType, Tool, ToolResult, UnifiedLLMProvider};
use ai_agents::tools::generate_schema;
use ai_agents_cli::{CliRepl, init_tracing};
use async_trait::async_trait;
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Deserialize, JsonSchema)]
struct NoteInput {
    /// Operation to perform: "add", "get", "list", "get_all", or "delete"
    operation: String,
    /// Note title (required for add, get, delete; ignored for list)
    #[serde(default)]
    title: Option<String>,
    /// Note content (required for add; ignored for other operations)
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct NoteOutput {
    operation: String,
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    titles: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<usize>,
}

fn note_ok(op: &str, title: Option<&str>) -> ToolResult {
    let output = NoteOutput {
        operation: op.to_string(),
        success: true,
        title: title.map(String::from),
        content: None,
        titles: None,
        count: None,
    };
    ToolResult::ok(serde_json::to_string(&output).unwrap())
}

struct NoteTool {
    notes: RwLock<HashMap<String, String>>,
}

impl NoteTool {
    fn new() -> Self {
        Self {
            notes: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl Tool for NoteTool {
    fn id(&self) -> &str { "notes" }
    fn name(&self) -> &str { "Notes" }
    fn description(&self) -> &str {
        "Store and retrieve personal notes. \
         Operations: add (save a note), get (retrieve by title), \
         list (show all titles), get_all (show all notes), delete (remove a note)."
    }
    fn input_schema(&self) -> Value { generate_schema::<NoteInput>() }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: NoteInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.as_str() {
            "add" => {
                let title = match input.title {
                    Some(t) if !t.is_empty() => t,
                    _ => return ToolResult::error("'title' is required for add"),
                };
                let content = input.content.unwrap_or_default();
                self.notes.write().insert(title.clone(), content);
                note_ok("add", Some(&title))
            }
            "get" => {
                let title = match input.title {
                    Some(t) if !t.is_empty() => t,
                    _ => return ToolResult::error("'title' is required for get"),
                };
                match self.notes.read().get(&title) {
                    Some(content) => {
                        let output = NoteOutput {
                            operation: "get".into(),
                            success: true,
                            title: Some(title),
                            content: Some(content.clone()),
                            titles: None,
                            count: None,
                        };
                        ToolResult::ok(serde_json::to_string(&output).unwrap())
                    }
                    None => ToolResult::error(format!("Note '{}' not found.", title)),
                }
            }
            "list" => {
                let notes = self.notes.read();
                let mut titles: Vec<String> = notes.keys().cloned().collect();
                titles.sort();
                let output = NoteOutput {
                    operation: "list".into(),
                    success: true,
                    title: None,
                    content: None,
                    count: Some(titles.len()),
                    titles: Some(titles),
                };
                ToolResult::ok(serde_json::to_string(&output).unwrap())
            }
            "get_all" => {
                let notes = self.notes.read();
                let mut entries: Vec<String> = notes
                    .iter()
                    .map(|(title, content)| format!("{}: {}", title, content))
                    .collect();
                entries.sort();
                let output = NoteOutput {
                    operation: "get_all".into(),
                    success: true,
                    title: None,
                    content: Some(entries.join("\n")),
                    count: Some(entries.len()),
                    titles: None,
                };
                ToolResult::ok(serde_json::to_string(&output).unwrap())
            }
            "delete" => {
                let title = match input.title {
                    Some(t) if !t.is_empty() => t,
                    _ => return ToolResult::error("'title' is required for delete"),
                };
                match self.notes.write().remove(&title) {
                    Some(_) => note_ok("delete", Some(&title)),
                    None => ToolResult::error(format!("Note '{}' not found.", title)),
                }
            }
            other => ToolResult::error(format!(
                "Unknown operation '{}'. Use: add, get, list, get_all, delete",
                other
            )),
        }
    }
}

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    init_tracing();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;

    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a personal assistant with a notes system. \
             Help the user manage their notes.",
        )
        .llm(Arc::new(llm))
        .auto_configure_features()?
        .tool(Arc::new(NoteTool::new()))
        .build()?;

    CliRepl::new(agent)
        .welcome("=== Stateful Tool Demo ===\n\nThis tool remembers notes across turns.")
        .show_tool_calls()
        .hint("Try: Remember to buy milk")
        .hint("Try: Save a note called 'meeting' with 'Team sync at 3pm'")
        .hint("Try: What are my notes?")
        .hint("Try: Delete the milk note")
        .run()
        .await
}
