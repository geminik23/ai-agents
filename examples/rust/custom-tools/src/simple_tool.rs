// Minimal custom tool
//
// teaches the `Tool` trait with the same patterns the framework's built-in tools use:
// schemars for input schema and serialized JSON for output.
//
// Key points:
// - Implementing `Tool` with id, name, description, input_schema, execute
// - `#[derive(JsonSchema)]` for input, `#[derive(Serialize)]` for output
// - `generate_schema::<T>()` helper (same as built-in tools)
// - Registering a custom tool via `.tool(Arc::new(...))`
// - `ToolResult::ok(...)` and `ToolResult::error(...)` return values
//
// The WordCountTool counts words in a string - simple enough to focus on the trait, not the business logic.
//
// Run: cd examples/rust/custom-tools && cargo run --bin simple-tool

use ai_agents::{
    AgentBuilder, ProviderType, Tool, ToolResult, UnifiedLLMProvider,
    tools::generate_schema,
};
use ai_agents_cli::{CliRepl, init_tracing};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Deserialize, JsonSchema)]
struct WordCountInput {
    /// The text to count words in
    text: String,
}

#[derive(Debug, Serialize)]
struct WordCountOutput {
    word_count: usize,
    text: String,
}

struct WordCountTool;

#[async_trait]
impl Tool for WordCountTool {
    fn id(&self) -> &str { "word_count" }
    fn name(&self) -> &str { "Word Counter" }

    // A clear description is the single most important factor for reliable
    // tool selection - it tells the LLM when and why to call this tool.
    fn description(&self) -> &str {
        "Count the number of words in a text string."
    }

    fn input_schema(&self) -> Value { generate_schema::<WordCountInput>() }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: WordCountInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let output = WordCountOutput {
            word_count: input.text.split_whitespace().count(),
            text: input.text,
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }
}

#[tokio::main]
async fn main() -> ai_agents::Result<()> {
    init_tracing();

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-mini")?;

    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a helpful assistant. \
             Use the word counter tool when asked about word counts.",
        )
        .llm(Arc::new(llm))
        // auto_configure_features registers built-in tools (calculator, datetime, etc.).
        .auto_configure_features()?
        // .tool() adds one custom tool on top - does NOT replace built-ins.
        .tool(Arc::new(WordCountTool))
        .build()?;

    CliRepl::new(agent)
        .welcome("=== Simple Custom Tool Demo ===")
        .show_tool_calls()
        .hint("Try: How many words in 'The quick brown fox jumps over the lazy dog'?")
        .hint("Try: Count words in 'hello world'")
        .run()
        .await
}
