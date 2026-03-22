// YAML agent + Rust domain tool - the recommended production pattern.
//
// Agent behavior is defined in YAML (prompt, tools list, memory, CLI metadata).
// Domain-specific logic is implemented in Rust and injected at build time.
//
// The YAML references "lookup_order" in its tools list, but no built-in tool
// has that ID. The Rust binary registers it via .tool(Arc::new(OrderLookupTool))
// before .build(). Built-in tools (calculator, datetime) are auto-registered
// by .auto_configure_features().
//
// Key points:
// - YAML-first agent design with Rust tool injection
// - .tool() adds to, does not replace, auto-configured built-ins
// - AgentBuilder::from_yaml_file() + .auto_configure_llms() pattern
// - Mock data pattern for domain tools (replace with real DB/API calls)
//
// Builds on: stateful-tool (schema + state patterns)
// See also: rust/custom-hitl/ - same YAML+Rust pattern but focused on HITL
//
// Run: cd examples/rust/custom-tools && cargo run --bin yaml-custom-tool

use ai_agents::{AgentBuilder, Tool, ToolResult, Result};
use ai_agents::tools::generate_schema;
use ai_agents_cli::{CliRepl, init_tracing};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

// Domain-specific tool - implemented in Rust, referenced from YAML.
// In a real application, execute() would query a database or call an API.
// Here we return mock data so the example runs without external dependencies.

#[derive(Debug, Deserialize, JsonSchema)]
struct OrderLookupInput {
    /// Order ID to look up (e.g., "ORD-1234")
    order_id: String,
}

struct OrderLookupTool;

#[async_trait]
impl Tool for OrderLookupTool {
    fn id(&self) -> &str {
        "lookup_order"
    }
    fn name(&self) -> &str {
        "Order Lookup"
    }
    fn description(&self) -> &str {
        "Look up an order by its ID. Returns order status, items, and delivery estimate."
    }
    fn input_schema(&self) -> Value {
        generate_schema::<OrderLookupInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: OrderLookupInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Mock order database - replace with real data source in production.
        match input.order_id.as_str() {
            "ORD-1234" => ToolResult::ok(
                serde_json::json!({
                    "order_id": "ORD-1234",
                    "status": "shipped",
                    "items": ["Widget A x2", "Gadget B x1"],
                    "total": 59.97,
                    "estimated_delivery": "2025-03-15"
                })
                .to_string(),
            ),
            "ORD-5678" => ToolResult::ok(
                serde_json::json!({
                    "order_id": "ORD-5678",
                    "status": "processing",
                    "items": ["Premium Kit x1"],
                    "total": 129.99,
                    "estimated_delivery": "2025-03-20"
                })
                .to_string(),
            ),
            _ => ToolResult::ok(
                serde_json::json!({
                    "order_id": input.order_id,
                    "status": "not_found",
                    "message": "No order found with this ID"
                })
                .to_string(),
            ),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    // Order matters:
    //   1. from_yaml_file  - loads the spec (prompt, tool list, memory, metadata)
    //   2. auto_configure_llms - creates LLM clients from the spec's llm/llms config
    //   3. auto_configure_features - registers built-in tools (calculator, datetime, etc.)
    //      Must come BEFORE .tool() - it only creates the builtin registry when self.tools is None.
    //      If .tool() runs first, self.tools is already Some and builtins are skipped.
    //   4. .tool() - adds the custom tool into the existing registry
    //   5. .build() - validates that every tool declared in the YAML exists in the registry.
    //      If "lookup_order" were not registered here, build() would return an error naming the missing tool.
    let agent = AgentBuilder::from_yaml_file("agents/custom_tool_agent.yaml")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .tool(Arc::new(OrderLookupTool))
        .build()?;

    // CliRepl picks up welcome/hints/show_tools from the YAML metadata.cli block, so we only need .show_tool_calls() here for runtime tool-call display.
    CliRepl::new(agent)
        .show_tool_calls()
        .run()
        .await
}
