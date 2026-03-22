// Schema Tool - a more complex tool with multi-field input and typed output.
//
// Both simple-tool and this example use schemars for input schema and JSON for output.
// This one shows how the pattern scales to real tools with multiple input fields, multi-branch logic, and richer output.
//
// Run: cd examples/rust/custom-tools && cargo run --bin schema-tool

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
struct ConvertInput {
    /// The numeric value to convert
    value: f64,
    /// Source unit (e.g., "km", "miles", "celsius", "fahrenheit", "kg", "lbs")
    from: String,
    /// Target unit (e.g., "km", "miles", "celsius", "fahrenheit", "kg", "lbs")
    to: String,
}

#[derive(Debug, Serialize)]
struct ConvertOutput {
    input_value: f64,
    from: String,
    to: String,
    result: f64,
}

struct UnitConverterTool;

#[async_trait]
impl Tool for UnitConverterTool {
    fn id(&self) -> &str {
        "unit_converter"
    }
    fn name(&self) -> &str {
        "Unit Converter"
    }
    fn description(&self) -> &str {
        "Convert between units of measurement: \
         distance (km/miles), temperature (celsius/fahrenheit), weight (kg/lbs)."
    }

    fn input_schema(&self) -> Value { generate_schema::<ConvertInput>() }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: ConvertInput = match serde_json::from_value(args) {
            Ok(i) => i,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let result = match (
            input.from.to_lowercase().as_str(),
            input.to.to_lowercase().as_str(),
        ) {
            ("km", "miles") => input.value * 0.621371,
            ("miles", "km") => input.value * 1.60934,
            ("celsius", "fahrenheit") | ("c", "f") => input.value * 9.0 / 5.0 + 32.0,
            ("fahrenheit", "celsius") | ("f", "c") => (input.value - 32.0) * 5.0 / 9.0,
            ("kg", "lbs") => input.value * 2.20462,
            ("lbs", "kg") => input.value * 0.453592,
            _ => {
                return ToolResult::error(format!(
                    "Unknown conversion: {} -> {}. Supported: km/miles, celsius/fahrenheit, kg/lbs",
                    input.from, input.to
                ))
            }
        };

        let output = ConvertOutput {
            input_value: input.value,
            from: input.from,
            to: input.to,
            result,
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

    let llm = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-nano")?;

    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a helpful assistant with unit conversion capabilities. \
             Use the unit converter tool for any conversion request.",
        )
        .llm(Arc::new(llm))
        .auto_configure_features()?
        .tool(Arc::new(UnitConverterTool))
        .build()?;

    CliRepl::new(agent)
        .welcome("=== Schema Tool Demo ===\n\nUses schemars to auto-generate input schemas.")
        .show_tool_calls()
        .hint("Try: Convert 100 km to miles")
        .hint("Try: What is 72F in celsius?")
        .hint("Try: How many lbs is 80 kg?")
        .run()
        .await
}
