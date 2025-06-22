use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::{Tool, ToolResult, generate_schema};

pub struct EchoTool;

impl EchoTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EchoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoInput {
    /// The message to echo back
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EchoOutput {
    message: String,
    length: usize,
}

#[async_trait]
impl Tool for EchoTool {
    fn id(&self) -> &str {
        "echo"
    }

    fn name(&self) -> &str {
        "Echo"
    }

    fn description(&self) -> &str {
        "Echoes back the input message. Useful for testing."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<EchoInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: EchoInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let output = EchoOutput {
            length: input.message.len(),
            message: input.message,
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo() {
        let echo = EchoTool::new();
        let result = echo
            .execute(serde_json::json!({"message": "Hello, world!"}))
            .await;

        assert!(result.success);
        let output: EchoOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.message, "Hello, world!");
        assert_eq!(output.length, 13);
    }

    #[tokio::test]
    async fn test_invalid_input() {
        let echo = EchoTool::new();
        let result = echo
            .execute(serde_json::json!({"wrong_field": "test"}))
            .await;
        assert!(!result.success);
    }
}
