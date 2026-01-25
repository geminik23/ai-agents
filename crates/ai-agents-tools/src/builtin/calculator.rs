use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::generate_schema;
use ai_agents_core::{Tool, ToolResult};

/// This Tool is for development purposes only.
/// It evaluates mathematical expressions.
/// It is not intended for production use.
pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CalculatorInput {
    /// Mathematical expression to evaluate (e.g., '2 + 3 * 4')
    expression: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CalculatorOutput {
    result: f64,
    expression: String,
}

#[async_trait]
impl Tool for CalculatorTool {
    fn id(&self) -> &str {
        "calculator"
    }

    fn name(&self) -> &str {
        "Calculator"
    }

    fn description(&self) -> &str {
        "Evaluates mathematical expressions. Supports +, -, *, /, ^ and parentheses."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<CalculatorInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: CalculatorInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match evalexpr::eval(&input.expression) {
            Ok(value) => {
                let result = match value {
                    evalexpr::Value::Float(f) => f,
                    evalexpr::Value::Int(i) => i as f64,
                    _ => return ToolResult::error("Expression must evaluate to a number"),
                };

                let output = CalculatorOutput {
                    result,
                    expression: input.expression,
                };

                match serde_json::to_string(&output) {
                    Ok(json) => ToolResult::ok(json),
                    Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
                }
            }
            Err(e) => ToolResult::error(format!("Calculation error: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_operations() {
        let calc = CalculatorTool::new();

        let result = calc
            .execute(serde_json::json!({"expression": "2 + 3"}))
            .await;
        assert!(result.success);

        let result = calc
            .execute(serde_json::json!({"expression": "10 * 5"}))
            .await;
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_operator_precedence() {
        let calc = CalculatorTool::new();
        let result = calc
            .execute(serde_json::json!({"expression": "2 + 3 * 4"}))
            .await;
        assert!(result.success);

        let output: CalculatorOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, 14.0);
    }

    #[tokio::test]
    async fn test_parentheses() {
        let calc = CalculatorTool::new();
        let result = calc
            .execute(serde_json::json!({"expression": "(2 + 3) * 4"}))
            .await;
        assert!(result.success);

        let output: CalculatorOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, 20.0);
    }

    #[tokio::test]
    async fn test_invalid_expression() {
        let calc = CalculatorTool::new();
        let result = calc.execute(serde_json::json!({"expression": "2 +"})).await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_invalid_input() {
        let calc = CalculatorTool::new();
        let result = calc
            .execute(serde_json::json!({"wrong_field": "test"}))
            .await;
        assert!(!result.success);
    }
}
