use async_trait::async_trait;
use rand::{Rng, seq::SliceRandom};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::generate_schema;
use ai_agents_core::{Tool, ToolResult};

pub struct RandomTool;

impl RandomTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RandomTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RandomInput {
    /// Operation to perform: uuid, number, integer, choice, shuffle, bool, string
    operation: String,
    /// Minimum value (for number/integer operations)
    #[serde(default)]
    min: Option<f64>,
    /// Maximum value (for number/integer operations)
    #[serde(default)]
    max: Option<f64>,
    /// Items to choose from or shuffle (for choice/shuffle operations)
    #[serde(default)]
    items: Option<Vec<Value>>,
    /// Number of items to choose (for choice operation, default: 1)
    #[serde(default)]
    count: Option<usize>,
    /// Length of random string (for string operation, default: 16)
    #[serde(default)]
    length: Option<usize>,
    /// Character set for string: alphanumeric, alpha, numeric, hex (default: alphanumeric)
    #[serde(default)]
    charset: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UuidOutput {
    uuid: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct NumberOutput {
    value: f64,
    min: f64,
    max: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct IntegerOutput {
    value: i64,
    min: i64,
    max: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChoiceOutput {
    selected: Vec<Value>,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShuffleOutput {
    shuffled: Vec<Value>,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct BoolOutput {
    value: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct StringOutput {
    value: String,
    length: usize,
}

#[async_trait]
impl Tool for RandomTool {
    fn id(&self) -> &str {
        "random"
    }

    fn name(&self) -> &str {
        "Random Generator"
    }

    fn description(&self) -> &str {
        "Generate random values. Operations: uuid (generate UUID v4), number (random float), integer (random int), choice (pick from list), shuffle (randomize list order), bool (random true/false), string (random string)."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<RandomInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: RandomInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.to_lowercase().as_str() {
            "uuid" => self.handle_uuid(),
            "number" => self.handle_number(&input),
            "integer" | "int" => self.handle_integer(&input),
            "choice" | "choose" | "pick" => self.handle_choice(&input),
            "shuffle" => self.handle_shuffle(&input),
            "bool" | "boolean" => self.handle_bool(),
            "string" | "str" => self.handle_string(&input),
            _ => ToolResult::error(format!(
                "Unknown operation: {}. Valid operations: uuid, number, integer, choice, shuffle, bool, string",
                input.operation
            )),
        }
    }
}

impl RandomTool {
    fn handle_uuid(&self) -> ToolResult {
        let output = UuidOutput {
            uuid: uuid::Uuid::new_v4().to_string(),
        };
        self.to_result(&output)
    }

    fn handle_number(&self, input: &RandomInput) -> ToolResult {
        let min = input.min.unwrap_or(0.0);
        let max = input.max.unwrap_or(1.0);

        if min >= max {
            return ToolResult::error("'min' must be less than 'max'");
        }

        let mut rng = rand::thread_rng();
        let value: f64 = rng.gen_range(min..max);

        let output = NumberOutput { value, min, max };
        self.to_result(&output)
    }

    fn handle_integer(&self, input: &RandomInput) -> ToolResult {
        let min = input.min.unwrap_or(0.0) as i64;
        let max = input.max.unwrap_or(100.0) as i64;

        if min >= max {
            return ToolResult::error("'min' must be less than 'max'");
        }

        let mut rng = rand::thread_rng();
        let value: i64 = rng.gen_range(min..=max);

        let output = IntegerOutput { value, min, max };
        self.to_result(&output)
    }

    fn handle_choice(&self, input: &RandomInput) -> ToolResult {
        let items = match &input.items {
            Some(i) if !i.is_empty() => i,
            Some(_) => return ToolResult::error("'items' cannot be empty"),
            None => return ToolResult::error("'items' is required for choice operation"),
        };

        let count = input.count.unwrap_or(1).min(items.len());

        let mut rng = rand::thread_rng();
        let selected: Vec<Value> = items.choose_multiple(&mut rng, count).cloned().collect();

        let output = ChoiceOutput {
            count: selected.len(),
            selected,
        };
        self.to_result(&output)
    }

    fn handle_shuffle(&self, input: &RandomInput) -> ToolResult {
        let items = match &input.items {
            Some(i) => i.clone(),
            None => return ToolResult::error("'items' is required for shuffle operation"),
        };

        let mut shuffled = items;
        let mut rng = rand::thread_rng();
        shuffled.shuffle(&mut rng);

        let output = ShuffleOutput {
            count: shuffled.len(),
            shuffled,
        };
        self.to_result(&output)
    }

    fn handle_bool(&self) -> ToolResult {
        let mut rng = rand::thread_rng();
        let output = BoolOutput { value: rng.r#gen() };
        self.to_result(&output)
    }

    fn handle_string(&self, input: &RandomInput) -> ToolResult {
        let length = input.length.unwrap_or(16);
        let charset = input.charset.as_deref().unwrap_or("alphanumeric");

        let chars: Vec<char> = match charset.to_lowercase().as_str() {
            "alphanumeric" | "alnum" => {
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                    .chars()
                    .collect()
            }
            "alpha" | "letters" => "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                .chars()
                .collect(),
            "numeric" | "digits" | "numbers" => "0123456789".chars().collect(),
            "hex" | "hexadecimal" => "0123456789abcdef".chars().collect(),
            "lower" | "lowercase" => "abcdefghijklmnopqrstuvwxyz".chars().collect(),
            "upper" | "uppercase" => "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect(),
            _ => {
                return ToolResult::error(format!(
                    "Unknown charset: {}. Valid: alphanumeric, alpha, numeric, hex, lower, upper",
                    charset
                ));
            }
        };

        let mut rng = rand::thread_rng();
        let value: String = (0..length)
            .map(|_| chars[rng.gen_range(0..chars.len())])
            .collect();

        let output = StringOutput { value, length };
        self.to_result(&output)
    }

    fn to_result<T: Serialize>(&self, output: &T) -> ToolResult {
        match serde_json::to_string(output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_uuid() {
        let tool = RandomTool::new();
        let result = tool.execute(serde_json::json!({"operation": "uuid"})).await;
        assert!(result.success);

        let output: UuidOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.uuid.len(), 36);
        assert!(output.uuid.contains('-'));
    }

    #[tokio::test]
    async fn test_number() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "number",
                "min": 10.0,
                "max": 20.0
            }))
            .await;
        assert!(result.success);

        let output: NumberOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.value >= 10.0 && output.value < 20.0);
    }

    #[tokio::test]
    async fn test_number_invalid_range() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "number",
                "min": 20.0,
                "max": 10.0
            }))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_integer() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "integer",
                "min": 1,
                "max": 10
            }))
            .await;
        assert!(result.success);

        let output: IntegerOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.value >= 1 && output.value <= 10);
    }

    #[tokio::test]
    async fn test_choice() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "choice",
                "items": ["a", "b", "c", "d"],
                "count": 2
            }))
            .await;
        assert!(result.success);

        let output: ChoiceOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.count, 2);
        assert_eq!(output.selected.len(), 2);
    }

    #[tokio::test]
    async fn test_choice_single() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "choice",
                "items": [1, 2, 3]
            }))
            .await;
        assert!(result.success);

        let output: ChoiceOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.count, 1);
    }

    #[tokio::test]
    async fn test_choice_empty() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "choice",
                "items": []
            }))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_shuffle() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "shuffle",
                "items": [1, 2, 3, 4, 5]
            }))
            .await;
        assert!(result.success);

        let output: ShuffleOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.count, 5);
    }

    #[tokio::test]
    async fn test_bool() {
        let tool = RandomTool::new();
        let result = tool.execute(serde_json::json!({"operation": "bool"})).await;
        assert!(result.success);

        let output: BoolOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.value == true || output.value == false);
    }

    #[tokio::test]
    async fn test_string_default() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "string"}))
            .await;
        assert!(result.success);

        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.length, 16);
        assert_eq!(output.value.len(), 16);
    }

    #[tokio::test]
    async fn test_string_hex() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "string",
                "length": 8,
                "charset": "hex"
            }))
            .await;
        assert!(result.success);

        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.length, 8);
        assert!(output.value.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn test_string_numeric() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "string",
                "length": 10,
                "charset": "numeric"
            }))
            .await;
        assert!(result.success);

        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.value.chars().all(|c| c.is_ascii_digit()));
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = RandomTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "invalid"}))
            .await;
        assert!(!result.success);
    }
}
