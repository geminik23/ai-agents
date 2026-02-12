use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::generate_schema;
use ai_agents_core::{Tool, ToolResult};

pub struct JsonTool;

impl JsonTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for JsonTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct JsonInput {
    /// Operation to perform: parse, get, set, merge, stringify, keys, values
    operation: String,
    /// JSON string or data to operate on
    #[serde(default)]
    data: Option<Value>,
    /// Path using dot notation (e.g., 'user.name', 'items.0.id')
    #[serde(default)]
    path: Option<String>,
    /// Value to set (for set operation)
    #[serde(default)]
    value: Option<Value>,
    /// Second JSON data (for merge operation)
    #[serde(default)]
    data2: Option<Value>,
    /// Pretty print output (default: true)
    #[serde(default)]
    pretty: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParseOutput {
    parsed: Value,
    valid: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct GetOutput {
    value: Value,
    path: String,
    found: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SetOutput {
    result: Value,
    path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct MergeOutput {
    result: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct StringifyOutput {
    result: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct KeysOutput {
    keys: Vec<String>,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ValuesOutput {
    values: Vec<Value>,
    count: usize,
}

#[async_trait]
impl Tool for JsonTool {
    fn id(&self) -> &str {
        "json"
    }

    fn name(&self) -> &str {
        "JSON Manipulation"
    }

    fn description(&self) -> &str {
        "Parse, query, and manipulate JSON data. Operations: parse (validate JSON string), get (extract value by path), set (set value at path), merge (combine two JSON objects), stringify (convert to string), keys (get object keys), values (get object values)."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<JsonInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: JsonInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.to_lowercase().as_str() {
            "parse" => self.handle_parse(&input),
            "get" => self.handle_get(&input),
            "set" => self.handle_set(&input),
            "merge" => self.handle_merge(&input),
            "stringify" => self.handle_stringify(&input),
            "keys" => self.handle_keys(&input),
            "values" => self.handle_values(&input),
            _ => ToolResult::error(format!(
                "Unknown operation: {}. Valid operations: parse, get, set, merge, stringify, keys, values",
                input.operation
            )),
        }
    }
}

impl JsonTool {
    fn handle_parse(&self, input: &JsonInput) -> ToolResult {
        match &input.data {
            Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                Ok(parsed) => {
                    let output = ParseOutput {
                        parsed,
                        valid: true,
                    };
                    self.to_result(&output)
                }
                Err(e) => ToolResult::error(format!("Parse error: {}", e)),
            },
            Some(v) => {
                let output = ParseOutput {
                    parsed: v.clone(),
                    valid: true,
                };
                self.to_result(&output)
            }
            None => ToolResult::error("'data' is required for parse operation"),
        }
    }

    fn handle_get(&self, input: &JsonInput) -> ToolResult {
        let data = match &input.data {
            Some(d) => d,
            None => return ToolResult::error("'data' is required for get operation"),
        };

        let path = match &input.path {
            Some(p) => p,
            None => return ToolResult::error("'path' is required for get operation"),
        };

        let value = self.get_by_path(data, path);
        let found = !value.is_null();

        let output = GetOutput {
            value,
            path: path.clone(),
            found,
        };

        self.to_result(&output)
    }

    fn handle_set(&self, input: &JsonInput) -> ToolResult {
        let data = match &input.data {
            Some(d) => d.clone(),
            None => return ToolResult::error("'data' is required for set operation"),
        };

        let path = match &input.path {
            Some(p) => p,
            None => return ToolResult::error("'path' is required for set operation"),
        };

        let value = match &input.value {
            Some(v) => v.clone(),
            None => return ToolResult::error("'value' is required for set operation"),
        };

        let result = match self.set_by_path(data, path, value) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };

        let output = SetOutput {
            result,
            path: path.clone(),
        };

        self.to_result(&output)
    }

    fn handle_merge(&self, input: &JsonInput) -> ToolResult {
        let data1 = match &input.data {
            Some(d) => d.clone(),
            None => return ToolResult::error("'data' is required for merge operation"),
        };

        let data2 = match &input.data2 {
            Some(d) => d.clone(),
            None => return ToolResult::error("'data2' is required for merge operation"),
        };

        let result = self.merge_values(data1, data2);

        let output = MergeOutput { result };

        self.to_result(&output)
    }

    fn handle_stringify(&self, input: &JsonInput) -> ToolResult {
        let data = match &input.data {
            Some(d) => d,
            None => return ToolResult::error("'data' is required for stringify operation"),
        };

        let pretty = input.pretty.unwrap_or(true);

        let result = if pretty {
            serde_json::to_string_pretty(data).unwrap_or_default()
        } else {
            serde_json::to_string(data).unwrap_or_default()
        };

        let output = StringifyOutput { result };

        self.to_result(&output)
    }

    fn handle_keys(&self, input: &JsonInput) -> ToolResult {
        let data = match &input.data {
            Some(d) => d,
            None => return ToolResult::error("'data' is required for keys operation"),
        };

        let keys: Vec<String> = match data {
            Value::Object(obj) => obj.keys().cloned().collect(),
            _ => return ToolResult::error("'data' must be a JSON object for keys operation"),
        };

        let count = keys.len();
        let output = KeysOutput { keys, count };

        self.to_result(&output)
    }

    fn handle_values(&self, input: &JsonInput) -> ToolResult {
        let data = match &input.data {
            Some(d) => d,
            None => return ToolResult::error("'data' is required for values operation"),
        };

        let values: Vec<Value> = match data {
            Value::Object(obj) => obj.values().cloned().collect(),
            Value::Array(arr) => arr.clone(),
            _ => {
                return ToolResult::error(
                    "'data' must be a JSON object or array for values operation",
                );
            }
        };

        let count = values.len();
        let output = ValuesOutput { values, count };

        self.to_result(&output)
    }

    fn get_by_path(&self, data: &Value, path: &str) -> Value {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = data;

        for part in parts {
            if part.is_empty() {
                continue;
            }

            if let Ok(index) = part.parse::<usize>() {
                match current.get(index) {
                    Some(v) => current = v,
                    None => return Value::Null,
                }
            } else {
                match current.get(part) {
                    Some(v) => current = v,
                    None => return Value::Null,
                }
            }
        }

        current.clone()
    }

    fn set_by_path(&self, mut data: Value, path: &str, value: Value) -> Result<Value, String> {
        let parts: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return Ok(value);
        }

        let mut current = &mut data;

        for (i, part) in parts.iter().enumerate() {
            let is_last = i == parts.len() - 1;

            if let Ok(index) = part.parse::<usize>() {
                if !current.is_array() {
                    *current = Value::Array(vec![]);
                }

                let arr = current.as_array_mut().unwrap();
                while arr.len() <= index {
                    arr.push(Value::Null);
                }

                if is_last {
                    arr[index] = value.clone();
                    break;
                } else {
                    current = &mut arr[index];
                }
            } else {
                if !current.is_object() {
                    *current = Value::Object(serde_json::Map::new());
                }

                let obj = current.as_object_mut().unwrap();

                if is_last {
                    obj.insert(part.to_string(), value.clone());
                    break;
                } else {
                    if !obj.contains_key(*part) {
                        obj.insert(part.to_string(), Value::Object(serde_json::Map::new()));
                    }
                    current = obj.get_mut(*part).unwrap();
                }
            }
        }

        Ok(data)
    }

    fn merge_values(&self, base: Value, overlay: Value) -> Value {
        match (base, overlay) {
            (Value::Object(mut base_obj), Value::Object(overlay_obj)) => {
                for (key, value) in overlay_obj {
                    let merged = if let Some(base_value) = base_obj.remove(&key) {
                        self.merge_values(base_value, value)
                    } else {
                        value
                    };
                    base_obj.insert(key, merged);
                }
                Value::Object(base_obj)
            }
            (_, overlay) => overlay,
        }
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
    async fn test_parse_string() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "parse",
                "data": "{\"name\": \"test\", \"value\": 42}"
            }))
            .await;
        assert!(result.success);

        let output: ParseOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.valid);
        assert_eq!(output.parsed["name"], "test");
        assert_eq!(output.parsed["value"], 42);
    }

    #[tokio::test]
    async fn test_parse_invalid() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "parse",
                "data": "{invalid json}"
            }))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_get_simple() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "get",
                "data": {"user": {"name": "Alice", "age": 30}},
                "path": "user.name"
            }))
            .await;
        assert!(result.success);

        let output: GetOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.found);
        assert_eq!(output.value, "Alice");
    }

    #[tokio::test]
    async fn test_get_array_index() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "get",
                "data": {"items": ["a", "b", "c"]},
                "path": "items.1"
            }))
            .await;
        assert!(result.success);

        let output: GetOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.found);
        assert_eq!(output.value, "b");
    }

    #[tokio::test]
    async fn test_get_not_found() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "get",
                "data": {"a": 1},
                "path": "b.c.d"
            }))
            .await;
        assert!(result.success);

        let output: GetOutput = serde_json::from_str(&result.output).unwrap();
        assert!(!output.found);
    }

    #[tokio::test]
    async fn test_set_simple() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "set",
                "data": {"user": {"name": "Alice"}},
                "path": "user.age",
                "value": 30
            }))
            .await;
        assert!(result.success);

        let output: SetOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result["user"]["age"], 30);
        assert_eq!(output.result["user"]["name"], "Alice");
    }

    #[tokio::test]
    async fn test_set_nested() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "set",
                "data": {},
                "path": "a.b.c",
                "value": "deep"
            }))
            .await;
        assert!(result.success);

        let output: SetOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result["a"]["b"]["c"], "deep");
    }

    #[tokio::test]
    async fn test_merge() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "merge",
                "data": {"a": 1, "b": {"x": 10}},
                "data2": {"b": {"y": 20}, "c": 3}
            }))
            .await;
        assert!(result.success);

        let output: MergeOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result["a"], 1);
        assert_eq!(output.result["b"]["x"], 10);
        assert_eq!(output.result["b"]["y"], 20);
        assert_eq!(output.result["c"], 3);
    }

    #[tokio::test]
    async fn test_stringify() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "stringify",
                "data": {"name": "test"},
                "pretty": false
            }))
            .await;
        assert!(result.success);

        let output: StringifyOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "{\"name\":\"test\"}");
    }

    #[tokio::test]
    async fn test_keys() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "keys",
                "data": {"a": 1, "b": 2, "c": 3}
            }))
            .await;
        assert!(result.success);

        let output: KeysOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.count, 3);
        assert!(output.keys.contains(&"a".to_string()));
        assert!(output.keys.contains(&"b".to_string()));
        assert!(output.keys.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn test_values() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "values",
                "data": [1, 2, 3]
            }))
            .await;
        assert!(result.success);

        let output: ValuesOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.count, 3);
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = JsonTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "invalid"}))
            .await;
        assert!(!result.success);
    }
}
