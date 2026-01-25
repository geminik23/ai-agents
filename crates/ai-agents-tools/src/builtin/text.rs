use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::generate_schema;
use ai_agents_core::{Tool, ToolResult};

pub struct TextTool;

impl TextTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TextTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TextInput {
    /// Operation: length, substring, uppercase, lowercase, trim, trim_start, trim_end, replace, split, join, contains, starts_with, ends_with, repeat, reverse, pad_left, pad_right, truncate, lines, words, char_at, index_of
    operation: String,
    /// Input text
    #[serde(default)]
    text: Option<String>,
    /// Start index (for substring)
    #[serde(default)]
    start: Option<usize>,
    /// End index (for substring)
    #[serde(default)]
    end: Option<usize>,
    /// Text to find (for replace/contains/index_of)
    #[serde(default)]
    find: Option<String>,
    /// Replacement text
    #[serde(default)]
    replace_with: Option<String>,
    /// Delimiter (for split/join)
    #[serde(default)]
    delimiter: Option<String>,
    /// Items to join
    #[serde(default)]
    items: Option<Vec<String>>,
    /// Repeat count
    #[serde(default)]
    count: Option<usize>,
    /// Target width (for padding/truncate)
    #[serde(default)]
    width: Option<usize>,
    /// Padding character
    #[serde(default)]
    pad_char: Option<String>,
    /// Index position
    #[serde(default)]
    index: Option<usize>,
    /// Suffix for truncation
    #[serde(default)]
    suffix: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LengthOutput {
    length: usize,
    bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct StringOutput {
    result: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BoolOutput {
    result: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct SplitOutput {
    parts: Vec<String>,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct CharAtOutput {
    char: Option<String>,
    found: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexOfOutput {
    index: Option<usize>,
    found: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct LinesOutput {
    lines: Vec<String>,
    count: usize,
}

#[async_trait]
impl Tool for TextTool {
    fn id(&self) -> &str {
        "text"
    }

    fn name(&self) -> &str {
        "Text Manipulation"
    }

    fn description(&self) -> &str {
        "String operations: length (character count), substring, uppercase, lowercase, trim, trim_start, trim_end, replace, split, join, contains, starts_with, ends_with, repeat, reverse, pad_left, pad_right, truncate, lines, words, char_at, index_of. Works with all Unicode text."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<TextInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: TextInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.to_lowercase().as_str() {
            "length" | "len" => self.handle_length(&input),
            "substring" | "substr" | "slice" => self.handle_substring(&input),
            "uppercase" | "upper" => self.handle_uppercase(&input),
            "lowercase" | "lower" => self.handle_lowercase(&input),
            "trim" => self.handle_trim(&input),
            "trim_start" | "ltrim" => self.handle_trim_start(&input),
            "trim_end" | "rtrim" => self.handle_trim_end(&input),
            "replace" => self.handle_replace(&input),
            "split" => self.handle_split(&input),
            "join" => self.handle_join(&input),
            "contains" | "includes" => self.handle_contains(&input),
            "starts_with" => self.handle_starts_with(&input),
            "ends_with" => self.handle_ends_with(&input),
            "repeat" => self.handle_repeat(&input),
            "reverse" => self.handle_reverse(&input),
            "pad_left" | "lpad" => self.handle_pad_left(&input),
            "pad_right" | "rpad" => self.handle_pad_right(&input),
            "truncate" => self.handle_truncate(&input),
            "lines" => self.handle_lines(&input),
            "words" => self.handle_words(&input),
            "char_at" => self.handle_char_at(&input),
            "index_of" | "find" => self.handle_index_of(&input),
            _ => ToolResult::error(format!(
                "Unknown operation: {}. Valid: length, substring, uppercase, lowercase, trim, replace, split, join, contains, starts_with, ends_with, repeat, reverse, pad_left, pad_right, truncate, lines, words, char_at, index_of",
                input.operation
            )),
        }
    }
}

impl TextTool {
    fn handle_length(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let output = LengthOutput {
            length: text.chars().count(),
            bytes: text.len(),
        };
        self.to_result(&output)
    }

    fn handle_substring(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let chars: Vec<char> = text.chars().collect();
        let start = input.start.unwrap_or(0);
        let end = input.end.unwrap_or(chars.len());

        let start = start.min(chars.len());
        let end = end.min(chars.len());

        let result: String = chars[start..end].iter().collect();
        let output = StringOutput { result };
        self.to_result(&output)
    }

    fn handle_uppercase(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let output = StringOutput {
            result: text.to_uppercase(),
        };
        self.to_result(&output)
    }

    fn handle_lowercase(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let output = StringOutput {
            result: text.to_lowercase(),
        };
        self.to_result(&output)
    }

    fn handle_trim(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let output = StringOutput {
            result: text.trim().to_string(),
        };
        self.to_result(&output)
    }

    fn handle_trim_start(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let output = StringOutput {
            result: text.trim_start().to_string(),
        };
        self.to_result(&output)
    }

    fn handle_trim_end(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let output = StringOutput {
            result: text.trim_end().to_string(),
        };
        self.to_result(&output)
    }

    fn handle_replace(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let find = input.find.as_deref().unwrap_or("");
        let replace_with = input.replace_with.as_deref().unwrap_or("");

        let output = StringOutput {
            result: text.replace(find, replace_with),
        };
        self.to_result(&output)
    }

    fn handle_split(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let delimiter = input.delimiter.as_deref().unwrap_or(" ");

        let parts: Vec<String> = text.split(delimiter).map(|s| s.to_string()).collect();
        let output = SplitOutput {
            count: parts.len(),
            parts,
        };
        self.to_result(&output)
    }

    fn handle_join(&self, input: &TextInput) -> ToolResult {
        let items = input.items.as_deref().unwrap_or(&[]);
        let delimiter = input.delimiter.as_deref().unwrap_or("");

        let output = StringOutput {
            result: items.join(delimiter),
        };
        self.to_result(&output)
    }

    fn handle_contains(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let find = input.find.as_deref().unwrap_or("");

        let output = BoolOutput {
            result: text.contains(find),
        };
        self.to_result(&output)
    }

    fn handle_starts_with(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let find = input.find.as_deref().unwrap_or("");

        let output = BoolOutput {
            result: text.starts_with(find),
        };
        self.to_result(&output)
    }

    fn handle_ends_with(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let find = input.find.as_deref().unwrap_or("");

        let output = BoolOutput {
            result: text.ends_with(find),
        };
        self.to_result(&output)
    }

    fn handle_repeat(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let count = input.count.unwrap_or(1);

        let output = StringOutput {
            result: text.repeat(count),
        };
        self.to_result(&output)
    }

    fn handle_reverse(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");

        let output = StringOutput {
            result: text.chars().rev().collect(),
        };
        self.to_result(&output)
    }

    fn handle_pad_left(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let width = input.width.unwrap_or(0);
        let pad_char = input
            .pad_char
            .as_deref()
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');

        let char_count = text.chars().count();
        let result = if char_count >= width {
            text.to_string()
        } else {
            let padding: String = std::iter::repeat(pad_char)
                .take(width - char_count)
                .collect();
            format!("{}{}", padding, text)
        };

        let output = StringOutput { result };
        self.to_result(&output)
    }

    fn handle_pad_right(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let width = input.width.unwrap_or(0);
        let pad_char = input
            .pad_char
            .as_deref()
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');

        let char_count = text.chars().count();
        let result = if char_count >= width {
            text.to_string()
        } else {
            let padding: String = std::iter::repeat(pad_char)
                .take(width - char_count)
                .collect();
            format!("{}{}", text, padding)
        };

        let output = StringOutput { result };
        self.to_result(&output)
    }

    fn handle_truncate(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let width = input.width.unwrap_or(text.chars().count());
        let suffix = input.suffix.as_deref().unwrap_or("...");

        let chars: Vec<char> = text.chars().collect();
        let result = if chars.len() <= width {
            text.to_string()
        } else {
            let suffix_len = suffix.chars().count();
            if width <= suffix_len {
                chars[..width].iter().collect()
            } else {
                let truncated: String = chars[..(width - suffix_len)].iter().collect();
                format!("{}{}", truncated, suffix)
            }
        };

        let output = StringOutput { result };
        self.to_result(&output)
    }

    fn handle_lines(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();

        let output = LinesOutput {
            count: lines.len(),
            lines,
        };
        self.to_result(&output)
    }

    fn handle_words(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        let output = SplitOutput {
            count: words.len(),
            parts: words,
        };
        self.to_result(&output)
    }

    fn handle_char_at(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let index = input.index.unwrap_or(0);

        let chars: Vec<char> = text.chars().collect();
        let output = if index < chars.len() {
            CharAtOutput {
                char: Some(chars[index].to_string()),
                found: true,
            }
        } else {
            CharAtOutput {
                char: None,
                found: false,
            }
        };
        self.to_result(&output)
    }

    fn handle_index_of(&self, input: &TextInput) -> ToolResult {
        let text = input.text.as_deref().unwrap_or("");
        let find = input.find.as_deref().unwrap_or("");

        let output = match text.find(find) {
            Some(byte_index) => {
                let char_index = text[..byte_index].chars().count();
                IndexOfOutput {
                    index: Some(char_index),
                    found: true,
                }
            }
            None => IndexOfOutput {
                index: None,
                found: false,
            },
        };
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
    async fn test_length_unicode() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "length",
                "text": "안녕하세요"
            }))
            .await;
        assert!(result.success);
        let output: LengthOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.length, 5);
    }

    #[tokio::test]
    async fn test_substring() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "substring",
                "text": "hello world",
                "start": 0,
                "end": 5
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "hello");
    }

    #[tokio::test]
    async fn test_uppercase_lowercase() {
        let tool = TextTool::new();

        let result = tool
            .execute(serde_json::json!({
                "operation": "uppercase",
                "text": "hello"
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("HELLO"));

        let result = tool
            .execute(serde_json::json!({
                "operation": "lowercase",
                "text": "HELLO"
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn test_trim() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "trim",
                "text": "  hello  "
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "hello");
    }

    #[tokio::test]
    async fn test_split_join() {
        let tool = TextTool::new();

        let result = tool
            .execute(serde_json::json!({
                "operation": "split",
                "text": "a,b,c",
                "delimiter": ","
            }))
            .await;
        assert!(result.success);
        let output: SplitOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.parts, vec!["a", "b", "c"]);
        assert_eq!(output.count, 3);

        let result = tool
            .execute(serde_json::json!({
                "operation": "join",
                "items": ["a", "b", "c"],
                "delimiter": "-"
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "a-b-c");
    }

    #[tokio::test]
    async fn test_replace() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "replace",
                "text": "hello world",
                "find": "world",
                "replace_with": "rust"
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "hello rust");
    }

    #[tokio::test]
    async fn test_contains() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "contains",
                "text": "hello world",
                "find": "world"
            }))
            .await;
        assert!(result.success);
        let output: BoolOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.result);
    }

    #[tokio::test]
    async fn test_repeat() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "repeat",
                "text": "ab",
                "count": 3
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "ababab");
    }

    #[tokio::test]
    async fn test_reverse() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "reverse",
                "text": "hello"
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "olleh");
    }

    #[tokio::test]
    async fn test_pad() {
        let tool = TextTool::new();

        let result = tool
            .execute(serde_json::json!({
                "operation": "pad_left",
                "text": "5",
                "width": 3,
                "pad_char": "0"
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "005");

        let result = tool
            .execute(serde_json::json!({
                "operation": "pad_right",
                "text": "hi",
                "width": 5
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "hi   ");
    }

    #[tokio::test]
    async fn test_truncate() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "truncate",
                "text": "hello world",
                "width": 8
            }))
            .await;
        assert!(result.success);
        let output: StringOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.result, "hello...");
    }

    #[tokio::test]
    async fn test_lines() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "lines",
                "text": "line1\nline2\nline3"
            }))
            .await;
        assert!(result.success);
        let output: LinesOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.count, 3);
    }

    #[tokio::test]
    async fn test_words() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "words",
                "text": "hello  world   test"
            }))
            .await;
        assert!(result.success);
        let output: SplitOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.count, 3);
        assert_eq!(output.parts, vec!["hello", "world", "test"]);
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = TextTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "invalid"
            }))
            .await;
        assert!(!result.success);
    }
}
