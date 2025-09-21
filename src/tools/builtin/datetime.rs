use async_trait::async_trait;
use chrono::{DateTime, Duration, NaiveDateTime, TimeZone, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::{Tool, ToolResult, generate_schema};

pub struct DateTimeTool;

impl DateTimeTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DateTimeTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DateTimeInput {
    /// Operation to perform: now, format, parse, add, diff
    operation: String,
    /// Date/time value (ISO 8601 format for most operations)
    #[serde(default)]
    value: Option<String>,
    /// Second date/time value (for diff operation)
    #[serde(default)]
    value2: Option<String>,
    /// Format string using strftime syntax (e.g., '%Y-%m-%d %H:%M:%S')
    #[serde(default)]
    format: Option<String>,
    /// Amount to add (for add operation)
    #[serde(default)]
    amount: Option<i64>,
    /// Unit for add operation: seconds, minutes, hours, days, weeks
    #[serde(default)]
    unit: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct NowOutput {
    iso: String,
    unix_timestamp: i64,
    formatted: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct FormatOutput {
    formatted: String,
    original: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParseOutput {
    iso: String,
    unix_timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct AddOutput {
    result: String,
    unix_timestamp: i64,
    original: String,
    added: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiffOutput {
    seconds: i64,
    minutes: i64,
    hours: i64,
    days: i64,
    value1: String,
    value2: String,
}

#[async_trait]
impl Tool for DateTimeTool {
    fn id(&self) -> &str {
        "datetime"
    }

    fn name(&self) -> &str {
        "DateTime"
    }

    fn description(&self) -> &str {
        "Get current time, format dates, parse date strings, add/subtract time, and calculate differences. All times are in UTC."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<DateTimeInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: DateTimeInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.to_lowercase().as_str() {
            "now" => self.handle_now(&input),
            "format" => self.handle_format(&input),
            "parse" => self.handle_parse(&input),
            "add" => self.handle_add(&input),
            "diff" => self.handle_diff(&input),
            _ => ToolResult::error(format!(
                "Unknown operation: {}. Valid operations: now, format, parse, add, diff",
                input.operation
            )),
        }
    }
}

impl DateTimeTool {
    fn handle_now(&self, input: &DateTimeInput) -> ToolResult {
        let now = Utc::now();
        let format = input.format.as_deref().unwrap_or("%Y-%m-%d %H:%M:%S UTC");

        let output = NowOutput {
            iso: now.to_rfc3339(),
            unix_timestamp: now.timestamp(),
            formatted: now.format(format).to_string(),
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }

    fn handle_format(&self, input: &DateTimeInput) -> ToolResult {
        let value = match &input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for format operation"),
        };

        let format = input.format.as_deref().unwrap_or("%Y-%m-%d");

        let dt = match self.parse_datetime(value) {
            Ok(dt) => dt,
            Err(e) => return ToolResult::error(e),
        };

        let output = FormatOutput {
            formatted: dt.format(format).to_string(),
            original: value.clone(),
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }

    fn handle_parse(&self, input: &DateTimeInput) -> ToolResult {
        let value = match &input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for parse operation"),
        };

        let dt = match self.parse_datetime(value) {
            Ok(dt) => dt,
            Err(e) => return ToolResult::error(e),
        };

        let output = ParseOutput {
            iso: dt.to_rfc3339(),
            unix_timestamp: dt.timestamp(),
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }

    fn handle_add(&self, input: &DateTimeInput) -> ToolResult {
        let value = match &input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for add operation"),
        };

        let amount = match input.amount {
            Some(a) => a,
            None => return ToolResult::error("'amount' is required for add operation"),
        };

        let unit = input.unit.as_deref().unwrap_or("days");

        let dt = match self.parse_datetime(value) {
            Ok(dt) => dt,
            Err(e) => return ToolResult::error(e),
        };

        let duration = match unit.to_lowercase().as_str() {
            "seconds" | "second" | "s" => Duration::seconds(amount),
            "minutes" | "minute" | "m" => Duration::minutes(amount),
            "hours" | "hour" | "h" => Duration::hours(amount),
            "days" | "day" | "d" => Duration::days(amount),
            "weeks" | "week" | "w" => Duration::weeks(amount),
            _ => {
                return ToolResult::error(format!(
                    "Unknown unit: {}. Valid units: seconds, minutes, hours, days, weeks",
                    unit
                ));
            }
        };

        let result = dt + duration;

        let output = AddOutput {
            result: result.to_rfc3339(),
            unix_timestamp: result.timestamp(),
            original: value.clone(),
            added: format!("{} {}", amount, unit),
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }

    fn handle_diff(&self, input: &DateTimeInput) -> ToolResult {
        let value1 = match &input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for diff operation"),
        };

        let value2 = match &input.value2 {
            Some(v) => v,
            None => return ToolResult::error("'value2' is required for diff operation"),
        };

        let dt1 = match self.parse_datetime(value1) {
            Ok(dt) => dt,
            Err(e) => return ToolResult::error(format!("Error parsing value: {}", e)),
        };

        let dt2 = match self.parse_datetime(value2) {
            Ok(dt) => dt,
            Err(e) => return ToolResult::error(format!("Error parsing value2: {}", e)),
        };

        let diff = dt2.signed_duration_since(dt1);
        let total_seconds = diff.num_seconds();

        let output = DiffOutput {
            seconds: total_seconds,
            minutes: total_seconds / 60,
            hours: total_seconds / 3600,
            days: total_seconds / 86400,
            value1: dt1.to_rfc3339(),
            value2: dt2.to_rfc3339(),
        };

        match serde_json::to_string(&output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }

    fn parse_datetime(&self, value: &str) -> Result<DateTime<Utc>, String> {
        if let Ok(ts) = value.parse::<i64>() {
            return Utc
                .timestamp_opt(ts, 0)
                .single()
                .ok_or_else(|| "Invalid unix timestamp".to_string());
        }

        if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
            return Ok(dt.with_timezone(&Utc));
        }

        if let Ok(dt) = DateTime::parse_from_rfc2822(value) {
            return Ok(dt.with_timezone(&Utc));
        }

        let formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
        ];

        for fmt in formats {
            if let Ok(ndt) = NaiveDateTime::parse_from_str(value, fmt) {
                return Ok(Utc.from_utc_datetime(&ndt));
            }
            if let Ok(nd) = chrono::NaiveDate::parse_from_str(value, fmt) {
                let ndt = nd.and_hms_opt(0, 0, 0).unwrap();
                return Ok(Utc.from_utc_datetime(&ndt));
            }
        }

        Err(format!(
            "Unable to parse date/time: '{}'. Supported formats: ISO 8601, RFC 3339, RFC 2822, YYYY-MM-DD, unix timestamp",
            value
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_now() {
        let tool = DateTimeTool::new();
        let result = tool.execute(serde_json::json!({"operation": "now"})).await;
        assert!(result.success);

        let output: NowOutput = serde_json::from_str(&result.output).unwrap();
        assert!(!output.iso.is_empty());
        assert!(output.unix_timestamp > 0);
    }

    #[tokio::test]
    async fn test_now_with_format() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "now",
                "format": "%Y-%m-%d"
            }))
            .await;
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_format() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "format",
                "value": "2024-12-25T10:30:00Z",
                "format": "%B %d, %Y"
            }))
            .await;
        assert!(result.success);

        let output: FormatOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.formatted, "December 25, 2024");
    }

    #[tokio::test]
    async fn test_parse_iso() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "parse",
                "value": "2024-01-15T12:00:00Z"
            }))
            .await;
        assert!(result.success);

        let output: ParseOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.unix_timestamp > 0);
    }

    #[tokio::test]
    async fn test_parse_simple_date() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "parse",
                "value": "2024-01-15"
            }))
            .await;
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_add_days() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "add",
                "value": "2024-01-15T00:00:00Z",
                "amount": 10,
                "unit": "days"
            }))
            .await;
        assert!(result.success);

        let output: AddOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.result.contains("2024-01-25"));
    }

    #[tokio::test]
    async fn test_add_negative() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "add",
                "value": "2024-01-15T00:00:00Z",
                "amount": -5,
                "unit": "days"
            }))
            .await;
        assert!(result.success);

        let output: AddOutput = serde_json::from_str(&result.output).unwrap();
        assert!(output.result.contains("2024-01-10"));
    }

    #[tokio::test]
    async fn test_diff() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "diff",
                "value": "2024-01-01T00:00:00Z",
                "value2": "2024-01-02T00:00:00Z"
            }))
            .await;
        assert!(result.success);

        let output: DiffOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.days, 1);
        assert_eq!(output.hours, 24);
        assert_eq!(output.seconds, 86400);
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "invalid"}))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_missing_value() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "format"}))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_parse_unix_timestamp() {
        let tool = DateTimeTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "parse",
                "value": "1704067200"
            }))
            .await;
        assert!(result.success);
    }
}
