use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::generate_schema;
use ai_agents_core::{Tool, ToolResult};

pub struct MathTool;

impl MathTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MathTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MathInput {
    /// Operation: mean, median, mode, stdev, variance, sum, min, max, abs, round, floor, ceil, clamp, percentage, sqrt, pow, log, range, count
    operation: String,
    /// Array of numbers (for statistical operations)
    #[serde(default)]
    values: Option<Vec<f64>>,
    /// Single number input
    #[serde(default)]
    value: Option<f64>,
    /// Decimal places (for round)
    #[serde(default)]
    decimals: Option<i32>,
    /// Minimum value (for clamp/range)
    #[serde(default)]
    min: Option<f64>,
    /// Maximum value (for clamp/range)
    #[serde(default)]
    max: Option<f64>,
    /// Base for pow/log
    #[serde(default)]
    base: Option<f64>,
    /// Exponent for pow
    #[serde(default)]
    exponent: Option<f64>,
    /// Total for percentage calculation
    #[serde(default)]
    total: Option<f64>,
    /// Step for range
    #[serde(default)]
    step: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StatOutput {
    result: f64,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct StdevOutput {
    stdev: f64,
    variance: f64,
    mean: f64,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModeOutput {
    mode: Vec<f64>,
    frequency: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct SingleOutput {
    result: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClampOutput {
    result: f64,
    clamped: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RangeOutput {
    range: Vec<f64>,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct MinMaxOutput {
    min: f64,
    max: f64,
    range: f64,
}

#[async_trait]
impl Tool for MathTool {
    fn id(&self) -> &str {
        "math"
    }

    fn name(&self) -> &str {
        "Advanced Math"
    }

    fn description(&self) -> &str {
        "Advanced math operations: mean (average), median, mode, stdev (standard deviation), variance, sum, min, max, minmax (both), abs, round, floor, ceil, clamp, percentage, sqrt, pow, log, log10, range, count."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<MathInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: MathInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.to_lowercase().as_str() {
            "mean" | "average" | "avg" => self.handle_mean(&input),
            "median" => self.handle_median(&input),
            "mode" => self.handle_mode(&input),
            "stdev" | "std" => self.handle_stdev(&input),
            "variance" | "var" => self.handle_variance(&input),
            "sum" => self.handle_sum(&input),
            "min" => self.handle_min(&input),
            "max" => self.handle_max(&input),
            "minmax" => self.handle_minmax(&input),
            "abs" => self.handle_abs(&input),
            "round" => self.handle_round(&input),
            "floor" => self.handle_floor(&input),
            "ceil" => self.handle_ceil(&input),
            "clamp" => self.handle_clamp(&input),
            "percentage" | "percent" => self.handle_percentage(&input),
            "sqrt" => self.handle_sqrt(&input),
            "pow" | "power" => self.handle_pow(&input),
            "log" => self.handle_log(&input),
            "log10" => self.handle_log10(&input),
            "range" => self.handle_range(&input),
            "count" => self.handle_count(&input),
            _ => ToolResult::error(format!(
                "Unknown operation: {}. Valid: mean, median, mode, stdev, variance, sum, min, max, minmax, abs, round, floor, ceil, clamp, percentage, sqrt, pow, log, log10, range, count",
                input.operation
            )),
        }
    }
}

impl MathTool {
    fn get_values(&self, input: &MathInput) -> Result<Vec<f64>, ToolResult> {
        input
            .values
            .clone()
            .ok_or_else(|| ToolResult::error("'values' array is required"))
    }

    fn handle_mean(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.is_empty() {
            return ToolResult::error("values array cannot be empty");
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let output = StatOutput {
            result: mean,
            count: values.len(),
        };
        self.to_result(&output)
    }

    fn handle_median(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.is_empty() {
            return ToolResult::error("values array cannot be empty");
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };

        let output = StatOutput {
            result: median,
            count: values.len(),
        };
        self.to_result(&output)
    }

    fn handle_mode(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.is_empty() {
            return ToolResult::error("values array cannot be empty");
        }

        use std::collections::HashMap;
        let mut counts: HashMap<String, usize> = HashMap::new();

        for v in &values {
            let key = format!("{:.10}", v);
            *counts.entry(key).or_insert(0) += 1;
        }

        let max_count = *counts.values().max().unwrap_or(&0);
        let modes: Vec<f64> = counts
            .iter()
            .filter(|&(_, &c)| c == max_count)
            .filter_map(|(k, _)| k.parse().ok())
            .collect();

        let output = ModeOutput {
            mode: modes,
            frequency: max_count,
        };
        self.to_result(&output)
    }

    fn handle_stdev(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.len() < 2 {
            return ToolResult::error("stdev requires at least 2 values");
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let stdev = variance.sqrt();

        let output = StdevOutput {
            stdev,
            variance,
            mean,
            count: values.len(),
        };
        self.to_result(&output)
    }

    fn handle_variance(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.len() < 2 {
            return ToolResult::error("variance requires at least 2 values");
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        let output = StatOutput {
            result: variance,
            count: values.len(),
        };
        self.to_result(&output)
    }

    fn handle_sum(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };

        let sum: f64 = values.iter().sum();
        let output = StatOutput {
            result: sum,
            count: values.len(),
        };
        self.to_result(&output)
    }

    fn handle_min(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.is_empty() {
            return ToolResult::error("values array cannot be empty");
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let output = SingleOutput { result: min };
        self.to_result(&output)
    }

    fn handle_max(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.is_empty() {
            return ToolResult::error("values array cannot be empty");
        }

        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let output = SingleOutput { result: max };
        self.to_result(&output)
    }

    fn handle_minmax(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if values.is_empty() {
            return ToolResult::error("values array cannot be empty");
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let output = MinMaxOutput {
            min,
            max,
            range: max - min,
        };
        self.to_result(&output)
    }

    fn handle_abs(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for abs operation"),
        };
        let output = SingleOutput {
            result: value.abs(),
        };
        self.to_result(&output)
    }

    fn handle_round(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for round operation"),
        };
        let decimals = input.decimals.unwrap_or(0);
        let multiplier = 10_f64.powi(decimals);
        let rounded = (value * multiplier).round() / multiplier;
        let output = SingleOutput { result: rounded };
        self.to_result(&output)
    }

    fn handle_floor(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for floor operation"),
        };
        let output = SingleOutput {
            result: value.floor(),
        };
        self.to_result(&output)
    }

    fn handle_ceil(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for ceil operation"),
        };
        let output = SingleOutput {
            result: value.ceil(),
        };
        self.to_result(&output)
    }

    fn handle_clamp(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for clamp operation"),
        };
        let min = match input.min {
            Some(m) => m,
            None => return ToolResult::error("'min' is required for clamp operation"),
        };
        let max = match input.max {
            Some(m) => m,
            None => return ToolResult::error("'max' is required for clamp operation"),
        };

        let clamped_value = value.max(min).min(max);
        let output = ClampOutput {
            result: clamped_value,
            clamped: value != clamped_value,
        };
        self.to_result(&output)
    }

    fn handle_percentage(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for percentage operation"),
        };
        let total = match input.total {
            Some(t) => t,
            None => return ToolResult::error("'total' is required for percentage operation"),
        };
        if total == 0.0 {
            return ToolResult::error("total cannot be zero");
        }

        let percentage = (value / total) * 100.0;
        let output = SingleOutput { result: percentage };
        self.to_result(&output)
    }

    fn handle_sqrt(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for sqrt operation"),
        };
        if value < 0.0 {
            return ToolResult::error("cannot calculate sqrt of negative number");
        }
        let output = SingleOutput {
            result: value.sqrt(),
        };
        self.to_result(&output)
    }

    fn handle_pow(&self, input: &MathInput) -> ToolResult {
        let base = input.value.or(input.base);
        let base = match base {
            Some(b) => b,
            None => return ToolResult::error("'value' or 'base' is required for pow operation"),
        };
        let exponent = match input.exponent {
            Some(e) => e,
            None => return ToolResult::error("'exponent' is required for pow operation"),
        };
        let output = SingleOutput {
            result: base.powf(exponent),
        };
        self.to_result(&output)
    }

    fn handle_log(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for log operation"),
        };
        if value <= 0.0 {
            return ToolResult::error("cannot calculate log of non-positive number");
        }
        let result = match input.base {
            Some(b) if b > 0.0 && b != 1.0 => value.log(b),
            Some(_) => return ToolResult::error("log base must be positive and not equal to 1"),
            None => value.ln(),
        };
        let output = SingleOutput { result };
        self.to_result(&output)
    }

    fn handle_log10(&self, input: &MathInput) -> ToolResult {
        let value = match input.value {
            Some(v) => v,
            None => return ToolResult::error("'value' is required for log10 operation"),
        };
        if value <= 0.0 {
            return ToolResult::error("cannot calculate log of non-positive number");
        }
        let output = SingleOutput {
            result: value.log10(),
        };
        self.to_result(&output)
    }

    fn handle_range(&self, input: &MathInput) -> ToolResult {
        let min = input.min.unwrap_or(0.0);
        let max = match input.max {
            Some(m) => m,
            None => return ToolResult::error("'max' is required for range operation"),
        };
        let step = input.step.unwrap_or(1.0);

        if step == 0.0 {
            return ToolResult::error("step cannot be zero");
        }
        if (max > min && step < 0.0) || (max < min && step > 0.0) {
            return ToolResult::error("step direction doesn't match min/max range");
        }

        let mut values = Vec::new();
        let mut current = min;

        if step > 0.0 {
            while current < max {
                values.push(current);
                current += step;
            }
        } else {
            while current > max {
                values.push(current);
                current += step;
            }
        }

        let output = RangeOutput {
            count: values.len(),
            range: values,
        };
        self.to_result(&output)
    }

    fn handle_count(&self, input: &MathInput) -> ToolResult {
        let values = match self.get_values(input) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let output = StatOutput {
            result: values.len() as f64,
            count: values.len(),
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
    async fn test_mean() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "mean",
                "values": [1, 2, 3, 4, 5]
            }))
            .await;
        assert!(result.success);
        let output: StatOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 3.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_median_odd() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "median",
                "values": [1, 3, 2, 5, 4]
            }))
            .await;
        assert!(result.success);
        let output: StatOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 3.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_median_even() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "median",
                "values": [1, 2, 3, 4]
            }))
            .await;
        assert!(result.success);
        let output: StatOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 2.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_stdev() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "stdev",
                "values": [2, 4, 4, 4, 5, 5, 7, 9]
            }))
            .await;
        assert!(result.success);
        let output: StdevOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.stdev - 2.138).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_sum() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "sum",
                "values": [1, 2, 3, 4, 5]
            }))
            .await;
        assert!(result.success);
        let output: StatOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 15.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_minmax() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "minmax",
                "values": [3, 1, 4, 1, 5, 9, 2, 6]
            }))
            .await;
        assert!(result.success);
        let output: MinMaxOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.min - 1.0).abs() < f64::EPSILON);
        assert!((output.max - 9.0).abs() < f64::EPSILON);
        assert!((output.range - 8.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_round() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "round",
                "value": 3.14159,
                "decimals": 2
            }))
            .await;
        assert!(result.success);
        let output: SingleOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 3.14).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_clamp() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "clamp",
                "value": 15,
                "min": 0,
                "max": 10
            }))
            .await;
        assert!(result.success);
        let output: ClampOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 10.0).abs() < f64::EPSILON);
        assert!(output.clamped);
    }

    #[tokio::test]
    async fn test_percentage() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "percentage",
                "value": 25,
                "total": 100
            }))
            .await;
        assert!(result.success);
        let output: SingleOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 25.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_sqrt() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "sqrt",
                "value": 16
            }))
            .await;
        assert!(result.success);
        let output: SingleOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 4.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_pow() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "pow",
                "value": 2,
                "exponent": 10
            }))
            .await;
        assert!(result.success);
        let output: SingleOutput = serde_json::from_str(&result.output).unwrap();
        assert!((output.result - 1024.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_range() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "range",
                "min": 0,
                "max": 5,
                "step": 1
            }))
            .await;
        assert!(result.success);
        let output: RangeOutput = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output.range, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "invalid"
            }))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_empty_values() {
        let tool = MathTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "mean",
                "values": []
            }))
            .await;
        assert!(!result.success);
    }
}
