use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Datelike, Local, Timelike, Utc};
use serde::Deserialize;
use serde_json::Value;

use crate::llm::LLMProvider;
use crate::state::{CompareOp, ContextMatcher, StateMatcher, TimeMatcher, ToolCondition};
use crate::{ChatMessage, Result};

#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    pub tool_id: String,
    pub result: Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct EvaluationContext {
    pub context: HashMap<String, Value>,
    pub state_name: Option<String>,
    pub state_turn_count: u32,
    pub previous_state: Option<String>,
    pub called_tools: Vec<ToolCallRecord>,
    pub recent_messages: Vec<ChatMessage>,
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self {
            context: HashMap::new(),
            state_name: None,
            state_turn_count: 0,
            previous_state: None,
            called_tools: Vec::new(),
            recent_messages: Vec::new(),
        }
    }
}

impl EvaluationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context(mut self, context: HashMap<String, Value>) -> Self {
        self.context = context;
        self
    }

    pub fn with_state(mut self, name: Option<String>, turn_count: u32, previous: Option<String>) -> Self {
        self.state_name = name;
        self.state_turn_count = turn_count;
        self.previous_state = previous;
        self
    }

    pub fn with_called_tools(mut self, tools: Vec<ToolCallRecord>) -> Self {
        self.called_tools = tools;
        self
    }

    pub fn with_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.recent_messages = messages;
        self
    }
}

#[async_trait]
pub trait LLMGetter: Send + Sync {
    fn get_llm(&self, alias: &str) -> Option<Arc<dyn LLMProvider>>;
}

pub struct ConditionEvaluator<G: LLMGetter> {
    llm_getter: G,
}

impl<G: LLMGetter> ConditionEvaluator<G> {
    pub fn new(llm_getter: G) -> Self {
        Self { llm_getter }
    }

    pub async fn evaluate(&self, condition: &ToolCondition, ctx: &EvaluationContext) -> Result<bool> {
        match condition {
            ToolCondition::Context(matchers) => Ok(self.evaluate_context(matchers, &ctx.context)),
            ToolCondition::State(matcher) => Ok(self.evaluate_state(matcher, ctx)),
            ToolCondition::AfterTool(tool_id) => {
                Ok(ctx.called_tools.iter().any(|t| &t.tool_id == tool_id))
            }
            ToolCondition::ToolResult { tool, result } => {
                Ok(self.evaluate_tool_result(tool, result, &ctx.called_tools))
            }
            ToolCondition::Semantic { when, llm, threshold } => {
                self.evaluate_semantic(when, llm, *threshold, ctx).await
            }
            ToolCondition::Time(matcher) => Ok(self.evaluate_time(matcher)),
            ToolCondition::All(conditions) => {
                for cond in conditions {
                    if !Box::pin(self.evaluate(cond, ctx)).await? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            ToolCondition::Any(conditions) => {
                for cond in conditions {
                    if Box::pin(self.evaluate(cond, ctx)).await? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            ToolCondition::Not(inner) => {
                Ok(!Box::pin(self.evaluate(inner, ctx)).await?)
            }
        }
    }

    fn evaluate_context(
        &self,
        matchers: &HashMap<String, ContextMatcher>,
        context: &HashMap<String, Value>,
    ) -> bool {
        for (path, matcher) in matchers {
            let value = self.get_context_value(path, context);
            if !self.match_value(value.as_ref(), matcher) {
                return false;
            }
        }
        true
    }

    fn get_context_value(&self, path: &str, context: &HashMap<String, Value>) -> Option<Value> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return None;
        }

        let mut current: Option<&Value> = context.get(parts[0]);

        for part in &parts[1..] {
            match current {
                Some(Value::Object(map)) => {
                    current = map.get(*part);
                }
                _ => return None,
            }
        }

        current.cloned()
    }

    fn match_value(&self, value: Option<&Value>, matcher: &ContextMatcher) -> bool {
        match matcher {
            ContextMatcher::Exact(expected) => value.map(|v| v == expected).unwrap_or(false),
            ContextMatcher::Exists { exists } => {
                let has_value = value.is_some() && value != Some(&Value::Null);
                *exists == has_value
            }
            ContextMatcher::Compare(op) => {
                let Some(val) = value else {
                    return false;
                };
                self.compare_value(val, op)
            }
        }
    }

    fn compare_value(&self, value: &Value, op: &CompareOp) -> bool {
        match op {
            CompareOp::Eq(expected) => value == expected,
            CompareOp::Neq(expected) => value != expected,
            CompareOp::Gt(n) => value.as_f64().map(|v| v > *n).unwrap_or(false),
            CompareOp::Gte(n) => value.as_f64().map(|v| v >= *n).unwrap_or(false),
            CompareOp::Lt(n) => value.as_f64().map(|v| v < *n).unwrap_or(false),
            CompareOp::Lte(n) => value.as_f64().map(|v| v <= *n).unwrap_or(false),
            CompareOp::In(values) => values.contains(value),
            CompareOp::Contains(s) => value
                .as_str()
                .map(|v| v.contains(s))
                .or_else(|| value.as_array().map(|arr| arr.iter().any(|v| v.as_str() == Some(s))))
                .unwrap_or(false),
        }
    }

    fn evaluate_state(&self, matcher: &StateMatcher, ctx: &EvaluationContext) -> bool {
        if let Some(ref expected_name) = matcher.name {
            if ctx.state_name.as_ref() != Some(expected_name) {
                return false;
            }
        }

        if let Some(ref turn_op) = matcher.turn_count {
            let turn_count = ctx.state_turn_count as f64;
            if !self.compare_value(&Value::Number(serde_json::Number::from_f64(turn_count).unwrap_or(0.into())), turn_op) {
                return false;
            }
        }

        if let Some(ref expected_prev) = matcher.previous {
            if ctx.previous_state.as_ref() != Some(expected_prev) {
                return false;
            }
        }

        true
    }

    fn evaluate_tool_result(
        &self,
        tool: &str,
        expected: &HashMap<String, Value>,
        called_tools: &[ToolCallRecord],
    ) -> bool {
        let tool_record = called_tools.iter().rev().find(|t| t.tool_id == tool);

        let Some(record) = tool_record else {
            return false;
        };

        let result_obj = match &record.result {
            Value::Object(obj) => obj,
            _ => return false,
        };

        for (key, expected_value) in expected {
            match result_obj.get(key) {
                Some(actual) if actual == expected_value => continue,
                _ => return false,
            }
        }

        true
    }

    fn evaluate_time(&self, matcher: &TimeMatcher) -> bool {
        let now = if let Some(ref tz) = matcher.timezone {
            if tz == "utc" || tz == "UTC" {
                Utc::now().with_timezone(&Utc).naive_local()
            } else {
                Local::now().naive_local()
            }
        } else {
            Local::now().naive_local()
        };

        if let Some(ref hours_op) = matcher.hours {
            let hour = now.hour() as f64;
            if !self.compare_value(&serde_json::json!(hour), hours_op) {
                return false;
            }
        }

        if let Some(ref days) = matcher.day_of_week {
            let day_name = match now.weekday() {
                chrono::Weekday::Mon => "monday",
                chrono::Weekday::Tue => "tuesday",
                chrono::Weekday::Wed => "wednesday",
                chrono::Weekday::Thu => "thursday",
                chrono::Weekday::Fri => "friday",
                chrono::Weekday::Sat => "saturday",
                chrono::Weekday::Sun => "sunday",
            };

            if !days.iter().any(|d| d.to_lowercase() == day_name) {
                return false;
            }
        }

        true
    }

    async fn evaluate_semantic(
        &self,
        condition: &str,
        llm_alias: &str,
        threshold: f32,
        ctx: &EvaluationContext,
    ) -> Result<bool> {
        let llm = match self.llm_getter.get_llm(llm_alias) {
            Some(l) => l,
            None => {
                tracing::warn!(llm = llm_alias, "LLM not found for semantic evaluation");
                return Ok(false);
            }
        };

        let conversation_summary = ctx
            .recent_messages
            .iter()
            .take(10)
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Based on the conversation below, evaluate if this condition is TRUE or FALSE.

Condition to evaluate: "{}"

Recent conversation:
{}

Respond with ONLY a JSON object:
{{"result": true, "confidence": 0.9, "reason": "brief explanation"}}
or
{{"result": false, "confidence": 0.9, "reason": "brief explanation"}}"#,
            condition, conversation_summary
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm.complete(&messages, None).await?;

        let parsed: SemanticEvalResult = serde_json::from_str(&response.content).unwrap_or(SemanticEvalResult {
            result: false,
            confidence: 0.0,
            reason: "Failed to parse response".to_string(),
        });

        tracing::debug!(
            condition = condition,
            result = parsed.result,
            confidence = parsed.confidence,
            threshold = threshold,
            reason = %parsed.reason,
            "Semantic evaluation"
        );

        Ok(parsed.result && parsed.confidence >= threshold)
    }
}

#[derive(Debug, Deserialize)]
struct SemanticEvalResult {
    result: bool,
    confidence: f32,
    reason: String,
}

pub struct SimpleLLMGetter {
    llms: HashMap<String, Arc<dyn LLMProvider>>,
}

impl SimpleLLMGetter {
    pub fn new() -> Self {
        Self {
            llms: HashMap::new(),
        }
    }

    pub fn with_llm(mut self, alias: &str, llm: Arc<dyn LLMProvider>) -> Self {
        self.llms.insert(alias.to_string(), llm);
        self
    }
}

impl Default for SimpleLLMGetter {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMGetter for SimpleLLMGetter {
    fn get_llm(&self, alias: &str) -> Option<Arc<dyn LLMProvider>> {
        self.llms.get(alias).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoOpLLMGetter;

    impl LLMGetter for NoOpLLMGetter {
        fn get_llm(&self, _alias: &str) -> Option<Arc<dyn LLMProvider>> {
            None
        }
    }

    #[tokio::test]
    async fn test_context_condition_exact() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("user".to_string(), serde_json::json!({
            "verified": true,
            "tier": "premium"
        }));

        let ctx = EvaluationContext::new().with_context(context);

        let mut matchers = HashMap::new();
        matchers.insert(
            "user.verified".to_string(),
            ContextMatcher::Exact(Value::Bool(true)),
        );

        let condition = ToolCondition::Context(matchers);
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_context_condition_exists() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("name".to_string(), Value::String("Alice".into()));

        let ctx = EvaluationContext::new().with_context(context);

        let mut matchers = HashMap::new();
        matchers.insert("name".to_string(), ContextMatcher::Exists { exists: true });
        matchers.insert("email".to_string(), ContextMatcher::Exists { exists: false });

        let condition = ToolCondition::Context(matchers);
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_context_condition_compare() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("balance".to_string(), serde_json::json!(150.0));

        let ctx = EvaluationContext::new().with_context(context);

        let mut matchers = HashMap::new();
        matchers.insert(
            "balance".to_string(),
            ContextMatcher::Compare(CompareOp::Gte(100.0)),
        );

        let condition = ToolCondition::Context(matchers);
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_state_condition() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let ctx = EvaluationContext::new().with_state(
            Some("checkout".to_string()),
            5,
            Some("browsing".to_string()),
        );

        let condition = ToolCondition::State(StateMatcher {
            name: Some("checkout".to_string()),
            turn_count: Some(CompareOp::Gte(3.0)),
            previous: Some("browsing".to_string()),
        });

        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_after_tool_condition() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let ctx = EvaluationContext::new().with_called_tools(vec![
            ToolCallRecord {
                tool_id: "search".to_string(),
                result: serde_json::json!({"found": true}),
                timestamp: Utc::now(),
            },
        ]);

        let condition = ToolCondition::AfterTool("search".to_string());
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());

        let condition2 = ToolCondition::AfterTool("calculate".to_string());
        assert!(!evaluator.evaluate(&condition2, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_tool_result_condition() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let ctx = EvaluationContext::new().with_called_tools(vec![ToolCallRecord {
            tool_id: "verify_purchase".to_string(),
            result: serde_json::json!({
                "valid": true,
                "refundable": true
            }),
            timestamp: Utc::now(),
        }]);

        let mut expected = HashMap::new();
        expected.insert("valid".to_string(), Value::Bool(true));
        expected.insert("refundable".to_string(), Value::Bool(true));

        let condition = ToolCondition::ToolResult {
            tool: "verify_purchase".to_string(),
            result: expected,
        };

        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_all_condition() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("verified".to_string(), Value::Bool(true));
        context.insert("balance".to_string(), serde_json::json!(100.0));

        let ctx = EvaluationContext::new().with_context(context);

        let mut m1 = HashMap::new();
        m1.insert("verified".to_string(), ContextMatcher::Exact(Value::Bool(true)));

        let mut m2 = HashMap::new();
        m2.insert("balance".to_string(), ContextMatcher::Compare(CompareOp::Gte(50.0)));

        let condition = ToolCondition::All(vec![
            ToolCondition::Context(m1),
            ToolCondition::Context(m2),
        ]);

        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_any_condition() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("tier".to_string(), Value::String("basic".into()));

        let ctx = EvaluationContext::new().with_context(context);

        let mut m1 = HashMap::new();
        m1.insert("tier".to_string(), ContextMatcher::Exact(Value::String("premium".into())));

        let mut m2 = HashMap::new();
        m2.insert("tier".to_string(), ContextMatcher::Exact(Value::String("basic".into())));

        let condition = ToolCondition::Any(vec![
            ToolCondition::Context(m1),
            ToolCondition::Context(m2),
        ]);

        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_not_condition() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("blocked".to_string(), Value::Bool(false));

        let ctx = EvaluationContext::new().with_context(context);

        let mut matchers = HashMap::new();
        matchers.insert("blocked".to_string(), ContextMatcher::Exact(Value::Bool(true)));

        let condition = ToolCondition::Not(Box::new(ToolCondition::Context(matchers)));
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_time_condition_day_of_week() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);
        let ctx = EvaluationContext::new();

        let all_days = vec![
            "monday".to_string(),
            "tuesday".to_string(),
            "wednesday".to_string(),
            "thursday".to_string(),
            "friday".to_string(),
            "saturday".to_string(),
            "sunday".to_string(),
        ];

        let condition = ToolCondition::Time(TimeMatcher {
            hours: None,
            day_of_week: Some(all_days),
            timezone: None,
        });

        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_compare_in() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("status".to_string(), Value::String("active".into()));

        let ctx = EvaluationContext::new().with_context(context);

        let mut matchers = HashMap::new();
        matchers.insert(
            "status".to_string(),
            ContextMatcher::Compare(CompareOp::In(vec![
                Value::String("active".into()),
                Value::String("pending".into()),
            ])),
        );

        let condition = ToolCondition::Context(matchers);
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_compare_contains_string() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert("email".to_string(), Value::String("user@example.com".into()));

        let ctx = EvaluationContext::new().with_context(context);

        let mut matchers = HashMap::new();
        matchers.insert(
            "email".to_string(),
            ContextMatcher::Compare(CompareOp::Contains("@example.com".into())),
        );

        let condition = ToolCondition::Context(matchers);
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn test_nested_context_path() {
        let evaluator = ConditionEvaluator::new(NoOpLLMGetter);

        let mut context = HashMap::new();
        context.insert(
            "user".to_string(),
            serde_json::json!({
                "profile": {
                    "settings": {
                        "notifications": true
                    }
                }
            }),
        );

        let ctx = EvaluationContext::new().with_context(context);

        let mut matchers = HashMap::new();
        matchers.insert(
            "user.profile.settings.notifications".to_string(),
            ContextMatcher::Exact(Value::Bool(true)),
        );

        let condition = ToolCondition::Context(matchers);
        assert!(evaluator.evaluate(&condition, &ctx).await.unwrap());
    }

    #[test]
    fn test_simple_llm_getter() {
        let getter = SimpleLLMGetter::new();
        assert!(getter.get_llm("nonexistent").is_none());
    }
}
