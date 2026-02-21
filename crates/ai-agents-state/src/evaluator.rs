use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use ai_agents_core::{ChatMessage, LLMProvider, Result};
use ai_agents_llm::mock::MockLLMProvider;

use super::config::{CompareOp, ContextMatcher, GuardConditions, Transition, TransitionGuard};

pub struct TransitionContext {
    pub user_message: String,
    pub assistant_response: String,
    pub current_state: String,
    pub context: HashMap<String, Value>,
}

impl TransitionContext {
    pub fn new(user_message: &str, assistant_response: &str, current_state: &str) -> Self {
        Self {
            user_message: user_message.to_string(),
            assistant_response: assistant_response.to_string(),
            current_state: current_state.to_string(),
            context: HashMap::new(),
        }
    }

    pub fn with_context(mut self, context: HashMap<String, Value>) -> Self {
        self.context = context;
        self
    }
}

#[async_trait]
pub trait TransitionEvaluator: Send + Sync {
    async fn select_transition(
        &self,
        transitions: &[Transition],
        context: &TransitionContext,
    ) -> Result<Option<usize>>;
}

pub struct LLMTransitionEvaluator {
    llm: Arc<dyn LLMProvider>,
}

impl LLMTransitionEvaluator {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self { llm }
    }

    fn evaluate_guard(&self, guard: &TransitionGuard, ctx: &TransitionContext) -> bool {
        match guard {
            TransitionGuard::Expression(expr) => self.evaluate_expression(expr, ctx),
            TransitionGuard::Conditions(conditions) => self.evaluate_conditions(conditions, ctx),
        }
    }

    fn evaluate_expression(&self, expr: &str, ctx: &TransitionContext) -> bool {
        let expr = expr.trim();

        if !expr.contains("{{") {
            return !expr.is_empty();
        }

        let inner = expr.trim_start_matches("{{").trim_end_matches("}}").trim();

        self.evaluate_simple_expression(inner, ctx)
    }

    fn evaluate_simple_expression(&self, expr: &str, ctx: &TransitionContext) -> bool {
        if expr.starts_with("context.") {
            let path = &expr[8..];
            return self.get_context_value(path, &ctx.context).is_some();
        }

        if expr.starts_with("state.") {
            let field = &expr[6..];
            return self.evaluate_state_expression(field, ctx);
        }

        if let Some(idx) = expr.find('>') {
            let (left, right) = expr.split_at(idx);
            let op = if right.starts_with(">=") { ">=" } else { ">" };
            let right = right.trim_start_matches(op).trim();
            let left = left.trim();

            if let (Some(left_val), Ok(right_val)) =
                (self.resolve_value(left, ctx), right.parse::<f64>())
            {
                if let Some(left_num) = left_val.as_f64() {
                    return if op == ">=" {
                        left_num >= right_val
                    } else {
                        left_num > right_val
                    };
                }
            }
        }

        if let Some(idx) = expr.find('<') {
            let (left, right) = expr.split_at(idx);
            let op = if right.starts_with("<=") { "<=" } else { "<" };
            let right = right.trim_start_matches(op).trim();
            let left = left.trim();

            if let (Some(left_val), Ok(right_val)) =
                (self.resolve_value(left, ctx), right.parse::<f64>())
            {
                if let Some(left_num) = left_val.as_f64() {
                    return if op == "<=" {
                        left_num <= right_val
                    } else {
                        left_num < right_val
                    };
                }
            }
        }

        if let Some(idx) = expr.find("==") {
            let (left, right) = expr.split_at(idx);
            let right = &right[2..].trim();
            let left = left.trim();

            if let Some(left_val) = self.resolve_value(left, ctx) {
                let right_val: Value = if right.starts_with('"') && right.ends_with('"') {
                    Value::String(right[1..right.len() - 1].to_string())
                } else if *right == "true" {
                    Value::Bool(true)
                } else if *right == "false" {
                    Value::Bool(false)
                } else if let Ok(n) = right.parse::<f64>() {
                    serde_json::json!(n)
                } else {
                    Value::String(right.to_string())
                };

                return left_val == right_val;
            }
        }

        if let Some(idx) = expr.find("!=") {
            let (left, right) = expr.split_at(idx);
            let right = &right[2..].trim();
            let left = left.trim();

            if let Some(left_val) = self.resolve_value(left, ctx) {
                let right_val: Value = if right.starts_with('"') && right.ends_with('"') {
                    Value::String(right[1..right.len() - 1].to_string())
                } else if *right == "true" {
                    Value::Bool(true)
                } else if *right == "false" {
                    Value::Bool(false)
                } else if let Ok(n) = right.parse::<f64>() {
                    serde_json::json!(n)
                } else {
                    Value::String(right.to_string())
                };

                return left_val != right_val;
            }
        }

        false
    }

    fn resolve_value(&self, expr: &str, ctx: &TransitionContext) -> Option<Value> {
        let expr = expr.trim();
        if expr.starts_with("context.") {
            let path = &expr[8..];
            return self.get_context_value(path, &ctx.context);
        }
        if expr.starts_with("state.") {
            let field = &expr[6..];
            return self.get_state_value(field, ctx);
        }
        None
    }

    fn evaluate_state_expression(&self, field: &str, _ctx: &TransitionContext) -> bool {
        match field {
            "turn_count" => true,
            _ => false,
        }
    }

    fn get_state_value(&self, field: &str, _ctx: &TransitionContext) -> Option<Value> {
        match field {
            "current" => Some(Value::String(_ctx.current_state.clone())),
            _ => None,
        }
    }

    fn evaluate_conditions(&self, conditions: &GuardConditions, ctx: &TransitionContext) -> bool {
        match conditions {
            GuardConditions::All(exprs) => exprs.iter().all(|e| self.evaluate_expression(e, ctx)),
            GuardConditions::Any(exprs) => exprs.iter().any(|e| self.evaluate_expression(e, ctx)),
            GuardConditions::Context(matchers) => {
                self.evaluate_context_matchers(matchers, &ctx.context)
            }
        }
    }

    fn evaluate_context_matchers(
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
                .or_else(|| {
                    value
                        .as_array()
                        .map(|arr| arr.iter().any(|v| v.as_str() == Some(s)))
                })
                .unwrap_or(false),
        }
    }
}

#[async_trait]
impl TransitionEvaluator for LLMTransitionEvaluator {
    async fn select_transition(
        &self,
        transitions: &[Transition],
        context: &TransitionContext,
    ) -> Result<Option<usize>> {
        if transitions.is_empty() {
            return Ok(None);
        }

        // Guard-based transitions (existing, no LLM)
        for (i, transition) in transitions.iter().enumerate() {
            if let Some(ref guard) = transition.guard {
                if self.evaluate_guard(guard, context) {
                    return Ok(Some(i));
                }
            }
        }

        // !!NOTE: Resolved-intent short-circuit
        //
        // If disambiguation has resolved an intent, try to match it against transitions that declare an `intent` field.
        // This is DETERMINISTIC - no LLM call.
        if let Some(resolved) = context.context.get("resolved_intent") {
            if let Some(resolved_str) = resolved.as_str() {
                // Skip null values (used to clear stale context)
                if !resolved_str.is_empty() {
                    for (i, transition) in transitions.iter().enumerate() {
                        if let Some(ref intent) = transition.intent {
                            if intent == resolved_str {
                                tracing::debug!(
                                    resolved_intent = resolved_str,
                                    target = %transition.to,
                                    "Deterministic routing via resolved_intent"
                                );
                                return Ok(Some(i));
                            }
                        }
                    }
                }
            }
        }

        // ── Phase 1: LLM-based evaluation (existing) ──
        let llm_transitions: Vec<(usize, &Transition)> = transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.when.is_empty() && t.guard.is_none())
            .collect();

        if llm_transitions.is_empty() {
            return Ok(None);
        }

        let conditions: Vec<String> = llm_transitions
            .iter()
            .enumerate()
            .map(|(display_idx, (_, t))| format!("{}. {}", display_idx + 1, t.when))
            .collect();

        let prompt = format!(
            r#"Based on the conversation, which condition is met?

Current state: {}
User message: {}
Assistant response: {}

Conditions:
{}
0. None of the above

Reply with ONLY the number (0-{})."#,
            context.current_state,
            context.user_message,
            context.assistant_response,
            conditions.join("\n"),
            llm_transitions.len()
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = self.llm.complete(&messages, None).await?;

        let choice: usize = response.content.trim().parse().unwrap_or(0);

        if choice == 0 || choice > llm_transitions.len() {
            Ok(None)
        } else {
            Ok(Some(llm_transitions[choice - 1].0))
        }
    }
}

pub struct GuardOnlyEvaluator;

impl GuardOnlyEvaluator {
    pub fn new() -> Self {
        Self
    }

    pub fn evaluate_guard(&self, guard: &TransitionGuard, ctx: &TransitionContext) -> bool {
        let evaluator = LLMTransitionEvaluator {
            llm: Arc::new(MockLLMProvider::new("guard_eval")),
        };
        evaluator.evaluate_guard(guard, ctx)
    }

    pub fn evaluate_guards(
        &self,
        transitions: &[Transition],
        ctx: &TransitionContext,
    ) -> Option<usize> {
        for (i, transition) in transitions.iter().enumerate() {
            if let Some(ref guard) = transition.guard {
                if self.evaluate_guard(guard, ctx) {
                    return Some(i);
                }
            }
        }
        None
    }
}

impl Default for GuardOnlyEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::{FinishReason, LLMResponse};
    use ai_agents_llm::mock::MockLLMProvider;

    #[tokio::test]
    async fn test_select_transition_none() {
        let mut mock = MockLLMProvider::new("evaluator_test");
        mock.add_response(LLMResponse::new("0", FinishReason::Stop));
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![Transition {
            to: "next".into(),
            when: "user says goodbye".into(),
            guard: None,
            intent: None,
            auto: true,
            priority: 0,
        }];

        let context = TransitionContext::new("hello", "hi there", "greeting");

        let result = evaluator.select_transition(&transitions, &context).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_select_transition_match() {
        let mut mock = MockLLMProvider::new("evaluator_test");
        mock.add_response(LLMResponse::new("1", FinishReason::Stop));
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![
            Transition {
                to: "support".into(),
                when: "user needs help".into(),
                guard: None,
                intent: None,
                auto: true,
                priority: 10,
            },
            Transition {
                to: "sales".into(),
                when: "user wants to buy".into(),
                guard: None,
                intent: None,
                auto: true,
                priority: 5,
            },
        ];

        let context = TransitionContext::new("I need help", "Sure!", "greeting");

        let result = evaluator.select_transition(&transitions, &context).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(0));
    }

    #[tokio::test]
    async fn test_empty_transitions() {
        let mock = MockLLMProvider::new("evaluator_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let context = TransitionContext::new("hi", "hello", "start");

        let result = evaluator.select_transition(&[], &context).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_guard_expression_simple() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert("has_data".to_string(), Value::Bool(true));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let guard = TransitionGuard::Expression("{{ context.has_data }}".into());
        assert!(evaluator.evaluate_guard(&guard, &ctx));
    }

    #[tokio::test]
    async fn test_guard_expression_missing() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(HashMap::new());

        let guard = TransitionGuard::Expression("{{ context.has_data }}".into());
        assert!(!evaluator.evaluate_guard(&guard, &ctx));
    }

    #[tokio::test]
    async fn test_guard_with_nested_context() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert(
            "user".to_string(),
            serde_json::json!({
                "name": "Alice",
                "verified": true
            }),
        );

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let guard = TransitionGuard::Expression("{{ context.user.verified }}".into());
        assert!(evaluator.evaluate_guard(&guard, &ctx));
    }

    #[tokio::test]
    async fn test_guard_conditions_all() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert("has_name".to_string(), Value::Bool(true));
        context_map.insert("has_email".to_string(), Value::Bool(true));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let guard = TransitionGuard::Conditions(GuardConditions::All(vec![
            "{{ context.has_name }}".into(),
            "{{ context.has_email }}".into(),
        ]));
        assert!(evaluator.evaluate_guard(&guard, &ctx));
    }

    #[tokio::test]
    async fn test_guard_conditions_any() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert("is_vip".to_string(), Value::Bool(true));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let guard = TransitionGuard::Conditions(GuardConditions::Any(vec![
            "{{ context.is_admin }}".into(),
            "{{ context.is_vip }}".into(),
        ]));
        assert!(evaluator.evaluate_guard(&guard, &ctx));
    }

    #[tokio::test]
    async fn test_guard_context_matchers() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert(
            "user".to_string(),
            serde_json::json!({
                "tier": "premium",
                "balance": 100.0
            }),
        );

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let mut matchers = HashMap::new();
        matchers.insert(
            "user.tier".to_string(),
            ContextMatcher::Exact(Value::String("premium".into())),
        );
        matchers.insert(
            "user.balance".to_string(),
            ContextMatcher::Compare(CompareOp::Gte(50.0)),
        );

        let guard = TransitionGuard::Conditions(GuardConditions::Context(matchers));
        assert!(evaluator.evaluate_guard(&guard, &ctx));
    }

    #[tokio::test]
    async fn test_guard_priority_over_llm() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert("ready".to_string(), Value::Bool(true));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let transitions = vec![
            Transition {
                to: "llm_based".into(),
                when: "user wants to proceed".into(),
                guard: None,
                intent: None,
                auto: true,
                priority: 10,
            },
            Transition {
                to: "guard_based".into(),
                when: String::new(),
                guard: Some(TransitionGuard::Expression("{{ context.ready }}".into())),
                intent: None,
                auto: true,
                priority: 5,
            },
        ];

        let result = evaluator.select_transition(&transitions, &ctx).await;
        assert_eq!(result.unwrap(), Some(1));
    }

    #[test]
    fn test_guard_only_evaluator() {
        let evaluator = GuardOnlyEvaluator::new();

        let mut context_map = HashMap::new();
        context_map.insert("ready".to_string(), Value::Bool(true));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let transitions = vec![
            Transition {
                to: "no_guard".into(),
                when: "some condition".into(),
                guard: None,
                intent: None,
                auto: true,
                priority: 10,
            },
            Transition {
                to: "with_guard".into(),
                when: String::new(),
                guard: Some(TransitionGuard::Expression("{{ context.ready }}".into())),
                intent: None,
                auto: true,
                priority: 5,
            },
        ];

        let result = evaluator.evaluate_guards(&transitions, &ctx);
        assert_eq!(result, Some(1));
    }

    #[tokio::test]
    async fn test_context_matcher_exists() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert("name".to_string(), Value::String("Alice".into()));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let mut matchers = HashMap::new();
        matchers.insert("name".to_string(), ContextMatcher::Exists { exists: true });
        matchers.insert(
            "email".to_string(),
            ContextMatcher::Exists { exists: false },
        );

        let guard = TransitionGuard::Conditions(GuardConditions::Context(matchers));
        assert!(evaluator.evaluate_guard(&guard, &ctx));
    }

    #[tokio::test]
    async fn test_compare_contains() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert("message".to_string(), Value::String("hello world".into()));
        context_map.insert("tags".to_string(), serde_json::json!(["urgent", "support"]));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let mut matchers1 = HashMap::new();
        matchers1.insert(
            "message".to_string(),
            ContextMatcher::Compare(CompareOp::Contains("world".into())),
        );
        let guard1 = TransitionGuard::Conditions(GuardConditions::Context(matchers1));
        assert!(evaluator.evaluate_guard(&guard1, &ctx));

        let mut matchers2 = HashMap::new();
        matchers2.insert(
            "tags".to_string(),
            ContextMatcher::Compare(CompareOp::Contains("urgent".into())),
        );
        let guard2 = TransitionGuard::Conditions(GuardConditions::Context(matchers2));
        assert!(evaluator.evaluate_guard(&guard2, &ctx));
    }

    #[tokio::test]
    async fn test_compare_in() {
        let mock = MockLLMProvider::new("guard_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let mut context_map = HashMap::new();
        context_map.insert("tier".to_string(), Value::String("premium".into()));

        let ctx = TransitionContext::new("hi", "hello", "start").with_context(context_map);

        let mut matchers = HashMap::new();
        matchers.insert(
            "tier".to_string(),
            ContextMatcher::Compare(CompareOp::In(vec![
                Value::String("premium".into()),
                Value::String("enterprise".into()),
            ])),
        );

        let guard = TransitionGuard::Conditions(GuardConditions::Context(matchers));
        assert!(evaluator.evaluate_guard(&guard, &ctx));
    }

    /// a transition's `intent` field, routing is deterministic - no LLM call.
    #[tokio::test]
    async fn test_intent_based_routing_deterministic() {
        // The mock has NO responses queued — if the LLM were called it would panic/fail.
        let mock = MockLLMProvider::new("intent_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![
            Transition {
                to: "cancel_order".into(),
                when: "User wants to cancel an order".into(),
                guard: None,
                intent: Some("cancel_order".into()),
                auto: true,
                priority: 10,
            },
            Transition {
                to: "cancel_reservation".into(),
                when: "User wants to cancel a reservation".into(),
                guard: None,
                intent: Some("cancel_reservation".into()),
                auto: true,
                priority: 10,
            },
            Transition {
                to: "cancel_subscription".into(),
                when: "User wants to cancel a subscription".into(),
                guard: None,
                intent: Some("cancel_subscription".into()),
                auto: true,
                priority: 10,
            },
        ];

        // Simulate disambiguation having resolved intent to "cancel_reservation"
        let mut context_map = HashMap::new();
        context_map.insert(
            "resolved_intent".to_string(),
            Value::String("cancel_reservation".into()),
        );

        let ctx =
            TransitionContext::new("あれキャンセルして", "", "greeting").with_context(context_map);

        let result = evaluator
            .select_transition(&transitions, &ctx)
            .await
            .unwrap();
        // Should pick index 1 (cancel_reservation) deterministically
        assert_eq!(result, Some(1));
    }

    /// the evaluator falls back to LLM-based `when` evaluation.
    #[tokio::test]
    async fn test_intent_routing_falls_back_to_llm_when_no_resolved_intent() {
        let mut mock = MockLLMProvider::new("intent_fallback_test");
        // LLM returns "1" → first LLM transition (cancel_order)
        mock.add_response(LLMResponse::new("1", FinishReason::Stop));
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![
            Transition {
                to: "cancel_order".into(),
                when: "User wants to cancel an order".into(),
                guard: None,
                intent: Some("cancel_order".into()),
                auto: true,
                priority: 10,
            },
            Transition {
                to: "cancel_reservation".into(),
                when: "User wants to cancel a reservation".into(),
                guard: None,
                intent: Some("cancel_reservation".into()),
                auto: true,
                priority: 10,
            },
        ];

        // No resolved_intent in context -> LLM evaluation fires
        let ctx = TransitionContext::new("Cancel order ORD-1042", "", "greeting")
            .with_context(HashMap::new());

        let result = evaluator
            .select_transition(&transitions, &ctx)
            .await
            .unwrap();
        // LLM said "1" → index 0 (first LLM transition)
        assert_eq!(result, Some(0));
    }

    /// transition's `intent` field, it falls through to LLM evaluation.
    #[tokio::test]
    async fn test_no_routing_when_resolved_intent_doesnt_match() {
        let mut mock = MockLLMProvider::new("intent_nomatch_test");
        // LLM returns "0" (none of the above)
        mock.add_response(LLMResponse::new("0", FinishReason::Stop));
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![
            Transition {
                to: "cancel_order".into(),
                when: "User wants to cancel an order".into(),
                guard: None,
                intent: Some("cancel_order".into()),
                auto: true,
                priority: 10,
            },
            Transition {
                to: "cancel_reservation".into(),
                when: "User wants to cancel a reservation".into(),
                guard: None,
                intent: Some("cancel_reservation".into()),
                auto: true,
                priority: 10,
            },
        ];

        // resolved_intent is "something_else" which matches no transition
        let mut context_map = HashMap::new();
        context_map.insert(
            "resolved_intent".to_string(),
            Value::String("something_else".into()),
        );

        let ctx = TransitionContext::new("do something", "", "greeting").with_context(context_map);

        let result = evaluator
            .select_transition(&transitions, &ctx)
            .await
            .unwrap();
        // LLM said "0" → None
        assert_eq!(result, None);
    }

    /// should be ignored - not treated as a valid intent.
    #[tokio::test]
    async fn test_null_resolved_intent_is_ignored() {
        let mut mock = MockLLMProvider::new("intent_null_test");
        mock.add_response(LLMResponse::new("1", FinishReason::Stop));
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![Transition {
            to: "cancel_order".into(),
            when: "User wants to cancel an order".into(),
            guard: None,
            intent: Some("cancel_order".into()),
            auto: true,
            priority: 10,
        }];

        // resolved_intent is Null (simulating clear_disambiguation_context)
        let mut context_map = HashMap::new();
        context_map.insert("resolved_intent".to_string(), Value::Null);

        let ctx =
            TransitionContext::new("Cancel my order", "", "greeting").with_context(context_map);

        let result = evaluator
            .select_transition(&transitions, &ctx)
            .await
            .unwrap();
        // Null resolved_intent should be ignored; LLM said "1" -> index 0
        assert_eq!(result, Some(0));
    }
}
