use serde_json::Value;
use std::collections::HashMap;

use crate::error::Result;
use crate::llm::LLMRegistry;

use super::config::{HITLConfig, StateApprovalTrigger, ToolApprovalConfig};
use super::handler::ApprovalHandler;
use super::localization::{MessageResolver, resolve_tool_message};
use super::types::{ApprovalTrigger, HITLCheckResult};

pub struct HITLEngine {
    config: HITLConfig,
}

impl HITLEngine {
    pub fn new(config: HITLConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &HITLConfig {
        &self.config
    }

    pub fn check_tool(&self, tool_name: &str, args: &Value) -> HITLCheckResult {
        if let Some(tool_config) = self.config.tools.get(tool_name) {
            if tool_config.require_approval {
                let context = self.build_tool_context(tool_config, args);
                let message = tool_config
                    .approval_message
                    .get_any()
                    .unwrap_or_else(|| format!("Approve execution of tool '{}'?", tool_name));
                let timeout = tool_config
                    .timeout_seconds
                    .or(Some(self.config.default_timeout_seconds));

                return HITLCheckResult::required(
                    ApprovalTrigger::tool(tool_name, args.clone()),
                    context,
                    message,
                    timeout,
                );
            }
        }
        HITLCheckResult::not_required()
    }

    pub async fn check_tool_with_localization(
        &self,
        tool_name: &str,
        args: &Value,
        handler: &dyn ApprovalHandler,
        llm_registry: Option<&LLMRegistry>,
    ) -> Result<HITLCheckResult> {
        if let Some(tool_config) = self.config.tools.get(tool_name) {
            if tool_config.require_approval {
                let context = self.build_tool_context(tool_config, args);

                let message = resolve_tool_message(
                    tool_config,
                    tool_name,
                    &self.config.message_language,
                    &context,
                    handler,
                    llm_registry,
                )
                .await?;

                let timeout = tool_config
                    .timeout_seconds
                    .or(Some(self.config.default_timeout_seconds));

                return Ok(HITLCheckResult::required(
                    ApprovalTrigger::tool(tool_name, args.clone()),
                    context,
                    message,
                    timeout,
                ));
            }
        }
        Ok(HITLCheckResult::not_required())
    }

    pub fn check_conditions(&self, data: &Value) -> HITLCheckResult {
        for condition in &self.config.conditions {
            if condition.require_approval && self.evaluate_condition(&condition.when, data) {
                let message = condition
                    .approval_message
                    .get_any()
                    .unwrap_or_else(|| format!("Condition '{}' matched", condition.name));

                let mut context = HashMap::new();
                context.insert(
                    "condition_name".to_string(),
                    Value::String(condition.name.clone()),
                );
                context.insert(
                    "condition_expression".to_string(),
                    Value::String(condition.when.clone()),
                );
                if let Some(obj) = data.as_object() {
                    for (k, v) in obj {
                        context.insert(k.clone(), v.clone());
                    }
                }

                return HITLCheckResult::required(
                    ApprovalTrigger::condition(&condition.name, &condition.when),
                    context,
                    message,
                    Some(self.config.default_timeout_seconds),
                );
            }
        }
        HITLCheckResult::not_required()
    }

    pub async fn check_conditions_with_localization(
        &self,
        data: &Value,
        handler: &dyn ApprovalHandler,
        llm_registry: Option<&LLMRegistry>,
    ) -> Result<HITLCheckResult> {
        for condition in &self.config.conditions {
            if condition.require_approval && self.evaluate_condition(&condition.when, data) {
                let mut context = HashMap::new();
                context.insert(
                    "condition_name".to_string(),
                    Value::String(condition.name.clone()),
                );
                context.insert(
                    "condition_expression".to_string(),
                    Value::String(condition.when.clone()),
                );
                if let Some(obj) = data.as_object() {
                    for (k, v) in obj {
                        context.insert(k.clone(), v.clone());
                    }
                }

                let message = if condition.approval_message.is_empty() {
                    format!("Condition '{}' matched", condition.name)
                } else {
                    let effective_config = condition
                        .message_language
                        .as_ref()
                        .unwrap_or(&self.config.message_language);

                    let mut resolver = MessageResolver::new(effective_config);
                    if let Some(registry) = llm_registry {
                        resolver = resolver.with_llm_registry(registry);
                    }

                    resolver
                        .resolve(
                            &condition.approval_message,
                            condition.message_language.as_ref(),
                            &context,
                            handler,
                        )
                        .await?
                };

                return Ok(HITLCheckResult::required(
                    ApprovalTrigger::condition(&condition.name, &condition.when),
                    context,
                    message,
                    Some(self.config.default_timeout_seconds),
                ));
            }
        }
        Ok(HITLCheckResult::not_required())
    }

    pub fn check_state_transition(&self, from: Option<&str>, to: &str) -> HITLCheckResult {
        if let Some(state_config) = self.config.states.get(to) {
            if state_config.on_enter == StateApprovalTrigger::RequireApproval {
                let message = state_config
                    .approval_message
                    .get_any()
                    .unwrap_or_else(|| format!("Approve transition to state '{}'?", to));

                let mut context = HashMap::new();
                if let Some(from_state) = from {
                    context.insert(
                        "from_state".to_string(),
                        Value::String(from_state.to_string()),
                    );
                }
                context.insert("to_state".to_string(), Value::String(to.to_string()));

                return HITLCheckResult::required(
                    ApprovalTrigger::state(from.map(String::from), to),
                    context,
                    message,
                    Some(self.config.default_timeout_seconds),
                );
            }
        }
        HITLCheckResult::not_required()
    }

    pub async fn check_state_transition_with_localization(
        &self,
        from: Option<&str>,
        to: &str,
        handler: &dyn ApprovalHandler,
        llm_registry: Option<&LLMRegistry>,
    ) -> Result<HITLCheckResult> {
        if let Some(state_config) = self.config.states.get(to) {
            if state_config.on_enter == StateApprovalTrigger::RequireApproval {
                let mut context = HashMap::new();
                if let Some(from_state) = from {
                    context.insert(
                        "from_state".to_string(),
                        Value::String(from_state.to_string()),
                    );
                }
                context.insert("to_state".to_string(), Value::String(to.to_string()));

                let message = if state_config.approval_message.is_empty() {
                    format!("Approve transition to state '{}'?", to)
                } else {
                    let effective_config = state_config
                        .message_language
                        .as_ref()
                        .unwrap_or(&self.config.message_language);

                    let mut resolver = MessageResolver::new(effective_config);
                    if let Some(registry) = llm_registry {
                        resolver = resolver.with_llm_registry(registry);
                    }

                    resolver
                        .resolve(
                            &state_config.approval_message,
                            state_config.message_language.as_ref(),
                            &context,
                            handler,
                        )
                        .await?
                };

                return Ok(HITLCheckResult::required(
                    ApprovalTrigger::state(from.map(String::from), to),
                    context,
                    message,
                    Some(self.config.default_timeout_seconds),
                ));
            }
        }
        Ok(HITLCheckResult::not_required())
    }

    fn build_tool_context(
        &self,
        config: &ToolApprovalConfig,
        args: &Value,
    ) -> HashMap<String, Value> {
        let mut context = HashMap::new();

        if config.approval_context.is_empty() {
            if let Some(obj) = args.as_object() {
                for (k, v) in obj {
                    context.insert(k.clone(), v.clone());
                }
            }
        } else {
            for field in &config.approval_context {
                if let Some(value) = args.get(field) {
                    context.insert(field.clone(), value.clone());
                }
            }
        }

        context
    }

    fn evaluate_condition(&self, expr: &str, data: &Value) -> bool {
        let expr = expr.trim();

        for (op_str, op_fn) in [
            (">=", compare_gte as fn(f64, f64) -> bool),
            ("<=", compare_lte),
            ("!=", compare_neq),
            (">", compare_gt),
            ("<", compare_lt),
            ("==", compare_eq),
        ] {
            if let Some((field, value_str)) = expr.split_once(op_str) {
                let field = field.trim();
                let value_str = value_str.trim();

                if let Some(field_value) = self.get_field_value(data, field) {
                    if let Some(compare_value) = parse_number(value_str) {
                        return op_fn(field_value, compare_value);
                    }
                }
            }
        }

        if let Some((field, list_str)) = expr.split_once(" in ") {
            let field = field.trim();
            let list_str = list_str.trim();
            if list_str.starts_with('[') && list_str.ends_with(']') {
                let inner = &list_str[1..list_str.len() - 1];
                let allowed: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();

                if let Some(value) = data.get(field) {
                    if let Some(s) = value.as_str() {
                        return allowed.contains(&s);
                    }
                }
            }
        }

        if let Some((field, list_str)) = expr.split_once(" not in ") {
            let field = field.trim();
            let list_str = list_str.trim();
            if list_str.starts_with('[') && list_str.ends_with(']') {
                let inner = &list_str[1..list_str.len() - 1];
                let blocked: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();

                if let Some(value) = data.get(field) {
                    if let Some(s) = value.as_str() {
                        return !blocked.contains(&s);
                    }
                }
            }
        }

        false
    }

    fn get_field_value(&self, data: &Value, field: &str) -> Option<f64> {
        let value = if field.contains('.') {
            let parts: Vec<&str> = field.split('.').collect();
            let mut current = data;
            for part in parts {
                current = current.get(part)?;
            }
            current
        } else {
            data.get(field)?
        };

        value.as_f64().or_else(|| value.as_i64().map(|i| i as f64))
    }
}

fn parse_number(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

fn compare_gt(a: f64, b: f64) -> bool {
    a > b
}
fn compare_lt(a: f64, b: f64) -> bool {
    a < b
}
fn compare_gte(a: f64, b: f64) -> bool {
    a >= b
}
fn compare_lte(a: f64, b: f64) -> bool {
    a <= b
}
fn compare_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < f64::EPSILON
}
fn compare_neq(a: f64, b: f64) -> bool {
    (a - b).abs() >= f64::EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hitl::ApprovalResult;
    use crate::hitl::config::{
        ApprovalCondition, ApprovalMessage, MessageLanguageConfig, MessageLanguageStrategy,
        StateApprovalConfig, TimeoutAction, ToolApprovalConfig,
    };

    struct TestHandler {
        language: Option<String>,
    }

    impl TestHandler {
        fn new() -> Self {
            Self { language: None }
        }

        fn with_language(language: impl Into<String>) -> Self {
            Self {
                language: Some(language.into()),
            }
        }
    }

    #[async_trait::async_trait]
    impl ApprovalHandler for TestHandler {
        async fn request_approval(&self, _request: crate::hitl::ApprovalRequest) -> ApprovalResult {
            ApprovalResult::Approved
        }

        fn preferred_language(&self) -> Option<String> {
            self.language.clone()
        }
    }

    fn create_test_config() -> HITLConfig {
        let mut tools = HashMap::new();
        tools.insert(
            "send_payment".to_string(),
            ToolApprovalConfig {
                require_approval: true,
                approval_context: vec!["amount".to_string(), "recipient".to_string()],
                approval_message: ApprovalMessage::simple("Approve payment?"),
                message_language: None,
                timeout_seconds: Some(120),
            },
        );
        tools.insert(
            "delete_record".to_string(),
            ToolApprovalConfig {
                require_approval: true,
                approval_context: vec![],
                approval_message: ApprovalMessage::default(),
                message_language: None,
                timeout_seconds: None,
            },
        );

        let conditions = vec![ApprovalCondition {
            name: "high_value".to_string(),
            when: "amount > 1000".to_string(),
            require_approval: true,
            approval_message: ApprovalMessage::simple("High value transaction"),
            message_language: None,
        }];

        let mut states = HashMap::new();
        states.insert(
            "escalation".to_string(),
            StateApprovalConfig {
                on_enter: StateApprovalTrigger::RequireApproval,
                approval_message: ApprovalMessage::simple("Escalate?"),
                message_language: None,
            },
        );

        HITLConfig {
            default_timeout_seconds: 300,
            on_timeout: TimeoutAction::Reject,
            message_language: MessageLanguageConfig::default(),
            tools,
            conditions,
            states,
        }
    }

    #[test]
    fn test_check_tool_requires_approval() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);

        let args = serde_json::json!({
            "amount": 500,
            "recipient": "John",
            "extra": "ignored"
        });

        let result = engine.check_tool("send_payment", &args);
        assert!(result.is_required());

        if let HITLCheckResult::Required {
            trigger,
            context,
            message,
            timeout,
        } = result
        {
            assert_eq!(trigger.trigger_type(), "tool");
            assert_eq!(context.get("amount").unwrap(), &Value::Number(500.into()));
            assert_eq!(
                context.get("recipient").unwrap(),
                &Value::String("John".to_string())
            );
            assert!(context.get("extra").is_none());
            assert_eq!(message, "Approve payment?");
            assert_eq!(timeout, Some(120));
        }
    }

    #[test]
    fn test_check_tool_not_configured() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);

        let result = engine.check_tool("unknown_tool", &serde_json::json!({}));
        assert!(!result.is_required());
    }

    #[test]
    fn test_check_tool_all_args_when_empty_context() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);

        let args = serde_json::json!({
            "id": 123,
            "name": "test"
        });

        let result = engine.check_tool("delete_record", &args);
        if let HITLCheckResult::Required { context, .. } = result {
            assert_eq!(context.get("id").unwrap(), &Value::Number(123.into()));
            assert_eq!(
                context.get("name").unwrap(),
                &Value::String("test".to_string())
            );
        }
    }

    #[test]
    fn test_check_condition_greater_than() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);

        let high_value = serde_json::json!({ "amount": 2000 });
        let result = engine.check_conditions(&high_value);
        assert!(result.is_required());

        let low_value = serde_json::json!({ "amount": 500 });
        let result = engine.check_conditions(&low_value);
        assert!(!result.is_required());
    }

    #[test]
    fn test_check_condition_greater_or_equal() {
        let mut config = create_test_config();
        config.conditions[0].when = "amount >= 1000".to_string();
        let engine = HITLEngine::new(config);

        let exact = serde_json::json!({ "amount": 1000 });
        assert!(engine.check_conditions(&exact).is_required());

        let below = serde_json::json!({ "amount": 999 });
        assert!(!engine.check_conditions(&below).is_required());
    }

    #[test]
    fn test_check_state_transition_requires_approval() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);

        let result = engine.check_state_transition(Some("greeting"), "escalation");
        assert!(result.is_required());

        if let HITLCheckResult::Required {
            trigger,
            context,
            message,
            ..
        } = result
        {
            assert_eq!(trigger.trigger_type(), "state");
            assert_eq!(
                context.get("from_state").unwrap(),
                &Value::String("greeting".to_string())
            );
            assert_eq!(
                context.get("to_state").unwrap(),
                &Value::String("escalation".to_string())
            );
            assert_eq!(message, "Escalate?");
        }
    }

    #[test]
    fn test_check_state_transition_not_configured() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);

        let result = engine.check_state_transition(None, "unknown_state");
        assert!(!result.is_required());
    }

    #[test]
    fn test_evaluate_condition_operators() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);
        let data = serde_json::json!({ "value": 100 });

        assert!(engine.evaluate_condition("value > 50", &data));
        assert!(engine.evaluate_condition("value >= 100", &data));
        assert!(engine.evaluate_condition("value < 200", &data));
        assert!(engine.evaluate_condition("value <= 100", &data));
        assert!(engine.evaluate_condition("value == 100", &data));
        assert!(engine.evaluate_condition("value != 99", &data));
    }

    #[test]
    fn test_evaluate_condition_nested_field() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);
        let data = serde_json::json!({
            "order": {
                "total": 500
            }
        });

        assert!(engine.evaluate_condition("order.total > 100", &data));
    }

    #[test]
    fn test_check_conditions_returns_first_match() {
        let mut config = create_test_config();
        config.conditions.push(ApprovalCondition {
            name: "very_high".to_string(),
            when: "amount > 5000".to_string(),
            require_approval: true,
            approval_message: ApprovalMessage::simple("Very high value"),
            message_language: None,
        });
        let engine = HITLEngine::new(config);

        let data = serde_json::json!({ "amount": 6000 });
        let result = engine.check_conditions(&data);

        if let HITLCheckResult::Required { message, .. } = result {
            assert_eq!(message, "High value transaction");
        }
    }

    #[test]
    fn test_default_timeout_used() {
        let mut config = create_test_config();
        config.tools.insert(
            "no_timeout".to_string(),
            ToolApprovalConfig {
                require_approval: true,
                approval_context: vec![],
                approval_message: ApprovalMessage::default(),
                message_language: None,
                timeout_seconds: None,
            },
        );
        let engine = HITLEngine::new(config);

        let result = engine.check_tool("no_timeout", &serde_json::json!({}));
        if let HITLCheckResult::Required { timeout, .. } = result {
            assert_eq!(timeout, Some(300));
        }
    }

    #[test]
    fn test_engine_config_accessor() {
        let config = create_test_config();
        let engine = HITLEngine::new(config);

        assert_eq!(engine.config().default_timeout_seconds, 300);
        assert_eq!(engine.config().on_timeout, TimeoutAction::Reject);
    }

    #[test]
    fn test_hitl_check_result_into_request() {
        let trigger = ApprovalTrigger::tool("test", serde_json::json!({}));
        let result = HITLCheckResult::required(trigger, HashMap::new(), "Test?", Some(60));

        let request = result.into_request().unwrap();
        assert_eq!(request.message, "Test?");
    }

    #[test]
    fn test_condition_in_list() {
        let mut config = create_test_config();
        config.conditions[0].when = "status in [pending, review]".to_string();
        let engine = HITLEngine::new(config);

        let pending = serde_json::json!({ "status": "pending" });
        assert!(engine.check_conditions(&pending).is_required());

        let review = serde_json::json!({ "status": "review" });
        assert!(engine.check_conditions(&review).is_required());

        let approved = serde_json::json!({ "status": "approved" });
        assert!(!engine.check_conditions(&approved).is_required());
    }

    #[test]
    fn test_condition_not_in_list() {
        let mut config = create_test_config();
        config.conditions[0].when = "status not in [approved, completed]".to_string();
        let engine = HITLEngine::new(config);

        let pending = serde_json::json!({ "status": "pending" });
        assert!(engine.check_conditions(&pending).is_required());

        let approved = serde_json::json!({ "status": "approved" });
        assert!(!engine.check_conditions(&approved).is_required());
    }

    #[tokio::test]
    async fn test_check_tool_with_localization_korean() {
        let mut config = create_test_config();

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "Approve payment?".to_string());
        messages.insert("ko".to_string(), "결제를 승인하시겠습니까?".to_string());

        config.tools.insert(
            "localized_payment".to_string(),
            ToolApprovalConfig {
                require_approval: true,
                approval_context: vec![],
                approval_message: ApprovalMessage::MultiLanguage {
                    messages,
                    description: None,
                },
                message_language: Some(MessageLanguageConfig {
                    strategy: MessageLanguageStrategy::Approver,
                    fallback: vec![],
                    explicit: None,
                    llm_generate: None,
                }),
                timeout_seconds: None,
            },
        );

        let engine = HITLEngine::new(config);
        let handler = TestHandler::with_language("ko");

        let result = engine
            .check_tool_with_localization(
                "localized_payment",
                &serde_json::json!({}),
                &handler,
                None,
            )
            .await
            .unwrap();

        if let HITLCheckResult::Required { message, .. } = result {
            assert_eq!(message, "결제를 승인하시겠습니까?");
        } else {
            panic!("Expected Required result");
        }
    }

    #[tokio::test]
    async fn test_check_state_transition_with_localization() {
        let mut config = create_test_config();

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "Escalate to human?".to_string());
        messages.insert(
            "ja".to_string(),
            "人間にエスカレーションしますか？".to_string(),
        );

        config.states.insert(
            "localized_escalation".to_string(),
            StateApprovalConfig {
                on_enter: StateApprovalTrigger::RequireApproval,
                approval_message: ApprovalMessage::MultiLanguage {
                    messages,
                    description: None,
                },
                message_language: Some(MessageLanguageConfig {
                    strategy: MessageLanguageStrategy::Approver,
                    fallback: vec![],
                    explicit: None,
                    llm_generate: None,
                }),
            },
        );

        let engine = HITLEngine::new(config);
        let handler = TestHandler::with_language("ja");

        let result = engine
            .check_state_transition_with_localization(
                Some("greeting"),
                "localized_escalation",
                &handler,
                None,
            )
            .await
            .unwrap();

        if let HITLCheckResult::Required { message, .. } = result {
            assert_eq!(message, "人間にエスカレーションしますか？");
        } else {
            panic!("Expected Required result");
        }
    }

    #[tokio::test]
    async fn test_check_conditions_with_localization() {
        let mut config = create_test_config();

        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "High value: {{ amount }}".to_string());
        messages.insert("ko".to_string(), "고액: {{ amount }}".to_string());

        config.conditions[0].approval_message = ApprovalMessage::MultiLanguage {
            messages,
            description: None,
        };
        config.conditions[0].message_language = Some(MessageLanguageConfig {
            strategy: MessageLanguageStrategy::User,
            fallback: vec![],
            explicit: None,
            llm_generate: None,
        });

        let engine = HITLEngine::new(config);
        let handler = TestHandler::new();

        let mut context_map = serde_json::Map::new();
        context_map.insert("amount".to_string(), serde_json::json!(2000));
        context_map.insert("user.language".to_string(), serde_json::json!("ko"));

        let data = Value::Object(context_map);

        let result = engine
            .check_conditions_with_localization(&data, &handler, None)
            .await
            .unwrap();

        if let HITLCheckResult::Required { message, .. } = result {
            assert_eq!(message, "고액: 2000");
        } else {
            panic!("Expected Required result");
        }
    }
}
