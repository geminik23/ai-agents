use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub id: String,
    pub trigger: ApprovalTrigger,
    pub context: HashMap<String, Value>,
    pub message: String,
    pub timeout: Option<Duration>,
}

impl ApprovalRequest {
    pub fn new(trigger: ApprovalTrigger, message: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            trigger,
            context: HashMap::new(),
            message: message.into(),
            timeout: None,
        }
    }

    pub fn with_context(mut self, context: HashMap<String, Value>) -> Self {
        self.context = context;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_timeout_seconds(mut self, seconds: u64) -> Self {
        self.timeout = Some(Duration::from_secs(seconds));
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApprovalTrigger {
    Tool { name: String, args: Value },
    Condition { name: String, matched: String },
    State { from: Option<String>, to: String },
}

impl ApprovalTrigger {
    pub fn tool(name: impl Into<String>, args: Value) -> Self {
        Self::Tool {
            name: name.into(),
            args,
        }
    }

    pub fn condition(name: impl Into<String>, matched: impl Into<String>) -> Self {
        Self::Condition {
            name: name.into(),
            matched: matched.into(),
        }
    }

    pub fn state(from: Option<String>, to: impl Into<String>) -> Self {
        Self::State {
            from,
            to: to.into(),
        }
    }

    pub fn trigger_type(&self) -> &'static str {
        match self {
            Self::Tool { .. } => "tool",
            Self::Condition { .. } => "condition",
            Self::State { .. } => "state",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ApprovalResult {
    Approved,
    Rejected { reason: Option<String> },
    Modified { changes: HashMap<String, Value> },
    Timeout,
}

impl ApprovalResult {
    pub fn approved() -> Self {
        Self::Approved
    }

    pub fn rejected(reason: Option<String>) -> Self {
        Self::Rejected { reason }
    }

    pub fn rejected_with_reason(reason: impl Into<String>) -> Self {
        Self::Rejected {
            reason: Some(reason.into()),
        }
    }

    pub fn modified(changes: HashMap<String, Value>) -> Self {
        Self::Modified { changes }
    }

    pub fn timeout() -> Self {
        Self::Timeout
    }

    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved | Self::Modified { .. })
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }

    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout)
    }
}

#[derive(Debug, Clone)]
pub enum HITLCheckResult {
    NotRequired,
    Required {
        trigger: ApprovalTrigger,
        context: HashMap<String, Value>,
        message: String,
        timeout: Option<u64>,
    },
}

impl HITLCheckResult {
    pub fn not_required() -> Self {
        Self::NotRequired
    }

    pub fn required(
        trigger: ApprovalTrigger,
        context: HashMap<String, Value>,
        message: impl Into<String>,
        timeout: Option<u64>,
    ) -> Self {
        Self::Required {
            trigger,
            context,
            message: message.into(),
            timeout,
        }
    }

    pub fn is_required(&self) -> bool {
        matches!(self, Self::Required { .. })
    }

    pub fn into_request(self) -> Option<ApprovalRequest> {
        match self {
            Self::NotRequired => None,
            Self::Required {
                trigger,
                context,
                message,
                timeout,
            } => {
                let mut request = ApprovalRequest::new(trigger, message).with_context(context);
                if let Some(secs) = timeout {
                    request = request.with_timeout_seconds(secs);
                }
                Some(request)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approval_request_new() {
        let trigger = ApprovalTrigger::tool("test_tool", serde_json::json!({"key": "value"}));
        let request = ApprovalRequest::new(trigger, "Test message");
        assert!(!request.id.is_empty());
        assert_eq!(request.message, "Test message");
        assert!(request.context.is_empty());
        assert!(request.timeout.is_none());
    }

    #[test]
    fn test_approval_request_with_timeout() {
        let trigger = ApprovalTrigger::tool("test", serde_json::json!({}));
        let request = ApprovalRequest::new(trigger, "Test").with_timeout_seconds(60);
        assert_eq!(request.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_approval_trigger_tool() {
        let trigger = ApprovalTrigger::tool("send_payment", serde_json::json!({"amount": 100}));
        assert_eq!(trigger.trigger_type(), "tool");
        if let ApprovalTrigger::Tool { name, args } = trigger {
            assert_eq!(name, "send_payment");
            assert_eq!(args["amount"], 100);
        } else {
            panic!("Expected Tool trigger");
        }
    }

    #[test]
    fn test_approval_trigger_condition() {
        let trigger = ApprovalTrigger::condition("high_value", "amount > 1000");
        assert_eq!(trigger.trigger_type(), "condition");
    }

    #[test]
    fn test_approval_trigger_state() {
        let trigger = ApprovalTrigger::state(Some("greeting".to_string()), "escalation");
        assert_eq!(trigger.trigger_type(), "state");
    }

    #[test]
    fn test_approval_result_approved() {
        let result = ApprovalResult::approved();
        assert!(result.is_approved());
        assert!(!result.is_rejected());
        assert!(!result.is_timeout());
    }

    #[test]
    fn test_approval_result_rejected() {
        let result = ApprovalResult::rejected_with_reason("User declined");
        assert!(!result.is_approved());
        assert!(result.is_rejected());
    }

    #[test]
    fn test_approval_result_modified() {
        let mut changes = HashMap::new();
        changes.insert("amount".to_string(), serde_json::json!(500));
        let result = ApprovalResult::modified(changes);
        assert!(result.is_approved());
        assert!(!result.is_rejected());
    }

    #[test]
    fn test_hitl_check_result_not_required() {
        let result = HITLCheckResult::not_required();
        assert!(!result.is_required());
        assert!(result.into_request().is_none());
    }

    #[test]
    fn test_hitl_check_result_required() {
        let trigger = ApprovalTrigger::tool("test", serde_json::json!({}));
        let result = HITLCheckResult::required(trigger, HashMap::new(), "Approve?", Some(60));
        assert!(result.is_required());

        let request = result.into_request().unwrap();
        assert_eq!(request.message, "Approve?");
        assert_eq!(request.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_approval_trigger_serialization() {
        let trigger = ApprovalTrigger::tool("payment", serde_json::json!({"amount": 100}));
        let json = serde_json::to_string(&trigger).unwrap();
        assert!(json.contains("tool"));
        assert!(json.contains("payment"));

        let parsed: ApprovalTrigger = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.trigger_type(), "tool");
    }

    #[test]
    fn test_approval_result_serialization() {
        let result = ApprovalResult::rejected_with_reason("Not allowed");
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("rejected"));
        assert!(json.contains("Not allowed"));

        let parsed: ApprovalResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_rejected());
    }
}
