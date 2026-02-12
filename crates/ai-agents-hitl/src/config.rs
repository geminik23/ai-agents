use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HITLConfig {
    #[serde(default = "default_hitl_timeout")]
    pub default_timeout_seconds: u64,

    #[serde(default)]
    pub on_timeout: TimeoutAction,

    #[serde(default)]
    pub message_language: MessageLanguageConfig,

    #[serde(default)]
    pub tools: HashMap<String, ToolApprovalConfig>,

    #[serde(default)]
    pub conditions: Vec<ApprovalCondition>,

    #[serde(default)]
    pub states: HashMap<String, StateApprovalConfig>,
}

fn default_hitl_timeout() -> u64 {
    300
}

impl Default for HITLConfig {
    fn default() -> Self {
        Self {
            default_timeout_seconds: default_hitl_timeout(),
            on_timeout: TimeoutAction::default(),
            message_language: MessageLanguageConfig::default(),
            tools: HashMap::new(),
            conditions: Vec::new(),
            states: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TimeoutAction {
    #[default]
    Reject,
    Approve,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageLanguageStrategy {
    #[default]
    Auto,
    User,
    Approver,
    Explicit,
    LlmGenerate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageLanguageConfig {
    #[serde(default)]
    pub strategy: MessageLanguageStrategy,

    #[serde(default = "default_fallback_chain")]
    pub fallback: Vec<MessageLanguageStrategy>,

    #[serde(default)]
    pub explicit: Option<String>,

    #[serde(default)]
    pub llm_generate: Option<LlmGenerateConfig>,
}

fn default_fallback_chain() -> Vec<MessageLanguageStrategy> {
    vec![
        MessageLanguageStrategy::Approver,
        MessageLanguageStrategy::User,
        MessageLanguageStrategy::Explicit,
        MessageLanguageStrategy::LlmGenerate,
    ]
}

impl Default for MessageLanguageConfig {
    fn default() -> Self {
        Self {
            strategy: MessageLanguageStrategy::default(),
            fallback: default_fallback_chain(),
            explicit: None,
            llm_generate: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmGenerateConfig {
    #[serde(default = "default_router")]
    pub llm: String,

    #[serde(default = "default_true")]
    pub include_context: bool,
}

fn default_router() -> String {
    "router".to_string()
}

fn default_true() -> bool {
    true
}

impl Default for LlmGenerateConfig {
    fn default() -> Self {
        Self {
            llm: default_router(),
            include_context: default_true(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApprovalMessage {
    Simple(String),
    MultiLanguage {
        #[serde(flatten)]
        messages: HashMap<String, String>,
        #[serde(default)]
        description: Option<String>,
    },
}

impl Default for ApprovalMessage {
    fn default() -> Self {
        ApprovalMessage::Simple(String::new())
    }
}

impl ApprovalMessage {
    pub fn simple(message: impl Into<String>) -> Self {
        ApprovalMessage::Simple(message.into())
    }

    pub fn multi_language(messages: HashMap<String, String>) -> Self {
        ApprovalMessage::MultiLanguage {
            messages,
            description: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        if let ApprovalMessage::MultiLanguage {
            ref mut description,
            ..
        } = self
        {
            *description = Some(desc.into());
        }
        self
    }

    pub fn get(&self, lang: &str) -> Option<String> {
        match self {
            ApprovalMessage::Simple(s) => Some(s.clone()),
            ApprovalMessage::MultiLanguage { messages, .. } => messages.get(lang).cloned(),
        }
    }

    pub fn get_any(&self) -> Option<String> {
        match self {
            ApprovalMessage::Simple(s) if !s.is_empty() => Some(s.clone()),
            ApprovalMessage::Simple(_) => None,
            ApprovalMessage::MultiLanguage { messages, .. } => messages
                .get("en")
                .cloned()
                .or_else(|| messages.values().next().cloned()),
        }
    }

    pub fn description(&self) -> Option<&str> {
        match self {
            ApprovalMessage::Simple(_) => None,
            ApprovalMessage::MultiLanguage { description, .. } => description.as_deref(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            ApprovalMessage::Simple(s) => s.is_empty(),
            ApprovalMessage::MultiLanguage { messages, .. } => messages.is_empty(),
        }
    }

    pub fn available_languages(&self) -> Vec<String> {
        match self {
            ApprovalMessage::Simple(_) => vec![],
            ApprovalMessage::MultiLanguage { messages, .. } => messages.keys().cloned().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolApprovalConfig {
    #[serde(default)]
    pub require_approval: bool,

    #[serde(default)]
    pub approval_context: Vec<String>,

    #[serde(default)]
    pub approval_message: ApprovalMessage,

    #[serde(default)]
    pub message_language: Option<MessageLanguageConfig>,

    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalCondition {
    pub name: String,

    pub when: String,

    #[serde(default)]
    pub require_approval: bool,

    #[serde(default)]
    pub approval_message: ApprovalMessage,

    #[serde(default)]
    pub message_language: Option<MessageLanguageConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateApprovalConfig {
    #[serde(default)]
    pub on_enter: StateApprovalTrigger,

    #[serde(default)]
    pub approval_message: ApprovalMessage,

    #[serde(default)]
    pub message_language: Option<MessageLanguageConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StateApprovalTrigger {
    #[default]
    None,
    RequireApproval,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hitl_config_default() {
        let config = HITLConfig::default();
        assert_eq!(config.default_timeout_seconds, 300);
        assert_eq!(config.on_timeout, TimeoutAction::Reject);
        assert!(config.tools.is_empty());
        assert!(config.conditions.is_empty());
        assert!(config.states.is_empty());
        assert_eq!(
            config.message_language.strategy,
            MessageLanguageStrategy::Auto
        );
    }

    #[test]
    fn test_hitl_config_from_yaml() {
        let yaml = r#"
default_timeout_seconds: 600
on_timeout: approve
message_language:
  strategy: approver
  fallback:
    - user
    - explicit
  explicit: en
tools:
  send_payment:
    require_approval: true
    approval_context:
      - amount
      - recipient
    approval_message: "Approve payment?"
    timeout_seconds: 120
  delete_record:
    require_approval: true
conditions:
  - name: high_value
    when: "amount > 1000"
    require_approval: true
    approval_message: "High value transaction"
states:
  escalation:
    on_enter: require_approval
    approval_message: "Escalate to human?"
"#;
        let config: HITLConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.default_timeout_seconds, 600);
        assert_eq!(config.on_timeout, TimeoutAction::Approve);
        assert_eq!(
            config.message_language.strategy,
            MessageLanguageStrategy::Approver
        );
        assert_eq!(config.message_language.explicit, Some("en".to_string()));
        assert_eq!(config.tools.len(), 2);

        let payment = config.tools.get("send_payment").unwrap();
        assert!(payment.require_approval);
        assert_eq!(payment.approval_context, vec!["amount", "recipient"]);
        assert_eq!(payment.timeout_seconds, Some(120));

        assert_eq!(config.conditions.len(), 1);
        assert_eq!(config.conditions[0].name, "high_value");

        let escalation = config.states.get("escalation").unwrap();
        assert_eq!(escalation.on_enter, StateApprovalTrigger::RequireApproval);
    }

    #[test]
    fn test_multi_language_approval_message() {
        let yaml = r#"
tools:
  process_payment:
    require_approval: true
    approval_message:
      en: "Approve payment of {{ amount }}?"
      ko: "{{ amount }} 결제를 승인하시겠습니까?"
      ja: "{{ amount }}の支払いを承認しますか？"
      description: "Payment approval request"
"#;
        let config: HITLConfig = serde_yaml::from_str(yaml).unwrap();
        let payment = config.tools.get("process_payment").unwrap();

        let msg = &payment.approval_message;
        assert_eq!(
            msg.get("en"),
            Some("Approve payment of {{ amount }}?".to_string())
        );
        assert_eq!(
            msg.get("ko"),
            Some("{{ amount }} 결제를 승인하시겠습니까?".to_string())
        );
        assert_eq!(
            msg.get("ja"),
            Some("{{ amount }}の支払いを承認しますか？".to_string())
        );
        assert_eq!(msg.description(), Some("Payment approval request"));
    }

    #[test]
    fn test_tool_level_message_language_override() {
        let yaml = r#"
message_language:
  strategy: auto
tools:
  admin_action:
    require_approval: true
    message_language:
      strategy: explicit
      explicit: en
    approval_message:
      en: "Admin action required"
"#;
        let config: HITLConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            config.message_language.strategy,
            MessageLanguageStrategy::Auto
        );

        let admin = config.tools.get("admin_action").unwrap();
        let tool_lang = admin.message_language.as_ref().unwrap();
        assert_eq!(tool_lang.strategy, MessageLanguageStrategy::Explicit);
        assert_eq!(tool_lang.explicit, Some("en".to_string()));
    }

    #[test]
    fn test_llm_generate_config() {
        let yaml = r#"
message_language:
  strategy: llm_generate
  llm_generate:
    llm: router
    include_context: true
tools:
  dynamic_action:
    require_approval: true
    approval_message:
      description: "Dynamic action requiring approval"
"#;
        let config: HITLConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            config.message_language.strategy,
            MessageLanguageStrategy::LlmGenerate
        );

        let llm_config = config.message_language.llm_generate.as_ref().unwrap();
        assert_eq!(llm_config.llm, "router");
        assert!(llm_config.include_context);
    }

    #[test]
    fn test_timeout_action_variants() {
        let yaml_reject = "reject";
        let yaml_approve = "approve";
        let yaml_error = "error";

        let action: TimeoutAction = serde_yaml::from_str(yaml_reject).unwrap();
        assert_eq!(action, TimeoutAction::Reject);

        let action: TimeoutAction = serde_yaml::from_str(yaml_approve).unwrap();
        assert_eq!(action, TimeoutAction::Approve);

        let action: TimeoutAction = serde_yaml::from_str(yaml_error).unwrap();
        assert_eq!(action, TimeoutAction::Error);
    }

    #[test]
    fn test_message_language_strategy_variants() {
        assert_eq!(
            serde_yaml::from_str::<MessageLanguageStrategy>("auto").unwrap(),
            MessageLanguageStrategy::Auto
        );
        assert_eq!(
            serde_yaml::from_str::<MessageLanguageStrategy>("user").unwrap(),
            MessageLanguageStrategy::User
        );
        assert_eq!(
            serde_yaml::from_str::<MessageLanguageStrategy>("approver").unwrap(),
            MessageLanguageStrategy::Approver
        );
        assert_eq!(
            serde_yaml::from_str::<MessageLanguageStrategy>("explicit").unwrap(),
            MessageLanguageStrategy::Explicit
        );
        assert_eq!(
            serde_yaml::from_str::<MessageLanguageStrategy>("llm_generate").unwrap(),
            MessageLanguageStrategy::LlmGenerate
        );
    }

    #[test]
    fn test_approval_message_simple() {
        let msg = ApprovalMessage::simple("Approve?");
        assert_eq!(msg.get("en"), Some("Approve?".to_string()));
        assert_eq!(msg.get("ko"), Some("Approve?".to_string()));
        assert!(msg.description().is_none());
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_approval_message_multi_language() {
        let mut messages = HashMap::new();
        messages.insert("en".to_string(), "Approve?".to_string());
        messages.insert("ko".to_string(), "승인?".to_string());

        let msg = ApprovalMessage::multi_language(messages).with_description("Test approval");
        assert_eq!(msg.get("en"), Some("Approve?".to_string()));
        assert_eq!(msg.get("ko"), Some("승인?".to_string()));
        assert_eq!(msg.get("ja"), None);
        assert_eq!(msg.description(), Some("Test approval"));
        assert!(!msg.is_empty());

        let langs = msg.available_languages();
        assert!(langs.contains(&"en".to_string()));
        assert!(langs.contains(&"ko".to_string()));
    }

    #[test]
    fn test_approval_message_get_any() {
        let msg = ApprovalMessage::simple("Test");
        assert_eq!(msg.get_any(), Some("Test".to_string()));

        let empty = ApprovalMessage::simple("");
        assert_eq!(empty.get_any(), None);

        let mut messages = HashMap::new();
        messages.insert("ko".to_string(), "한국어".to_string());
        let msg = ApprovalMessage::multi_language(messages);
        assert_eq!(msg.get_any(), Some("한국어".to_string()));

        let mut messages_with_en = HashMap::new();
        messages_with_en.insert("en".to_string(), "English".to_string());
        messages_with_en.insert("ko".to_string(), "한국어".to_string());
        let msg = ApprovalMessage::multi_language(messages_with_en);
        assert_eq!(msg.get_any(), Some("English".to_string()));
    }

    #[test]
    fn test_tool_approval_config_default() {
        let config = ToolApprovalConfig::default();
        assert!(!config.require_approval);
        assert!(config.approval_context.is_empty());
        assert!(config.approval_message.is_empty());
        assert!(config.message_language.is_none());
        assert!(config.timeout_seconds.is_none());
    }

    #[test]
    fn test_backward_compatible_simple_message() {
        let yaml = r#"
tools:
  old_tool:
    require_approval: true
    approval_message: "Simple approval message"
"#;
        let config: HITLConfig = serde_yaml::from_str(yaml).unwrap();
        let tool = config.tools.get("old_tool").unwrap();
        assert_eq!(
            tool.approval_message.get_any(),
            Some("Simple approval message".to_string())
        );
    }

    #[test]
    fn test_condition_with_multi_language() {
        let yaml = r#"
conditions:
  - name: high_value
    when: "amount > 1000"
    require_approval: true
    approval_message:
      en: "High value: {{ amount }}"
      ko: "고액 거래: {{ amount }}"
    message_language:
      strategy: user
"#;
        let config: HITLConfig = serde_yaml::from_str(yaml).unwrap();
        let condition = &config.conditions[0];
        assert_eq!(condition.name, "high_value");
        assert_eq!(
            condition.approval_message.get("en"),
            Some("High value: {{ amount }}".to_string())
        );
        assert_eq!(
            condition.message_language.as_ref().unwrap().strategy,
            MessageLanguageStrategy::User
        );
    }

    #[test]
    fn test_state_with_multi_language() {
        let yaml = r#"
states:
  escalation:
    on_enter: require_approval
    approval_message:
      en: "Escalate to human agent?"
      ko: "상담원에게 연결하시겠습니까?"
      ja: "人間のエージェントにエスカレーションしますか？"
    message_language:
      strategy: approver
"#;
        let config: HITLConfig = serde_yaml::from_str(yaml).unwrap();
        let state = config.states.get("escalation").unwrap();
        assert_eq!(state.on_enter, StateApprovalTrigger::RequireApproval);
        assert_eq!(
            state.approval_message.get("ko"),
            Some("상담원에게 연결하시겠습니까?".to_string())
        );
    }

    #[test]
    fn test_default_fallback_chain() {
        let chain = default_fallback_chain();
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0], MessageLanguageStrategy::Approver);
        assert_eq!(chain[1], MessageLanguageStrategy::User);
        assert_eq!(chain[2], MessageLanguageStrategy::Explicit);
        assert_eq!(chain[3], MessageLanguageStrategy::LlmGenerate);
    }

    #[test]
    fn test_llm_generate_config_defaults() {
        let config = LlmGenerateConfig::default();
        assert_eq!(config.llm, "router");
        assert!(config.include_context);
    }
}
