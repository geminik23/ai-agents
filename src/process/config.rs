//! Process configuration types for input/output transformation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessConfig {
    #[serde(default)]
    pub input: Vec<ProcessStage>,
    #[serde(default)]
    pub output: Vec<ProcessStage>,
    #[serde(default)]
    pub settings: ProcessSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessStage {
    Normalize(NormalizeStage),
    Detect(DetectStage),
    Extract(ExtractStage),
    Sanitize(SanitizeStage),
    Transform(TransformStage),
    Validate(ValidateStage),
    Format(FormatStage),
    Enrich(EnrichStage),
    Conditional(ConditionalStage),
}

impl ProcessStage {
    pub fn condition(&self) -> Option<&ConditionExpr> {
        match self {
            ProcessStage::Normalize(s) => s.condition.as_ref(),
            ProcessStage::Detect(s) => s.condition.as_ref(),
            ProcessStage::Extract(s) => s.condition.as_ref(),
            ProcessStage::Sanitize(s) => s.condition.as_ref(),
            ProcessStage::Transform(s) => s.condition.as_ref(),
            ProcessStage::Validate(s) => s.condition.as_ref(),
            ProcessStage::Format(s) => s.condition.as_ref(),
            ProcessStage::Enrich(s) => s.condition.as_ref(),
            ProcessStage::Conditional(_) => None,
        }
    }

    pub fn id(&self) -> Option<&str> {
        match self {
            ProcessStage::Normalize(s) => s.id.as_deref(),
            ProcessStage::Detect(s) => s.id.as_deref(),
            ProcessStage::Extract(s) => s.id.as_deref(),
            ProcessStage::Sanitize(s) => s.id.as_deref(),
            ProcessStage::Transform(s) => s.id.as_deref(),
            ProcessStage::Validate(s) => s.id.as_deref(),
            ProcessStage::Format(s) => s.id.as_deref(),
            ProcessStage::Enrich(s) => s.id.as_deref(),
            ProcessStage::Conditional(s) => s.id.as_deref(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NormalizeStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: NormalizeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizeConfig {
    #[serde(default = "default_true")]
    pub trim: bool,
    #[serde(default)]
    pub unicode: Option<UnicodeNormalization>,
    #[serde(default)]
    pub collapse_whitespace: bool,
    #[serde(default)]
    pub lowercase: bool,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            trim: true,
            unicode: None,
            collapse_whitespace: false,
            lowercase: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnicodeNormalization {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DetectStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: DetectConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DetectConfig {
    #[serde(default)]
    pub llm: Option<String>,
    #[serde(default)]
    pub detect: Vec<DetectionType>,
    #[serde(default)]
    pub intents: Vec<IntentDefinition>,
    #[serde(default)]
    pub store_in_context: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionType {
    Language,
    Sentiment,
    Intent,
    Topic,
    Formality,
    Urgency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDefinition {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: ExtractConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractConfig {
    #[serde(default)]
    pub llm: Option<String>,
    #[serde(default)]
    pub schema: HashMap<String, FieldSchema>,
    #[serde(default)]
    pub store_in_context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldSchema {
    #[serde(rename = "type", default)]
    pub field_type: FieldType,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub values: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    #[default]
    String,
    Number,
    Integer,
    Boolean,
    Date,
    Enum,
    Array,
    Object,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SanitizeStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: SanitizeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SanitizeConfig {
    #[serde(default)]
    pub llm: Option<String>,
    #[serde(default)]
    pub pii: Option<PIISanitizeConfig>,
    #[serde(default)]
    pub harmful: Option<HarmfulContentConfig>,
    #[serde(default)]
    pub remove: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIISanitizeConfig {
    #[serde(default)]
    pub action: PIIAction,
    #[serde(default)]
    pub types: Vec<PIIType>,
    #[serde(default = "default_mask_char")]
    pub mask_char: String,
}

impl Default for PIISanitizeConfig {
    fn default() -> Self {
        Self {
            action: PIIAction::Mask,
            types: Vec::new(),
            mask_char: default_mask_char(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PIIAction {
    #[default]
    Mask,
    Remove,
    Flag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PIIType {
    Email,
    Phone,
    CreditCard,
    Ssn,
    IpAddress,
    Name,
    Address,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HarmfulContentConfig {
    #[serde(default)]
    pub detect: Vec<HarmfulContentType>,
    #[serde(default)]
    pub action: HarmfulAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarmfulContentType {
    HateSpeech,
    Violence,
    SexualContent,
    Harassment,
    SelfHarm,
    IllegalActivity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HarmfulAction {
    #[default]
    Flag,
    Block,
    Remove,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransformStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: TransformConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransformConfig {
    #[serde(default)]
    pub llm: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidateStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: ValidateConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidateConfig {
    #[serde(default)]
    pub rules: Vec<ValidationRule>,
    #[serde(default)]
    pub llm: Option<String>,
    #[serde(default)]
    pub criteria: Vec<String>,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default)]
    pub on_fail: ValidationFailAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ValidationRule {
    MinLength {
        min_length: usize,
        #[serde(default)]
        on_fail: ValidationAction,
    },
    MaxLength {
        max_length: usize,
        #[serde(default)]
        on_fail: ValidationAction,
    },
    Pattern {
        pattern: String,
        #[serde(default)]
        on_fail: ValidationAction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationAction {
    #[serde(default)]
    pub action: ValidationActionType,
    #[serde(default)]
    pub message: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValidationActionType {
    #[default]
    Reject,
    Truncate,
    Warn,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationFailAction {
    #[serde(default)]
    pub action: ValidationFailType,
    #[serde(default)]
    pub max_retries: Option<u32>,
    #[serde(default)]
    pub feedback_to_agent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValidationFailType {
    #[default]
    Reject,
    Regenerate,
    Warn,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FormatStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: FormatConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FormatConfig {
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub channels: HashMap<String, ChannelFormat>,
    #[serde(default)]
    pub channel: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChannelFormat {
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub format: Option<OutputFormat>,
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub markdown: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    #[default]
    Text,
    Html,
    Json,
    Markdown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnrichStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default)]
    pub config: EnrichConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnrichConfig {
    #[serde(default)]
    pub source: EnrichSource,
    #[serde(default)]
    pub store_in_context: Option<String>,
    #[serde(default)]
    pub on_error: EnrichErrorAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum EnrichSource {
    #[default]
    None,
    Api {
        url: String,
        #[serde(default = "default_method")]
        method: String,
        #[serde(default)]
        headers: HashMap<String, String>,
        #[serde(default)]
        body: Option<serde_json::Value>,
        #[serde(default)]
        extract: HashMap<String, String>,
    },
    File {
        path: String,
        #[serde(default)]
        format: Option<String>,
    },
    Tool {
        tool: String,
        #[serde(default)]
        args: serde_json::Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnrichErrorAction {
    #[default]
    Continue,
    Stop,
    Warn,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConditionalStage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub config: ConditionalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConditionalConfig {
    #[serde(default)]
    pub condition: Option<ConditionExpr>,
    #[serde(default, rename = "then")]
    pub then_stages: Vec<ProcessStage>,
    #[serde(default, rename = "else")]
    pub else_stages: Vec<ProcessStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConditionExpr {
    All { all: Vec<ConditionExpr> },
    Any { any: Vec<ConditionExpr> },
    Simple(HashMap<String, serde_json::Value>),
}

impl Default for ConditionExpr {
    fn default() -> Self {
        ConditionExpr::Simple(HashMap::new())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessSettings {
    #[serde(default)]
    pub on_stage_error: StageErrorConfig,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub cache: ProcessCacheConfig,
    #[serde(default)]
    pub debug: ProcessDebugConfig,
}

impl Default for ProcessSettings {
    fn default() -> Self {
        Self {
            on_stage_error: StageErrorConfig::default(),
            timeout_ms: default_timeout(),
            cache: ProcessCacheConfig::default(),
            debug: ProcessDebugConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageErrorConfig {
    #[serde(default)]
    pub default: StageErrorAction,
    #[serde(default)]
    pub retry: Option<StageRetryConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StageErrorAction {
    #[default]
    Continue,
    Stop,
    Retry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRetryConfig {
    #[serde(default = "default_retry")]
    pub max_retries: u32,
    #[serde(default = "default_backoff")]
    pub backoff_ms: u64,
}

impl Default for StageRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_retry(),
            backoff_ms: default_backoff(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessCacheConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub stages: Vec<String>,
    #[serde(default = "default_cache_ttl")]
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessDebugConfig {
    #[serde(default)]
    pub log_stages: bool,
    #[serde(default)]
    pub include_timing: bool,
}

fn default_true() -> bool {
    true
}

fn default_mask_char() -> String {
    "*".to_string()
}

fn default_threshold() -> f32 {
    0.7
}

fn default_timeout() -> u64 {
    5000
}

fn default_retry() -> u32 {
    2
}

fn default_backoff() -> u64 {
    100
}

fn default_cache_ttl() -> u64 {
    300
}

fn default_method() -> String {
    "GET".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProcessConfig::default();
        assert!(config.input.is_empty());
        assert!(config.output.is_empty());
        assert_eq!(config.settings.timeout_ms, 5000);
    }

    #[test]
    fn test_yaml_parsing() {
        let yaml = r#"
input:
  - type: normalize
    id: basic_normalize
    config:
      trim: true
      collapse_whitespace: true
  - type: detect
    id: detect_language
    config:
      llm: fast
      detect:
        - language
        - sentiment
      intents:
        - id: greeting
          description: "User is saying hello"
  - type: extract
    config:
      llm: fast
      schema:
        user_name:
          type: string
          description: "User's name if mentioned"
output:
  - type: validate
    config:
      llm: fast
      criteria:
        - "Response is helpful"
      threshold: 0.8
settings:
  timeout_ms: 3000
"#;
        let config: ProcessConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.input.len(), 3);
        assert_eq!(config.output.len(), 1);
        assert_eq!(config.settings.timeout_ms, 3000);
    }

    #[test]
    fn test_normalize_config() {
        let config = NormalizeConfig::default();
        assert!(config.trim);
        assert!(!config.collapse_whitespace);
    }

    #[test]
    fn test_field_type() {
        let yaml = r#"
type: enum
values:
  - low
  - medium
  - high
description: "Priority level"
"#;
        let schema: FieldSchema = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(schema.field_type, FieldType::Enum);
        assert_eq!(schema.values.len(), 3);
    }

    #[test]
    fn test_condition_parsing_simple() {
        let yaml = r#"
type: extract
condition:
  context.session.user_name:
    exists: false
config:
  schema:
    user_name:
      type: string
"#;
        let stage: ExtractStage = serde_yaml::from_str(yaml).unwrap();
        assert!(stage.condition.is_some());
    }

    #[test]
    fn test_condition_parsing_all() {
        let yaml = r#"
type: enrich
condition:
  all:
    - context.input.extracted.user_name:
        exists: true
    - context.session.user_profile:
        exists: false
config: {}
"#;
        let stage: EnrichStage = serde_yaml::from_str(yaml).unwrap();
        assert!(stage.condition.is_some());
        match stage.condition.unwrap() {
            ConditionExpr::All { all } => assert_eq!(all.len(), 2),
            _ => panic!("Expected All condition"),
        }
    }

    #[test]
    fn test_condition_parsing_any() {
        let yaml = r#"
type: detect
condition:
  any:
    - context.session.language:
        exists: false
    - context.session.force_detect: true
config:
  detect:
    - language
"#;
        let stage: DetectStage = serde_yaml::from_str(yaml).unwrap();
        assert!(stage.condition.is_some());
        match stage.condition.unwrap() {
            ConditionExpr::Any { any } => assert_eq!(any.len(), 2),
            _ => panic!("Expected Any condition"),
        }
    }

    #[test]
    fn test_process_stage_condition_accessor() {
        let stage = ProcessStage::Extract(ExtractStage {
            id: Some("test".to_string()),
            condition: Some(ConditionExpr::Simple(HashMap::new())),
            config: ExtractConfig::default(),
        });
        assert!(stage.condition().is_some());
        assert_eq!(stage.id(), Some("test"));
    }

    #[test]
    fn test_process_stage_no_condition() {
        let stage = ProcessStage::Normalize(NormalizeStage::default());
        assert!(stage.condition().is_none());
    }
}
