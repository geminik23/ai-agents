//! Error recovery configuration types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorRecoveryConfig {
    #[serde(default)]
    pub default: RetryConfig,
    #[serde(default)]
    pub llm: LLMRecoveryConfig,
    #[serde(default)]
    pub tools: ToolRecoveryConfig,
    #[serde(default)]
    pub parsing: ParsingRecoveryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    #[serde(default)]
    pub max_retries: u32,
    #[serde(default)]
    pub backoff: BackoffConfig,
    #[serde(default)]
    pub retry_on: Vec<ErrorType>,
    #[serde(default)]
    pub no_retry_on: Vec<ErrorType>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 0,
            backoff: BackoffConfig::default(),
            retry_on: vec![
                ErrorType::Timeout,
                ErrorType::RateLimit,
                ErrorType::ConnectionError,
                ErrorType::ServerError,
            ],
            no_retry_on: vec![ErrorType::InvalidApiKey, ErrorType::InvalidRequest],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffConfig {
    #[serde(default = "default_backoff_type", rename = "type")]
    pub backoff_type: BackoffType,
    #[serde(default = "default_initial_ms")]
    pub initial_ms: u64,
    #[serde(default = "default_max_ms")]
    pub max_ms: u64,
    #[serde(default = "default_multiplier")]
    pub multiplier: f64,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            backoff_type: default_backoff_type(),
            initial_ms: default_initial_ms(),
            max_ms: default_max_ms(),
            multiplier: default_multiplier(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackoffType {
    Fixed,
    Linear,
    #[default]
    Exponential,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    Timeout,
    RateLimit,
    ConnectionError,
    ServerError,
    InvalidApiKey,
    ContextTooLong,
    InvalidRequest,
    InvalidResponse,
    ToolError,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LLMRecoveryConfig {
    #[serde(default)]
    pub on_failure: LLMFailureAction,
    #[serde(default)]
    pub on_rate_limit: RateLimitAction,
    #[serde(default)]
    pub on_context_overflow: ContextOverflowAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum LLMFailureAction {
    #[default]
    Error,
    FallbackLlm {
        fallback_llm: String,
    },
    FallbackResponse {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RateLimitAction {
    #[default]
    Error,
    WaitAndRetry {
        #[serde(default = "default_rate_limit_wait")]
        max_wait_ms: u64,
    },
    SwitchModel {
        fallback_llm: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ContextOverflowAction {
    #[default]
    Error,
    Truncate {
        #[serde(default = "default_keep_recent")]
        keep_recent: usize,
    },
    Summarize {
        summarizer_llm: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolRecoveryConfig {
    #[serde(default)]
    pub default: ToolRetryConfig,
    #[serde(default, flatten)]
    pub per_tool: HashMap<String, ToolRetryConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRetryConfig {
    #[serde(default)]
    pub max_retries: u32,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub on_failure: ToolFailureAction,
}

impl Default for ToolRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 0,
            timeout_ms: None,
            on_failure: ToolFailureAction::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ToolFailureAction {
    #[default]
    ReportError,
    Skip,
    Fallback {
        fallback_tool: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParsingRecoveryConfig {
    #[serde(default)]
    pub on_invalid_json: ParseErrorAction,
    #[serde(default)]
    pub on_invalid_tool_call: ParseErrorAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ParseErrorAction {
    #[default]
    Error,
    RetryWithHint {
        #[serde(default = "default_parse_retries")]
        max_retries: u32,
    },
    ExtractPartial,
}

fn default_backoff_type() -> BackoffType {
    BackoffType::Exponential
}

fn default_initial_ms() -> u64 {
    100
}

fn default_max_ms() -> u64 {
    5000
}

fn default_multiplier() -> f64 {
    2.0
}

fn default_parse_retries() -> u32 {
    2
}

fn default_rate_limit_wait() -> u64 {
    30000
}

fn default_keep_recent() -> usize {
    20
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ErrorRecoveryConfig::default();
        assert_eq!(config.default.max_retries, 0);
        assert_eq!(
            config.default.backoff.backoff_type,
            BackoffType::Exponential
        );
    }

    #[test]
    fn test_yaml_parsing() {
        let yaml = r#"
default:
  max_retries: 5
  backoff:
    type: linear
    initial_ms: 200
llm:
  on_failure:
    action: fallback_llm
    fallback_llm: local
"#;
        let config: ErrorRecoveryConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.default.max_retries, 5);
        assert!(matches!(
            config.default.backoff.backoff_type,
            BackoffType::Linear
        ));
        assert!(matches!(
            config.llm.on_failure,
            LLMFailureAction::FallbackLlm { .. }
        ));
    }
}
