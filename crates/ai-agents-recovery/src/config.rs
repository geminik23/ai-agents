use ai_agents_core::{AgentError, MAX_TOOL_TIMEOUT_MS, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level error recovery configuration for an agent.
/// Controls active retry, LLM failure and context handling, tool failure handling, and compatibility parsing settings.
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

impl ErrorRecoveryConfig {
    /// Validates default and per-tool invocation timeout caps before a recovery manager is installed.
    pub fn validate(&self) -> Result<()> {
        let mut invalid_paths = Vec::new();
        if self
            .tools
            .default
            .timeout_ms
            .is_some_and(|timeout_ms| timeout_ms > MAX_TOOL_TIMEOUT_MS)
        {
            invalid_paths.push("error_recovery.tools.default.timeout_ms".to_string());
        }
        invalid_paths.extend(
            self.tools
                .per_tool
                .iter()
                .filter(|(_, config)| {
                    config
                        .timeout_ms
                        .is_some_and(|timeout_ms| timeout_ms > MAX_TOOL_TIMEOUT_MS)
                })
                .map(|(tool_id, _)| format!("error_recovery.tools.{tool_id}.timeout_ms")),
        );
        invalid_paths.sort();
        if invalid_paths.is_empty() {
            return Ok(());
        }
        Err(AgentError::Config(format!(
            "{} must be no greater than {MAX_TOOL_TIMEOUT_MS} milliseconds",
            invalid_paths.join(", ")
        )))
    }
}

/// Retry policy used by `RecoveryManager::with_retry` after an initial failed attempt.
/// Runtime main-provider calls use the top-level default, while tool calls use `ToolRetryConfig`.
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

/// Backoff timing configuration used between retry attempts.
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

/// Strategy for computing the wait duration between successive retry attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackoffType {
    Fixed,
    Linear,
    #[default]
    Exponential,
}

/// Classified error kinds used to decide whether a failure should be retried.
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

/// LLM-specific failure and context-overflow settings plus compatibility rate-limit configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LLMRecoveryConfig {
    #[serde(default)]
    pub on_failure: LLMFailureAction,
    #[serde(default)]
    pub on_rate_limit: RateLimitAction,
    #[serde(default)]
    pub on_context_overflow: ContextOverflowAction,
}

/// Action to take when the primary LLM attempt and configured retry policy end in failure.
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

/// Compatibility configuration for rate-limit-specific recovery.
/// The current runtime uses `RetryConfig` for rate-limit retries and `LLMFailureAction` after exhaustion instead of executing these variants separately.
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

/// Action to take when the accumulated message history exceeds max_context_tokens.
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
        #[serde(default)]
        summarizer_llm: Option<String>,
        #[serde(default = "default_max_summary_tokens")]
        max_summary_tokens: u32,
        #[serde(default)]
        custom_prompt: Option<String>,
        #[serde(default = "default_keep_recent")]
        keep_recent: usize,
        #[serde(default)]
        filter: Option<FilterConfig>,
    },
}

/// Selects which messages are passed to the summarizer during context compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FilterConfig {
    KeepRecent(#[serde(default = "default_keep_recent")] usize),
    ByRole { keep_roles: Vec<String> },
    SkipPattern { skip_if_contains: Vec<String> },
    Custom { name: String },
}

/// Tool-level recovery: a default policy plus optional per-tool overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolRecoveryConfig {
    #[serde(default)]
    pub default: ToolRetryConfig,
    #[serde(default, flatten)]
    pub per_tool: HashMap<String, ToolRetryConfig>,
}

/// Retry and failure policy for a single tool (or the default for all tools).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolRetryConfig {
    #[serde(default)]
    pub max_retries: u32,
    /// Optional invocation timeout cap, up to [`MAX_TOOL_TIMEOUT_MS`].
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub on_failure: ToolFailureAction,
}

/// Action to take when a tool call and its retry policy end in failure.
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

/// Compatibility settings for malformed LLM output.
/// The current main-provider path does not execute these actions separately.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParsingRecoveryConfig {
    #[serde(default)]
    pub on_invalid_json: ParseErrorAction,
    #[serde(default)]
    pub on_invalid_tool_call: ParseErrorAction,
}

/// Compatibility action retained for parsing recovery configuration.
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
    10
}

fn default_max_summary_tokens() -> u32 {
    200
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
    fn tool_recovery_timeouts_accept_the_documented_range() {
        let config = ErrorRecoveryConfig {
            tools: ToolRecoveryConfig {
                default: ToolRetryConfig {
                    timeout_ms: Some(MAX_TOOL_TIMEOUT_MS),
                    ..Default::default()
                },
                per_tool: HashMap::from([(
                    "slow".to_string(),
                    ToolRetryConfig {
                        timeout_ms: Some(MAX_TOOL_TIMEOUT_MS),
                        ..Default::default()
                    },
                )]),
            },
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn tool_recovery_timeouts_reject_unrepresentable_values_with_full_paths() {
        for timeout_ms in [MAX_TOOL_TIMEOUT_MS + 1, u64::MAX] {
            let config = ErrorRecoveryConfig {
                tools: ToolRecoveryConfig {
                    default: ToolRetryConfig {
                        timeout_ms: Some(timeout_ms),
                        ..Default::default()
                    },
                    per_tool: HashMap::from([(
                        "slow".to_string(),
                        ToolRetryConfig {
                            timeout_ms: Some(timeout_ms),
                            ..Default::default()
                        },
                    )]),
                },
                ..Default::default()
            };
            let error = config.validate().unwrap_err();
            let message = error.to_string();
            assert!(message.contains("error_recovery.tools.default.timeout_ms"));
            assert!(message.contains("error_recovery.tools.slow.timeout_ms"));
            assert!(message.contains("3153600000000000 milliseconds"));
        }
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

    #[test]
    fn test_summarize_config_parsing() {
        let yaml = r#"
llm:
  on_context_overflow:
    action: summarize
    summarizer_llm: fast
    max_summary_tokens: 300
    keep_recent: 5
    filter:
      type: by_role
      keep_roles:
        - user
        - assistant
"#;
        let config: ErrorRecoveryConfig = serde_yaml::from_str(yaml).unwrap();
        match &config.llm.on_context_overflow {
            ContextOverflowAction::Summarize {
                summarizer_llm,
                max_summary_tokens,
                keep_recent,
                filter,
                ..
            } => {
                assert_eq!(summarizer_llm.as_deref(), Some("fast"));
                assert_eq!(max_summary_tokens, &300);
                assert_eq!(keep_recent, &5);
                assert!(matches!(filter, Some(FilterConfig::ByRole { .. })));
            }
            _ => panic!("Expected Summarize action"),
        }
    }

    #[test]
    fn test_filter_config_parsing() {
        let yaml = r#"
type: skip_pattern
skip_if_contains:
  - "[DEBUG]"
  - "[TOOL]"
"#;
        let filter: FilterConfig = serde_yaml::from_str(yaml).unwrap();
        match filter {
            FilterConfig::SkipPattern { skip_if_contains } => {
                assert_eq!(skip_if_contains.len(), 2);
            }
            _ => panic!("Expected SkipPattern"),
        }
    }
}
