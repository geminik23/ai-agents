//! Recovery manager with retry logic and backoff strategies

use super::{
    BackoffType, ErrorRecoveryConfig, ErrorType, IntoClassifiedError, RecoveryError, RetryConfig,
    RetryFailure, classify_llm_error,
};
use std::future::Future;
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct RecoveryManager {
    config: ErrorRecoveryConfig,
}

impl RecoveryManager {
    /// Creates a validated recovery manager and panics when host configuration is invalid.
    pub fn new(config: ErrorRecoveryConfig) -> Self {
        Self::try_new(config).expect("invalid error recovery configuration")
    }

    /// Creates a validated recovery manager with a recoverable configuration error.
    pub fn try_new(config: ErrorRecoveryConfig) -> ai_agents_core::Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &ErrorRecoveryConfig {
        &self.config
    }

    /// Execute operation with retry logic
    pub async fn with_retry<T, E, F, Fut>(
        &self,
        operation_name: &str,
        retry_config: Option<&RetryConfig>,
        mut operation: F,
    ) -> Result<T, RecoveryError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: IntoClassifiedError,
    {
        let config = retry_config.unwrap_or(&self.config.default);
        let mut attempts = 0u32;
        let mut retries = 0u32;

        loop {
            attempts += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let classified = e.classify();

                    if !self.should_retry(&classified.error_type, config) {
                        return Err(RecoveryError::NonRetryable(classified));
                    }

                    if retries >= config.max_retries {
                        return Err(RecoveryError::MaxRetriesExceeded {
                            attempts,
                            last_error: classified,
                        });
                    }

                    retries += 1;
                    let wait = self.calculate_backoff(retries, &config.backoff);
                    tracing::warn!(
                        "[Recovery] {} failed (retry {}/{}), retrying in {:?}",
                        operation_name,
                        retries,
                        config.max_retries,
                        wait
                    );

                    tokio::time::sleep(wait).await;
                }
            }
        }
    }

    /// Executes an LLM operation with retry policy while retaining each original typed error.
    ///
    /// The terminal predicate runs immediately after an attempt fails and before generic error
    /// classification, retry scheduling, or backoff. This lets runtime preserve provider-owned
    /// protocol/history failures while continuing to apply its existing fallback policy only to
    /// non-terminal or exhausted failures.
    pub async fn with_llm_retry<T, F, Fut, P>(
        &self,
        operation_name: &str,
        retry_config: Option<&RetryConfig>,
        mut operation: F,
        is_terminal: P,
    ) -> Result<T, RetryFailure<ai_agents_llm::LLMError>>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, ai_agents_llm::LLMError>>,
        P: Fn(&ai_agents_llm::LLMError) -> bool,
    {
        let config = retry_config.unwrap_or(&self.config.default);
        let mut attempts = 0u32;
        let mut retries = 0u32;

        loop {
            attempts += 1;
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if is_terminal(&error) {
                        return Err(RetryFailure::Terminal { attempts, error });
                    }

                    let classified = classify_llm_error(&error);
                    if !self.should_retry(&classified.error_type, config) {
                        return Err(RetryFailure::NonRetryable {
                            attempts,
                            error,
                            classified,
                        });
                    }
                    if retries >= config.max_retries {
                        return Err(RetryFailure::Exhausted {
                            attempts,
                            error,
                            classified,
                        });
                    }

                    retries += 1;
                    let wait = self.calculate_backoff(retries, &config.backoff);
                    tracing::warn!(
                        "[Recovery] {} failed (retry {}/{}), retrying in {:?}",
                        operation_name,
                        retries,
                        config.max_retries,
                        wait
                    );
                    tokio::time::sleep(wait).await;
                }
            }
        }
    }

    fn should_retry(&self, error_type: &ErrorType, config: &RetryConfig) -> bool {
        // Check blacklist first
        if config.no_retry_on.contains(error_type) {
            return false;
        }

        // If whitelist is specified, only retry those
        if !config.retry_on.is_empty() {
            return config.retry_on.contains(error_type);
        }

        // Default: retry transient errors
        matches!(
            error_type,
            ErrorType::Timeout
                | ErrorType::RateLimit
                | ErrorType::ConnectionError
                | ErrorType::ServerError
        )
    }

    fn calculate_backoff(&self, attempt: u32, config: &super::BackoffConfig) -> Duration {
        let base = config.initial_ms as f64;

        let wait_ms = match config.backoff_type {
            BackoffType::Fixed => base,
            BackoffType::Linear => base * attempt as f64,
            BackoffType::Exponential => base * config.multiplier.powi(attempt as i32 - 1),
        };

        Duration::from_millis(wait_ms.min(config.max_ms as f64) as u64)
    }

    /// Returns the tool-specific retry policy or the complete default policy.
    pub fn get_tool_config(&self, tool_id: &str) -> &super::ToolRetryConfig {
        self.config
            .tools
            .per_tool
            .get(tool_id)
            .unwrap_or(&self.config.tools.default)
    }

    /// Returns the timeout from the complete retry policy selected for this tool.
    pub fn get_tool_timeout(&self, tool_id: &str) -> Option<u64> {
        self.get_tool_config(tool_id).timeout_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClassifiedError;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn try_new_rejects_invalid_tool_recovery_timeout() {
        let config = super::super::ErrorRecoveryConfig {
            tools: super::super::ToolRecoveryConfig {
                default: super::super::ToolRetryConfig {
                    timeout_ms: Some(ai_agents_core::MAX_TOOL_TIMEOUT_MS + 1),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };

        let error = RecoveryManager::try_new(config).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("error_recovery.tools.default.timeout_ms")
        );
    }

    #[test]
    fn per_tool_timeout_follows_complete_policy_selection() {
        let manager = RecoveryManager::new(super::super::ErrorRecoveryConfig {
            tools: super::super::ToolRecoveryConfig {
                default: super::super::ToolRetryConfig {
                    timeout_ms: Some(5_000),
                    ..Default::default()
                },
                per_tool: std::collections::HashMap::from([
                    (
                        "without_timeout".to_string(),
                        super::super::ToolRetryConfig {
                            max_retries: 1,
                            ..Default::default()
                        },
                    ),
                    (
                        "with_timeout".to_string(),
                        super::super::ToolRetryConfig {
                            timeout_ms: Some(1_000),
                            ..Default::default()
                        },
                    ),
                ]),
            },
            ..Default::default()
        });

        assert_eq!(manager.get_tool_timeout("other"), Some(5_000));
        assert_eq!(manager.get_tool_timeout("without_timeout"), None);
        assert_eq!(manager.get_tool_timeout("with_timeout"), Some(1_000));
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = super::super::ErrorRecoveryConfig {
            default: super::super::RetryConfig {
                max_retries: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = RecoveryManager::new(config);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<&str, RecoveryError> = manager
            .with_retry("test", None, || {
                let c = counter_clone.clone();
                async move {
                    let count = c.fetch_add(1, Ordering::SeqCst);
                    if count < 2 {
                        Err(ClassifiedError::timeout("temp failure"))
                    } else {
                        Ok("success")
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_max_retries_counts_retries_after_the_initial_attempt() {
        let config = super::super::ErrorRecoveryConfig {
            default: super::super::RetryConfig {
                max_retries: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = RecoveryManager::new(config);
        let attempts = Arc::new(AtomicU32::new(0));
        let observed_attempts = Arc::clone(&attempts);

        let result: Result<(), RecoveryError> = manager
            .with_retry("test", None, || {
                let attempts = Arc::clone(&observed_attempts);
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>(ClassifiedError::timeout("always fails"))
                }
            })
            .await;

        assert!(matches!(
            result,
            Err(RecoveryError::MaxRetriesExceeded { attempts: 4, .. })
        ));
        assert_eq!(attempts.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn test_non_retryable_error() {
        let config = super::super::ErrorRecoveryConfig {
            default: super::super::RetryConfig {
                max_retries: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = RecoveryManager::new(config);

        let result: Result<(), RecoveryError> = manager
            .with_retry("test", None, || async {
                Err::<(), _>(ClassifiedError::invalid_api_key("bad key"))
            })
            .await;

        assert!(matches!(result, Err(RecoveryError::NonRetryable(_))));
    }

    #[tokio::test]
    async fn typed_llm_retry_stops_terminal_error_before_retry() {
        let manager = RecoveryManager::new(super::super::ErrorRecoveryConfig {
            default: super::super::RetryConfig {
                max_retries: 3,
                ..Default::default()
            },
            ..Default::default()
        });
        let attempts = Arc::new(AtomicU32::new(0));
        let observed_attempts = Arc::clone(&attempts);

        let result = manager
            .with_llm_retry(
                "terminal",
                None,
                || {
                    let attempts = Arc::clone(&observed_attempts);
                    async move {
                        attempts.fetch_add(1, Ordering::SeqCst);
                        Err::<(), _>(ai_agents_llm::LLMError::Network(
                            "typed terminal".to_string(),
                        ))
                    }
                },
                |_| true,
            )
            .await;

        let failure = result.unwrap_err();
        assert!(failure.is_terminal());
        assert_eq!(failure.attempts(), 1);
        assert!(failure.classified().is_none());
        assert!(matches!(
            failure.into_error(),
            ai_agents_llm::LLMError::Network(message) if message == "typed terminal"
        ));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn typed_llm_retry_preserves_transient_retry_behavior() {
        let manager = RecoveryManager::new(super::super::ErrorRecoveryConfig {
            default: super::super::RetryConfig {
                max_retries: 3,
                backoff: super::super::BackoffConfig {
                    initial_ms: 0,
                    max_ms: 0,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let attempts = Arc::new(AtomicU32::new(0));
        let observed_attempts = Arc::clone(&attempts);

        let result = manager
            .with_llm_retry(
                "transient",
                None,
                || {
                    let attempts = Arc::clone(&observed_attempts);
                    async move {
                        let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                        if attempt < 2 {
                            Err(ai_agents_llm::LLMError::Network(format!(
                                "attempt-{attempt}"
                            )))
                        } else {
                            Ok("success")
                        }
                    }
                },
                |_| false,
            )
            .await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn typed_llm_retry_preserves_exhausted_error_variant_and_value() {
        let manager = RecoveryManager::new(super::super::ErrorRecoveryConfig {
            default: super::super::RetryConfig {
                max_retries: 2,
                backoff: super::super::BackoffConfig {
                    initial_ms: 0,
                    max_ms: 0,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let attempts = Arc::new(AtomicU32::new(0));
        let observed_attempts = Arc::clone(&attempts);

        let failure = manager
            .with_llm_retry(
                "exhausted",
                None,
                || {
                    let attempts = Arc::clone(&observed_attempts);
                    async move {
                        let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                        Err::<(), _>(ai_agents_llm::LLMError::Network(format!(
                            "attempt-{attempt}"
                        )))
                    }
                },
                |_| false,
            )
            .await
            .unwrap_err();

        assert!(!failure.is_terminal());
        assert_eq!(failure.attempts(), 3);
        assert_eq!(
            failure.classified().unwrap().error_type,
            super::super::ErrorType::ConnectionError
        );
        assert!(matches!(
            failure.into_error(),
            ai_agents_llm::LLMError::Network(message) if message == "attempt-3"
        ));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn typed_llm_retry_preserves_non_retryable_error() {
        let manager = RecoveryManager::default();
        let failure = manager
            .with_llm_retry(
                "config",
                None,
                || async {
                    Err::<(), _>(ai_agents_llm::LLMError::Config(
                        "invalid typed config".to_string(),
                    ))
                },
                |_| false,
            )
            .await
            .unwrap_err();

        assert!(matches!(
            &failure,
            RetryFailure::NonRetryable { classified, .. }
                if classified.error_type == super::super::ErrorType::InvalidRequest
        ));
        assert!(matches!(
            failure.into_error(),
            ai_agents_llm::LLMError::Config(message) if message == "invalid typed config"
        ));
    }

    #[test]
    fn test_backoff_calculation() {
        let manager = RecoveryManager::default();
        let config = super::super::BackoffConfig {
            backoff_type: BackoffType::Exponential,
            initial_ms: 100,
            max_ms: 5000,
            multiplier: 2.0,
        };

        assert_eq!(
            manager.calculate_backoff(1, &config),
            Duration::from_millis(100)
        );
        assert_eq!(
            manager.calculate_backoff(2, &config),
            Duration::from_millis(200)
        );
        assert_eq!(
            manager.calculate_backoff(3, &config),
            Duration::from_millis(400)
        );
    }
}
