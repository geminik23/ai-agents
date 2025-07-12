//! Recovery manager with retry logic and backoff strategies

use super::{
    BackoffType, ErrorRecoveryConfig, ErrorType, IntoClassifiedError, RecoveryError, RetryConfig,
};
use std::future::Future;
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct RecoveryManager {
    config: ErrorRecoveryConfig,
}

impl RecoveryManager {
    pub fn new(config: ErrorRecoveryConfig) -> Self {
        Self { config }
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

        loop {
            attempts += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let classified = e.classify();

                    if !self.should_retry(&classified.error_type, config) {
                        return Err(RecoveryError::NonRetryable(classified));
                    }

                    if attempts >= config.max_retries {
                        return Err(RecoveryError::MaxRetriesExceeded {
                            attempts,
                            last_error: classified,
                        });
                    }

                    let wait = self.calculate_backoff(attempts, &config.backoff);
                    tracing::warn!(
                        "[Recovery] {} failed (attempt {}/{}), retrying in {:?}",
                        operation_name,
                        attempts,
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

    /// Get tool-specific retry config
    pub fn get_tool_config(&self, tool_id: &str) -> &super::ToolRetryConfig {
        self.config
            .tools
            .per_tool
            .get(tool_id)
            .unwrap_or(&self.config.tools.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::ClassifiedError;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

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
    async fn test_max_retries_exceeded() {
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
                Err::<(), _>(ClassifiedError::timeout("always fails"))
            })
            .await;

        assert!(matches!(
            result,
            Err(RecoveryError::MaxRetriesExceeded { .. })
        ));
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
