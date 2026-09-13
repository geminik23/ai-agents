//! Recovery error types

use super::ErrorType;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct ClassifiedError {
    pub error_type: ErrorType,
    pub message: String,
    pub retryable: bool,
}

impl ClassifiedError {
    pub fn new(error_type: ErrorType, message: impl Into<String>) -> Self {
        let retryable = matches!(
            error_type,
            ErrorType::Timeout
                | ErrorType::RateLimit
                | ErrorType::ConnectionError
                | ErrorType::ServerError
        );
        Self {
            error_type,
            message: message.into(),
            retryable,
        }
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorType::Timeout, message)
    }

    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::new(ErrorType::RateLimit, message)
    }

    pub fn connection(message: impl Into<String>) -> Self {
        Self::new(ErrorType::ConnectionError, message)
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self::new(ErrorType::ServerError, message)
    }

    pub fn invalid_api_key(message: impl Into<String>) -> Self {
        Self::new(ErrorType::InvalidApiKey, message)
    }

    pub fn context_too_long(message: impl Into<String>) -> Self {
        Self::new(ErrorType::ContextTooLong, message)
    }

    pub fn tool_error(message: impl Into<String>) -> Self {
        Self::new(ErrorType::ToolError, message)
    }
}

impl fmt::Display for ClassifiedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.error_type, self.message)
    }
}

impl std::error::Error for ClassifiedError {}

#[derive(Debug, Error)]
pub enum RecoveryError {
    #[error("Retry limit exceeded after {attempts} attempts: {last_error}")]
    MaxRetriesExceeded {
        attempts: u32,
        last_error: ClassifiedError,
    },

    #[error("Non-retryable error: {0}")]
    NonRetryable(ClassifiedError),

    #[error("Circuit breaker open for: {resource}")]
    CircuitOpen { resource: String },

    #[error("Timeout after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    #[error("No fallback available: {0}")]
    NoFallback(String),

    #[error("{0}")]
    Other(String),
}

impl RecoveryError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, RecoveryError::Timeout { .. })
    }

    pub fn last_error(&self) -> Option<&ClassifiedError> {
        match self {
            RecoveryError::MaxRetriesExceeded { last_error, .. } => Some(last_error),
            RecoveryError::NonRetryable(e) => Some(e),
            _ => None,
        }
    }
}

/// Typed failure from a retry loop that preserves the operation's original error.
///
/// Callers can stop terminal protocol failures immediately while retaining the distinction
/// between policy non-retryability and retry exhaustion for their existing fallback logic.
#[derive(Debug)]
pub enum RetryFailure<E> {
    /// The caller declared this exact error terminal before retry classification or backoff.
    Terminal { attempts: u32, error: E },
    /// Retry policy rejected this error type on the current attempt.
    NonRetryable {
        attempts: u32,
        error: E,
        classified: ClassifiedError,
    },
    /// The initial attempt and all configured retries failed.
    Exhausted {
        attempts: u32,
        error: E,
        classified: ClassifiedError,
    },
}

impl<E> RetryFailure<E> {
    /// Returns the number of attempts completed before this failure.
    pub fn attempts(&self) -> u32 {
        match self {
            Self::Terminal { attempts, .. }
            | Self::NonRetryable { attempts, .. }
            | Self::Exhausted { attempts, .. } => *attempts,
        }
    }

    /// Returns whether the caller explicitly classified this failure as terminal.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal { .. })
    }

    /// Borrows the original typed operation error.
    pub fn error(&self) -> &E {
        match self {
            Self::Terminal { error, .. }
            | Self::NonRetryable { error, .. }
            | Self::Exhausted { error, .. } => error,
        }
    }

    /// Returns retry classification evidence when policy evaluated the error.
    pub fn classified(&self) -> Option<&ClassifiedError> {
        match self {
            Self::Terminal { .. } => None,
            Self::NonRetryable { classified, .. } | Self::Exhausted { classified, .. } => {
                Some(classified)
            }
        }
    }

    /// Consumes the failure and returns the original typed operation error.
    pub fn into_error(self) -> E {
        match self {
            Self::Terminal { error, .. }
            | Self::NonRetryable { error, .. }
            | Self::Exhausted { error, .. } => error,
        }
    }
}

impl<E: fmt::Display> fmt::Display for RetryFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Terminal { attempts, error } => {
                write!(
                    formatter,
                    "Terminal error after {attempts} attempt(s): {error}"
                )
            }
            Self::NonRetryable {
                attempts, error, ..
            } => write!(
                formatter,
                "Non-retryable error after {attempts} attempt(s): {error}"
            ),
            Self::Exhausted {
                attempts, error, ..
            } => write!(
                formatter,
                "Retry limit exceeded after {attempts} attempt(s): {error}"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for RetryFailure<E> {}

pub trait IntoClassifiedError {
    fn classify(self) -> ClassifiedError;
}

impl IntoClassifiedError for ClassifiedError {
    fn classify(self) -> ClassifiedError {
        self
    }
}

impl IntoClassifiedError for ai_agents_llm::LLMError {
    fn classify(self) -> ClassifiedError {
        classify_llm_error(&self)
    }
}

/// Classifies an LLM error by reference so typed retry paths can retain the original value.
pub fn classify_llm_error(error: &ai_agents_llm::LLMError) -> ClassifiedError {
    match error {
        ai_agents_llm::LLMError::RateLimit { .. } => ClassifiedError::rate_limit(error.to_string()),
        ai_agents_llm::LLMError::Network(_) => ClassifiedError::connection(error.to_string()),
        ai_agents_llm::LLMError::API { status, .. } => {
            if let Some(code) = status {
                if *code >= 500 {
                    return ClassifiedError::server(error.to_string());
                }
                if *code == 401 || *code == 403 {
                    return ClassifiedError::invalid_api_key(error.to_string());
                }
            }
            ClassifiedError::new(ErrorType::InvalidRequest, error.to_string())
        }
        ai_agents_llm::LLMError::Config(_) => {
            ClassifiedError::new(ErrorType::InvalidRequest, error.to_string())
        }
        _ => ClassifiedError::new(ErrorType::InvalidResponse, error.to_string()),
    }
}

impl IntoClassifiedError for ai_agents_core::AgentError {
    fn classify(self) -> ClassifiedError {
        match &self {
            ai_agents_core::AgentError::Tool(msg) => ClassifiedError::tool_error(msg),
            ai_agents_core::AgentError::LLM(msg) | ai_agents_core::AgentError::LLMError(msg) => {
                if msg.to_lowercase().contains("timeout") {
                    ClassifiedError::timeout(msg)
                } else if msg.to_lowercase().contains("rate") {
                    ClassifiedError::rate_limit(msg)
                } else {
                    ClassifiedError::new(ErrorType::InvalidResponse, msg)
                }
            }
            _ => ClassifiedError::new(ErrorType::InvalidRequest, self.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classified_error() {
        let err = ClassifiedError::timeout("request timed out");
        assert!(err.retryable);
        assert_eq!(err.error_type, ErrorType::Timeout);
    }

    #[test]
    fn test_non_retryable() {
        let err = ClassifiedError::invalid_api_key("bad key");
        assert!(!err.retryable);
    }
}
