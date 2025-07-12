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
    #[error("Max retries ({attempts}) exceeded: {last_error}")]
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

pub trait IntoClassifiedError {
    fn classify(self) -> ClassifiedError;
}

impl IntoClassifiedError for ClassifiedError {
    fn classify(self) -> ClassifiedError {
        self
    }
}

impl IntoClassifiedError for crate::llm::LLMError {
    fn classify(self) -> ClassifiedError {
        match &self {
            crate::llm::LLMError::RateLimit { .. } => ClassifiedError::rate_limit(self.to_string()),
            crate::llm::LLMError::Network(_) => ClassifiedError::connection(self.to_string()),
            crate::llm::LLMError::API { status, .. } => {
                if let Some(code) = status {
                    if *code >= 500 {
                        return ClassifiedError::server(self.to_string());
                    }
                    if *code == 401 || *code == 403 {
                        return ClassifiedError::invalid_api_key(self.to_string());
                    }
                }
                ClassifiedError::new(ErrorType::InvalidRequest, self.to_string())
            }
            crate::llm::LLMError::Config(_) => {
                ClassifiedError::new(ErrorType::InvalidRequest, self.to_string())
            }
            _ => ClassifiedError::new(ErrorType::InvalidResponse, self.to_string()),
        }
    }
}

impl IntoClassifiedError for crate::error::AgentError {
    fn classify(self) -> ClassifiedError {
        match &self {
            crate::error::AgentError::Tool(msg) => ClassifiedError::tool_error(msg),
            crate::error::AgentError::LLM(msg) | crate::error::AgentError::LLMError(msg) => {
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
