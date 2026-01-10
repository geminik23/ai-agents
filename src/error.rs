//! Error types for the AI agents framework

use crate::llm::LLMError;
use thiserror::Error;

/// Result type alias for agent operations
pub type Result<T> = std::result::Result<T, AgentError>;

/// Main error type for the framework
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("Skill error: {0}")]
    Skill(String),

    #[error("LLM error: {0}")]
    LLM(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Template error: {0}")]
    TemplateError(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Invalid spec: {0}")]
    InvalidSpec(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("HITL timeout")]
    HITLTimeout,

    #[error("HITL rejected: {0}")]
    HITLRejected(String),

    #[error("Persistence error: {0}")]
    Persistence(String),

    #[error("Memory budget exceeded: used {used} tokens, budget {budget} tokens")]
    MemoryBudgetExceeded { used: u32, budget: u32 },

    #[error("{0}")]
    Other(String),
}

impl From<LLMError> for AgentError {
    fn from(err: LLMError) -> Self {
        AgentError::LLM(err.to_string())
    }
}
