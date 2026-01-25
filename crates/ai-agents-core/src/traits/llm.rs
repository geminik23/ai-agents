//! LLM provider traits

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::message::ChatMessage;
use crate::types::{LLMChunk, LLMConfig, LLMFeature, LLMResponse};

/// Core LLM provider trait
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError>;

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>;

    fn provider_name(&self) -> &str;

    fn supports(&self, feature: LLMFeature) -> bool;
}

/// Higher-level LLM capabilities for agent operations
#[async_trait]
pub trait LLMCapability: Send + Sync {
    async fn select_tool(
        &self,
        context: &TaskContext,
        user_input: &str,
    ) -> Result<ToolSelection, LLMError>;

    async fn generate_tool_args(
        &self,
        tool_id: &str,
        user_input: &str,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value, LLMError>;

    async fn evaluate_yesno(
        &self,
        question: &str,
        context: &TaskContext,
    ) -> Result<(bool, String), LLMError>;

    async fn classify(&self, input: &str, categories: &[String])
    -> Result<(String, f32), LLMError>;

    async fn process_task(
        &self,
        context: &TaskContext,
        system_prompt: &str,
    ) -> Result<LLMResponse, LLMError>;
}

/// Task context for LLM operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub current_state: Option<String>,
    pub available_tools: Vec<String>,
    pub memory_slots: HashMap<String, serde_json::Value>,
    pub recent_messages: Vec<ChatMessage>,
}

/// Tool selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSelection {
    pub tool_id: String,
    pub confidence: f32,
    pub reasoning: Option<String>,
}

/// LLM error types
#[derive(Debug, Error)]
pub enum LLMError {
    #[error("API error: {message}")]
    API {
        message: String,
        status: Option<u16>,
    },

    #[error("Network error: {0}")]
    Network(String),

    #[error("Rate limit exceeded: {retry_after:?}")]
    RateLimit {
        retry_after: Option<std::time::Duration>,
    },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Content filtered: {0}")]
    ContentFiltered(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<serde_json::Error> for LLMError {
    fn from(err: serde_json::Error) -> Self {
        LLMError::Serialization(err.to_string())
    }
}
