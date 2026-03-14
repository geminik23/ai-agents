//! OpenAI-Compatible Provider Example
//!
//! Demonstrates implementing `LLMProvider` as an HTTP adapter for any OpenAI-compatible API server
//!
//! Environment variables:
//!   LOCAL_LLM_BASE_URL  - API base URL (default: http://localhost:11434/v1)
//!   LOCAL_LLM_MODEL     - Model name  (default: local-model)
//!   LOCAL_LLM_API_KEY   - API key     (default: not-needed)
//!
//! Run: cargo run --bin openai-compatible

use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider,
    LLMResponse, Role, TokenUsage,
};
use ai_agents::{AgentBuilder, Result};
use ai_agents_cli::{CliRepl as Repl, init_tracing};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── OpenAI API types ─────────────────────────────────────────────

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    stream: bool,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<ApiUsage>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<DeltaContent>,
    #[serde(default)]
    delta: Option<DeltaContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct DeltaContent {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct ApiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize)]
struct StreamChunkResponse {
    choices: Vec<Choice>,
}

// ── Provider implementation ──────────────────────────────────────

struct OpenAICompatibleProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl OpenAICompatibleProvider {
    fn new(base_url: impl Into<String>, model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    fn from_env() -> Self {
        Self::new(
            std::env::var("LOCAL_LLM_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
            std::env::var("LOCAL_LLM_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string()),
            std::env::var("LOCAL_LLM_API_KEY").unwrap_or_else(|_| "not-needed".to_string()),
        )
    }

    fn build_messages(&self, messages: &[ChatMessage]) -> Vec<ApiMessage> {
        messages
            .iter()
            .map(|m| ApiMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::Function => "function".to_string(),
                    Role::Tool => "tool".to_string(),
                },
                content: m.content.clone(),
            })
            .collect()
    }

    fn map_finish_reason(reason: &str) -> FinishReason {
        match reason {
            "stop" | "end_turn" => FinishReason::Stop,
            "length" | "max_tokens" => FinishReason::Length,
            "tool_calls" | "function_call" => FinishReason::ToolCall,
            "content_filter" => FinishReason::ContentFilter,
            _ => FinishReason::Other,
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAICompatibleProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: self.build_messages(messages),
            temperature: config.and_then(|c| c.temperature),
            max_tokens: config.and_then(|c| c.max_tokens),
            top_p: config.and_then(|c| c.top_p),
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::Network(format!("Connection failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::API {
                message: format!("HTTP {}: {}", status, body),
                status: Some(status),
            });
        }

        let body: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LLMError::Serialization(format!("Failed to parse response: {}", e)))?;

        let content = body
            .choices
            .first()
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.content.clone())
            .unwrap_or_default();

        let finish_reason = body
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .map(Self::map_finish_reason)
            .unwrap_or(FinishReason::Stop);

        let usage = body.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LLMResponse {
            content,
            finish_reason,
            usage,
            model: body.model,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: self.build_messages(messages),
            temperature: config.and_then(|c| c.temperature),
            max_tokens: config.and_then(|c| c.max_tokens),
            top_p: config.and_then(|c| c.top_p),
            stream: true,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::Network(format!("Connection failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(LLMError::API {
                message: format!("HTTP {}: {}", status, body),
                status: Some(status),
            });
        }

        let byte_stream = response.bytes_stream();

        // Buffer for partial SSE lines across chunk boundaries
        let mapped = byte_stream
            .scan(String::new(), |buffer, chunk| {
                let chunk = match chunk {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        return std::future::ready(Some(vec![Err(LLMError::Network(
                            format!("Stream read error: {}", e),
                        ))]));
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));
                let mut results = Vec::new();

                // Process complete lines
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            results.push(Ok(LLMChunk::final_chunk(
                                "",
                                FinishReason::Stop,
                                None,
                            )));
                            continue;
                        }

                        match serde_json::from_str::<StreamChunkResponse>(data) {
                            Ok(chunk_resp) => {
                                if let Some(choice) = chunk_resp.choices.first() {
                                    if let Some(ref delta) = choice.delta {
                                        if let Some(ref content) = delta.content {
                                            if !content.is_empty() {
                                                results
                                                    .push(Ok(LLMChunk::new(content.clone(), false)));
                                            }
                                        }
                                    }
                                }
                            }
                            Err(_) => { /* skip unparseable chunks */ }
                        }
                    }
                }

                std::future::ready(Some(results))
            })
            .flat_map(futures::stream::iter);

        Ok(Box::new(Box::pin(mapped)))
    }

    fn provider_name(&self) -> &str {
        "openai-compatible"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::Streaming | LLMFeature::SystemMessages)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let provider = OpenAICompatibleProvider::from_env();
    println!("Connecting to: {} (model: {})",
        std::env::var("LOCAL_LLM_BASE_URL").unwrap_or_else(|_| "http://localhost:1234/v1".to_string()),
        std::env::var("LOCAL_LLM_MODEL").unwrap_or_else(|_| "local-model".to_string()),
    );

    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a helpful assistant running on a local model server. \
             Keep responses concise and informative.",
        )
        .llm(Arc::new(provider))
        .build()?;

    Repl::new(agent)
        .welcome("=== OpenAI-Compatible Local LLM ===")
        .hint("Requires a running OpenAI-compatible server (LM Studio, Ollama, vLLM, etc.).")
        .hint("Set LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL, LOCAL_LLM_API_KEY to configure.")
        .run()
        .await
}
