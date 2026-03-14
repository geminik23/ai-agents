use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider,
    LLMResponse, TokenUsage,
};
use ai_agents::{AgentBuilder, Result};
use ai_agents_cli::{CliRepl as Repl, init_tracing};
use async_trait::async_trait;
use std::sync::Arc;

/// A simple rule-based LLM provider that works entirely offline.
///
/// This demonstrates the minimum required to implement `LLMProvider`:
/// - `complete()` - returns a full response
/// - `complete_stream()` - returns a stream of word-by-word chunks
/// - `provider_name()` - identifier string
/// - `supports()` - feature flags
struct EchoProvider {
    model_name: String,
}

impl EchoProvider {
    fn new(model: impl Into<String>) -> Self {
        Self {
            model_name: model.into(),
        }
    }

    /// Simple keyword-based response generation.
    /// In a real provider, this would call an API or run local inference.
    fn generate_response(&self, messages: &[ChatMessage]) -> String {
        let last_user_msg = messages
            .iter()
            .rev()
            .find(|m| m.role == ai_agents::Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let lower = last_user_msg.to_lowercase();

        if lower.contains("hello") || lower.contains("hi") {
            "Hello! I'm the EchoProvider — a custom LLM that runs entirely offline. \
             I respond to a few keywords: try 'help', 'weather', or 'rust'."
                .to_string()
        } else if lower.contains("help") {
            "I'm a demo provider showing how to implement the `LLMProvider` trait. \
             I understand: hello, help, weather, rust, and echo back anything else."
                .to_string()
        } else if lower.contains("weather") {
            "I can't actually check the weather — I'm an offline demo! \
             But a real custom provider could call any API here."
                .to_string()
        } else if lower.contains("rust") {
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             This entire framework is written in Rust!"
                .to_string()
        } else {
            format!(
                "Echo: \"{}\"\n\n(I'm a custom offline provider. \
                 Try: hello, help, weather, or rust.)",
                last_user_msg
            )
        }
    }
}

#[async_trait]
impl LLMProvider for EchoProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let content = self.generate_response(messages);

        // Estimate token usage (chars / 4 is a rough approximation)
        let prompt_tokens: u32 = messages.iter().map(|m| m.content.len() as u32 / 4).sum();
        let completion_tokens = content.len() as u32 / 4;

        Ok(LLMResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            model: Some(self.model_name.clone()),
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        _config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        let content = self.generate_response(messages);
        let prompt_tokens: u32 = messages.iter().map(|m| m.content.len() as u32 / 4).sum();
        let completion_tokens = content.len() as u32 / 4;

        // Split into word-by-word chunks for realistic streaming
        let words: Vec<String> = content
            .split_inclusive(' ')
            .map(|w| w.to_string())
            .collect();

        let mut chunks: Vec<std::result::Result<LLMChunk, LLMError>> = Vec::new();
        let last_idx = words.len().saturating_sub(1);

        for (i, word) in words.into_iter().enumerate() {
            if i == last_idx {
                chunks.push(Ok(LLMChunk::final_chunk(
                    word,
                    FinishReason::Stop,
                    Some(TokenUsage::new(prompt_tokens, completion_tokens)),
                )));
            } else {
                chunks.push(Ok(LLMChunk::new(word, false)));
            }
        }

        Ok(Box::new(futures::stream::iter(chunks)))
    }

    fn provider_name(&self) -> &str {
        "echo"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::Streaming | LLMFeature::SystemMessages)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let provider = EchoProvider::new("echo-v1");

    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a helpful assistant powered by a custom echo provider. \
             Greet the user and help them explore your capabilities.",
        )
        .llm(Arc::new(provider))
        .build()?;

    Repl::new(agent)
        .welcome("=== Custom LLM Provider Demo ===")
        .hint("This example runs entirely offline — no API key needed.")
        .hint("Try: 'hello', 'help', 'weather', 'rust', or any message.")
        .run()
        .await
}
