//! Multi-Provider Routing Example
//!
//! Demonstrates mixing custom and built-in LLM providers with cost-optimized routing
//! - expensive model for user-facing responses, cheap custom provider for internal classification and routing tasks.
//!
//! Requires: OPENAI_API_KEY environment variable
//!
//! Run: cargo run --bin multi-provider

use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider,
    LLMResponse, TokenUsage,
};
use ai_agents::{AgentBuilder, LLMRegistry, ProviderType, Result, UnifiedLLMProvider};
use ai_agents_cli::{CliRepl as Repl, init_tracing};
use ai_agents_llm::multi::MultiLLMRouter;
use async_trait::async_trait;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// A cheap, fast provider for internal routing tasks.
///
/// In production this could be a small local model or a fast API.
/// Here we simulate it with deterministic responses that the framework's
/// internal systems (tool selection, guard evaluation, classification) can parse.
struct CheapRouterProvider {
    call_count: AtomicU32,
}

impl CheapRouterProvider {
    fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
        }
    }

    fn calls(&self) -> u32 {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for CheapRouterProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        _config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);

        let last_msg = messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // The framework sends structured prompts for internal tasks.
        // This cheap provider returns minimal valid responses.
        let content = format!(
            "Routed by cheap provider (call #{}). Input length: {} chars.",
            self.call_count.load(Ordering::Relaxed),
            last_msg.len()
        );

        let prompt_tokens = last_msg.len() as u32 / 4;
        let completion_tokens = content.len() as u32 / 4;

        Ok(LLMResponse {
            content,
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            model: Some("cheap-router-v1".to_string()),
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
        // Router tasks don't need streaming — fall back to single-chunk response
        let response = self.complete(messages, config).await?;
        let chunk = LLMChunk::final_chunk(
            response.content,
            response.finish_reason,
            response.usage,
        );
        Ok(Box::new(futures::stream::iter(vec![Ok(chunk)])))
    }

    fn provider_name(&self) -> &str {
        "cheap-router"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::SystemMessages)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    // 1. Create the main (expensive) provider for user-facing responses
    let main_provider = UnifiedLLMProvider::from_env(ProviderType::OpenAI, "gpt-4.1-mini")?;

    // 2. Create the cheap router for internal classification tasks
    let router_provider = Arc::new(CheapRouterProvider::new());

    // 3. Set up multi-provider routing
    //    - Primary: handles complete() and complete_stream() for user responses
    //    - Tool selector, guard evaluator, classifier: use the cheap router
    let router = MultiLLMRouter::new(Arc::new(main_provider))
        .with_tool_selector(router_provider.clone())
        .with_guard_evaluator(router_provider.clone())
        .with_classifier(router_provider.clone());

    // 4. Optionally, set up a named registry for explicit alias-based access
    let mut registry = LLMRegistry::new();
    registry.register("default", Arc::new(router));
    registry.set_default("default");

    let agent = AgentBuilder::new()
        .system_prompt(
            "You are a helpful assistant. The framework routes your internal tasks \
             (tool selection, classification) through a cheap provider, while your \
             user-facing responses go through the main model.",
        )
        .llm_registry(registry)
        .build()?;

    println!(
        "Router provider calls before chat: {}",
        router_provider.calls()
    );

    Repl::new(agent)
        .welcome("=== Multi-Provider Routing Demo ===")
        .hint("User-facing responses: OpenAI gpt-4.1-mini")
        .hint("Internal routing tasks: CheapRouterProvider (local, fast)")
        .hint("Requires: OPENAI_API_KEY")
        .run()
        .await
}
