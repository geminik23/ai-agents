//! Memory implementations for AI Agents framework

mod compacting;
mod context;
mod events;
mod in_memory;
mod summarizer;
mod token_budget;

use async_trait::async_trait;

pub use ai_agents_core::MemorySnapshot;
pub use compacting::{CompactingMemory, CompactingMemoryConfig, CompressionEvent};
pub use context::{estimate_message_tokens, estimate_tokens, CompressResult, ConversationContext};
pub use events::{
    EvictionReason, FactExtractedEvent, MemoryBudgetEvent, MemoryCompressEvent, MemoryEvictEvent,
};
pub use in_memory::InMemoryStore;
pub use summarizer::{LLMSummarizer, NoopSummarizer, Summarizer};
pub use token_budget::{MemoryBudgetState, MemoryTokenBudget, OverflowStrategy, TokenAllocation};

/// Extended memory trait that preserves the original interface.
#[async_trait]
pub trait Memory: ai_agents_core::Memory {
    async fn get_context(&self) -> ai_agents_core::Result<ConversationContext> {
        let messages = self.get_messages(None).await?;
        Ok(ConversationContext::with_messages(messages))
    }

    async fn compress(
        &self,
        _summarizer: Option<&dyn Summarizer>,
    ) -> ai_agents_core::Result<CompressResult> {
        Ok(CompressResult::NotNeeded)
    }

    fn needs_compression(&self) -> bool {
        false
    }
}
