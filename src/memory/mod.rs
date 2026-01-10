//! Memory system for storing conversation history

mod compacting;
mod context;
mod events;
mod in_memory;
mod summarizer;
mod token_budget;

use async_trait::async_trait;
pub use compacting::{CompactingMemory, CompactingMemoryConfig, CompressionEvent};
pub use context::{CompressResult, ConversationContext, estimate_message_tokens, estimate_tokens};
pub use events::{
    EvictionReason, FactExtractedEvent, MemoryBudgetEvent, MemoryCompressEvent, MemoryEvictEvent,
};
pub use in_memory::InMemoryStore;
use std::sync::Arc;
pub use summarizer::{LLMSummarizer, NoopSummarizer, Summarizer};
pub use token_budget::{MemoryBudgetState, MemoryTokenBudget, OverflowStrategy, TokenAllocation};

use crate::error::Result;
use crate::llm::{ChatMessage, LLMProvider};
use crate::persistence::MemorySnapshot;
use crate::spec::MemoryConfig;

#[async_trait]
pub trait Memory: Send + Sync {
    async fn add_message(&self, message: ChatMessage) -> Result<()>;
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ChatMessage>>;
    async fn clear(&self) -> Result<()>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    async fn snapshot(&self) -> Result<MemorySnapshot> {
        Ok(MemorySnapshot::new(self.get_messages(None).await?))
    }

    async fn restore(&self, snapshot: MemorySnapshot) -> Result<()>;

    async fn get_context(&self) -> Result<ConversationContext> {
        let messages = self.get_messages(None).await?;
        Ok(ConversationContext::with_messages(messages))
    }

    async fn compress(&self, _summarizer: Option<&dyn Summarizer>) -> Result<CompressResult> {
        Ok(CompressResult::NotNeeded)
    }

    fn needs_compression(&self) -> bool {
        false
    }

    /// Evict oldest messages from memory, returns evicted messages
    async fn evict_oldest(&self, _count: usize) -> Result<Vec<ChatMessage>> {
        Ok(vec![])
    }
}

pub fn create_memory(memory_type: &str, max_messages: usize) -> Arc<dyn Memory> {
    match memory_type {
        "in-memory" => Arc::new(InMemoryStore::new(max_messages)),
        "compacting" => {
            let summarizer: Arc<dyn Summarizer> = Arc::new(NoopSummarizer);
            Arc::new(CompactingMemory::with_default_config(summarizer))
        }
        _ => Arc::new(InMemoryStore::new(max_messages)),
    }
}

pub fn create_memory_from_config(config: &MemoryConfig) -> Arc<dyn Memory> {
    if config.is_compacting() {
        let summarizer: Arc<dyn Summarizer> = Arc::new(NoopSummarizer);
        let compacting_config = config.to_compacting_config();
        Arc::new(CompactingMemory::new(summarizer, compacting_config))
    } else {
        Arc::new(InMemoryStore::new(config.max_messages))
    }
}

/// Create memory from config with an LLM provider for summarization
pub fn create_memory_from_config_with_llm(
    config: &MemoryConfig,
    llm: Option<Arc<dyn LLMProvider>>,
) -> Arc<dyn Memory> {
    if config.is_compacting() {
        let summarizer: Arc<dyn Summarizer> = match llm {
            Some(provider) => Arc::new(LLMSummarizer::new(provider)),
            None => Arc::new(NoopSummarizer),
        };
        let compacting_config = config.to_compacting_config();
        Arc::new(CompactingMemory::new(summarizer, compacting_config))
    } else {
        Arc::new(InMemoryStore::new(config.max_messages))
    }
}

pub fn create_compacting_memory(
    summarizer: Arc<dyn Summarizer>,
    config: CompactingMemoryConfig,
) -> Arc<dyn Memory> {
    Arc::new(CompactingMemory::new(summarizer, config))
}

pub fn create_compacting_memory_from_config(
    summarizer: Arc<dyn Summarizer>,
    config: &MemoryConfig,
) -> Arc<dyn Memory> {
    let compacting_config = config.to_compacting_config();
    Arc::new(CompactingMemory::new(summarizer, compacting_config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_memory_variants() {
        let in_memory = create_memory("in-memory", 100);
        assert!(in_memory.is_empty());

        let unknown = create_memory("unknown", 50);
        assert!(unknown.is_empty());

        let compacting = create_memory("compacting", 100);
        assert!(compacting.is_empty());

        let config = MemoryConfig::default();
        let from_config = create_memory_from_config(&config);
        assert!(from_config.is_empty());
    }

    #[test]
    fn test_create_compacting_memory_from_config() {
        let yaml = r#"
type: compacting
max_recent_messages: 20
compress_threshold: 15
"#;
        let config: MemoryConfig = serde_yaml::from_str(yaml).unwrap();
        let memory = create_memory_from_config(&config);
        assert!(memory.is_empty());
    }
}
