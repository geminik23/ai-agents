//! CompactingMemory implementation with auto-summarization

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::Memory;
use super::context::{CompressResult, ConversationContext, estimate_tokens};
use super::summarizer::Summarizer;
use crate::error::Result;
use crate::llm::ChatMessage;
use crate::persistence::MemorySnapshot;

pub struct CompactingMemory {
    summary: RwLock<Option<String>>,
    messages: RwLock<Vec<ChatMessage>>,
    summarized_count: RwLock<usize>,
    config: CompactingMemoryConfig,
    summarizer: Arc<dyn Summarizer>,
    compression_history: RwLock<Vec<CompressionEvent>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactingMemoryConfig {
    // This field is never used: currently only `compress_threshold` and `summary_batch_size` control the compression behavior.
    // FIXME: implement for this one later.
    #[serde(default = "default_max_recent_messages")]
    pub max_recent_messages: usize,

    #[serde(default = "default_compress_threshold")]
    pub compress_threshold: usize,

    #[serde(default = "default_summarize_batch_size")]
    pub summarize_batch_size: usize,

    // FIXME: unlimited length as default value?
    #[serde(default = "default_max_summary_length")]
    pub max_summary_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub messages_compressed: usize,
    pub summary_length_before: usize,
    pub summary_length_after: usize,
}

fn default_max_recent_messages() -> usize {
    50
}

fn default_compress_threshold() -> usize {
    30
}

fn default_summarize_batch_size() -> usize {
    10
}

fn default_max_summary_length() -> usize {
    2000
}

impl Default for CompactingMemoryConfig {
    fn default() -> Self {
        Self {
            max_recent_messages: default_max_recent_messages(),
            compress_threshold: default_compress_threshold(),
            summarize_batch_size: default_summarize_batch_size(),
            max_summary_length: default_max_summary_length(),
        }
    }
}

impl CompactingMemory {
    pub fn new(summarizer: Arc<dyn Summarizer>, config: CompactingMemoryConfig) -> Self {
        Self {
            summary: RwLock::new(None),
            messages: RwLock::new(Vec::new()),
            summarized_count: RwLock::new(0),
            config,
            summarizer,
            compression_history: RwLock::new(Vec::new()),
        }
    }

    pub fn with_default_config(summarizer: Arc<dyn Summarizer>) -> Self {
        Self::new(summarizer, CompactingMemoryConfig::default())
    }

    pub fn config(&self) -> &CompactingMemoryConfig {
        &self.config
    }

    pub fn summary(&self) -> Option<String> {
        self.summary.read().clone()
    }

    pub fn summarized_count(&self) -> usize {
        *self.summarized_count.read()
    }

    pub fn compression_history(&self) -> Vec<CompressionEvent> {
        self.compression_history.read().clone()
    }

    fn record_compression(&self, messages_compressed: usize, before: usize, after: usize) {
        let event = CompressionEvent {
            timestamp: chrono::Utc::now(),
            messages_compressed,
            summary_length_before: before,
            summary_length_after: after,
        };
        self.compression_history.write().push(event);
    }
}

#[async_trait]
impl Memory for CompactingMemory {
    async fn add_message(&self, message: ChatMessage) -> Result<()> {
        self.messages.write().push(message);
        Ok(())
    }

    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ChatMessage>> {
        let messages = self.messages.read();
        match limit {
            Some(n) => {
                let start = messages.len().saturating_sub(n);
                Ok(messages[start..].to_vec())
            }
            None => Ok(messages.clone()),
        }
    }

    async fn clear(&self) -> Result<()> {
        *self.summary.write() = None;
        self.messages.write().clear();
        *self.summarized_count.write() = 0;
        self.compression_history.write().clear();
        Ok(())
    }

    fn len(&self) -> usize {
        self.messages.read().len()
    }

    async fn snapshot(&self) -> Result<MemorySnapshot> {
        let messages = self.messages.read().clone();
        let summary = self.summary.read().clone();

        let mut snapshot = MemorySnapshot::new(messages);
        if let Some(s) = summary {
            snapshot = snapshot.with_summary(s);
        }
        Ok(snapshot)
    }

    async fn restore(&self, snapshot: MemorySnapshot) -> Result<()> {
        *self.messages.write() = snapshot.messages;
        *self.summary.write() = snapshot.summary;
        *self.summarized_count.write() = 0;
        self.compression_history.write().clear();
        Ok(())
    }

    async fn get_context(&self) -> Result<ConversationContext> {
        let messages = self.messages.read().clone();
        let summary = self.summary.read().clone();
        let summarized_count = *self.summarized_count.read();
        let total_messages = messages.len() + summarized_count;

        let mut ctx = ConversationContext::with_messages(messages);
        ctx.total_messages = total_messages;

        if let Some(s) = summary {
            ctx = ctx.with_summary(s, summarized_count);
        }

        Ok(ctx)
    }

    async fn compress(&self, summarizer: Option<&dyn Summarizer>) -> Result<CompressResult> {
        let message_count = self.messages.read().len();

        if message_count < self.config.compress_threshold {
            return Ok(CompressResult::NotNeeded);
        }

        let summarizer = summarizer.unwrap_or(self.summarizer.as_ref());
        let batch_size = self.config.summarize_batch_size.min(message_count);

        let messages_to_summarize: Vec<ChatMessage> = {
            let messages = self.messages.read();
            messages[..batch_size].to_vec()
        };

        let new_summary = summarizer.summarize(&messages_to_summarize).await?;

        let summary_before_len = self.summary.read().as_ref().map(|s| s.len()).unwrap_or(0);

        let existing_summary = self.summary.read().clone();
        let combined_summary = match existing_summary {
            Some(existing) => summarizer.merge_summaries(&[existing, new_summary]).await?,
            None => new_summary,
        };

        let final_summary = if combined_summary.len() > self.config.max_summary_length {
            combined_summary[..self.config.max_summary_length].to_string()
        } else {
            combined_summary
        };

        let summary_after_len = final_summary.len();

        {
            let mut messages = self.messages.write();
            messages.drain(..batch_size);
        }

        *self.summary.write() = Some(final_summary.clone());
        *self.summarized_count.write() += batch_size;

        self.record_compression(batch_size, summary_before_len, summary_after_len);

        let tokens_before: u32 = messages_to_summarize
            .iter()
            .map(|m| estimate_tokens(&m.content))
            .sum();
        let tokens_after = estimate_tokens(&final_summary);
        let tokens_saved = tokens_before.saturating_sub(tokens_after);

        Ok(CompressResult::Compressed {
            messages_summarized: batch_size,
            new_summary_length: summary_after_len,
            tokens_saved,
        })
    }

    fn needs_compression(&self) -> bool {
        self.messages.read().len() >= self.config.compress_threshold
    }

    async fn evict_oldest(&self, count: usize) -> Result<Vec<ChatMessage>> {
        let mut messages = self.messages.write();
        let evict_count = count.min(messages.len());
        let evicted: Vec<ChatMessage> = messages.drain(..evict_count).collect();
        Ok(evicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use crate::memory::summarizer::NoopSummarizer;

    fn make_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: content.to_string(),
            name: None,
            timestamp: None,
        }
    }

    fn create_test_memory() -> CompactingMemory {
        let summarizer = Arc::new(NoopSummarizer);
        let config = CompactingMemoryConfig {
            max_recent_messages: 10,
            compress_threshold: 5,
            summarize_batch_size: 3,
            max_summary_length: 500,
        };
        CompactingMemory::new(summarizer, config)
    }

    #[tokio::test]
    async fn test_basic_add_and_get() {
        let memory = create_test_memory();

        memory.add_message(make_message("Hello")).await.unwrap();
        memory.add_message(make_message("World")).await.unwrap();

        let messages = memory.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(messages[1].content, "World");
    }

    #[tokio::test]
    async fn test_get_messages_with_limit() {
        let memory = create_test_memory();

        for i in 0..5 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let messages = memory.get_messages(Some(2)).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "msg3");
        assert_eq!(messages[1].content, "msg4");
    }

    #[tokio::test]
    async fn test_clear() {
        let memory = create_test_memory();

        memory.add_message(make_message("test")).await.unwrap();
        assert!(!memory.is_empty());

        memory.clear().await.unwrap();
        assert!(memory.is_empty());
        assert!(memory.summary().is_none());
    }

    #[tokio::test]
    async fn test_needs_compression() {
        let memory = create_test_memory();

        for i in 0..4 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }
        assert!(!memory.needs_compression());

        memory.add_message(make_message("msg4")).await.unwrap();
        assert!(memory.needs_compression());
    }

    #[tokio::test]
    async fn test_compress_not_needed() {
        let memory = create_test_memory();

        memory.add_message(make_message("msg1")).await.unwrap();
        memory.add_message(make_message("msg2")).await.unwrap();

        let result = memory.compress(None).await.unwrap();
        assert!(matches!(result, CompressResult::NotNeeded));
    }

    #[tokio::test]
    async fn test_compress_when_needed() {
        let memory = create_test_memory();

        for i in 0..6 {
            memory
                .add_message(make_message(&format!("message number {}", i)))
                .await
                .unwrap();
        }

        assert!(memory.needs_compression());

        let result = memory.compress(None).await.unwrap();

        if let CompressResult::Compressed {
            messages_summarized,
            ..
        } = result
        {
            assert_eq!(messages_summarized, 3);
        } else {
            panic!("Expected Compressed result");
        }

        assert_eq!(memory.len(), 3);
        assert!(memory.summary().is_some());
        assert_eq!(memory.summarized_count(), 3);
    }

    #[tokio::test]
    async fn test_get_context() {
        let memory = create_test_memory();

        for i in 0..6 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        memory.compress(None).await.unwrap();

        let ctx = memory.get_context().await.unwrap();
        assert!(ctx.summary.is_some());
        assert_eq!(ctx.messages.len(), 3);
        assert_eq!(ctx.summarized_count, 3);
    }

    #[tokio::test]
    async fn test_snapshot_restore() {
        let memory = create_test_memory();

        memory.add_message(make_message("msg1")).await.unwrap();
        memory.add_message(make_message("msg2")).await.unwrap();

        let snapshot = memory.snapshot().await.unwrap();
        assert_eq!(snapshot.messages.len(), 2);

        memory.clear().await.unwrap();
        assert!(memory.is_empty());

        memory.restore(snapshot).await.unwrap();
        let messages = memory.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[tokio::test]
    async fn test_compression_history() {
        let memory = create_test_memory();

        for i in 0..6 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        memory.compress(None).await.unwrap();

        let history = memory.compression_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].messages_compressed, 3);
    }

    #[test]
    fn test_config_default() {
        let config = CompactingMemoryConfig::default();
        assert_eq!(config.max_recent_messages, 50);
        assert_eq!(config.compress_threshold, 30);
        assert_eq!(config.summarize_batch_size, 10);
        assert_eq!(config.max_summary_length, 2000);
    }

    #[tokio::test]
    async fn test_evict_oldest() {
        let memory = create_test_memory();
        for i in 0..5 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let evicted = memory.evict_oldest(2).await.unwrap();
        assert_eq!(evicted.len(), 2);
        assert_eq!(evicted[0].content, "msg0");
        assert_eq!(evicted[1].content, "msg1");

        let remaining = memory.get_messages(None).await.unwrap();
        assert_eq!(remaining.len(), 3);
        assert_eq!(remaining[0].content, "msg2");
    }
}
