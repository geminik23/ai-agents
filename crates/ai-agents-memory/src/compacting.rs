//! CompactingMemory implementation with auto-summarization

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use ai_agents_core::{ChatMessage, MemorySnapshot, Result};

use super::Memory;
use super::context::{CompressResult, ConversationContext, estimate_tokens};
use super::summarizer::Summarizer;

fn prefix_at_char_boundary(text: &str, max_chars: usize) -> &str {
    if max_chars == 0 {
        return "";
    }

    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

pub struct CompactingMemory {
    operation_lock: AsyncMutex<()>,
    summary: RwLock<Option<String>>,
    messages: RwLock<Vec<ChatMessage>>,
    summarized_count: RwLock<usize>,
    config: CompactingMemoryConfig,
    summarizer: Arc<dyn Summarizer>,
    compression_history: RwLock<Vec<CompressionEvent>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactingMemoryConfig {
    /// Maximum recent messages retained verbatim, clamped below the compression threshold.
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

fn protected_recent_count(config: &CompactingMemoryConfig, message_count: usize) -> usize {
    if config.max_recent_messages < config.compress_threshold {
        return config.max_recent_messages.min(message_count);
    }

    let batch_at_threshold = config
        .summarize_batch_size
        .max(1)
        .min(config.compress_threshold);
    let retention_cap = config.compress_threshold.saturating_sub(batch_at_threshold);
    config
        .max_recent_messages
        .min(retention_cap)
        .min(message_count)
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
            operation_lock: AsyncMutex::new(()),
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
impl ai_agents_core::Memory for CompactingMemory {
    async fn add_message(&self, message: ChatMessage) -> Result<()> {
        let _operation = self.operation_lock.lock().await;
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
        let _operation = self.operation_lock.lock().await;
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
        let _operation = self.operation_lock.lock().await;
        let messages = self.messages.read().clone();
        let summary = self.summary.read().clone();
        let summarized_count = *self.summarized_count.read();

        let mut snapshot = MemorySnapshot::new(messages).with_summarized_count(summarized_count);
        if let Some(s) = summary {
            snapshot = snapshot.with_summary(s);
        }
        Ok(snapshot)
    }

    async fn restore(&self, snapshot: MemorySnapshot) -> Result<()> {
        let _operation = self.operation_lock.lock().await;
        *self.messages.write() = snapshot.messages;
        *self.summary.write() = snapshot.summary;
        *self.summarized_count.write() = snapshot.summarized_count;
        self.compression_history.write().clear();
        Ok(())
    }

    async fn evict_oldest(&self, count: usize) -> Result<Vec<ChatMessage>> {
        let _operation = self.operation_lock.lock().await;
        let mut messages = self.messages.write();
        let evict_count = count.min(messages.len());
        let evicted: Vec<ChatMessage> = messages.drain(..evict_count).collect();
        Ok(evicted)
    }
}

#[async_trait]
impl Memory for CompactingMemory {
    async fn get_context(&self) -> Result<ConversationContext> {
        let _operation = self.operation_lock.lock().await;
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
        let _operation = self.operation_lock.lock().await;
        let message_count = self.messages.read().len();

        if message_count == 0 || message_count < self.config.compress_threshold {
            return Ok(CompressResult::NotNeeded);
        }

        let summarizer = summarizer.unwrap_or(self.summarizer.as_ref());
        let protected_count = protected_recent_count(&self.config, message_count);
        let compressible_count = message_count - protected_count;
        let batch_size = self
            .config
            .summarize_batch_size
            .max(1)
            .min(compressible_count);

        let messages_to_summarize: Vec<ChatMessage> = {
            let messages = self.messages.read();
            messages[..batch_size].to_vec()
        };

        let new_summary = summarizer.summarize(&messages_to_summarize).await?;

        let summary_before_len = self.summary.read().as_ref().map(|s| s.len()).unwrap_or(0);

        let existing_summary = self.summary.read().clone();
        let existing_summary_tokens = existing_summary
            .as_deref()
            .map(estimate_tokens)
            .unwrap_or(0);
        let combined_summary = match existing_summary {
            Some(existing) => summarizer.merge_summaries(&[existing, new_summary]).await?,
            None => new_summary,
        };

        let truncated = prefix_at_char_boundary(&combined_summary, self.config.max_summary_length);
        let final_summary = if truncated.len() < combined_summary.len() {
            truncated.to_string()
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

        let tokens_before: u32 = existing_summary_tokens.saturating_add(
            messages_to_summarize
                .iter()
                .map(|m| estimate_tokens(&m.content))
                .sum(),
        );
        let tokens_after = estimate_tokens(&final_summary);
        let tokens_saved = tokens_before.saturating_sub(tokens_after);

        Ok(CompressResult::Compressed {
            messages_summarized: batch_size,
            new_summary_length: summary_after_len,
            tokens_saved,
        })
    }

    fn needs_compression(&self) -> bool {
        let message_count = self.messages.read().len();
        message_count > 0 && message_count >= self.config.compress_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summarizer::NoopSummarizer;
    use ai_agents_core::{AgentError, Memory as CoreMemory, Role};
    use tokio::time::{Duration, timeout};

    fn make_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: content.to_string(),
            name: None,
            timestamp: None,
        }
    }

    struct BlockingSummarizer {
        started: tokio::sync::Notify,
        release: tokio::sync::Notify,
        batches: RwLock<Vec<Vec<String>>>,
    }

    impl BlockingSummarizer {
        fn new() -> Self {
            Self {
                started: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(),
                batches: RwLock::new(Vec::new()),
            }
        }

        async fn wait_until_started(&self) {
            self.started.notified().await;
        }

        fn release(&self) {
            self.release.notify_one();
        }

        fn batches(&self) -> Vec<Vec<String>> {
            self.batches.read().clone()
        }
    }

    #[async_trait]
    impl Summarizer for BlockingSummarizer {
        async fn summarize(&self, messages: &[ChatMessage]) -> Result<String> {
            let contents: Vec<_> = messages
                .iter()
                .map(|message| message.content.clone())
                .collect();
            self.batches.write().push(contents.clone());
            self.started.notify_one();
            self.release.notified().await;
            Ok(contents.join(" | "))
        }
    }

    struct FailingMergeSummarizer;

    #[async_trait]
    impl Summarizer for FailingMergeSummarizer {
        async fn summarize(&self, _messages: &[ChatMessage]) -> Result<String> {
            Ok("new summary".to_string())
        }

        async fn merge_summaries(&self, _summaries: &[String]) -> Result<String> {
            Err(AgentError::MemoryError("merge failed".to_string()))
        }
    }

    fn create_test_memory() -> CompactingMemory {
        let summarizer = Arc::new(NoopSummarizer);
        let config = CompactingMemoryConfig {
            max_recent_messages: 3,
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
    async fn test_compress_preserves_configured_recent_tail() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 3,
            compress_threshold: 5,
            summarize_batch_size: 10,
            max_summary_length: 500,
        };
        let memory = CompactingMemory::new(Arc::new(NoopSummarizer), config);

        for i in 0..7 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let result = memory.compress(None).await.unwrap();
        assert!(matches!(
            result,
            CompressResult::Compressed {
                messages_summarized: 4,
                ..
            }
        ));
        let remaining = memory.get_messages(None).await.unwrap();
        let contents: Vec<_> = remaining
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(contents, vec!["msg4", "msg5", "msg6"]);
    }

    #[tokio::test]
    async fn test_compress_clamps_recent_tail_for_configured_batch() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 100,
            compress_threshold: 5,
            summarize_batch_size: 10,
            max_summary_length: 500,
        };
        let memory = CompactingMemory::new(Arc::new(NoopSummarizer), config);

        for i in 0..5 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let result = memory.compress(None).await.unwrap();
        assert!(matches!(
            result,
            CompressResult::Compressed {
                messages_summarized: 5,
                ..
            }
        ));
        assert!(memory.get_messages(None).await.unwrap().is_empty());
        assert!(!memory.needs_compression());
    }

    #[tokio::test]
    async fn test_default_config_compresses_full_batches_at_steady_state() {
        let memory =
            CompactingMemory::new(Arc::new(NoopSummarizer), CompactingMemoryConfig::default());

        for i in 0..30 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        for round in 0..4 {
            let result = memory.compress(None).await.unwrap();
            assert!(matches!(
                result,
                CompressResult::Compressed {
                    messages_summarized: 10,
                    ..
                }
            ));
            assert_eq!(memory.len(), 20);
            assert!(!memory.needs_compression());

            if round < 3 {
                let start = 30 + round * 10;
                for i in start..start + 10 {
                    memory
                        .add_message(make_message(&format!("msg{}", i)))
                        .await
                        .unwrap();
                }
                assert_eq!(memory.len(), 30);
                assert!(memory.needs_compression());
            }
        }

        assert_eq!(memory.summarized_count(), 40);
        let remaining = memory.get_messages(None).await.unwrap();
        assert_eq!(remaining.first().unwrap().content, "msg40");
        assert_eq!(remaining.last().unwrap().content, "msg59");
    }

    #[test]
    fn test_protected_recent_count_edge_cases() {
        let non_conflicting = CompactingMemoryConfig {
            max_recent_messages: 25,
            compress_threshold: 30,
            summarize_batch_size: 10,
            max_summary_length: 500,
        };
        assert_eq!(protected_recent_count(&non_conflicting, 30), 25);

        let conflicting = CompactingMemoryConfig {
            max_recent_messages: 50,
            compress_threshold: 30,
            summarize_batch_size: 10,
            max_summary_length: 500,
        };
        assert_eq!(protected_recent_count(&conflicting, 30), 20);

        let oversized_batch = CompactingMemoryConfig {
            max_recent_messages: 5,
            compress_threshold: 5,
            summarize_batch_size: 10,
            max_summary_length: 500,
        };
        assert_eq!(protected_recent_count(&oversized_batch, 5), 0);

        let zero_batch = CompactingMemoryConfig {
            max_recent_messages: 5,
            compress_threshold: 5,
            summarize_batch_size: 0,
            max_summary_length: 500,
        };
        assert_eq!(protected_recent_count(&zero_batch, 5), 4);

        let zero_threshold = CompactingMemoryConfig {
            max_recent_messages: 5,
            compress_threshold: 0,
            summarize_batch_size: 10,
            max_summary_length: 500,
        };
        assert_eq!(protected_recent_count(&zero_threshold, 5), 0);
    }

    #[tokio::test]
    async fn test_compression_failure_is_non_destructive() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 2,
            compress_threshold: 5,
            summarize_batch_size: 3,
            max_summary_length: 500,
        };
        let memory = CompactingMemory::new(Arc::new(NoopSummarizer), config);

        for i in 0..5 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }
        memory.compress(None).await.unwrap();
        for i in 5..8 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let messages_before = memory.get_messages(None).await.unwrap();
        let summary_before = memory.summary();
        let summarized_count_before = memory.summarized_count();
        let history_len_before = memory.compression_history().len();

        let error = memory.compress(Some(&FailingMergeSummarizer)).await;
        assert!(error.is_err());
        let messages_after = memory.get_messages(None).await.unwrap();
        let before_contents: Vec<_> = messages_before
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        let after_contents: Vec<_> = messages_after
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(after_contents, before_contents);
        assert_eq!(memory.summary(), summary_before);
        assert_eq!(memory.summarized_count(), summarized_count_before);
        assert_eq!(memory.compression_history().len(), history_len_before);
    }

    #[tokio::test]
    async fn test_concurrent_compressions_are_serialized() {
        let summarizer = Arc::new(BlockingSummarizer::new());
        let config = CompactingMemoryConfig {
            max_recent_messages: 2,
            compress_threshold: 5,
            summarize_batch_size: 3,
            max_summary_length: 500,
        };
        let memory = Arc::new(CompactingMemory::new(summarizer.clone(), config));
        for i in 0..5 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let first_memory = memory.clone();
        let first = tokio::spawn(async move { first_memory.compress(None).await });
        summarizer.wait_until_started().await;

        let second_memory = memory.clone();
        let mut second = tokio::spawn(async move { second_memory.compress(None).await });
        assert!(
            timeout(Duration::from_millis(50), &mut second)
                .await
                .is_err()
        );

        summarizer.release();
        assert!(matches!(
            first.await.unwrap().unwrap(),
            CompressResult::Compressed {
                messages_summarized: 3,
                ..
            }
        ));
        assert!(matches!(
            second.await.unwrap().unwrap(),
            CompressResult::NotNeeded
        ));
        assert_eq!(summarizer.batches(), vec![vec!["msg0", "msg1", "msg2"]]);
        let remaining = memory.get_messages(None).await.unwrap();
        let contents: Vec<_> = remaining
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(contents, vec!["msg3", "msg4"]);
    }

    #[tokio::test]
    async fn test_compression_serializes_add_message() {
        let summarizer = Arc::new(BlockingSummarizer::new());
        let config = CompactingMemoryConfig {
            max_recent_messages: 2,
            compress_threshold: 5,
            summarize_batch_size: 3,
            max_summary_length: 500,
        };
        let memory = Arc::new(CompactingMemory::new(summarizer.clone(), config));
        for i in 0..5 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let compress_memory = memory.clone();
        let compress = tokio::spawn(async move { compress_memory.compress(None).await });
        summarizer.wait_until_started().await;

        let add_memory = memory.clone();
        let mut add =
            tokio::spawn(async move { add_memory.add_message(make_message("msg5")).await });
        assert!(timeout(Duration::from_millis(50), &mut add).await.is_err());

        summarizer.release();
        compress.await.unwrap().unwrap();
        add.await.unwrap().unwrap();
        assert_eq!(summarizer.batches(), vec![vec!["msg0", "msg1", "msg2"]]);
        let remaining = memory.get_messages(None).await.unwrap();
        let contents: Vec<_> = remaining
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(contents, vec!["msg3", "msg4", "msg5"]);
    }

    #[tokio::test]
    async fn test_compression_serializes_eviction() {
        let summarizer = Arc::new(BlockingSummarizer::new());
        let config = CompactingMemoryConfig {
            max_recent_messages: 2,
            compress_threshold: 5,
            summarize_batch_size: 3,
            max_summary_length: 500,
        };
        let memory = Arc::new(CompactingMemory::new(summarizer.clone(), config));
        for i in 0..5 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let compress_memory = memory.clone();
        let compress = tokio::spawn(async move { compress_memory.compress(None).await });
        summarizer.wait_until_started().await;

        let evict_memory = memory.clone();
        let mut evict = tokio::spawn(async move { evict_memory.evict_oldest(1).await });
        assert!(
            timeout(Duration::from_millis(50), &mut evict)
                .await
                .is_err()
        );

        summarizer.release();
        compress.await.unwrap().unwrap();
        let evicted = evict.await.unwrap().unwrap();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].content, "msg3");
        assert_eq!(summarizer.batches(), vec![vec!["msg0", "msg1", "msg2"]]);
        let remaining = memory.get_messages(None).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].content, "msg4");
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
    async fn test_snapshot_restore_preserves_recent_tail() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 3,
            compress_threshold: 5,
            summarize_batch_size: 10,
            max_summary_length: 500,
        };
        let memory = CompactingMemory::new(Arc::new(NoopSummarizer), config);

        for i in 0..7 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }
        memory.compress(None).await.unwrap();
        let snapshot = memory.snapshot().await.unwrap();
        assert_eq!(snapshot.summarized_count, 4);

        memory.clear().await.unwrap();
        memory.restore(snapshot).await.unwrap();
        assert!(memory.summary().is_some());
        assert_eq!(memory.summarized_count(), 4);
        for i in 7..9 {
            memory
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }
        memory.compress(None).await.unwrap();

        let remaining = memory.get_messages(None).await.unwrap();
        let contents: Vec<_> = remaining
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(contents, vec!["msg6", "msg7", "msg8"]);
        let context = memory.get_context().await.unwrap();
        assert_eq!(context.summarized_count, 6);
        assert_eq!(context.total_messages, 9);
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

    #[test]
    fn test_config_rejects_unknown_fields() {
        let yaml = r#"
max_recent_messages: 5
compress_thresold: 10
"#;
        let error = serde_yaml::from_str::<CompactingMemoryConfig>(yaml).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unknown field `compress_thresold`")
        );
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

    #[test]
    fn test_prefix_at_char_boundary_handles_unicode() {
        let text = "계약서 내용을 확인하고 싶어서";
        let prefix = prefix_at_char_boundary(text, 5);
        assert_eq!(prefix.chars().count(), 5);
        assert!(text.starts_with(prefix));
    }
}
