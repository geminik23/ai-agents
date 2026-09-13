//! CompactingMemory implementation with auto-summarization

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use ai_agents_core::{ChatMessage, MemorySnapshot, Result};

use super::Memory;
use super::context::{CompressResult, ConversationContext, estimate_tokens};
use super::native::{NativeRetentionInspection, readable_projection};
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
        let requested = count.min(messages.len());
        let inspection = NativeRetentionInspection::inspect(&messages)?;
        let evict_count = if requested == 0 {
            0
        } else {
            inspection
                .safe_prefix_len_between(requested, messages.len())
                .ok_or_else(|| {
                    ai_agents_core::AgentError::MemoryError(
                        "eviction would split the protected signed native exchange".to_string(),
                    )
                })?
        };
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
        let configured_compressible_count = message_count - protected_count;
        let inspection = {
            let messages = self.messages.read();
            NativeRetentionInspection::inspect(&messages)?
        };
        let compressible_count = inspection
            .protected_suffix_start()
            .map_or(configured_compressible_count, |start| {
                configured_compressible_count.min(start)
            });
        if compressible_count == 0 {
            return Ok(CompressResult::NotNeeded);
        }
        let requested_batch = self
            .config
            .summarize_batch_size
            .max(1)
            .min(compressible_count);
        // Prefer finishing the group containing the configured batch boundary. If that would
        // enter the retained suffix, summarize the largest earlier complete group instead.
        let batch_size = inspection
            .safe_prefix_len_between(requested_batch, compressible_count)
            .or_else(|| {
                let earlier = inspection.safe_prefix_len_at_most(requested_batch);
                (earlier > 0).then_some(earlier)
            })
            .unwrap_or(0);
        if batch_size == 0 {
            return Ok(CompressResult::NotNeeded);
        }

        let original_messages_to_summarize: Vec<ChatMessage> = {
            let messages = self.messages.read();
            messages[..batch_size].to_vec()
        };
        // Projection happens before the dynamic summarizer boundary so custom implementations
        // cannot observe or accidentally persist provider replay state.
        let mut messages_to_summarize = readable_projection(&original_messages_to_summarize)?;
        let incomplete_exchanges = inspection
            .incomplete_exchanges()
            .into_iter()
            .filter(|(message_index, _)| *message_index < batch_size)
            .collect::<Vec<_>>();
        for (message_index, missing_result_ids) in &incomplete_exchanges {
            let status = serde_json::json!({
                "native_exchange_status": "incomplete",
                "missing_result_ids": missing_result_ids,
                "description": "final execution result was not recorded"
            });
            messages_to_summarize[*message_index].content.push('\n');
            messages_to_summarize[*message_index]
                .content
                .push_str(&status.to_string());
        }

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

        // The framework, rather than a fallible/custom summarizer, owns the durable uncertainty
        // marker so completed compression cannot erase evidence that a tool result was missing.
        let incomplete_suffix = incomplete_exchanges
            .iter()
            .map(|(_, missing_result_ids)| {
                serde_json::json!({
                    "native_exchange_status": "incomplete",
                    "missing_result_ids": missing_result_ids,
                    "description": "final execution result was not recorded"
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let suffix = (!incomplete_suffix.is_empty()).then(|| format!("\n{incomplete_suffix}"));
        let suffix_chars = suffix
            .as_deref()
            .map(|suffix| suffix.chars().count())
            .unwrap_or(0);
        if suffix_chars > self.config.max_summary_length {
            return Err(ai_agents_core::AgentError::MemoryError(
                "summary limit cannot retain incomplete native exchange status".to_string(),
            ));
        }
        let content_limit = self.config.max_summary_length - suffix_chars;
        let truncated = prefix_at_char_boundary(&combined_summary, content_limit);
        let mut final_summary = if truncated.len() < combined_summary.len() {
            truncated.to_string()
        } else {
            combined_summary
        };
        if let Some(suffix) = suffix {
            final_summary.push_str(&suffix);
        }

        let summary_after_len = final_summary.len();

        {
            let mut messages = self.messages.write();
            messages.drain(..batch_size);
        }

        *self.summary.write() = Some(final_summary.clone());
        *self.summarized_count.write() += batch_size;

        self.record_compression(batch_size, summary_before_len, summary_after_len);

        let tokens_before: u32 = existing_summary_tokens.saturating_add(
            original_messages_to_summarize
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
    use ai_agents_core::{
        AgentError, Memory as CoreMemory, NativeCallBinding, NativeProviderState,
        NativeProviderTarget, Role, ToolCall, encode_native_tool_call_markers,
        encode_native_tool_result_marker,
    };
    use tokio::time::{Duration, timeout};

    fn make_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: content.to_string(),
            name: None,
            timestamp: None,
        }
    }

    fn signed_turn(exchange_id: &str) -> Vec<ChatMessage> {
        let call = ToolCall {
            id: format!("{exchange_id}-call"),
            name: "lookup".to_string(),
            arguments: serde_json::json!({"query":"fixture"}),
        };
        let state = NativeProviderState::new(
            exchange_id,
            "google",
            "generateContent",
            NativeProviderTarget::new("https://example.invalid/v1beta/", "fixture-model").unwrap(),
            serde_json::json!({
                "role":"model",
                "parts":[{
                    "functionCall":{"name":"lookup","args":{"query":"fixture"}},
                    "thoughtSignature":"fixture-signature"
                }]
            }),
            vec![NativeCallBinding::new(&call.id, 0).unwrap()],
        )
        .unwrap();
        vec![
            ChatMessage::user("signed user request"),
            ChatMessage::assistant(
                encode_native_tool_call_markers(std::slice::from_ref(&call), Some(&state)).unwrap(),
            ),
            ChatMessage::function(
                "lookup",
                encode_native_tool_result_marker(&call, serde_json::json!({"ok":true})).unwrap(),
            ),
            ChatMessage::assistant("signed turn final response"),
        ]
    }

    fn message_contents(messages: &[ChatMessage]) -> Vec<&str> {
        messages
            .iter()
            .map(|message| message.content.as_str())
            .collect()
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

    struct FailingSummarizer;

    #[async_trait]
    impl Summarizer for FailingSummarizer {
        async fn summarize(&self, _messages: &[ChatMessage]) -> Result<String> {
            Err(AgentError::MemoryError("summary failed".to_string()))
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

    #[derive(Default)]
    struct CapturingSummarizer {
        batches: RwLock<Vec<Vec<ChatMessage>>>,
    }

    #[async_trait]
    impl Summarizer for CapturingSummarizer {
        async fn summarize(&self, messages: &[ChatMessage]) -> Result<String> {
            self.batches.write().push(messages.to_vec());
            Ok(messages
                .iter()
                .map(|message| message.content.clone())
                .collect::<Vec<_>>()
                .join(" | "))
        }
    }

    struct DroppingStatusSummarizer;

    #[async_trait]
    impl Summarizer for DroppingStatusSummarizer {
        async fn summarize(&self, _messages: &[ChatMessage]) -> Result<String> {
            Ok("summary that omitted the native status".to_string())
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
    async fn test_multilingual_recent_tail_remains_verbatim() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 4,
            compress_threshold: 6,
            summarize_batch_size: 3,
            max_summary_length: 100_000,
        };
        let memory = CompactingMemory::new(Arc::new(NoopSummarizer), config);
        let large_mixed_message = "protected 한국어 日本語 العربية English emoji 🧭".to_string();
        let source = vec![
            "Old English context".to_string(),
            "이전 한국어 문맥".to_string(),
            "古い日本語の文脈".to_string(),
            "سياق عربي قديم".to_string(),
            "पुराना हिंदी संदर्भ".to_string(),
            "Keep 한국어 and English together".to_string(),
            "保留する日本語メッセージ".to_string(),
            "احتفظ بهذه الرسالة العربية".to_string(),
            large_mixed_message,
        ];
        let expected_tail = source[source.len() - 4..].to_vec();

        for content in &source {
            memory.add_message(make_message(content)).await.unwrap();
        }

        memory.compress(None).await.unwrap();
        memory.compress(None).await.unwrap();
        let remaining = memory.get_messages(None).await.unwrap();

        assert!(!memory.needs_compression());
        assert_eq!(
            remaining
                .iter()
                .map(|message| message.content.clone())
                .collect::<Vec<_>>(),
            expected_tail
        );
        assert_eq!(memory.summarized_count() + memory.len(), source.len());
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
    async fn test_initial_summary_failure_rolls_back_all_accounting() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 3,
            compress_threshold: 5,
            summarize_batch_size: 2,
            max_summary_length: 100_000,
        };
        let memory = CompactingMemory::new(Arc::new(FailingSummarizer), config);
        let source = vec![
            "English before failure".to_string(),
            "실패 전 한국어".to_string(),
            "失敗前の日本語".to_string(),
            "قبل الفشل".to_string(),
            "large mixed message 한界🙂abc".to_string(),
        ];
        for content in &source {
            memory.add_message(make_message(content)).await.unwrap();
        }

        let snapshot_before = memory.snapshot().await.unwrap();
        let context_before = memory.get_context().await.unwrap();
        let error = memory.compress(None).await.unwrap_err();
        let snapshot_after = memory.snapshot().await.unwrap();
        let context_after = memory.get_context().await.unwrap();

        assert!(error.to_string().contains("summary failed"));
        assert_eq!(
            message_contents(&snapshot_after.messages),
            message_contents(&snapshot_before.messages)
        );
        assert_eq!(snapshot_after.summary, snapshot_before.summary);
        assert_eq!(
            snapshot_after.summarized_count,
            snapshot_before.summarized_count
        );
        assert_eq!(context_after.total_messages, context_before.total_messages);
        assert_eq!(
            context_after.summarized_count,
            context_before.summarized_count
        );
        assert_eq!(
            context_after.estimated_tokens(),
            context_before.estimated_tokens()
        );
        assert!(memory.compression_history().is_empty());
        assert!(memory.needs_compression());

        let retry = memory.compress(Some(&NoopSummarizer)).await.unwrap();
        assert!(matches!(
            retry,
            CompressResult::Compressed {
                messages_summarized: 2,
                ..
            }
        ));
        assert_eq!(memory.summarized_count() + memory.len(), source.len());
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
        let serialized = serde_json::to_string(&snapshot).unwrap();
        let persisted: MemorySnapshot = serde_json::from_str(&serialized).unwrap();

        memory.clear().await.unwrap();
        memory.restore(persisted).await.unwrap();
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

    #[tokio::test]
    async fn compression_expands_batch_to_complete_signed_past_turn_and_projects_state() {
        let summarizer = Arc::new(CapturingSummarizer::default());
        let config = CompactingMemoryConfig {
            max_recent_messages: 1,
            compress_threshold: 5,
            summarize_batch_size: 2,
            max_summary_length: 100_000,
        };
        let memory = CompactingMemory::new(summarizer.clone(), config);
        for message in signed_turn("compress-past") {
            memory.add_message(message).await.unwrap();
        }
        memory
            .add_message(ChatMessage::user("new user turn"))
            .await
            .unwrap();

        let result = memory.compress(None).await.unwrap();

        assert!(matches!(
            result,
            CompressResult::Compressed {
                messages_summarized: 4,
                ..
            }
        ));
        let remaining = memory.get_messages(None).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].content, "new user turn");
        let batches = summarizer.batches.read();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 4);
        let projected = batches[0]
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!projected.contains("fixture-signature"));
        assert!(!projected.contains("_ai_agents_provider_state"));
        assert!(projected.contains("native_tool_calls"));
        assert!(projected.contains("native_tool_results"));
        assert!(!memory.summary().unwrap().contains("fixture-signature"));
    }

    #[tokio::test]
    async fn compression_keeps_latest_signed_turn_intact() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 0,
            compress_threshold: 4,
            summarize_batch_size: 2,
            max_summary_length: 100_000,
        };
        let memory = CompactingMemory::new(Arc::new(NoopSummarizer), config);
        for message in signed_turn("compress-active") {
            memory.add_message(message).await.unwrap();
        }

        let before = memory.get_messages(None).await.unwrap();
        let result = memory.compress(None).await.unwrap();
        let after = memory.get_messages(None).await.unwrap();

        assert!(matches!(result, CompressResult::NotNeeded));
        assert_eq!(after.len(), before.len());
        assert!(after[1].content.contains("fixture-signature"));
        assert!(memory.summary().is_none());
    }

    #[tokio::test]
    async fn compression_projects_missing_result_for_ended_signed_turn() {
        let summarizer = Arc::new(CapturingSummarizer::default());
        let config = CompactingMemoryConfig {
            max_recent_messages: 1,
            compress_threshold: 4,
            summarize_batch_size: 2,
            max_summary_length: 100_000,
        };
        let memory = CompactingMemory::new(summarizer.clone(), config);
        let mut incomplete = signed_turn("compress-incomplete");
        incomplete.remove(2);
        for message in incomplete {
            memory.add_message(message).await.unwrap();
        }
        memory
            .add_message(ChatMessage::user("new user turn"))
            .await
            .unwrap();

        let result = memory.compress(None).await.unwrap();

        assert!(matches!(
            result,
            CompressResult::Compressed {
                messages_summarized: 3,
                ..
            }
        ));
        let batches = summarizer.batches.read();
        let projected = batches[0]
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(projected.contains("native_exchange_status"));
        assert!(projected.contains("final execution result was not recorded"));
        assert!(!projected.contains("fixture-signature"));
        assert!(memory.summary().unwrap().contains("native_exchange_status"));
    }

    #[tokio::test]
    async fn framework_retains_incomplete_status_when_summarizer_drops_it() {
        let config = CompactingMemoryConfig {
            max_recent_messages: 1,
            compress_threshold: 4,
            summarize_batch_size: 2,
            max_summary_length: 100_000,
        };
        let memory = CompactingMemory::new(Arc::new(DroppingStatusSummarizer), config);
        let mut incomplete = signed_turn("compress-dropped-status");
        incomplete.remove(2);
        for message in incomplete {
            memory.add_message(message).await.unwrap();
        }
        memory
            .add_message(ChatMessage::user("new user turn"))
            .await
            .unwrap();

        memory.compress(None).await.unwrap();

        let summary = memory.summary().unwrap();
        assert!(summary.contains("summary that omitted the native status"));
        assert!(summary.contains("native_exchange_status"));
        assert!(!summary.contains("fixture-signature"));
    }

    #[tokio::test]
    async fn compacting_eviction_keeps_signed_turn_atomic() {
        let memory = create_test_memory();
        for message in signed_turn("evict-past") {
            memory.add_message(message).await.unwrap();
        }
        memory
            .add_message(ChatMessage::user("new user turn"))
            .await
            .unwrap();

        let evicted = memory.evict_oldest(1).await.unwrap();

        assert_eq!(evicted.len(), 4);
        let remaining = memory.get_messages(None).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].content, "new user turn");
    }

    #[test]
    fn test_prefix_at_char_boundary_handles_unicode() {
        let text = "계약서 내용을 확인하고 싶어서";
        let prefix = prefix_at_char_boundary(text, 5);
        assert_eq!(prefix.chars().count(), 5);
        assert!(text.starts_with(prefix));
    }
}
