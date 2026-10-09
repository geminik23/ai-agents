//! Adds persisted input provenance without rewriting provider-native message content or tool names.

use ai_agents_core::{ChatMessage, MemorySnapshot, Result, Role};
use ai_agents_memory::{CompressResult, ConversationContext, Memory, Summarizer};
use async_trait::async_trait;
use std::sync::Arc;

pub(crate) struct TaskMemory {
    inner: Arc<dyn Memory>,
    gate: crate::runtime::RootTurnGate,
}

impl TaskMemory {
    /// The decorator is installed before any turn, and keeps the original storage and compression authority.
    pub(crate) fn new(inner: Arc<dyn Memory>, gate: crate::runtime::RootTurnGate) -> Self {
        Self { inner, gate }
    }
}

#[async_trait]
impl ai_agents_core::Memory for TaskMemory {
    /// Only the initial objective is conversational; task-produced history stays out of later actor extraction.
    async fn add_message(&self, mut message: ChatMessage) -> Result<()> {
        if let Some(input) = super::current_turn_input(&self.gate)
            && (message.role != Role::User || input.controller_only())
        {
            message.provenance = Some(ai_agents_core::message::MessageProvenance {
                run_id: input.owner.run_id.clone(),
            });
        }
        self.inner.add_message(message).await
    }
    /// Exact message content and provenance are returned unchanged for native provider replay.
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ChatMessage>> {
        self.inner.get_messages(limit).await
    }
    /// Reset ownership is checked by the runtime before the backing store is cleared.
    async fn clear(&self) -> Result<()> {
        self.inner.clear().await
    }
    /// Length continues to describe the single backing conversation store.
    fn len(&self) -> usize {
        self.inner.len()
    }
    /// Snapshot serialization retains provenance while preserving old omitted-field readability.
    async fn snapshot(&self) -> Result<MemorySnapshot> {
        self.inner.snapshot().await
    }
    /// Restore keeps native messages and their projection classification in one snapshot.
    async fn restore(&self, snapshot: MemorySnapshot) -> Result<()> {
        self.inner.restore(snapshot).await
    }
    /// Eviction retains classification so extraction cannot turn controller output into user facts.
    async fn evict_oldest(&self, count: usize) -> Result<Vec<ChatMessage>> {
        self.inner.evict_oldest(count).await
    }
}

#[async_trait]
impl Memory for TaskMemory {
    /// The wrapper cannot make hidden or unwrapped provider work in a custom memory conforming.
    fn supports_task_admission(&self) -> bool {
        self.inner.supports_task_admission()
    }

    /// Provider context uses the backing store's exact native-history grouping and summary policy.
    async fn get_context(&self) -> Result<ConversationContext> {
        self.inner.get_context().await
    }
    /// Compression remains owned by the existing implementation, not a second task memory store.
    async fn compress(&self, summarizer: Option<&dyn Summarizer>) -> Result<CompressResult> {
        self.inner.compress(summarizer).await
    }
    /// The decorator does not redefine compression thresholds.
    fn needs_compression(&self) -> bool {
        self.inner.needs_compression()
    }
}
