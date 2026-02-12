//! Memory trait for conversation storage

use async_trait::async_trait;

use crate::error::Result;
use crate::message::ChatMessage;

/// Snapshot of memory state for persistence
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct MemorySnapshot {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub summary: Option<String>,
}

impl MemorySnapshot {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            summary: None,
        }
    }

    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary = Some(summary);
        self
    }
}

/// Core memory trait for storing conversation history.
///
/// Built-in implementations: `InMemoryStore` (simple) and `CompactingMemory`
/// (with LLM-based summarization). Implement this for custom storage strategies.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Append a message to conversation history.
    async fn add_message(&self, message: ChatMessage) -> Result<()>;
    /// Get messages. `Some(n)` returns the most recent N messages.
    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<ChatMessage>>;
    /// Remove all messages and reset state.
    async fn clear(&self) -> Result<()>;
    /// Number of messages currently stored.
    fn len(&self) -> usize;

    /// Returns `true` if no messages are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize current state for persistence.
    async fn snapshot(&self) -> Result<MemorySnapshot> {
        Ok(MemorySnapshot::new(self.get_messages(None).await?))
    }

    /// Restore from a previously saved snapshot, replacing current state.
    async fn restore(&self, snapshot: MemorySnapshot) -> Result<()>;

    /// Remove the oldest N messages. Returns empty vec by default.
    async fn evict_oldest(&self, _count: usize) -> Result<Vec<ChatMessage>> {
        Ok(vec![])
    }
}
