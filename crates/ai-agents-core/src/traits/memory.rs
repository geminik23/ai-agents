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
    #[serde(default)]
    pub summarized_count: usize,
}

impl MemorySnapshot {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            summary: None,
            summarized_count: 0,
        }
    }

    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary = Some(summary);
        self
    }

    pub fn with_summarized_count(mut self, summarized_count: usize) -> Self {
        self.summarized_count = summarized_count;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_snapshot_defaults_summarized_count() {
        let json = r#"{"messages":[],"summary":"existing"}"#;
        let snapshot: MemorySnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snapshot.summary.as_deref(), Some("existing"));
        assert_eq!(snapshot.summarized_count, 0);
    }

    #[test]
    fn snapshot_roundtrip_preserves_summarized_count() {
        let snapshot = MemorySnapshot::new(Vec::new())
            .with_summary("existing".to_string())
            .with_summarized_count(12);
        let json = serde_json::to_string(&snapshot).unwrap();
        let restored: MemorySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.summary.as_deref(), Some("existing"));
        assert_eq!(restored.summarized_count, 12);
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
