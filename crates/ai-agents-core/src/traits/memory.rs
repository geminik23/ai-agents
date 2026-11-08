//! Memory trait for conversation storage

use async_trait::async_trait;

use crate::error::Result;
use crate::message::ChatMessage;

/// An observation that a signed native exchange has no recorded result for some calls.
///
/// This is advisory history, not proof of non-execution, cleanup, or authority to retry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IncompleteNativeExchange {
    /// The recorded exchange identity, unavailable for pre-structured legacy summaries.
    pub exchange_id: Option<String>,
    pub missing_result_ids: Vec<String>,
    /// Legacy text may have come from a model or user and cannot prove a native exchange existed.
    #[serde(default)]
    pub unverified_legacy: bool,
}

/// Snapshot of memory state for persistence
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct MemorySnapshot {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub summarized_count: usize,
    /// Incomplete signed exchanges retained separately from model-generated summary text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub incomplete_native_exchanges: Vec<IncompleteNativeExchange>,
}

impl MemorySnapshot {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            summary: None,
            summarized_count: 0,
            incomplete_native_exchanges: Vec::new(),
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

    /// Records advisory incomplete history without treating summary text as execution evidence.
    pub fn with_incomplete_native_exchanges(
        mut self,
        exchanges: Vec<IncompleteNativeExchange>,
    ) -> Self {
        self.incomplete_native_exchanges = exchanges;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_snapshot_defaults_summarized_count() {
        let json = r#"{"messages":[],"summary":"existing"}"#;
        let snapshot: MemorySnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snapshot.summary.as_deref(), Some("existing"));
        assert_eq!(snapshot.summarized_count, 0);
        assert!(snapshot.incomplete_native_exchanges.is_empty());
    }

    #[test]
    fn incomplete_observations_survive_snapshot_roundtrip() {
        let snapshot = MemorySnapshot::default().with_incomplete_native_exchanges(vec![
            IncompleteNativeExchange {
                exchange_id: Some("exchange".into()),
                missing_result_ids: vec!["call".into()],
                unverified_legacy: false,
            },
            IncompleteNativeExchange {
                exchange_id: None,
                missing_result_ids: vec!["legacy".into()],
                unverified_legacy: true,
            },
        ]);
        let restored: MemorySnapshot =
            serde_json::from_str(&serde_json::to_string(&snapshot).unwrap()).unwrap();
        assert_eq!(
            restored.incomplete_native_exchanges,
            snapshot.incomplete_native_exchanges
        );
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
