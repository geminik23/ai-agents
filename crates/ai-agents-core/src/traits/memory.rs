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

/// Core memory trait for storing conversation history
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

    async fn evict_oldest(&self, _count: usize) -> Result<Vec<ChatMessage>> {
        Ok(vec![])
    }
}
