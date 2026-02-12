use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;

use ai_agents_core::{ChatMessage, MemorySnapshot, Result};

use super::Memory;

pub struct InMemoryStore {
    messages: Arc<RwLock<Vec<ChatMessage>>>,
    max_messages: usize,
}

impl InMemoryStore {
    pub fn new(max_messages: usize) -> Self {
        Self {
            messages: Arc::new(RwLock::new(Vec::new())),
            max_messages,
        }
    }

    pub fn max_messages(&self) -> usize {
        self.max_messages
    }
}

impl Clone for InMemoryStore {
    fn clone(&self) -> Self {
        Self {
            messages: Arc::clone(&self.messages),
            max_messages: self.max_messages,
        }
    }
}

#[async_trait]
impl ai_agents_core::Memory for InMemoryStore {
    async fn add_message(&self, message: ChatMessage) -> Result<()> {
        let mut messages = self.messages.write();
        messages.push(message);

        while messages.len() > self.max_messages {
            messages.remove(0);
        }

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
        self.messages.write().clear();
        Ok(())
    }

    fn len(&self) -> usize {
        self.messages.read().len()
    }

    async fn restore(&self, snapshot: MemorySnapshot) -> Result<()> {
        let mut messages = self.messages.write();
        *messages = snapshot.messages;
        while messages.len() > self.max_messages {
            messages.remove(0);
        }
        Ok(())
    }

    async fn evict_oldest(&self, count: usize) -> Result<Vec<ChatMessage>> {
        let mut messages = self.messages.write();
        let evict_count = count.min(messages.len());
        let evicted: Vec<ChatMessage> = messages.drain(..evict_count).collect();
        Ok(evicted)
    }
}

#[async_trait]
impl Memory for InMemoryStore {}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::{Memory as CoreMemory, Role};

    fn make_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: content.to_string(),
            name: None,
            timestamp: None,
        }
    }

    #[tokio::test]
    async fn test_add_and_get_messages() {
        let store = InMemoryStore::new(10);

        store.add_message(make_message("hello")).await.unwrap();
        store.add_message(make_message("world")).await.unwrap();

        let messages = store.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].content, "world");
    }

    #[tokio::test]
    async fn test_max_messages_limit() {
        let store = InMemoryStore::new(3);

        for i in 0..5 {
            store
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let messages = store.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "msg2");
        assert_eq!(messages[1].content, "msg3");
        assert_eq!(messages[2].content, "msg4");
    }

    #[tokio::test]
    async fn test_get_messages_with_limit() {
        let store = InMemoryStore::new(10);

        for i in 0..5 {
            store
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let messages = store.get_messages(Some(2)).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "msg3");
        assert_eq!(messages[1].content, "msg4");
    }

    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryStore::new(10);

        store.add_message(make_message("test")).await.unwrap();
        assert!(!store.is_empty());

        store.clear().await.unwrap();
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn test_clone_shares_state() {
        let store1 = InMemoryStore::new(10);
        let store2 = store1.clone();

        store1
            .add_message(make_message("from store1"))
            .await
            .unwrap();

        let messages = store2.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "from store1");
    }

    #[tokio::test]
    async fn test_snapshot_restore() {
        let store = InMemoryStore::new(10);
        store.add_message(make_message("msg1")).await.unwrap();
        store.add_message(make_message("msg2")).await.unwrap();

        let snapshot = store.snapshot().await.unwrap();
        assert_eq!(snapshot.messages.len(), 2);

        store.clear().await.unwrap();
        assert!(store.is_empty());

        store.restore(snapshot).await.unwrap();
        let messages = store.get_messages(None).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "msg1");
    }

    #[tokio::test]
    async fn test_evict_oldest() {
        let store = InMemoryStore::new(10);
        for i in 0..5 {
            store
                .add_message(make_message(&format!("msg{}", i)))
                .await
                .unwrap();
        }

        let evicted = store.evict_oldest(2).await.unwrap();
        assert_eq!(evicted.len(), 2);
        assert_eq!(evicted[0].content, "msg0");
        assert_eq!(evicted[1].content, "msg1");

        let remaining = store.get_messages(None).await.unwrap();
        assert_eq!(remaining.len(), 3);
        assert_eq!(remaining[0].content, "msg2");
    }
}
