//! Memory system for storing conversation history

mod in_memory;

use async_trait::async_trait;
pub use in_memory::InMemoryStore;
use std::sync::Arc;

use crate::error::Result;
use crate::llm::ChatMessage;
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
}

pub fn create_memory(memory_type: &str, max_messages: usize) -> Result<Arc<dyn Memory>> {
    match memory_type {
        "in-memory" => Ok(Arc::new(InMemoryStore::new(max_messages))),
        _ => Ok(Arc::new(InMemoryStore::new(max_messages))),
    }
}

pub fn create_memory_from_config(config: &MemoryConfig) -> Result<Arc<dyn Memory>> {
    create_memory(&config.memory_type, config.max_messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_memory_in_memory() {
        let memory = create_memory("in-memory", 100).unwrap();
        assert!(memory.is_empty());
    }

    #[test]
    fn test_create_memory_unknown_defaults_to_in_memory() {
        let memory = create_memory("unknown", 50).unwrap();
        assert!(memory.is_empty());
    }

    #[test]
    fn test_create_memory_from_config() {
        let config = MemoryConfig::default();
        let memory = create_memory_from_config(&config).unwrap();
        assert!(memory.is_empty());
    }
}
