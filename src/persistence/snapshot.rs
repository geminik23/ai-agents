use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::llm::ChatMessage;
use crate::state::StateMachineSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSnapshot {
    // FIXME: change later with SCHEMA_VERSION (future feature)
    pub version: String,
    pub agent_id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub state_machine: Option<StateMachineSnapshot>,
    pub memory: MemorySnapshot,
    #[serde(default)]
    pub context: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemorySnapshot {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub summary: Option<String>,
}

impl AgentSnapshot {
    pub fn new(agent_id: String) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            agent_id,
            timestamp: Utc::now(),
            state_machine: None,
            memory: MemorySnapshot::default(),
            context: HashMap::new(),
        }
    }

    pub fn with_state_machine(mut self, snapshot: StateMachineSnapshot) -> Self {
        self.state_machine = Some(snapshot);
        self
    }

    pub fn with_memory(mut self, snapshot: MemorySnapshot) -> Self {
        self.memory = snapshot;
        self
    }

    pub fn with_context(mut self, context: HashMap<String, serde_json::Value>) -> Self {
        self.context = context;
        self
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;

    #[test]
    fn test_agent_snapshot_new() {
        let snapshot = AgentSnapshot::new("agent-123".into());
        assert_eq!(snapshot.agent_id, "agent-123");
        assert!(snapshot.state_machine.is_none());
        assert!(snapshot.memory.messages.is_empty());
        assert!(snapshot.context.is_empty());
    }

    #[test]
    fn test_memory_snapshot() {
        let messages = vec![
            ChatMessage {
                role: Role::User,
                content: "Hello".into(),
                name: None,
                timestamp: None,
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Hi!".into(),
                name: None,
                timestamp: None,
            },
        ];
        let snapshot = MemorySnapshot::new(messages.clone());
        assert_eq!(snapshot.messages.len(), 2);
        assert!(snapshot.summary.is_none());
    }

    #[test]
    fn test_snapshot_serialization() {
        let mut context = HashMap::new();
        context.insert("user".into(), serde_json::json!({"name": "Alice"}));

        let snapshot = AgentSnapshot::new("test-agent".into()).with_context(context);

        let json = serde_json::to_string(&snapshot).unwrap();
        let restored: AgentSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.agent_id, "test-agent");
        assert!(restored.context.contains_key("user"));
    }
}
