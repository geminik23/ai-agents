//! Agent registry for tracking and messaging spawned agents.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::spec::AgentSpec;
use crate::{Agent, RuntimeAgent};
use ai_agents_core::{AgentError, AgentResponse, Result};

use super::spawner::SpawnedAgent;

/// Summary information for a registered agent, returned by `list()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnedAgentInfo {
    pub id: String,
    pub name: String,
    pub spawned_at: DateTime<Utc>,
}

/// Tracks spawned agents and provides inter-agent messaging.
pub struct AgentRegistry {
    agents: RwLock<HashMap<String, Arc<SpawnedAgent>>>,
    hooks: Option<Arc<dyn RegistryHooks>>,
    /// When true, `send()` prefixes messages with `[From {sender}]: `.
    send_with_context: bool,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            hooks: None,
            send_with_context: true,
        }
    }

    /// Attach lifecycle hooks to the registry.
    pub fn with_hooks(mut self, hooks: Arc<dyn RegistryHooks>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Configure whether `send()` injects sender identity into messages.
    pub fn with_send_context(mut self, enabled: bool) -> Self {
        self.send_with_context = enabled;
        self
    }

    /// Register a spawned agent. Returns error if the ID already exists.
    pub async fn register(&self, agent: SpawnedAgent) -> Result<()> {
        let id = agent.id.clone();
        let spec_clone = agent.spec.clone();
        {
            let mut agents = self.agents.write();
            if agents.contains_key(&id) {
                return Err(AgentError::Config(format!(
                    "Agent already registered: {}",
                    id
                )));
            }
            agents.insert(id.clone(), Arc::new(agent));
        }
        info!(agent_id = %id, "Agent registered in registry");
        if let Some(ref hooks) = self.hooks {
            hooks.on_agent_spawned(&id, &spec_clone).await;
        }
        Ok(())
    }

    /// Clone an Arc handle to a registered agent's RuntimeAgent.
    pub fn get(&self, id: &str) -> Option<Arc<RuntimeAgent>> {
        let agents = self.agents.read();
        agents.get(id).map(|sa| Arc::clone(&sa.agent))
    }

    /// Get the full SpawnedAgent metadata (agent + spec + timestamp).
    pub fn get_spawned(&self, id: &str) -> Option<Arc<SpawnedAgent>> {
        let agents = self.agents.read();
        agents.get(id).cloned()
    }

    /// List metadata for all registered agents.
    pub fn list(&self) -> Vec<SpawnedAgentInfo> {
        let agents = self.agents.read();
        agents
            .values()
            .map(|sa| SpawnedAgentInfo {
                id: sa.id.clone(),
                name: sa.spec.name.clone(),
                spawned_at: sa.spawned_at,
            })
            .collect()
    }

    /// Remove an agent from the registry and return it.
    pub async fn remove(&self, id: &str) -> Option<Arc<SpawnedAgent>> {
        let removed = {
            let mut agents = self.agents.write();
            agents.remove(id)
        };
        if removed.is_some() {
            info!(agent_id = %id, "Agent removed from registry");
            if let Some(ref hooks) = self.hooks {
                hooks.on_agent_removed(id).await;
            }
        } else {
            debug!(agent_id = %id, "Attempted to remove non-existent agent");
        }
        removed
    }

    /// Send a message from one agent to another and return the response.
    pub async fn send(&self, from: &str, to: &str, message: &str) -> Result<AgentResponse> {
        let target = {
            // The read lock is held only long enough to clone the target Arc, then released before the async `chat()` call.
            let agents = self.agents.read();
            agents.get(to).map(|sa| Arc::clone(&sa.agent))
        };
        let target =
            target.ok_or_else(|| AgentError::Other(format!("Target agent not found: {}", to)))?;

        if let Some(ref hooks) = self.hooks {
            hooks.on_message_sent(from, to, message).await;
        }

        let formatted = if self.send_with_context {
            format!("[From {}]: {}", from, message)
        } else {
            message.to_string()
        };

        debug!(from = %from, to = %to, "Sending inter-agent message");
        target.chat(&formatted).await
    }

    /// Broadcast a message to all agents except the sender.
    ///
    /// Clones all target Arcs under a single brief read lock, then drives all `chat()` calls concurrently after releasing the lock.
    pub async fn broadcast(
        &self,
        from: &str,
        message: &str,
    ) -> Vec<(String, Result<AgentResponse>)> {
        let targets: Vec<(String, Arc<RuntimeAgent>)> = {
            let agents = self.agents.read();
            agents
                .iter()
                .filter(|(id, _)| id.as_str() != from)
                .map(|(id, sa)| (id.clone(), Arc::clone(&sa.agent)))
                .collect()
        };

        if targets.is_empty() {
            return Vec::new();
        }

        let formatted = if self.send_with_context {
            format!("[From {}]: {}", from, message)
        } else {
            message.to_string()
        };

        debug!(
            from = %from,
            target_count = targets.len(),
            "Broadcasting message"
        );

        let mut handles = Vec::with_capacity(targets.len());
        for (id, agent) in targets {
            let msg = formatted.clone();
            handles.push(tokio::spawn(async move {
                let result = agent.chat(&msg).await;
                (id, result)
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((id, res)) => results.push((id, res)),
                Err(e) => {
                    warn!(error = %e, "Broadcast task panicked");
                }
            }
        }
        results
    }

    /// Number of currently registered agents.
    pub fn count(&self) -> usize {
        self.agents.read().len()
    }

    /// Returns true if the registry contains an agent with this ID.
    pub fn contains(&self, id: &str) -> bool {
        self.agents.read().contains_key(id)
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Debug impl avoids printing agent internals.
impl std::fmt::Debug for AgentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.agents.read().len();
        f.debug_struct("AgentRegistry")
            .field("agent_count", &count)
            .field("send_with_context", &self.send_with_context)
            .field("has_hooks", &self.hooks.is_some())
            .finish()
    }
}

/// Optional lifecycle hooks for registry events.
#[async_trait]
pub trait RegistryHooks: Send + Sync {
    /// Called after an agent is successfully registered.
    async fn on_agent_spawned(&self, _id: &str, _spec: &AgentSpec) {}

    /// Called after an agent is removed from the registry.
    async fn on_agent_removed(&self, _id: &str) {}

    /// Called before a message is delivered via `send()`.
    async fn on_message_sent(&self, _from: &str, _to: &str, _message: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentBuilder;
    use ai_agents_core::{
        ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider,
        LLMResponse,
    };
    use ai_agents_llm::LLMRegistry;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct EchoProvider;

    #[async_trait]
    impl LLMProvider for EchoProvider {
        async fn complete(
            &self,
            messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<LLMResponse, LLMError> {
            let last = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(LLMResponse::new(
                format!("Echo: {}", last),
                FinishReason::Stop,
            ))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<
            Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
            LLMError,
        > {
            Err(LLMError::Other("not implemented".into()))
        }

        fn provider_name(&self) -> &str {
            "echo"
        }

        fn supports(&self, _feature: LLMFeature) -> bool {
            false
        }
    }

    fn make_test_agent(name: &str) -> RuntimeAgent {
        let mut registry = LLMRegistry::new();
        registry.register("default", Arc::new(EchoProvider));

        AgentBuilder::new()
            .system_prompt(format!("You are {}.", name))
            .llm_registry(registry)
            .build()
            .unwrap()
    }

    fn make_spawned(id: &str) -> SpawnedAgent {
        let agent = make_test_agent(id);
        SpawnedAgent {
            id: id.to_string(),
            agent: Arc::new(agent),
            spec: AgentSpec {
                name: id.to_string(),
                ..AgentSpec::default()
            },
            spawned_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = AgentRegistry::new();
        registry.register(make_spawned("agent_a")).await.unwrap();

        assert!(registry.get("agent_a").is_some());
        assert!(registry.get("agent_b").is_none());
        assert_eq!(registry.count(), 1);
    }

    #[tokio::test]
    async fn test_duplicate_register() {
        let registry = AgentRegistry::new();
        registry.register(make_spawned("dup")).await.unwrap();
        let result = registry.register(make_spawned("dup")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_and_remove() {
        let registry = AgentRegistry::new();
        registry.register(make_spawned("a")).await.unwrap();
        registry.register(make_spawned("b")).await.unwrap();

        assert_eq!(registry.list().len(), 2);

        let removed = registry.remove("a").await;
        assert!(removed.is_some());
        assert_eq!(registry.count(), 1);
        assert!(registry.get("a").is_none());
    }

    #[tokio::test]
    async fn test_send_message() {
        let registry = AgentRegistry::new();
        registry.register(make_spawned("sender")).await.unwrap();
        registry.register(make_spawned("receiver")).await.unwrap();

        let response = registry.send("sender", "receiver", "hello").await.unwrap();
        assert!(response.content.contains("hello"));
    }

    #[tokio::test]
    async fn test_send_to_missing() {
        let registry = AgentRegistry::new();
        registry.register(make_spawned("sender")).await.unwrap();

        let result = registry.send("sender", "nobody", "hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_broadcast() {
        let registry = AgentRegistry::new();
        registry
            .register(make_spawned("broadcaster"))
            .await
            .unwrap();
        registry.register(make_spawned("listener_1")).await.unwrap();
        registry.register(make_spawned("listener_2")).await.unwrap();

        let results = registry.broadcast("broadcaster", "hey everyone").await;
        // Should have 2 results (excluding broadcaster)
        assert_eq!(results.len(), 2);
        for (_, res) in &results {
            assert!(res.is_ok());
        }
    }

    #[tokio::test]
    async fn test_hooks() {
        struct CountingHooks {
            spawned: AtomicU32,
            removed: AtomicU32,
            sent: AtomicU32,
        }

        #[async_trait]
        impl RegistryHooks for CountingHooks {
            async fn on_agent_spawned(&self, _id: &str, _spec: &AgentSpec) {
                self.spawned.fetch_add(1, Ordering::Relaxed);
            }
            async fn on_agent_removed(&self, _id: &str) {
                self.removed.fetch_add(1, Ordering::Relaxed);
            }
            async fn on_message_sent(&self, _from: &str, _to: &str, _msg: &str) {
                self.sent.fetch_add(1, Ordering::Relaxed);
            }
        }

        let hooks = Arc::new(CountingHooks {
            spawned: AtomicU32::new(0),
            removed: AtomicU32::new(0),
            sent: AtomicU32::new(0),
        });

        let registry = AgentRegistry::new().with_hooks(hooks.clone());
        registry.register(make_spawned("h1")).await.unwrap();
        registry.register(make_spawned("h2")).await.unwrap();
        assert_eq!(hooks.spawned.load(Ordering::Relaxed), 2);

        registry.send("h1", "h2", "ping").await.unwrap();
        assert_eq!(hooks.sent.load(Ordering::Relaxed), 1);

        registry.remove("h1").await;
        assert_eq!(hooks.removed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_contains() {
        let registry = AgentRegistry::new();
        assert!(!registry.contains("x"));
        registry.register(make_spawned("x")).await.unwrap();
        assert!(registry.contains("x"));
    }

    #[tokio::test]
    async fn test_send_without_context() {
        let registry = AgentRegistry::new().with_send_context(false);
        registry.register(make_spawned("a")).await.unwrap();
        registry.register(make_spawned("b")).await.unwrap();

        let response = registry.send("a", "b", "raw msg").await.unwrap();
        // Without context prefix, the message should be passed as-is
        assert!(response.content.contains("raw msg"));
        assert!(!response.content.contains("[From"));
    }
}
