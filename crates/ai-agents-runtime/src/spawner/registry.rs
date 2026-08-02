//! Agent registry for tracking and messaging spawned agents.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::spec::AgentSpec;
use crate::{Agent, RuntimeAgent, TurnActorContext};
use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_observability::{current_observation_context, with_observation_context};

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
        self.register_batch(vec![agent]).await
    }

    /// Register all spawned agents atomically after checking every ID under one lock.
    pub async fn register_batch(&self, batch: Vec<SpawnedAgent>) -> Result<()> {
        let hook_entries = {
            let mut agents = self.agents.write();
            let mut batch_ids = HashSet::with_capacity(batch.len());

            for agent in &batch {
                if !batch_ids.insert(agent.id.clone()) {
                    return Err(AgentError::Config(format!(
                        "Duplicate agent ID in registration batch: {}",
                        agent.id
                    )));
                }
                if agents.contains_key(&agent.id) {
                    return Err(AgentError::Config(format!(
                        "Agent already registered: {}",
                        agent.id
                    )));
                }
            }

            let hook_entries: Vec<(String, AgentSpec)> = batch
                .iter()
                .map(|agent| (agent.id.clone(), agent.spec.clone()))
                .collect();
            for agent in batch {
                agents.insert(agent.id.clone(), Arc::new(agent));
            }
            hook_entries
        };

        for (id, spec) in hook_entries {
            info!(agent_id = %id, "Agent registered in registry");
            if let Some(ref hooks) = self.hooks {
                hooks.on_agent_spawned(&id, &spec).await;
            }
        }
        Ok(())
    }

    pub(crate) async fn reconcile(
        &self,
        target_ids: &HashSet<String>,
        additions: Vec<SpawnedAgent>,
    ) -> Result<()> {
        let (removed, added_hooks) = {
            let mut agents = self.agents.write();
            let mut addition_ids = HashSet::with_capacity(additions.len());
            for addition in &additions {
                if !target_ids.contains(&addition.id) {
                    return Err(AgentError::Config(format!(
                        "Restored agent is absent from target topology: {}",
                        addition.id
                    )));
                }
                if !addition_ids.insert(addition.id.clone()) || agents.contains_key(&addition.id) {
                    return Err(AgentError::Config(format!(
                        "Agent already registered during topology restore: {}",
                        addition.id
                    )));
                }
            }
            for id in target_ids {
                if !agents.contains_key(id) && !addition_ids.contains(id) {
                    return Err(AgentError::Config(format!(
                        "Target topology has no retained or staged agent: {}",
                        id
                    )));
                }
            }

            //
            // Build the complete replacement map before swapping it so validation failures preserve the prior topology.
            //
            let mut next = agents.clone();
            let removed_ids = next
                .keys()
                .filter(|id| !target_ids.contains(*id))
                .cloned()
                .collect::<Vec<_>>();
            let removed = removed_ids
                .into_iter()
                .filter_map(|id| next.remove(&id))
                .collect::<Vec<_>>();
            let added_hooks = additions
                .iter()
                .map(|agent| (agent.id.clone(), agent.spec.clone()))
                .collect::<Vec<_>>();
            for addition in additions {
                next.insert(addition.id.clone(), Arc::new(addition));
            }
            *agents = next;
            (removed, added_hooks)
        };

        for agent in removed {
            agent.release_capacity();
            info!(agent_id = %agent.id, "Agent removed during topology restore");
            if let Some(ref hooks) = self.hooks {
                hooks.on_agent_removed(&agent.id).await;
            }
        }
        for (id, spec) in added_hooks {
            info!(agent_id = %id, "Agent registered during topology restore");
            if let Some(ref hooks) = self.hooks {
                hooks.on_agent_spawned(&id, &spec).await;
            }
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

    /// List all registered agents with their specs serialized as YAML for session persistence.
    pub fn list_with_specs(&self) -> Vec<ai_agents_core::SpawnedAgentEntry> {
        let agents = self.agents.read();
        agents
            .values()
            .filter_map(|sa| {
                let spec_yaml = match serde_yaml::to_string(&sa.spec) {
                    Ok(y) => y,
                    Err(e) => {
                        warn!(agent_id = %sa.id, error = %e, "Failed to serialize agent spec");
                        return None;
                    }
                };
                Some(ai_agents_core::SpawnedAgentEntry {
                    id: sa.id.clone(),
                    name: sa.spec.name.clone(),
                    spec_yaml,
                })
            })
            .collect()
    }

    /// Remove an agent from the registry and return it.
    pub async fn remove(&self, id: &str) -> Option<Arc<SpawnedAgent>> {
        let removed = {
            let mut agents = self.agents.write();
            agents.remove(id)
        };
        if let Some(agent) = removed.as_ref() {
            // Registry removal is the ownership boundary for an active slot. The reservation also releases on drop, so external Arc handles cannot double-decrement it.
            agent.release_capacity();
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
        self.send_inner(from, to, message, None).await
    }

    /// Send a message with structured actor context for actor-scoped memory.
    pub async fn send_with_actor_context(
        &self,
        from: &str,
        to: &str,
        message: &str,
        actor_context: TurnActorContext,
    ) -> Result<AgentResponse> {
        self.send_inner(from, to, message, Some(actor_context))
            .await
    }

    async fn send_inner(
        &self,
        from: &str,
        to: &str,
        message: &str,
        actor_context: Option<TurnActorContext>,
    ) -> Result<AgentResponse> {
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

        debug!(from = %from, to = %to, has_actor_context = actor_context.is_some(), "Sending inter-agent message");
        if let Some(context) = actor_context {
            target.chat_with_actor_context(&formatted, context).await
        } else {
            target.chat(&formatted).await
        }
    }

    /// Broadcast a message to all agents except the sender.
    ///
    /// Clones all target Arcs under a single brief read lock, then drives all `chat()` calls concurrently after releasing the lock.
    pub async fn broadcast(
        &self,
        from: &str,
        message: &str,
    ) -> Vec<(String, Result<AgentResponse>)> {
        self.broadcast_inner(from, message, None).await
    }

    /// Broadcast a message with structured actor context for actor-scoped memory.
    pub async fn broadcast_with_actor_context(
        &self,
        from: &str,
        message: &str,
        actor_context: TurnActorContext,
    ) -> Vec<(String, Result<AgentResponse>)> {
        self.broadcast_inner(from, message, Some(actor_context))
            .await
    }

    async fn broadcast_inner(
        &self,
        from: &str,
        message: &str,
        actor_context: Option<TurnActorContext>,
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
            has_actor_context = actor_context.is_some(),
            "Broadcasting message"
        );

        let mut handles = Vec::with_capacity(targets.len());
        let observation_context = current_observation_context();
        for (id, agent) in targets {
            let msg = formatted.clone();
            let context = actor_context.clone();
            let observation_context = observation_context.clone();
            handles.push(tokio::spawn(async move {
                let run = async move {
                    if let Some(context) = context {
                        agent.chat_with_actor_context(&msg, context).await
                    } else {
                        agent.chat(&msg).await
                    }
                };
                let result = if let Some(context) = observation_context {
                    with_observation_context(context, run).await
                } else {
                    run.await
                };
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
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Mutex, Weak};

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
        SpawnedAgent::untracked(
            id.to_string(),
            agent,
            AgentSpec {
                name: id.to_string(),
                ..AgentSpec::default()
            },
        )
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
    async fn test_register_batch_inserts_all_agents() {
        let registry = AgentRegistry::new();
        registry
            .register_batch(vec![make_spawned("a"), make_spawned("b")])
            .await
            .unwrap();

        assert!(registry.contains("a"));
        assert!(registry.contains("b"));
        assert_eq!(registry.count(), 2);
    }

    #[tokio::test]
    async fn test_register_batch_rejects_duplicate_ids_without_inserting() {
        let registry = AgentRegistry::new();
        let result = registry
            .register_batch(vec![make_spawned("dup"), make_spawned("dup")])
            .await;

        assert!(result.is_err());
        assert!(!registry.contains("dup"));
        assert_eq!(registry.count(), 0);
    }

    #[tokio::test]
    async fn test_register_batch_rejects_existing_collision_without_inserting() {
        let registry = AgentRegistry::new();
        registry.register(make_spawned("existing")).await.unwrap();

        let result = registry
            .register_batch(vec![make_spawned("new"), make_spawned("existing")])
            .await;

        assert!(result.is_err());
        assert!(!registry.contains("new"));
        assert_eq!(registry.count(), 1);
    }

    #[tokio::test]
    async fn reconcile_commits_additions_retained_agents_and_removals_together() {
        let registry = AgentRegistry::new();
        registry
            .register_batch(vec![make_spawned("a"), make_spawned("b")])
            .await
            .unwrap();
        let retained = registry.get("b").unwrap();

        registry
            .reconcile(
                &HashSet::from(["b".to_string(), "c".to_string()]),
                vec![make_spawned("c")],
            )
            .await
            .unwrap();

        assert!(!registry.contains("a"));
        assert!(Arc::ptr_eq(&registry.get("b").unwrap(), &retained));
        assert!(registry.contains("c"));
        assert_eq!(registry.count(), 2);

        registry
            .reconcile(&HashSet::new(), Vec::new())
            .await
            .unwrap();
        assert_eq!(registry.count(), 0);
    }

    #[tokio::test]
    async fn reconcile_validation_failure_preserves_prior_topology() {
        let registry = AgentRegistry::new();
        registry
            .register_batch(vec![make_spawned("a"), make_spawned("b")])
            .await
            .unwrap();
        let before_a = registry.get("a").unwrap();
        let before_b = registry.get("b").unwrap();

        let result = registry
            .reconcile(
                &HashSet::from(["a".to_string(), "missing".to_string()]),
                Vec::new(),
            )
            .await;

        assert!(result.is_err());
        assert!(Arc::ptr_eq(&registry.get("a").unwrap(), &before_a));
        assert!(Arc::ptr_eq(&registry.get("b").unwrap(), &before_b));
        assert_eq!(registry.count(), 2);
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
    async fn test_send_agent_message() {
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
    async fn test_batch_hooks_run_only_after_successful_commit() {
        struct CommitObservingHooks {
            registry: Mutex<Option<Weak<AgentRegistry>>>,
            spawned: AtomicU32,
            observed_full_batch: AtomicBool,
        }

        #[async_trait]
        impl RegistryHooks for CommitObservingHooks {
            async fn on_agent_spawned(&self, _id: &str, _spec: &AgentSpec) {
                self.spawned.fetch_add(1, Ordering::Relaxed);
                let registry = self
                    .registry
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap();
                if registry.contains("a") && registry.contains("b") {
                    self.observed_full_batch.store(true, Ordering::Relaxed);
                }
            }
        }

        let hooks = Arc::new(CommitObservingHooks {
            registry: Mutex::new(None),
            spawned: AtomicU32::new(0),
            observed_full_batch: AtomicBool::new(false),
        });
        let registry = Arc::new(AgentRegistry::new().with_hooks(hooks.clone()));
        *hooks.registry.lock().unwrap() = Some(Arc::downgrade(&registry));

        assert!(
            registry
                .register_batch(vec![make_spawned("dup"), make_spawned("dup")])
                .await
                .is_err()
        );
        assert_eq!(hooks.spawned.load(Ordering::Relaxed), 0);

        registry
            .register_batch(vec![make_spawned("a"), make_spawned("b")])
            .await
            .unwrap();
        assert_eq!(registry.count(), 2);
        assert_eq!(hooks.spawned.load(Ordering::Relaxed), 2);
        assert!(hooks.observed_full_batch.load(Ordering::Relaxed));
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
