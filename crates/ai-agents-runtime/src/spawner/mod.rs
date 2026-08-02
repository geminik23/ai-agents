//! Dynamic agent spawning, registry, and inter-agent messaging.

pub mod config;
pub mod registry;
// Keep the repeated module name because renaming it would break the public v1 `spawner::spawner` path.
#[allow(clippy::module_inception)]
pub mod spawner;
pub mod storage;
pub mod tools;

pub use config::{
    auto_configure_spawner, configure_spawner_tools, resolve_templates, spawner_from_config,
};
pub use registry::{AgentRegistry, RegistryHooks, SpawnedAgentInfo};
pub use spawner::{AgentSpawner, ResolvedTemplate, SpawnedAgent};
pub use storage::NamespacedStorage;
pub use tools::{GenerateAgentTool, ListAgentsTool, RemoveAgentTool, SendMessageTool};
