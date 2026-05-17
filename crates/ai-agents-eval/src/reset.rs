use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResetProfile {
    Conversation,
    Session,
    FullRuntime,
    EvalAttempt,
    Persistence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetOptions {
    #[serde(default = "default_profile")]
    pub profile: ResetProfile,
    #[serde(default)]
    pub preserve_host_context: bool,
    #[serde(default)]
    pub preserve_actor_id: bool,
    #[serde(default = "default_true")]
    pub preserve_storage: bool,
    #[serde(default = "default_true")]
    pub reset_spawner_registry: bool,
    #[serde(default = "default_true")]
    pub reset_observability: bool,
    #[serde(default)]
    pub delete_persistence: bool,
}

impl Default for ResetOptions {
    fn default() -> Self {
        Self {
            profile: ResetProfile::FullRuntime,
            preserve_host_context: false,
            preserve_actor_id: false,
            preserve_storage: true,
            reset_spawner_registry: true,
            reset_observability: true,
            delete_persistence: false,
        }
    }
}

fn default_profile() -> ResetProfile {
    ResetProfile::FullRuntime
}

fn default_true() -> bool {
    true
}
