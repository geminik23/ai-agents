use serde::{Deserialize, Serialize};

/// Preset reset strategies for eval reset steps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResetProfile {
    Conversation,
    Session,
    FullRuntime,
    EvalAttempt,
    Persistence,
}

/// Options controlling an eval reset step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResetOptions {
    /// Reset profile applied by the reset step.
    #[serde(default = "default_profile")]
    pub profile: ResetProfile,
    /// Reapply fixture and scenario context after reset.
    #[serde(default)]
    pub preserve_host_context: bool,
    /// Preserve the active actor ID after reset.
    #[serde(default)]
    pub preserve_actor_id: bool,
    /// Keep the temp storage path across rebuilds.
    #[serde(default = "default_true")]
    pub preserve_storage: bool,
    /// Whether spawner registry state should be rebuilt.
    #[serde(default = "default_true")]
    pub reset_spawner_registry: bool,
    /// Whether observability state should be recreated.
    #[serde(default = "default_true")]
    pub reset_observability: bool,
    /// Delete temp persistence before rebuilding.
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
