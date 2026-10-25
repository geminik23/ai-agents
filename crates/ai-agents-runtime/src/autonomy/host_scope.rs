//! Declarative host namespaces are resolved without constructing providers or executable restoration graphs.

use super::{AutomaticTaskOptions, ScopedTaskRunStore};
use crate::spec::{AgentSpec, StorageConfig};
use ai_agents_core::{AgentError, Result};
use ai_agents_skills::{SkillDefinition, SkillLoader};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};

/// An explicit YAML-host namespace shared by execution and store-only management.
/// This opt-in identity covers the normalized declaration and loaded skills, not opaque implementation binding or caller authentication.
/// Hosts must prepare it from their authorized configuration, never from an untrusted checkpoint; existing default runtime identities are unchanged.
#[derive(Clone)]
pub struct TaskHostScope {
    agent_id: String,
    actor_id: Option<String>,
    identity: String,
    storage: StorageConfig,
}

impl TaskHostScope {
    /// Resolves external skills using the same loader/base directory as the builder, without providers, environment credentials or startup work.
    pub fn from_spec(spec: &AgentSpec, base_dir: &Path, actor_id: Option<String>) -> Result<Self> {
        spec.validate()?;
        if actor_id.as_ref().is_some_and(|id| id.trim().is_empty()) {
            return Err(AgentError::Config("task actor must not be empty".into()));
        }
        let skills = SkillLoader::default()
            .with_base_dir(base_dir)
            .load_refs(&spec.skills)?;
        super::validate_loaded_skills(&spec.autonomy, &skills)?;
        Ok(Self {
            agent_id: spec.name.clone(),
            actor_id,
            identity: declaration_identity(spec, &skills)?,
            storage: spec.storage.clone(),
        })
    }

    // The execution builder compares its once-loaded definitions, not a second filesystem sample which could hide a preparation race.
    pub(crate) fn validate_prepared(
        &self,
        spec: &AgentSpec,
        skills: &[SkillDefinition],
    ) -> Result<String> {
        let identity = declaration_identity(spec, skills)?;
        if identity != self.identity || spec.name != self.agent_id {
            return Err(AgentError::Config(
                "task host declaration changed during preparation".into(),
            ));
        }
        Ok(identity)
    }

    /// Supplies only the explicit namespace; hosts with external definitions should prefer the builder's checked task_host_scope binding.
    pub fn task_options(&self) -> AutomaticTaskOptions {
        AutomaticTaskOptions {
            config_identity: Some(self.identity.clone()),
            ..Default::default()
        }
    }

    /// Opens only an existing configured SQLite task journal; snapshot-only stores do not gain task capability.
    /// This does not construct providers, claim owners, reconstruct a runtime or consume pending decisions.
    pub async fn open_store(&self) -> Result<Arc<ScopedTaskRunStore>> {
        let StorageConfig::Sqlite(config) = &self.storage else {
            return Err(AgentError::Config("task management requires an existing SQLite task journal; volatile tasks remain in their original process".into()));
        };
        if !Path::new(&config.path).is_file() {
            return Err(AgentError::Config(
                "configured task journal does not exist".into(),
            ));
        }
        let storage = ai_agents_storage::create_storage(&crate::spec::storage::to_storage_config(
            &self.storage,
        ))
        .await?
        .ok_or_else(|| AgentError::Config("task storage is unavailable".into()))?;
        Ok(Arc::new(ScopedTaskRunStore::new(
            storage,
            self.agent_id.clone(),
            self.actor_id.clone(),
            self.identity.clone(),
        )?))
    }
}

// Value uses deterministic object ordering; raw YAML formatting and HashMap iteration do not change the declared namespace.
fn declaration_identity(spec: &AgentSpec, skills: &[SkillDefinition]) -> Result<String> {
    let value = serde_json::json!({"contract":1,"spec":spec,"loaded_skills":skills});
    Ok(format!(
        "yaml-host.v1:{:x}",
        Sha256::digest(serde_json::to_vec(&value)?)
    ))
}
