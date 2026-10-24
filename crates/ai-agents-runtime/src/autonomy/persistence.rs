//! Persistence policy uses the existing conditional journal and prepared restoration, not a second snapshot or startup executor.

use super::*;

/// Startup selection carries current host configuration, never a guessed latest run or a serialized execution permit.
pub struct TaskStartupRequest {
    pub config: AutonomyConfig,
    pub ceilings: AutonomyHostCeilings,
    pub config_identity: String,
    pub profile: Option<String>,
}

// Omission preserves existing explicit execution; a declared persistence section opts in unless explicitly disabled.
fn enabled(settings: &AutonomyProfile) -> bool {
    settings
        .persistence
        .as_ref()
        .is_some_and(|p| p.enabled != Some(false))
}

impl AutonomyRunner {
    /// Installs the trusted constructor for automatic safe-pause preparation; ordinary chat and task entry remain explicit.
    pub fn with_restore_host(mut self, host: Arc<dyn TaskRestoreHost>) -> Self {
        self.restore_host = Some(host);
        self
    }

    // Invalid persistence requests fail before reserving runtime ownership or creating any task journal.
    pub(super) fn preflight_persistence(&self, settings: &AutonomyProfile) -> Result<()> {
        if let Some(policy) = &settings.persistence {
            if !enabled(settings)
                && (policy.save_after_each_step == Some(true)
                    || policy.resume_on_start == Some(true))
            {
                return Err(AgentError::Config(
                    "disabled task persistence conflicts with enabled persistence controls".into(),
                ));
            }
            if enabled(settings)
                && (self.restore_host.is_none() || !self.store.supports_reconstruction())
            {
                return Err(AgentError::Config("task persistence requires an installed restore host and reconstruction-capable store".into()));
            }
        }
        Ok(())
    }

    // A returned persisted pause is a complete prepared restart point, not just a DTO that happens to have pending IDs.
    // Preparation failure returns an error and retains its actual claim phase rather than reporting a durable pause or retrying.
    pub(super) async fn persist_returned_pause(
        &self,
        result: TaskRunResult,
    ) -> Result<TaskRunResult> {
        if result.run.status != TaskRunStatus::Paused {
            return Ok(result);
        }
        // The admitted policy is already in the exact acknowledged execution; omitted/disabled persistence adds no failure-capable read.
        let saved = self
            .active
            .lock()
            .as_ref()
            .filter(|execution| execution.run_id == result.run.key.run_id)
            .map(|execution| execution.acknowledged_snapshot())
            .ok_or(TaskRunStorageError::NotResumable)?;
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload.clone())?;
        if !enabled(&payload.settings) {
            return Ok(result);
        }
        let host = self
            .restore_host
            .as_ref()
            .ok_or(TaskRunStorageError::NotResumable)?;
        Box::pin(self.checkpoint_for_restore(&saved.key.run_id, result.run.revision, host.as_ref()))
            .await
    }

    // Optional per-step snapshots are separate from mandatory effect/continuation journals, which are never disabled by policy.
    // This refreshes the full current runtime image only after children and owned maintenance have settled.
    pub(super) async fn checkpoint_persistence_step(
        &self,
        execution: &RunExecution,
        settings: &AutonomyProfile,
        cycle: u64,
    ) -> Result<()> {
        if !enabled(settings)
            || settings
                .persistence
                .as_ref()
                .is_some_and(|p| p.save_after_each_step == Some(false))
        {
            return Ok(());
        }
        let runtime = TaskRuntimeCheckpoint::between_turns(self.agent.save_state_full().await?)?;
        execution
            .update(|payload| {
                payload.runtime = runtime;
                let marker = TaskAdapterCheckpoint {
                    id: "runtime.persistence_step".into(),
                    adapter: "runtime.persistence_step".into(),
                    contract_version: 1,
                    config: json!({"contract":"settled_step_snapshot_v1"}),
                    state: json!({"cycle":cycle}),
                };
                if let Some(old) = payload.adapters.iter_mut().find(|a| a.id == marker.id) {
                    *old = marker;
                } else {
                    payload.adapters.push(marker);
                }
                Ok(())
            })
            .await?;
        Ok(())
    }

    /// Selects only one compatible unowned prepared pause in the supplied store's fixed agent/actor scope.
    /// Zero candidates returns None, ambiguity is an error, and no approval, question, external signal or abandoned owner is consumed.
    /// Memory remains process-local; durable process restart requires a retained compatible backend such as SQLite.
    pub async fn restore_on_start(
        request: TaskStartupRequest,
        store: Arc<dyn TaskRunStore>,
        host: Arc<dyn TaskRestoreHost>,
    ) -> Result<Option<(Self, TaskRunResult)>> {
        let mut selected = None;
        for summary in store.list().await? {
            if summary.status != TaskRunStatus::Paused || summary.owned || summary.cancel_requested
            {
                continue;
            }
            let snapshot = store
                .inspect_candidate(&summary.key.run_id)
                .await?
                .ok_or(TaskRunStorageError::Conflict)?;
            if snapshot.revision != summary.revision {
                return Err(TaskRunStorageError::Conflict.into());
            }
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
            if payload.config_identity != request.config_identity
                || request
                    .profile
                    .as_ref()
                    .is_some_and(|profile| payload.profile.as_ref() != Some(profile))
            {
                continue;
            }
            if !enabled(&payload.settings)
                || payload
                    .settings
                    .persistence
                    .as_ref()
                    .is_none_or(|p| p.resume_on_start != Some(true))
            {
                continue;
            }
            let candidate = TaskRestoreRequest {
                run_id: snapshot.key.run_id.clone(),
                expected_revision: snapshot.revision,
                config_identity: request.config_identity.clone(),
                config: request.config.clone(),
                ceilings: request.ceilings.clone(),
            };
            if !restoration::validate_startup_candidate(
                &candidate,
                &snapshot,
                &payload,
                host.as_ref(),
            )? {
                continue;
            }
            if selected.replace(candidate).is_some() {
                return Err(AgentError::Config(
                    "multiple compatible task checkpoints require an explicit run selection".into(),
                ));
            }
        }
        let Some(candidate) = selected else {
            return Ok(None);
        };
        Self::restore_paused(candidate, store, host).await.map(Some)
    }
}
