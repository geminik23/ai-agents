//! Authenticated Rust host stop reuses exact safe cleanup without treating an answer as control authority.

use super::*;
use std::future::Future;

#[derive(Clone)]
pub(super) struct HostStop {
    reason: String,
    kind: &'static str,
    status: TaskRunStatus,
    request_id: Option<String>,
}

tokio::task_local! { static HOST_STOP: HostStop; }

/// Only the explicit Rust host API installs a stop reason; external signals and tool responses cannot deserialize it.
async fn scope_stop<F: Future>(control: HostStop, future: F) -> F::Output {
    HOST_STOP.scope(control, future).await
}

/// Cancellation helpers retain their original semantics unless the host explicitly owns a stop operation.
pub(super) fn stop_reason() -> Option<HostStop> {
    HOST_STOP.try_with(Clone::clone).ok()
}

impl AutonomyRunner {
    /// Stops a known-safe retained pause without invoking its pending operation; the host authenticates its caller.
    /// A conditional claim excludes competing responses, and protection is released only after terminal acknowledgement.
    pub async fn stop_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
        reason: &str,
    ) -> ai_agents_core::Result<TaskRunResult> {
        if reason.trim().is_empty() || reason.chars().count() > 512 {
            return Err(ai_agents_core::AgentError::Config(
                "host stop requires a bounded audit reason".into(),
            ));
        }
        scope_stop(
            HostStop {
                reason: reason.into(),
                kind: "stop",
                status: TaskRunStatus::Incomplete,
                request_id: None,
            },
            self.cancel_paused(run_id, expected_revision),
        )
        .await
    }

    /// Reports host-confirmed delivery loss for one retained external wait, never inferred from silence or signal payloads.
    /// Frozen stop/fail policy uses existing safe cleanup and CAS; stale freshness cannot prevent known-safe retirement.
    pub async fn external_unavailable_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
        request_id: &str,
        reason: &str,
    ) -> ai_agents_core::Result<TaskRunResult> {
        if reason.trim().is_empty() || reason.chars().count() > 512 {
            return Err(ai_agents_core::AgentError::Config(
                "external unavailability requires a bounded audit reason".into(),
            ));
        }
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;
        let state = self.retained_controller(&snapshot, &payload)?;
        let wait = state
            .driver
            .as_ref()
            .and_then(|driver| driver.wait.as_ref())
            .filter(|wait| wait.request_id == request_id)
            .ok_or(TaskRunStorageError::NotResumable)?;
        if self.agent.autonomy_extensions().condition(&wait.binding.id) != Some(&wait.binding) {
            return Err(TaskRunStorageError::NotResumable.into());
        }
        let status = match self
            .agent
            .autonomy_extensions()
            .condition_unavailable_policy(&wait.binding.id)
        {
            ExternalUnavailablePolicy::Stop => TaskRunStatus::Incomplete,
            ExternalUnavailablePolicy::Fail => TaskRunStatus::Failed,
        };
        scope_stop(
            HostStop {
                reason: reason.into(),
                kind: "external_unavailable",
                status,
                request_id: Some(request_id.into()),
            },
            self.cancel_paused(run_id, expected_revision),
        )
        .await
    }

    /// Revises numeric ceilings without consuming the pending response or executing work; the host authenticates its caller.
    /// Failed or lost acknowledgement preserves protection, while a successful revision keeps individual operation deadlines unchanged.
    pub async fn revise_limits_paused(
        &self,
        run_id: &str,
        expected_revision: u64,
        limits: TaskRunLimits,
        reason: &str,
    ) -> ai_agents_core::Result<TaskRunResult> {
        let _resume = self.resume_gate.lock().await;
        if !self.store.supports_paused_limit_revision() {
            return Err(ai_agents_core::AgentError::Config(
                "paused limit revision is unsupported by this task store".into(),
            ));
        }
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;
        let (proposed, _) = super::super::control::prepare_paused_limit_revision(
            &snapshot,
            limits.clone(),
            reason,
            &self.host,
        )?;
        let (owner, whole_expiry) = {
            let paused = self.paused.lock();
            let paused = paused
                .get(run_id)
                .ok_or(TaskRunStorageError::NotResumable)?;
            paused.owner.check()?;
            if paused.participants.unsettled()
                || paused
                    .whole_expiry
                    .is_some_and(|expiry| std::time::Instant::now() >= expiry)
            {
                return Err(TaskRunStorageError::NotResumable.into());
            }
            (paused.owner.clone(), paused.whole_expiry)
        };
        let revised_expiry = match (
            whole_expiry,
            payload.clocks.expires_at,
            proposed.clocks.expires_at,
        ) {
            (Some(original), Some(old), Some(new)) => {
                let delta = new.signed_duration_since(old);
                Some(
                    if delta >= chrono::Duration::zero() {
                        original.checked_add(
                            delta
                                .to_std()
                                .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?,
                        )
                    } else {
                        original.checked_sub(
                            (-delta)
                                .to_std()
                                .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?,
                        )
                    }
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?,
                )
            }
            (None, None, Some(new)) => Some(
                std::time::Instant::now()
                    .checked_add(
                        (new - chrono::Utc::now())
                            .to_std()
                            .map_err(|_| TaskRunStorageError::AdmissionExpired)?,
                    )
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?,
            ),
            (None, None, None) => None,
            _ => return Err(TaskRunStorageError::InvalidCheckpoint.into()),
        };
        if revised_expiry.is_some_and(|expiry| std::time::Instant::now() >= expiry) {
            return Err(TaskRunStorageError::AdmissionExpired.into());
        }
        // Audit growth and checkpoint validation are proven pre-write failures, so reject them before arming abandonment.
        let expected = snapshot.transition(
            &TaskRunMutation::HostControl {
                expected_revision,
                owner_token: None,
                status: TaskRunStatus::Paused,
                payload: serde_json::to_value(&proposed)?,
                deadline: whole_expiry,
                expires_at: payload.clocks.expires_at,
            },
            chrono::Utc::now(),
        )?;
        proposed.validate(&expected, &self.config_identity)?;
        proposed.validate_successor_with_control(&payload, expected.revision, false, true)?;
        let mut cleanup = OwnedTurnCleanup::new(owner);
        let saved = self
            .store
            .revise_paused_limits(
                run_id,
                expected_revision,
                limits,
                reason,
                &self.host,
                whole_expiry,
            )
            .await?;
        let Some(saved) = saved else {
            cleanup.finish();
            return Err(TaskRunStorageError::Conflict.into());
        };
        if saved.key != snapshot.key
            || saved.actor_id != snapshot.actor_id
            || saved.schema_version != snapshot.schema_version
            || saved.revision
                != snapshot
                    .revision
                    .checked_add(1)
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            || saved.created_at != snapshot.created_at
            || saved.updated_at < snapshot.updated_at
            || saved.owner_token.is_some()
            || saved.cancel_requested != snapshot.cancel_requested
            || saved.status != TaskRunStatus::Paused
            || saved.payload != serde_json::to_value(&proposed)?
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let run = TaskRun::from_checkpoint(&saved, &self.config_identity)?;
        self.paused
            .lock()
            .get_mut(run_id)
            .ok_or(TaskRunStorageError::NotResumable)?
            .whole_expiry = revised_expiry;
        cleanup.finish();
        Ok(TaskRunResult {
            run,
            final_response: None,
        })
    }

    /// Stop claims rather than fabricating cancellation; every cleanup admission requires an exact successor receipt.
    pub(super) async fn admit_paused_cleanup(
        &self,
        snapshot: TaskRunSnapshot,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        let mutation = if let Some(control) = stop_reason() {
            if let Some(request_id) = &control.request_id {
                let payload: TaskCheckpointPayload =
                    serde_json::from_value(snapshot.payload.clone())?;
                if payload.pending.as_ref().is_none_or(|pending| {
                    pending.id != *request_id
                        || !matches!(pending.kind, TaskPendingKind::ExternalCondition)
                }) {
                    return Err(TaskRunStorageError::NotResumable.into());
                }
            }
            if snapshot.cancel_requested {
                return Err(TaskRunStorageError::NotResumable.into());
            }
            TaskRunMutation::Claim {
                expected_revision: snapshot.revision,
                owner_token: uuid::Uuid::new_v4().to_string(),
            }
        } else if snapshot.cancel_requested {
            return Ok(snapshot);
        } else {
            TaskRunMutation::RequestCancel {
                expected_revision: snapshot.revision,
            }
        };
        let saved = self.store.mutate(&snapshot.key.run_id, &mutation).await?;
        super::super::store::verify_checkpoint_receipt(&snapshot, &mutation, &saved)?;
        Ok(saved)
    }

    /// A host-stop terminal checkpoint retains usage and exact settled histories and never publishes task completion.
    pub(super) async fn acknowledge_paused_cleanup(
        &self,
        admitted: &TaskRunSnapshot,
        mut payload: TaskCheckpointPayload,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        let mutation = if let Some(control) = stop_reason() {
            payload.stop_reason = Some(
                if control.kind == "stop" {
                    "user_stopped"
                } else {
                    "external_unavailable"
                }
                .into(),
            );
            let sequence = payload
                .evidence
                .records
                .iter()
                .map(|record| record.sequence)
                .chain(
                    payload
                        .evidence
                        .tool_calls
                        .iter()
                        .map(|record| record.sequence),
                )
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| {
                    ai_agents_core::AgentError::Config("audit sequence overflow".into())
                })?;
            payload.evidence.records.push(TaskEvidenceRecord { run_id: admitted.key.run_id.clone(), sequence, stage: payload.stage.clone(), attempt: None,
                origin: "host.control".into(), data: serde_json::json!({"action":{"kind":control.kind,"request_id":control.request_id},"reason":control.reason,"expected_revision":admitted.revision.saturating_sub(1)}) });
            TaskRunMutation::FinalCheckpoint {
                expected_revision: admitted.revision,
                owner_token: admitted
                    .owner_token
                    .clone()
                    .ok_or(TaskRunStorageError::Owned)?,
                status: control.status,
                payload: serde_json::to_value(payload)?,
                release: true,
                expires_at: None,
                evidence: None,
                deadline: std::time::Instant::now(),
            }
        } else {
            TaskRunMutation::AcknowledgeCancel {
                expected_revision: admitted.revision,
                payload: serde_json::to_value(payload)?,
            }
        };
        let saved = self.store.mutate(&admitted.key.run_id, &mutation).await?;
        super::super::store::verify_checkpoint_receipt(admitted, &mutation, &saved)?;
        Ok(saved)
    }
}
