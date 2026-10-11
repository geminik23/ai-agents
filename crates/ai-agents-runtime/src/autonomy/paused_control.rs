//! Authenticated Rust host stop reuses exact safe cleanup without treating an answer as control authority.

use super::*;
use std::future::Future;

tokio::task_local! { static HOST_STOP: String; }

/// Only the explicit Rust host API installs a stop reason; external signals and tool responses cannot deserialize it.
pub(super) async fn scope_stop<F: Future>(reason: String, future: F) -> F::Output {
    HOST_STOP.scope(reason, future).await
}

/// Cancellation helpers retain their original semantics unless the host explicitly owns a stop operation.
pub(super) fn stop_reason() -> Option<String> {
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
        scope_stop(reason.into(), self.cancel_paused(run_id, expected_revision)).await
    }

    /// Stop claims rather than fabricating cancellation; ordinary cancellation keeps its stored-request and acknowledgement protocol.
    pub(super) async fn admit_paused_cleanup(
        &self,
        snapshot: TaskRunSnapshot,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        if stop_reason().is_some() {
            if snapshot.cancel_requested {
                return Err(TaskRunStorageError::NotResumable.into());
            }
            self.store
                .mutate(
                    &snapshot.key.run_id,
                    &TaskRunMutation::Claim {
                        expected_revision: snapshot.revision,
                        owner_token: uuid::Uuid::new_v4().to_string(),
                    },
                )
                .await
        } else if snapshot.cancel_requested {
            Ok(snapshot)
        } else {
            self.store
                .mutate(
                    &snapshot.key.run_id,
                    &TaskRunMutation::RequestCancel {
                        expected_revision: snapshot.revision,
                    },
                )
                .await
        }
    }

    /// A host-stop terminal checkpoint retains usage and exact settled histories and never publishes task completion.
    pub(super) async fn acknowledge_paused_cleanup(
        &self,
        admitted: &TaskRunSnapshot,
        mut payload: TaskCheckpointPayload,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        let mutation = if let Some(reason) = stop_reason() {
            payload.stop_reason = Some("user_stopped".into());
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
                origin: "host.control".into(), data: serde_json::json!({"action":{"kind":"stop"},"reason":reason,"expected_revision":admitted.revision.saturating_sub(1)}) });
            TaskRunMutation::FinalCheckpoint {
                expected_revision: admitted.revision,
                owner_token: admitted
                    .owner_token
                    .clone()
                    .ok_or(TaskRunStorageError::Owned)?,
                status: TaskRunStatus::Incomplete,
                payload: serde_json::to_value(payload)?,
                release: true,
                expires_at: None,
                deadline: std::time::Instant::now(),
            }
        } else {
            TaskRunMutation::AcknowledgeCancel {
                expected_revision: admitted.revision,
                payload: serde_json::to_value(payload)?,
            }
        };
        self.store.mutate(&admitted.key.run_id, &mutation).await
    }
}
