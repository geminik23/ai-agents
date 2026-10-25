//! Host-facing projections and cancellation reuse conditional ownership; display values never become execution permits.

use super::*;
use crate::autonomy::controller::{ControllerProposal, ControllerReview};
use crate::autonomy::events::{safe_text, safe_value};
use crate::autonomy::suspension::BatchApproval;
use serde::Serialize;
use serde_json::Value;

#[cfg(test)]
#[path = "hosting_control_tests.rs"]
mod tests;

/// Maximum serialized read-only display snapshot, independent of exact checkpoint and event-envelope limits.
pub const MAX_TASK_DISPLAY_BYTES: usize = 16_384;

/// Cancellation acknowledgement means actual retained cleanup returned successfully, not just that a flag was stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationDisposition {
    Requested,
    Acknowledged,
    AlreadyTerminal,
}

/// A scoped cancellation receipt; terminal metadata alone never certifies live cleanup, rollback or exactly-once effects.
#[derive(Debug, Clone, Serialize)]
pub struct CancellationOutcome {
    pub disposition: CancellationDisposition,
    pub run: TaskRunSummary,
    /// None means a store-only caller cannot observe the live owner's process.
    pub locally_retained: Option<bool>,
}

impl CancellationOutcome {
    /// Records an exact-revision request without constructing providers, adopting owners or executing a pending continuation.
    /// A remote request always remains Requested; errors can be ambiguous and callers must inspect rather than blindly retry.
    pub async fn request(store: &dyn TaskRunStore, run_id: &str, revision: u64) -> Result<Self> {
        let snapshot = store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        if snapshot.status.is_terminal() {
            return Ok(Self {
                disposition: CancellationDisposition::AlreadyTerminal,
                run: snapshot.summary(),
                locally_retained: None,
            });
        }
        if snapshot.cancel_requested {
            return Ok(Self {
                disposition: CancellationDisposition::Requested,
                run: snapshot.summary(),
                locally_retained: None,
            });
        }
        let mutation = TaskRunMutation::RequestCancel {
            expected_revision: revision,
        };
        let saved = store.mutate(run_id, &mutation).await?;
        super::super::store::verify_checkpoint_receipt(&snapshot, &mutation, &saved)?;
        Ok(Self {
            disposition: CancellationDisposition::Requested,
            run: saved.summary(),
            locally_retained: None,
        })
    }
}

/// Bounded default-redacted snapshot of acknowledged facts, not an event replay, fresh completion proof or control authority.
/// Optional observations remain unavailable when no acknowledged source exists; clocks are stored facts rather than a UI timer.
#[derive(Debug, Clone, Serialize)]
pub struct TaskRunDisplay {
    pub summary: TaskRunSummary,
    /// Local runtime custody is separate from the stored owner token and is not global quiescence evidence.
    /// Store-only reads have no attached runtime and report false without certifying the remote owner's cleanup.
    pub locally_retained: bool,
    pub objective: Value,
    pub profile: Option<Value>,
    pub stage: Option<Value>,
    pub pause_reason: Option<Value>,
    pub stop_reason: Option<Value>,
    pub pending_kind: Option<TaskPendingKind>,
    pub counters: TaskCounters,
    pub limits: TaskRunLimits,
    pub clocks: TaskClocks,
    pub todos: Value,
    pub progress: Value,
    pub last_gate: Option<GateOutcome>,
    pub last_gate_cycle: Option<u64>,
    pub last_gate_objective_revision: Option<u64>,
    pub artifacts: Option<Value>,
    /// None means there is no local retained review from which to establish this fact.
    pub restoration_review_required: Option<bool>,
}

impl TaskRunDisplay {
    /// Reads a safe snapshot using only an authorized scoped store, without constructing providers, claiming owners or consuming responses.
    /// False local retention means this store-only view has no attached runtime; remote custody and nonterminal restoration review remain unknown.
    pub async fn read(store: &dyn TaskRunStore, run_id: &str) -> Result<Self> {
        let snapshot = store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        // The store owns agent/actor/configuration admission; the shared projector additionally requires structurally valid checkpoint data.
        payload.validate(&snapshot, &payload.config_identity)?;
        let review = snapshot.status.is_terminal().then_some(false);
        Self::project(&snapshot, &payload, false, review)
    }

    // Both local and store-only observation use the same bounded redaction; only a retained controller may supply local custody/review facts.
    fn project(
        snapshot: &TaskRunSnapshot,
        payload: &TaskCheckpointPayload,
        locally_retained: bool,
        restoration_review_required: Option<bool>,
    ) -> Result<Self> {
        let latest = payload
            .evidence
            .records
            .iter()
            .rev()
            .find(|r| r.origin == "controller");
        let last_gate = latest
            .and_then(|r| r.data.get("gate"))
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?;
        // Historical controller evidence is display data, not an executable cursor or a fresh completion permit.
        let evidence: Option<EvaluationEvidence> = payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == "runtime.controller")
            .and_then(|adapter| adapter.state.get("evidence"))
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()?;
        let artifacts = evidence
            .as_ref()
            .filter(|state| !state.artifacts.is_empty())
            .map(|state| {
                let observations: Vec<_> = state
                    .artifacts
                    .iter()
                    .map(|a| json!({"path":a.value.path,"exists":a.value.exists}))
                    .collect();
                safe_value(&json!(observations))
            });
        let view = Self {
            summary: snapshot.summary(),
            locally_retained,
            objective: safe_text(&payload.objective),
            profile: payload.profile.as_deref().map(safe_text),
            stage: payload.stage.as_deref().map(safe_text),
            pause_reason: payload.pause_reason.as_deref().map(safe_text),
            stop_reason: payload.stop_reason.as_deref().map(safe_text),
            pending_kind: payload.pending.as_ref().map(|p| p.kind),
            counters: payload.counters.clone(),
            limits: payload.limits.clone(),
            clocks: payload.clocks.clone(),
            todos: safe_value(&serde_json::to_value(
                payload.todos.as_ref().map(|t| &t.items),
            )?),
            progress: safe_value(
                &json!({"observations":payload.progress.observation_sequence,
                "cycles_without_progress":payload.progress.cycles_without_progress,"replans":payload.progress.replans,
                "high_water_marks":payload.progress.high_water_marks}),
            ),
            last_gate,
            last_gate_cycle: latest.and_then(|r| r.data["scope"]["cycle"].as_u64()),
            last_gate_objective_revision: latest
                .and_then(|r| r.data["scope"]["objective_revision"].as_u64()),
            artifacts,
            restoration_review_required,
        };
        if serde_json::to_vec(&view)?.len() > MAX_TASK_DISPLAY_BYTES {
            return Err(AgentError::Config(
                "task display exceeds the bounded snapshot limit".into(),
            ));
        }
        Ok(view)
    }
}

/// Exact allowlisted prompt data for an authenticated host, deliberately not serializable or content-bearing Debug output.
/// Approval context and question defaults may be sensitive; they must not be logged or substituted for fresh controller admission.
pub enum TaskHostPrompt {
    Approval {
        message: String,
        context: Value,
    },
    Question(ai_agents_tools::QuestionRequest),
    Continue,
    Capacity {
        reason: String,
        original_limits: TaskRunLimits,
    },
    /// Completing required stages still needs review and does not itself establish task success.
    StageReview {
        next_stage: Option<String>,
        instruction: Option<String>,
        finishes_lifecycle: bool,
    },
    PlanReview {
        instruction: String,
    },
    ExternalCondition {
        binding: String,
    },
    ValidationObservation,
}

impl std::fmt::Debug for TaskHostPrompt {
    // Diagnostic formatting exposes only the kind, never exact reviewed context, question text or domain bindings.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Approval { .. } => "Approval",
            Self::Question(_) => "Question",
            Self::Continue => "Continue",
            Self::Capacity { .. } => "Capacity",
            Self::StageReview { .. } => "StageReview",
            Self::PlanReview { .. } => "PlanReview",
            Self::ExternalCondition { .. } => "ExternalCondition",
            Self::ValidationObservation => "ValidationObservation",
        })
    }
}

/// One exact pending exchange; nested prompts retain the coordinating request ID for the existing response driver.
/// A view can become stale immediately, so the host must return its original revision and request ID without auto-refresh/replay.
pub struct TaskHostInteraction {
    pub key: TaskRunKey,
    pub revision: u64,
    pub request_id: String,
    pub issued_revision: u64,
    pub locally_retained: bool,
    pub prompt: TaskHostPrompt,
    pub restoration_review: Option<TaskRestorationReview>,
}

impl std::fmt::Debug for TaskHostInteraction {
    // Even opaque request/actor identifiers stay out of diagnostics; displaying a prompt never authenticates its recipient.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskHostInteraction")
            .field("revision", &self.revision)
            .field("prompt", &self.prompt)
            .finish_non_exhaustive()
    }
}

impl AutonomyRunner {
    /// Applies cancellation under the retained execution's serial boundary, or records a store-only remote request.
    /// Only an awaited known-safe pause cleanup yields Acknowledged; running callers must keep polling their original operation.
    pub async fn cancel_task(&self, run_id: &str, revision: u64) -> Result<CancellationOutcome> {
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        if snapshot.status.is_terminal() {
            return Ok(CancellationOutcome {
                disposition: CancellationDisposition::AlreadyTerminal,
                run: snapshot.summary(),
                locally_retained: Some(
                    self.agent
                        .captured_task_owner()
                        .is_some_and(|o| o.run_id == run_id),
                ),
            });
        }
        let paused = self.paused.lock().contains_key(run_id);
        if paused {
            let result = self.cancel_paused(run_id, revision).await?;
            return Ok(CancellationOutcome {
                disposition: CancellationDisposition::Acknowledged,
                locally_retained: Some(false),
                run: TaskRunSummary {
                    key: result.run.key,
                    actor_id: result.run.actor_id,
                    revision: result.run.revision,
                    status: result.run.status,
                    owned: false,
                    cancel_requested: true,
                    created_at: result.run.created_at,
                    updated_at: result.run.updated_at,
                },
            });
        }
        let execution = self
            .active
            .lock()
            .as_ref()
            .filter(|e| e.run_id == run_id)
            .cloned();
        if let Some(execution) = execution {
            let run = execution.request_cancel_at_revision(Some(revision)).await?;
            return Ok(CancellationOutcome {
                disposition: CancellationDisposition::Requested,
                run,
                locally_retained: Some(
                    self.agent
                        .captured_task_owner()
                        .is_some_and(|o| o.run_id == run_id),
                ),
            });
        }
        CancellationOutcome::request(self.store.as_ref(), run_id, revision).await
    }

    /// Reads one validated acknowledged revision without claiming, restoring, calling a provider or consuming a pending decision.
    /// This fixed projection excludes recovery adapters and exact responses; historical gate facts never establish current eligibility.
    pub async fn task_display(&self, run_id: &str) -> Result<TaskRunDisplay> {
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;

        // Inspection also supports terminal or recovery data, where no executable paused cursor may exist.
        let restoration_review_required = if snapshot.status.is_terminal() {
            Some(false)
        } else {
            self.paused.lock().get(run_id).map(|paused| {
                paused
                    .restoration
                    .as_ref()
                    .is_some_and(|review| !review.view.approved)
            })
        };
        TaskRunDisplay::project(
            &snapshot,
            &payload,
            self.agent
                .captured_task_owner()
                .is_some_and(|o| o.run_id == run_id),
            restoration_review_required,
        )
    }

    /// Projects only a validated pending prompt for the exact requested revision; it grants no response or restoration authority.
    /// The host authenticates its recipient and keeps sensitive prompt data separate from display/export snapshots.
    pub async fn task_interaction(
        &self,
        run_id: &str,
        revision: u64,
    ) -> Result<Option<TaskHostInteraction>> {
        let snapshot = self
            .store
            .load(run_id)
            .await?
            .ok_or(TaskRunStorageError::NotFound)?;
        if snapshot.revision != revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.validate(&snapshot, &self.config_identity)?;
        if snapshot.status != TaskRunStatus::Paused || snapshot.cancel_requested {
            return Ok(None);
        }
        let Some(pending) = payload.pending.as_ref() else {
            return Ok(None);
        };
        let prompt = prompt(pending, &payload.settings)?;
        Ok(Some(TaskHostInteraction {
            key: snapshot.key,
            revision,
            request_id: pending.id.clone(),
            issued_revision: pending.issued_revision,
            locally_retained: self
                .agent
                .captured_task_owner()
                .is_some_and(|o| o.run_id == run_id),
            prompt,
            restoration_review: self.paused.lock().get(run_id).and_then(|paused| {
                paused
                    .restoration
                    .as_ref()
                    .map(|review| review.view.clone())
            }),
        }))
    }
}

// Only known framework prompt schemas are projected; traversal preserves the outer response identity and never copies executable cursor state.
fn prompt(pending: &TaskPendingRequest, settings: &AutonomyProfile) -> Result<TaskHostPrompt> {
    let mut leaf = pending.clone();
    for _ in 0..32 {
        let value = &leaf.reviewed_action;
        if value.get("child_operation").is_some() && value.get("request").is_some() {
            leaf = serde_json::from_value(value["request"].clone())?;
            continue;
        }
        if value.get("call_id").is_some() {
            let approval: BatchApproval = serde_json::from_value(value.clone())?;
            if let Some(limit) = approval.capacity()? {
                return Ok(TaskHostPrompt::Capacity {
                    reason: limit.reason,
                    original_limits: limit.original,
                });
            }
            if let Some(question) = approval.question {
                return Ok(TaskHostPrompt::Question(serde_json::from_value(question)?));
            }
            return Ok(TaskHostPrompt::Approval {
                message: approval.message,
                context: approval.context,
            });
        }
        if value.get("proposal").is_some() {
            let review: ControllerReview = serde_json::from_value(value.clone())?;
            return Ok(match review.proposal {
                ControllerProposal::Continue => TaskHostPrompt::Continue,
                ControllerProposal::Question { prompt, .. } => {
                    TaskHostPrompt::Question(ai_agents_tools::QuestionRequest {
                        question: prompt,
                        allow_other: true,
                        ..Default::default()
                    })
                }
                ControllerProposal::Capacity { limit } => TaskHostPrompt::Capacity {
                    reason: limit.reason,
                    original_limits: limit.original,
                },
                ControllerProposal::Stage { lifecycle } => {
                    lifecycle.validate(settings)?;
                    let stages = settings.lifecycle.as_deref().unwrap_or_default();
                    let stage = stages.get(lifecycle.current_stage);
                    TaskHostPrompt::StageReview {
                        next_stage: stage.map(|stage| stage.id.clone()),
                        instruction: stage.and_then(|stage| stage.instruction.clone()),
                        finishes_lifecycle: lifecycle.current_stage == stages.len(),
                    }
                }
                ControllerProposal::Plan { instruction } => {
                    TaskHostPrompt::PlanReview { instruction }
                }
            });
        }
        return match leaf.kind {
            TaskPendingKind::ExternalCondition => {
                let wait: ExternalWait = serde_json::from_value(value.clone())?;
                Ok(TaskHostPrompt::ExternalCondition {
                    binding: wait.binding.id,
                })
            }
            TaskPendingKind::ValidationObservation => Ok(TaskHostPrompt::ValidationObservation),
            _ => Err(TaskRunStorageError::InvalidCheckpoint.into()),
        };
    }
    Err(TaskRunStorageError::InvalidCheckpoint.into())
}
