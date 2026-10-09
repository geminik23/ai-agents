//! Explicit Rust host control actions; model output and external signals are never deserialized into authority.

use super::*;
use ai_agents_core::autonomy::{TaskRunSnapshot, TaskRunStatus};
use ai_agents_core::{AgentError, Result};
use serde_json::json;

/// The host must authenticate the caller before invoking the control plane.
/// These actions deliberately have no Deserialize implementation or model-callable tool.
pub enum HostControlAction {
    Stop {
        reason: String,
    },
    ModifyObjective {
        objective: String,
        reason: String,
    },
    ReviseLimits {
        limits: TaskRunLimits,
        reason: String,
    },
}

/// Prepares an auditable safe-boundary update without changing memory, replaying work or refilling counters.
pub(crate) fn prepare_host_control(
    snapshot: &TaskRunSnapshot,
    action: HostControlAction,
    host: &AutonomyHostCeilings,
) -> Result<(TaskCheckpointPayload, TaskRunStatus)> {
    let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
    if snapshot.status.is_terminal()
        || snapshot.cancel_requested
        || payload
            .clocks
            .expires_at
            .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        || payload.pending.is_some()
        || payload.children.iter().any(|c| c.pending.is_some())
        || !matches!(payload.runtime.continuation, TaskContinuation::BetweenTurns)
        || payload.clocks.active_interval_started_at.is_some()
        || payload.reservations.iter().any(|r| {
            matches!(
                r.state,
                TaskEffectState::Dispatched | TaskEffectState::Uncertain
            )
        })
    {
        return Err(AgentError::Config(
            "host action requires a safe nonterminal boundary".into(),
        ));
    }
    let (reason, audit, status) = match action {
        HostControlAction::Stop { reason } => {
            payload.stop_reason = Some("user_stopped".into());
            (reason, json!({"kind":"stop"}), TaskRunStatus::Incomplete)
        }
        HostControlAction::ModifyObjective { objective, reason } => {
            if objective.trim().is_empty() {
                return Err(AgentError::Config("empty revised objective".into()));
            }
            let old = payload.objective.clone();
            payload.objective = objective.clone();
            payload.objective_revision = payload
                .objective_revision
                .checked_add(1)
                .ok_or_else(|| AgentError::Config("objective revision overflow".into()))?;
            payload.progress.high_water_marks = serde_json::Value::Null;
            payload.progress.cycles_without_progress = 0;
            if let Some(todos) = &mut payload.todos {
                todos.items.clear();
                todos.binding.objective_revision = payload.objective_revision;
                todos.binding.token = uuid::Uuid::new_v4().to_string();
            }
            // Old exact evidence and effects remain as audit data; their old objective identity is now ineligible.
            (
                reason,
                json!({"kind":"modify_objective","old_objective":old,"new_objective":objective,"objective_revision":payload.objective_revision}),
                snapshot.status,
            )
        }
        HostControlAction::ReviseLimits { limits, reason } => {
            if payload
                .clocks
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
            {
                return Err(AgentError::Config("cannot revise an expired task".into()));
            }
            validate_limits(&limits, &payload, host)?;
            let old = payload.limits.clone();
            if limits.max_wall_time_seconds != old.max_wall_time_seconds {
                payload.clocks.expires_at = limits
                    .max_wall_time_seconds
                    .map(|seconds| {
                        let duration = chrono::Duration::try_seconds(
                            i64::try_from(seconds)
                                .map_err(|_| AgentError::Config("wall lifetime overflow".into()))?,
                        )
                        .ok_or_else(|| AgentError::Config("wall lifetime overflow".into()))?;
                        snapshot
                            .created_at
                            .checked_add_signed(duration)
                            .ok_or_else(|| AgentError::Config("wall expiry overflow".into()))
                    })
                    .transpose()?;
            }
            if payload
                .clocks
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
            {
                return Err(AgentError::Config(
                    "revised lifetime already expired from original admission".into(),
                ));
            }
            payload.limits = limits.clone();
            (
                reason,
                json!({"kind":"revise_limits","old_limits":old,"new_limits":limits}),
                snapshot.status,
            )
        }
    };
    if reason.trim().is_empty() || reason.chars().count() > 512 {
        return Err(AgentError::Config(
            "host control requires a bounded audit reason".into(),
        ));
    }
    let sequence = payload
        .evidence
        .records
        .iter()
        .map(|r| r.sequence)
        .chain(payload.evidence.tool_calls.iter().map(|r| r.sequence))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| AgentError::Config("audit sequence overflow".into()))?;
    payload.evidence.records.push(TaskEvidenceRecord {
        run_id: snapshot.key.run_id.clone(),
        sequence,
        stage: payload.stage.clone(),
        attempt: None,
        origin: "host.control".into(),
        data: json!({"expected_revision":snapshot.revision,"reason":reason,"action":audit}),
    });
    Ok((payload, status))
}

// Revised ceilings include consumed/reserved usage, original lifetime and fixed optional enforcement capabilities.
fn validate_limits(
    limits: &TaskRunLimits,
    payload: &TaskCheckpointPayload,
    host: &AutonomyHostCeilings,
) -> Result<()> {
    let invalid = || {
        AgentError::Config(
            "limit revision exceeds host authority or consumed/reserved usage".into(),
        )
    };
    for (requested, consumed, ceiling) in [
        (
            u64::from(limits.max_turns),
            payload.counters.turns,
            host.max_turns.map(u64::from),
        ),
        (
            u64::from(limits.max_llm_calls),
            payload.counters.llm_attempts,
            host.max_llm_calls.map(u64::from),
        ),
        (
            u64::from(limits.max_tool_calls),
            payload.counters.tool_attempts,
            host.max_tool_calls.map(u64::from),
        ),
        (
            u64::from(limits.max_command_calls),
            payload.counters.command_attempts,
            host.max_command_calls.map(u64::from),
        ),
        (
            limits.max_active_time_seconds,
            payload
                .clocks
                .active_millis
                .saturating_add(payload.clocks.interrupted_interval_millis)
                .div_ceil(1000),
            host.max_active_time_seconds,
        ),
    ] {
        if requested == 0 || requested < consumed || ceiling.is_some_and(|c| requested > c) {
            return Err(invalid());
        }
    }
    if limits.max_micro_usd.is_some() != payload.limits.max_micro_usd.is_some()
        || limits.max_declared_write_paths.is_some()
            != payload.limits.max_declared_write_paths.is_some()
        || ((payload.limits.max_wall_time_seconds.is_some()
            || host.max_wall_time_seconds.is_some())
            && limits.max_wall_time_seconds.is_none())
    {
        return Err(invalid());
    }
    if let Some(cost) = limits.max_micro_usd {
        let reserved = payload
            .reservations
            .iter()
            .filter(|r| r.state != TaskEffectState::Completed)
            .try_fold(payload.counters.charged_micro_usd, |sum, r| {
                sum.checked_add(r.reserved_micro_usd.saturating_sub(r.charged_micro_usd))
            })
            .ok_or_else(invalid)?;
        if cost < reserved
            || host
                .max_cost_usd
                .as_ref()
                .is_none_or(|c| cost > c.micro_usd())
        {
            return Err(invalid());
        }
    }
    if let Some(paths) = limits.max_declared_write_paths
        && (u64::from(paths) < payload.declared_write_targets.len() as u64
            || host.max_declared_write_paths.is_none_or(|c| paths > c))
    {
        return Err(invalid());
    }
    if let Some(wall) = limits.max_wall_time_seconds
        && (wall == 0 || host.max_wall_time_seconds.is_some_and(|c| wall > c))
    {
        return Err(invalid());
    }
    Ok(())
}
