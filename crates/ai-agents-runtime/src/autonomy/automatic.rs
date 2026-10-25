//! Automatic state and selected-skill entry share the explicit controller, storage and retained ownership.

use super::*;
use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_process::ProcessData;
use ai_agents_skills::SkillDefinition;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;

/// Host-installed storage, ceilings and reconstruction binding for automatic state/skill tasks.
/// Omission uses a retained volatile backend; durable behavior still requires an explicit reconstruction host.
#[derive(Default)]
pub struct AutomaticTaskOptions {
    pub ceilings: AutonomyHostCeilings,
    pub store: Option<Arc<dyn TaskRunStore>>,
    pub config_identity: Option<String>,
    pub restore_host: Option<Arc<dyn TaskRestoreHost>>,
}

/// The service owns paused controllers while controller runtime handles hold only a weak service reference.
pub(crate) struct AutomaticTaskHost {
    pub(crate) config: AutonomyConfig,
    pub(crate) options: AutomaticTaskOptions,
    pub(crate) interaction_delivery: bool,
    // Prepared session backends can be shared, but task adapters retain independent actor/configuration validation.
    pub(crate) storage: Arc<dyn ai_agents_core::AgentStorage>,
    pub(crate) runner: parking_lot::Mutex<Option<Arc<AutonomyRunner>>>,
    pub(crate) outcome: parking_lot::Mutex<Arc<AutomaticOutcome>>,
}

/// Each admitted task owns its delivery slot so a later input cannot replace an earlier composing caller's result.
#[derive(Default)]
pub(crate) struct AutomaticOutcome {
    pub(crate) key: Option<TaskRunKey>,
    pub(crate) result: parking_lot::Mutex<Option<TaskRunResult>>,
    pub(crate) changed: tokio::sync::Notify,
    pub(crate) failed: std::sync::atomic::AtomicBool,
}

impl AutomaticOutcome {
    /// Publication wakes ordinary composing callers without treating the pause itself as a successful child response.
    pub(crate) fn publish(&self, result: &TaskRunResult) {
        if self.key.as_ref() != Some(&result.run.key) {
            return;
        }
        let mut current = self.result.lock();
        if current.as_ref().is_some_and(|prior| {
            prior.run.status.is_terminal() || prior.run.revision >= result.run.revision
        }) {
            return;
        }
        *current = Some(result.clone());
        self.failed
            .store(false, std::sync::atomic::Ordering::Release);
        drop(current);
        self.changed.notify_waiters();
    }

    /// Ambiguous execution failure wakes ordinary callers without replacing a protected task with a fabricated terminal status.
    pub(crate) fn failure(&self) {
        let current = self.result.lock();
        if current
            .as_ref()
            .is_some_and(|result| result.run.status.is_terminal())
        {
            return;
        }
        self.failed
            .store(true, std::sync::atomic::Ordering::Release);
        drop(current);
        self.changed.notify_waiters();
    }
}

/// A dropped host operation wakes an ordinary waiter only after its actual owner becomes abandoned, never on a cancellation receipt.
pub(crate) struct AutomaticDeliveryGuard {
    pub(crate) owner: std::sync::Weak<RunOwner>,
    pub(crate) outcome: std::sync::Weak<AutomaticOutcome>,
}

impl Drop for AutomaticDeliveryGuard {
    // Drop performs no I/O and grants no cleanup success; the protected task remains available for explicit recovery.
    fn drop(&mut self) {
        if self
            .owner
            .upgrade()
            .is_some_and(|owner| owner.is_abandoned())
            && let Some(outcome) = self.outcome.upgrade()
        {
            outcome.failure();
        }
    }
}

/// One ordinary clarification preserves its prepared data under the original selected skill and state/manager lineage.
pub(crate) struct PreparedSkillInput {
    pub(crate) skill: String,
    pub(crate) epoch: u64,
    pub(crate) generation: Option<u64>,
    pub(crate) data: Box<ProcessData>,
}

/// Exact activation data is separate from a transition cursor and survives movement outside the original state.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AutomaticBinding {
    pub(crate) state: Option<String>,
    pub(crate) settings: ai_agents_core::autonomy::AutonomyOverride,
    pub(crate) skill: Option<SkillDefinition>,
    pub(crate) prepared: ProcessData,
    pub(crate) user_committed: bool,
    pub(crate) state_generation: Option<u64>,
}

impl AutomaticBinding {
    /// A state-owned run can carry a selected nested skill without becoming a skill-owned run.
    pub(crate) fn scope(&self) -> AutonomyScope {
        if self.state.is_some() {
            AutonomyScope::State
        } else {
            AutonomyScope::Skill
        }
    }
}

/// Only the currently admitted root can publish this live handoff; serialized metadata cannot create it.
pub(crate) struct AutomaticEntry {
    pub(crate) owner: Arc<RunOwner>,
    pub(crate) binding: Box<AutomaticBinding>,
    pub(crate) outcome: Arc<AutomaticOutcome>,
}

pub(crate) type Promotion = Arc<parking_lot::Mutex<Option<Box<AutomaticEntry>>>>;
tokio::task_local! {
    static PROMOTION: (crate::runtime::RootTurnGate, Promotion);
    static COMPOSING_CALL: bool;
}

/// Poll-local routing prevents child runtimes or callbacks from consuming another root's promotion slot.
pub(crate) async fn scope_promotion<F: Future>(
    gate: crate::runtime::RootTurnGate,
    slot: Promotion,
    future: F,
) -> F::Output {
    PROMOTION.scope((gate, slot), future).await
}

/// Marks an ordinary framework composition call as a result consumer, not as task execution authority.
pub(crate) async fn scope_composing<F: Future>(future: F) -> F::Output {
    COMPOSING_CALL.scope(true, future).await
}

/// SDK composition without a runtime root still needs to await the actual child outcome instead of aggregating a pause.
pub(crate) fn composing() -> bool {
    COMPOSING_CALL.try_with(|active| *active).unwrap_or(false)
}

/// Gate pointer identity, not an input label or run ID, qualifies the pending root handoff.
pub(crate) fn promotion(gate: &crate::runtime::RootTurnGate) -> Option<Promotion> {
    PROMOTION
        .try_with(|(actual, slot)| Arc::ptr_eq(actual, gate).then(|| slot.clone()))
        .ok()
        .flatten()
}

/// The binding is exact private recovery data; absence preserves older explicit task and skill checkpoints.
pub(crate) fn binding(payload: &TaskCheckpointPayload) -> Result<Option<Box<AutomaticBinding>>> {
    payload
        .adapters
        .iter()
        .find(|adapter| adapter.id == "runtime.activation")
        .map(|adapter| {
            if adapter.contract_version != 1 {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            Ok(serde_json::from_value(adapter.config.clone())?)
        })
        .transpose()
}

/// Supplies bounded acknowledged script inputs to controller continuation without exposing execution records or treating old results as fresh proof.
/// Exact step data remains in the cursor; an oversized model-context projection is explicitly unavailable rather than truncating recovery data.
pub(crate) fn skill_results(payload: &TaskCheckpointPayload) -> serde_json::Value {
    let values: Vec<_> = payload.adapters.iter().filter(|adapter| adapter.adapter == "runtime.skill").filter_map(|adapter| {
        let cursor = &adapter.state["cursor"];
        let steps = cursor["context"]["step_results"].as_array()?;
        Some(serde_json::json!({
            "execution":cursor["execution_id"],"skill":cursor["definition"]["id"],"completion_authority":false,
            "steps":steps.iter().map(|step|serde_json::json!({"index":step["step_index"],"arguments":step["args"],"result":step["result"]})).collect::<Vec<_>>()
        }))
    }).collect();
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    let value = serde_json::Value::Array(values);
    if bounded_value(&value, 65_536).is_err() {
        serde_json::json!({"available":false,"reason":"script results exceed controller context limit; exact cursor retained"})
    } else {
        value
    }
}

/// Display output carries truthful status but never exports raw pending requests, objective or recovery payloads.
pub(crate) fn response(result: &TaskRunResult) -> AgentResponse {
    let mut response = result
        .final_response
        .clone()
        .unwrap_or_else(|| AgentResponse::new(""));
    response = response.with_metadata("task_run", serde_json::json!({
        "run_id":result.run.key.run_id,"revision":result.run.revision,"status":result.run.status,
        "pending_kind":result.run.pending.as_ref().map(|pending| pending.kind),
    }));
    response
}

/// An ordinary parent must not aggregate an incomplete child as success; exact task control stays with its host.
pub(crate) fn completed_child(result: &TaskRunResult) -> Result<AgentResponse> {
    if result.run.status != TaskRunStatus::Completed {
        return Err(AgentError::Other(format!(
            "automatic child task ended with {:?}",
            result.run.status
        )));
    }
    Ok(response(result))
}
