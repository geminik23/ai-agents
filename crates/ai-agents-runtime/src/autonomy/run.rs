//! Exact task recovery data and separate runtime-facing results; no task loop is installed here.

use std::collections::HashSet;

use ai_agents_core::autonomy::*;
use ai_agents_core::{
    AgentError, AgentResponse, AgentSnapshot, ChatMessage, Result, ToolExecutionRecord,
    inspect_native_history,
};
use ai_agents_tools::{TodoItem, TodoRunBinding};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const TASK_PAYLOAD_VERSION: u32 = 1;
pub const MAX_TASK_CHECKPOINT_RECORDS: usize = 4096;

/// Exact persisted native-history identity, separate from display and actor-memory projections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskNativeExchange {
    pub exchange_id: String,
    pub call_ids: Vec<String>,
}

/// A continuation describes a safe boundary, never a saved Rust future.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskContinuation {
    BetweenTurns,
    Suspended {
        request_id: String,
        turn_id: String,
        user_message_committed: bool,
        finalized: bool,
        skill: Option<TaskSkillCursor>,
        batch: Option<TaskToolBatchCursor>,
    },
}

/// Completed skill results are exact inputs to later templates, not truncated previews.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSkillCursor {
    pub skill_id: String,
    pub next_step: usize,
    pub results: Vec<Value>,
}

/// Completed calls are retained in order so a later driver cannot replay prior effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskToolBatchCursor {
    pub messages: Vec<ChatMessage>,
    pub call_ids: Vec<String>,
    pub next_call: usize,
    pub completed_results: Vec<Value>,
}

/// Runtime state and native history belong to the same outer checkpoint revision.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRuntimeCheckpoint {
    pub snapshot: AgentSnapshot,
    pub native_exchanges: Vec<TaskNativeExchange>,
    pub continuation: TaskContinuation,
}

impl TaskRuntimeCheckpoint {
    /// Builds a between-turn checkpoint only from complete, structurally valid native history.
    pub fn between_turns(snapshot: AgentSnapshot) -> Result<Self> {
        let inspection = inspect_native_history(&snapshot.memory.messages)
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
        if inspection.exchanges().iter().any(|e| !e.is_complete()) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let native_exchanges = inspection
            .exchanges()
            .iter()
            .map(|e| TaskNativeExchange {
                exchange_id: e.state().exchange_id().to_owned(),
                call_ids: e.call_ids().to_vec(),
            })
            .collect();
        Ok(Self {
            snapshot,
            native_exchanges,
            continuation: TaskContinuation::BetweenTurns,
        })
    }

    // Includes nested cursors in the shared record cap so exact continuation results cannot grow unnoticed.
    fn record_count(&self) -> usize {
        let mut count = self.snapshot.memory.messages.len()
            + self.native_exchanges.len()
            + self
                .native_exchanges
                .iter()
                .map(|e| e.call_ids.len())
                .sum::<usize>()
            + self
                .snapshot
                .state_machine
                .as_ref()
                .map_or(0, |s| s.history.len())
            + self.snapshot.spawned_agents.as_ref().map_or(0, Vec::len);
        if let TaskContinuation::Suspended { skill, batch, .. } = &self.continuation {
            if let Some(skill) = skill {
                count += skill.results.len();
            }
            if let Some(batch) = batch {
                count +=
                    batch.messages.len() + batch.call_ids.len() + batch.completed_results.len();
            }
        }
        count
    }

    /// Checks history identities and cursor ordering before any live runtime can be mutated.
    pub fn validate(&self) -> Result<()> {
        let inspection = inspect_native_history(&self.snapshot.memory.messages)
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
        let actual: Vec<_> = inspection
            .exchanges()
            .iter()
            .map(|e| TaskNativeExchange {
                exchange_id: e.state().exchange_id().to_owned(),
                call_ids: e.call_ids().to_vec(),
            })
            .collect();
        if actual != self.native_exchanges {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        match &self.continuation {
            TaskContinuation::BetweenTurns => {
                if inspection.exchanges().iter().any(|e| !e.is_complete()) {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
            }
            TaskContinuation::Suspended {
                request_id,
                turn_id,
                finalized,
                skill,
                batch,
                ..
            } => {
                if request_id.is_empty() || turn_id.is_empty() || *finalized {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
                if let Some(skill) = skill
                    && (skill.skill_id.is_empty() || skill.results.len() != skill.next_step)
                {
                    return Err(TaskRunStorageError::InvalidCheckpoint.into());
                }
                validate_suspended_history(&self.snapshot.memory.messages, batch.as_ref())?;
                if let Some(batch) = batch {
                    let ids: HashSet<_> = batch.call_ids.iter().collect();
                    if batch.next_call >= batch.call_ids.len()
                        || batch.completed_results.len() != batch.next_call
                        || ids.len() != batch.call_ids.len()
                        || batch.call_ids.iter().any(String::is_empty)
                    {
                        return Err(TaskRunStorageError::InvalidCheckpoint.into());
                    }
                    validate_suspended_history(&batch.messages, Some(batch))?;
                    validate_native_batch_authority(&self.snapshot.memory.messages, batch)?;
                }
            }
        }
        Ok(())
    }
}

// Only the final exchange of the current user turn may be incomplete, with results matching the exact batch cursor.
fn validate_suspended_history(
    messages: &[ChatMessage],
    batch: Option<&TaskToolBatchCursor>,
) -> Result<()> {
    let history =
        inspect_native_history(messages).map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
    for (index, exchange) in history.exchanges().iter().enumerate() {
        if exchange.is_complete() {
            continue;
        }
        let batch = batch.ok_or(TaskRunStorageError::InvalidCheckpoint)?;
        let latest_user = messages
            .iter()
            .rposition(|message| message.role == ai_agents_core::Role::User);
        if index + 1 != history.exchanges().len()
            || latest_user.is_some_and(|user| user > exchange.message_start())
            || exchange.call_ids() != batch.call_ids
            || batch.next_call > batch.call_ids.len()
            || exchange.result_ids() != &batch.call_ids[..batch.next_call]
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    Ok(())
}

// Selects the exact current native batch; a later user or assistant message cannot masquerade as a pending batch.
fn active_batch_segment<'a>(
    messages: &'a [ChatMessage],
    batch: &TaskToolBatchCursor,
) -> Result<Option<&'a [ChatMessage]>> {
    for (index, message) in messages.iter().enumerate().rev() {
        if message.role != ai_agents_core::Role::Assistant {
            continue;
        }
        let calls = ai_agents_core::decode_native_tool_call_markers(&message.content)
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
        if let Some(calls) = calls
            && calls
                .calls()
                .iter()
                .map(|call| &call.id)
                .eq(batch.call_ids.iter())
        {
            if messages[index + 1..].iter().any(|m| {
                matches!(
                    m.role,
                    ai_agents_core::Role::User | ai_agents_core::Role::Assistant
                )
            }) {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            return Ok(Some(&messages[index..]));
        }
    }
    Ok(None)
}

// Snapshot, provider continuation and completed-result ledger must agree exactly rather than only sharing call IDs.
fn validate_native_batch_authority(
    snapshot: &[ChatMessage],
    batch: &TaskToolBatchCursor,
) -> Result<()> {
    let stored = active_batch_segment(snapshot, batch)?;
    let pending = active_batch_segment(&batch.messages, batch)?;
    if let (Some(stored), Some(pending)) = (stored, pending) {
        if serde_json::to_value(stored)? != serde_json::to_value(pending)? {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        let mut results = Vec::new();
        let mut ids = Vec::new();
        for message in pending {
            if !matches!(
                message.role,
                ai_agents_core::Role::Tool | ai_agents_core::Role::Function
            ) {
                continue;
            }
            let decoded = ai_agents_core::decode_native_tool_result_markers(&message.content)
                .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
            if let Some(decoded) = decoded {
                for result in decoded {
                    let (id, _, output) = result.into_parts();
                    ids.push(id);
                    results.push(output);
                }
            }
        }
        if ids != batch.call_ids[..batch.next_call] || results != batch.completed_results {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    } else if stored.is_some() || pending.is_some() {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    Ok(())
}

// An unchanged suspended turn must retain completed cursor prefixes so already committed effects cannot be replayed.
fn validate_continuation_successor(
    next: &TaskRuntimeCheckpoint,
    previous: &TaskRuntimeCheckpoint,
    retired: &[String],
) -> Result<()> {
    let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
    if let (TaskContinuation::Suspended { request_id, .. }, TaskContinuation::BetweenTurns) =
        (&previous.continuation, &next.continuation)
        && !retired.contains(request_id)
    {
        return Err(invalid());
    }
    if let (
        TaskContinuation::Suspended {
            turn_id: old_turn,
            user_message_committed: old_committed,
            skill: old_skill,
            batch: old_batch,
            ..
        },
        TaskContinuation::Suspended {
            turn_id: new_turn,
            user_message_committed: new_committed,
            skill: new_skill,
            batch: new_batch,
            ..
        },
    ) = (&previous.continuation, &next.continuation)
    {
        if old_turn != new_turn || (old_committed != new_committed) {
            return Err(invalid());
        }
        if let Some(old) = old_skill {
            let new = new_skill.as_ref().ok_or_else(invalid)?;
            if old.skill_id != new.skill_id
                || new.next_step < old.next_step
                || serde_json::to_value(&new.results[..old.next_step])?
                    != serde_json::to_value(&old.results)?
            {
                return Err(invalid());
            }
        }
        if let Some(old) = old_batch {
            let new = new_batch.as_ref().ok_or_else(invalid)?;
            let old_native = active_batch_segment(&old.messages, old)?;
            let new_native = active_batch_segment(&new.messages, new)?;
            if let Some(old_native) = old_native {
                let new_native = new_native.ok_or_else(invalid)?;
                if new_native.len() < old_native.len()
                    || serde_json::to_value(&new_native[..old_native.len()])?
                        != serde_json::to_value(old_native)?
                {
                    return Err(invalid());
                }
            }
            if old.call_ids != new.call_ids
                || new.next_call < old.next_call
                || serde_json::to_value(&new.completed_results[..old.next_call])?
                    != serde_json::to_value(&old.completed_results)?
            {
                return Err(invalid());
            }
        }
    }
    Ok(())
}

// Removing a request retires its identity even during recovery; changing a reviewed action requires a fresh revision-bound request.
fn validate_pending_successor(
    next: Option<&TaskPendingRequest>,
    previous: Option<&TaskPendingRequest>,
    retired: &[String],
    revision: u64,
) -> Result<()> {
    if let Some(old) = previous {
        if let Some(new) = next
            && new.id == old.id
        {
            if serde_json::to_value(new)? != serde_json::to_value(old)? {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            return Ok(());
        }
        if !retired.contains(&old.id) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
    }
    if let Some(new) = next
        && new.issued_revision != revision
    {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    Ok(())
}

/// Pending response identity is bound to the revision at which the request was issued.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPendingRequest {
    pub id: String,
    pub issued_revision: u64,
    pub kind: TaskPendingKind,
    pub reviewed_action: Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskPendingKind {
    Approval,
    UserQuestion,
    ExternalCondition,
    ValidationObservation,
}

/// Actual effect outcome, including the crash window in which automatic replay is forbidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskEffectState {
    Reserved,
    Dispatched,
    /// A framework-owned message cursor has acknowledged quiescence without completing or refunding this invocation.
    Suspended,
    Completed,
    Uncertain,
}

/// Durable attempt, priced amount and operation-aware target declarations share one identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskReservation {
    pub id: String,
    pub state: TaskEffectState,
    pub reserved_micro_usd: u64,
    pub charged_micro_usd: u64,
    pub write_targets: Vec<String>,
    pub result: Option<Value>,
}

/// Resolved numeric ceilings persist independently from raw settings and are not capability proofs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRunLimits {
    pub max_turns: u32,
    pub max_active_time_seconds: u64,
    pub max_wall_time_seconds: Option<u64>,
    pub max_llm_calls: u32,
    pub max_tool_calls: u32,
    pub max_command_calls: u32,
    pub max_micro_usd: Option<u64>,
    pub max_declared_write_paths: Option<u32>,
}

impl TaskRunLimits {
    /// Copies final host-clamped values without implying that provider or write admission is installed.
    pub fn from_profile(profile: &super::EffectiveAutonomyProfile) -> Self {
        Self {
            max_turns: profile.max_turns,
            max_active_time_seconds: profile.max_active_time_seconds,
            max_wall_time_seconds: profile.max_wall_time_seconds,
            max_llm_calls: profile.max_llm_calls,
            max_tool_calls: profile.max_tool_calls,
            max_command_calls: profile.max_command_calls,
            max_micro_usd: profile.max_cost_usd.as_ref().map(UsdAmount::micro_usd),
            max_declared_write_paths: profile.max_declared_write_paths,
        }
    }
}

/// Counters are cumulative across pauses, children, retries and objective revisions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCounters {
    pub turns: u64,
    pub llm_attempts: u64,
    pub tool_attempts: u64,
    pub command_attempts: u64,
    pub continuations: u64,
    pub charged_micro_usd: u64,
}

/// Active allowance is independent of original expiry; interrupted intervals remain explicit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskClocks {
    pub active_millis: u64,
    pub active_interval_started_at: Option<DateTime<Utc>>,
    pub interrupted_interval_millis: u64,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Immutable adapter binding data is checkpointed with its exact opaque recovery state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskAdapterCheckpoint {
    pub id: String,
    pub adapter: String,
    pub contract_version: u32,
    pub config: Value,
    pub state: Value,
}

/// Scoped evidence records retain eligibility identity; gate evaluation is implemented separately.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskEvidenceRecord {
    pub run_id: String,
    pub sequence: u64,
    pub stage: Option<String>,
    pub attempt: Option<String>,
    pub origin: String,
    pub data: Value,
}

/// Shared executor evidence with explicit run, stage and invocation-attempt eligibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskToolEvidence {
    pub run_id: String,
    pub sequence: u64,
    pub stage: Option<String>,
    pub attempt: Option<String>,
    pub origin: String,
    pub record: ToolExecutionRecord,
}

/// Exact evidence is not exposed by storage listing or implicitly converted into a passing gate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRunEvidence {
    pub records: Vec<TaskEvidenceRecord>,
    pub tool_calls: Vec<TaskToolEvidence>,
}

/// Persisted progress decisions survive resume without resetting stagnation or replanning limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskProgressCheckpoint {
    pub observation_sequence: u64,
    pub cycles_without_progress: u64,
    pub replans: u64,
    pub high_water_marks: Value,
    pub history: Vec<TaskEvidenceRecord>,
}

/// The list is a checkpoint copy of the canonical store, not another live todo authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskTodoCheckpoint {
    pub binding: TodoRunBinding,
    pub items: Vec<TodoItem>,
}

/// Child state remains associated with parent ownership instead of becoming a separately budgeted run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskChildCheckpoint {
    pub child_id: String,
    pub parent_run_id: String,
    pub config_identity: String,
    pub runtime: TaskRuntimeCheckpoint,
    pub pending: Option<TaskPendingRequest>,
    pub result: Option<Value>,
}

/// Versioned runtime payload; core storage treats it as an opaque, size-bounded object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCheckpointPayload {
    pub version: u32,
    pub objective: String,
    pub objective_revision: u64,
    pub profile: Option<String>,
    pub config_identity: String,
    pub settings: AutonomyProfile,
    pub limits: TaskRunLimits,
    pub stage: Option<String>,
    pub stage_attempt: Option<String>,
    pub pause_reason: Option<String>,
    pub stop_reason: Option<String>,
    pub last_response: Option<String>,
    pub controller_state: Value,
    pub runtime: TaskRuntimeCheckpoint,
    pub pending: Option<TaskPendingRequest>,
    pub consumed_request_ids: Vec<String>,
    pub counters: TaskCounters,
    pub clocks: TaskClocks,
    pub reservations: Vec<TaskReservation>,
    pub declared_write_targets: Vec<String>,
    pub adapters: Vec<TaskAdapterCheckpoint>,
    pub progress: TaskProgressCheckpoint,
    pub evidence: TaskRunEvidence,
    pub todos: Option<TaskTodoCheckpoint>,
    pub children: Vec<TaskChildCheckpoint>,
}

impl TaskCheckpointPayload {
    /// Creates initial controller data without granting task execution or resetting a resumed run.
    pub fn new(
        objective: String,
        config_identity: String,
        profile: &super::EffectiveAutonomyProfile,
        runtime: TaskRuntimeCheckpoint,
    ) -> Self {
        Self {
            version: TASK_PAYLOAD_VERSION,
            objective,
            objective_revision: 0,
            profile: profile.profile.clone(),
            config_identity,
            settings: profile.settings.clone(),
            limits: TaskRunLimits::from_profile(profile),
            stage: None,
            stage_attempt: None,
            pause_reason: None,
            stop_reason: None,
            last_response: None,
            controller_state: Value::Null,
            runtime,
            pending: None,
            consumed_request_ids: vec![],
            counters: TaskCounters::default(),
            clocks: TaskClocks::default(),
            reservations: vec![],
            declared_write_targets: vec![],
            adapters: vec![],
            progress: TaskProgressCheckpoint::default(),
            evidence: TaskRunEvidence::default(),
            todos: None,
            children: vec![],
        }
    }

    /// Validates identity, size/count bounds, pending ownership and native history before restoration.
    pub fn validate(&self, envelope: &TaskRunSnapshot, config_identity: &str) -> Result<()> {
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        if self.version != TASK_PAYLOAD_VERSION
            || self.objective.trim().is_empty()
            || self.config_identity.is_empty()
            || self.config_identity != config_identity
            || self.runtime.snapshot.agent_id != envelope.key.agent_id
        {
            return Err(invalid());
        }
        if [
            self.limits.max_turns,
            self.limits.max_llm_calls,
            self.limits.max_tool_calls,
            self.limits.max_command_calls,
        ]
        .contains(&0)
            || self.limits.max_active_time_seconds == 0
            || self.limits.max_wall_time_seconds == Some(0)
            || self.limits.max_micro_usd == Some(0)
            || self.limits.max_declared_write_paths == Some(0)
        {
            return Err(invalid());
        }
        self.runtime.validate()?;
        let consumed: HashSet<_> = self.consumed_request_ids.iter().collect();
        if consumed.len() != self.consumed_request_ids.len()
            || consumed.iter().any(|id| id.is_empty())
        {
            return Err(invalid());
        }
        let mut pending_ids = HashSet::new();
        let mut count = self.evidence.records.len()
            + self.evidence.tool_calls.len()
            + self.progress.history.len()
            + self.reservations.len()
            + self.adapters.len()
            + self.children.len()
            + self.declared_write_targets.len()
            + self.runtime.record_count()
            + self.consumed_request_ids.len();
        if let Some(todos) = &self.todos {
            count += todos.items.len();
            if todos.binding.agent_id != envelope.key.agent_id
                || todos.binding.run_id != envelope.key.run_id
                || todos.binding.token.is_empty()
                || todos.binding.objective_revision != self.objective_revision
            {
                return Err(invalid());
            }
            super::todo::validate_todo_items(&todos.items)?;
        }
        if let Some(pending) = &self.pending {
            if pending.id.is_empty()
                || pending.issued_revision > envelope.revision
                || consumed.contains(&pending.id)
                || !pending_ids.insert(&pending.id)
            {
                return Err(invalid());
            }
            if let TaskContinuation::Suspended { request_id, .. } = &self.runtime.continuation
                && request_id != &pending.id
            {
                return Err(invalid());
            }
        } else if matches!(
            self.runtime.continuation,
            TaskContinuation::Suspended { .. }
        ) {
            return Err(invalid());
        }
        if matches!(
            self.runtime.continuation,
            TaskContinuation::Suspended { .. }
        ) && self.pause_reason.as_deref() != Some("child_interaction")
            && let Some(adapter) = self
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.tool_batch")
        {
            let batch: super::TaskBatchState = serde_json::from_value(adapter.state.clone())?;
            batch.validate_checkpoint(self)?;
        }
        if envelope.status == TaskRunStatus::Paused
            && (matches!(
                self.runtime.continuation,
                TaskContinuation::Suspended { batch: None, .. }
            ) || self.pause_reason.as_deref() == Some("child_interaction"))
            && let Some(adapter) = self
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.group")
        {
            let group: super::TaskGroupState = serde_json::from_value(adapter.state.clone())?;
            group.validate_checkpoint(self)?;
        }
        if envelope.status.is_terminal() && self.pending.is_some() {
            return Err(invalid());
        }
        super::message::validate_suspended_attempts(self, envelope.status)?;
        let uncertain = self.reservations.iter().any(|r| {
            matches!(
                r.state,
                TaskEffectState::Dispatched | TaskEffectState::Uncertain
            )
        });
        if (uncertain || self.clocks.active_interval_started_at.is_some())
            && !matches!(
                envelope.status,
                TaskRunStatus::Running | TaskRunStatus::RecoveryRequired
            )
        {
            return Err(invalid());
        }
        let declared: HashSet<_> = self.declared_write_targets.iter().collect();
        if declared.len() != self.declared_write_targets.len()
            || declared.iter().any(|target| target.is_empty())
            || self.counters.command_attempts > self.counters.tool_attempts
            || self
                .clocks
                .active_interval_started_at
                .is_some_and(|start| start < envelope.created_at || start > envelope.updated_at)
            || self
                .clocks
                .expires_at
                .is_some_and(|expiry| expiry <= envelope.created_at)
        {
            return Err(invalid());
        }
        let charged = self
            .reservations
            .iter()
            .try_fold(0_u64, |sum, r| sum.checked_add(r.charged_micro_usd))
            .ok_or_else(invalid)?;
        if charged > self.counters.charged_micro_usd {
            return Err(invalid());
        }
        // Outstanding bounds remain committed across recovery; settled usage alone cannot hide in-flight overspend.
        let commitment = self
            .reservations
            .iter()
            .filter(|reservation| {
                matches!(
                    reservation.state,
                    TaskEffectState::Reserved
                        | TaskEffectState::Dispatched
                        | TaskEffectState::Suspended
                        | TaskEffectState::Uncertain
                )
            })
            .try_fold(self.counters.charged_micro_usd, |sum, reservation| {
                sum.checked_add(
                    reservation
                        .reserved_micro_usd
                        .saturating_sub(reservation.charged_micro_usd),
                )
            })
            .ok_or_else(invalid)?;
        if self
            .limits
            .max_micro_usd
            .is_some_and(|limit| commitment > limit)
            || self
                .limits
                .max_declared_write_paths
                .is_some_and(|limit| declared.len() > limit as usize)
        {
            return Err(invalid());
        }
        if envelope.status == TaskRunStatus::Completed
            && (self.counters.turns > u64::from(self.limits.max_turns)
                || self.counters.llm_attempts > u64::from(self.limits.max_llm_calls)
                || self.counters.tool_attempts > u64::from(self.limits.max_tool_calls)
                || self.counters.command_attempts > u64::from(self.limits.max_command_calls)
                || self
                    .clocks
                    .active_millis
                    .saturating_add(self.clocks.interrupted_interval_millis)
                    > self.limits.max_active_time_seconds.saturating_mul(1000)
                || self
                    .limits
                    .max_micro_usd
                    .is_some_and(|limit| self.counters.charged_micro_usd > limit)
                || self
                    .limits
                    .max_declared_write_paths
                    .is_some_and(|limit| declared.len() as u64 > u64::from(limit))
                || self
                    .clocks
                    .expires_at
                    .is_some_and(|expiry| envelope.updated_at > expiry))
        {
            return Err(invalid());
        }
        let mut ids = HashSet::new();
        for reservation in &self.reservations {
            count += reservation.write_targets.len();
            if reservation.id.is_empty()
                || !ids.insert(&reservation.id)
                || (reservation
                    .write_targets
                    .iter()
                    .any(|target| !declared.contains(target))
                    && !(reservation.state == TaskEffectState::Completed
                        && reservation
                            .result
                            .as_ref()
                            .and_then(|result| result.get("not_invoked"))
                            == Some(&Value::Bool(true))))
                || (reservation.state == TaskEffectState::Completed && reservation.result.is_none())
            {
                return Err(invalid());
            }
        }
        let mut adapters = HashSet::new();
        for adapter in &self.adapters {
            if adapter.id.is_empty()
                || adapter.adapter.is_empty()
                || adapter.contract_version == 0
                || !adapters.insert(&adapter.id)
            {
                return Err(invalid());
            }
        }
        let mut children = HashSet::new();
        for child in &self.children {
            if child.child_id.is_empty()
                || child.parent_run_id != envelope.key.run_id
                || child.config_identity.is_empty()
                || !children.insert(&child.child_id)
            {
                return Err(invalid());
            }
            child.runtime.validate()?;
            count += child.runtime.record_count();
            if let Some(pending) = &child.pending {
                if pending.id.is_empty()
                    || pending.issued_revision > envelope.revision
                    || consumed.contains(&pending.id)
                    || !pending_ids.insert(&pending.id)
                    || envelope.status.is_terminal()
                {
                    return Err(invalid());
                }
                if let TaskContinuation::Suspended { request_id, .. } = &child.runtime.continuation
                    && request_id != &pending.id
                {
                    return Err(invalid());
                }
            } else if matches!(
                child.runtime.continuation,
                TaskContinuation::Suspended { .. }
            ) {
                return Err(invalid());
            }
        }
        for record in self.evidence.records.iter().chain(&self.progress.history) {
            if record.run_id != envelope.key.run_id || record.origin.is_empty() {
                return Err(invalid());
            }
        }
        for record in &self.evidence.tool_calls {
            if record.run_id != envelope.key.run_id || record.origin.is_empty() {
                return Err(invalid());
            }
        }
        if count > MAX_TASK_CHECKPOINT_RECORDS {
            return Err(TaskRunStorageError::CheckpointTooLarge.into());
        }
        envelope.validate()
    }

    /// Rejects budget resets, configuration hot-swaps, replayed requests and lost committed effects.
    pub fn validate_successor(
        &self,
        previous: &Self,
        revision: u64,
        reconciled: bool,
    ) -> Result<()> {
        self.validate_successor_with_control(previous, revision, reconciled, false)
    }

    /// Host control may revise only explicitly authorized objective/ceilings, while every effect and resource invariant remains protected.
    pub(crate) fn validate_successor_with_control(
        &self,
        previous: &Self,
        revision: u64,
        reconciled: bool,
        host_control: bool,
    ) -> Result<()> {
        let invalid = || AgentError::from(TaskRunStorageError::InvalidCheckpoint);
        if self.config_identity != previous.config_identity
            || self.profile != previous.profile
            || (!host_control && self.limits != previous.limits)
            || serde_json::to_value(&self.settings)? != serde_json::to_value(&previous.settings)?
            || (!host_control && self.clocks.expires_at != previous.clocks.expires_at)
            || self.clocks.active_millis < previous.clocks.active_millis
            || self.clocks.interrupted_interval_millis < previous.clocks.interrupted_interval_millis
            || self.progress.replans < previous.progress.replans
            || self.progress.observation_sequence < previous.progress.observation_sequence
            || self.objective_revision < previous.objective_revision
            || (!host_control
                && (self.objective_revision != previous.objective_revision
                    || self.objective != previous.objective))
            || (self.objective != previous.objective
                && self.objective_revision
                    != previous
                        .objective_revision
                        .checked_add(1)
                        .ok_or_else(invalid)?)
        {
            return Err(invalid());
        }
        // Only a newly acknowledged, proven non-invocation can refund its original typed slot.
        let mut refunds = [0_u64; 3];
        for prior in &previous.reservations {
            if prior.state != TaskEffectState::Dispatched {
                continue;
            }
            if let Some(next) = self.reservations.iter().find(|next| next.id == prior.id)
                && next.state == TaskEffectState::Completed
                && next
                    .result
                    .as_ref()
                    .and_then(|result| result.get("not_invoked"))
                    == Some(&Value::Bool(true))
            {
                // Message frames prove the parent implementation already dispatched; a later resume cannot relabel that attempt as unused.
                if previous.adapters.iter().any(|adapter| {
                    adapter.adapter == "runtime.delegate"
                        && adapter.state.get("dispatch").and_then(Value::as_str)
                            == Some("tool_message")
                        && adapter
                            .state
                            .get("cursor")
                            .and_then(|cursor| cursor.get("attempt"))
                            .and_then(Value::as_str)
                            == Some(prior.id.as_str())
                }) {
                    return Err(invalid());
                }
                let invocation = previous
                    .adapters
                    .iter()
                    .find(|binding| binding.id == format!("invocation:{}", prior.id))
                    .ok_or_else(invalid)?;
                match invocation.config.get("tool").and_then(Value::as_str) {
                    Some("command") => {
                        refunds[1] += 1;
                        refunds[2] += 1;
                    }
                    Some(_) => refunds[1] += 1,
                    None => refunds[0] += 1,
                }
            }
        }
        for (new, old, refund) in [
            (
                self.counters.llm_attempts,
                previous.counters.llm_attempts,
                refunds[0],
            ),
            (
                self.counters.tool_attempts,
                previous.counters.tool_attempts,
                refunds[1],
            ),
            (
                self.counters.command_attempts,
                previous.counters.command_attempts,
                refunds[2],
            ),
        ] {
            if new
                .checked_add(refund)
                .is_none_or(|accounted| accounted < old)
            {
                return Err(invalid());
            }
        }
        for (new, old) in [
            (self.counters.turns, previous.counters.turns),
            (self.counters.continuations, previous.counters.continuations),
            (
                self.counters.charged_micro_usd,
                previous.counters.charged_micro_usd,
            ),
        ] {
            if new < old {
                return Err(invalid());
            }
        }
        if previous
            .consumed_request_ids
            .iter()
            .any(|id| !self.consumed_request_ids.contains(id))
            || previous.declared_write_targets.iter().any(|target| {
                !self.declared_write_targets.contains(target)
                    && (previous
                        .reservations
                        .iter()
                        .all(|reservation| !reservation.write_targets.contains(target))
                        || self.reservations.iter().any(|reservation| {
                            reservation.write_targets.contains(target)
                                && !(reservation.state == TaskEffectState::Completed
                                    && reservation
                                        .result
                                        .as_ref()
                                        .and_then(|result| result.get("not_invoked"))
                                        == Some(&Value::Bool(true)))
                        }))
            })
        {
            return Err(invalid());
        }
        validate_pending_successor(
            self.pending.as_ref(),
            previous.pending.as_ref(),
            &self.consumed_request_ids,
            revision,
        )?;
        validate_continuation_successor(
            &self.runtime,
            &previous.runtime,
            &self.consumed_request_ids,
        )?;
        if previous.clocks.active_interval_started_at.is_some()
            && self.clocks.active_interval_started_at != previous.clocks.active_interval_started_at
            && (self.clocks.active_millis == previous.clocks.active_millis
                && self.clocks.interrupted_interval_millis
                    == previous.clocks.interrupted_interval_millis)
        {
            return Err(invalid());
        }
        for prior in &previous.adapters {
            let next = self
                .adapters
                .iter()
                .find(|adapter| adapter.id == prior.id)
                .ok_or_else(invalid)?;
            if next.adapter != prior.adapter
                || next.contract_version != prior.contract_version
                || next.config != prior.config
            {
                return Err(invalid());
            }
        }
        if self.evidence.records.len() < previous.evidence.records.len()
            || serde_json::to_value(&self.evidence.records[..previous.evidence.records.len()])?
                != serde_json::to_value(&previous.evidence.records)?
            || self.evidence.tool_calls.len() < previous.evidence.tool_calls.len()
            || serde_json::to_value(
                &self.evidence.tool_calls[..previous.evidence.tool_calls.len()],
            )? != serde_json::to_value(&previous.evidence.tool_calls)?
            || self.progress.history.len() < previous.progress.history.len()
            || serde_json::to_value(&self.progress.history[..previous.progress.history.len()])?
                != serde_json::to_value(&previous.progress.history)?
        {
            return Err(invalid());
        }
        for prior in &previous.children {
            let next = self
                .children
                .iter()
                .find(|child| child.child_id == prior.child_id)
                .ok_or_else(invalid)?;
            validate_pending_successor(
                next.pending.as_ref(),
                prior.pending.as_ref(),
                &self.consumed_request_ids,
                revision,
            )?;
            validate_continuation_successor(
                &next.runtime,
                &prior.runtime,
                &self.consumed_request_ids,
            )?;
            if next.config_identity != prior.config_identity
                || (prior.result.is_some() && next.result != prior.result)
            {
                return Err(invalid());
            }
        }
        for child in &self.children {
            if !previous
                .children
                .iter()
                .any(|old| old.child_id == child.child_id)
            {
                validate_pending_successor(
                    child.pending.as_ref(),
                    None,
                    &self.consumed_request_ids,
                    revision,
                )?;
            }
        }
        if let Some(prior) = &previous.todos
            && self.todos.as_ref().is_none_or(|next| {
                next.binding != prior.binding
                    && !(host_control
                        && self.objective_revision != previous.objective_revision
                        && next.items.is_empty())
            })
        {
            return Err(invalid());
        }
        for prior in &previous.reservations {
            let next = self
                .reservations
                .iter()
                .find(|r| r.id == prior.id)
                .ok_or_else(invalid)?;
            if next.reserved_micro_usd != prior.reserved_micro_usd
                || next.write_targets != prior.write_targets
                || next.charged_micro_usd < prior.charged_micro_usd
                || (prior.state == TaskEffectState::Completed
                    && serde_json::to_value(next)? != serde_json::to_value(prior)?)
                || (matches!(
                    prior.state,
                    TaskEffectState::Dispatched | TaskEffectState::Suspended
                ) && next.state == TaskEffectState::Reserved)
                || (prior.state == TaskEffectState::Suspended
                    && next
                        .result
                        .as_ref()
                        .and_then(|result| result.get("not_invoked"))
                        == Some(&Value::Bool(true)))
                || (prior.state == TaskEffectState::Uncertain
                    && next.state != TaskEffectState::Uncertain
                    && (!reconciled || next.state != TaskEffectState::Completed))
            {
                return Err(invalid());
            }
        }
        Ok(())
    }

    /// Encodes exact data only after checking the final envelope that will be persisted.
    pub fn bind(self, mut envelope: TaskRunSnapshot) -> Result<TaskRunSnapshot> {
        envelope.payload = serde_json::to_value(&self)?;
        self.validate(&envelope, &self.config_identity)?;
        Ok(envelope)
    }
}

/// Runtime-facing task view is not the core storage ABI and may contain sensitive objective text.
#[derive(Debug, Clone)]
pub struct TaskRun {
    pub key: TaskRunKey,
    pub status: TaskRunStatus,
    pub revision: u64,
    pub objective: String,
    pub actor_id: Option<String>,
    pub profile: Option<String>,
    pub current_stage: Option<String>,
    pub pause_reason: Option<String>,
    /// Acknowledged interaction data is host-visible and may contain sensitive review context.
    pub pending: Option<TaskPendingRequest>,
    pub stop_reason: Option<String>,
    pub todos: Vec<TodoItem>,
    pub progress: TaskProgressCheckpoint,
    pub limits: TaskRunLimits,
    pub counters: TaskCounters,
    pub clocks: TaskClocks,
    pub evidence: TaskRunEvidence,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl TaskRun {
    /// Converts validated recovery data into a runtime-facing view without making it a storage ABI or safe display summary.
    pub fn from_checkpoint(snapshot: &TaskRunSnapshot, config_identity: &str) -> Result<Self> {
        snapshot.validate()?;
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())
            .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
        payload.validate(snapshot, config_identity)?;
        Ok(Self {
            key: snapshot.key.clone(),
            status: snapshot.status,
            revision: snapshot.revision,
            objective: payload.objective,
            actor_id: snapshot.actor_id.clone(),
            profile: payload.profile,
            current_stage: payload.stage,
            pause_reason: payload.pause_reason,
            pending: payload.pending,
            stop_reason: payload.stop_reason,
            todos: payload.todos.map_or_else(Vec::new, |todos| todos.items),
            progress: payload.progress,
            limits: payload.limits,
            counters: payload.counters,
            clocks: payload.clocks,
            evidence: payload.evidence,
            created_at: snapshot.created_at,
            updated_at: snapshot.updated_at,
        })
    }
}

/// A task result describes the acknowledged run status rather than treating one turn response as completion.
#[derive(Debug, Clone)]
pub struct TaskRunResult {
    pub run: TaskRun,
    pub final_response: Option<AgentResponse>,
}
