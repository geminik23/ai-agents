//! Foreground task events project existing execution boundaries without detaching work or granting authority.

use super::*;
use ai_agents_core::{AgentError, Result};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Weak},
    task::{Context, Poll},
};
use tokio::sync::mpsc;

/// Maximum serialized display envelope size, independent of exact recovery checkpoint limits.
pub const MAX_TASK_EVENT_BYTES: usize = 4096;

/// Bounds delivery memory independently of exact task recovery data.
#[derive(Debug, Clone, Copy)]
pub struct TaskStreamConfig {
    pub capacity: usize,
}
impl Default for TaskStreamConfig {
    /// A small bounded queue applies backpressure without a detached producer.
    fn default() -> Self {
        Self { capacity: 32 }
    }
}

/// Event categories describe runtime facts, never commands or executable checkpoints.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskRunEventKind {
    TaskStarted,
    TaskResumed,
    StageStarted,
    StageCompleted,
    TodoUpdated,
    ArtifactObserved,
    ToolStarted,
    ToolCompleted,
    ValidationStarted,
    ValidationCompleted,
    CompletionGateChecked,
    ProgressObserved,
    InterventionSelected,
    ReplanCompleted,
    ApprovalRequired,
    HostContinueRequired,
    CapacityRequired,
    ValidationObservationRequired,
    UserQuestion,
    ExternalConditionRequired,
    ContinuationForced,
    TurnCompleted,
    StateTransition,
    HostControlApplied,
    TaskPaused,
    TaskCompleted,
    TaskIncomplete,
    TaskLimitReached,
    TaskRecoveryRequired,
    TaskFailed,
    TaskCancelled,
    OperationError,
}

/// A run cursor orders live delivery by coordinating checkpoint revision, then by event index within that revision.
/// Fresh execution/control claims advance the revision; gaps are permitted and this is not durable event replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskEventCursor {
    pub revision: u64,
    pub index: u64,
}

/// A bounded default-redacted event; exact pending data and results remain available to the trusted host through the stream result.
/// Revision identifies the source ACK; cursor is its delivery high-water and sequence is local to operation_id.
/// Snapshot redisplay is not durable event replay and never authorizes new execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRunEvent {
    pub operation_id: String,
    pub sequence: u64,
    pub cursor: Option<TaskEventCursor>,
    pub run_id: Option<String>,
    pub revision: Option<u64>,
    pub kind: TaskRunEventKind,
    pub data: Value,
}

/// Explicit operations share existing runner APIs, including paused controls that do not execute another model turn.
#[derive(Debug)]
pub enum TaskStreamOperation {
    Run {
        objective: String,
        profile: Option<String>,
    },
    Skill {
        objective: String,
        skill_id: String,
        profile: Option<String>,
    },
    Resume {
        run_id: String,
        revision: u64,
        input: TaskResumeInput,
    },
    Cancel {
        run_id: String,
        revision: u64,
    },
    Stop {
        run_id: String,
        revision: u64,
        reason: String,
    },
    ModifyObjective {
        run_id: String,
        revision: u64,
        objective: String,
        reason: String,
    },
    ReviseLimits {
        run_id: String,
        revision: u64,
        limits: TaskRunLimits,
        reason: String,
    },
    ExternalUnavailable {
        run_id: String,
        revision: u64,
        request_id: String,
        reason: String,
    },
    ControllerUnavailable {
        run_id: String,
        revision: u64,
        request_id: String,
        reason: String,
    },
    RestorationReview {
        run_id: String,
        revision: u64,
        review_id: String,
        decision: ai_agents_hitl::ApprovalResult,
    },
}

pub(crate) struct EventSink {
    sender: mpsc::Sender<TaskRunEvent>,
    operation: String,
    delivery: tokio::sync::Mutex<(u64, Option<TaskEventCursor>)>,
    execution: parking_lot::Mutex<Weak<RunExecution>>,
    owners: parking_lot::Mutex<Vec<Weak<RunOwner>>>,
    active: std::sync::atomic::AtomicBool,
}
tokio::task_local! { static EVENTS: Arc<EventSink>; }

/// Captures delivery only from the owning operation or an explicitly inherited execution, never a global subscriber.
pub(crate) fn current_sink() -> Option<Arc<EventSink>> {
    EVENTS
        .try_with(Clone::clone)
        .ok()
        .or_else(|| current_execution().and_then(|e| e.owner.events.read().clone()))
        .filter(|sink| sink.active.load(std::sync::atomic::Ordering::Acquire))
}

impl EventSink {
    /// Tracks actual live owners so interrupted delivery cannot detach protected work.
    pub(crate) fn owner(&self, owner: &Arc<RunOwner>) {
        self.owners.lock().push(Arc::downgrade(owner));
    }
    /// Only the first coordinating execution owns awaited stream cancellation; children retain their existing parent custody.
    pub(crate) fn execution(&self, execution: &Arc<RunExecution>) {
        let mut slot = self.execution.lock();
        if slot.upgrade().is_none() {
            *slot = Arc::downgrade(execution);
        }
    }
    /// Sends only bounded projections outside ledger locks; receiver loss never creates successful delivery.
    async fn emit(
        &self,
        run_id: Option<String>,
        revision: Option<u64>,
        kind: TaskRunEventKind,
        data: Value,
    ) {
        let mut delivery = self.delivery.lock().await;
        let sequence = delivery.0;
        delivery.0 += 1;
        let cursor = revision.map(|revision| TaskEventCursor {
            // Concurrent producers can sample an older ACK before waiting for delivery serialization.
            revision: delivery
                .1
                .map_or(revision, |old| old.revision.max(revision)),
            index: delivery
                .1
                .filter(|old| old.revision >= revision)
                .map_or(0, |old| old.index + 1),
        });
        delivery.1 = cursor;
        let mut event = TaskRunEvent {
            operation_id: self.operation.clone(),
            sequence,
            cursor,
            run_id,
            revision,
            kind,
            data,
        };
        if serde_json::to_vec(&event).map_or(true, |bytes| bytes.len() > MAX_TASK_EVENT_BYTES) {
            event.data = json!({"display_unavailable":"envelope_limit"});
            if serde_json::to_vec(&event).map_or(true, |bytes| bytes.len() > MAX_TASK_EVENT_BYTES) {
                event.run_id = None;
            }
        }
        // Producers supply bounded structural fields and redacted text, never exact recovery payloads.
        if self.sender.send(event).await.is_err() {
            self.abandon();
        }
    }
    /// Drop blocks further admission synchronously but cannot certify cleanup or change durable terminal facts.
    fn abandon(&self) {
        if !self.active.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        if let Some(execution) = self.execution.lock().upgrade() {
            execution
                .cancellation
                .store(true, std::sync::atomic::Ordering::Release);
        }
        for owner in self.owners.lock().iter().filter_map(Weak::upgrade) {
            owner.abandon();
        }
    }
}

/// Text is summarized before queueing, including host-defined identifiers and error messages.
pub(crate) fn safe_text(text: &str) -> Value {
    ai_agents_observability::Redactor::new(Default::default()).redact_text(text)
}

/// Source categories remain machine-readable while host-selected skill/state/fallback identifiers stay private.
pub(crate) fn safe_source(source: &ai_agents_core::ToolCallSource) -> Value {
    use ai_agents_core::ToolCallSource;
    match source {
        ToolCallSource::Model => json!({"source":"model"}),
        ToolCallSource::Skill {
            skill_id,
            step_index,
        } => json!({"source":"skill","skill":safe_text(skill_id),"step_index":step_index}),
        ToolCallSource::StateAction {
            state,
            action_index,
        } => {
            json!({"source":"state_action","state":state.as_deref().map(safe_text),"action_index":action_index})
        }
        ToolCallSource::Plan { step_index } => json!({"source":"plan","step_index":step_index}),
        ToolCallSource::Orchestration => json!({"source":"orchestration"}),
        ToolCallSource::Spawner => json!({"source":"spawner"}),
        ToolCallSource::Fallback { original_tool } => {
            json!({"source":"fallback","original_tool":safe_text(original_tool)})
        }
        ToolCallSource::Task => json!({"source":"task"}),
        ToolCallSource::Manual => json!({"source":"manual"}),
        ToolCallSource::EvalFixture => json!({"source":"eval_fixture"}),
    }
}

/// Structured numeric observations retain their meaning while all host text is redacted and display growth is bounded.
pub(crate) fn safe_value(value: &Value) -> Value {
    let mut nodes = 256;
    let projected = project_untrusted(value, 0, &mut nodes);
    if serde_json::to_vec(&projected).map_or(true, |bytes| bytes.len() > 2048) {
        json!({"display_unavailable":"payload_limit"})
    } else {
        projected
    }
}

// Domain keys and reserved-looking fields are untrusted text, not a pre-redacted observability envelope.
// Encode dynamic map keys as redacted name/value entries so numeric facts survive without leaking identifiers.
fn project_untrusted(value: &Value, depth: usize, nodes: &mut usize) -> Value {
    if depth > 16 || *nodes == 0 {
        return json!({"display_unavailable":"structure_limit"});
    }
    *nodes -= 1;
    match value {
        Value::String(text) => safe_text(text),
        Value::Array(items) if items.len() > 256 => json!({"display_unavailable":"structure_limit"}),
        Value::Object(items) if items.len() > 256 => json!({"display_unavailable":"structure_limit"}),
        Value::Array(items) => Value::Array(items.iter().take(256).map(|item|project_untrusted(item, depth + 1, nodes)).collect()),
        Value::Object(items) => Value::Array(items.iter().take(256).map(|(name, value)|json!({"name":safe_text(name),"value":project_untrusted(value, depth + 1, nodes)})).collect()),
        other => other.clone(),
    }
}

/// Existing execution supplies authority and revision; events cannot feed back into task completion.
/// Lazy projection and future erasure keep optional event allocation and polling frames out of deeply nested ordinary task execution.
pub(crate) fn emit<'a>(
    execution: &'a RunExecution,
    kind: TaskRunEventKind,
    data: impl FnOnce() -> Value + Send + 'a,
) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
    let sink = current_sink()
        .or_else(|| execution.owner.events.read().clone())
        .filter(|sink| sink.active.load(std::sync::atomic::Ordering::Acquire));
    Box::pin(async move {
        if let Some(sink) = sink {
            let root = sink.execution.lock().upgrade();
            let coordinating = root.as_deref().unwrap_or(execution);
            sink.emit(
                Some(coordinating.run_id.clone()),
                Some(
                    coordinating
                        .revision
                        .load(std::sync::atomic::Ordering::Acquire),
                ),
                kind,
                data(),
            )
            .await;
        }
    })
}

/// Validation results are projected only from the shared post-ACK driver boundary, including retained and reconstructed reentry.
pub(crate) fn validation_result(
    result: &ValidationResult,
    replayed: bool,
) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
    Box::pin(async move {
        if let Some(execution) = current_execution() {
            emit(&execution, TaskRunEventKind::ValidationCompleted, || json!({"check":safe_text(&result.check_id),"attempt":result.identity.attempt.as_deref().map(safe_text),"outcome":result.outcome,"replayed":replayed})).await;
        }
    })
}

/// Pull-driven foreground operation; no worker survives this value, and paused owners remain with the retained runner.
/// EOF without result is never success. take_result is trusted-host access and is not a redacted display/export API.
pub struct TaskRunStream<'a> {
    future: Option<Pin<Box<dyn Future<Output = Result<TaskRunResult>> + Send + 'a>>>,
    receiver: mpsc::Receiver<TaskRunEvent>,
    sink: Arc<EventSink>,
    result: Option<Result<TaskRunResult>>,
    finished: bool,
    polled: bool,
}
impl Stream for TaskRunStream<'_> {
    type Item = TaskRunEvent;
    /// Polls the same operation future under its captured task locals, with bounded delivery and no detached execution.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.polled = true;
        if !this.finished
            && let Some(future) = &mut this.future
            && let Poll::Ready(result) = future.as_mut().poll(cx)
        {
            this.result = Some(result);
            this.future = None;
            this.finished = true;
            // A cooperative Pending is not empty/EOF, even after the producer completed.
            // Closing admits no new sends but retains the complete buffered terminal/control suffix.
            this.receiver.close();
        }
        this.receiver.poll_recv(cx)
    }
}
impl TaskRunStream<'_> {
    /// Drives one event without requiring a host to import StreamExt; it has exactly the same foreground lifetime.
    pub async fn next_event(&mut self) -> Option<TaskRunEvent> {
        self.next().await
    }

    /// Returns the exact host result only after the underlying operation has completed; events remain independently redacted.
    pub fn take_result(&mut self) -> Option<Result<TaskRunResult>> {
        self.result.take()
    }
    /// Requests run cancellation and drives acknowledged settlement rather than detaching on a cancellation receipt.
    /// Before first poll nothing was admitted. Interrupted pre-execution/control work retains the existing recovery guards.
    pub async fn close(mut self) -> Option<Result<TaskRunResult>> {
        if self.finished {
            return self.take_result();
        }
        if !self.polled {
            return None;
        }
        if !self.sink.active.load(std::sync::atomic::Ordering::Acquire) {
            while self.next().await.is_some() {}
            return self.take_result();
        }
        let execution = self.sink.execution.lock().upgrade();
        if let Some(execution) = execution {
            // Cancellation can wait for a journal mutex owned by this very foreground future.
            // Poll both until one completes; terminal publication may legitimately win the cancellation race.
            let cancellation = execution.request_cancel();
            tokio::pin!(cancellation);
            let requested = tokio::select! {
                biased;
                result = &mut cancellation => Some(result),
                _ = async { while self.next().await.is_some() {} } => None,
            };
            if let Some(Err(error)) = requested
                && self.sink.active.load(std::sync::atomic::Ordering::Acquire)
                && execution.acknowledged_snapshot().status == TaskRunStatus::Running
            {
                return Some(Err(error));
            }
            while self.next().await.is_some() {}
            self.take_result()
        } else {
            self.sink.abandon();
            Some(Err(AgentError::Other("task stream interrupted before execution acknowledgement; inspect retained ownership before recovery".into())))
        }
    }
}
impl Drop for TaskRunStream<'_> {
    /// A returned pause or terminal result has handed custody back; only interrupted operations abandon their live owners.
    fn drop(&mut self) {
        if !self.finished {
            self.sink.abandon();
        }
    }
}

impl AutonomyRunner {
    /// Streams the existing explicit task controller without enabling YAML/state/skill automatic entry.
    pub fn run_stream<'a>(
        &'a self,
        objective: &str,
        selected: Option<&str>,
    ) -> Result<TaskRunStream<'a>> {
        self.stream_operation(
            TaskStreamOperation::Run {
                objective: objective.into(),
                profile: selected.map(str::to_owned),
            },
            Default::default(),
        )
    }
    /// Streams one retained response under its exact revision and original execution authority.
    pub fn resume_stream(
        &self,
        run_id: &str,
        revision: u64,
        input: TaskResumeInput,
    ) -> Result<TaskRunStream<'_>> {
        self.stream_operation(
            TaskStreamOperation::Resume {
                run_id: run_id.into(),
                revision,
                input,
            },
            Default::default(),
        )
    }
    /// Streams an explicitly admitted operation; bounded display output never replaces existing authentication or conditional storage.
    pub fn stream_operation(
        &self,
        operation: TaskStreamOperation,
        config: TaskStreamConfig,
    ) -> Result<TaskRunStream<'_>> {
        let host_control = !matches!(
            &operation,
            TaskStreamOperation::Run { .. }
                | TaskStreamOperation::Skill { .. }
                | TaskStreamOperation::Resume { .. }
        );
        self.stream_work(
            async move {
                match operation {
                    TaskStreamOperation::Run { objective, profile } => {
                        self.run(&objective, profile.as_deref()).await
                    }
                    TaskStreamOperation::Skill {
                        objective,
                        skill_id,
                        profile,
                    } => {
                        self.run_skill(&objective, &skill_id, profile.as_deref())
                            .await
                    }
                    TaskStreamOperation::Resume {
                        run_id,
                        revision,
                        input,
                    } => self.resume(&run_id, revision, input).await,
                    TaskStreamOperation::Cancel { run_id, revision } => {
                        self.cancel_paused(&run_id, revision).await
                    }
                    TaskStreamOperation::Stop {
                        run_id,
                        revision,
                        reason,
                    } => self.stop_paused(&run_id, revision, &reason).await,
                    TaskStreamOperation::ModifyObjective {
                        run_id,
                        revision,
                        objective,
                        reason,
                    } => {
                        self.modify_objective_paused(&run_id, revision, &objective, &reason)
                            .await
                    }
                    TaskStreamOperation::ReviseLimits {
                        run_id,
                        revision,
                        limits,
                        reason,
                    } => {
                        self.revise_limits_paused(&run_id, revision, limits, &reason)
                            .await
                    }
                    TaskStreamOperation::ExternalUnavailable {
                        run_id,
                        revision,
                        request_id,
                        reason,
                    } => {
                        self.external_unavailable_paused(&run_id, revision, &request_id, &reason)
                            .await
                    }
                    TaskStreamOperation::ControllerUnavailable {
                        run_id,
                        revision,
                        request_id,
                        reason,
                    } => {
                        self.controller_unavailable_paused(&run_id, revision, &request_id, &reason)
                            .await
                    }
                    TaskStreamOperation::RestorationReview {
                        run_id,
                        revision,
                        review_id,
                        decision,
                    } => {
                        self.approve_restoration(&run_id, revision, &review_id, decision)
                            .await
                    }
                }
            },
            config,
            host_control,
        )
    }

    /// An admitted ordinary root uses the same bounded delivery and terminal projection as explicit operations, not another producer.
    pub(crate) fn stream_automatic(
        &self,
        entry: Box<super::automatic::AutomaticEntry>,
    ) -> Result<TaskRunStream<'_>> {
        let owner = entry.owner.clone();
        let mut cleanup = super::OwnedTurnCleanup::new(owner.clone());
        self.stream_work(
            async move {
                if let Some(sink) = current_sink() {
                    sink.owner(&owner);
                    *owner.events.write() = Some(sink);
                }
                let result = self.run_automatic(entry).await;
                cleanup.finish();
                result
            },
            Default::default(),
            false,
        )
    }

    /// Shared foreground delivery owns the actual operation future; backpressure, cleanup and final ACK ordering are identical for every entry.
    fn stream_work<'a, F>(
        &'a self,
        work: F,
        config: TaskStreamConfig,
        host_control: bool,
    ) -> Result<TaskRunStream<'a>>
    where
        F: Future<Output = Result<TaskRunResult>> + Send + 'a,
    {
        if config.capacity == 0 || config.capacity > 1024 {
            return Err(AgentError::Config(
                "task stream capacity must be 1..=1024".into(),
            ));
        }
        let (sender, receiver) = mpsc::channel(config.capacity);
        let sink = Arc::new(EventSink {
            sender,
            operation: uuid::Uuid::new_v4().to_string(),
            delivery: Default::default(),
            execution: Default::default(),
            owners: Default::default(),
            active: std::sync::atomic::AtomicBool::new(true),
        });
        let context = sink.clone();
        let future = Box::pin(EVENTS.scope(sink.clone(), async move {
            let result = work.await;
            // Existing APIs have already transferred safe pause/terminal custody; display loss cannot reverse that transfer.
            context.active.store(false, std::sync::atomic::Ordering::Release);
            match &result {
                Ok(result) => {
                    let run = &result.run;
                    if host_control {
                        context.emit(Some(run.key.run_id.clone()), Some(run.revision), TaskRunEventKind::HostControlApplied, json!({"status":run.status,"counters":run.counters})).await;
                    }
                    if let Some(pending) = &run.pending {
                        context.emit(Some(run.key.run_id.clone()), Some(run.revision), match pending.kind { TaskPendingKind::UserQuestion => TaskRunEventKind::UserQuestion, TaskPendingKind::ExternalCondition => TaskRunEventKind::ExternalConditionRequired, TaskPendingKind::HostContinue => TaskRunEventKind::HostContinueRequired, TaskPendingKind::Limit => TaskRunEventKind::CapacityRequired, TaskPendingKind::ValidationObservation => TaskRunEventKind::ValidationObservationRequired, TaskPendingKind::Approval => TaskRunEventKind::ApprovalRequired }, json!({"pending":safe_text(&pending.id),"details":"available through trusted host result"})).await;
                    }
                    let kind = match run.status {
                        TaskRunStatus::Running => TaskRunEventKind::HostControlApplied,
                        TaskRunStatus::Paused => TaskRunEventKind::TaskPaused,
                        TaskRunStatus::Completed => TaskRunEventKind::TaskCompleted,
                        TaskRunStatus::Incomplete => TaskRunEventKind::TaskIncomplete,
                        TaskRunStatus::LimitReached => TaskRunEventKind::TaskLimitReached,
                        TaskRunStatus::RecoveryRequired => TaskRunEventKind::TaskRecoveryRequired,
                        TaskRunStatus::Failed => TaskRunEventKind::TaskFailed,
                        TaskRunStatus::Cancelled => TaskRunEventKind::TaskCancelled,
                    };
                    context.emit(Some(run.key.run_id.clone()), Some(run.revision), kind, json!({"status":run.status,"counters":run.counters,"clocks":run.clocks,"pause_reason":run.pause_reason.as_deref().map(safe_text),"stop_reason":run.stop_reason.as_deref().map(safe_text),"response":result.final_response.as_ref().map(|r|safe_text(&r.content))})).await;
                }
                Err(error) => context.emit(None, None, TaskRunEventKind::OperationError, json!({"error":safe_text(&error.to_string())})).await,
            }
            result
        }));
        Ok(TaskRunStream {
            future: Some(future),
            receiver,
            sink,
            result: None,
            finished: false,
            polled: false,
        })
    }
}

impl crate::RuntimeAgent {
    /// Adapts this actual runtime to an explicitly configured shared runner, without constructing a second controller or enabling YAML entry.
    /// A runner for another runtime is rejected before polling or any task side effect.
    pub fn task_operation_stream<'a>(
        &'a self,
        runner: &'a AutonomyRunner,
        operation: TaskStreamOperation,
        config: TaskStreamConfig,
    ) -> Result<TaskRunStream<'a>> {
        if !self.same_runtime(runner.runtime().as_ref()) {
            return Err(AgentError::Config(
                "task stream runner belongs to another runtime".into(),
            ));
        }
        runner.stream_operation(operation, config)
    }

    /// Streams an explicit objective through the supplied runner and the same live runtime ownership boundary.
    pub fn run_task_stream<'a>(
        &'a self,
        runner: &'a AutonomyRunner,
        objective: &str,
        profile: Option<&str>,
    ) -> Result<TaskRunStream<'a>> {
        self.task_operation_stream(
            runner,
            TaskStreamOperation::Run {
                objective: objective.into(),
                profile: profile.map(str::to_owned),
            },
            Default::default(),
        )
    }
}
