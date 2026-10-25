//! Typed task operations use the configured runtime controller; exact observations never become export data.

use crate::assertion::{Assertion, PathAssertion};
use crate::fixtures::AttemptFixtureContext;
use crate::{EvalError, Result};
use ai_agents_hitl::ApprovalResult;
use ai_agents_runtime::autonomy::*;
use ai_agents_runtime::{AgentBuilder, RuntimeAgent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Caller-installed domain integration, applied before the builder freezes captured implementations.
/// Implementations must preserve the supplied fixtures, hooks and managed providers; validation must not construct or call providers.
#[async_trait::async_trait]
pub trait EvalHost: Send + Sync {
    fn configure(
        &self,
        builder: AgentBuilder,
        context: &AttemptFixtureContext,
    ) -> Result<AgentBuilder>;
    fn validate(&self, _spec: &ai_agents_runtime::spec::AgentSpec) -> Result<()> {
        Ok(())
    }
    /// Optionally supplies an actual prepared SDK operation, using this attempt's freshly configured runtime and fixtures.
    /// Hosts retain responsibility for pure staging, implementation/clock compatibility, original accounting and explicit reconstruction review.
    async fn prepare(
        &self,
        _agent: std::sync::Arc<RuntimeAgent>,
        _context: &AttemptFixtureContext,
    ) -> Result<Option<(std::sync::Arc<AutonomyRunner>, TaskRunResult)>> {
        Ok(None)
    }
}

/// One operation on the same configured host. Resume/control target the last exact acknowledged result, never a display lookup.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStep {
    pub operation: TaskAction,
    #[serde(rename = "assert")]
    pub assertions: Assertion,
    #[serde(default)]
    pub expect_error: Option<crate::suite::ExpectedError>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskAction {
    /// Inspects the exact supplied startup result without consuming its pending request or replaying events.
    Inspect,
    /// Fresh reconstruction review is separate from answering the original question/approval.
    RestorationReview {
        decision: ApprovalResult,
    },
    Run {
        objective: String,
        #[serde(default)]
        profile: Option<String>,
    },
    Skill {
        objective: String,
        skill: String,
        #[serde(default)]
        profile: Option<String>,
    },
    Resume {
        response: TaskReply,
    },
    Cancel,
    Stop {
        reason: String,
    },
    Objective {
        objective: String,
        reason: String,
    },
    Limits {
        limits: TaskRunLimits,
        reason: String,
    },
}

/// Explicit human fixture data; request IDs are taken from the retained result and still checked by the runtime.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskReply {
    Continue,
    Answer {
        value: Value,
    },
    Approval {
        decision: ApprovalResult,
    },
    QuestionTimeout,
    External {
        event_id: String,
        #[serde(default)]
        target_revision: Option<String>,
        payload: Value,
    },
}
impl TaskReply {
    pub(crate) fn bind(&self, pending: &TaskPendingRequest) -> TaskResumeInput {
        match self {
            Self::Continue => TaskResumeInput::Continue,
            Self::Answer { value } => TaskResumeInput::UserAnswer {
                request_id: pending.id.clone(),
                answer: value.clone(),
            },
            Self::Approval { decision } => TaskResumeInput::Approval {
                request_id: pending.id.clone(),
                result: decision.clone(),
            },
            Self::QuestionTimeout => TaskResumeInput::QuestionTimeout {
                request_id: pending.id.clone(),
            },
            Self::External {
                event_id,
                target_revision,
                payload,
            } => TaskResumeInput::ExternalSignal(ExternalSignal {
                request_id: pending.id.clone(),
                event_id: event_id.clone(),
                target_revision: target_revision.clone(),
                payload: payload.clone(),
            }),
        }
    }
}
impl TaskAction {
    pub(crate) fn validate(&self, config: &AutonomyConfig) -> Result<()> {
        match self {
            Self::Run { objective, profile }
            | Self::Skill {
                objective, profile, ..
            } => {
                if objective.trim().is_empty() {
                    return Err(EvalError::Config("task objective must not be empty".into()));
                }
                if profile
                    .as_ref()
                    .is_some_and(|profile| !config.profiles.contains_key(profile))
                {
                    return Err(EvalError::Config("task selects an unknown profile".into()));
                }
            }
            Self::Stop { reason }
            | Self::Objective { reason, .. }
            | Self::Limits { reason, .. }
                if reason.trim().is_empty() =>
            {
                return Err(EvalError::Config(
                    "task control reason must not be empty".into(),
                ));
            }
            _ => {}
        }
        if matches!(self, Self::Objective { objective, .. } if objective.trim().is_empty()) {
            return Err(EvalError::Config("task objective must not be empty".into()));
        }
        if matches!(self, Self::Skill { skill, .. } if skill.trim().is_empty()) {
            return Err(EvalError::Config("task skill ID must not be empty".into()));
        }
        Ok(())
    }

    pub(crate) fn operation(&self, last: Option<&TaskRunResult>) -> Result<TaskStreamOperation> {
        match self {
            Self::Run { objective, profile } => {
                return Ok(TaskStreamOperation::Run {
                    objective: objective.clone(),
                    profile: profile.clone(),
                });
            }
            Self::Skill {
                objective,
                skill,
                profile,
            } => {
                return Ok(TaskStreamOperation::Skill {
                    objective: objective.clone(),
                    skill_id: skill.clone(),
                    profile: profile.clone(),
                });
            }
            _ => {}
        }
        let run = &last
            .ok_or_else(|| {
                EvalError::Config(
                    "task control requires a preceding acknowledged task result".into(),
                )
            })?
            .run;
        let run_id = run.key.run_id.clone();
        let revision = run.revision;
        Ok(match self {
            Self::Resume { response } => TaskStreamOperation::Resume {
                run_id,
                revision,
                input: response.bind(run.pending.as_ref().ok_or_else(|| {
                    EvalError::Config("task resume requires a pending request".into())
                })?),
            },
            Self::Cancel => TaskStreamOperation::Cancel { run_id, revision },
            Self::Stop { reason } => TaskStreamOperation::Stop {
                run_id,
                revision,
                reason: reason.clone(),
            },
            Self::Objective { objective, reason } => TaskStreamOperation::ModifyObjective {
                run_id,
                revision,
                objective: objective.clone(),
                reason: reason.clone(),
            },
            Self::Limits { limits, reason } => TaskStreamOperation::ReviseLimits {
                run_id,
                revision,
                limits: limits.clone(),
                reason: reason.clone(),
            },
            Self::Run { .. }
            | Self::Skill { .. }
            | Self::Inspect
            | Self::RestorationReview { .. } => unreachable!(),
        })
    }
}

/// Structural expectations on an acknowledged task; missing inspection is an assertion error even underneath negation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskAssertion {
    #[serde(default)]
    pub status: Option<TaskRunStatus>,
    #[serde(default)]
    pub pending_kind: Option<TaskPendingKind>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub same_run: Option<bool>,
    #[serde(default)]
    pub counters: Option<PathAssertion>,
    #[serde(default)]
    pub progress: Option<PathAssertion>,
    #[serde(default)]
    pub checks: Vec<TaskCheckAssertion>,
    #[serde(default)]
    pub events_include: Vec<TaskRunEventKind>,
    #[serde(default)]
    pub event_result_parity: Option<bool>,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCheckAssertion {
    pub id: String,
    pub outcome: GateOutcome,
}

/// Assertion-only host evidence: no serde implementation, no recovery payload and no content-bearing Debug.
#[derive(Clone)]
pub struct TaskEvidence {
    pub inspection: TaskRunInspection,
    pub events: Vec<TaskRunEvent>,
    pub events_complete: bool,
    pub same_run: Option<bool>,
}
impl std::fmt::Debug for TaskEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskEvidence")
            .field("revision", &self.inspection.run.revision)
            .field("events_complete", &self.events_complete)
            .finish_non_exhaustive()
    }
}

/// Bounded safe task report is independent of assertion-time domain data and does not certify current completion eligibility.
#[derive(Debug, Clone, Serialize)]
pub struct TaskReport {
    pub display: TaskRunDisplay,
    pub validations: Vec<TaskValidationReport>,
    pub validations_truncated: bool,
    pub event_kinds: Vec<TaskRunEventKind>,
    pub events_complete: bool,
    pub same_run: Option<bool>,
}

/// Validation outcomes retain observation identity without exporting domain labels, metrics or proof payloads.
#[derive(Debug, Clone, Serialize)]
pub struct TaskValidationReport {
    pub check_hash: String,
    pub outcome: GateOutcome,
    pub cycle: u64,
    pub objective_revision: u64,
    pub eligible_at_capture: bool,
}

pub(crate) struct TaskContext {
    pub cleanup_uncertain: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub initializing: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub runner_override: Option<std::sync::Arc<AutonomyRunner>>,
    pub session: TaskHostSession,
    pub deadline: Option<tokio::time::Instant>,
    pub workspace_hold: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub last: Option<TaskRunResult>,
    pub interactions: crate::interactions::Interactions,
    pub redact_interactions: bool,
    pub events: Vec<TaskRunEvent>,
    pub events_complete: bool,
}
impl Default for TaskContext {
    fn default() -> Self {
        Self {
            cleanup_uncertain: None,
            initializing: None,
            runner_override: None,
            session: TaskHostSession::new(true),
            deadline: None,
            workspace_hold: None,
            last: None,
            interactions: std::sync::Arc::new(parking_lot::Mutex::new(Default::default())),
            redact_interactions: true,
            events: Vec::new(),
            events_complete: true,
        }
    }
}
impl Drop for TaskContext {
    fn drop(&mut self) {
        // A known unowned attempt can remove its files; interrupted task custody retains them for explicit recovery.
        if !self.session.has_protected_work()
            && !self
                .cleanup_uncertain
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
            && !self
                .initializing
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
            && let Some(hold) = &self.workspace_hold
        {
            hold.store(false, std::sync::atomic::Ordering::Release);
        }
    }
}

impl TaskContext {
    pub(crate) fn with_session(session: TaskHostSession) -> Self {
        let mut context = Self::default();
        context.session = session;
        context
    }

    pub(crate) fn runner(&self, agent: &RuntimeAgent) -> Result<std::sync::Arc<AutonomyRunner>> {
        self.runner_override
            .clone()
            .map_or_else(|| agent.task_runner().map_err(EvalError::from), Ok)
    }

    pub(crate) fn remaining(&self, requested: u64) -> u64 {
        self.deadline.map_or(requested, |deadline| {
            requested.min(
                deadline
                    .saturating_duration_since(tokio::time::Instant::now())
                    .as_millis() as u64,
            )
        })
    }

    pub(crate) fn record(&mut self, event: TaskRunEvent) {
        // Delivery remains bounded even for long runs. Truncation cannot prove event absence or parity.
        if self.events.len() < 4096 {
            self.events.push(event);
        } else {
            self.events_complete = false;
        }
    }

    pub(crate) async fn operate(
        &mut self,
        agent: &RuntimeAgent,
        step: &TaskStep,
        timeout_ms: u64,
    ) -> Result<(TaskRunResult, Option<String>)> {
        let runner = self.runner(agent)?;
        if matches!(step.operation, TaskAction::Inspect) {
            self.events_complete = false;
            return Ok((
                self.last.clone().ok_or_else(|| {
                    EvalError::Config("task inspect requires an acknowledged result".into())
                })?,
                None,
            ));
        }
        let operation = if let TaskAction::RestorationReview { decision } = &step.operation {
            let run = &self
                .last
                .as_ref()
                .ok_or_else(|| {
                    EvalError::Config("restoration review requires an acknowledged result".into())
                })?
                .run;
            let review = runner
                .restoration_review(&run.key.run_id)
                .await?
                .ok_or_else(|| EvalError::Config("no pending restoration review".into()))?;
            TaskStreamOperation::RestorationReview {
                run_id: run.key.run_id.clone(),
                revision: run.revision,
                review_id: review.id,
                decision: decision.clone(),
            }
        } else {
            step.operation.operation(self.last.as_ref())?
        };
        let session = self.session.clone();
        let interactions = self.interactions.clone();
        let redact = self.redact_interactions;
        let foreground_runtime = agent as *const RuntimeAgent as usize;
        let mut stream = runner.stream_operation(operation, Default::default())?;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let work = async {
            while let Some(event) = stream.next_event().await {
                self.record(event);
            }
            stream
                .take_result()
                .ok_or_else(|| {
                    EvalError::Runtime("task stream ended without an operation result".into())
                })?
                .map_err(EvalError::from)
        };
        let (result, deadline_error) = drive(
            &session,
            crate::interactions::service(
                &session,
                session.scope_runner(runner.clone(), work),
                interactions,
                redact,
                foreground_runtime,
            ),
            deadline,
        )
        .await?;
        Ok((result??, deadline_error))
    }

    pub(crate) async fn capture(
        &mut self,
        agent: &RuntimeAgent,
        result: TaskRunResult,
    ) -> Result<(TaskEvidence, TaskReport)> {
        let runner = self.runner(agent)?;
        self.session.observe_result(runner.clone(), &result).await?;
        let inspection = runner
            .task_inspection(&result.run.key.run_id, result.run.revision)
            .await?;
        let same_run = self.last.as_ref().map(|old| old.run.key == result.run.key);
        let display = runner.task_display(&result.run.key.run_id).await?;
        if display.summary.key != result.run.key || display.summary.revision != result.run.revision
        {
            return Err(EvalError::Runtime(
                "task changed during observation capture".into(),
            ));
        }
        let validations = inspection
            .evidence
            .validations
            .iter()
            .rev()
            .take(64)
            .map(|result| {
                use sha2::{Digest, Sha256};
                TaskValidationReport {
                    check_hash: format!("{:x}", Sha256::digest(result.check_id.as_bytes())),
                    outcome: result.outcome,
                    cycle: result.identity.cycle,
                    objective_revision: result.identity.objective_revision,
                    eligible_at_capture: inspection
                        .scope
                        .as_ref()
                        .and_then(|scope| {
                            latest_validation(
                                &inspection.evidence.validations,
                                &result.check_id,
                                scope,
                            )
                        })
                        .is_some_and(|latest| std::ptr::eq(latest, result)),
                }
            })
            .collect();
        let report = TaskReport {
            display,
            validations,
            validations_truncated: inspection.evidence.validations.len() > 64,
            event_kinds: self.events.iter().map(|event| event.kind).collect(),
            events_complete: self.events_complete,
            same_run,
        };
        let evidence = TaskEvidence {
            inspection,
            events: std::mem::take(&mut self.events),
            events_complete: self.events_complete,
            same_run,
        };
        self.last = Some(result);
        self.events_complete = true;
        Ok((evidence, report))
    }

    pub(crate) async fn cleanup(&self) -> Result<()> {
        // Mark before awaited inspection: a read failure is not proof that custody or effects have settled.
        if let Some(flag) = &self.cleanup_uncertain {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        for run in self.session.runs() {
            if run.execution_admitted() {
                let display =
                    tokio::time::timeout_at(deadline, run.runner.task_display(&run.key.run_id))
                        .await
                        .map_err(|_| {
                            EvalError::Runtime("task cleanup inspection timed out".into())
                        })??;
                if !display.summary.status.is_terminal() {
                    tokio::time::timeout_at(deadline, run.runner.request_cancel(&run.key.run_id))
                        .await
                        .map_err(|_| {
                            EvalError::Runtime("task cleanup unacknowledged after timeout".into())
                        })??;
                }
            }
        }
        if self.session.has_protected_work() {
            return Err(EvalError::Runtime(
                "task cleanup unacknowledged; retry/reset and fixture removal are unsafe".into(),
            ));
        }
        if let Some(flag) = &self.cleanup_uncertain {
            flag.store(false, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }
}

/// Drives the original future during cancellation, including work admitted through automatic or composed entry.
/// The bounded drain is a recovery boundary, not a cleanup certificate; callers retain files and prohibit retry when custody remains.
pub(crate) async fn drive<F: std::future::Future>(
    session: &TaskHostSession,
    work: F,
    deadline: tokio::time::Instant,
) -> Result<(F::Output, Option<String>)> {
    let mut original = Box::pin(session.scope(work));
    tokio::select! {
        biased;
        result = &mut original => return Ok((result, None)),
        _ = tokio::time::sleep_until(deadline) => {}
    }
    if !session.has_protected_work() {
        return Err(EvalError::Runtime(
            "operation timed out without protected task work".into(),
        ));
    }
    session.close_input();
    let drain_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let cancellation = cancel_running(session);
    tokio::pin!(cancellation);
    tokio::select! {
        biased;
        result = &mut original => Ok((result, Some("operation timed out; original operation drained".into()))),
        cancellation_result = &mut cancellation => {
            let reason = match cancellation_result {
                Err(error) => format!("operation timed out; cancellation request failed: {error}"),
                Ok(()) => "operation timed out; cancellation stopped before settlement".into(),
            };
            drain_after_cancel_error(original.as_mut(), drain_deadline, reason).await
        }
        _ = tokio::time::sleep_until(drain_deadline) => Err(EvalError::Runtime("task cleanup unacknowledged after bounded drain".into())),
    }
}

// A cancellation write error is not a reason to abandon an original future which can still settle.
pub(crate) async fn drain_after_cancel_error<F: std::future::Future>(
    original: std::pin::Pin<&mut F>,
    deadline: tokio::time::Instant,
    reason: String,
) -> Result<(F::Output, Option<String>)> {
    match tokio::time::timeout_at(deadline, original).await {
        Ok(result) => Ok((result, Some(reason))),
        Err(_) => Err(EvalError::Runtime(format!(
            "task cleanup unacknowledged after bounded drain: {reason}"
        ))),
    }
}

// Serial cancellation follows actual admitted handles while the owning foreground future keeps polling.
pub(crate) async fn cancel_running(session: &TaskHostSession) -> Result<()> {
    let mut requested = std::collections::BTreeSet::new();
    loop {
        for run in session.runs() {
            if !run.execution_admitted() || requested.contains(&run.key.run_id) {
                continue;
            }
            if run
                .runner
                .request_running_cancel(&run.key.run_id)
                .await?
                .is_some()
            {
                requested.insert(run.key.run_id);
            } else if let Some(revision) = run.published_revision() {
                // A child can already be publicly paused while its ordinary parent is still awaiting completion.
                let display = run.runner.task_display(&run.key.run_id).await?;
                if display.summary.status == TaskRunStatus::Paused
                    && display.summary.revision == revision
                {
                    run.runner.cancel_paused(&run.key.run_id, revision).await?;
                    requested.insert(run.key.run_id);
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
}

pub(crate) fn terminal_kind(status: TaskRunStatus) -> Option<TaskRunEventKind> {
    Some(match status {
        TaskRunStatus::Paused => TaskRunEventKind::TaskPaused,
        TaskRunStatus::Completed => TaskRunEventKind::TaskCompleted,
        TaskRunStatus::Incomplete => TaskRunEventKind::TaskIncomplete,
        TaskRunStatus::LimitReached => TaskRunEventKind::TaskLimitReached,
        TaskRunStatus::Failed => TaskRunEventKind::TaskFailed,
        TaskRunStatus::Cancelled => TaskRunEventKind::TaskCancelled,
        TaskRunStatus::RecoveryRequired => TaskRunEventKind::TaskRecoveryRequired,
        TaskRunStatus::Running => return None,
    })
}
