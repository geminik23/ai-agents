//! Scripted foreground controls consume the same typed hosting API as embedded applications.

use crate::assertion::{
    Assertion, AssertionEvalContext, AssertionOutcome, AssertionResultDetail, evaluate_assertion,
};
use crate::evidence::{TurnEvidence, collect_turn_evidence_since};
use crate::fixtures::RecordingToolLog;
use crate::redaction::{RedactedString, redact_text};
use crate::task::{TaskAction, TaskContext, TaskReport};
use crate::{EvalError, Result};
use ai_agents_runtime::autonomy::*;
use futures::{StreamExt, future::BoxFuture, stream::FuturesUnordered};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, future::Future, sync::Arc};

/// Controls an actual independent participant while the foreground parent is pending.
/// Names select scoped participants, never authority; duplicate matching live participants fail closed.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskInteraction {
    pub agent: String,
    pub pending_kind: TaskPendingKind,
    pub operation: TaskAction,
    #[serde(default, rename = "assert")]
    pub assertions: Option<Assertion>,
    #[serde(default)]
    pub expect_error: Option<crate::suite::ExpectedError>,
}

/// Bounded acknowledged outcome, available even when a later store inspection fails.
/// The hashed whole key correlates observations but is neither a resume token nor a cleanup certificate.
#[derive(Debug, Clone, Serialize)]
pub struct TaskOperationOutcome {
    pub run_hash: String,
    pub revision: u64,
    pub status: TaskRunStatus,
    pub counters: TaskCounters,
}

pub(crate) fn run_hash(key: &TaskRunKey) -> String {
    use sha2::{Digest, Sha256};
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(key).expect("serializable task key"))
    )
}

/// Control evidence is separate from the parent's response and is retained even if the parent cannot settle.
#[derive(Clone, Serialize)]
pub struct TaskInteractionResult {
    pub index: usize,
    pub task: Option<TaskReport>,
    pub outcome: Option<TaskOperationOutcome>,
    pub runtime_error: Option<RedactedString>,
    pub capture_error: Option<RedactedString>,
    pub event_kinds: Vec<TaskRunEventKind>,
    pub events_complete: bool,
    pub assertion_results: Vec<AssertionResultDetail>,
    #[serde(skip_serializing)]
    pub evidence: TurnEvidence,
    #[serde(skip_serializing)]
    pub acknowledged_result: Option<TaskRunResult>,
    #[serde(skip_serializing)]
    pub events: Vec<TaskRunEvent>,
}
impl std::fmt::Debug for TaskInteractionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskInteractionResult")
            .field("index", &self.index)
            .field("events_complete", &self.events_complete)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
pub(crate) struct InteractionState {
    pub plan: Vec<TaskInteraction>,
    pub claimed: BTreeSet<usize>,
    pub results: Vec<TaskInteractionResult>,
    pub reported: usize,
    pub error: Option<&'static str>,
    pub cancellation_failed: bool,
    pub foreground_started: Option<std::time::Instant>,
}
pub(crate) type Interactions = Arc<parking_lot::Mutex<InteractionState>>;
type Ready = (
    usize,
    TaskInteraction,
    Vec<(TaskHostRun, TaskRunInspection)>,
);

/// Boxes optional host polling so deeply nested runtimes keep using their default thread stacks.
/// Allocation does not poll, start, detach or extend the authority of either work or controls.
pub(crate) fn service<'a, F: Future + Send + 'a>(
    session: &'a TaskHostSession,
    work: F,
    state: Interactions,
    redact: bool,
    foreground_runtime: usize,
) -> BoxFuture<'a, Result<F::Output>>
where
    F::Output: Send + 'a,
{
    Box::pin(service_inner(
        session,
        work,
        state,
        redact,
        foreground_runtime,
    ))
}

// Discovery, the parent and all started controls are polled together; storage reads cannot hold the other futures hostage.
async fn service_inner<F: Future>(
    session: &TaskHostSession,
    work: F,
    state: Interactions,
    redact: bool,
    foreground_runtime: usize,
) -> Result<F::Output> {
    let mut original = Box::pin(work);
    let mut primary = None;
    let mut controls: FuturesUnordered<BoxFuture<'static, String>> = FuturesUnordered::new();
    let mut discovery: Option<BoxFuture<'static, Vec<Ready>>> = None;
    let mut discovery_catalogue = Vec::new();
    let mut busy = BTreeSet::new();
    let mut cancellation: Option<BoxFuture<'_, Result<()>>> = None;
    let mut cancellation_started = false;
    let mut dirty = true;
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(10));
    loop {
        if primary.is_some() && controls.is_empty() {
            return Ok(primary.take().unwrap());
        }
        if dirty && primary.is_none() && session.interaction_available() && discovery.is_none() {
            let scripts = {
                let locked = state.lock();
                locked
                    .plan
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !locked.claimed.contains(index))
                    .map(|(index, script)| (index, script.clone()))
                    .collect()
            };
            dirty = false;
            discovery_catalogue = catalogue(session);
            discovery = Some(Box::pin(discover(
                session.runs(),
                scripts,
                busy.clone(),
                foreground_runtime,
            )));
        }
        if state.lock().error.is_some() && !cancellation_started {
            cancellation_started = true;
            session.close_input();
            cancellation = Some(Box::pin(crate::task::cancel_running(session)));
        }
        tokio::select! {
            biased;
            result = original.as_mut(), if primary.is_none() => { primary = Some(result); discovery = None; }
            Some(id) = controls.next(), if !controls.is_empty() => { busy.remove(&id); discovery = None; dirty = true; }
            ready = async { discovery.as_mut().unwrap().await }, if discovery.is_some() => {
                discovery = None;
                if primary.is_some() || !session.interaction_available() { continue; }
                                // A new or newly published participant can make a formerly unique selector ambiguous.
                                if discovery_catalogue != catalogue(session) { dirty = true; continue; }
                for (index, script, mut candidates) in ready {
                    if state.lock().claimed.contains(&index) { continue; }
                    candidates.retain(|(run, _)| !busy.contains(&run.key.run_id));
                    if candidates.len() > 1 {
                        state.lock().error = Some("ambiguous scoped task interaction target");
                        session.close_input();
                        break;
                    }
                    let Some((run, inspection)) = candidates.pop() else { continue; };
                    state.lock().claimed.insert(index);
                    busy.insert(run.key.run_id.clone());
                    let scoped = session.clone();
                    let shared = state.clone();
                    controls.push(Box::pin(async move {
                        let id = run.key.run_id.clone();
                        control(&scoped, run, inspection, index, script, redact, &shared).await;
                        id
                    }));
                }
            }
            outcome = async { cancellation.as_mut().unwrap().await }, if cancellation.is_some() => {
                            state.lock().cancellation_failed |= outcome.is_err();
                            cancellation = None;
                        }
            _ = session.changed() => { dirty = true; }
            _ = tick.tick() => { dirty = true; }
        }
    }
}

// A read-only catalogue fingerprint prevents stale discovery from choosing among changed participants.
type Catalogue = Vec<(String, String, usize, Option<u64>, bool)>;
fn catalogue(session: &TaskHostSession) -> Catalogue {
    session
        .runs()
        .iter()
        .map(|run| {
            (
                run.key.agent_id.clone(),
                run.key.run_id.clone(),
                Arc::as_ptr(&run.runner) as usize,
                run.published_revision(),
                run.execution_admitted(),
            )
        })
        .collect()
}

// A public pause ACK gates discovery; exact control admission still revalidates revision/request and current bindings.
async fn discover(
    runs: Vec<TaskHostRun>,
    scripts: Vec<(usize, TaskInteraction)>,
    busy: BTreeSet<String>,
    foreground_runtime: usize,
) -> Vec<Ready> {
    let mut ready = Vec::new();
    for (index, script) in scripts {
        let mut candidates = Vec::new();
        for run in &runs {
            // Exclude the actual foreground runtime, not a name/ID which an independent child may legitimately share.
            if busy.contains(&run.key.run_id)
                || Arc::as_ptr(run.runner.runtime()) as usize == foreground_runtime
                || run.runner.runtime().info().name != script.agent
            {
                continue;
            }
            let Some(revision) = run.published_revision() else {
                continue;
            };
            let Ok(inspection) = run.runner.task_inspection(&run.key.run_id, revision).await else {
                continue;
            };
            if inspection.run.status == TaskRunStatus::Paused
                && inspection.run.pending.as_ref().is_some_and(|pending| {
                    std::mem::discriminant(&pending.kind)
                        == std::mem::discriminant(&script.pending_kind)
                })
                && matches!(
                    run.runner.task_interaction(&run.key.run_id, revision).await,
                    Ok(Some(_))
                )
            {
                candidates.push((run.clone(), inspection));
            }
        }
        ready.push((index, script, candidates));
    }
    ready
}

// Stores acknowledged outcomes before further awaited capture, so outer timeout cannot erase completed control work.
fn retain(state: &Interactions, result: TaskInteractionResult) {
    let mut state = state.lock();
    if let Some(previous) = state
        .results
        .iter_mut()
        .find(|previous| previous.index == result.index)
    {
        *previous = result;
    } else {
        state.results.push(result);
    }
}

// Operation errors and observation errors are distinct: expect_error never accepts failed evidence collection.
async fn control(
    session: &TaskHostSession,
    run: TaskHostRun,
    inspection: TaskRunInspection,
    index: usize,
    script: TaskInteraction,
    redact: bool,
    state: &Interactions,
) {
    let agent = run.runner.runtime().clone();
    let cursor = agent.observability().map(|manager| manager.event_cursor());
    let prefix = inspection.run.evidence.tool_calls.len();
    let before = TaskRunResult {
        run: inspection.run,
        final_response: None,
    };
    let mut task = TaskContext::with_session(session.clone());
    task.runner_override = Some(run.runner.clone());
    task.last = Some(before.clone());
    let executed: Result<TaskRunResult> = session
        .scope_runner(run.runner.clone(), async {
            let operation = script.operation.operation(Some(&before))?;
            let mut stream = run.runner.stream_operation(operation, Default::default())?;
            while let Some(event) = stream.next_event().await {
                task.record(event);
            }
            stream
                .take_result()
                .ok_or_else(|| {
                    EvalError::Runtime("control stream ended without an acknowledged result".into())
                })?
                .map_err(EvalError::from)
        })
        .await;
    let (acknowledged, operation_error) = match executed {
        Ok(result) => (Some(result), None),
        Err(error) => {
            task.events_complete = false;
            (None, Some(error.to_string()))
        }
    };
    let log = RecordingToolLog::new();
    if let Some(result) = &acknowledged {
        for record in result.run.evidence.tool_calls.iter().skip(prefix) {
            log.push_executor_record(&record.record);
        }
    }
    let response = acknowledged
        .as_ref()
        .and_then(|r| r.final_response.as_ref())
        .map(|r| r.content.clone())
        .unwrap_or_default();
    let mut result = TaskInteractionResult {
        index,
        task: None,
        outcome: acknowledged.as_ref().map(|result| TaskOperationOutcome {
            run_hash: run_hash(&result.run.key),
            revision: result.run.revision,
            status: result.run.status,
            counters: result.run.counters.clone(),
        }),
        runtime_error: operation_error
            .as_deref()
            .map(|error| redact_text(error, redact, 0)),
        capture_error: None,
        event_kinds: task.events.iter().map(|event| event.kind).collect(),
        events_complete: task.events_complete,
        assertion_results: vec![detail("interaction.capture", false)],
        evidence: collect_turn_evidence_since(&agent, None, &log, 0, None, cursor),
        acknowledged_result: acknowledged.clone(),
        events: task.events.clone(),
    };
    retain(state, result.clone());
    let mut capture_ok = true;
    if let Some(acknowledged) = acknowledged {
        match task.capture(&agent, acknowledged).await {
            Ok((evidence, report)) => {
                result.evidence.task = Some(evidence);
                result.task = Some(report);
            }
            Err(error) => {
                capture_ok = false;
                result.capture_error = Some(redact_text(&error.to_string(), redact, 0));
            }
        }
    } else {
        // Historical executor facts can survive an errored operation, but cannot masquerade as an acknowledged task outcome or prove absence.
        if let Ok(display) = run.runner.task_display(&run.key.run_id).await
            && let Ok(inspection) = run
                .runner
                .task_inspection(&run.key.run_id, display.summary.revision)
                .await
        {
            let partial = RecordingToolLog::new();
            for record in inspection.run.evidence.tool_calls.iter().skip(prefix) {
                partial.push_executor_record(&record.record);
            }
            result.evidence.tool_executions = partial.records_since(0);
        }
    }
    let mut details = Vec::new();
    if let Some(assertion) = &script.assertions {
        if operation_error.is_some() {
            details.push(detail("interaction.evidence_unavailable", false));
        } else {
            details = match evaluate_assertion(
                assertion,
                AssertionEvalContext {
                    evidence: &result.evidence,
                    response: &response,
                    user_input: None,
                    scenario_id: None,
                    language: None,
                    judge_resolver: None,
                },
            )
            .await
            {
                AssertionOutcome::Passed(details) | AssertionOutcome::Failed(details) => details,
                AssertionOutcome::Error(_) => {
                    vec![detail("interaction.evidence_unavailable", false)]
                }
            };
        }
    }
    details.push(detail("interaction.capture", capture_ok));
    details.push(detail(
        "interaction.operation",
        match (script.expect_error.as_ref(), operation_error.as_deref()) {
            (None, None) => true,
            (Some(expected), Some(error)) => expected.matches(error),
            _ => false,
        },
    ));
    if redact {
        for detail in &mut details {
            detail.actual = Value::Null;
            detail.expected = Value::Null;
            detail.message = None;
        }
    }
    result.assertion_results = details;
    retain(state, result);
}

fn detail(name: &str, passed: bool) -> AssertionResultDetail {
    AssertionResultDetail {
        assertion: name.into(),
        passed,
        actual: json!(passed),
        expected: json!(true),
        message: None,
    }
}

// Reject unavailable hook-level evidence instead of letting an empty child log prove a negative claim.
fn validate_assertion_scope(assertion: &Assertion) -> Result<()> {
    let value = serde_json::to_value(assertion)?;
    for (key, value) in value.as_object().expect("assertion object") {
        if !value.is_null()
            && !matches!(
                key.as_str(),
                "task"
                    | "state"
                    | "state_in"
                    | "state_not"
                    | "response_contains"
                    | "response_contains_any"
                    | "response_not_contains"
                    | "response_not_empty"
                    | "tool_called"
                    | "tool_not_called"
                    | "context_path"
                    | "all"
                    | "any"
                    | "not"
            )
        {
            return Err(EvalError::Config("participant assertions require task, response, executor, state or context evidence; hook/judge assertions belong on the foreground turn".into()));
        }
    }
    for child in assertion
        .all
        .iter()
        .flatten()
        .chain(assertion.any.iter().flatten())
        .chain(assertion.not.iter().map(|a| a.as_ref()))
    {
        validate_assertion_scope(child)?;
    }
    Ok(())
}

pub(crate) fn validate(script: &TaskInteraction, index: usize) -> Result<()> {
    if script.agent.trim().is_empty()
        || matches!(
            script.operation,
            TaskAction::Run { .. }
                | TaskAction::Skill { .. }
                | TaskAction::Inspect
                | TaskAction::RestorationReview { .. }
        )
    {
        return Err(EvalError::Config(
            "interactions require an agent name and a retained control operation".into(),
        ));
    }
    if let Some(assertion) = &script.assertions {
        assertion.validate(&format!("interactions[{index}].assert"))?;
        validate_assertion_scope(assertion)?;
    } else if script.expect_error.is_none() {
        return Err(EvalError::Config(
            "interaction requires assert or expect_error".into(),
        ));
    }
    script.operation.validate(&AutonomyConfig::default())
}
