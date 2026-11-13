// Verify application-owned records through the public EvalRunner, not a test-only completion engine.
// Flow: repository suite -> the application's domain installer -> exact in-memory evidence -> redacted reports.
// These deterministic fixtures validate contracts, not web-source authenticity or live model quality.

#[path = "../src/task_domain.rs"]
mod task_domain;

use ai_agents::AgentBuilder;
use ai_agents::autonomy::{GateOutcome, TaskRunEventKind, TaskRunStatus};
use ai_agents_eval::fixtures::AttemptFixtureContext;
use ai_agents_eval::{EvalHost, EvalResult, EvalRunner, EvalRunnerOptions, ScenarioStatus};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct DomainHost {
    installations: AtomicUsize,
}

#[async_trait]
impl EvalHost for DomainHost {
    fn configure(
        &self,
        builder: AgentBuilder,
        _: &AttemptFixtureContext,
    ) -> ai_agents_eval::Result<AgentBuilder> {
        self.installations.fetch_add(1, Ordering::SeqCst);
        Ok(task_domain::install_domain(builder)?)
    }
}

// Locate declared repository sources from the manifest, independent of Cargo's workspace-relative file!() paths.
// Both the ordinary examples workspace and an external local validation manifest exercise the same fixtures.
fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|directory| {
            directory
                .join("examples/rust/custom-llm/src/task_domain.rs")
                .is_file()
                && directory.join("crates/ai-agents/Cargo.toml").is_file()
        })
        .expect("the validation manifest must be inside the source repository")
        .to_path_buf()
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        Self(AttemptFixtureContext::create(None).unwrap().workspace)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        // Only this outer suite/report directory is ours; the runner owns its distinct attempt workspaces and cleanup.
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

async fn evaluate(suite: &Path, output: &Path, host: &Arc<DomainHost>) -> EvalResult {
    EvalRunner::from_file(
        suite,
        EvalRunnerOptions {
            output: output.to_path_buf(),
            ..Default::default()
        },
    )
    .unwrap()
    .with_host(host.clone())
    .run()
    .await
    .unwrap()
}

async fn accepted(name: &str) -> EvalResult {
    let scratch = Scratch::new();
    let host = Arc::new(DomainHost::default());
    let result = evaluate(
        &repository().join(format!("examples/eval/mocked/autonomy/{name}_mocked.yaml")),
        &scratch.0.join("report"),
        &host,
    )
    .await;
    assert_eq!(
        (result.total, result.passed, result.failed, result.skipped),
        (1, 1, 0, 0),
        "{:?}",
        result
            .scenarios
            .iter()
            .map(|scenario| (&scenario.id, &scenario.status))
            .collect::<Vec<_>>()
    );
    assert_eq!(host.installations.load(Ordering::SeqCst), 1);
    result
}

#[tokio::test]
async fn research_rejects_duplicate_missing_and_unrecorded_citations() {
    let result = accepted("research").await;
    let task = result.scenarios[0].attempts[0].turns[0]
        .evidence
        .task
        .as_ref()
        .unwrap();
    let checks: Vec<_> = task
        .inspection
        .evidence
        .validations
        .iter()
        .filter(|check| check.check_id == "records")
        .map(|check| (check.metrics["unique"].as_u64().unwrap(), check.outcome))
        .collect();
    assert_eq!(
        checks,
        vec![
            (1, GateOutcome::Fail),
            (1, GateOutcome::Fail),
            (2, GateOutcome::Fail),
            (2, GateOutcome::Fail),
            (2, GateOutcome::Pass)
        ]
    );
    let operations: Vec<_> = task
        .inspection
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "domain" && entry.record.executed)
        .map(|entry| entry.record.executed_arguments["value"].clone())
        .collect();
    assert_eq!(operations, vec![json!("A"), json!("A"), json!("B")]);
    assert_eq!(task.inspection.run.counters.turns, 5);
    // Exact checks remain assertion-only even though their outcomes are safe to report.
    let report = serde_json::to_string(&result).unwrap();
    assert!(!report.contains("unrecorded-source"));
    assert!(!report.contains("observed domain predicate"));
}

#[tokio::test]
async fn support_retains_pre_question_state_limits_and_usage_without_replay() {
    let result = accepted("support").await;
    let turns = &result.scenarios[0].attempts[0].turns;
    assert_eq!(turns.len(), 2);
    let before = &turns[0].evidence.task.as_ref().unwrap().inspection.run;
    let after = &turns[1].evidence.task.as_ref().unwrap().inspection.run;
    assert_eq!(before.status, TaskRunStatus::Paused);
    assert_eq!(after.status, TaskRunStatus::Completed);
    assert_eq!(before.key, after.key);
    assert!(after.revision > before.revision);
    assert_eq!(
        (before.counters.llm_attempts, after.counters.llm_attempts),
        (2, 4)
    );
    assert_eq!(
        serde_json::to_value(&before.limits).unwrap(),
        serde_json::to_value(&after.limits).unwrap()
    );
    let records: Vec<_> = after
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "domain" && entry.record.executed)
        .map(|entry| {
            (
                &entry.record.executed_arguments,
                serde_json::from_str::<Value>(&entry.record.output).unwrap(),
            )
        })
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].0["op"], "source");
    assert_eq!(records[0].1["ticket"], "open");
    assert_eq!(records[1].0["op"], "resolve");
    assert_eq!(records[1].1["ticket"], "resolved");
    for (_, state) in &records {
        assert_eq!(state["sources"], json!(["ticket://request-context"]));
    }
    assert_eq!(
        turns
            .iter()
            .map(|turn| turn.evidence.llm_requests.len())
            .sum::<usize>(),
        4
    );
    assert!(
        !serde_json::to_string(&result)
            .unwrap()
            .contains("ticket://request-context")
    );
}

#[tokio::test]
async fn support_domain_installation_cannot_refill_the_enclosing_eval_budget() {
    let root = repository();
    let scratch = Scratch::new();
    let suite =
        std::fs::read_to_string(root.join("examples/eval/mocked/autonomy/support_mocked.yaml"))
            .unwrap()
            .replace("max_llm_calls: 4", "max_llm_calls: 3")
            .replace(
                "../../../yaml/autonomy/support_resolution.yaml",
                root.join("examples/yaml/autonomy/support_resolution.yaml")
                    .to_str()
                    .unwrap(),
            );
    // The original model-loop error has no new acknowledged task result; assert that exact budget error instead of inventing a terminal snapshot.
    let (before_resume, _) = suite.split_once("    steps:\n").unwrap();
    let suite = format!(
        r#"{before_resume}    steps:
      - !task
        operation: {{action: resume, response: {{kind: answer, value: {{answered: true, selected: [yes]}}}}}}
        expect_error: 'eval budget exceeded: max_llm_calls=3'
        assert:
          all:
            - tool_called: {{id: domain, count: 1, executed: true, success: true, args_executed: {{path: op, eq: resolve}}}}
            - tool_called: {{id: domain, result_path: {{path: sources, eq: ['ticket://request-context']}}}}
"#
    );
    let path = scratch.0.join("suite.yaml");
    std::fs::write(&path, suite).unwrap();
    let host = Arc::new(DomainHost::default());
    let result = evaluate(&path, &scratch.0.join("report"), &host).await;
    // Refusal during the original retained model loop preserves its semantic recovery cursor, not a new terminal ACK.
    // Require this exact protected error and the matched budget refusal; an arbitrary failed scenario is not evidence.
    assert_eq!(
        (result.total, result.passed, result.failed, result.skipped),
        (1, 0, 1, 0)
    );
    assert!(
        matches!(&result.scenarios[0].status, ScenarioStatus::Error { message }
        if message == "task cleanup unacknowledged; attempt workspace retained; no automatic retry")
    );
    assert_eq!(result.scenarios[0].attempts.len(), 1);
    assert_eq!(host.installations.load(Ordering::SeqCst), 1);
    let turns = &result.scenarios[0].attempts[0].turns;
    assert_eq!(turns.len(), 2);
    let before = &turns[0].evidence.task.as_ref().unwrap().inspection.run;
    assert_eq!(before.status, TaskRunStatus::Paused);
    assert_eq!(before.counters.llm_attempts, 2);
    assert!(turns[1].evidence.task.is_none());
    assert!(turns[1].runtime_error.is_some());
    let retained_workspace = turns[1].evidence.context["eval"]["workspace"]
        .as_str()
        .unwrap();
    assert!(Path::new(retained_workspace).is_dir());
    assert!(
        turns[1]
            .assertion_results
            .iter()
            .all(|assertion| assertion.passed)
    );
    assert!(
        turns[1]
            .assertion_results
            .iter()
            .any(|assertion| assertion.assertion == "expect_error" && assertion.passed)
    );
    assert_eq!(
        turns
            .iter()
            .map(|turn| turn.evidence.llm_requests.len())
            .sum::<usize>(),
        3
    );
}

#[tokio::test]
async fn operations_escalation_keeps_failed_health_and_completes_required_review_last() {
    let result = accepted("ops_remediation").await;
    let task = result.scenarios[0].attempts[0].turns[0]
        .evidence
        .task
        .as_ref()
        .unwrap();
    let checkpoints: Vec<_> = task
        .inspection
        .run
        .evidence
        .records
        .iter()
        .filter(|record| record.origin == "controller")
        .collect();
    assert_eq!(
        checkpoints
            .iter()
            .map(|record| record.stage.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("remediate"), Some("verify"), Some("review")]
    );
    for record in &checkpoints[..2] {
        assert_ne!(record.data["decision"]["action"], "terminal");
    }
    assert_eq!(checkpoints[2].data["decision"]["status"], "completed");
    let verification = checkpoints[1].data["validations"].as_array().unwrap();
    assert!(
        verification
            .iter()
            .any(|check| check["check_id"] == "health" && check["outcome"] == "fail")
    );
    assert!(
        verification
            .iter()
            .any(|check| check["check_id"] == "records" && check["outcome"] == "pass")
    );
    assert!(task.events_complete);
    let events: Vec<_> = task
        .events
        .iter()
        .filter_map(|event| match event.kind {
            TaskRunEventKind::StageStarted
            | TaskRunEventKind::StageCompleted
            | TaskRunEventKind::TaskCompleted => Some((
                event.kind,
                event.data.get("stage_index").and_then(Value::as_u64),
            )),
            _ => None,
        })
        .collect();
    use TaskRunEventKind::{StageCompleted, StageStarted, TaskCompleted};
    assert_eq!(
        events,
        vec![
            (StageStarted, None),
            (StageCompleted, Some(0)),
            (StageStarted, None),
            (StageCompleted, Some(1)),
            (StageStarted, None),
            (StageCompleted, Some(2)),
            (TaskCompleted, None)
        ]
    );
}

#[tokio::test]
async fn offline_application_executes_its_actual_retained_round_trip() {
    use std::process::Stdio;
    use std::time::Duration;
    use tokio::io::AsyncReadExt;

    let scratch = Scratch::new();
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_task-domain-host"))
        .env_clear()
        .current_dir(&scratch.0)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    // This is a hang guard, not a latency assertion; wait and both drains remain owned by this test.
    let observed = tokio::time::timeout(Duration::from_secs(60), async {
        let (status, _, _) = tokio::try_join!(
            child.wait(),
            stdout.read_to_end(&mut out),
            stderr.read_to_end(&mut err)
        )?;
        Ok::<_, std::io::Error>(status)
    })
    .await;
    drop(stdout);
    drop(stderr);
    let status = match observed {
        Ok(Ok(status)) => status,
        other => {
            let kill = child.start_kill();
            let reaped = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
            panic!(
                "application observation failed: {other:?}; direct-child kill={kill:?}, reap={reaped:?}"
            );
        }
    };
    assert!(
        status.success(),
        "application failed: {}",
        String::from_utf8_lossy(&err)
    );
    let display: Value = serde_json::from_slice(&out).unwrap();
    assert!(!display.is_null());
    assert!(
        !String::from_utf8(out)
            .unwrap()
            .contains("ticket://request-context")
    );
}
