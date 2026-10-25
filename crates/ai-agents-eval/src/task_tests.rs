use crate::fixtures::AttemptFixtureContext;
use crate::{EvalRunner, EvalRunnerOptions};

const AGENT: &str = r#"
name: TaskEval
system_prompt: Produce a verified report.
tools: []
autonomy:
  enabled: true
  max_turns: 4
  max_llm_calls: 8
  completion: {response_contains: verified report}
  premature_finish: {action: ask_user, on_unavailable: stop}
"#;

async fn evaluate(agent: &str, steps: &str) -> crate::suite::EvalResult {
    let context = AttemptFixtureContext::create(None).unwrap();
    std::fs::write(context.workspace.join("agent.yaml"), agent).unwrap();
    let path = context.workspace.join("suite.yaml");
    std::fs::write(
        &path,
        format!(
            r#"
name: Task evidence
agent: agent.yaml
settings: {{retries: 0, timeout_per_turn_ms: 2000}}
fixtures:
  llm:
    mode: mock
    responses: [draft, verified report]
scenarios:
  - id: typed-task
    {steps}
"#
        ),
    )
    .unwrap();
    let runner = EvalRunner::from_file(&path, EvalRunnerOptions::default()).unwrap();
    let result = runner.run().await.unwrap();
    std::fs::remove_dir_all(&context.workspace).unwrap();
    result
}

#[tokio::test]
async fn explicit_task_pause_resume_preserves_identity_accounting_events_and_redaction() {
    let result = evaluate(
        AGENT,
        r#"steps:
      - !task
        operation: {action: run, objective: private-objective-marker}
        assert:
          task:
            status: paused
            pending_kind: user_question
            event_result_parity: true
            events_include: [task_started, user_question, task_paused]
            counters: {path: llm_attempts, eq: 1}
      - !task
        operation: {action: resume, response: {kind: answer, value: private-answer-marker}}
        assert:
          all:
            - response_contains: verified report
            - task:
                status: completed
                same_run: true
                event_result_parity: true
                counters: {path: llm_attempts, eq: 2}
"#,
    )
    .await;
    assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
    let attempts = &result.scenarios[0].attempts;
    assert_eq!(attempts[0].turns.len(), 2);
    assert!(
        attempts[0].turns[1]
            .evidence
            .task
            .as_ref()
            .unwrap()
            .same_run
            .unwrap()
    );
    let report = serde_json::to_string(&result).unwrap();
    for secret in [
        "private-objective-marker",
        "private-answer-marker",
        "verified report",
        "runtime.controller",
        "owner_token",
    ] {
        assert!(
            !report.contains(secret),
            "serialized report leaked {secret}"
        );
    }
    assert!(report.contains("completed"));
}

#[tokio::test]
async fn paused_task_is_acknowledged_cancelled_at_attempt_end_without_answer_or_extra_calls() {
    let result = evaluate(
        AGENT,
        r#"steps:
      - !task
        operation: {action: run, objective: report}
        assert: {task: {status: paused, counters: {path: llm_attempts, eq: 1}}}
"#,
    )
    .await;
    assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
    let task = result.scenarios[0].attempts[0].turns[0]
        .evidence
        .task
        .as_ref()
        .unwrap();
    assert_eq!(task.inspection.run.counters.llm_attempts, 1);
}

#[tokio::test]
async fn automatic_state_task_has_exact_result_and_lifecycle_evidence() {
    let agent = r#"
name: StateTask
system_prompt: Produce a report.
tools: []
states:
  initial: reporting
  states:
    reporting:
      autonomy:
        enabled: true
        mode: state_until_exit
        max_turns: 4
        max_llm_calls: 8
        completion: {response_contains: verified report}
        premature_finish: {action: ask_user, on_unavailable: stop}
"#;
    let result = evaluate(
        agent,
        r#"turns:
      - input: report
        stream: true
        assert: {task: {status: paused, event_result_parity: true}}
    steps:
      - !task
        operation: {action: resume, response: {kind: answer, value: proceed}}
        assert: {task: {status: completed, same_run: true, counters: {path: llm_attempts, eq: 2}}}
"#,
    )
    .await;
    assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
}

#[tokio::test]
async fn missing_task_observation_cannot_pass_through_not() {
    let result = evaluate(
        "name: Ordinary\nsystem_prompt: Answer\ntools: []\n",
        r#"turns:
      - input: hello
        assert: {not: {task: {status: completed}}}
"#,
    )
    .await;
    assert_eq!(result.passed, 0);
}

async fn check_captured(
    assertion: &str,
    evidence: &crate::evidence::TurnEvidence,
) -> crate::AssertionOutcome {
    let assertion = serde_yaml::from_str(assertion).unwrap();
    crate::evaluate_assertion(
        &assertion,
        crate::assertion::AssertionEvalContext {
            evidence,
            response: "draft",
            user_input: None,
            scenario_id: None,
            language: None,
            judge_resolver: None,
        },
    )
    .await
}

#[tokio::test]
async fn any_success_does_not_skip_sibling_task_status() {
    let result = evaluate(
        AGENT,
        r#"steps:
      - !task
        operation: {action: run, objective: report}
        assert:
          task: {status: completed}
          any: [{response_not_empty: true}]
"#,
    )
    .await;
    assert_eq!(result.passed, 0);
    assert!(
        result.scenarios[0].attempts[0].turns[0]
            .assertion_results
            .iter()
            .any(|a| a.assertion == "task.status" && !a.passed)
    );
}

#[tokio::test]
async fn truncated_events_remain_unavailable_through_not_and_any() {
    let result = evaluate(
        AGENT,
        r#"steps:
      - !task
        operation: {action: run, objective: report}
        assert: {task: {status: paused}}
"#,
    )
    .await;
    let mut evidence = result.scenarios[0].attempts[0].turns[0].evidence.clone();
    evidence.task.as_mut().unwrap().events_complete = false;
    for assertion in [
        "not: {task: {events_include: [task_paused]}}",
        "not: {any: [{task: {event_result_parity: true}}]}",
    ] {
        assert!(matches!(
            check_captured(assertion, &evidence).await,
            crate::AssertionOutcome::Error(_)
        ));
    }
}

#[tokio::test]
async fn missing_task_observation_is_not_a_false_fact_inside_nested_any() {
    let result = evaluate(
        "name: Ordinary\nsystem_prompt: Answer\ntools: []\n",
        r#"turns:
      - input: hello
        assert: {not: {any: [{task: {status: completed}}]}}
"#,
    )
    .await;
    assert_eq!(result.passed, 0);
    assert_eq!(
        result.scenarios[0].attempts[0].turns.len(),
        1,
        "available partial evidence must survive assertion errors"
    );
}

#[tokio::test]
async fn task_path_assertions_do_not_export_dynamic_domain_keys() {
    let agent = AGENT.replace("completion: {response_contains: verified report}", "completion: {response_not_empty: true}\n  validation:\n    checks: [{id: private-domain-key-marker, adapter: builtin.evidence, config: {assertion: {path: absent, exists: false}}}]");
    let result = evaluate(
        &agent,
        r#"steps:
      - !task
        operation: {action: run, objective: report}
        assert: {task: {status: completed, progress: {path: high_water_marks, exists: true}}}
"#,
    )
    .await;
    assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
    let evidence = result.scenarios[0].attempts[0].turns[0]
        .evidence
        .task
        .as_ref()
        .unwrap();
    assert!(
        serde_json::to_string(&evidence.inspection.run.progress)
            .unwrap()
            .contains("private-domain-key-marker")
    );
    assert!(!format!("{evidence:?}").contains("private-domain-key-marker"));
    assert!(
        !serde_json::to_string(&result)
            .unwrap()
            .contains("private-domain-key-marker")
    );
}

#[tokio::test]
async fn task_report_metrics_deduplicate_resume_and_observe_both_operations() {
    let agent = format!("{AGENT}\nobservability:\n  enabled: true\n");
    let result = evaluate(
        &agent,
        r#"steps:
      - !task
        operation: {action: run, objective: report}
        assert: {task: {status: paused}}
      - !task
        operation: {action: resume, response: {kind: answer, value: proceed}}
        assert: {task: {status: completed}}
"#,
    )
    .await;
    assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
    assert_eq!(result.schema_version, 3);
    assert_eq!(result.metrics.observed_task_runs, 1);
    assert_eq!(result.metrics.observed_task_llm_attempts, 2);
    let report = result.observability.as_ref().unwrap();
    assert_eq!(report.summary.total_llm_calls, 2);
    assert!(report.by_purpose.iter().any(|metric| {
        metric.dimensions.get("purpose").map(String::as_str) == Some("autonomy_completion_gate")
    }));
    assert!(result.scenarios[0].attempts[0].turns.iter().all(|turn| {
        turn.evidence
            .observability
            .as_ref()
            .unwrap()
            .events
            .iter()
            .any(|event| {
                matches!(
                    &event.event_type,
                    ai_agents_observability::EventType::TaskLifecycle { .. }
                )
            })
    }));
}

#[test]
fn attempt_files_are_bounded_relative_and_never_overwrite_completed_effects() {
    let context = AttemptFixtureContext::create(None).unwrap();
    let mut config = crate::FixturesConfig::default();
    for invalid in ["../escape", "/absolute", "a/../b", "C:/drive", "a\\b"] {
        config.files.clear();
        config.files.insert(invalid.into(), "fixture".into());
        assert!(config.validate().is_err(), "{invalid}");
    }
    config.files.clear();
    config.files.insert("src/code.txt".into(), "before".into());
    context.materialize_files(&config).unwrap();
    let path = context.workspace.join("src/code.txt");
    std::fs::write(&path, "actual effect").unwrap();
    assert!(context.materialize_files(&config).is_err());
    assert_eq!(std::fs::read_to_string(path).unwrap(), "actual effect");
    std::fs::remove_dir_all(context.workspace).unwrap();
}

#[test]
fn configuration_checks_reject_task_profiles_without_constructing_providers() {
    let context = AttemptFixtureContext::create(None).unwrap();
    std::fs::write(context.workspace.join("agent.yaml"), AGENT).unwrap();
    let suite = context.workspace.join("suite.yaml");
    std::fs::write(&suite, "name: Pure task check\nagent: agent.yaml\nfixtures: {llm: {mode: real}}\nscenarios:\n  - id: check\n    steps:\n      - !task\n        operation: {action: run, objective: report}\n        assert: {task: {status: completed}}\n").unwrap();
    assert!(
        EvalRunner::validate_file(&suite, None).is_ok(),
        "validation is provider-free even for declared live configuration"
    );
    let invalid = std::fs::read_to_string(&suite)
        .unwrap()
        .replace("objective: report", "objective: report, profile: missing");
    std::fs::write(&suite, invalid).unwrap();
    assert!(
        EvalRunner::validate_file(&suite, None)
            .unwrap_err()
            .to_string()
            .contains("unknown profile")
    );
    std::fs::remove_dir_all(context.workspace).unwrap();
}

#[tokio::test]
async fn cancellation_error_drains_the_original_future_instead_of_dropping_it() {
    let release = std::sync::Arc::new(tokio::sync::Semaphore::new(0));
    let original_release = release.clone();
    let mut original = Box::pin(async move {
        original_release.acquire().await.unwrap().forget();
        "actual settlement"
    });
    let mut drain = Box::pin(crate::task::drain_after_cancel_error(
        original.as_mut(),
        tokio::time::Instant::now() + std::time::Duration::from_secs(5),
        "cancellation write failed".into(),
    ));
    assert!(futures::poll!(drain.as_mut()).is_pending());
    release.add_permits(1);
    let (result, error) = drain.await.unwrap();
    assert_eq!(result, "actual settlement");
    assert_eq!(error.as_deref(), Some("cancellation write failed"));
}

struct ReadFailureStore {
    inner: std::sync::Arc<dyn ai_agents_runtime::autonomy::TaskRunStore>,
    fail: std::sync::atomic::AtomicBool,
}
#[async_trait::async_trait]
impl ai_agents_runtime::autonomy::TaskRunStore for ReadFailureStore {
    async fn list(
        &self,
    ) -> ai_agents_core::Result<Vec<ai_agents_runtime::autonomy::TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(id, revision).await
    }
    async fn create(
        &self,
        snapshot: &ai_agents_runtime::autonomy::TaskRunSnapshot,
    ) -> ai_agents_core::Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(
        &self,
        id: &str,
    ) -> ai_agents_core::Result<Option<ai_agents_runtime::autonomy::TaskRunSnapshot>> {
        if self.fail.load(std::sync::atomic::Ordering::Acquire) {
            return Err(ai_agents_core::AgentError::Persistence(
                "injected cleanup read failure".into(),
            ));
        }
        self.inner.load(id).await
    }
    async fn mutate(
        &self,
        id: &str,
        mutation: &ai_agents_runtime::autonomy::TaskRunMutation,
    ) -> ai_agents_core::Result<ai_agents_runtime::autonomy::TaskRunSnapshot> {
        self.inner.mutate(id, mutation).await
    }
}

#[tokio::test]
async fn cleanup_read_failure_retains_files_even_without_a_live_owner() {
    use ai_agents_runtime::autonomy::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let workspace = crate::fixtures::AttemptWorkspace::create(None).unwrap();
    let path = workspace.workspace.clone();
    let spec = ai_agents_runtime::spec::AgentSpec::from_yaml_strict(AGENT).unwrap();
    let fixtures = crate::fixtures::LlmFixtureConfig {
        responses: vec!["verified report".into()],
        ..Default::default()
    };
    let (providers, _) =
        crate::fixtures::build_llm_registry(&spec, &fixtures, &workspace.workspace).unwrap();
    let agent = Arc::new(
        ai_agents_runtime::AgentBuilder::from_spec(spec.clone())
            .llm_registry(providers)
            .build()
            .unwrap(),
    );
    let store = Arc::new(ReadFailureStore {
        inner: Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "cleanup.v1".into()).unwrap(),
        ),
        fail: AtomicBool::new(false),
    });
    let runner = Arc::new(
        AutonomyRunner::try_new(
            agent.clone(),
            spec.autonomy,
            Default::default(),
            store.clone(),
            "cleanup.v1".into(),
        )
        .unwrap(),
    );
    let session = TaskHostSession::new(true);
    let mut task = crate::task::TaskContext::with_session(session.clone());
    task.runner_override = Some(runner);
    let hold = workspace.hold_for_tasks();
    let uncertain = Arc::new(AtomicBool::new(false));
    task.workspace_hold = Some(hold.clone());
    task.cleanup_uncertain = Some(uncertain.clone());
    let step = serde_yaml::from_str(
        "operation: {action: run, objective: report}\nassert: {task: {status: completed}}\n",
    )
    .unwrap();
    let (result, error) = task.operate(&agent, &step, 2000).await.unwrap();
    assert!(error.is_none());
    task.capture(&agent, result).await.unwrap();
    assert!(!session.has_protected_work());
    store.fail.store(true, Ordering::Release);
    assert!(task.cleanup().await.is_err());
    assert!(uncertain.load(Ordering::Acquire));
    drop(task);
    assert!(
        hold.load(Ordering::Acquire),
        "absence of a local owner is not cleanup acknowledgement"
    );
    assert!(path.exists());
    store.fail.store(false, Ordering::Release);
    let mut acknowledged = crate::task::TaskContext::with_session(session);
    acknowledged.workspace_hold = Some(hold.clone());
    acknowledged.cleanup_uncertain = Some(uncertain.clone());
    acknowledged.cleanup().await.unwrap();
    assert!(!uncertain.load(Ordering::Acquire));
    drop(acknowledged);
    drop(workspace);
    assert!(!path.exists());
}

#[tokio::test]
async fn inspection_without_event_collection_cannot_prove_event_absence() {
    let result = evaluate(
        AGENT,
        r#"steps:
      - !task
        operation: {action: run, objective: report}
        assert: {task: {status: paused}}
      - !task
        operation: {action: inspect}
        assert: {not: {task: {events_include: [task_completed]}}}
"#,
    )
    .await;
    assert_eq!(result.passed, 0);
    let inspection = &result.scenarios[0].attempts[0].turns[1];
    assert!(!inspection.task.as_ref().unwrap().events_complete);
    assert_eq!(
        inspection
            .evidence
            .task
            .as_ref()
            .unwrap()
            .inspection
            .run
            .counters
            .llm_attempts,
        1
    );
}

#[test]
fn task_jsonl_fields_are_rejected_instead_of_silently_becoming_chat() {
    assert!(
        crate::compatibility::suite_from_jsonl(
            "task".into(),
            r#"{"id":"task","input":"hello","task":{"objective":"work"}}"#
        )
        .is_err()
    );
    assert!(
        crate::compatibility::suite_from_jsonl(
            "ordinary".into(),
            r#"{"id":"chat","input":"hello","unrelated_legacy_field":true}"#
        )
        .is_ok()
    );
}
