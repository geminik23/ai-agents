//! End-to-end consumers use ordinary declared composition, not a substitute task engine.
use crate::{EvalRunner, EvalRunnerOptions};

async fn suite(interactions: &str) -> crate::suite::EvalResult {
    let context = crate::fixtures::AttemptFixtureContext::create(None).unwrap();
    for id in ["first", "second"] {
        std::fs::write(
            context.workspace.join(format!("{id}.yaml")),
            format!(
                r#"
name: {id}
system_prompt: Produce a verified finding.
llm: {{default: {id}}}
tools: []
states:
  initial: work
  states:
    work:
      autonomy:
        enabled: true
        mode: state_until_exit
        max_turns: 4
        max_llm_calls: 4
        completion: {{response_contains: verified}}
        premature_finish: {{action: ask_user, on_unavailable: stop}}
"#
            ),
        )
        .unwrap();
    }
    std::fs::write(
        context.workspace.join("agent.yaml"),
        r#"
name: OrdinaryCoordinator
system_prompt: Coordinate actual findings.
llms:
  first: {provider: openai, model: offline-first}
  second: {provider: openai, model: offline-second}
llm: {default: first}
tools: []
spawner:
  shared_llms: true
  auto_spawn:
    - {id: first, agent: first.yaml}
    - {id: second, agent: second.yaml}
states:
  initial: research
  states:
    research:
      concurrent:
        agents: [first, second]
        aggregation: {strategy: all}
"#,
    )
    .unwrap();
    let path = context.workspace.join("suite.yaml");
    std::fs::write(
        &path,
        format!(
            r#"
name: Independently hosted participants
agent: agent.yaml
settings: {{retries: 0, timeout_per_turn_ms: 2000}}
fixtures:
  llm:
    mode: mock
    responses_by_alias:
      first: [draft, verified-first]
      second: [draft, verified-second]
scenarios:
  - id: original-parent-retains-both-findings
    interactions:
{interactions}
    turns:
      - input: Research both findings.
        assert:
          all:
            - response_contains: verified-first
            - response_contains: verified-second
            - orchestration: {{type: concurrent, agents_include: [first, second]}}
"#
        ),
    )
    .unwrap();
    let result = EvalRunner::from_file(&path, EvalRunnerOptions::default())
        .unwrap()
        .run()
        .await
        .unwrap();
    std::fs::remove_dir_all(context.workspace).unwrap();
    result
}

#[tokio::test]
async fn independent_children_are_answered_before_ordinary_parent_finishes() {
    let result = suite(r#"      - agent: second
        pending_kind: user_question
        operation: {action: resume, response: {kind: answer, value: continue}}
        assert: {task: {status: completed, same_run: true, counters: {path: llm_attempts, eq: 2}, event_result_parity: true}}
      - agent: first
        pending_kind: user_question
        operation: {action: resume, response: {kind: answer, value: continue}}
        assert: {task: {status: completed, same_run: true, counters: {path: llm_attempts, eq: 2}, event_result_parity: true}}"#).await;
    assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
    let turn = &result.scenarios[0].attempts[0].turns[0];
    assert!(
        turn.task.is_none(),
        "ordinary parent must not impersonate a child task"
    );
    assert_eq!(turn.interactions.len(), 2);
    assert_eq!(result.metrics.observed_task_runs, 2);
    assert_eq!(result.metrics.observed_task_llm_attempts, 4);
    assert!(
        turn.interactions
            .iter()
            .all(|result| result.acknowledged_result.is_some())
    );
    let exported = serde_json::to_string(&result).unwrap();
    for secret in [
        "verified-first",
        "verified-second",
        "request_id",
        "owner_token",
        "acknowledged_result",
    ] {
        assert!(!exported.contains(secret), "export leaked {secret}");
    }
}

#[tokio::test]
async fn wrong_kind_is_a_separate_error_and_does_not_consume_the_real_question() {
    let result = suite(
        r#"      - agent: first
        pending_kind: user_question
        operation: {action: resume, response: {kind: approval, decision: {status: approved}}}
        expect_error: task is not resumable
      - agent: first
        pending_kind: user_question
        operation: {action: resume, response: {kind: answer, value: continue}}
        assert: {task: {status: completed, counters: {path: llm_attempts, eq: 2}}}
      - agent: second
        pending_kind: user_question
        operation: {action: resume, response: {kind: answer, value: continue}}
        assert: {task: {status: completed, counters: {path: llm_attempts, eq: 2}}}"#,
    )
    .await;
    assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
    let interactions = &result.scenarios[0].attempts[0].turns[0].interactions;
    let rejected = interactions.iter().find(|r| r.index == 0).unwrap();
    assert!(rejected.runtime_error.is_some());
    assert!(rejected.acknowledged_result.is_none());
    assert!(rejected.task.is_none());
    assert!(!rejected.events_complete);
    assert_eq!(result.metrics.observed_task_llm_attempts, 4);
}

// A long timeout fixture runs in its own process so it does not hold the process-wide environment reader across unrelated writer-lock tests.
#[test]
fn missing_child_answer_cancels_without_erasing_the_completed_sibling() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "interactions_tests::missing_child_answer_worker",
            "--ignored",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
#[ignore = "bounded timeout worker, executed by its process-isolated parent test"]
async fn missing_child_answer_worker() {
    let result = suite(
        r#"      - agent: first
        pending_kind: user_question
        operation: {action: resume, response: {kind: answer, value: continue}}
        assert: {task: {status: completed, counters: {path: llm_attempts, eq: 2}}}"#,
    )
    .await;
    assert_eq!(result.failed, 1);
    assert_eq!(result.metrics.observed_task_runs, 1);
    let interactions: Vec<_> = result.scenarios[0].attempts[0]
        .turns
        .iter()
        .flat_map(|turn| &turn.interactions)
        .collect();
    assert_eq!(interactions.len(), 1);
    assert!(interactions[0].acknowledged_result.is_some());
}

struct DiscoveryStore {
    inner: std::sync::Arc<dyn ai_agents_runtime::autonomy::TaskRunStore>,
    gated: std::sync::atomic::AtomicBool,
    fail: std::sync::atomic::AtomicBool,
    failures: std::sync::atomic::AtomicUsize,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait::async_trait]
impl ai_agents_runtime::autonomy::TaskRunStore for DiscoveryStore {
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
        if self.gated.swap(false, std::sync::atomic::Ordering::SeqCst) {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        self.inner.load(id).await
    }
    async fn mutate(
        &self,
        id: &str,
        mutation: &ai_agents_runtime::autonomy::TaskRunMutation,
    ) -> ai_agents_core::Result<ai_agents_runtime::autonomy::TaskRunSnapshot> {
        if self.fail.load(std::sync::atomic::Ordering::SeqCst) {
            self.failures
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return Err(ai_agents_core::AgentError::Persistence(
                "injected cancellation rejection".into(),
            ));
        }
        self.inner.mutate(id, mutation).await
    }
}

#[tokio::test]
async fn changed_catalogue_rejects_old_selection_and_cancellation_error_does_not_spin() {
    use ai_agents_runtime::autonomy::*;
    use std::sync::{Arc, atomic::Ordering};
    let yaml = "name: SameName\nsystem_prompt: Help\ntools: []\nautonomy:\n  enabled: true\n  max_turns: 3\n  completion: {response_contains: verified}\n  premature_finish: {action: ask_user, on_unavailable: stop}\n";
    let session = TaskHostSession::new(true);
    let mut runners = Vec::new();
    let mut stores = Vec::new();
    let mut results = Vec::new();
    for _ in 0..2 {
        let mut provider = ai_agents_llm::mock::MockLLMProvider::new("offline");
        provider.set_responses(vec!["draft".into(), "verified".into()], false);
        let agent = Arc::new(
            ai_agents_runtime::AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(Arc::new(provider))
                .build()
                .unwrap(),
        );
        let store = Arc::new(DiscoveryStore {
            inner: Arc::new(
                ScopedTaskRunStore::in_memory(agent.info().id, None, "discovery.v1".into())
                    .unwrap(),
            ),
            gated: false.into(),
            fail: false.into(),
            failures: 0.into(),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        });
        let config = ai_agents_runtime::spec::AgentSpec::from_yaml_strict(yaml)
            .unwrap()
            .autonomy;
        let runner = Arc::new(
            AutonomyRunner::try_new(
                agent,
                config,
                Default::default(),
                store.clone(),
                "discovery.v1".into(),
            )
            .unwrap(),
        );
        let result = runner.run("research", None).await.unwrap();
        results.push(result);
        runners.push(runner);
        stores.push(store);
    }
    session
        .observe_result(runners[0].clone(), &results[0])
        .await
        .unwrap();
    let completed_parent = Arc::new(parking_lot::Mutex::new(
        crate::interactions::InteractionState::default(),
    ));
    completed_parent.lock().plan.push(serde_yaml::from_str("agent: SameName\npending_kind: user_question\noperation: {action: resume, response: {kind: answer, value: continue}}\nassert: {task: {status: completed}}\n").unwrap());
    crate::interactions::service(
        &session,
        std::future::ready(()),
        completed_parent.clone(),
        true,
        usize::MAX,
    )
    .await
    .unwrap();
    assert!(
        completed_parent.lock().claimed.is_empty(),
        "a completed foreground parent cannot authorize a new scripted response"
    );
    assert_eq!(
        runners[0]
            .task_display(&results[0].run.key.run_id)
            .await
            .unwrap()
            .summary
            .status,
        TaskRunStatus::Paused
    );
    stores[0].gated.store(true, Ordering::SeqCst);
    for store in &stores {
        store.fail.store(true, Ordering::SeqCst);
    }
    let state = Arc::new(parking_lot::Mutex::new(
        crate::interactions::InteractionState::default(),
    ));
    state.lock().plan.push(serde_yaml::from_str("agent: SameName\npending_kind: user_question\noperation: {action: resume, response: {kind: answer, value: continue}}\nassert: {task: {status: completed}}\n").unwrap());
    let work = async {
        stores[0].entered.acquire().await.unwrap().forget();
        session
            .observe_result(runners[1].clone(), &results[1])
            .await
            .unwrap();
        stores[0].release.add_permits(1);
        std::future::pending::<()>().await;
    };
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(50),
            crate::interactions::service(&session, work, state.clone(), true, usize::MAX)
        )
        .await
        .is_err()
    );
    assert!(
        state.lock().claimed.is_empty(),
        "stale unique discovery must not dispatch after a matching participant joins"
    );
    assert!(state.lock().error.is_some());
    assert!(state.lock().cancellation_failed);
    assert_eq!(
        stores
            .iter()
            .map(|store| store.failures.load(Ordering::SeqCst))
            .sum::<usize>(),
        1,
        "an immediate storage error must not become an automatic cancellation retry loop"
    );
    let mut recovery_required = false;
    for (runner, store) in runners.iter().zip(&stores) {
        store.fail.store(false, Ordering::SeqCst);
        recovery_required |= runner
            .request_cancel(&runner.task_store().list().await.unwrap()[0].key.run_id)
            .await
            .is_err();
    }
    // A storage rejection without a certified no-write receipt remains uncertain even after the injected failure is disabled.
    assert!(recovery_required);
    assert!(session.has_protected_work());
}

#[test]
fn configuration_rejects_unasserted_or_authority_creating_interactions() {
    for script in [
        "agent: child\npending_kind: user_question\noperation: {action: run, objective: hidden}\nexpect_error: refused",
        "agent: child\npending_kind: user_question\noperation: {action: resume, response: {kind: continue}}",
        "agent: child\npending_kind: user_question\noperation: {action: resume, response: {kind: continue}}\nassert: {llm_request: {count: 0}}",
    ] {
        let script = serde_yaml::from_str(script).unwrap();
        assert!(crate::interactions::validate(&script, 0).is_err());
    }
}
