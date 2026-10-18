use super::*;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use std::sync::atomic::AtomicBool;

struct FaultStore {
    inner: Arc<ScopedTaskRunStore>,
    phase: &'static str,
    mode: &'static str,
    armed: AtomicBool,
    consumed_writes: std::sync::atomic::AtomicUsize,
    reached: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
impl FaultStore {
    /// A single chosen protocol boundary distinguishes pre-write refusal from lost or interrupted acknowledgement.
    fn new(inner: Arc<ScopedTaskRunStore>, phase: &'static str, mode: &'static str) -> Arc<Self> {
        Arc::new(Self {
            inner,
            phase,
            mode,
            armed: AtomicBool::new(false),
            consumed_writes: std::sync::atomic::AtomicUsize::new(0),
            reached: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        })
    }
    /// Corrupts only the receipt after the actual backend operation, never the stored authoritative snapshot.
    async fn receipt(&self, mut saved: TaskRunSnapshot) -> Result<TaskRunSnapshot> {
        self.reached.add_permits(1);
        match self.mode {
            "lost" => Err(AgentError::Persistence(
                "lost control acknowledgement".into(),
            )),
            "stale" => {
                saved.revision -= 1;
                Ok(saved)
            }
            "drop" => std::future::pending().await,
            "hold" => {
                self.release.acquire().await.unwrap().forget();
                Ok(saved)
            }
            _ => Ok(saved),
        }
    }
}
#[async_trait]
impl TaskRunStore for FaultStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.inner.delete(id, revision).await
    }
    fn supports_objective_revision(&self) -> bool {
        true
    }
    fn supports_paused_limit_revision(&self) -> bool {
        true
    }
    async fn revise_paused_limits(
        &self,
        id: &str,
        revision: u64,
        limits: TaskRunLimits,
        reason: &str,
        host: &AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        self.inner
            .revise_paused_limits(id, revision, limits, reason, host, deadline)
            .await
    }
    async fn apply_host_control(
        &self,
        id: &str,
        revision: u64,
        action: HostControlAction,
        host: &AutonomyHostCeilings,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<TaskRunSnapshot>> {
        let selected = self.phase == "objective" && self.armed.swap(false, Ordering::SeqCst);
        if selected && self.mode == "reject" {
            self.reached.add_permits(1);
            return Ok(None);
        }
        let saved = self
            .inner
            .apply_host_control(id, revision, action, host, deadline)
            .await?;
        if selected {
            Ok(Some(self.receipt(saved.unwrap()).await?))
        } else {
            Ok(saved)
        }
    }
    async fn claim_paused(
        &self,
        id: &str,
        revision: u64,
        token: String,
    ) -> Result<Option<TaskRunSnapshot>> {
        if self.phase == "claim"
            && self.armed.load(Ordering::SeqCst)
            && matches!(self.mode, "no_write" | "cancel")
        {
            self.armed.store(false, Ordering::SeqCst);
            if self.mode == "cancel" {
                self.inner
                    .mutate(
                        id,
                        &TaskRunMutation::RequestCancel {
                            expected_revision: revision,
                        },
                    )
                    .await?;
            }
            return Ok(None);
        }
        self.mutate(
            id,
            &TaskRunMutation::Claim {
                expected_revision: revision,
                owner_token: token,
            },
        )
        .await
        .map(Some)
    }
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let matches = match (self.phase, mutation) {
            ("claim", TaskRunMutation::Claim { .. }) => true,
            (
                "pause",
                TaskRunMutation::FinalCheckpoint {
                    status: TaskRunStatus::Paused,
                    ..
                },
            ) => true,
            ("consume", TaskRunMutation::Checkpoint { payload, .. }) => {
                payload["consumed_request_ids"]
                    .as_array()
                    .is_some_and(|ids| !ids.is_empty())
            }
            ("stage_apply", TaskRunMutation::Checkpoint { payload, .. }) => {
                payload["consumed_request_ids"]
                    .as_array()
                    .is_some_and(|ids| !ids.is_empty())
                    && self.consumed_writes.fetch_add(1, Ordering::SeqCst) == 1
            }
            ("adopt", TaskRunMutation::Checkpoint { payload, .. }) => {
                payload["evidence"]["records"]
                    .as_array()
                    .and_then(|records| records.last())
                    .is_some_and(|record| record["origin"] == "runtime.plan.adopt")
            }
            _ => false,
        };
        let selected = matches && self.armed.swap(false, Ordering::SeqCst);
        if selected && self.mode == "reject" {
            self.reached.add_permits(1);
            return Err(AgentError::Persistence(
                "rejected control checkpoint".into(),
            ));
        }
        let saved = self.inner.mutate(id, mutation).await?;
        if selected {
            self.receipt(saved).await
        } else {
            Ok(saved)
        }
    }
}

/// Root and child executable plan reviews preserve authority and original deadlines through every acknowledgement wait.
#[tokio::test]
async fn host_control_failure_plan_adoption_rechecks_binding_and_deadline_at_root_and_child() {
    for child in [false, true] {
        for phase in ["claim", "consume", "adopt"] {
            for expired in [false, true] {
                let plan = r#"{"steps":[{"id":"one","description":"Answer","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
                let provider = RecordingProvider::new(&[plan, "child done"]);
                let yaml = "name: PlanReviewRace\nsystem_prompt: test\nreasoning:\n  mode: plan_and_execute\nhitl:\n  default_timeout_seconds: 1\n  on_timeout: reject\n";
                let worker = AgentBuilder::from_yaml(yaml)
                    .unwrap()
                    .llm(provider.clone())
                    .auto_configure_features()
                    .unwrap()
                    .build()
                    .unwrap();
                let (root, reviewed) = if child {
                    let registry = Arc::new(crate::spawner::AgentRegistry::new());
                    registry
                        .register(crate::spawner::SpawnedAgent::from_runtime(
                            "worker".into(),
                            worker,
                            crate::spec::AgentSpec::from_yaml_strict(yaml).unwrap(),
                        ))
                        .await
                        .unwrap();
                    let reviewed = registry.get("worker").unwrap();
                    let marker = ai_agents_core::encode_native_tool_call_markers(
                        &[ai_agents_core::ToolCall {
                            id: "send".into(),
                            name: "send_agent_message".into(),
                            arguments: json!({"to":"worker","message":"objective"}),
                        }],
                        None,
                    )
                    .unwrap();
                    let root_provider = RecordingProvider::new(&[&marker, "done"]);
                    let parent = AgentBuilder::from_yaml(
                        "name: ParentReview\nsystem_prompt: parent\ntools: [send_agent_message]\n",
                    )
                    .unwrap()
                    .llm(root_provider)
                    .auto_configure_features()
                    .unwrap()
                    .tool(Arc::new(crate::spawner::tools::SendMessageTool::new(
                        registry.clone(),
                        "parent",
                    )))
                    .build()
                    .unwrap()
                    .with_spawner_handles(Arc::new(crate::spawner::AgentSpawner::new()), registry);
                    (Arc::new(parent), reviewed)
                } else {
                    let reviewed = Arc::new(worker);
                    (reviewed.clone(), reviewed)
                };
                let inner = support::store(root.info().id, "plan-race.v1", false).await;
                let store = FaultStore::new(inner, phase, "hold");
                let mut config = super::super::runner_tests::config(8);
                config.defaults.max_tool_calls = Some(8);
                config.defaults.supervision = Some(SupervisionConfig {
                    stage_changes: None,
                    plan_changes: Some(ReviewPolicy::RequireApproval),
                });
                config.defaults.hitl = Some(AutonomyHitlConfig {
                    on_user_question: Some(InteractionAction::PauseRun),
                    on_approval_required: Some(InteractionAction::PauseRun),
                });
                let runner = AutonomyRunner::try_new(
                    root.clone(),
                    config,
                    Default::default(),
                    store.clone(),
                    "plan-race.v1".into(),
                )
                .unwrap();
                let first = runner.run("objective", None).await.unwrap();
                assert_eq!(first.run.status, TaskRunStatus::Paused);
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                store.armed.store(true, Ordering::SeqCst);
                let mut future = Box::pin(approve(&runner, &first.run));
                tokio::select! { result = &mut future => panic!("returned before selected wait child={child} phase={phase}: {result:?}"), permit = tokio::time::timeout(std::time::Duration::from_secs(5), store.reached.acquire()) => { permit.unwrap().unwrap().forget(); } }
                if expired {
                    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
                } else {
                    reviewed.runtime_control().set_emergency_deny(true);
                }
                store.release.add_permits(1);
                let result = future.await;
                if expired {
                    let done = result.unwrap();
                    assert_eq!(
                        done.run.status,
                        TaskRunStatus::Incomplete,
                        "child={child} phase={phase} reason={:?}",
                        done.run.stop_reason
                    );
                } else {
                    match result {
                        Ok(done) if child => {
                            assert_eq!(
                                done.run.status,
                                TaskRunStatus::Failed,
                                "phase={phase} calls={} reason={:?}",
                                provider.calls.load(Ordering::SeqCst),
                                done.run.stop_reason
                            );
                            assert_eq!(
                                done.run.stop_reason.as_deref(),
                                Some("required_child_failure")
                            );
                        }
                        Err(_) => assert!(root.set_context("unrelated", json!(true)).is_err()),
                        other => {
                            panic!("stale review executed child={child} phase={phase}: {other:?}")
                        }
                    }
                }
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    1,
                    "child={child} phase={phase}"
                );
            }
        }
    }
}

/// Stage credit is still provisional when the post-consumption controller checkpoint crosses its original deadline.
#[tokio::test]
async fn host_control_failure_stage_deadline_is_checked_at_actual_application() {
    let provider = RecordingProvider::new(&["done", "done"]);
    let agent = Arc::new(AgentBuilder::from_yaml("name: ApplyStage\nsystem_prompt: test\nhitl:\n  default_timeout_seconds: 1\n  on_timeout: reject\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().build().unwrap());
    let inner = support::store(agent.info().id, "host.v1", false).await;
    let store = FaultStore::new(inner, "stage_apply", "hold");
    let runner = runner(agent.clone(), store.clone());
    let first = runner.run("objective", None).await.unwrap();
    store.armed.store(true, Ordering::SeqCst);
    let mut future = Box::pin(approve(&runner, &first.run));
    tokio::select! { result = &mut future => panic!("returned before application: {result:?}"), permit = tokio::time::timeout(std::time::Duration::from_secs(5), store.reached.acquire()) => { permit.unwrap().unwrap().forget(); } }
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    store.release.add_permits(1);
    let done = future.await.unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Incomplete);
    assert_eq!(done.run.stop_reason.as_deref(), Some("supervision_timeout"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(agent.set_context("released", json!(true)).is_ok());
}

/// Explicit host timeout settlement preserves configured approval without granting late ordinary approvals or future proposals.
#[tokio::test]
async fn host_control_failure_explicit_timeout_policy_approval_is_bound_to_exact_review() {
    for executable in [false, true] {
        let plan = r#"{"steps":[{"id":"one","description":"Answer","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
        let replies = [if executable { plan } else { "done" }, "done"];
        let provider = RecordingProvider::new(&replies);
        let yaml = format!(
            "name: TimeoutApproval\nsystem_prompt: test\nhitl:\n  default_timeout_seconds: 1\n  on_timeout: approve\n{}",
            if executable {
                "reasoning:\n  mode: plan_and_execute\n"
            } else {
                ""
            }
        );
        let agent = Arc::new(
            AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "timeout.v1", false).await;
        let mut config = super::super::runner_tests::config(8);
        config.defaults.supervision = Some(SupervisionConfig {
            stage_changes: if executable {
                None
            } else {
                Some(ReviewPolicy::RequireApproval)
            },
            plan_changes: if executable {
                Some(ReviewPolicy::RequireApproval)
            } else {
                None
            },
        });
        if !executable {
            config.defaults.lifecycle =
                Some(serde_json::from_value(json!([{"id":"work"}])).unwrap());
        }
        let runner = AutonomyRunner::try_new(
            agent,
            config,
            Default::default(),
            store,
            "timeout.v1".into(),
        )
        .unwrap();
        let first = runner.run("objective", None).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(approve(&runner, &first.run).await.is_err());
        let done = runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Approval {
                    request_id: first.run.pending.unwrap().id,
                    result: ApprovalResult::Timeout,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "executable={executable} reason={:?}",
            done.run.stop_reason
        );
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            if executable { 2 } else { 1 }
        );
    }
}

/// Claim, consumption and re-pause acknowledgement faults never install reusable execution authority.
#[tokio::test]
async fn host_control_failure_review_receipts_and_drop_preserve_protection() {
    for sqlite in support::backends() {
        for phase in ["claim", "consume", "pause"] {
            for mode in ["reject", "lost", "stale", "drop"] {
                let (agent, _, inner, provider) = fixture(*sqlite).await;
                let store = FaultStore::new(inner, phase, mode);
                let runner = runner(agent.clone(), store.clone());
                let first = runner.run("objective", None).await.unwrap();
                store.armed.store(true, Ordering::SeqCst);
                let mut future = Box::pin(approve(&runner, &first.run));
                if mode == "drop" {
                    tokio::select! { result = &mut future => panic!("returned before drop: {result:?}"), permit = tokio::time::timeout(std::time::Duration::from_secs(5), store.reached.acquire()) => { permit.unwrap().unwrap().forget(); } }
                    drop(future);
                } else {
                    assert!(future.await.is_err(), "phase={phase} mode={mode}");
                }
                assert!(!store.armed.load(Ordering::SeqCst));
                assert!(agent.set_context("unrelated", json!(true)).is_err());
                assert!(approve(&runner, &first.run).await.is_err());
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    if phase == "pause" { 2 } else { 1 }
                );
            }
        }
    }
}

/// Objective cleanup and commit faults cannot revive old requests or silently execute new work.
#[tokio::test]
async fn host_control_failure_objective_handoff_receipts_preserve_protection() {
    for sqlite in support::backends() {
        for phase in ["claim", "consume", "objective", "pause"] {
            for mode in ["reject", "lost", "stale", "drop"] {
                let (agent, _, inner, provider) = fixture(*sqlite).await;
                let store = FaultStore::new(inner, phase, mode);
                let runner = runner(agent.clone(), store.clone());
                let first = runner.run("old", None).await.unwrap();
                store.armed.store(true, Ordering::SeqCst);
                let mut future = Box::pin(runner.modify_objective_paused(
                    &first.run.key.run_id,
                    first.run.revision,
                    "new",
                    "revise",
                ));
                if mode == "drop" {
                    tokio::select! { result = &mut future => panic!("returned before drop: {result:?}"), permit = tokio::time::timeout(std::time::Duration::from_secs(5), store.reached.acquire()) => { permit.unwrap().unwrap().forget(); } }
                    drop(future);
                } else {
                    assert!(future.await.is_err(), "phase={phase} mode={mode}");
                }
                assert!(!store.armed.load(Ordering::SeqCst));
                assert!(agent.set_context("unrelated", json!(true)).is_err());
                let saved = store.load(&first.run.key.run_id).await.unwrap().unwrap();
                assert!(
                    runner
                        .resume(&saved.key.run_id, saved.revision, TaskResumeInput::Continue)
                        .await
                        .is_err()
                );
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            }
        }
    }
}

/// A certified no-write refusal preserves the pause; cancellation winning that CAS remains normally cleanable.
#[tokio::test]
async fn host_control_failure_certified_claim_refusal_preserves_owner_and_cancel_cleanup() {
    for operation in ["objective", "approve"] {
        for mode in ["no_write", "cancel"] {
            let (agent, _, inner, provider) = fixture(false).await;
            let store = FaultStore::new(inner, "claim", mode);
            let runner = runner(agent.clone(), store.clone());
            let first = runner.run("old", None).await.unwrap();
            store.armed.store(true, Ordering::SeqCst);
            let result = if operation == "objective" {
                runner
                    .modify_objective_paused(
                        &first.run.key.run_id,
                        first.run.revision,
                        "new",
                        "revise",
                    )
                    .await
            } else {
                approve(&runner, &first.run).await
            };
            assert!(result.is_err());
            if mode == "cancel" {
                let saved = store.load(&first.run.key.run_id).await.unwrap().unwrap();
                let done = runner
                    .cancel_paused(&saved.key.run_id, saved.revision)
                    .await
                    .unwrap();
                assert_eq!(done.run.status, TaskRunStatus::Cancelled);
                assert!(agent.set_context("released", json!(true)).is_ok());
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            } else {
                assert_eq!(
                    approve(&runner, &first.run).await.unwrap().run.status,
                    TaskRunStatus::Paused
                );
            }
        }
    }
}

/// Approval arriving before expiry cannot apply after delayed claim acknowledgement crosses its original deadline.
#[tokio::test]
async fn host_control_failure_review_deadline_is_rechecked_after_claim_acknowledgement() {
    let provider = RecordingProvider::new(&["done", "done"]);
    let agent = Arc::new(AgentBuilder::from_yaml("name: TimedReview\nsystem_prompt: test\nhitl:\n  default_timeout_seconds: 1\n  on_timeout: reject\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().build().unwrap());
    let inner = support::store(agent.info().id, "host.v1", false).await;
    let store = FaultStore::new(inner, "claim", "hold");
    let runner = runner(agent.clone(), store.clone());
    let first = runner.run("old", None).await.unwrap();
    store.armed.store(true, Ordering::SeqCst);
    let mut future = Box::pin(approve(&runner, &first.run));
    tokio::select! { result = &mut future => panic!("returned before acknowledgement: {result:?}"), permit = tokio::time::timeout(std::time::Duration::from_secs(5), store.reached.acquire()) => { permit.unwrap().unwrap().forget(); } }
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    store.release.add_permits(1);
    let stopped = future.await.unwrap();
    assert_eq!(stopped.run.status, TaskRunStatus::Incomplete);
    assert_eq!(
        stopped.run.stop_reason.as_deref(),
        Some("supervision_timeout")
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(agent.set_context("released", json!(true)).is_ok());
}

/// Limit revision and approval race on one revision; the winner does not auto-consume the loser's response.
#[tokio::test]
async fn host_control_failure_response_limit_and_objective_races_have_one_winner() {
    for sqlite in support::backends() {
        for objective in [false, true] {
            let (_, runner, _, provider) = fixture(*sqlite).await;
            let first = runner.run("old", None).await.unwrap();
            let mut limits = first.run.limits.clone();
            limits.max_turns += 1;
            let control = async {
                if objective {
                    runner
                        .modify_objective_paused(
                            &first.run.key.run_id,
                            first.run.revision,
                            "new",
                            "revise",
                        )
                        .await
                } else {
                    runner
                        .revise_limits_paused(
                            &first.run.key.run_id,
                            first.run.revision,
                            limits,
                            "capacity",
                        )
                        .await
                }
            };
            let (control, response) = tokio::join!(control, approve(&runner, &first.run));
            assert_ne!(control.is_ok(), response.is_ok());
            assert!(approve(&runner, &first.run).await.is_err());
            assert!(provider.calls.load(Ordering::SeqCst) <= 2);
        }
    }
}
