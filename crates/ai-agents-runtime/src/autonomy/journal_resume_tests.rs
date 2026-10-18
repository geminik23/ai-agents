use super::evaluation_tests::{bound_script, current, identity};
use super::*;
use ai_agents_core::{AgentError, ToolCancellationToken};
use async_trait::async_trait;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct HostedExecutor {
    runtime: RuntimeObservationExecutor,
    attempts: AtomicUsize,
}
#[async_trait]
impl ValidationObservationExecutor for HostedExecutor {
    // Two host review boundaries occur before one actual shared-executor invocation.
    async fn execute(
        &self,
        request: &ValidationObservationRequest,
        identity: &EvidenceIdentity,
    ) -> ai_agents_core::Result<ObservationResult> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        if attempt < 2 {
            return Err(AgentError::TaskSuspended(format!("host-review-{attempt}")));
        }
        self.runtime.execute(request, identity).await
    }
    fn take_suspension(
        &self,
        request: &ValidationObservationRequest,
        identity: &EvidenceIdentity,
        before: &EvaluationScope,
    ) -> ai_agents_core::Result<Option<ValidationObservationPause>> {
        let call = super::authority::observation_tool_request(request, identity)?;
        let request_id = format!("host-review-{}", self.attempts.load(Ordering::SeqCst) - 1);
        let review = super::suspension::BatchApproval {
            call_id: call.call_id.clone(),
            request_id: request_id.clone(),
            trigger: serde_json::Value::Null,
            context: call.arguments.clone(),
            message: "Review observation".into(),
            timeout_millis: None,
            expires_at: None,
            question: None,
        };
        Ok(Some(ValidationObservationPause {
            observation: request.id().into(),
            request: call,
            pending: TaskPendingRequest {
                id: request_id,
                issued_revision: self.runtime.run_revision.load(Ordering::SeqCst) + 1,
                kind: TaskPendingKind::Approval,
                reviewed_action: serde_json::to_value(review)?,
            },
            executor: None,
            before: before.clone(),
            authorized: Vec::new(),
            group: None,
        }))
    }
}

// Standalone conditional journals count logical observations, unlike the runner's real-invocation accounting mode.
async fn contract(storage: Arc<dyn ai_agents_core::AgentStorage>) {
    let (bound, mut scope, _) = bound_script("complete");
    let check = &bound.checks[0];
    struct Probe;
    #[async_trait]
    impl ai_agents_core::Tool for Probe {
        fn id(&self) -> &str {
            "probe"
        }
        fn name(&self) -> &str {
            "probe"
        }
        fn description(&self) -> &str {
            "Independent SDK observation"
        }
        fn input_schema(&self) -> serde_json::Value {
            json!({"type":"object"})
        }
        fn safety_metadata(&self) -> ai_agents_core::ToolSafetyMetadata {
            ai_agents_core::ToolSafetyMetadata {
                read_only: true,
                concurrency_safe: true,
                ..Default::default()
            }
        }
        async fn execute(
            &self,
            _: serde_json::Value,
            _: ai_agents_core::ToolExecutionContext,
        ) -> ai_agents_core::ToolResult {
            ai_agents_core::ToolResult::ok("observed")
        }
    }
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(super::runner_tests::RecordingProvider::new(&["unused"]))
            .tool(Arc::new(Probe))
            .build()
            .unwrap(),
    );
    scope.key.agent_id = agent.info().id;
    let store = Arc::new(
        ScopedTaskRunStore::new(
            storage,
            scope.key.agent_id.clone(),
            None,
            "config-v1".into(),
        )
        .unwrap(),
    );
    let mut snapshot = super::tests::checkpoint("run");
    let mut payload: TaskCheckpointPayload =
        serde_json::from_value(snapshot.payload.clone()).unwrap();
    payload.limits.max_tool_calls = 1;
    snapshot.payload = serde_json::to_value(payload).unwrap();
    store.create(&snapshot).await.unwrap();
    let claimed = store
        .mutate(
            "run",
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let journal = TaskValidationJournal::new(
        store.clone(),
        "run".into(),
        "owner".into(),
        claimed.revision,
    );
    let executor = HostedExecutor {
        runtime: RuntimeObservationExecutor {
            agent,
            profile: None,
            scope_mode: "task".into(),
            run_revision: journal.revision_handle(),
            validator: check.check.adapter.clone(),
            contract_version: 1,
        },
        attempts: AtomicUsize::new(0),
    };
    let mut identity = identity(&scope);
    identity.config_identity = check.config_identity.clone();
    let mut driver = ValidationDriverState::new(check, identity).unwrap();
    let evidence = current(&scope);
    let cancellation = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let context = ValidationDriveContext {
        scope: &scope,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancellation,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(10),
    };
    for _ in 0..2 {
        assert!(matches!(
            driver.drive(check, &context).await.unwrap(),
            ValidationDriverOutcome::Suspended
        ));
        let snapshot = store.load("run").await.unwrap().unwrap();
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
        assert_eq!(payload.counters.tool_attempts, 1);
        assert_eq!(payload.reservations.len(), 1);
        assert_eq!(payload.reservations[0].state, TaskEffectState::Suspended);
        let pending = payload.pending.take().unwrap();
        payload.consumed_request_ids.push(pending.id);
        let saved = store
            .mutate(
                "run",
                &TaskRunMutation::Checkpoint {
                    expected_revision: snapshot.revision,
                    owner_token: "owner".into(),
                    status: TaskRunStatus::Running,
                    payload: serde_json::to_value(payload).unwrap(),
                    release: false,
                },
            )
            .await
            .unwrap();
        journal
            .revision_handle()
            .store(saved.revision, Ordering::SeqCst);
        driver = ValidationDriverState::restore_retained(
            serde_json::to_value(&driver).unwrap(),
            &driver,
            check,
            &scope,
            &bound.extensions,
        )
        .unwrap();
    }
    assert!(matches!(
        driver.drive(check, &context).await.unwrap(),
        ValidationDriverOutcome::Complete(_)
    ));
    let snapshot = store.load("run").await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert_eq!(payload.counters.tool_attempts, 1);
    assert_eq!(payload.reservations.len(), 1);
    assert_eq!(payload.reservations[0].state, TaskEffectState::Completed);
    assert_eq!(executor.attempts.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn journal_resume_preserves_one_logical_charge_across_host_repauses() {
    contract(Arc::new(ai_agents_storage::InMemoryTaskStorage::default())).await;
}
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn journal_resume_sqlite_preserves_one_logical_charge_across_host_repauses() {
    contract(Arc::new(
        ai_agents_storage::SqliteStorage::in_memory().await.unwrap(),
    ))
    .await;
}
