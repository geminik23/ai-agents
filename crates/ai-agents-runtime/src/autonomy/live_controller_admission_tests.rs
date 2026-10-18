use super::*;
use ai_agents_core::autonomy::{ProviderCostBound, ProviderCostSettlement, ProviderRequest};
use ai_agents_core::{ChatMessage, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse};

struct DelayedClaim(Arc<ScopedTaskRunStore>);
#[async_trait]
impl TaskRunStore for DelayedClaim {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.0.create(snapshot).await
    }
    async fn load(&self, id: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        self.0.load(id).await
    }
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.0.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.0.delete(id, revision).await
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
    ) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        self.0
            .revise_paused_limits(id, revision, limits, reason, host, deadline)
            .await
    }
    // A committed claim can arrive after the original check deadline without granting a fresh observation window.
    async fn mutate(
        &self,
        id: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        let saved = self.0.mutate(id, mutation).await?;
        if matches!(mutation, TaskRunMutation::Claim { .. }) {
            tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        }
        Ok(saved)
    }
}

#[tokio::test]
async fn live_controller_capacity_claim_cannot_extend_the_original_judge_deadline() {
    for sqlite in support::backends() {
        let provider = super::super::super::runner_tests::RecordingProvider::new(&["done"]);
        let (agent, _, _) = super::super::super::runner_tests::fixture(provider.clone(), 1);
        let store = support::store(agent.info().id, "prepared-v1", *sqlite).await;
        let mut config = super::super::super::runner_tests::config(1);
        config.defaults.validation = Some(serde_yaml::from_str("checks: [{id: quality, adapter: builtin.judge, timeout_seconds: 1, config: {criteria: [report quality]}}]").unwrap());
        config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
        let runner = live_runner(
            agent.clone(),
            Arc::new(DelayedClaim(store.clone())),
            config,
            "prepared-v1",
        );
        let paused = runner.run("report", None).await.unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let mut limits = payload.limits;
        limits.max_llm_calls = 2;
        let revised = runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                paused.run.revision,
                limits,
                "allow judge",
            )
            .await
            .unwrap();
        let done = runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::LimitReached);
        assert_eq!(done.run.stop_reason.as_deref(), Some("validation_timeout"));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(agent.set_context("released", json!(true)).is_ok());
    }
}

struct Priced {
    inner: Arc<super::super::super::runner_tests::RecordingProvider>,
}
#[async_trait]
impl LLMProvider for Priced {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.inner.complete(messages, config).await
    }
    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<
            dyn futures::Stream<Item = std::result::Result<ai_agents_core::LLMChunk, LLMError>>
                + Unpin
                + Send,
        >,
        LLMError,
    > {
        self.inner.complete_stream(messages, config).await
    }
    fn provider_name(&self) -> &str {
        "priced-capacity-test"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
    fn priced_capability_identity(&self) -> Option<String> {
        Some("test.price.v1".into())
    }
    fn request_cost_bound(
        &self,
        request: &ProviderRequest<'_>,
    ) -> std::result::Result<Option<ProviderCostBound>, LLMError> {
        Ok(Some(ProviderCostBound {
            provider_identity: "test.price.v1".into(),
            pricing_identity: "fixed.v1".into(),
            request_id: request.request_id.into(),
            max_micro_usd: 10,
        }))
    }
    fn settle_request_cost(
        &self,
        bound: &ProviderCostBound,
        _: &LLMResponse,
    ) -> std::result::Result<Option<ProviderCostSettlement>, LLMError> {
        Ok(Some(ProviderCostSettlement {
            bound: bound.clone(),
            charged_micro_usd: 10,
        }))
    }
}

#[tokio::test]
async fn live_controller_actual_judge_cost_refusal_has_safe_logical_custody_and_no_refund_or_work_replay()
 {
    for sqlite in support::backends() {
        for disposition in ["resume", "cancel", "unavailable", "objective"] {
            let provider = super::super::super::runner_tests::RecordingProvider::new(&[
                "done",
                "{\"overall_score\":1.0}",
            ]);
            let agent = Arc::new(
                AgentBuilder::new()
                    .system_prompt("report")
                    .llm(Arc::new(Priced {
                        inner: provider.clone(),
                    }))
                    .build()
                    .unwrap(),
            );
            let store = support::store(agent.info().id, "priced.v1", *sqlite).await;
            let mut config = super::super::super::runner_tests::config(4);
            config.defaults.max_cost_usd = Some(UsdAmount::parse("0.000010").unwrap());
            config.defaults.validation = Some(serde_yaml::from_str("checks:\n  - id: quality\n    adapter: builtin.judge\n    config: {criteria: [report quality]}").unwrap());
            config.defaults.on_limit =
                Some(serde_yaml::from_str("action: pause_run\non_unavailable: fail").unwrap());
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                config,
                AutonomyHostCeilings {
                    max_cost_usd: Some(UsdAmount::parse("0.000030").unwrap()),
                    ..Default::default()
                },
                store.clone(),
                "priced.v1".into(),
            )
            .unwrap();
            let paused = runner.run("report", None).await.unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{disposition}: {:?}",
                paused.run.stop_reason
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(paused.run.counters.llm_attempts, 1);
            assert_eq!(paused.run.counters.charged_micro_usd, 10);
            let done = if disposition == "cancel" {
                runner
                    .cancel_paused(&paused.run.key.run_id, paused.run.revision)
                    .await
                    .unwrap()
            } else if disposition == "objective" {
                let revised = runner
                    .modify_objective_paused(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        "new report objective",
                        "host changed the task",
                    )
                    .await;
                assert!(
                    revised.is_ok(),
                    "{revised:?}; old_revision={} latest_revision={} pending={:?}",
                    paused.run.revision,
                    store
                        .load(&paused.run.key.run_id)
                        .await
                        .unwrap()
                        .unwrap()
                        .revision,
                    store
                        .load(&paused.run.key.run_id)
                        .await
                        .unwrap()
                        .unwrap()
                        .payload["pending"]
                );
                let revised = revised.unwrap();
                assert_eq!(revised.run.status, TaskRunStatus::Paused);
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                runner
                    .stop_paused(
                        &revised.run.key.run_id,
                        revised.run.revision,
                        "stop revised task",
                    )
                    .await
                    .unwrap()
            } else if disposition == "unavailable" {
                runner
                    .controller_unavailable_paused(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        &paused.run.pending.as_ref().unwrap().id,
                        "delivery lost",
                    )
                    .await
                    .unwrap()
            } else {
                let payload: TaskCheckpointPayload = serde_json::from_value(
                    store
                        .load(&paused.run.key.run_id)
                        .await
                        .unwrap()
                        .unwrap()
                        .payload,
                )
                .unwrap();
                let mut limits = payload.limits;
                limits.max_micro_usd = Some(20);
                let revised = runner
                    .revise_limits_paused(
                        &paused.run.key.run_id,
                        paused.run.revision,
                        limits,
                        "allow required judge",
                    )
                    .await
                    .unwrap();
                runner
                    .resume(
                        &revised.run.key.run_id,
                        revised.run.revision,
                        TaskResumeInput::Continue,
                    )
                    .await
                    .unwrap()
            };
            assert_eq!(
                done.run.status,
                match disposition {
                    "resume" => TaskRunStatus::Completed,
                    "cancel" => TaskRunStatus::Cancelled,
                    "objective" => TaskRunStatus::Incomplete,
                    _ => TaskRunStatus::Failed,
                }
            );
            assert_eq!(done.run.counters.turns, 1);
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                if disposition == "resume" { 2 } else { 1 }
            );
            assert_eq!(
                done.run.counters.charged_micro_usd,
                if disposition == "resume" { 20 } else { 10 }
            );
            assert!(agent.set_context("released", json!(true)).is_ok());
        }
    }
}

struct Writes(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Writes {
    fn id(&self) -> &str {
        "writes"
    }
    fn name(&self) -> &str {
        "writes"
    }
    fn description(&self) -> &str {
        "Test an actual footprint admission"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: false,
            ..Default::default()
        }
    }
    fn declared_write_footprint(
        &self,
        _: &Value,
        _: &ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ToolWriteFootprint>> {
        Ok(Some(ToolWriteFootprint {
            binding_identity: "writes.v1".into(),
            targets: vec!["a".into(), "b".into()],
        }))
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("written")
    }
}
struct WriteValidator(AdapterDescriptor);
impl AutonomyValidator for WriteValidator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.0
    }
    fn validate_config(&self, _: &Value) -> ai_agents_core::Result<()> {
        Ok(())
    }
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        if input.observations.is_empty() {
            return Ok(ValidationDecision::NeedObservations {
                requests: vec![ValidationObservationRequest::Tool {
                    id: "writes".into(),
                    tool: "writes".into(),
                    arguments: json!({}),
                    host_binding: None,
                }],
                checkpoint: json!({}),
            });
        }
        Ok(ValidationDecision::Complete {
            outcome: GateOutcome::Pass,
            reason: "observed".into(),
            metrics: json!({}),
            evidence_refs: vec![],
        })
    }
}

#[tokio::test]
async fn live_controller_validator_actual_footprint_capacity_reuses_the_exact_observation() {
    for sqlite in support::backends() {
        let provider = super::super::super::runner_tests::RecordingProvider::new(&["done"]);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut extensions = AutonomyExtensions::builtins();
        extensions
            .register_validator(Arc::new(WriteValidator(AdapterDescriptor {
                id: "host.writes".into(),
                contract_version: 1,
                tools: vec!["writes".into()],
                ..Default::default()
            })))
            .unwrap();
        let agent = Arc::new(
            AgentBuilder::from_yaml("name: validation\nsystem_prompt: test\ntools: [writes]\n")
                .unwrap()
                .llm(provider.clone())
                .tool(Arc::new(Writes(calls.clone())))
                .autonomy_extensions(extensions)
                .build()
                .unwrap(),
        );
        let store = support::store(agent.info().id, "write.v1", *sqlite).await;
        let mut config = super::super::super::runner_tests::config(2);
        config.defaults.max_declared_write_paths = Some(1);
        config.defaults.completion = Some(CompletionGate::ValidationPassed(true));
        config.defaults.validation =
            Some(serde_yaml::from_str("checks: [{id: writes, adapter: host.writes}]").unwrap());
        config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            config,
            AutonomyHostCeilings {
                max_declared_write_paths: Some(2),
                ..Default::default()
            },
            store.clone(),
            "write.v1".into(),
        )
        .unwrap();
        let paused = runner.run("report", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert!(matches!(
            paused.run.pending.as_ref().unwrap().kind,
            TaskPendingKind::Limit
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&paused.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        let mut limits = payload.limits;
        limits.max_declared_write_paths = Some(2);
        let revised = runner
            .revise_limits_paused(
                &paused.run.key.run_id,
                paused.run.revision,
                limits,
                "allow declared targets",
            )
            .await
            .unwrap();
        let done = runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}; pending={:?}",
            done.run.stop_reason,
            done.run.pending
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(done.run.counters.turns, 1);
        assert_eq!(done.run.counters.tool_attempts, 1);
    }
}
