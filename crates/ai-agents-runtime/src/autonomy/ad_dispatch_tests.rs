use super::*;
use crate::autonomy::continuation_tests::shared_executor_fixture;
use crate::autonomy::*;
use ai_agents_core::autonomy::TaskFinalAdmission;
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize};

struct FinalProof {
    calls: AtomicUsize,
    cancel: crate::RuntimeControlHandle,
}
impl std::fmt::Debug for FinalProof {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("FinalProof")
    }
}
impl TaskFinalAdmission for FinalProof {
    // The second sample is inside the actual first-poll closure, after all earlier admission checks.
    fn validate(&self) -> ai_agents_core::Result<()> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
            self.cancel.cancel_all();
        }
        Ok(())
    }
}

// Cancellation completed by the last proof must prevent the first tool poll and refund only this acknowledged non-invocation.
#[tokio::test]
async fn ad_final_tool_last_proof_rechecks_cancellation() {
    let (agent, execution, calls) = shared_executor_fixture(Default::default(), false).await;
    let proof = Arc::new(FinalProof {
        calls: AtomicUsize::new(0),
        cancel: agent.runtime_control(),
    });
    let tool = agent.tools.get("work_probe").unwrap();
    let mut context = ToolExecutionContext::test("work_probe");
    context.registry_version = agent.tools.version();
    let safety = agent.runtime_safety_snapshot();
    context.runtime_control_version = safety.version;
    context.policy_version = safety.tool_security.policy_version();
    let result = scope_execution(
        execution.clone(),
        ai_agents_llm::scope_invocation_evidence(
            vec![proof.clone()],
            agent.execute_resolved_tool_once(
                tool,
                json!({}),
                context,
                RuntimeAgent::validated_tool_timeout(10_000).unwrap(),
            ),
        ),
    )
    .await
    .unwrap();
    assert!(proof.calls.load(Ordering::SeqCst) >= 2);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(!result.3);
    let payload: TaskCheckpointPayload =
        serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
    assert_eq!(payload.counters.tool_attempts, 0);
    assert!(payload.reservations.iter().all(|entry| {
        entry
            .result
            .as_ref()
            .is_some_and(|value| value["not_invoked"] == true)
    }));
}

// A non-preemptible body may finish late; factual success remains stored, but forged metadata cannot grant task credit or retry authority.
#[tokio::test]
async fn ad_final_synchronous_tool_overrun_retains_effect_without_completion_credit() {
    let (agent, execution, calls) = shared_executor_fixture(Default::default(), false).await;
    let bounds = ai_agents_llm::InvocationBounds::new(
        std::time::SystemTime::now() + std::time::Duration::from_secs(20),
        std::time::Instant::now() + std::time::Duration::from_millis(100),
    );
    let result = ai_agents_llm::scope_invocation_bounds(
        Some(bounds),
        scope_execution(
            execution.clone(),
            agent.invoke_tool(ToolExecutionRequest::new(
                "overrun",
                "work_probe",
                json!({"synchronous_overrun":true}),
                ToolCallSource::Task,
            )),
        ),
    )
    .await;
    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload =
        serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
    assert_eq!(payload.counters.tool_attempts, 1);
    assert!(payload.reservations.iter().all(|entry| {
        entry.state == TaskEffectState::Completed
            && entry
                .result
                .as_ref()
                .is_some_and(|value| value["success"] == true)
    }));
    let record = &payload.evidence.tool_calls[0].record;
    assert!(
        record.executed && record.success,
        "actual effect facts must survive"
    );
    assert_eq!(record.metadata["_task_result_eligible"], false);
    assert!(!crate::autonomy::executed(record));
    assert!(!execution.owner.work.unsettled());
}

struct OrdinaryTimeoutTool {
    id: &'static str,
    pending: bool,
}
#[async_trait]
impl ai_agents_core::Tool for OrdinaryTimeoutTool {
    fn id(&self) -> &str {
        self.id
    }
    fn name(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        "isolated timeout recovery"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object"})
    }
    fn safety_metadata(&self) -> ai_agents_core::ToolSafetyMetadata {
        ai_agents_core::ToolSafetyMetadata {
            read_only: true,
            ..Default::default()
        }
    }
    async fn execute(&self, _: Value, _: ToolExecutionContext) -> ToolResult {
        if self.pending {
            std::future::pending::<()>().await;
        }
        ToolResult::ok("recovered")
    }
}

// Task result adoption must not change ordinary agents' existing timeout skip/fallback policy.
#[tokio::test]
async fn ad_final_ordinary_timeout_preserves_skip_and_fallback() {
    use ai_agents_recovery::{ErrorRecoveryConfig, ToolRecoveryConfig, ToolRetryConfig};
    for skip in [false, true] {
        let agent = crate::AgentBuilder::new()
            .system_prompt("ordinary")
            .llm(Arc::new(ai_agents_llm::mock::MockLLMProvider::new(
                "unused",
            )))
            .tool(Arc::new(OrdinaryTimeoutTool {
                id: "pending",
                pending: true,
            }))
            .tool(Arc::new(OrdinaryTimeoutTool {
                id: "rescue",
                pending: false,
            }))
            .recovery_manager(RecoveryManager::new(ErrorRecoveryConfig {
                tools: ToolRecoveryConfig {
                    default: ToolRetryConfig {
                        max_retries: 0,
                        timeout_ms: Some(10),
                        on_failure: if skip {
                            ToolFailureAction::Skip
                        } else {
                            ToolFailureAction::Fallback {
                                fallback_tool: "rescue".into(),
                            }
                        },
                    },
                    ..Default::default()
                },
                ..Default::default()
            }))
            .build()
            .unwrap();
        let record = agent
            .invoke_tool(ToolExecutionRequest::new(
                "ordinary",
                "pending",
                json!({}),
                ToolCallSource::Model,
            ))
            .await
            .unwrap();
        assert!(record.success);
        if skip {
            assert!(record.output.contains("skipped"));
        } else {
            assert_eq!(record.canonical_id, "rescue");
        }
    }
}

struct AckStore {
    inner: Arc<dyn TaskRunStore>,
    armed: AtomicBool,
    reached: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
    maintenance: bool,
}
#[async_trait]
impl TaskRunStore for AckStore {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(id, revision).await
    }
    // Delay only a successful actual settlement receipt; the backend has already committed its exact successor.
    async fn mutate(
        &self,
        id: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        let selected = if let TaskRunMutation::Checkpoint { payload, .. } = mutation {
            let payload: TaskCheckpointPayload = serde_json::from_value(payload.clone())?;
            payload.reservations.iter().any(|entry| {
                entry.state == TaskEffectState::Completed
                    && entry.result.as_ref().is_some_and(|result| {
                        if self.maintenance {
                            result["maintenance"] == "ack-tail"
                        } else {
                            result["success"] == true
                        }
                    })
            })
        } else {
            false
        };
        let saved = self.inner.mutate(id, mutation).await?;
        if selected && self.armed.swap(false, Ordering::SeqCst) {
            self.reached.add_permits(1);
            self.release.acquire().await.unwrap().forget();
        }
        Ok(saved)
    }
}

#[derive(Debug)]
struct LiveProof(Arc<AtomicBool>);
impl TaskFinalAdmission for LiveProof {
    fn validate(&self) -> ai_agents_core::Result<()> {
        if self.0.load(Ordering::SeqCst) {
            Err(ai_agents_core::AgentError::Other(
                "current host proof changed".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
struct AdmissionOnlyProof(Arc<AtomicBool>);
impl TaskFinalAdmission for AdmissionOnlyProof {
    fn validate(&self) -> ai_agents_core::Result<()> {
        LiveProof(self.0.clone()).validate()
    }
    // Closing new work is deliberately not revocation of an already dispatched result or its settlement.
    fn validate_dispatched(&self) -> ai_agents_core::Result<()> {
        Ok(())
    }
}

// Known effect settlement stays completed even when its ACK outwaits proof, cancellation or the original invocation clock.
#[tokio::test]
async fn ad_final_tool_and_maintenance_success_ack_rechecks_eligibility() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for maintenance in [false, true] {
            for mode in ["normal", "admission_closed", "proof", "cancel", "expiry"] {
                let (agent, original, calls) =
                    shared_executor_fixture(Default::default(), false).await;
                let snapshot = original.load_owned().await.unwrap();
                let inner = crate::autonomy::continuation_test_support::store(
                    agent.info().id,
                    "shared-v1",
                    sqlite,
                )
                .await;
                inner.create(&snapshot).await.unwrap();
                let store = Arc::new(AckStore {
                    inner,
                    armed: AtomicBool::new(true),
                    reached: tokio::sync::Semaphore::new(0),
                    release: tokio::sync::Semaphore::new(0),
                    maintenance,
                });
                let execution = RunExecution::new(
                    store.clone(),
                    &snapshot,
                    LifecycleState::new(&Default::default(), false).unwrap(),
                    original.owner.clone(),
                    std::time::Instant::now(),
                )
                .unwrap();
                execution.set_scope(&EvaluationScope {
                    key: snapshot.key.clone(),
                    objective_revision: 0,
                    cycle: 1,
                    stage: None,
                    mutation_generation: 0,
                    target_revisions: Default::default(),
                    target_generations: Default::default(),
                    validation_bindings: Default::default(),
                    validation_attempts: Default::default(),
                    validation_dependencies: Default::default(),
                });
                let changed = Arc::new(AtomicBool::new(false));
                let bounds = ai_agents_llm::InvocationBounds::new(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(30),
                    std::time::Instant::now()
                        + std::time::Duration::from_secs(if mode == "expiry" { 1 } else { 20 }),
                );
                let mut work = Box::pin(ai_agents_llm::scope_invocation_evidence(
                    vec![if mode == "admission_closed" {
                        Arc::new(AdmissionOnlyProof(changed.clone())) as Arc<dyn TaskFinalAdmission>
                    } else {
                        Arc::new(LiveProof(changed.clone()))
                    }],
                    ai_agents_llm::scope_invocation_bounds(
                        Some(bounds),
                        scope_execution(execution.clone(), async {
                            if maintenance {
                                crate::autonomy::owned_maintenance("ack-tail", async {
                                    crate::autonomy::work::maintenance_effect();
                                    calls.fetch_add(1, Ordering::SeqCst);
                                    Ok(())
                                })
                                .await
                            } else {
                                let record = agent
                                    .invoke_tool(ToolExecutionRequest::new(
                                        "ack",
                                        "work_probe",
                                        json!({}),
                                        ToolCallSource::Task,
                                    ))
                                    .await?;
                                if record.success {
                                    Ok(())
                                } else {
                                    Err(ai_agents_core::AgentError::Other(
                                        "ineligible tool result".into(),
                                    ))
                                }
                            }
                        }),
                    ),
                ));
                tokio::select! { result = &mut work => panic!("settlement did not wait: {result:?}"), permit = store.reached.acquire() => permit.unwrap().forget() }
                assert_eq!(calls.load(Ordering::SeqCst), 1);
                match mode {
                    "proof" | "admission_closed" => changed.store(true, Ordering::SeqCst),
                    "cancel" => {
                        execution.cancellation.store(true, Ordering::SeqCst);
                    }
                    "expiry" => {
                        std::thread::sleep(bounds.remaining() + std::time::Duration::from_millis(2))
                    }
                    _ => {}
                }
                store.release.add_permits(1);
                assert_eq!(
                    work.await.is_ok(),
                    matches!(mode, "normal" | "admission_closed"),
                    "{sqlite}/{maintenance}/{mode}"
                );
                let saved = store.load(&snapshot.key.run_id).await.unwrap().unwrap();
                let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
                assert_eq!(payload.counters.tool_attempts, u64::from(!maintenance));
                assert!(
                    payload
                        .reservations
                        .iter()
                        .all(|entry| entry.state == TaskEffectState::Completed)
                );
                assert!(!execution.owner.work.unsettled());
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    1,
                    "settled effects cannot replay"
                );
            }
        }
    }
}
