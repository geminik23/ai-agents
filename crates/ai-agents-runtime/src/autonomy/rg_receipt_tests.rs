//! Batch and composed safe pauses require exact intermediate and final ACKs before cursor or owner transfer.
use super::*;
use ai_agents_core::AgentError;

struct CorruptPauseStore {
    inner: Arc<dyn TaskRunStore>,
    anchor: bool,
    armed: std::sync::atomic::AtomicBool,
}
#[async_trait]
impl TaskRunStore for CorruptPauseStore {
    // The fixture delegates every real transaction, then corrupts only the selected receipt.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    // Reads always reveal the authoritative backend, not the altered receipt.
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    // No listing can claim execution.
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    // Actual deletion keeps the backend's conditional semantics.
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.inner.delete(id, revision).await
    }
    // A correct revision/status cannot hide an incomplete payload ACK, including numeric evidence omitted from the receipt.
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let selected = if self.anchor {
            matches!(mutation,TaskRunMutation::Checkpoint {payload,..} if !payload["pending"].is_null() && !payload["pause_reason"].is_null())
        } else {
            matches!(
                mutation,
                TaskRunMutation::FinalCheckpoint {
                    status: TaskRunStatus::Paused,
                    ..
                }
            )
        };
        let mut saved = self.inner.mutate(id, mutation).await?;
        if selected && self.armed.swap(false, Ordering::SeqCst) {
            let mut found = false;
            for adapter in saved.payload["adapters"].as_array_mut().unwrap() {
                if adapter["id"] == "runtime.telemetry" {
                    adapter["state"]["complete"] = json!(false);
                    adapter["state"]["events"] = json!({});
                    found = true;
                }
            }
            if !found {
                return Err(AgentError::Other(
                    "selected pause lacks telemetry fixture".into(),
                ));
            }
        }
        Ok(saved)
    }
}

// A malformed ACK leaves actual custody held even when the backend committed a valid pause; it cannot retire collector cuts or admit ordinary work.
#[tokio::test]
async fn rg_batch_and_group_pause_reject_inexact_telemetry_receipts() {
    for sqlite in super::super::continuation_test_support::backends() {
        for group in [false, true] {
            for anchor in [false, true] {
                let state = DomainState::new();
                let provider = RecordingProvider::new(&[
                    r#"{"tool":"ask_user","arguments":{"question":"Confirm","options":["yes"]}}"#,
                    "done",
                ]);
                let agent = if group {
                    use crate::spawner::{AgentRegistry, AgentSpawner};
                    let registry = Arc::new(AgentRegistry::new());
                    let child =
                        Arc::try_unwrap(domain_agent_observed(state.clone(), provider, true, true))
                            .unwrap_or_else(|_| panic!("child already shared"));
                    super::super::co_completion_tests::register(&registry,"worker",child,"name: RGDomain\nsystem_prompt: Pursue the configured objective; only host facts prove completion.\ntools: [domain, ask_user]\n").await;
                    let mut observation = ai_agents_observability::ObservabilityConfig {
                        enabled: true,
                        ..Default::default()
                    };
                    observation.export.write_report = false;
                    Arc::new(crate::AgentBuilder::from_yaml("name: ReceiptRoot\nsystem_prompt: parent\nstates:\n  initial: work\n  states:\n    work:\n      delegate: worker\n").unwrap().llm(RecordingProvider::new(&[])).observability(ai_agents_observability::ObservabilityManager::new(observation)).auto_configure_features().unwrap().build().unwrap().with_spawner_handles(Arc::new(AgentSpawner::new()),registry))
                } else {
                    domain_agent_observed(state, provider, true, true)
                };
                let inner = super::super::continuation_test_support::store(
                    agent.info().id,
                    "rg.v1",
                    *sqlite,
                )
                .await;
                let store = Arc::new(CorruptPauseStore {
                    inner: inner.clone(),
                    anchor,
                    armed: std::sync::atomic::AtomicBool::new(true),
                });
                let runner = AutonomyRunner::try_new(agent.clone(),config("completion: {response_contains: done}\nhitl: {on_user_question: pause_run}\n"),Default::default(),store.clone(),"rg.v1".into()).unwrap();
                assert!(
                    runner
                        .run("verify exact pause acknowledgement", None)
                        .await
                        .is_err()
                );
                assert!(!store.armed.load(Ordering::SeqCst));
                let run = inner.list().await.unwrap().remove(0);
                let saved = inner.load(&run.key.run_id).await.unwrap().unwrap();
                assert_eq!(
                    saved.status,
                    if anchor {
                        TaskRunStatus::Running
                    } else {
                        TaskRunStatus::Paused
                    }
                );
                assert!(
                    agent
                        .set_context("must remain protected", json!(true))
                        .is_err()
                );
            }
        }
    }
}
