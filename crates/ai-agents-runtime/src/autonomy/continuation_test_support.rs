use super::*;
use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(super) struct ReceiptStore {
    pub inner: Arc<ScopedTaskRunStore>,
    pub armed: AtomicBool,
    pub mode: &'static str,
    pub status: TaskRunStatus,
    pub checkpoint: parking_lot::Mutex<Option<serde_json::Value>>,
    pub reached: tokio::sync::Semaphore,
}
impl ReceiptStore {
    // A single selected final write distinguishes proven rejection from an ambiguous receipt.
    pub fn new(
        inner: Arc<ScopedTaskRunStore>,
        mode: &'static str,
        status: TaskRunStatus,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner,
            armed: AtomicBool::new(false),
            mode,
            status,
            checkpoint: parking_lot::Mutex::new(None),
            reached: tokio::sync::Semaphore::new(0),
        })
    }

    // Checkpoint selection is bound to consumed request identity or a new acknowledged provider result.
    pub fn checkpoint(
        inner: Arc<ScopedTaskRunStore>,
        mode: &'static str,
        checkpoint: serde_json::Value,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner,
            armed: AtomicBool::new(false),
            mode,
            status: TaskRunStatus::Running,
            checkpoint: parking_lot::Mutex::new(Some(checkpoint)),
            reached: tokio::sync::Semaphore::new(0),
        })
    }

    pub fn assert_hit(&self) {
        assert!(
            !self.armed.load(Ordering::SeqCst),
            "selected receipt fault did not fire"
        );
    }

    pub async fn assert_final_write(&self, id: &str) {
        self.assert_hit();
        let snapshot = self.inner.load(id).await.unwrap().unwrap();
        if self.mode == "reject" {
            assert_ne!(snapshot.status, self.status);
        } else {
            assert_eq!(snapshot.status, self.status);
        }
    }
}
#[async_trait]
impl TaskRunStore for ReceiptStore {
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
    // Faults target the selected consumption, provider settlement or final write rather than an incidental nearby checkpoint.
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let selected = if let Some(binding) = &*self.checkpoint.lock() {
            if let TaskRunMutation::Checkpoint { payload, .. } = mutation {
                let payload: TaskCheckpointPayload = serde_json::from_value(payload.clone())?;
                if let Some(id) = binding.get("consume").and_then(serde_json::Value::as_str) {
                    payload
                        .consumed_request_ids
                        .iter()
                        .any(|consumed| consumed == id)
                } else {
                    payload
                        .reservations
                        .iter()
                        .filter(|r| {
                            r.state == TaskEffectState::Completed
                                && r.result
                                    .as_ref()
                                    .is_some_and(|r| r.get("response").is_some())
                        })
                        .count()
                        > binding["completed"].as_u64().unwrap() as usize
                }
            } else {
                false
            }
        } else {
            matches!(mutation, TaskRunMutation::FinalCheckpoint { status, release: true, .. } if *status == self.status)
        };
        if selected && self.armed.swap(false, Ordering::SeqCst) {
            if self.mode == "reject" {
                self.reached.add_permits(1);
                return Err(AgentError::Persistence(
                    "rejected continuation receipt".into(),
                ));
            }
            let saved = self.inner.mutate(id, mutation).await?;
            self.reached.add_permits(1);
            if self.mode == "drop" {
                std::future::pending::<()>().await;
            }
            if self.mode == "lost" {
                return Err(AgentError::Persistence("lost continuation receipt".into()));
            }
            return Ok(saved);
        }
        self.inner.mutate(id, mutation).await
    }
}

// Each test runs the same conditional journal against process-local and SQL storage when enabled.
pub(super) async fn store(
    agent_id: String,
    identity: &str,
    sqlite: bool,
) -> Arc<ScopedTaskRunStore> {
    #[cfg(feature = "sqlite")]
    if sqlite {
        return Arc::new(
            ScopedTaskRunStore::new(
                Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
                agent_id,
                None,
                identity.into(),
            )
            .unwrap(),
        );
    }
    assert!(!sqlite);
    Arc::new(ScopedTaskRunStore::in_memory(agent_id, None, identity.into()).unwrap())
}
pub(super) fn backends() -> &'static [bool] {
    if cfg!(feature = "sqlite") {
        &[false, true]
    } else {
        &[false]
    }
}

pub(super) struct FaultProvider {
    replies: Vec<String>,
    pub calls: std::sync::atomic::AtomicUsize,
    pub fault_at: usize,
    pub mode: &'static str,
    pub reached: tokio::sync::Semaphore,
    pub release: tokio::sync::Semaphore,
}
impl FaultProvider {
    pub fn new(replies: &[&str], fault_at: usize, mode: &'static str) -> Arc<Self> {
        Arc::new(Self {
            replies: replies.iter().map(|s| (*s).into()).collect(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            fault_at,
            mode,
            reached: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        })
    }
}
#[async_trait]
impl ai_agents_core::LLMProvider for FaultProvider {
    // The gate targets one semantic dispatch so completed parsing is observable before generation fails or drops.
    async fn complete(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<ai_agents_core::LLMResponse, ai_agents_core::LLMError> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        if index == self.fault_at {
            self.reached.add_permits(1);
            if self.mode == "drop" {
                std::future::pending::<()>().await;
            }
            if self.mode == "gate" {
                self.release.acquire().await.unwrap().forget();
            }
            if self.mode == "error" {
                return Err(ai_agents_core::LLMError::Config(
                    "semantic provider failed".into(),
                ));
            }
        }
        Ok(ai_agents_core::LLMResponse::new(
            self.replies
                .get(index)
                .map(String::as_str)
                .unwrap_or("invalid response"),
            ai_agents_core::FinishReason::Stop,
        ))
    }
    async fn complete_stream(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<
        Box<
            dyn futures::Stream<
                    Item = std::result::Result<ai_agents_core::LLMChunk, ai_agents_core::LLMError>,
                > + Unpin
                + Send,
        >,
        ai_agents_core::LLMError,
    > {
        Err(ai_agents_core::LLMError::Config("unexpected stream".into()))
    }
    fn provider_name(&self) -> &str {
        "continuation-fault"
    }
    fn supports(&self, _: ai_agents_core::LLMFeature) -> bool {
        false
    }
}

pub(super) struct ResponseGate {
    pub calls: std::sync::atomic::AtomicUsize,
    pub reached: tokio::sync::Semaphore,
}
impl ResponseGate {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
            reached: tokio::sync::Semaphore::new(0),
        })
    }
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for ResponseGate {
    // A dropped finalization callback is not acknowledgement that the root completed.
    async fn on_response(&self, _: &ai_agents_core::AgentResponse) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.reached.add_permits(1);
        std::future::pending::<()>().await;
    }
}
