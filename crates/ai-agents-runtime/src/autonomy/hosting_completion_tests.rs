//! Common hosting acceptance covers unserviceable pauses and store-only observation without a production restart host.

use super::continuation_test_support::{self as support, ReceiptStore};
use super::runner_tests::RecordingProvider;
use super::*;
use crate::{Agent, AgentBuilder};
use std::sync::atomic::Ordering;

struct CancellationRaceStore {
    inner: std::sync::Arc<ScopedTaskRunStore>,
    phase: &'static str,
    armed: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl TaskRunStore for CancellationRaceStore {
    // The real scoped backend owns schema and scope validation; race injection adds no alternate journal.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.inner.create(snapshot).await
    }
    // A cancellation-only write occurs at a deterministic read boundary, without using timing to infer the competing operation.
    async fn load(&self, id: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        let snapshot = self.inner.load(id).await?;
        if let Some(paused) = &snapshot
            && paused.status == TaskRunStatus::Paused
            && self.phase != "claim"
            && self.armed.swap(false, Ordering::SeqCst)
        {
            let _receipt =
                CancellationOutcome::request(self.inner.as_ref(), id, paused.revision).await?;
            if self.phase == "before_load" {
                return self.inner.load(id).await;
            }
        }
        Ok(snapshot)
    }
    // Conditional writes retain their production semantics and exact receipts.
    async fn mutate(
        &self,
        id: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        self.inner.mutate(id, mutation).await
    }
    // Only the actual scoped backend may certify that the racing claim made no write.
    async fn claim_paused(
        &self,
        id: &str,
        revision: u64,
        token: String,
    ) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        if self.phase == "claim" && self.armed.swap(false, Ordering::SeqCst) {
            let _receipt = CancellationOutcome::request(self.inner.as_ref(), id, revision).await?;
        }
        self.inner.claim_paused(id, revision, token).await
    }
    // Metadata observation does not execute the competing cancellation or adopt an owner.
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    // Deletion preserves the actual backend's scope and conditional protection.
    async fn delete(&self, id: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(id, revision).await
    }
}

// Cancellation can arrive before a read, after its sample, or at CAS; all three converge only through an exact successor/no-write witness.
#[tokio::test]
async fn host_completion_cancel_races_through_read_and_claim_are_acknowledged() {
    for &sqlite in support::backends() {
        for phase in ["before_load", "after_load", "claim"] {
            let store = std::sync::Arc::new(CancellationRaceStore {
                inner: support::store("HostingExit".into(), "hosting-exit.v1", sqlite).await,
                phase,
                armed: std::sync::atomic::AtomicBool::new(true),
            });
            let provider = RecordingProvider::new(&["done"]);
            let agent = AgentBuilder::from_yaml(STAGE)
                .unwrap()
                .llm(provider.clone())
                .automatic_task_options(AutomaticTaskOptions {
                    store: Some(store.clone()),
                    config_identity: Some("hosting-exit.v1".into()),
                    ..Default::default()
                })
                .task_interaction_delivery(false)
                .build_task_host()
                .await
                .unwrap();
            let result = agent.run_task("Prepare report", None).await.unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Cancelled, "{phase}");
            assert!(!store.armed.load(Ordering::SeqCst));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert!(
                agent
                    .set_context("released", serde_json::json!(true))
                    .is_ok()
            );
        }
    }
}

// An ordinary pipeline parent must receive the actual unavailable-child result, not wait forever on a superseded paused outcome.
#[tokio::test]
async fn host_completion_unavailable_automatic_child_returns_to_ordinary_parent() {
    use crate::orchestration::{PipelineStage, pipeline};
    use crate::spawner::{AgentRegistry, SpawnedAgent};
    let yaml = format!(
        "{STAGE}states:\n  initial: work\n  states:\n    work:\n      autonomy: {{enabled: true, mode: state_until_exit}}\n"
    );
    let provider = RecordingProvider::new(&["done"]);
    let spec = crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
    let runtime = AgentBuilder::from_spec(spec.clone())
        .llm(provider.clone())
        .task_interaction_delivery(false)
        .build_task_host()
        .await
        .unwrap();
    let registry = AgentRegistry::new();
    registry
        .register(SpawnedAgent::from_runtime("child".into(), runtime, spec))
        .await
        .unwrap();
    let stages = vec![PipelineStage::id("child")];
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        pipeline(
            &registry,
            "Prepare report",
            &stages,
            Some(30_000),
            None,
            None,
        ),
    )
    .await
    .unwrap();
    assert!(result.is_err());
    let child = registry.get("child").unwrap();
    assert_eq!(
        child.automatic_task_result().unwrap().run.status,
        TaskRunStatus::Incomplete
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// A lost cleanup claim is not a no-write witness and must not be retried as cancellation or reported as successful retirement.
#[tokio::test]
async fn host_completion_unavailable_claim_ack_faults_keep_protection() {
    for &sqlite in support::backends() {
        for mode in ["reject", "lost", "drop"] {
            let fault = ReceiptStore::checkpoint(
                support::store("HostingExit".into(), "hosting-exit.v1", sqlite).await,
                mode,
                serde_json::json!({"claim":true}),
            );
            fault.armed.store(true, Ordering::SeqCst);
            let provider = RecordingProvider::new(&["done"]);
            let agent = AgentBuilder::from_yaml(STAGE)
                .unwrap()
                .llm(provider.clone())
                .automatic_task_options(AutomaticTaskOptions {
                    store: Some(fault.clone()),
                    config_identity: Some("hosting-exit.v1".into()),
                    ..Default::default()
                })
                .task_interaction_delivery(false)
                .build_task_host()
                .await
                .unwrap();
            let mut operation = Box::pin(agent.run_task("Prepare report", None));
            if mode == "drop" {
                tokio::time::timeout(std::time::Duration::from_secs(10), async {
                    tokio::select! {
                        result = &mut operation => panic!("claim did not hold: {result:?}"),
                        permit = fault.reached.acquire() => permit.unwrap().forget(),
                    }
                })
                .await
                .unwrap();
                drop(operation);
            } else {
                assert!(operation.await.is_err());
            }
            fault.assert_hit();
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert!(agent.chat("must not replay").await.is_err());
        }
    }
}

const STAGE: &str = "name: HostingExit\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  lifecycle: [{id: work, instruction: Prepare the report}]\n  supervision: {stage_changes: require_approval}\n  completion: {response_contains: done}\n";
const QUESTION: &str = "name: HostingExit\nsystem_prompt: Helpful\ntools: [ask_user]\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  hitl: {on_user_question: pause_run}\n  completion: {response_contains: done}\n";

// Store-only inspection has no runtime/restore dependency and preserves the same privacy and schema/scope checks as retained inspection.
#[tokio::test]
async fn host_completion_store_only_display_is_read_only_redacted_and_scope_bound() {
    let storage = std::sync::Arc::new(ai_agents_storage::InMemoryTaskStorage::default());
    let provider = RecordingProvider::new(&["PRIVATE_DRAFT"]);
    let agent = AgentBuilder::from_yaml(STAGE)
        .unwrap()
        .llm(provider.clone())
        .storage(storage.clone())
        .build_task_host()
        .await
        .unwrap();
    agent.set_actor_id("alice").unwrap();
    let paused = agent.run_task("PRIVATE_OBJECTIVE", None).await.unwrap();
    let runner = agent.task_runner().unwrap();
    let local = agent.task_display(&paused.run.key.run_id).await.unwrap();
    assert!(local.locally_retained);
    for _ in 0..2 {
        let stored = TaskRunDisplay::read(runner.task_store().as_ref(), &paused.run.key.run_id)
            .await
            .unwrap();
        assert_eq!(stored.summary.revision, paused.run.revision);
        assert!(!stored.locally_retained);
        assert_eq!(stored.restoration_review_required, None);
        let mut local_json = serde_json::to_value(&local).unwrap();
        local_json["locally_retained"] = serde_json::json!(false);
        local_json["restoration_review_required"] = serde_json::Value::Null;
        assert_eq!(serde_json::to_value(&stored).unwrap(), local_json);
        let encoded = serde_json::to_string(&stored).unwrap();
        for secret in [
            "PRIVATE_OBJECTIVE",
            "PRIVATE_DRAFT",
            "owner_token",
            "reviewed_action",
        ] {
            assert!(!encoded.contains(secret));
        }
        assert!(encoded.len() <= MAX_TASK_DISPLAY_BYTES);
    }
    let snapshot = runner
        .task_store()
        .load(&paused.run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    let identity = serde_json::from_value::<TaskCheckpointPayload>(snapshot.payload)
        .unwrap()
        .config_identity;
    for (actor, identity) in [
        (Some("bob".into()), identity),
        (Some("alice".into()), "wrong-config".into()),
    ] {
        let wrong = ScopedTaskRunStore::new(storage.clone(), "HostingExit".into(), actor, identity)
            .unwrap();
        assert!(
            TaskRunDisplay::read(&wrong, &paused.run.key.run_id)
                .await
                .is_err()
        );
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let cancelled = agent
        .cancel_task(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    let terminal = TaskRunDisplay::read(runner.task_store().as_ref(), &paused.run.key.run_id)
        .await
        .unwrap();
    assert_eq!(terminal.summary.revision, cancelled.run.revision);
    assert_eq!(terminal.summary.status, TaskRunStatus::Cancelled);
}

// Automatic dispatch cannot publish an obsolete usable pause after the host's common return path acknowledges unavailable cleanup.
#[tokio::test]
async fn host_completion_automatic_unavailable_pause_publishes_the_actual_terminal_result() {
    use futures::StreamExt;
    for (yaml, reply) in [
        (STAGE, "done"),
        (
            QUESTION,
            r#"{"tool":"ask_user","arguments":{"question":"Choose next work"}}"#,
        ),
    ] {
        let yaml = format!(
            "{yaml}states:\n  initial: work\n  states:\n    work:\n      autonomy: {{enabled: true, mode: state_until_exit}}\n"
        );
        let provider = RecordingProvider::new(&[reply]);
        let agent = AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .auto_configure_features()
            .unwrap()
            .task_interaction_delivery(false)
            .build_task_host()
            .await
            .unwrap();
        let mut stream = agent.chat_stream_events("Prepare a report").await.unwrap();
        let mut final_status = None;
        while let Some(event) = stream.next().await {
            if let crate::AgentStreamEvent::Final(response) = event {
                final_status = response
                    .metadata
                    .and_then(|m| m.get("task_run").cloned())
                    .map(|m| m["status"].clone());
            }
        }
        drop(stream);
        assert_eq!(final_status, Some(serde_json::json!("incomplete")));
        let outcome = agent.automatic_task_result().unwrap();
        assert_eq!(outcome.run.status, TaskRunStatus::Incomplete);
        assert_eq!(
            outcome.run.stop_reason.as_deref(),
            Some("capability_unavailable")
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(
            agent
                .set_context("released", serde_json::json!(true))
                .is_ok()
        );
    }
}

// A cancel request racing the acknowledged pause wins only when its receipt is the exact cancellation-only successor.
#[tokio::test]
async fn host_completion_cancel_after_pause_ack_is_not_relabelled_as_unavailable() {
    for &sqlite in support::backends() {
        let inner = support::store("HostingExit".into(), "hosting-exit.v1", sqlite).await;
        let held = ReceiptStore::new(inner.clone(), "hold", TaskRunStatus::Paused);
        held.armed.store(true, Ordering::SeqCst);
        let provider = RecordingProvider::new(&["done"]);
        let agent = AgentBuilder::from_yaml(STAGE)
            .unwrap()
            .llm(provider.clone())
            .automatic_task_options(AutomaticTaskOptions {
                store: Some(held.clone()),
                config_identity: Some("hosting-exit.v1".into()),
                ..Default::default()
            })
            .task_interaction_delivery(false)
            .build_task_host()
            .await
            .unwrap();
        let mut operation = Box::pin(agent.run_task("Prepare a report", None));
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::select! {
                result = &mut operation => panic!("pause did not hold: {result:?}"),
                permit = held.reached.acquire() => permit.unwrap().forget(),
            }
        })
        .await
        .unwrap();
        let paused = inner.list().await.unwrap().pop().unwrap();
        let receipt =
            CancellationOutcome::request(inner.as_ref(), &paused.key.run_id, paused.revision)
                .await
                .unwrap();
        assert_eq!(receipt.disposition, CancellationDisposition::Requested);
        held.release.add_permits(1);
        let done = tokio::time::timeout(std::time::Duration::from_secs(10), operation)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Cancelled);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(
            agent
                .set_context("released", serde_json::json!(true))
                .is_ok()
        );
    }
}

// A volatile stage review or tool question cannot be returned to a host which explicitly has no response endpoint.
#[tokio::test]
async fn host_completion_unserviceable_pauses_retire_without_approving_pending_work() {
    for (yaml, reply) in [
        (STAGE, "done"),
        (
            QUESTION,
            r#"{"tool":"ask_user","arguments":{"question":"Choose next work"}}"#,
        ),
    ] {
        for &sqlite in support::backends() {
            let provider = RecordingProvider::new(&[reply]);
            let store = support::store("HostingExit".into(), "hosting-exit.v1", sqlite).await;
            let agent = AgentBuilder::from_yaml(yaml)
                .unwrap()
                .llm(provider.clone())
                .auto_configure_features()
                .unwrap()
                .automatic_task_options(AutomaticTaskOptions {
                    store: Some(store.clone()),
                    config_identity: Some("hosting-exit.v1".into()),
                    ..Default::default()
                })
                .task_interaction_delivery(false)
                .build_task_host()
                .await
                .unwrap();
            let result = agent.run_task("Prepare a report", None).await.unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Incomplete);
            assert_eq!(
                result.run.stop_reason.as_deref(),
                Some("capability_unavailable")
            );
            assert!(result.run.pending.is_none());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                store
                    .load(&result.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                TaskRunStatus::Incomplete
            );
            assert!(
                !agent
                    .task_display(&result.run.key.run_id)
                    .await
                    .unwrap()
                    .locally_retained
            );
            assert!(
                agent
                    .set_context("released", serde_json::json!(true))
                    .is_ok()
            );
        }
    }
}

// A real retained endpoint is the positive control; it keeps the exact prompt and single-use response instead of silently stopping.
#[tokio::test]
async fn host_completion_retained_stage_review_preserves_typed_response_and_exact_revision() {
    let provider = RecordingProvider::new(&["done"]);
    let agent = AgentBuilder::from_yaml(STAGE)
        .unwrap()
        .llm(provider.clone())
        .build_task_host()
        .await
        .unwrap();
    let paused = agent.run_task("Prepare a report", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let prompt = agent
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(prompt.prompt, TaskHostPrompt::StageReview { .. }));
    assert!(
        agent
            .resume_task(
                &paused.run.key.run_id,
                paused.run.revision + 1,
                TaskResumeInput::Approval {
                    request_id: prompt.request_id.clone(),
                    result: ai_agents_hitl::ApprovalResult::approved()
                }
            )
            .await
            .is_err()
    );
    let done = agent
        .resume_task(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: prompt.request_id,
                result: ai_agents_hitl::ApprovalResult::approved(),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Failed, lost and dropped cleanup acknowledgements retain actual runtime custody and never certify that a pending operation ran.
#[tokio::test]
async fn host_completion_unavailable_cleanup_ack_faults_keep_protection() {
    for &sqlite in support::backends() {
        for mode in ["reject", "lost", "drop"] {
            let inner = support::store("HostingExit".into(), "hosting-exit.v1", sqlite).await;
            let fault = ReceiptStore::new(inner, mode, TaskRunStatus::Incomplete);
            fault.armed.store(true, Ordering::SeqCst);
            let provider = RecordingProvider::new(&["done"]);
            let agent = AgentBuilder::from_yaml(STAGE)
                .unwrap()
                .llm(provider.clone())
                .automatic_task_options(AutomaticTaskOptions {
                    store: Some(fault.clone()),
                    config_identity: Some("hosting-exit.v1".into()),
                    ..Default::default()
                })
                .task_interaction_delivery(false)
                .build_task_host()
                .await
                .unwrap();
            let operation = agent.run_task("Prepare a report", None);
            let mut operation = Box::pin(operation);
            if mode == "drop" {
                tokio::time::timeout(std::time::Duration::from_secs(10), async {
                    tokio::select! {
                        result = &mut operation => panic!("cleanup did not reach the selected receipt: {result:?}"),
                        permit = fault.reached.acquire() => permit.unwrap().forget(),
                    }
                }).await.unwrap();
                drop(operation);
            } else {
                assert!(operation.await.is_err());
            }
            fault.assert_hit();
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert!(agent.chat("must not replay").await.is_err());
            assert!(
                agent
                    .set_context("not_released", serde_json::json!(true))
                    .is_err()
            );
        }
    }
}
