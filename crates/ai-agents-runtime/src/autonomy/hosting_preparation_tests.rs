//! Hosting preparation uses the actual builder, storage selection and controller without provider credentials or external services.

use super::runner_tests::RecordingProvider;
use super::*;
use crate::{Agent, AgentBuilder};
use ai_agents_core::{AgentStorage, StorageCapability};
use std::sync::{Arc, atomic::Ordering};

const TASK: &str = "name: PreparedHost\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  completion: {response_contains: done}\n  premature_finish: {action: ask_user, on_unavailable: stop}\n";
const STATE: &str = "name: PreparedState\nsystem_prompt: Helpful\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        max_turns: 4\n        max_llm_calls: 4\n        completion: {response_contains: done}\n        premature_finish: {action: ask_user, on_unavailable: stop}\n";
const SKILL: &str = "name: PreparedSkill\nsystem_prompt: Helpful\nllm: {default: default, router: router}\nskills:\n  - id: research\n    description: Research a report\n    trigger: Research requested\n    autonomy:\n      enabled: true\n      mode: skill_until_complete\n      max_turns: 4\n      max_llm_calls: 4\n      completion: {response_contains: done}\n      premature_finish: {action: ask_user, on_unavailable: stop}\n    steps:\n      - prompt: Prepare report\n";

// Preparation is not task startup, and explicit/state/skill entry shares the selected backend and exact retained controls.
#[tokio::test]
async fn host_preparation_explicit_state_and_skill_retain_one_scoped_store() {
    for (yaml, automatic) in [(TASK, false), (STATE, true), (SKILL, true)] {
        let provider = RecordingProvider::new(&["draft", "done"]);
        let storage = Arc::new(ai_agents_storage::InMemoryTaskStorage::default());
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .llm_alias("router", RecordingProvider::new(&["research"]))
            .storage(storage.clone())
            .build_task_host()
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert!(agent.automatic_task_runner().is_none());
        agent.set_actor_id("alice").unwrap();
        let runner = agent.task_runner().unwrap();
        assert!(runner.task_store().list().await.unwrap().is_empty());
        let paused = if automatic {
            agent.chat("Prepare research report").await.unwrap();
            agent.automatic_task_result().unwrap()
        } else {
            agent
                .run_task("Research without files", None)
                .await
                .unwrap()
        };
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        let saved = storage
            .load_task_run(&paused.run.key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(saved.revision, paused.run.revision);
        assert_eq!(saved.actor_id.as_deref(), Some("alice"));
        let prompt = agent
            .task_interaction(&paused.run.key.run_id, paused.run.revision)
            .await
            .unwrap()
            .unwrap();
        let done = agent
            .resume_task(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: prompt.request_id,
                    answer: serde_json::json!("continue"),
                },
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(done.run.key, paused.run.key);
        assert_eq!(done.run.counters.llm_attempts, 2);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert!(Arc::ptr_eq(&runner, &agent.task_runner().unwrap()));
        agent.set_actor_id("bob").unwrap();
        let next = agent.task_runner().unwrap();
        assert!(!Arc::ptr_eq(&runner, &next));
        assert!(next.task_store().list().await.unwrap().is_empty());
        assert!(next.task_store().load(&done.run.key.run_id).await.is_err());
    }
}

// Both construction paths pass actual delivery availability to the controller, not just to a frontend widget.
#[tokio::test]
async fn host_preparation_unavailable_endpoint_follows_stop_and_fail_without_extra_work() {
    for asynchronous in [false, true] {
        for fail in [false, true] {
            let yaml = if fail {
                TASK.replace("on_unavailable: stop", "on_unavailable: fail")
            } else {
                TASK.into()
            };
            let provider = RecordingProvider::new(&["draft"]);
            let builder = AgentBuilder::from_yaml(&yaml)
                .unwrap()
                .llm(provider.clone())
                .task_interaction_delivery(false);
            let agent = if asynchronous {
                builder.build_task_host().await
            } else {
                builder.build()
            }
            .unwrap();
            let result = agent.run_task("Analyze without files", None).await.unwrap();
            assert_eq!(
                result.run.status,
                if fail {
                    TaskRunStatus::Failed
                } else {
                    TaskRunStatus::Incomplete
                }
            );
            assert!(result.run.pending.is_none());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert!(
                agent
                    .set_context("released", serde_json::json!(true))
                    .is_ok()
            );
        }
    }
}

// The host's endpoint declaration applies equally to automatic state/skill promotion; ordinary dispatch is not completion authority.
#[tokio::test]
async fn host_preparation_automatic_unavailable_endpoint_does_not_return_a_volatile_pause() {
    for yaml in [STATE, SKILL] {
        let provider = RecordingProvider::new(&["draft"]);
        let agent = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(provider.clone())
            .llm_alias("router", RecordingProvider::new(&["research"]))
            .task_interaction_delivery(false)
            .build_task_host()
            .await
            .unwrap();
        agent.chat("Research a report").await.unwrap();
        let result = agent.automatic_task_result().unwrap();
        assert_eq!(result.run.status, TaskRunStatus::Incomplete);
        assert!(result.run.pending.is_none());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }
}

// No-storage and snapshot-only backends preserve usable volatile hosting without opening files or connecting to Redis.
#[tokio::test]
async fn host_preparation_snapshot_only_storage_does_not_construct_task_capability() {
    for storage in [
        crate::spec::StorageConfig::none(),
        crate::spec::StorageConfig::file("must-not-open/hosting-preparation"),
        crate::spec::StorageConfig::redis("redis://127.0.0.1:1/"),
    ] {
        let provider = RecordingProvider::new(&["done"]);
        let agent = AgentBuilder::from_yaml(TASK)
            .unwrap()
            .llm(provider.clone())
            .storage_config(storage)
            .build_task_host()
            .await
            .unwrap();
        assert!(agent.storage().is_none());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        // Task lookup is also side-effect free; actual chat would initialize its independently configured session storage.
        assert!(
            agent
                .task_runner()
                .unwrap()
                .task_store()
                .list()
                .await
                .unwrap()
                .is_empty()
        );
    }
    let provider = RecordingProvider::new(&["done"]);
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider.clone())
        .build_task_host()
        .await
        .unwrap();
    assert_eq!(
        agent
            .run_task("No-storage analysis", None)
            .await
            .unwrap()
            .run
            .status,
        TaskRunStatus::Completed
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Injected session handles take precedence over configured paths, including a deliberate snapshot-only Noop adapter.
#[tokio::test]
async fn host_preparation_injected_session_backend_preserves_builder_precedence() {
    let injected: Arc<dyn AgentStorage> = Arc::new(ai_agents_core::NoopStorage);
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(RecordingProvider::new(&["done"]))
        .storage_config(crate::spec::StorageConfig::sqlite(
            "must-not-open/unused.sqlite",
        ))
        .storage(injected.clone())
        .build_task_host()
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&agent.storage().unwrap(), &injected));
    assert!(!injected.supports(StorageCapability::TaskRuns));
    let result = agent
        .run_task("No task backend supplied", None)
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
}

// An explicitly scoped task store remains authoritative and preparation does not silently replace it or open session storage.
#[tokio::test]
async fn host_preparation_explicit_task_store_precedes_configured_session_storage() {
    let store: Arc<dyn TaskRunStore> = Arc::new(
        ScopedTaskRunStore::in_memory("PreparedHost".into(), None, "explicit.v1".into()).unwrap(),
    );
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(RecordingProvider::new(&["done"]))
        .storage_config(crate::spec::StorageConfig::sqlite(
            "must-not-open/unused.sqlite",
        ))
        .automatic_task_options(AutomaticTaskOptions {
            store: Some(store.clone()),
            config_identity: Some("explicit.v1".into()),
            ..Default::default()
        })
        .build_task_host()
        .await
        .unwrap();
    assert!(agent.storage().is_none());
    assert!(Arc::ptr_eq(
        agent.task_runner().unwrap().task_store(),
        &store
    ));
}

// Missing automatic reconstruction infrastructure rejects before opening configured resources or invoking any provider.
#[tokio::test]
async fn host_preparation_missing_restore_host_is_not_replaced_by_sqlite_configuration() {
    let yaml = STATE.replace(
        "        mode: state_until_exit",
        "        persistence: {enabled: true}\n        mode: state_until_exit",
    );
    let provider = RecordingProvider::new(&["must not execute"]);
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/hosting-preparation-rejected")
        .join(uuid::Uuid::new_v4().to_string());
    let path = directory.join("tasks.sqlite");
    assert!(!directory.exists());
    let result = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider.clone())
        .storage_config(crate::spec::StorageConfig::sqlite(path.to_str().unwrap()))
        .build_task_host()
        .await;
    assert!(
        !directory.exists(),
        "preflight must precede backend construction"
    );
    assert!(
        matches!(result, Err(ai_agents_core::AgentError::Config(ref message)) if message.contains("restore host"))
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

// Actual selected judge bindings fail before resource preparation, rather than being accepted by alias spelling alone.
#[tokio::test]
async fn host_preparation_invalid_alias_precedes_storage_io() {
    let yaml = STATE.replace(
        "completion: {response_contains: done}",
        "completion: {judge: {llm: missing, criteria: [Correctness]}}",
    );
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/hosting-preparation-rejected")
        .join(uuid::Uuid::new_v4().to_string());
    let provider = RecordingProvider::new(&["must not execute"]);
    let result = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider.clone())
        .storage_config(crate::spec::StorageConfig::sqlite(
            directory.join("tasks.sqlite").to_str().unwrap(),
        ))
        .build_task_host()
        .await;
    assert!(result.is_err());
    assert!(!directory.exists());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

// Explicit persistence is checked at invocation even when its profile is not an automatic declaration.
#[tokio::test]
async fn host_preparation_explicit_persistence_refuses_before_task_creation() {
    let yaml = TASK.replace(
        "  enabled: true",
        "  enabled: true\n  persistence: {enabled: true}",
    );
    let provider = RecordingProvider::new(&["must not execute"]);
    let agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider.clone())
        .build_task_host()
        .await
        .unwrap();
    assert!(agent.run_task("Requires restoration", None).await.is_err());
    assert!(
        agent
            .task_runner()
            .unwrap()
            .task_store()
            .list()
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

// Preparation does not opt a disabled profile into execution or preselect any stored startup candidate.
#[tokio::test]
async fn host_preparation_disabled_profile_remains_ordinary_and_nonexecuting() {
    let provider = RecordingProvider::new(&["ordinary"]);
    let agent = AgentBuilder::new()
        .system_prompt("Helpful")
        .llm(provider.clone())
        .build_task_host()
        .await
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(agent.run_task("Not enabled", None).await.is_err());
    assert!(
        agent
            .task_runner()
            .unwrap()
            .task_store()
            .list()
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        agent.chat("ordinary input").await.unwrap().content,
        "ordinary"
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// A builder override selects the actual backend once, and supplied-runner streaming consumes that same prepared service.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn host_preparation_sqlite_override_and_stream_share_actual_backend() {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/hosting-preparation")
        .join(uuid::Uuid::new_v4().to_string());
    let yaml = format!(
        "{TASK}storage:\n  type: sqlite\n  path: {}\n",
        directory.join("unselected/tasks.sqlite").to_str().unwrap()
    );
    let provider = RecordingProvider::new(&["done"]);
    let agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider.clone())
        .storage_config(crate::spec::StorageConfig::sqlite(
            directory.join("selected.sqlite").to_str().unwrap(),
        ))
        .build_task_host()
        .await
        .unwrap();
    assert!(!directory.join("unselected").exists());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let runner = agent.task_runner().unwrap();
    let done = {
        let mut stream = agent
            .run_task_stream(&runner, "Analyze without files", None)
            .unwrap();
        while stream.next_event().await.is_some() {}
        stream.take_result().unwrap().unwrap()
    };
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    let stored = agent
        .storage()
        .unwrap()
        .load_task_run(&done.run.key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.status, TaskRunStatus::Completed);
    assert_eq!(stored.revision, done.run.revision);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    drop(runner);
    drop(agent);
    crate::remove_sqlite_test_directory(&directory)
        .await
        .unwrap();
}

// Opening the real configured SQLite backend persists journals, not an executable restore host or a prepared restart acknowledgement.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn host_preparation_sqlite_journal_uses_effective_configuration_and_survives_reopen() {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/hosting-preparation")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("tasks.sqlite");
    let provider = RecordingProvider::new(&["draft"]);
    let yaml = format!(
        "{TASK}storage:\n  type: sqlite\n  path: {}\n",
        path.to_str().unwrap()
    );
    let agent = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider.clone())
        .build_task_host()
        .await
        .unwrap();
    assert!(
        agent
            .storage()
            .unwrap()
            .supports(StorageCapability::TaskRuns)
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    agent.set_actor_id("alice").unwrap();
    let paused = agent.run_task("Research report", None).await.unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let snapshot = agent
        .storage()
        .unwrap()
        .load_task_run(&paused.run.key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.revision, paused.run.revision);
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert!(
        !payload
            .adapters
            .iter()
            .any(|adapter| adapter.id == "runtime.restore")
    );
    // Complete cleanup before dropping the live runtime; the persisted pause itself never authorizes owner takeover.
    let cancelled = agent
        .cancel_task(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap();
    assert_eq!(cancelled.disposition, CancellationDisposition::Acknowledged);
    drop(agent);
    let reopened = ai_agents_storage::SqliteStorage::new(path.to_str().unwrap())
        .await
        .unwrap();
    let saved = reopened
        .load_task_run(&paused.run.key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(saved.status, TaskRunStatus::Cancelled);
    assert_eq!(saved.actor_id.as_deref(), Some("alice"));
    reopened.close().await;
    crate::remove_sqlite_test_directory(&directory)
        .await
        .unwrap();
}
