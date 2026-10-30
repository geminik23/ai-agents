//! Host sessions witness real configured participants, public ACK ordering and exact scoped controls without terminal dependencies.

use super::continuation_test_support::{self as support, ReceiptStore};
use super::runner_tests::RecordingProvider;
use super::*;
use crate::AgentBuilder;
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};
const TASK: &str = "name: SessionHost\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  completion: {response_contains: done}\n  premature_finish: {action: ask_user, on_unavailable: stop}\n";
const STATE: &str = "name: SessionChild\nsystem_prompt: Helpful\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        max_turns: 4\n        max_llm_calls: 4\n        completion: {response_contains: done}\n        premature_finish: {action: ask_user, on_unavailable: stop}\n";

// A supplied SDK controller joins only when its exact pointer enters real work; equivalent labels cannot replace the retained handle.
#[tokio::test]
async fn session_supplied_sdk_runner_is_admitted_before_dispatch_and_cannot_be_rebound() {
    let provider = RecordingProvider::gated();
    let agent = Arc::new(
        AgentBuilder::from_yaml(TASK)
            .unwrap()
            .llm(provider.clone())
            .build()
            .unwrap(),
    );
    let config = crate::spec::AgentSpec::from_yaml_strict(TASK)
        .unwrap()
        .autonomy;
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "sdk-session.v1".into()).unwrap(),
    );
    let runner = Arc::new(
        AutonomyRunner::try_new(
            agent.clone(),
            config.clone(),
            Default::default(),
            store.clone(),
            "sdk-session.v1".into(),
        )
        .unwrap(),
    );
    let session = TaskHostSession::new(true);
    let work = session.scope_runner(runner.clone(), runner.run("analysis", None));
    tokio::pin!(work);
    tokio::select! { _ = provider.entered.as_ref().unwrap().acquire() => {}, result = &mut work => panic!("provider did not gate: {result:?}") }
    let admitted = session.runs();
    assert_eq!(admitted.len(), 1);
    assert!(admitted[0].execution_admitted());
    assert_eq!(admitted[0].published_revision(), None);
    assert!(Arc::ptr_eq(&admitted[0].runner, &runner));
    provider.release.as_ref().unwrap().add_permits(1);
    let result = work.await.unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    session
        .observe_result(runner.clone(), &result)
        .await
        .unwrap();
    let other = Arc::new(
        AutonomyRunner::try_new(
            agent,
            config,
            Default::default(),
            store,
            "sdk-session.v1".into(),
        )
        .unwrap(),
    );
    assert!(session.observe_result(other, &result).await.is_err());
    assert_eq!(session.runs().len(), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Checked scope binding freezes the builder's actual loaded definitions instead of trusting two matching but unrelated filesystem samples.
#[tokio::test]
async fn session_scope_checks_and_retains_actual_external_skill_definitions() {
    let dir = std::env::temp_dir().join(format!("task-scope-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("work.skill.yaml");
    let skill = |text: &str| {
        format!(
            "id: work\ndescription: Research\ntrigger: Research requested\nsteps:\n  - prompt: '{text}'\n"
        )
    };
    std::fs::write(&file, skill("first definition")).unwrap();
    let spec = crate::spec::AgentSpec::from_yaml_strict(&format!(
        "{TASK}skills: [{{file: work.skill.yaml}}]\n"
    ))
    .unwrap();
    let old = TaskHostScope::from_spec(&spec, &dir, None).unwrap();
    std::fs::write(&file, skill("second definition")).unwrap();
    let provider = RecordingProvider::new(&[]);
    assert!(
        AgentBuilder::from_spec_with_base_dir(spec.clone(), &dir)
            .llm(provider.clone())
            .task_host_scope(old)
            .is_err()
    );
    let current = TaskHostScope::from_spec(&spec, &dir, None).unwrap();
    let builder = AgentBuilder::from_spec_with_base_dir(spec, &dir)
        .llm(provider.clone())
        .task_host_scope(current)
        .unwrap();
    std::fs::write(&file, skill("third definition")).unwrap();
    let agent = builder.build_task_host().await.unwrap();
    let actual = serde_json::to_string(agent.skills()).unwrap();
    assert!(actual.contains("second definition"));
    assert!(!actual.contains("third definition"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    std::fs::remove_dir_all(dir).unwrap();
}

// Snapshot-only and missing durable stores cannot create a task management backend or open a hidden network service.
#[tokio::test]
async fn session_scope_management_rejects_snapshot_only_and_missing_journals() {
    let mut spec = crate::spec::AgentSpec::from_yaml_strict(TASK).unwrap();
    let dir = std::env::temp_dir().join(format!("task-missing-{}", uuid::Uuid::new_v4()));
    for storage in [
        crate::spec::StorageConfig::None,
        crate::spec::StorageConfig::file(dir.to_string_lossy()),
        crate::spec::StorageConfig::redis("redis://127.0.0.1:1"),
        crate::spec::StorageConfig::sqlite(dir.join("missing.sqlite").to_string_lossy()),
    ] {
        spec.storage = storage;
        let scope = TaskHostScope::from_spec(&spec, std::path::Path::new("."), None).unwrap();
        assert!(scope.open_store().await.is_err());
        assert!(!dir.exists());
    }
}

// A backend terminal row with a lost ACK is not public completion and cannot erase live protected ownership.
#[tokio::test]
async fn session_lost_terminal_ack_never_publishes_a_successful_host_result() {
    for &sqlite in support::backends() {
        let inner = support::store("SessionHost".into(), "session.lost", sqlite).await;
        let fault = ReceiptStore::new(inner.clone(), "lost", TaskRunStatus::Completed);
        fault.armed.store(true, Ordering::SeqCst);
        let provider = RecordingProvider::new(&["done"]);
        let agent = AgentBuilder::from_yaml(TASK)
            .unwrap()
            .llm(provider.clone())
            .automatic_task_options(AutomaticTaskOptions {
                store: Some(fault.clone()),
                config_identity: Some("session.lost".into()),
                ..Default::default()
            })
            .build_task_host()
            .await
            .unwrap();
        let session = TaskHostSession::new(true);
        assert!(
            session
                .scope(agent.run_task("analysis", None))
                .await
                .is_err()
        );
        fault.assert_hit();
        let runs = session.runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].published_revision(), None);
        assert!(session.has_protected_work());
        assert_eq!(
            inner
                .load(&runs[0].key.run_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskRunStatus::Completed
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(agent.set_context("must-not-release", json!(true)).is_err());
    }
}

// Host shutdown targets the actual admitted execution, not a stale display revision, while user-supplied revision checks remain strict.
#[tokio::test]
async fn session_running_shutdown_uses_live_serial_cancellation_without_rebinding_responses() {
    let provider = RecordingProvider::gated();
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider.clone())
        .build_task_host()
        .await
        .unwrap();
    let runner = agent.task_runner().unwrap();
    assert!(
        runner
            .request_running_cancel("not-admitted")
            .await
            .unwrap()
            .is_none()
    );
    let session = TaskHostSession::new(true);
    let work = session.scope(agent.run_task("operations", None));
    tokio::pin!(work);
    tokio::select! {_=provider.entered.as_ref().unwrap().acquire()=>{},result=&mut work=>panic!("provider did not gate: {result:?}")}
    let runs = session.runs();
    assert_eq!(runs.len(), 1);
    assert!(runs[0].execution_admitted());
    let run = &runs[0];
    let snapshot = runner
        .task_store()
        .load(&run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    assert!(snapshot.revision > 0);
    assert!(runner.cancel_task(&run.key.run_id, 0).await.is_err());
    let requested = runner
        .request_running_cancel(&run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    assert!(requested.cancel_requested);
    provider.release.as_ref().unwrap().add_permits(1);
    let result = work.await.unwrap();
    assert_ne!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// The catalogue is inert before real admission, and later lookup cannot manufacture a second participating runtime.
#[tokio::test]
async fn session_catalogue_requires_actual_admission_and_preserves_same_run() {
    let provider = RecordingProvider::new(&["draft", "done"]);
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider.clone())
        .build_task_host()
        .await
        .unwrap();
    let session = TaskHostSession::new(true);
    let runner = agent.task_runner().unwrap();
    assert!(session.runs().is_empty());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let paused = session
        .scope(agent.run_task("research", None))
        .await
        .unwrap();
    let runs = session.runs();
    assert_eq!(runs.len(), 1);
    assert!(Arc::ptr_eq(&runs[0].runner, &runner));
    assert_eq!(runs[0].published_revision(), Some(paused.run.revision));
    let request = runner
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await
        .unwrap()
        .unwrap();
    let done = session
        .scope(runner.resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: request.request_id,
                answer: json!({"answered":true,"other_text":"continue"}),
            },
        ))
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.key, paused.run.key);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        session.runs()[0].published_revision(),
        Some(done.run.revision)
    );
}

// Ordinary concurrent workers inherit one actual foreground catalogue; an unrelated registered child never appears in it.
#[tokio::test]
async fn session_multiple_ordinary_children_are_serviceable_before_parent_final() {
    use crate::spawner::{AgentRegistry, SpawnedAgent};
    let registry = AgentRegistry::new();
    let mut providers = vec![];
    for id in ["first", "second", "unrelated"] {
        let yaml = STATE.replace("SessionChild", id);
        let spec = crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
        let provider = RecordingProvider::new(&["draft", "done"]);
        providers.push(provider.clone());
        let agent = AgentBuilder::from_spec(spec.clone())
            .llm(provider)
            .build_task_host()
            .await
            .unwrap();
        registry
            .register(SpawnedAgent::from_runtime(id.into(), agent, spec))
            .await
            .unwrap();
    }
    let refs = vec![
        ai_agents_state::ConcurrentAgentRef::Id("first".into()),
        ai_agents_state::ConcurrentAgentRef::Id("second".into()),
    ];
    let aggregation = serde_yaml::from_str("strategy: all").unwrap();
    let session = TaskHostSession::new(true);
    let work = session.scope(crate::orchestration::concurrent(
        &registry,
        "research",
        &refs,
        &aggregation,
        None,
        Some(2),
        Some(30_000),
        ai_agents_state::PartialFailureAction::Abort,
        None,
    ));
    tokio::pin!(work);
    tokio::time::timeout(std::time::Duration::from_secs(10),async {
        loop {
            tokio::select! {
                result=&mut work=>panic!("parent returned before both decisions: {result:?}"),
                _=session.changed()=>{
                    let runs=session.runs();
                    if runs.len()==2 && runs.iter().all(|run|run.published_revision().is_some()){break;}
                }
            }
        }
    }).await.unwrap();
    let mut runs = session.runs();
    runs.reverse();
    for run in runs {
        let revision = run.published_revision().unwrap();
        let prompt = run
            .runner
            .task_interaction(&run.key.run_id, revision)
            .await
            .unwrap()
            .unwrap();
        let done = session
            .scope(run.runner.resume(
                &run.key.run_id,
                revision,
                TaskResumeInput::UserAnswer {
                    request_id: prompt.request_id,
                    answer: json!("continue"),
                },
            ))
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
    }
    let result = work.await.unwrap();
    assert!(result.agent_results.iter().all(|r| r.success));
    assert_eq!(providers[0].calls.load(Ordering::SeqCst), 2);
    assert_eq!(providers[1].calls.load(Ordering::SeqCst), 2);
    assert_eq!(providers[2].calls.load(Ordering::SeqCst), 0);
}

// A stored pause is not a public host response boundary until its original ACK and preparation complete.
#[tokio::test]
async fn session_does_not_publish_prompt_before_pause_ack() {
    let inner = support::store("SessionHost".into(), "session.ack", false).await;
    let fault = ReceiptStore::new(inner, "hold", TaskRunStatus::Paused);
    fault.armed.store(true, Ordering::SeqCst);
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(RecordingProvider::new(&["draft"]))
        .automatic_task_options(AutomaticTaskOptions {
            store: Some(fault.clone()),
            config_identity: Some("session.ack".into()),
            ..Default::default()
        })
        .build_task_host()
        .await
        .unwrap();
    let session = TaskHostSession::new(true);
    let work = session.scope(agent.run_task("research", None));
    tokio::pin!(work);
    tokio::select! { _=fault.reached.acquire()=>{},result=&mut work=>panic!("pause ACK was not held: {result:?}") }
    assert_eq!(session.runs().len(), 1);
    assert_eq!(session.runs()[0].published_revision(), None);
    fault.release.add_permits(1);
    let result = work.await.unwrap();
    assert_eq!(
        session.runs()[0].published_revision(),
        Some(result.run.revision)
    );
    session
        .scope(agent.cancel_task(&result.run.key.run_id, result.run.revision))
        .await
        .unwrap();
}

// Endpoint loss uses frozen unavailable policy and typed cleanup rather than replaying the pending model invocation.
#[tokio::test]
async fn session_confirmed_input_loss_preserves_stop_and_fail() {
    for fail in [false, true] {
        let provider = RecordingProvider::new(&["draft", "must-not-run"]);
        let yaml = if fail {
            TASK.replace("on_unavailable: stop", "on_unavailable: fail")
        } else {
            TASK.into()
        };
        let agent = AgentBuilder::from_yaml(&yaml)
            .unwrap()
            .llm(provider.clone())
            .build_task_host()
            .await
            .unwrap();
        let session = TaskHostSession::new(true);
        let paused = session
            .scope(agent.run_task("research", None))
            .await
            .unwrap();
        session.close_input();
        let runner = agent.task_runner().unwrap();
        let done = session
            .scope(runner.input_unavailable_paused(&paused.run.key.run_id, paused.run.revision))
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            if fail {
                TaskRunStatus::Failed
            } else {
                TaskRunStatus::Incomplete
            }
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(agent.set_context("released", json!(true)).is_ok());
    }
}

// An explicitly supplied declarative scope is deterministic but never claims to bind opaque implementations or authenticate callers.
#[tokio::test]
async fn session_scope_normalization_and_configuration_filter_are_shared() {
    let spec = crate::spec::AgentSpec::from_yaml_strict(TASK).unwrap();
    let scope =
        TaskHostScope::from_spec(&spec, std::path::Path::new("."), Some("alice".into())).unwrap();
    let other =
        TaskHostScope::from_spec(&spec, std::path::Path::new("."), Some("alice".into())).unwrap();
    assert_eq!(
        scope.task_options().config_identity,
        other.task_options().config_identity
    );
    let storage = Arc::new(ai_agents_storage::InMemoryTaskStorage::default());
    let agent = AgentBuilder::from_spec(spec.clone())
        .llm(RecordingProvider::new(&["done"]))
        .storage(storage.clone())
        .automatic_task_options(scope.task_options())
        .build_task_host()
        .await
        .unwrap();
    agent.set_actor_id("alice").unwrap();
    let done = agent.run_task("research", None).await.unwrap();
    let store = ScopedTaskRunStore::new(
        storage.clone(),
        spec.name.clone(),
        Some("alice".into()),
        scope.task_options().config_identity.unwrap(),
    )
    .unwrap();
    assert_eq!(store.list_compatible().await.unwrap().len(), 1);
    let mut changed = spec.clone();
    changed.system_prompt = "different contract".into();
    let changed =
        TaskHostScope::from_spec(&changed, std::path::Path::new("."), Some("alice".into()))
            .unwrap();
    let other = ScopedTaskRunStore::new(
        storage,
        spec.name,
        Some("alice".into()),
        changed.task_options().config_identity.unwrap(),
    )
    .unwrap();
    assert!(other.list_compatible().await.unwrap().is_empty());
    assert!(other.load(&done.run.key.run_id).await.is_err());
}
