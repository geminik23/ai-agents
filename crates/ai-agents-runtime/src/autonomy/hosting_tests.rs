//! Configured hosting exercises the real controller and scoped store without provider credentials or external effects.

use super::runner_tests::RecordingProvider;
use super::*;
use crate::{Agent, AgentBuilder};
use std::sync::{Arc, atomic::Ordering};

const TASK: &str = "name: HostedTask\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  completion: {response_contains: done}\n";
const STATE: &str = "name: HostedState\nsystem_prompt: Helpful\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        max_turns: 4\n        max_llm_calls: 4\n        completion: {response_contains: done}\n        premature_finish: {action: ask_user, on_unavailable: stop}\n";

#[tokio::test]
// Obtaining a runner performs no semantic work; repeated explicit operations retain the same runtime and scoped store.
async fn hosted_explicit_tasks_reuse_the_configured_runner_without_starting_on_lookup() {
    let provider = RecordingProvider::new(&["draft", "done", "done"]);
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider.clone())
        .build()
        .unwrap();
    assert!(agent.automatic_task_runner().is_none());
    let runner = agent.task_runner().unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let first = agent
        .run_task("Research without files", None)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Completed);
    let second = agent
        .run_task("Another independent objective", None)
        .await
        .unwrap();
    assert_eq!(second.run.status, TaskRunStatus::Completed);
    assert_ne!(first.run.key, second.run.key);
    assert!(Arc::ptr_eq(&runner, &agent.task_runner().unwrap()));
    assert_eq!(runner.task_store().list().await.unwrap().len(), 2);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
// Automatic admission binds its delivery slot to the existing configured runner, and convenience resume preserves the exact run.
async fn hosted_automatic_pause_and_convenience_resume_share_runner_and_outcome() {
    let provider = RecordingProvider::new(&["draft", "done"]);
    let agent = AgentBuilder::from_yaml(STATE)
        .unwrap()
        .llm(provider.clone())
        .build()
        .unwrap();
    let runner = agent.task_runner().unwrap();
    let response = agent.chat("Prepare research").await.unwrap();
    assert_eq!(response.metadata.unwrap()["task_run"]["status"], "paused");
    assert!(Arc::ptr_eq(
        &runner,
        &agent.automatic_task_runner().unwrap()
    ));
    let paused = agent.automatic_task_result().unwrap();
    let result = agent
        .resume_task(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                answer: serde_json::json!("Continue research"),
            },
        )
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.key, paused.run.key);
    assert_eq!(result.run.counters.llm_attempts, 2);
    assert_eq!(
        agent.automatic_task_result().unwrap().run.revision,
        result.run.revision
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
// Host infrastructure opt-in does not enable a disabled profile or create a provider attempt.
async fn hosted_disabled_profile_stays_disabled_with_explicit_options() {
    let provider = RecordingProvider::new(&["must not execute"]);
    let agent = AgentBuilder::new()
        .system_prompt("Helpful")
        .llm(provider.clone())
        .automatic_task_options(AutomaticTaskOptions::default())
        .build()
        .unwrap();
    let runner = agent.task_runner().unwrap();
    assert!(agent.run_task("Not enabled", None).await.is_err());
    assert!(runner.task_store().list().await.unwrap().is_empty());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
// An actor switch after terminal settlement gets a different scoped runner; earlier actor data is not exposed by the new store.
async fn hosted_actor_switch_does_not_reuse_another_actors_store() {
    let provider = RecordingProvider::new(&["done", "done"]);
    let agent = AgentBuilder::from_yaml(TASK)
        .unwrap()
        .llm(provider)
        .build()
        .unwrap();
    agent.set_actor_id("first").unwrap();
    let first = agent.task_runner().unwrap();
    let old = agent.run_task("First actor", None).await.unwrap();
    agent.set_actor_id("second").unwrap();
    let second = agent.task_runner().unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert!(second.task_store().list().await.unwrap().is_empty());
    let current = agent.run_task("Second actor", None).await.unwrap();
    assert_eq!(current.run.actor_id.as_deref(), Some("second"));
    assert!(matches!(
        second.task_store().load(&old.run.key.run_id).await,
        Err(ai_agents_core::AgentError::TaskRunStorage(
            TaskRunStorageError::InvalidCheckpoint
        ))
    ));
    assert_eq!(first.task_store().list().await.unwrap().len(), 1);
    assert_eq!(second.task_store().list().await.unwrap().len(), 1);
}

#[tokio::test]
// A trusted host answers the retained task after turn-local actor context has ended without rebuilding its store from session defaults.
async fn hosted_actor_turn_pause_resumes_in_its_original_scope() {
    let provider = RecordingProvider::new(&["draft", "done"]);
    let agent = AgentBuilder::from_yaml(STATE)
        .unwrap()
        .llm(provider.clone())
        .build()
        .unwrap();
    agent
        .chat_as_actor("alice", "Prepare a report")
        .await
        .unwrap();
    let paused = agent.automatic_task_result().unwrap();
    assert_eq!(paused.run.actor_id.as_deref(), Some("alice"));
    let runner = agent.automatic_task_runner().unwrap();
    let done = agent
        .resume_task(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                answer: serde_json::json!("Continue the report"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.key, paused.run.key);
    assert_eq!(done.run.actor_id.as_deref(), Some("alice"));
    assert!(Arc::ptr_eq(
        &runner,
        &agent.automatic_task_runner().unwrap()
    ));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}
