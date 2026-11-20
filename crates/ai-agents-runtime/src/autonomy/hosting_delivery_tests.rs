//! Delivery regression for reuse of a configured runner across successive automatic admissions.

use super::*;
use crate::{Agent, AgentBuilder, autonomy::runner_tests::RecordingProvider};

#[tokio::test]
// A later admission must not replace an earlier parent's live slot before the earlier public result is delivered.
async fn hosted_late_publication_keeps_each_live_run_delivery_slot() {
    let agent = AgentBuilder::from_yaml(
        "name: DeliveryHost\nsystem_prompt: Helpful\nautonomy:\n  enabled: true\n  completion: {response_contains: done}\n",
    ).unwrap().llm(RecordingProvider::new(&["done", "done"])).build().unwrap();
    let runner = agent.task_runner().unwrap();
    let first = agent.run_task("First objective", None).await.unwrap();
    let first_slot = Arc::new(super::super::automatic::AutomaticOutcome {
        key: Some(first.run.key.clone()),
        ..Default::default()
    });
    runner.bind_automatic_outcome(first_slot.clone());
    let second = agent.run_task("Second objective", None).await.unwrap();
    let second_slot = Arc::new(super::super::automatic::AutomaticOutcome {
        key: Some(second.run.key.clone()),
        ..Default::default()
    });
    runner.bind_automatic_outcome(second_slot.clone());
    assert!(second_slot.result.lock().is_none());
    runner.publish_automatic(&first);
    assert_eq!(
        first_slot.result.lock().as_ref().unwrap().run.key,
        first.run.key
    );
    assert!(second_slot.result.lock().is_none());
    runner.publish_automatic(&second);
    assert_eq!(
        second_slot.result.lock().as_ref().unwrap().run.key,
        second.run.key
    );
    drop(first_slot);
    runner.bind_automatic_outcome(second_slot.clone());
    assert!(
        !runner
            .automatic_outcomes
            .lock()
            .contains_key(&first.run.key.run_id)
    );
}

#[tokio::test]
// Reproduce the release-before-retirement window with a real completed execution, then reject a new automatic objective before its execution exists.
async fn hosted_preexecution_failure_releases_its_owner_with_an_old_active_slot() {
    let provider = RecordingProvider::gated();
    let agent = AgentBuilder::from_yaml(
        "name: CleanupHost\nsystem_prompt: Helpful\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        objective_template: '{{ context.objective }}'\n        completion: {response_contains: done}\n",
    ).unwrap().llm(provider.clone()).build().unwrap();
    agent
        .set_context("objective", json!("valid objective"))
        .unwrap();
    let runner = agent.task_runner().unwrap();
    let old_execution = {
        let work = agent.chat("First task");
        tokio::pin!(work);
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::select! {
                _ = provider.entered.as_ref().unwrap().acquire() => {},
                result = &mut work => panic!("provider did not park: {result:?}"),
            }
        })
        .await
        .unwrap();
        let execution = runner.active.lock().as_ref().unwrap().clone();
        provider.release.as_ref().unwrap().add_permits(1);
        work.await.unwrap();
        execution
    };
    // This slot is exactly the old execution visible after runtime release but before the old caller retires it.
    *runner.active.lock() = Some(old_execution.clone());
    agent.set_context("objective", json!("")).unwrap();
    assert!(agent.chat("Invalid new objective").await.is_err());
    agent
        .set_context("objective", json!("valid again"))
        .expect("failed pre-execution owner must be released");
    runner.retire_owner(&old_execution.owner);
    provider.release.as_ref().unwrap().add_permits(1);
    let response = agent.chat("Next task").await.unwrap();
    assert_eq!(
        response.metadata.unwrap()["task_run"]["status"],
        "completed"
    );
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}
