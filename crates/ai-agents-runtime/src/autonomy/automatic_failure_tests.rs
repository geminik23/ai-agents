//! Ordinary composing callers receive truthful failure when a resumed automatic child loses execution acknowledgement.

use super::*;
use crate::autonomy::continuation_test_support::{self as support, ReceiptStore};
use crate::orchestration::{PipelineStage, pipeline};
use crate::spawner::{AgentRegistry, SpawnedAgent};

// A direct public resume failure cannot leave its original ordinary parent silently waiting on an obsolete safe pause.
#[tokio::test]
async fn auto_uncertain_resumed_child_wakes_parent_without_fabricating_success_or_replay() {
    for mode in ["reject", "lost", "drop"] {
        let registry = AgentRegistry::new();
        let inner = support::store("auto".into(), "automatic.v1", false).await;
        let fault = ReceiptStore::new(inner, mode, TaskRunStatus::Completed);
        let yaml = STATE.replace("        completion:", "        premature_finish: {action: ask_user, on_unavailable: stop}\n        completion:");
        let main = RecordingProvider::new(&["draft", "done"]);
        let spec = crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
        let runtime = AgentBuilder::from_spec(spec.clone())
            .llm(main.clone())
            .automatic_task_options(AutomaticTaskOptions {
                store: Some(fault.clone()),
                config_identity: Some("automatic.v1".into()),
                ..Default::default()
            })
            .build()
            .unwrap();
        registry
            .register(SpawnedAgent::from_runtime("child".into(), runtime, spec))
            .await
            .unwrap();
        let child = registry.get("child").unwrap();
        let stages = vec![PipelineStage::id("child")];
        let mut parent = Box::pin(pipeline(
            &registry,
            "objective",
            &stages,
            Some(30_000),
            None,
            None,
        ));
        let paused = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                tokio::select! {
                    result = &mut parent => panic!("parent returned before child decision: {result:?}"),
                    _ = tokio::task::yield_now() => if let Some(result) = child.automatic_task_result() { break result; },
                }
            }
        }).await.unwrap();
        fault.armed.store(true, Ordering::SeqCst);
        let runner = child.automatic_task_runner().unwrap();
        let mut operation = Box::pin(runner.resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.unwrap().id,
                answer: json!("continue"),
            },
        ));
        if mode == "drop" {
            tokio::select! {
                result = &mut operation => panic!("receipt wait unexpectedly finished: {result:?}"),
                permit = fault.reached.acquire() => permit.unwrap().forget(),
            }
            drop(operation);
        } else {
            assert!(operation.await.is_err());
        }
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(10), parent)
                .await
                .unwrap()
                .is_err()
        );
        assert!(child.chat("cannot replay uncertain work").await.is_err());
        assert_eq!(main.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            child.automatic_task_result().unwrap().run.status,
            TaskRunStatus::Paused
        );
    }
}
