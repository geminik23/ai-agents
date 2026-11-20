//! Existing conversation algorithms retain their own topology and positions across executable reconstruction.
use super::*;

// Group/handoff reconstruction restores typed conversation positions, not a generic persisted DAG or a new routing decision.
#[tokio::test]
async fn restoration_declared_group_styles_and_handoff_preserve_saved_positions() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for (pattern, effects) in [
            (
                "handoff:\n        initial_agent: worker\n        available_agents: [worker, peer]\n        max_handoffs: 2",
                1,
            ),
            (
                "group_chat:\n        participants:\n          - id: worker\n          - id: peer\n        style: brainstorm\n        max_rounds: 1\n        termination:\n          method: max_rounds",
                2,
            ),
            (
                "group_chat:\n        participants:\n          - id: worker\n          - id: peer\n        style: consensus\n        max_rounds: 1\n        termination:\n          method: max_rounds",
                2,
            ),
            (
                "group_chat:\n        participants:\n          - id: worker\n          - id: peer\n        style: maker_checker\n        maker_checker:\n          max_iterations: 1\n          acceptance_criteria: complete",
                2,
            ),
            (
                "group_chat:\n        participants:\n          - id: worker\n          - id: peer\n        style: debate\n        debate:\n          rounds: 1\n          synthesizer: worker",
                3,
            ),
        ] {
            let mut host = host();
            Arc::get_mut(&mut host).unwrap().root=AgentSpec::from_yaml_strict(&format!("name: RestoreRoot\nsystem_prompt: root\nstates:\n  initial: work\n  states:\n    work:\n      {pattern}\n")).unwrap();
            let root = declared_source(&host, pattern).await;
            let old = Arc::downgrade(&root);
            let store = crate::autonomy::continuation_test_support::store(
                root.info().id,
                "restore.v1",
                sqlite,
            )
            .await;
            let runner = AutonomyRunner::try_new(
                root.clone(),
                config(),
                Default::default(),
                store.clone(),
                "restore.v1".into(),
            )
            .unwrap();
            let source = runner.run("objective", None).await.unwrap();
            assert_eq!(
                source.run.status,
                TaskRunStatus::Paused,
                "{pattern}: {:?}",
                source.run.stop_reason
            );
            let before = host.calls.load(Ordering::SeqCst);
            let committed = host.effects.load(Ordering::SeqCst);
            let prepared = runner
                .checkpoint_for_restore(&source.run.key.run_id, source.run.revision, host.as_ref())
                .await
                .unwrap();
            drop(runner);
            drop(root);
            assert!(old.upgrade().is_none());
            let request = TaskRestoreRequest {
                run_id: prepared.run.key.run_id,
                expected_revision: prepared.run.revision,
                config_identity: "restore.v1".into(),
                config: config(),
                ceilings: Default::default(),
            };
            let (restored, paused) = AutonomyRunner::restore_paused(request, store, host.clone())
                .await
                .unwrap();
            let mut current = approve(&restored, &paused.run).await;
            assert_eq!(host.calls.load(Ordering::SeqCst), before);
            assert_eq!(host.effects.load(Ordering::SeqCst), committed);
            let mut pauses = 0;
            while current.run.status == TaskRunStatus::Paused {
                pauses += 1;
                assert!(pauses <= effects, "extra conversation turn: {pattern}");
                current = answer(&restored, &current.run).await.unwrap();
            }
            assert_eq!(
                current.run.status,
                TaskRunStatus::Completed,
                "{pattern}: {:?}",
                current.run.stop_reason
            );
            assert_eq!(
                host.effects.load(Ordering::SeqCst),
                effects,
                "completed effects replayed: {pattern}"
            );
        }
    }
}
