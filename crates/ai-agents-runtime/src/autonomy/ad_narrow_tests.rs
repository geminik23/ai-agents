use super::*;

// A source label cannot enter a reserved runtime; explicit spawned callbacks inherit the same narrow-only executor and ledger.
#[tokio::test]
async fn ad_narrow_owned_and_spawned_callbacks_use_final_lifecycle_admission() {
    for spawned in [false, true] {
        let profile = AutonomyProfile {
            lifecycle: Some(
                serde_yaml::from_str("- id: explore\n  required_before_tools: [work_probe]\n")
                    .unwrap(),
            ),
            max_tool_calls: Some(1),
            ..Default::default()
        };
        let (agent, execution, calls) = shared_executor_fixture(profile.clone(), false).await;
        let forged =
            ToolExecutionRequest::new("forged", "work_probe", json!({}), ToolCallSource::Task);
        assert!(agent.invoke_tool(forged).await.is_err());
        scope_execution(execution.clone(), async {
            let target = agent.clone();
            let denied = super::super::owned_maintenance("callback", async move {
                assert!(
                    target
                        .invoke_tool(ToolExecutionRequest::new(
                            "denied",
                            "work_probe",
                            json!({}),
                            ToolCallSource::Task
                        ))
                        .await
                        .is_err()
                );
                Ok(())
            });
            if spawned {
                tokio::spawn(denied).await.unwrap().unwrap();
            } else {
                denied.await.unwrap();
            }
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            let before: TaskCheckpointPayload =
                serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
            assert_eq!(before.counters.tool_attempts, 0);
            let mut lifecycle = LifecycleState::new(&profile, false).unwrap();
            lifecycle
                .complete_stage(&profile, "explore", GateOutcome::Pass, GateOutcome::Pass)
                .unwrap();
            execution.set_lifecycle(&profile, &lifecycle);
            let target = agent.clone();
            let allowed = super::super::owned_maintenance("callback", async move {
                let record = target
                    .invoke_tool(ToolExecutionRequest::new(
                        "allowed",
                        "work_probe",
                        json!({}),
                        ToolCallSource::Task,
                    ))
                    .await?;
                assert!(record.executed && record.success);
                Ok(())
            });
            if spawned {
                tokio::spawn(allowed).await.unwrap().unwrap();
            } else {
                allowed.await.unwrap();
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let after: TaskCheckpointPayload =
                serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
            assert_eq!(after.counters.tool_attempts, 1);
            assert!(!execution.owner.work.unsettled());
        })
        .await;
        agent.release_autonomy_run(&execution.owner).await.unwrap();
    }
}
