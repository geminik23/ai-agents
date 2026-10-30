use ai_agents_core::autonomy::*;
use ai_agents_core::{AgentError, AgentStorage};
use serde_json::json;
use std::sync::Arc;

fn snapshot(run_id: &str) -> TaskRunSnapshot {
    let now = chrono::Utc::now();
    TaskRunSnapshot {
        schema_version: TASK_RUN_SCHEMA_VERSION,
        key: TaskRunKey {
            agent_id: "agent".into(),
            run_id: run_id.into(),
        },
        actor_id: Some("actor".into()),
        revision: 0,
        status: TaskRunStatus::Paused,
        owner_token: None,
        cancel_requested: false,
        created_at: now,
        updated_at: now,
        payload: json!({"exact": "retain"}),
    }
}

async fn terminal_contract(storage: Arc<dyn AgentStorage>) {
    let paused = snapshot("paused-cancel");
    storage.create_task_run(&paused).await.unwrap();
    storage
        .mutate_task_run(
            &paused.key,
            &TaskRunMutation::RequestCancel {
                expected_revision: 0,
            },
        )
        .await
        .unwrap();
    let result = storage
        .mutate_task_run(
            &paused.key,
            &TaskRunMutation::AcknowledgeCancel {
                expected_revision: 1,
                payload: paused.payload.clone(),
            },
        )
        .await
        .unwrap();
    assert_eq!(result.status, TaskRunStatus::Cancelled);
    assert_eq!(result.payload, paused.payload);
    for (index, status) in [
        TaskRunStatus::Completed,
        TaskRunStatus::Incomplete,
        TaskRunStatus::Failed,
        TaskRunStatus::Cancelled,
        TaskRunStatus::LimitReached,
    ]
    .into_iter()
    .enumerate()
    {
        let s = snapshot(&format!("terminal-{index}"));
        storage.create_task_run(&s).await.unwrap();
        storage
            .mutate_task_run(
                &s.key,
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
            )
            .await
            .unwrap();
        let terminal = storage
            .mutate_task_run(
                &s.key,
                &TaskRunMutation::Checkpoint {
                    expected_revision: 1,
                    owner_token: "owner".into(),
                    status,
                    payload: s.payload.clone(),
                    release: true,
                },
            )
            .await
            .unwrap();
        for mutation in [
            TaskRunMutation::Claim {
                expected_revision: 2,
                owner_token: "new".into(),
            },
            TaskRunMutation::Checkpoint {
                expected_revision: 2,
                owner_token: "owner".into(),
                status: TaskRunStatus::Running,
                payload: json!({}),
                release: false,
            },
            TaskRunMutation::RequestCancel {
                expected_revision: 2,
            },
            TaskRunMutation::AcknowledgeCancel {
                expected_revision: 2,
                payload: json!({}),
            },
            TaskRunMutation::Recover {
                expected_revision: 2,
                owner_token: "owner".into(),
            },
            TaskRunMutation::ResolveRecovery {
                expected_revision: 2,
                payload: json!({}),
            },
        ] {
            assert!(matches!(
                storage.mutate_task_run(&s.key, &mutation).await,
                Err(AgentError::TaskRunStorage(
                    TaskRunStorageError::NotResumable
                ))
            ));
            assert_eq!(
                serde_json::to_value(storage.load_task_run(&s.key).await.unwrap().unwrap())
                    .unwrap(),
                serde_json::to_value(&terminal).unwrap()
            );
        }
    }
}

#[tokio::test]
async fn task_memory_terminal_and_unowned_cancellation_contract() {
    terminal_contract(Arc::new(crate::InMemoryTaskStorage::default())).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn task_sqlite_terminal_and_unowned_cancellation_contract() {
    terminal_contract(Arc::new(crate::SqliteStorage::in_memory().await.unwrap())).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn task_sqlite_actor_delete_races_claim_and_tombstones_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("race.db");
    let left = crate::SqliteStorage::new(path.to_str().unwrap())
        .await
        .unwrap();
    let right = crate::SqliteStorage::new(path.to_str().unwrap())
        .await
        .unwrap();
    let initial = snapshot("race");
    left.create_task_run(&initial).await.unwrap();
    let mutation = TaskRunMutation::Claim {
        expected_revision: 0,
        owner_token: "claim".into(),
    };
    let (claim, delete) = tokio::join!(
        left.mutate_task_run(&initial.key, &mutation),
        right.delete_actor_data("agent", "actor")
    );
    assert_ne!(claim.is_ok(), delete.is_ok());
    match claim {
        Ok(claimed) => {
            assert!(matches!(
                delete,
                Err(AgentError::TaskRunStorage(TaskRunStorageError::Owned))
            ));
            left.mutate_task_run(
                &initial.key,
                &TaskRunMutation::Checkpoint {
                    expected_revision: claimed.revision,
                    owner_token: "claim".into(),
                    status: TaskRunStatus::Paused,
                    payload: initial.payload.clone(),
                    release: true,
                },
            )
            .await
            .unwrap();
            right.delete_actor_data("agent", "actor").await.unwrap();
        }
        Err(AgentError::TaskRunStorage(TaskRunStorageError::NotFound)) => {}
        other => panic!("unexpected claim result: {other:?}"),
    }
    let direct = snapshot("direct-delete");
    left.create_task_run(&direct).await.unwrap();
    left.delete_task_run(&direct.key, 0).await.unwrap();
    left.close().await;
    right.close().await;
    let reopened = crate::SqliteStorage::new(path.to_str().unwrap())
        .await
        .unwrap();
    for old in [&initial, &direct] {
        assert!(reopened.load_task_run(&old.key).await.unwrap().is_none());
        assert!(matches!(
            reopened.create_task_run(old).await,
            Err(AgentError::TaskRunStorage(
                TaskRunStorageError::AlreadyExists
            ))
        ));
    }
    reopened.close().await;
}
