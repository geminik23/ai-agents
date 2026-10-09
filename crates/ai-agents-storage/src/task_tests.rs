use ai_agents_core::autonomy::*;
use ai_agents_core::{AgentError, AgentStorage};
use serde_json::json;
use std::sync::Arc;

fn snapshot(agent: &str, run: &str) -> TaskRunSnapshot {
    let now = chrono::Utc::now();
    TaskRunSnapshot {
        schema_version: TASK_RUN_SCHEMA_VERSION,
        key: TaskRunKey {
            agent_id: agent.into(),
            run_id: run.into(),
        },
        actor_id: Some("actor".into()),
        revision: 0,
        status: TaskRunStatus::Paused,
        owner_token: None,
        cancel_requested: false,
        created_at: now,
        updated_at: now,
        payload: json!({"objective": "sensitive", "pending": {"secret": "never list"}}),
    }
}

fn assert_error<T: std::fmt::Debug>(
    result: ai_agents_core::Result<T>,
    expected: TaskRunStorageError,
) {
    assert!(
        matches!(result, Err(AgentError::TaskRunStorage(actual)) if actual == expected),
        "expected {expected:?}"
    );
}

async fn contract(storage: Arc<dyn AgentStorage>) {
    let original = snapshot("a", "run");
    let key = &original.key;
    storage.create_task_run(&original).await.unwrap();
    assert_error(
        storage.create_task_run(&original).await,
        TaskRunStorageError::AlreadyExists,
    );
    let other = snapshot("b", "run");
    storage.create_task_run(&other).await.unwrap();
    let filter = TaskRunFilter {
        agent_id: "a".into(),
        actor_id: Some("actor".into()),
        ..Default::default()
    };
    let summary = storage.list_task_runs(&filter).await.unwrap();
    assert_eq!(summary.len(), 1);
    let display = serde_json::to_string(&summary).unwrap();
    assert!(!display.contains("sensitive") && !display.contains("never list"));
    let left_claim = TaskRunMutation::Claim {
        expected_revision: 0,
        owner_token: "left".into(),
    };
    let right_claim = TaskRunMutation::Claim {
        expected_revision: 0,
        owner_token: "right".into(),
    };
    let (left, right) = tokio::join!(
        storage.mutate_task_run(key, &left_claim),
        storage.mutate_task_run(key, &right_claim)
    );
    assert_ne!(left.is_ok(), right.is_ok());
    let claimed = storage.load_task_run(key).await.unwrap().unwrap();
    let owner = claimed.owner_token.clone().unwrap();
    assert_eq!(claimed.revision, 1);
    assert_error(
        storage.delete_task_run(key, 1).await,
        TaskRunStorageError::Owned,
    );
    assert_error(
        storage
            .mutate_task_run(
                key,
                &TaskRunMutation::Checkpoint {
                    expected_revision: 1,
                    owner_token: "wrong".into(),
                    status: TaskRunStatus::Completed,
                    payload: json!({}),
                    release: true,
                },
            )
            .await,
        TaskRunStorageError::Conflict,
    );
    let cancelled = storage
        .mutate_task_run(
            key,
            &TaskRunMutation::RequestCancel {
                expected_revision: 1,
            },
        )
        .await
        .unwrap();
    assert_eq!(cancelled.status, TaskRunStatus::Running);
    assert_eq!(cancelled.owner_token.as_ref(), Some(&owner));
    assert_error(
        storage
            .mutate_task_run(
                key,
                &TaskRunMutation::Checkpoint {
                    expected_revision: 1,
                    owner_token: owner.clone(),
                    status: TaskRunStatus::Completed,
                    payload: json!({}),
                    release: true,
                },
            )
            .await,
        TaskRunStorageError::Conflict,
    );
    assert_error(
        storage
            .mutate_task_run(
                key,
                &TaskRunMutation::Checkpoint {
                    expected_revision: 2,
                    owner_token: owner.clone(),
                    status: TaskRunStatus::Completed,
                    payload: json!({}),
                    release: true,
                },
            )
            .await,
        TaskRunStorageError::InvalidCheckpoint,
    );
    let terminal = storage
        .mutate_task_run(
            key,
            &TaskRunMutation::Checkpoint {
                expected_revision: 2,
                owner_token: owner,
                status: TaskRunStatus::Cancelled,
                payload: json!({"exact": "retained"}),
                release: true,
            },
        )
        .await
        .unwrap();
    assert_eq!(terminal.revision, 3);
    assert_error(
        storage
            .mutate_task_run(
                key,
                &TaskRunMutation::Claim {
                    expected_revision: 3,
                    owner_token: "new".into(),
                },
            )
            .await,
        TaskRunStorageError::NotResumable,
    );
    assert_error(
        storage.delete_task_run(key, 2).await,
        TaskRunStorageError::Conflict,
    );
    storage.delete_task_run(key, 3).await.unwrap();
    assert!(storage.load_task_run(key).await.unwrap().is_none());
    assert_error(
        storage
            .mutate_task_run(
                key,
                &TaskRunMutation::RequestCancel {
                    expected_revision: 3,
                },
            )
            .await,
        TaskRunStorageError::NotFound,
    );
    assert_error(
        storage.create_task_run(&original).await,
        TaskRunStorageError::AlreadyExists,
    );
    assert!(storage.load_task_run(&other.key).await.unwrap().is_some());

    let abandoned = snapshot("a", "abandoned");
    storage.create_task_run(&abandoned).await.unwrap();
    let key = &abandoned.key;
    storage
        .mutate_task_run(
            key,
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "dead-owner".into(),
            },
        )
        .await
        .unwrap();
    assert_error(
        storage
            .mutate_task_run(
                key,
                &TaskRunMutation::Claim {
                    expected_revision: 1,
                    owner_token: "takeover".into(),
                },
            )
            .await,
        TaskRunStorageError::Owned,
    );
    let recovered = storage
        .mutate_task_run(
            key,
            &TaskRunMutation::Recover {
                expected_revision: 1,
                owner_token: "dead-owner".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(recovered.status, TaskRunStatus::RecoveryRequired);
    assert_eq!(recovered.payload, abandoned.payload);
    assert_error(
        storage
            .mutate_task_run(
                key,
                &TaskRunMutation::Claim {
                    expected_revision: 2,
                    owner_token: "new".into(),
                },
            )
            .await,
        TaskRunStorageError::NotResumable,
    );
    storage
        .mutate_task_run(
            key,
            &TaskRunMutation::ResolveRecovery {
                expected_revision: 2,
                payload: json!({"explicitly_reconciled": true}),
            },
        )
        .await
        .unwrap();
    let resumed = storage
        .mutate_task_run(
            key,
            &TaskRunMutation::Claim {
                expected_revision: 3,
                owner_token: "new".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(resumed.revision, 4);

    // Both writers may propose a terminal decision, but only one observed revision wins.
    let race = snapshot("a", "race");
    storage.create_task_run(&race).await.unwrap();
    storage
        .mutate_task_run(
            &race.key,
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let request = TaskRunMutation::RequestCancel {
        expected_revision: 1,
    };
    let finish = TaskRunMutation::Checkpoint {
        expected_revision: 1,
        owner_token: "owner".into(),
        status: TaskRunStatus::Completed,
        payload: json!({}),
        release: true,
    };
    let (cancel, finish) = tokio::join!(
        storage.mutate_task_run(&race.key, &request),
        storage.mutate_task_run(&race.key, &finish)
    );
    assert_ne!(cancel.is_ok(), finish.is_ok());

    let mut oversized = snapshot("a", "large");
    oversized.payload = json!({"required": "x".repeat(MAX_TASK_CHECKPOINT_BYTES)});
    assert_error(
        storage.create_task_run(&oversized).await,
        TaskRunStorageError::CheckpointTooLarge,
    );
    assert!(
        storage
            .load_task_run(&oversized.key)
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn task_memory_conditional_contract() {
    contract(Arc::new(crate::InMemoryTaskStorage::default())).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn task_sqlite_conditional_contract() {
    contract(Arc::new(crate::SqliteStorage::in_memory().await.unwrap())).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn task_sqlite_restart_preserves_claim_and_actor_deletion_is_atomic() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tasks.db");
    let storage = crate::SqliteStorage::new(path.to_str().unwrap())
        .await
        .unwrap();
    let original = snapshot("a", "run");
    storage.create_task_run(&original).await.unwrap();
    storage
        .mutate_task_run(
            &original.key,
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "dead".into(),
            },
        )
        .await
        .unwrap();
    storage.close().await;
    let storage = crate::SqliteStorage::new(path.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(
        storage
            .load_task_run(&original.key)
            .await
            .unwrap()
            .unwrap()
            .owner_token
            .as_deref(),
        Some("dead")
    );
    assert_error(
        storage.delete_actor_data("a", "actor").await,
        TaskRunStorageError::Owned,
    );
    storage
        .mutate_task_run(
            &original.key,
            &TaskRunMutation::Checkpoint {
                expected_revision: 1,
                owner_token: "dead".into(),
                status: TaskRunStatus::Paused,
                payload: original.payload.clone(),
                release: true,
            },
        )
        .await
        .unwrap();
    let sibling = snapshot("b", "run");
    storage.create_task_run(&sibling).await.unwrap();
    // A later failure must roll back task tombstones and deletion together with actor tables.
    sqlx::query("CREATE TRIGGER reject_actor_delete BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT, 'fixture'); END").execute(&storage.pool).await.unwrap();
    let session = ai_agents_core::AgentSnapshot::new("a".into());
    let metadata = ai_agents_core::SessionMetadata {
        actor_id: Some("actor".into()),
        ..Default::default()
    };
    storage
        .save_snapshot_with_metadata("session", &session, &metadata)
        .await
        .unwrap();
    assert!(storage.delete_actor_data("a", "actor").await.is_err());
    assert!(
        storage
            .load_task_run(&original.key)
            .await
            .unwrap()
            .is_some()
    );
    sqlx::query("DROP TRIGGER reject_actor_delete")
        .execute(&storage.pool)
        .await
        .unwrap();
    storage.delete_actor_data("a", "actor").await.unwrap();
    assert!(
        storage
            .load_task_run(&original.key)
            .await
            .unwrap()
            .is_none()
    );
    assert!(storage.load_task_run(&sibling.key).await.unwrap().is_some());
    assert_error(
        storage.create_task_run(&original).await,
        TaskRunStorageError::AlreadyExists,
    );
    assert_error(
        storage
            .mutate_task_run(
                &original.key,
                &TaskRunMutation::Checkpoint {
                    expected_revision: 2,
                    owner_token: "dead".into(),
                    status: TaskRunStatus::Paused,
                    payload: json!({}),
                    release: true,
                },
            )
            .await,
        TaskRunStorageError::NotFound,
    );
    storage.close().await;
}

#[tokio::test]
async fn task_unsupported_defaults_never_return_empty_success() {
    let storage = ai_agents_core::NoopStorage;
    let s = snapshot("a", "r");
    assert!(matches!(
        storage.create_task_run(&s).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage.load_task_run(&s.key).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage.list_task_runs(&TaskRunFilter::default()).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage
            .mutate_task_run(
                &s.key,
                &TaskRunMutation::RequestCancel {
                    expected_revision: 0
                }
            )
            .await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage.delete_task_run(&s.key, 0).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
}
