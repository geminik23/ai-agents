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

#[derive(Debug)]
struct FinalAdmissionGuard {
    calls: std::sync::atomic::AtomicUsize,
    reject_at: std::sync::atomic::AtomicUsize,
}

impl TaskFinalAdmission for FinalAdmissionGuard {
    fn validate(&self) -> ai_agents_core::Result<()> {
        use std::sync::atomic::Ordering;
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call >= self.reject_at.load(Ordering::SeqCst) {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        Ok(())
    }
}

// Time-free publication still requires current live proof, the original owner, an exact revision and no cancellation.
async fn optional_final_deadline_contract(storage: Arc<dyn AgentStorage>, checks: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    for status in [TaskRunStatus::Completed, TaskRunStatus::Paused] {
        let original = snapshot("optional-final", &format!("{status:?}"));
        storage.create_task_run(&original).await.unwrap();
        let running = storage
            .mutate_task_run(
                &original.key,
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
            )
            .await
            .unwrap();
        let guard = Arc::new(FinalAdmissionGuard {
            calls: AtomicUsize::new(0),
            reject_at: AtomicUsize::new(0),
        });
        let mut mutation = TaskRunMutation::FinalCheckpoint {
            expected_revision: running.revision,
            owner_token: "owner".into(),
            status,
            payload: json!({"exact": "successor"}),
            release: true,
            deadline: None,
            expires_at: None,
            evidence: Some(guard.clone()),
        };
        assert_error(
            storage.mutate_task_run(&original.key, &mutation).await,
            TaskRunStorageError::InvalidCheckpoint,
        );
        assert_eq!(guard.calls.load(Ordering::SeqCst), 1);
        let retained = storage.load_task_run(&original.key).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&retained).unwrap(),
            serde_json::to_value(&running).unwrap()
        );

        guard.reject_at.store(usize::MAX, Ordering::SeqCst);
        // Expiry from either clock remains binding even when the other clock has no limit.
        if let TaskRunMutation::FinalCheckpoint { deadline, .. } = &mut mutation {
            *deadline = Some(Instant::now() - Duration::from_secs(1));
        }
        assert_error(
            storage.mutate_task_run(&original.key, &mutation).await,
            TaskRunStorageError::AdmissionExpired,
        );
        if let TaskRunMutation::FinalCheckpoint {
            deadline,
            expires_at,
            ..
        } = &mut mutation
        {
            *deadline = None;
            *expires_at = Some(chrono::Utc::now() - chrono::Duration::seconds(1));
        }
        assert_error(
            storage.mutate_task_run(&original.key, &mutation).await,
            TaskRunStorageError::AdmissionExpired,
        );
        if let TaskRunMutation::FinalCheckpoint {
            expires_at,
            expected_revision,
            ..
        } = &mut mutation
        {
            *expires_at = None;
            *expected_revision = running.revision - 1;
        }
        assert_error(
            storage.mutate_task_run(&original.key, &mutation).await,
            TaskRunStorageError::Conflict,
        );
        if let TaskRunMutation::FinalCheckpoint {
            expected_revision,
            owner_token,
            ..
        } = &mut mutation
        {
            *expected_revision = running.revision;
            *owner_token = "wrong".into();
        }
        assert_error(
            storage.mutate_task_run(&original.key, &mutation).await,
            TaskRunStorageError::Conflict,
        );
        if let TaskRunMutation::FinalCheckpoint { owner_token, .. } = &mut mutation {
            *owner_token = "owner".into();
        }
        guard.calls.store(0, Ordering::SeqCst);
        let saved = storage
            .mutate_task_run(&original.key, &mutation)
            .await
            .unwrap();
        assert_eq!(guard.calls.load(Ordering::SeqCst), checks);
        assert_eq!(saved.revision, running.revision + 1);
        assert_eq!(saved.status, status);
        assert!(saved.owner_token.is_none());
        assert_eq!(saved.payload, json!({"exact": "successor"}));
    }

    let original = snapshot("optional-final", "cancelled");
    storage.create_task_run(&original).await.unwrap();
    let running = storage
        .mutate_task_run(
            &original.key,
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let cancelled = storage
        .mutate_task_run(
            &original.key,
            &TaskRunMutation::RequestCancel {
                expected_revision: running.revision,
            },
        )
        .await
        .unwrap();
    for status in [TaskRunStatus::Completed, TaskRunStatus::Paused] {
        assert_error(
            storage
                .mutate_task_run(
                    &original.key,
                    &TaskRunMutation::FinalCheckpoint {
                        expected_revision: cancelled.revision,
                        owner_token: "owner".into(),
                        status,
                        payload: json!({"must_not_commit": true}),
                        release: true,
                        deadline: None,
                        expires_at: None,
                        evidence: None,
                    },
                )
                .await,
            TaskRunStorageError::InvalidCheckpoint,
        );
    }
    let retained = storage.load_task_run(&original.key).await.unwrap().unwrap();
    assert_eq!(
        serde_json::to_value(&retained).unwrap(),
        serde_json::to_value(&cancelled).unwrap()
    );

    // SQLite's second proof check must roll back the already staged SQL update, not publish a stale successor.
    if checks > 1 {
        let guard = Arc::new(FinalAdmissionGuard {
            calls: AtomicUsize::new(0),
            reject_at: AtomicUsize::new(checks - 1),
        });
        let original = snapshot("optional-final", "late-proof");
        storage.create_task_run(&original).await.unwrap();
        let running = storage
            .mutate_task_run(
                &original.key,
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
            )
            .await
            .unwrap();
        assert_error(
            storage
                .mutate_task_run(
                    &original.key,
                    &TaskRunMutation::FinalCheckpoint {
                        expected_revision: running.revision,
                        owner_token: "owner".into(),
                        status: TaskRunStatus::Completed,
                        payload: json!({"must_not_commit": true}),
                        release: true,
                        deadline: None,
                        expires_at: None,
                        evidence: Some(guard.clone()),
                    },
                )
                .await,
            TaskRunStorageError::InvalidCheckpoint,
        );
        assert_eq!(guard.calls.load(Ordering::SeqCst), checks);
        let retained = storage.load_task_run(&original.key).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&retained).unwrap(),
            serde_json::to_value(&running).unwrap()
        );
    }
}

#[tokio::test]
async fn task_memory_optional_final_deadline_preserves_admission() {
    optional_final_deadline_contract(Arc::new(crate::InMemoryTaskStorage::default()), 1).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn task_sqlite_optional_final_deadline_preserves_admission() {
    optional_final_deadline_contract(
        Arc::new(crate::SqliteStorage::in_memory().await.unwrap()),
        2,
    )
    .await;
}

#[derive(Debug)]
struct ExpiringFinalGuard {
    expiry: chrono::DateTime<chrono::Utc>,
    calls: std::sync::atomic::AtomicUsize,
    expire_on: usize,
}

impl TaskFinalAdmission for ExpiringFinalGuard {
    // Cross the actual UTC cut inside the selected backend guard, rather than asserting a machine-speed threshold.
    fn validate(&self) -> ai_agents_core::Result<()> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        if call == self.expire_on {
            let guard = std::time::Instant::now();
            while chrono::Utc::now() < self.expiry {
                assert!(
                    guard.elapsed() < std::time::Duration::from_secs(5),
                    "fixture clock failed to reach expiry"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        Ok(())
    }
}

// A UTC-only cut must be rechecked after the last live proof callback, including SQLite's second precommit check.
async fn final_utc_only_guard_expiry_contract(storage: Arc<dyn AgentStorage>, expire_on: usize) {
    let original = snapshot("utc-only-final", "expires-in-guard");
    storage.create_task_run(&original).await.unwrap();
    let running = storage
        .mutate_task_run(
            &original.key,
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let expiry = chrono::Utc::now() + chrono::Duration::seconds(1);
    let guard = Arc::new(ExpiringFinalGuard {
        expiry,
        calls: std::sync::atomic::AtomicUsize::new(0),
        expire_on,
    });
    let result = storage
        .mutate_task_run(
            &original.key,
            &TaskRunMutation::FinalCheckpoint {
                expected_revision: running.revision,
                owner_token: "owner".into(),
                status: TaskRunStatus::Completed,
                payload: json!({"must_not_commit": true}),
                release: true,
                deadline: None,
                expires_at: Some(expiry),
                evidence: Some(guard.clone()),
            },
        )
        .await;
    assert_eq!(
        guard.calls.load(std::sync::atomic::Ordering::SeqCst),
        expire_on
    );
    assert_error(result, TaskRunStorageError::AdmissionExpired);
    let retained = storage.load_task_run(&original.key).await.unwrap().unwrap();
    assert_eq!(
        serde_json::to_value(&retained).unwrap(),
        serde_json::to_value(&running).unwrap()
    );
}

#[tokio::test]
async fn task_memory_final_utc_only_expiry_during_guard_cannot_commit() {
    final_utc_only_guard_expiry_contract(Arc::new(crate::InMemoryTaskStorage::default()), 1).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn task_sqlite_final_utc_only_expiry_during_guard_cannot_commit() {
    final_utc_only_guard_expiry_contract(
        Arc::new(crate::SqliteStorage::in_memory().await.unwrap()),
        2,
    )
    .await;
}

// A controlled backend clock covers advances before and during proof admission without sleeping or changing the process clock.
#[test]
fn final_admission_resamples_the_backend_clock_without_rebinding_its_domain() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for scenario in ["valid", "before_guard", "during_guard", "clock_error"] {
        let original = snapshot("controlled-clock", scenario);
        let now = original.updated_at;
        let running = original
            .transition(
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
                now,
            )
            .unwrap();
        let expiry = now + chrono::Duration::seconds(60);
        let guard = Arc::new(FinalAdmissionGuard {
            calls: AtomicUsize::new(0),
            reject_at: AtomicUsize::new(usize::MAX),
        });
        let mutation = TaskRunMutation::FinalCheckpoint {
            expected_revision: running.revision,
            owner_token: "owner".into(),
            status: TaskRunStatus::Completed,
            payload: json!({"exact": "clock-bound"}),
            release: true,
            deadline: None,
            expires_at: Some(expiry),
            evidence: Some(guard.clone()),
        };
        let samples = std::cell::Cell::new(0);
        let result = running.transition_with_clock(&mutation, || {
            let sample = samples.get();
            samples.set(sample + 1);
            if scenario == "clock_error" && sample == 2 {
                return Err(TaskRunStorageError::InvalidCheckpoint.into());
            }
            if (scenario == "before_guard" && sample == 1)
                || (scenario == "during_guard" && sample == 2)
            {
                Ok(expiry)
            } else {
                Ok(now)
            }
        });
        if scenario == "valid" {
            let next = result.unwrap();
            assert_eq!(next.status, TaskRunStatus::Completed);
            assert_eq!(next.updated_at, now);
        } else {
            assert_error(
                result,
                if scenario == "clock_error" {
                    TaskRunStorageError::InvalidCheckpoint
                } else {
                    TaskRunStorageError::AdmissionExpired
                },
            );
        }
        assert_eq!(
            guard.calls.load(Ordering::SeqCst),
            usize::from(scenario != "before_guard")
        );
        assert_eq!(
            samples.get(),
            if scenario == "before_guard" { 2 } else { 3 }
        );
    }
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
