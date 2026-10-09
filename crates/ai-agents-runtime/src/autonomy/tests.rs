use super::*;
use ai_agents_core::{
    AgentError, AgentSnapshot, AgentStorage, ChatMessage, NativeCallBinding, NativeProviderState,
    NativeProviderTarget, Role, ToolCall, encode_native_tool_call_markers,
    encode_native_tool_result_marker,
};
use ai_agents_tools::{TodoItem, TodoStatus, TodoStore};
use serde_json::json;
use std::sync::Arc;

// Fixture configuration identity is host-owned and immutable, not a mutable profile name.
pub(super) fn checkpoint(run_id: &str) -> TaskRunSnapshot {
    let runtime = TaskRuntimeCheckpoint::between_turns(AgentSnapshot::new("agent".into())).unwrap();
    let profile = resolve_profile(
        &AutonomyConfig::default(),
        AutonomyScope::Task,
        None,
        None,
        None,
        None,
        &AutonomyHostCeilings::default(),
    )
    .unwrap();
    let payload =
        TaskCheckpointPayload::new("objective".into(), "config-v1".into(), &profile, runtime);
    let now = chrono::Utc::now();
    payload
        .bind(TaskRunSnapshot {
            schema_version: TASK_RUN_SCHEMA_VERSION,
            key: TaskRunKey {
                agent_id: "agent".into(),
                run_id: run_id.into(),
            },
            actor_id: None,
            revision: 0,
            status: TaskRunStatus::Paused,
            owner_token: None,
            cancel_requested: false,
            created_at: now,
            updated_at: now,
            payload: json!({}),
        })
        .unwrap()
}

// A completed item fixture uses the same public canonical type as the real todo tool.
fn item(id: &str, status: TodoStatus) -> TodoItem {
    TodoItem {
        id: id.into(),
        content: format!("work {id}"),
        active_form: None,
        status,
    }
}

// Each backend gets the same scoped schema, stale-writer, budget and crash recovery tests.
async fn scoped_contract(storage: Arc<dyn AgentStorage>) {
    let store =
        ScopedTaskRunStore::new(storage.clone(), "agent".into(), None, "config-v1".into()).unwrap();
    let initial = checkpoint("run");
    let mut initial_payload: TaskCheckpointPayload =
        serde_json::from_value(initial.payload.clone()).unwrap();
    initial_payload.controller_state = json!({"exact": "sensitive controller context"});
    initial_payload.clocks.expires_at = Some(initial.created_at + chrono::Duration::days(1));
    initial_payload.adapters.push(TaskAdapterCheckpoint {
        id: "validator".into(),
        adapter: "host.validator".into(),
        contract_version: 2,
        config: json!({"schema": "configured"}),
        state: json!({"exact": "opaque continuation"}),
    });
    initial_payload.progress.replans = 1;
    initial_payload.progress.high_water_marks = json!({"sources": 3});
    initial_payload.progress.history.push(TaskEvidenceRecord {
        run_id: "run".into(),
        sequence: 1,
        stage: Some("review".into()),
        attempt: Some("attempt".into()),
        origin: "host".into(),
        data: json!({"decision": "replan"}),
    });
    initial_payload.children.push(TaskChildCheckpoint {
        child_id: "child".into(),
        parent_run_id: "run".into(),
        config_identity: "child-config".into(),
        runtime: TaskRuntimeCheckpoint::between_turns(AgentSnapshot::new("child-runtime".into()))
            .unwrap(),
        pending: Some(TaskPendingRequest {
            id: "child-question".into(),
            issued_revision: 0,
            kind: TaskPendingKind::UserQuestion,
            reviewed_action: json!({"question": "exact pending child input"}),
        }),
        result: None,
    });
    let initial = initial_payload.bind(initial).unwrap();
    store.create(&initial).await.unwrap();
    assert_eq!(
        serde_json::to_value(store.load("run").await.unwrap().unwrap()).unwrap(),
        serde_json::to_value(&initial).unwrap()
    );
    let wrong_config = ScopedTaskRunStore::new(
        storage.clone(),
        "agent".into(),
        None,
        "different-config".into(),
    )
    .unwrap();
    assert!(wrong_config.load("run").await.is_err());
    let wrong_actor = ScopedTaskRunStore::new(
        storage,
        "agent".into(),
        Some("other".into()),
        "config-v1".into(),
    )
    .unwrap();
    assert!(wrong_actor.load("run").await.is_err());
    assert!(wrong_actor.list().await.unwrap().is_empty());
    let claimed = store
        .mutate(
            "run",
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let mut payload: TaskCheckpointPayload = serde_json::from_value(claimed.payload).unwrap();
    payload.counters.llm_attempts = 3;
    payload.clocks.active_millis = 42;
    payload.counters.charged_micro_usd = 7;
    payload.declared_write_targets.push("ticket:123".into());
    payload.reservations.push(TaskReservation {
        id: "external-effect".into(),
        state: TaskEffectState::Dispatched,
        reserved_micro_usd: 7,
        charged_micro_usd: 7,
        write_targets: vec!["ticket:123".into()],
        result: None,
    });
    let running = store
        .mutate(
            "run",
            &TaskRunMutation::Checkpoint {
                expected_revision: 1,
                owner_token: "owner".into(),
                status: TaskRunStatus::Running,
                payload: serde_json::to_value(&payload).unwrap(),
                release: false,
            },
        )
        .await
        .unwrap();
    assert_eq!(running.revision, 2);
    assert!(
        store
            .mutate(
                "run",
                &TaskRunMutation::Checkpoint {
                    expected_revision: 2,
                    owner_token: "owner".into(),
                    status: TaskRunStatus::Paused,
                    payload: serde_json::to_value(&payload).unwrap(),
                    release: true
                }
            )
            .await
            .is_err()
    );
    let recovered = store
        .mutate(
            "run",
            &TaskRunMutation::Recover {
                expected_revision: 2,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(recovered.status, TaskRunStatus::RecoveryRequired);
    assert!(
        store
            .mutate(
                "run",
                &TaskRunMutation::Claim {
                    expected_revision: 3,
                    owner_token: "another".into()
                }
            )
            .await
            .is_err()
    );
    payload.reservations[0].state = TaskEffectState::Completed;
    payload.reservations[0].result = Some(json!({"effect_performed": true, "exact": "한국어"}));
    let resolved = store
        .mutate(
            "run",
            &TaskRunMutation::ResolveRecovery {
                expected_revision: 3,
                payload: serde_json::to_value(&payload).unwrap(),
            },
        )
        .await
        .unwrap();
    let claimed = store
        .mutate(
            "run",
            &TaskRunMutation::Claim {
                expected_revision: resolved.revision,
                owner_token: "next".into(),
            },
        )
        .await
        .unwrap();
    let before = serde_json::to_value(&claimed).unwrap();
    let mut reset = payload.clone();
    reset.counters.llm_attempts = 0;
    assert!(
        store
            .mutate(
                "run",
                &TaskRunMutation::Checkpoint {
                    expected_revision: claimed.revision,
                    owner_token: "next".into(),
                    status: TaskRunStatus::Running,
                    payload: serde_json::to_value(reset).unwrap(),
                    release: false
                }
            )
            .await
            .is_err()
    );
    let mut lost_effect = payload.clone();
    lost_effect.reservations.clear();
    assert!(
        store
            .mutate(
                "run",
                &TaskRunMutation::Checkpoint {
                    expected_revision: claimed.revision,
                    owner_token: "next".into(),
                    status: TaskRunStatus::Running,
                    payload: serde_json::to_value(lost_effect).unwrap(),
                    release: false
                }
            )
            .await
            .is_err()
    );
    let mut changed = payload.clone();
    changed.settings.max_turns = Some(999);
    assert!(
        store
            .mutate(
                "run",
                &TaskRunMutation::Checkpoint {
                    expected_revision: claimed.revision,
                    owner_token: "next".into(),
                    status: TaskRunStatus::Running,
                    payload: serde_json::to_value(changed).unwrap(),
                    release: false
                }
            )
            .await
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(store.load("run").await.unwrap().unwrap()).unwrap(),
        before
    );
    payload.children[0].pending = None;
    payload.children[0].result = Some(json!({"child_completed": true}));
    payload.consumed_request_ids.push("child-question".into());
    store
        .mutate(
            "run",
            &TaskRunMutation::Checkpoint {
                expected_revision: claimed.revision,
                owner_token: "next".into(),
                status: TaskRunStatus::Completed,
                payload: serde_json::to_value(payload).unwrap(),
                release: true,
            },
        )
        .await
        .unwrap();
    let summary = serde_json::to_string(&store.list().await.unwrap()).unwrap();
    assert!(
        !summary.contains("objective")
            && !summary.contains("ticket:123")
            && !summary.contains("한국어")
    );
}

// In-memory persistence validates exact data with the same adapter as durable backends.
#[tokio::test]
async fn task_scoped_memory_validates_recovery_and_monotonic_accounting() {
    scoped_contract(Arc::new(ai_agents_storage::InMemoryTaskStorage::default())).await;
}

// SQLite runs identical runtime payload validation without invoking a model or tool.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn task_scoped_sqlite_validates_recovery_and_monotonic_accounting() {
    scoped_contract(Arc::new(
        ai_agents_storage::SqliteStorage::in_memory().await.unwrap(),
    ))
    .await;
}

// Lists never complete from a prior run, clearing, or cancellation alone.
#[test]
fn task_todos_share_one_authority_and_reject_stale_views() {
    let canonical = TodoStore::default();
    canonical.set(vec![item("old", TodoStatus::Completed)]);
    let key = TaskRunKey {
        agent_id: "agent".into(),
        run_id: "first".into(),
    };
    let adapter = RunTodoAdapter::begin(canonical.clone(), &key).unwrap();
    assert!(!adapter.all_done().unwrap());
    assert!(RunTodoAdapter::begin(canonical.clone(), &key).is_err());
    canonical.set(vec![item("work", TodoStatus::InProgress)]);
    assert!(!adapter.all_done().unwrap());
    canonical.update("work", None, None, Some(TodoStatus::Completed));
    assert!(adapter.all_done().unwrap());
    canonical.clear();
    assert!(!adapter.all_done().unwrap());
    canonical.set(vec![item("cancelled", TodoStatus::Cancelled)]);
    assert!(!adapter.all_done().unwrap());
    canonical.set(vec![item("done", TodoStatus::Completed)]);
    let saved = adapter.checkpoint().unwrap();
    adapter.release().unwrap();
    let new_key = TaskRunKey {
        agent_id: "agent".into(),
        run_id: "second".into(),
    };
    let next = RunTodoAdapter::begin(canonical.clone(), &new_key).unwrap();
    assert!(!next.all_done().unwrap());
    assert!(adapter.checkpoint().is_err());
    assert!(adapter.release().is_err());
    next.release().unwrap();
    assert!(RunTodoAdapter::restore(canonical.clone(), &new_key, saved.clone()).is_err());
    let restored = RunTodoAdapter::restore(canonical, &key, saved).unwrap();
    assert!(restored.all_done().unwrap());
    assert!(adapter.release().is_err());
    assert!(adapter.checkpoint().is_err());
    assert!(restored.all_done().unwrap());
}

// Signed provider content and completed results round-trip exactly, including opaque signatures.
#[test]
fn task_native_history_exact_round_trip_and_corruption_rejection() {
    let call = ToolCall {
        id: "call-1".into(),
        name: "echo".into(),
        arguments: json!({"input": "hello"}),
    };
    let state = NativeProviderState::new("exchange", "google", "generate_content", NativeProviderTarget::new("https://example.invalid", "model").unwrap(),
        json!({"role": "model", "parts": [{"functionCall": {"name": "echo", "args": {"input": "hello"}}, "thoughtSignature": "exact-signature"}]}),
        vec![NativeCallBinding::new("call-1", 0).unwrap()]).unwrap();
    let mut snapshot = AgentSnapshot::new("agent".into());
    snapshot.memory.messages = vec![
        ChatMessage::user("objective"),
        ChatMessage::assistant(
            encode_native_tool_call_markers(std::slice::from_ref(&call), Some(&state)).unwrap(),
        ),
        ChatMessage {
            role: Role::Tool,
            content: encode_native_tool_result_marker(&call, json!({"exact": "result"})).unwrap(),
            name: None,
            timestamp: None,
        },
    ];
    let runtime = TaskRuntimeCheckpoint::between_turns(snapshot).unwrap();
    let round_trip: TaskRuntimeCheckpoint =
        serde_json::from_value(serde_json::to_value(&runtime).unwrap()).unwrap();
    round_trip.validate().unwrap();
    assert_eq!(
        serde_json::to_value(&runtime).unwrap(),
        serde_json::to_value(&round_trip).unwrap()
    );
    let mut corrupt = runtime.clone();
    corrupt.snapshot.memory.messages.remove(1);
    assert!(corrupt.validate().is_err());
    let mut missing = runtime;
    missing.snapshot.memory.messages.pop();
    assert!(missing.validate().is_err());
    missing.continuation = TaskContinuation::Suspended {
        request_id: "request".into(),
        turn_id: "turn".into(),
        user_message_committed: true,
        finalized: false,
        skill: None,
        batch: None,
    };
    assert!(missing.validate().is_err());
    missing.continuation = TaskContinuation::Suspended {
        request_id: "request".into(),
        turn_id: "turn".into(),
        user_message_committed: true,
        finalized: false,
        skill: None,
        batch: Some(TaskToolBatchCursor {
            messages: missing.snapshot.memory.messages.clone(),
            call_ids: vec!["call-1".into()],
            next_call: 0,
            completed_results: vec![],
        }),
    };
    missing.validate().unwrap();
    missing
        .snapshot
        .memory
        .messages
        .push(ChatMessage::user("unrelated later turn"));
    assert!(missing.validate().is_err());
}

// Pending answers are single-use identities and cannot silently rewrite reviewed actions.
#[tokio::test]
async fn task_pending_requests_bound_to_revision_and_consumed_once() {
    let store = ScopedTaskRunStore::in_memory("agent".into(), None, "config-v1".into()).unwrap();
    let mut original = checkpoint("pending");
    let mut payload: TaskCheckpointPayload =
        serde_json::from_value(original.payload.clone()).unwrap();
    payload.pending = Some(TaskPendingRequest {
        id: "request-1".into(),
        issued_revision: 0,
        kind: TaskPendingKind::Approval,
        reviewed_action: json!({"operation": "write"}),
    });
    payload.runtime.continuation = TaskContinuation::Suspended {
        request_id: "request-1".into(),
        turn_id: "turn".into(),
        user_message_committed: true,
        finalized: false,
        skill: Some(TaskSkillCursor {
            skill_id: "skill".into(),
            next_step: 1,
            results: vec![json!({"exact": "required template result"})],
        }),
        batch: Some(TaskToolBatchCursor {
            messages: vec![],
            call_ids: vec!["done".into(), "awaiting".into()],
            next_call: 1,
            completed_results: vec![json!({"completed_effect": true})],
        }),
    };
    original = payload.clone().bind(original).unwrap();
    store.create(&original).await.unwrap();
    let claimed = store
        .mutate(
            "pending",
            &TaskRunMutation::Claim {
                expected_revision: 0,
                owner_token: "owner".into(),
            },
        )
        .await
        .unwrap();
    let exact: TaskCheckpointPayload = serde_json::from_value(claimed.payload).unwrap();
    assert_eq!(
        serde_json::to_value(&exact.runtime).unwrap(),
        serde_json::to_value(&payload.runtime).unwrap()
    );
    payload.pending = None;
    payload.runtime.continuation = TaskContinuation::BetweenTurns;
    let mutation = |p: &TaskCheckpointPayload| TaskRunMutation::Checkpoint {
        expected_revision: 1,
        owner_token: "owner".into(),
        status: TaskRunStatus::Running,
        payload: serde_json::to_value(p).unwrap(),
        release: false,
    };
    assert!(store.mutate("pending", &mutation(&payload)).await.is_err());
    payload.consumed_request_ids.push("request-1".into());
    store.mutate("pending", &mutation(&payload)).await.unwrap();
    payload.pending = exact.pending;
    assert!(
        payload
            .bind(store.load("pending").await.unwrap().unwrap())
            .is_err()
    );
}

// Namespaced snapshots must not falsely advertise task operations over a capable backend.
#[tokio::test]
async fn task_namespaced_storage_explicitly_unsupported() {
    let inner = Arc::new(ai_agents_storage::InMemoryTaskStorage::default());
    let storage = crate::spawner::NamespacedStorage::new(inner.clone(), "child");
    assert!(!storage.supports(ai_agents_core::StorageCapability::TaskRuns));
    let snapshot = checkpoint("run");
    assert!(matches!(
        storage.create_task_run(&snapshot).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage.load_task_run(&snapshot.key).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage.list_task_runs(&TaskRunFilter::default()).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage
            .mutate_task_run(
                &snapshot.key,
                &TaskRunMutation::RequestCancel {
                    expected_revision: 0
                }
            )
            .await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(matches!(
        storage.delete_task_run(&snapshot.key, 0).await,
        Err(AgentError::UnsupportedStorageCapability(_))
    ));
    assert!(inner.load_task_run(&snapshot.key).await.unwrap().is_none());
}

// Malformed and excessive recovery records fail without truncating the required state.
#[test]
fn task_checkpoint_versions_counts_and_scopes_fail_closed() {
    let s = checkpoint("run");
    let mut payload: TaskCheckpointPayload = serde_json::from_value(s.payload.clone()).unwrap();
    payload.version = 99;
    assert!(payload.clone().bind(s.clone()).is_err());
    payload.version = TASK_PAYLOAD_VERSION;
    payload.evidence.records = (0..=MAX_TASK_CHECKPOINT_RECORDS)
        .map(|i| TaskEvidenceRecord {
            run_id: "run".into(),
            sequence: i as u64,
            stage: None,
            attempt: None,
            origin: "host".into(),
            data: json!({}),
        })
        .collect();
    assert!(payload.clone().bind(s.clone()).is_err());
    payload.evidence.records.clear();
    payload.pending = Some(TaskPendingRequest {
        id: "future".into(),
        issued_revision: 1,
        kind: TaskPendingKind::UserQuestion,
        reviewed_action: json!({}),
    });
    assert!(payload.clone().bind(s.clone()).is_err());
    payload.pending = None;
    payload.runtime.snapshot.agent_id = "other".into();
    assert!(payload.bind(s).is_err());
}
