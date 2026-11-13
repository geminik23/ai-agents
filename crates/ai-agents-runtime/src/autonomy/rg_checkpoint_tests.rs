//! Exact serialized boundaries cover every reconstruction phase before a store can claim the paused owner.
use super::*;
use chrono::Timelike;

// Maximum timestamp precision and a real UUID-sized owner match the production preclaim envelope rather than just the inner payload.
fn proposed(source: &TaskRunSnapshot, payload: &TaskCheckpointPayload) -> TaskRunSnapshot {
    let mut snapshot = source.clone();
    snapshot.revision = source.revision + 3;
    snapshot.status = TaskRunStatus::Running;
    snapshot.owner_token = Some("00000000-0000-0000-0000-000000000000".into());
    snapshot.updated_at = snapshot.updated_at.with_nanosecond(999_999_999).unwrap();
    snapshot.payload = serde_json::to_value(payload).unwrap();
    snapshot
}

// The old ready image fits while its digest-bearing running journal does not; refusal must leave exact backend pause data untouched.
#[tokio::test]
async fn rg_all_reconstruction_phases_preflight_intermediate_envelope_before_claim() {
    for sqlite in crate::autonomy::continuation_test_support::backends() {
        for kind in ["capture", "restore", "review"] {
            let source = crate::autonomy::tests::checkpoint("rg-boundary");
            let mut ready: TaskCheckpointPayload =
                serde_json::from_value(source.payload.clone()).unwrap();
            let budget = std::time::Duration::from_secs(1);
            mark_work(&mut ready, kind, "paused", budget, source.updated_at).unwrap();
            ready
                .runtime
                .snapshot
                .context
                .insert("padding".into(), json!("한글\"\\"));
            let size = serde_json::to_vec(&proposed(&source, &ready))
                .unwrap()
                .len();
            let padding = format!(
                "한글\"\\{}",
                "x".repeat(MAX_TASK_CHECKPOINT_BYTES - size - 16)
            );
            ready
                .runtime
                .snapshot
                .context
                .insert("padding".into(), json!(padding));
            let original = ready.clone().bind(source).unwrap();
            let old_preflight = proposed(&original, &ready);
            old_preflight.validate().unwrap();
            assert_eq!(
                serde_json::to_vec(&old_preflight).unwrap().len(),
                MAX_TASK_CHECKPOINT_BYTES - 16
            );
            let running = work_payload(&ready, kind, budget, original.updated_at).unwrap();
            assert!(
                serde_json::to_vec(&proposed(&original, &running))
                    .unwrap()
                    .len()
                    > MAX_TASK_CHECKPOINT_BYTES
            );
            let store = crate::autonomy::continuation_test_support::store(
                "agent".into(),
                "config-v1",
                *sqlite,
            )
            .await;
            store.create(&original).await.unwrap();
            assert!(matches!(
                preflight_work(&original, &ready, kind, budget, original.updated_at),
                Err(AgentError::TaskRunStorage(
                    TaskRunStorageError::CheckpointTooLarge
                ))
            ));
            let after = store.load(&original.key.run_id).await.unwrap().unwrap();
            assert_eq!(
                serde_json::to_value(after).unwrap(),
                serde_json::to_value(original).unwrap()
            );
        }
    }
}

// Inclusive limits use serialized UTF-8 bytes, including escaping, rather than character or inner-payload length.
#[test]
fn rg_checkpoint_envelope_exact_cap_is_inclusive() {
    let mut source = crate::autonomy::tests::checkpoint("rg-inclusive");
    source.payload["runtime"]["snapshot"]["context"]["padding"] = json!("한\"\\");
    let size = serde_json::to_vec(&source).unwrap().len();
    source.payload["runtime"]["snapshot"]["context"]["padding"] = json!(format!(
        "한\"\\{}",
        "x".repeat(MAX_TASK_CHECKPOINT_BYTES - size)
    ));
    assert_eq!(
        serde_json::to_vec(&source).unwrap().len(),
        MAX_TASK_CHECKPOINT_BYTES
    );
    source.validate().unwrap();
    source.payload["runtime"]["snapshot"]["context"]["padding"]
        .as_str()
        .unwrap();
    let mut value = source.payload["runtime"]["snapshot"]["context"]["padding"]
        .as_str()
        .unwrap()
        .to_string();
    value.push('x');
    source.payload["runtime"]["snapshot"]["context"]["padding"] = json!(value);
    assert!(source.validate().is_err());
}
