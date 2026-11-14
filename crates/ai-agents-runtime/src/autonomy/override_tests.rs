//! Invocation acceptance exercises real controller admission, retained controls and the shared journal.

use super::runner_tests::{RecordingProvider, config, fixture};
use super::*;
use crate::AgentBuilder;
use std::sync::{Arc, atomic::Ordering};

// The raw partial profile must be checked before merging can hide contradictory fields.
#[tokio::test]
async fn overrides_invalid_input_never_admits_work() {
    let provider = RecordingProvider::new(&[]);
    let (_, runner, store) = fixture(provider.clone(), 4);
    let invalid = AutonomyProfile {
        objective: Some("first".into()),
        objective_template: Some("second".into()),
        ..Default::default()
    };
    assert!(
        runner
            .run_with_overrides("work", None, &invalid)
            .await
            .is_err()
    );
    for yaml in [
        "objective: first\nobjective_template: second",
        "max_llm_calls: 0",
        "max_llm_call: 2",
        "<<: {enabled: true}",
        "progress: {stagnation: {typo: true}}",
    ] {
        assert!(parse_task_overrides(yaml).is_err(), "{yaml}");
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(store.list().await.unwrap().is_empty());
}

// Completion, allowance and output come from the actual invocation, while the next run retains its defaults.
#[tokio::test]
async fn overrides_blocking_stream_and_later_run_are_isolated() {
    let provider = RecordingProvider::new(&["accepted", "accepted", "done"]);
    let (_, runner, store) = fixture(provider.clone(), 4);
    let overlay =
        parse_task_overrides("completion: {response_contains: accepted}\nmax_llm_calls: 1")
            .unwrap();
    let first = runner
        .run_with_overrides("work", None, &overlay)
        .await
        .unwrap();
    let mut stream = runner
        .run_stream_with_overrides("work", None, &overlay)
        .unwrap();
    while stream.next_event().await.is_some() {}
    let second = stream.take_result().unwrap().unwrap();
    drop(stream);
    let third = runner.run("work", None).await.unwrap();
    for result in [&first, &second, &third] {
        assert_eq!(result.run.status, TaskRunStatus::Completed);
        assert_eq!(result.run.counters.llm_attempts, 1);
    }
    assert_eq!(first.run.limits.max_llm_calls, Some(1));
    assert_eq!(second.run.limits.max_llm_calls, Some(1));
    assert_eq!(third.run.limits.max_llm_calls, Some(4));
    assert_eq!(store.list().await.unwrap().len(), 3);
    for (result, expected) in [
        (&first, Some(&overlay)),
        (&second, Some(&overlay)),
        (&third, None),
    ] {
        let saved = store.load(&result.run.key.run_id).await.unwrap().unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        assert_eq!(payload.invocation_overrides.as_deref(), expected);
        assert_eq!(payload.config_identity, "prepared-v1");
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
}

// A partial override inherits completion, but independent host ceilings still bound actual calls.
#[tokio::test]
async fn overrides_host_ceiling_stops_real_work() {
    let provider = RecordingProvider::new(&["draft", "must not execute"]);
    let (agent, _, store) = fixture(provider.clone(), 4);
    let runner = AutonomyRunner::try_new(
        agent,
        config(4),
        AutonomyHostCeilings {
            max_llm_calls: Some(1),
            ..Default::default()
        },
        store,
        "prepared-v1".into(),
    )
    .unwrap();
    let overlay = parse_task_overrides("max_llm_calls: unlimited").unwrap();
    let result = runner
        .run_with_overrides("work", None, &overlay)
        .await
        .unwrap();
    assert_eq!(result.run.status, TaskRunStatus::LimitReached);
    assert_eq!(result.run.limits.max_llm_calls, Some(1));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Resuming retains the invocation gate and preserves explicitly revised limits instead of reapplying the initial allowance.
#[tokio::test]
async fn overrides_retained_question_preserves_control_and_prefix() {
    let provider = RecordingProvider::new(&["draft", "accepted"]);
    let (_, runner, store) = fixture(provider.clone(), 6);
    let overlay = parse_task_overrides("completion: {response_contains: accepted}\nmax_llm_calls: 3\npremature_finish: {action: ask_user, on_unavailable: stop}").unwrap();
    let first = runner
        .run_with_overrides("work", None, &overlay)
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Paused);
    let request = first.run.pending.as_ref().unwrap().id.clone();
    let mut limits = first.run.limits.clone();
    limits.max_llm_calls = Some(4);
    let revised = runner
        .revise_limits_paused(
            &first.run.key.run_id,
            first.run.revision,
            limits,
            "host revision",
        )
        .await
        .unwrap();
    assert_eq!(revised.run.pending.as_ref().unwrap().id, request);
    assert!(
        runner
            .resume(
                &first.run.key.run_id,
                first.run.revision,
                TaskResumeInput::Continue
            )
            .await
            .is_err()
    );
    let done = runner
        .resume(
            &revised.run.key.run_id,
            revised.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: request,
                answer: serde_json::json!("continue"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.key, first.run.key);
    assert_eq!(done.run.limits.max_llm_calls, Some(4));
    assert_eq!(done.run.counters.llm_attempts, 2);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let saved = store.load(&done.run.key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
    assert_eq!(payload.invocation_overrides.as_deref(), Some(&overlay));
}

// The same runner rejects weakened inherited reviews and retains the actual stage approval before completion.
#[tokio::test]
async fn overrides_required_review_cannot_be_waived() {
    let provider = RecordingProvider::new(&["done"]);
    let (agent, _, store) = fixture(provider.clone(), 4);
    let mut cfg = config(4);
    cfg.defaults.lifecycle = Some(serde_yaml::from_str("[{id: work}]").unwrap());
    cfg.defaults.supervision = Some(
        serde_yaml::from_str("stage_changes: require_approval\nplan_changes: require_approval")
            .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        agent,
        cfg,
        Default::default(),
        store.clone(),
        "prepared-v1".into(),
    )
    .unwrap();
    let plan_denied = parse_task_overrides("supervision: {plan_changes: inherit}").unwrap();
    assert!(
        runner
            .run_with_overrides("work", None, &plan_denied)
            .await
            .is_err()
    );
    let denied = parse_task_overrides("supervision: {stage_changes: inherit}").unwrap();
    assert!(
        runner
            .run_with_overrides("work", None, &denied)
            .await
            .is_err()
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(store.list().await.unwrap().is_empty());
    let paused = runner
        .run_with_overrides("work", None, &AutonomyProfile::default())
        .await
        .unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// The configured public host also admits the additive operation without needing an earlier retained run.
#[tokio::test]
async fn overrides_runtime_convenience_and_operation_share_host() {
    let provider = RecordingProvider::new(&["accepted", "accepted"]);
    let agent = AgentBuilder::from_yaml("name: OverlayHost\nsystem_prompt: test\nautonomy:\n  enabled: true\n  completion: {response_contains: done}\n").unwrap().llm(provider.clone()).build_task_host().await.unwrap();
    let overlay =
        parse_task_overrides("completion: {response_contains: accepted}\nmax_llm_calls: 1")
            .unwrap();
    let first = agent
        .run_task_with_overrides("work", None, &overlay)
        .await
        .unwrap();
    let second = agent
        .task_operation(TaskStreamOperation::RunWithOverrides {
            objective: "work".into(),
            profile: None,
            overrides: Box::new(overlay),
        })
        .await
        .unwrap();
    assert_eq!(first.run.status, TaskRunStatus::Completed);
    assert_eq!(second.run.status, TaskRunStatus::Completed);
    assert_ne!(first.run.key, second.run.key);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

// The invocation overrides named and selected-skill settings without changing the admitted script or replaying steps.
#[tokio::test]
async fn overrides_skill_layers_and_stream_use_actual_script() {
    for streaming in [false, true] {
        let provider = RecordingProvider::new(&["source", "accepted"]);
        let agent = Arc::new(AgentBuilder::from_yaml("name: SkillOverlay\nsystem_prompt: test\nskills:\n  - id: report\n    description: report\n    trigger: report\n    autonomy: {max_llm_calls: 1, completion: {response_contains: scoped}}\n    steps:\n      - prompt: gather\n      - prompt: synthesize {{ steps[0].result }}\n").unwrap().llm(provider.clone()).auto_configure_features().unwrap().build().unwrap());
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(agent.info().id, None, "skill.v1".into()).unwrap(),
        );
        let mut cfg = config(8);
        cfg.profiles.insert(
            "selected".into(),
            parse_task_overrides("max_llm_calls: 3\ncompletion: {response_contains: named}")
                .unwrap(),
        );
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            cfg,
            Default::default(),
            store,
            "skill.v1".into(),
        )
        .unwrap();
        let overlay =
            parse_task_overrides("max_llm_calls: 2\ncompletion: {response_contains: accepted}")
                .unwrap();
        let result = if streaming {
            let mut stream = agent
                .task_operation_stream(
                    &runner,
                    TaskStreamOperation::SkillWithOverrides {
                        objective: "report".into(),
                        skill_id: "report".into(),
                        profile: Some("selected".into()),
                        overrides: Box::new(overlay),
                    },
                    Default::default(),
                )
                .unwrap();
            while stream.next_event().await.is_some() {}
            stream.take_result().unwrap().unwrap()
        } else {
            runner
                .run_skill_with_overrides("report", "report", Some("selected"), &overlay)
                .await
                .unwrap()
        };
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(result.run.limits.max_llm_calls, Some(2));
        assert_eq!(result.run.counters.llm_attempts, 2);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert!(
            provider.captured_messages()[1]
                .iter()
                .any(|message| message.content.contains("source"))
        );
    }
}

// Legacy versions remain readable only without the new invocation field; payload versioning cannot inject automatic overrides.
#[test]
fn overrides_checkpoint_legacy_and_automatic_reject_injected_input() {
    let original = super::tests::checkpoint("overlay-legacy");
    for version in [1, 2, TASK_PAYLOAD_VERSION] {
        let mut snapshot = original.clone();
        snapshot.payload["version"] = serde_json::json!(version);
        let payload: TaskCheckpointPayload =
            serde_json::from_value(snapshot.payload.clone()).unwrap();
        payload
            .validate(&snapshot, &payload.config_identity)
            .unwrap();
        snapshot.payload["invocation_overrides"] = serde_json::json!({"max_llm_calls": 2});
        let payload: TaskCheckpointPayload =
            serde_json::from_value(snapshot.payload.clone()).unwrap();
        assert_eq!(
            payload
                .validate(&snapshot, &payload.config_identity)
                .is_ok(),
            version == TASK_PAYLOAD_VERSION
        );
    }
    let mut snapshot = original;
    snapshot.payload["version"] = serde_json::json!(TASK_PAYLOAD_VERSION);
    snapshot.payload["invocation_overrides"] = serde_json::json!({});
    let mut payload: TaskCheckpointPayload =
        serde_json::from_value(snapshot.payload.clone()).unwrap();
    payload.adapters.push(TaskAdapterCheckpoint {
        id: "runtime.activation".into(),
        adapter: "runtime.activation".into(),
        contract_version: 1,
        config: serde_json::json!({}),
        state: serde_json::json!({}),
    });
    assert!(
        payload
            .validate(&snapshot, &payload.config_identity)
            .is_err()
    );
}

// Invocation-selected checks execute through the same validator and tool approval path; a name alone grants no binding.
#[tokio::test]
async fn overrides_validation_replacement_keeps_actual_binding_and_hitl() {
    let f = super::controller_tests::fixture("plain");
    let mut overlay = AutonomyProfile {
        validation: f.config.defaults.validation.clone(),
        completion: Some(CompletionGate::ValidatorPassed("replacement".into())),
        ..Default::default()
    };
    let check = &mut overlay
        .validation
        .as_mut()
        .unwrap()
        .checks
        .as_mut()
        .unwrap()[0];
    check.id = "replacement".into();
    check.adapter = "missing".into();
    assert!(
        f.runner
            .run_with_overrides("work", None, &overlay)
            .await
            .is_err()
    );
    assert_eq!(f.provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    overlay
        .validation
        .as_mut()
        .unwrap()
        .checks
        .as_mut()
        .unwrap()[0]
        .adapter = "host.observation".into();
    let paused = f
        .runner
        .run_with_overrides("work", None, &overlay)
        .await
        .unwrap();
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    let done = f
        .runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: paused.run.pending.as_ref().unwrap().id.clone(),
                result: ai_agents_hitl::ApprovalResult::Approved,
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(f.provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert!(f.callbacks.load(Ordering::SeqCst) >= 2);
}

// A lost terminal receipt protects the original owner even when its effects and invocation settings reached storage.
#[tokio::test]
async fn overrides_lost_ack_and_dropped_stream_keep_custody() {
    let provider = RecordingProvider::new(&["accepted"]);
    let (agent, _, store) = fixture(provider.clone(), 4);
    let fault = super::continuation_test_support::ReceiptStore::new(
        store.clone(),
        "lost",
        TaskRunStatus::Completed,
    );
    fault.armed.store(true, Ordering::SeqCst);
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config(4),
        Default::default(),
        fault.clone(),
        "prepared-v1".into(),
    )
    .unwrap();
    let overlay =
        parse_task_overrides("completion: {response_contains: accepted}\nmax_llm_calls: 1")
            .unwrap();
    assert!(
        runner
            .run_with_overrides("work", None, &overlay)
            .await
            .is_err()
    );
    fault.assert_hit();
    assert!(
        agent
            .set_context("blocked", serde_json::json!(true))
            .is_err()
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let rows = store.list().await.unwrap();
    assert_eq!(rows.len(), 1);
    let snapshot = store.load(&rows[0].key.run_id).await.unwrap().unwrap();
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
    assert_eq!(payload.invocation_overrides.as_deref(), Some(&overlay));

    let (agent, runner, store) = fixture(RecordingProvider::gated(), 4);
    let mut stream = runner
        .run_stream_with_overrides("work", None, &overlay)
        .unwrap();
    assert_eq!(
        stream.next_event().await.unwrap().kind,
        TaskRunEventKind::TaskStarted
    );
    drop(stream);
    assert!(agent.captured_task_owner().unwrap().is_abandoned());
    assert_ne!(
        store.list().await.unwrap()[0].status,
        TaskRunStatus::Completed
    );
}
