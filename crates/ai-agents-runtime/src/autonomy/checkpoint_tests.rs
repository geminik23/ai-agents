use super::tests::checkpoint;
use super::*;
use serde_json::json;

// Suspended fixture includes already completed batch and skill work, not another executor.
fn suspended() -> TaskCheckpointPayload {
    let s = checkpoint("run");
    let mut payload: TaskCheckpointPayload = serde_json::from_value(s.payload).unwrap();
    payload.pending = Some(TaskPendingRequest {
        id: "request".into(),
        issued_revision: 0,
        kind: TaskPendingKind::Approval,
        reviewed_action: json!({"write": "reviewed"}),
    });
    payload.runtime.continuation = TaskContinuation::Suspended {
        request_id: "request".into(),
        turn_id: "turn".into(),
        user_message_committed: true,
        finalized: false,
        skill: Some(TaskSkillCursor {
            skill_id: "skill".into(),
            next_step: 1,
            results: vec![json!({"exact": "skill result"})],
        }),
        batch: Some(TaskToolBatchCursor {
            messages: vec![],
            call_ids: vec!["completed".into(), "pending".into()],
            next_call: 1,
            completed_results: vec![json!({"effect": "completed"})],
        }),
    };
    payload
}

// Prefix checks reject both rewinding a cursor and rewriting results behind its unchanged position.
#[test]
fn task_suspended_successors_cannot_rewind_completed_work() {
    let old = suspended();
    let mut next = old.clone();
    if let TaskContinuation::Suspended {
        batch: Some(batch), ..
    } = &mut next.runtime.continuation
    {
        batch.next_call = 0;
        batch.completed_results.clear();
    }
    next.runtime.validate().unwrap();
    assert!(next.validate_successor(&old, 1, false).is_err());
    let mut next = old.clone();
    if let TaskContinuation::Suspended {
        skill: Some(skill), ..
    } = &mut next.runtime.continuation
    {
        skill.results[0] = json!("rewritten");
    }
    assert!(next.validate_successor(&old, 1, false).is_err());
    let mut bypass = old.clone();
    bypass.runtime.continuation = TaskContinuation::BetweenTurns;
    assert!(bypass.validate_successor(&old, 1, false).is_err());
    let mut next = old.clone();
    next.pending = None;
    next.runtime.continuation = TaskContinuation::BetweenTurns;
    assert!(next.validate_successor(&old, 1, true).is_err());
    next.consumed_request_ids.push("request".into());
    next.validate_successor(&old, 1, true).unwrap();
    next.pending = old.pending.clone();
    let mut envelope = checkpoint("run");
    envelope.revision = 1;
    assert!(next.bind(envelope).is_err());
}

// Duplicated native continuation data must agree on exact replay state, arguments and completed outputs.
#[test]
fn task_native_batch_representations_and_successors_agree_exactly() {
    use ai_agents_core::{
        AgentSnapshot, ChatMessage, NativeCallBinding, NativeProviderState, NativeProviderTarget,
        ToolCall, encode_native_tool_call_markers, encode_native_tool_result_marker,
    };
    let calls = vec![
        ToolCall {
            id: "completed".into(),
            name: "echo".into(),
            arguments: json!({}),
        },
        ToolCall {
            id: "pending".into(),
            name: "echo".into(),
            arguments: json!({}),
        },
    ];
    let state = NativeProviderState::new("exchange", "google", "generate_content", NativeProviderTarget::new("https://example.invalid", "model").unwrap(),
        json!({"parts": [{"functionCall": {"name": "echo", "args": {}}, "thoughtSignature": "original"}, {"functionCall": {"name": "echo", "args": {}}}]}),
        vec![NativeCallBinding::new("completed", 0).unwrap(), NativeCallBinding::new("pending", 1).unwrap()]).unwrap();
    let messages = vec![
        ChatMessage::user("objective"),
        ChatMessage::assistant(encode_native_tool_call_markers(&calls, Some(&state)).unwrap()),
        ChatMessage::function(
            "echo",
            encode_native_tool_result_marker(&calls[0], json!({"output": "A"})).unwrap(),
        ),
    ];
    let mut old = suspended();
    let mut snapshot = AgentSnapshot::new("agent".into());
    snapshot.memory.messages = messages.clone();
    old.runtime.snapshot = snapshot;
    old.runtime.native_exchanges = vec![TaskNativeExchange {
        exchange_id: "exchange".into(),
        call_ids: vec!["completed".into(), "pending".into()],
    }];
    if let TaskContinuation::Suspended {
        batch: Some(batch), ..
    } = &mut old.runtime.continuation
    {
        batch.messages = messages;
        batch.completed_results = vec![json!({"output": "A"})];
    }
    old.runtime.validate().unwrap();
    let mut tool_role = old.clone();
    tool_role.runtime.snapshot.memory.messages[2].role = ai_agents_core::Role::Tool;
    if let TaskContinuation::Suspended {
        batch: Some(batch), ..
    } = &mut tool_role.runtime.continuation
    {
        batch.messages[2].role = ai_agents_core::Role::Tool;
    }
    tool_role.runtime.validate().unwrap();
    let mut corrupt = old.clone();
    if let TaskContinuation::Suspended {
        batch: Some(batch), ..
    } = &mut corrupt.runtime.continuation
    {
        batch.messages[2].content =
            encode_native_tool_result_marker(&calls[0], json!({"output": "B"})).unwrap();
    }
    assert!(corrupt.runtime.validate().is_err());
    corrupt.runtime.snapshot.memory.messages[2].content =
        encode_native_tool_result_marker(&calls[0], json!({"output": "B"})).unwrap();
    assert!(corrupt.runtime.validate().is_err());
    let mut changed = old.clone();
    let replacement = NativeProviderState::new("exchange", "google", "generate_content", NativeProviderTarget::new("https://example.invalid", "model").unwrap(),
        json!({"parts": [{"functionCall": {"name": "echo", "args": {}}, "thoughtSignature": "changed"}, {"functionCall": {"name": "echo", "args": {}}}]}),
        vec![NativeCallBinding::new("completed", 0).unwrap(), NativeCallBinding::new("pending", 1).unwrap()]).unwrap();
    let marker = encode_native_tool_call_markers(&calls, Some(&replacement)).unwrap();
    changed.runtime.snapshot.memory.messages[1].content = marker.clone();
    if let TaskContinuation::Suspended {
        batch: Some(batch), ..
    } = &mut changed.runtime.continuation
    {
        batch.messages[1].content = marker;
    }
    changed.runtime.validate().unwrap();
    assert!(changed.validate_successor(&old, 1, false).is_err());
}

// Parent and child requests share single-use identity and immutable reviewed-action rules.
#[test]
fn task_child_requests_and_adapter_bindings_cannot_hot_swap() {
    let s = checkpoint("run");
    let mut old: TaskCheckpointPayload = serde_json::from_value(s.payload.clone()).unwrap();
    old.adapters.push(TaskAdapterCheckpoint {
        id: "quality".into(),
        adapter: "host.quality".into(),
        contract_version: 1,
        config: json!({"threshold": 1}),
        state: json!({"round": 1}),
    });
    old.children.push(TaskChildCheckpoint {
        child_id: "child".into(),
        parent_run_id: "run".into(),
        config_identity: "child-v1".into(),
        runtime: old.runtime.clone(),
        pending: Some(TaskPendingRequest {
            id: "child-request".into(),
            issued_revision: 0,
            kind: TaskPendingKind::Approval,
            reviewed_action: json!({"operation": "old"}),
        }),
        result: None,
    });
    old.validate(&s, "config-v1").unwrap();
    let mut next = old.clone();
    next.children[0].pending.as_mut().unwrap().reviewed_action = json!({"operation": "changed"});
    assert!(next.validate_successor(&old, 1, false).is_err());
    next = old.clone();
    next.children[0].pending = None;
    assert!(next.validate_successor(&old, 1, true).is_err());
    next.consumed_request_ids.push("child-request".into());
    next.validate_successor(&old, 1, true).unwrap();
    next.children[0].pending = old.children[0].pending.clone();
    assert!(next.validate(&s, "config-v1").is_err());
    next = old.clone();
    next.adapters[0].config = json!({"threshold": 999});
    assert!(next.validate_successor(&old, 1, false).is_err());
    next = old.clone();
    next.adapters[0].state = json!({"round": 2});
    next.validate_successor(&old, 1, false).unwrap();
    next.pending = old.children[0].pending.clone();
    assert!(next.validate(&s, "config-v1").is_err());
}

// Checkpoints must agree with their own ledger, and interval cleanup must retain charged time.
#[test]
fn task_accounting_and_nested_record_limits_are_consistent() {
    let mut s = checkpoint("run");
    let mut payload: TaskCheckpointPayload = serde_json::from_value(s.payload.clone()).unwrap();
    payload.counters.command_attempts = 1;
    assert!(payload.clone().bind(s.clone()).is_err());
    payload.counters.command_attempts = 0;
    payload.reservations.push(TaskReservation {
        id: "effect".into(),
        state: TaskEffectState::Completed,
        reserved_micro_usd: 2,
        charged_micro_usd: 2,
        write_targets: vec!["target".into()],
        result: Some(json!({"exact": true})),
    });
    assert!(payload.clone().bind(s.clone()).is_err());
    payload.counters.charged_micro_usd = 2;
    payload.declared_write_targets.push("target".into());
    payload.clone().bind(s.clone()).unwrap();
    s.status = TaskRunStatus::Running;
    s.owner_token = Some("owner".into());
    payload.clocks.active_interval_started_at = Some(s.created_at);
    payload.validate(&s, "config-v1").unwrap();
    let mut next = payload.clone();
    next.clocks.active_interval_started_at = None;
    assert!(next.validate_successor(&payload, 1, true).is_err());
    next.clocks.interrupted_interval_millis = 1;
    next.validate_successor(&payload, 1, true).unwrap();
    next.runtime.snapshot.state_machine = Some(ai_agents_core::StateMachineSnapshot {
        current_state: "state".into(),
        previous_state: None,
        turn_count: 0,
        no_transition_count: 0,
        history: (0..MAX_TASK_CHECKPOINT_RECORDS)
            .map(|_| ai_agents_core::StateTransitionEvent {
                from: "a".into(),
                to: "b".into(),
                reason: "test".into(),
                timestamp: s.created_at,
            })
            .collect(),
    });
    assert!(next.bind(s).is_err());
}

// A newly introduced pending identity must be issued in the proposed checkpoint revision.
#[test]
fn task_new_request_revision_and_over_limit_completion_are_rejected() {
    let s = checkpoint("run");
    let old: TaskCheckpointPayload = serde_json::from_value(s.payload.clone()).unwrap();
    let mut next = old.clone();
    next.pending = Some(TaskPendingRequest {
        id: "new".into(),
        issued_revision: 0,
        kind: TaskPendingKind::UserQuestion,
        reviewed_action: json!({}),
    });
    assert!(next.validate_successor(&old, 1, false).is_err());
    next.pending.as_mut().unwrap().issued_revision = 1;
    next.validate_successor(&old, 1, false).unwrap();
    next.pending = None;
    next.counters.llm_attempts = u64::from(next.limits.max_llm_calls) + 1;
    let mut terminal = s;
    terminal.status = TaskRunStatus::Completed;
    assert!(next.bind(terminal).is_err());
}
