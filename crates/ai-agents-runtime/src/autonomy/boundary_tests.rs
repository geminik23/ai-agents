use super::*;
use crate::autonomy::{AutonomyTurnInput, AutonomyTurnSource, scope_turn};

// Drop outside the task-local scope must transfer uncertain resource protection to the retained owner.
#[tokio::test]
async fn autonomy_resource_guards_retain_uncertain_locks_without_an_owner_cycle() {
    let agent = boundary_agent();
    let owner = agent.reserve_autonomy_run("run".into()).await.unwrap();
    let key = "test-owned-effect".to_string();
    let mut guards = agent
        .acquire_tool_resource_locks(std::slice::from_ref(&key))
        .await
        .unwrap();
    let lock = agent
        .resource_locks
        .read()
        .get(&key)
        .unwrap()
        .upgrade()
        .unwrap();
    guards.task_owner = Some(owner.clone());
    guards.retain_on_drop = true;
    drop(guards);
    assert!(lock.try_lock().is_err());
    agent.release_autonomy_run(&owner).await.unwrap();
    drop(owner);
    assert!(lock.try_lock().is_ok());
}

// Composite restoration protects both state and actor/session identity before the next run reserves them.
#[tokio::test]
async fn autonomy_reservation_observes_complete_restored_session_identity() {
    let agent = boundary_agent();
    let mut snapshot = AgentSnapshot::new(agent.info().id);
    snapshot
        .memory
        .messages
        .push(ChatMessage::user("restored user"));
    let metadata = ai_agents_core::SessionMetadata {
        actor_id: Some("restored-actor".into()),
        ..Default::default()
    };
    agent
        .apply_session_restore(
            "restored-session",
            StoredSessionRestore {
                snapshot,
                metadata: Some(metadata),
            },
        )
        .await
        .unwrap();
    let owner = agent.reserve_autonomy_run("run".into()).await.unwrap();
    assert_eq!(owner.actor_id.as_deref(), Some("restored-actor"));
    assert_eq!(
        agent.current_session_id.read().as_deref(),
        Some("restored-session")
    );
    assert_eq!(
        agent.memory.get_messages(None).await.unwrap()[0].content,
        "restored user"
    );
    agent.release_autonomy_run(&owner).await.unwrap();
}

// Separate runtimes use the existing mock provider and production builder, never a replacement turn executor.
fn boundary_agent() -> Arc<RuntimeAgent> {
    let mut provider = ai_agents_llm::mock::MockLLMProvider::new("default");
    provider.add_response(LLMResponse::new(
        "ordinary",
        ai_agents_core::FinishReason::Stop,
    ));
    Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(Arc::new(provider))
            .build()
            .unwrap(),
    )
}

// Later actor extraction must exclude persisted task history, not merely skip immediate maintenance.
#[test]
fn autonomy_history_projection_survives_message_round_trip() {
    let user = ChatMessage::user("actual user objective");
    let mut controller_reply = ChatMessage::assistant("controller-only material");
    controller_reply.provenance = Some(ai_agents_core::MessageProvenance {
        run_id: "run".into(),
    });
    let exact = vec![user, controller_reply];
    let restored: Vec<ChatMessage> =
        serde_json::from_value(serde_json::to_value(&exact).unwrap()).unwrap();
    assert_eq!(restored.len(), 2);
    let projected = RuntimeAgent::actor_memory_messages(restored).unwrap();
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].content, "actual user objective");
    assert_eq!(exact[1].content, "controller-only material");
}

// A nested ordinary runtime must not inherit another runtime's controller-message classification.
#[tokio::test]
async fn autonomy_input_classification_is_runtime_qualified() {
    let a = boundary_agent();
    let b = boundary_agent();
    let owner = a.reserve_autonomy_run("run-a".into()).await.unwrap();
    scope_turn(
        AutonomyTurnInput {
            owner: owner.clone(),
            objective: "objective".into(),
            controller_message: "private controller".into(),
            source: AutonomyTurnSource::Continuation,
        },
        async {
            assert!(crate::autonomy::current_turn_input(&a.root_turn_gate).is_some());
            assert!(crate::autonomy::current_turn_input(&b.root_turn_gate).is_none());
            assert_eq!(b.chat("ordinary input").await.unwrap().content, "ordinary");
        },
    )
    .await;
    let messages = b.memory.get_messages(None).await.unwrap();
    assert_eq!(messages[0].content, "ordinary input");
    assert!(messages.iter().all(|message| message.provenance.is_none()));
    a.release_autonomy_run(&owner).await.unwrap();
}

// FIFO mutex waiters make release-before-entry deterministic; capability validity must be rechecked after waiting.
#[tokio::test]
async fn autonomy_queued_turn_rechecks_owner_after_release() {
    let agent = boundary_agent();
    let owner = agent.reserve_autonomy_run("run".into()).await.unwrap();
    let gate = agent.root_turn_gate.clone().lock_owned().await;
    let release = agent.release_autonomy_run(&owner);
    let entry = agent.run_autonomy_turn(AutonomyTurnInput {
        owner: owner.clone(),
        objective: "objective".into(),
        controller_message: String::new(),
        source: AutonomyTurnSource::InitialObjective,
    });
    futures::pin_mut!(release, entry);
    assert!(futures::poll!(release.as_mut()).is_pending());
    assert!(futures::poll!(entry.as_mut()).is_pending());
    drop(gate);
    release.await.unwrap();
    assert!(
        entry
            .await
            .unwrap_err()
            .to_string()
            .contains("owner mismatch")
    );
    assert!(agent.memory.is_empty());
}

// Copying the task source label cannot authorize direct tools or state mutation on a reserved runtime.
#[tokio::test]
async fn autonomy_reservation_blocks_direct_host_operations() {
    let agent = boundary_agent();
    let owner = agent.reserve_autonomy_run("run".into()).await.unwrap();
    let request = ToolExecutionRequest::new(
        "host",
        "echo",
        serde_json::json!({"message":"unrelated"}),
        ToolCallSource::Task,
    );
    assert!(
        agent
            .invoke_tool(request)
            .await
            .unwrap_err()
            .to_string()
            .contains("reserved")
    );
    assert!(agent.transition_to("unrelated").await.is_err());
    agent.release_autonomy_run(&owner).await.unwrap();
}
