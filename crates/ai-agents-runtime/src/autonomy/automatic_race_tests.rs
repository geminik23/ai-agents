//! Gated delivery tests cover public runner aliases, spawned consumers and failed acknowledgement without timing-based inference.

use super::*;
use crate::orchestration::{PipelineStage, pipeline};
use crate::spawner::{AgentRegistry, SpawnedAgent};

// A public runner alias publishes the same acknowledged result; a later explicit run cannot replace a waiting parent's outcome.
#[tokio::test]
async fn auto_raw_runner_resume_wakes_parent_without_cross_run_result_substitution() {
    let registry = AgentRegistry::new();
    let yaml = SKILL.replace(
        "      completion:",
        "      premature_finish: {action: ask_user, on_unavailable: stop}\n      completion:",
    );
    let main = RecordingProvider::new(&["draft", "done-a", "done-b"]);
    let spec = crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
    let runtime = agent(&yaml, main.clone(), RecordingProvider::new(&["work"]));
    registry
        .register(SpawnedAgent::from_runtime("child".into(), runtime, spec))
        .await
        .unwrap();
    let child = registry.get("child").unwrap();
    let stages = vec![PipelineStage::id("child")];
    let mut work = Box::pin(pipeline(
        &registry,
        "objective",
        &stages,
        Some(30_000),
        None,
        None,
    ));
    let paused = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            tokio::select! {
                result = &mut work => panic!("returned paused child as success: {result:?}"),
                _ = tokio::task::yield_now() => if let Some(result) = child.automatic_task_result() { break result; },
            }
        }
    }).await.unwrap();
    let runner = child.automatic_task_runner().unwrap();
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: paused.run.pending.unwrap().id,
                answer: json!("continue"),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    // Do not poll the parent until another public operation has completed on the now released runtime.
    let other = runner
        .run_skill("another objective", "work", None)
        .await
        .unwrap();
    assert_eq!(other.run.status, TaskRunStatus::Completed);
    assert_ne!(done.run.key, other.run.key);
    assert_eq!(other.final_response.unwrap().content, "done-b");
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), work)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.response.content, "done-a");
    assert_eq!(child.automatic_task_result().unwrap().run.key, done.run.key);
}

// Spawned SDK workers must carry the composing role explicitly because Tokio task-locals do not cross spawn boundaries.
#[tokio::test]
async fn auto_spawned_concurrent_and_broadcast_wait_for_hosted_child() {
    for broadcast in [false, true] {
        let registry = AgentRegistry::new();
        let yaml = STATE.replace("        completion:", "        premature_finish: {action: ask_user, on_unavailable: stop}\n        completion:");
        let spec = crate::spec::AgentSpec::from_yaml_strict(&yaml).unwrap();
        let runtime = agent(
            &yaml,
            RecordingProvider::new(&["draft", "done"]),
            RecordingProvider::new(&[]),
        );
        registry
            .register(SpawnedAgent::from_runtime("child".into(), runtime, spec))
            .await
            .unwrap();
        let child = registry.get("child").unwrap();
        let agents = vec![ai_agents_state::ConcurrentAgentRef::Id("child".into())];
        let aggregation: ai_agents_state::AggregationConfig =
            serde_yaml::from_str("strategy: all").unwrap();
        let mut work: std::pin::Pin<
            Box<dyn std::future::Future<Output = ai_agents_core::Result<String>> + Send + '_>,
        > = if broadcast {
            Box::pin(async {
                let mut results = registry.broadcast("sender", "objective").await;
                Ok(results.remove(0).1?.content)
            })
        } else {
            Box::pin(async {
                let result = crate::orchestration::concurrent(
                    &registry,
                    "objective",
                    &agents,
                    &aggregation,
                    None,
                    Some(1),
                    Some(30_000),
                    ai_agents_state::PartialFailureAction::Abort,
                    None,
                )
                .await?;
                Ok(result.agent_results[0]
                    .response
                    .as_ref()
                    .unwrap()
                    .content
                    .clone())
            })
        };
        let paused = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                tokio::select! {
                    result = &mut work => panic!("spawned caller returned before host decision: {result:?}"),
                    _ = tokio::task::yield_now() => if let Some(result) = child.automatic_task_result() { break result; },
                }
            }
        }).await.unwrap();
        let runner = child.automatic_task_runner().unwrap();
        let mut events = runner
            .resume_stream(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: paused.run.pending.unwrap().id,
                    answer: json!("continue"),
                },
            )
            .unwrap();
        while events.next_event().await.is_some() {}
        assert_eq!(
            events.take_result().unwrap().unwrap().run.status,
            TaskRunStatus::Completed
        );
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(10), work)
                .await
                .unwrap()
                .unwrap(),
            "done"
        );
    }
}

struct StreamFailure;
#[async_trait]
impl ai_agents_core::LLMProvider for StreamFailure {
    // The ordinary follow-up proves that a retained failed stream cannot suppress the next input commit.
    async fn complete(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<ai_agents_core::LLMResponse, ai_agents_core::LLMError> {
        Ok(ai_agents_core::LLMResponse::new(
            "ordinary",
            ai_agents_core::FinishReason::Stop,
        ))
    }
    // No external work is performed; this is a genuine provider-stream error after ordinary root preparation.
    async fn complete_stream(
        &self,
        _: &[ai_agents_core::ChatMessage],
        _: Option<&ai_agents_core::LLMConfig>,
    ) -> std::result::Result<
        Box<
            dyn futures::Stream<
                    Item = std::result::Result<ai_agents_llm::LLMChunk, ai_agents_core::LLMError>,
                > + Unpin
                + Send,
        >,
        ai_agents_core::LLMError,
    > {
        Ok(Box::new(futures::stream::iter(vec![Err(
            ai_agents_core::LLMError::Config("isolated stream failure".into()),
        )])))
    }
    // Distinguish this isolated fixture from real provider transports during preflight.
    fn provider_name(&self) -> &str {
        "isolated-stream-failure"
    }
    // Use the legacy protocol so the test isolates ordinary error cleanup rather than native history.
    fn supports(&self, _: ai_agents_core::LLMFeature) -> bool {
        false
    }
}

// Error delivery must finish the old generator's cleanup before the next root can acquire the gate.
#[tokio::test]
async fn auto_ordinary_stream_error_cannot_corrupt_next_root_while_stream_is_retained() {
    let agent = AgentBuilder::from_yaml(SKILL)
        .unwrap()
        .llm(Arc::new(StreamFailure))
        .llm_alias("router", RecordingProvider::new(&["none", "none"]))
        .build()
        .unwrap();
    let mut stream = agent.chat_stream_events("first input").await.unwrap();
    loop {
        if matches!(
            stream.next().await,
            Some(AgentStreamEvent::Chunk(crate::StreamChunk::Error { .. }))
        ) {
            break;
        }
    }
    let response = agent.chat("second input").await.unwrap();
    assert_eq!(response.content, "ordinary");
    let snapshot = agent.save_state().await.unwrap();
    let users: Vec<_> = snapshot
        .memory
        .messages
        .iter()
        .filter(|message| message.role == ai_agents_core::Role::User)
        .collect();
    assert_eq!(users.len(), 2);
    assert_eq!(users[0].content, "first input");
    assert_eq!(users[1].content, "second input");
    drop(stream);
}

// A lost or rejected terminal acknowledgement never becomes a completed host result or ordinary Final.
#[tokio::test]
async fn auto_terminal_storage_fault_preserves_recovery_without_success_delivery() {
    use crate::autonomy::continuation_test_support::{self as support, ReceiptStore};
    for &sqlite in support::backends() {
        for mode in ["reject", "lost"] {
            let inner = support::store("auto".into(), "automatic.v1", sqlite).await;
            let store = ReceiptStore::new(inner.clone(), mode, TaskRunStatus::Completed);
            store.armed.store(true, Ordering::SeqCst);
            let runtime = AgentBuilder::from_yaml(STATE)
                .unwrap()
                .llm(RecordingProvider::new(&["done"]))
                .automatic_task_options(AutomaticTaskOptions {
                    store: Some(store.clone()),
                    config_identity: Some("automatic.v1".into()),
                    ..Default::default()
                })
                .build()
                .unwrap();
            let mut stream = runtime.chat_stream_events("request").await.unwrap();
            let mut finals = 0;
            let mut errors = 0;
            while let Some(event) = stream.next().await {
                match event {
                    AgentStreamEvent::Final(_) => finals += 1,
                    AgentStreamEvent::Chunk(crate::StreamChunk::Error { .. }) => errors += 1,
                    _ => {}
                }
            }
            store.assert_hit();
            assert_eq!(finals, 0);
            assert_eq!(errors, 1);
            assert!(runtime.automatic_task_result().is_none());
            assert!(runtime.chat("cannot reuse uncertain owner").await.is_err());
        }
    }
}

// A configured hard priced guarantee is capability checked before any automatic execution or tool effect.
#[test]
fn auto_unsupported_priced_cap_rejects_without_provider_calls() {
    let provider = RecordingProvider::new(&[]);
    let yaml = STATE.replace(
        "        max_turns: 4",
        "        max_turns: 4\n        max_cost_usd: '1.00'",
    );
    let result = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(provider.clone())
        .automatic_task_options(AutomaticTaskOptions {
            ceilings: AutonomyHostCeilings {
                max_cost_usd: Some(serde_json::from_value(json!("1.00")).unwrap()),
                ..Default::default()
            },
            ..Default::default()
        })
        .build();
    assert!(result.is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

struct CreateGate {
    inner: Arc<ScopedTaskRunStore>,
    reached: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl TaskRunStore for CreateGate {
    // The barrier holds the real initial checkpoint acknowledgement without replacing any runtime or effect implementation.
    async fn create(&self, snapshot: &TaskRunSnapshot) -> ai_agents_core::Result<()> {
        self.reached.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        self.inner.create(snapshot).await
    }
    // Other operations retain the exact production conditional storage contract.
    async fn load(&self, id: &str) -> ai_agents_core::Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    // Forward CAS without relaxing revision, owner or receipt checks.
    async fn mutate(
        &self,
        id: &str,
        mutation: &TaskRunMutation,
    ) -> ai_agents_core::Result<TaskRunSnapshot> {
        self.inner.mutate(id, mutation).await
    }
    // Expose only the underlying scoped metadata rather than synthesizing recovery data.
    async fn list(&self) -> ai_agents_core::Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    // Gating creation must not bypass exact-revision deletion protection.
    async fn delete(&self, id: &str, revision: u64) -> ai_agents_core::Result<()> {
        self.inner.delete(id, revision).await
    }
}

// Initial handoff cannot relabel a selected skill under a newer state generation after an awaited storage operation.
#[tokio::test]
async fn auto_selected_generation_is_rechecked_after_initial_checkpoint_wait() {
    let store = Arc::new(CreateGate {
        inner: Arc::new(
            ScopedTaskRunStore::in_memory("auto".into(), None, "automatic.v1".into()).unwrap(),
        ),
        reached: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    let yaml = SKILL.to_string() + "states:\n  initial: a\n  states:\n    a: {}\n    b: {}\n";
    let main = RecordingProvider::new(&["must not execute"]);
    let runtime = AgentBuilder::from_yaml(&yaml)
        .unwrap()
        .llm(main.clone())
        .llm_alias("router", RecordingProvider::new(&["work"]))
        .automatic_task_options(AutomaticTaskOptions {
            store: Some(store.clone()),
            config_identity: Some("automatic.v1".into()),
            ..Default::default()
        })
        .build()
        .unwrap();
    let machine = runtime.state_machine().unwrap().clone();
    let mut work = Box::pin(runtime.chat("work"));
    tokio::select! {
        result = &mut work => panic!("initial checkpoint was not awaited: {result:?}"),
        permit = store.reached.acquire() => permit.unwrap().forget(),
    }
    machine.transition_to("b", "host changes state").unwrap();
    machine
        .transition_to("a", "same label is not same generation")
        .unwrap();
    store.release.add_permits(1);
    work.await.unwrap();
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        runtime.automatic_task_result().unwrap().run.status,
        TaskRunStatus::Failed
    );
}

// Separate ordinary clarification turns retain extraction data when the selected skill is finally promoted into a task.
#[tokio::test]
async fn auto_ordinary_clarification_keeps_prepared_context_without_reextraction_or_routing() {
    let clear = r#"{"is_ambiguous":false,"confidence":1.0}"#;
    let ambiguous = r#"{"is_ambiguous":true,"confidence":0.1,"what_is_unclear":["target"],"reasoning":"missing","ambiguity_type":"missing_target"}"#;
    let question = r#"{"question":"Which target?","style":"open","clarifying":["target"]}"#;
    let answer = r#"{"status":"answered","enriched_input":"request for target one","resolved":{"target":"one"}}"#;
    let yaml = SKILL.replace(
        "    steps:",
        "    disambiguation: {enabled: true, required_clarity: [target]}\n    steps:",
    ) + "disambiguation:\n  enabled: true\n  detection: {llm: router}\n  clarification: {llm: router}\nprocess:\n  input:\n    - type: extract\n      config:\n        llm: router\n        schema:\n          tag: {type: string, required: true}\n        store_in_context: extracted\n  output:\n    - type: format\n      config:\n        template: '{{ response }} : {{ context.extracted }}'\n";
    let main = RecordingProvider::new(&["done"]);
    let router = RecordingProvider::new(&[
        clear,
        r#"{"tag":"original"}"#,
        "work",
        ambiguous,
        question,
        answer,
        clear,
    ]);
    let runtime = agent(&yaml, main.clone(), router.clone());
    let question = runtime.chat("request").await.unwrap();
    assert!(question.content.contains("Which target"));
    assert!(runtime.automatic_task_result().is_none());
    assert_eq!(main.calls.load(Ordering::SeqCst), 0);
    let done = runtime.chat("target one").await.unwrap();
    assert_eq!(
        runtime.automatic_task_result().unwrap().run.status,
        TaskRunStatus::Completed
    );
    assert!(done.content.contains("original"), "{}", done.content);
    assert_eq!(router.calls.load(Ordering::SeqCst), 7);
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);
}
