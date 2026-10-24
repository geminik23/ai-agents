// Development-only task events with an offline provider and no tools, credentials or external effects.
// Flow: bounded objective -> unmet gate -> retained question -> explicit answer -> acknowledged completion.
// Replace the fixture provider and completion predicate for your domain; events are not provider token deltas.

use ai_agents::autonomy::{
    AutonomyConfig, AutonomyHostCeilings, AutonomyRunner, ScopedTaskRunStore, TaskResumeInput,
    TaskRunStatus,
};
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::{AgentBuilder, Result};
use async_trait::async_trait;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct FixtureProvider(AtomicUsize);
#[async_trait]
impl LLMProvider for FixtureProvider {
    // Fixed replies demonstrate the runtime protocol, not semantic task-quality evaluation.
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        Ok(LLMResponse::new(
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                "draft"
            } else {
                "verified report"
            },
            FinishReason::Stop,
        ))
    }
    // Task lifecycle events also work with providers that supply no incremental token stream.
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Ok(Box::new(futures::stream::empty()))
    }
    fn provider_name(&self) -> &str {
        "offline-task-fixture"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Arc::new(
        AgentBuilder::new()
            .system_prompt("Produce a report.")
            .llm(Arc::new(FixtureProvider(AtomicUsize::new(0))))
            .build()?,
    );
    let config: AutonomyConfig = serde_json::from_value(serde_json::json!({
        "enabled":true,"max_turns":3,"max_llm_calls":3,
        "completion":{"response_contains":"verified report"},
        "premature_finish":{"action":"ask_user","on_unavailable":"stop"}
    }))?;
    // Volatile pause is valid while this host retains the runner; it is not restart persistence.
    let store = Arc::new(ScopedTaskRunStore::in_memory(
        agent.info().id,
        None,
        "offline-task.v1".into(),
    )?);
    let runner = AutonomyRunner::try_new(
        agent.clone(),
        config,
        AutonomyHostCeilings::default(),
        store,
        "offline-task.v1".into(),
    )?;
    let mut events = agent.run_task_stream(&runner, "Prepare a report", None)?;
    while let Some(event) = events.next_event().await {
        println!("{}", serde_json::to_string(&event)?);
    }
    let paused = events.take_result().expect("drained operation result")?;
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    // Only trusted host code reads exact question data; do not export the raw result as an event.
    let pending = paused.run.pending.as_ref().expect("retained question");
    let mut resumed = runner.resume_stream(
        &paused.run.key.run_id,
        paused.run.revision,
        TaskResumeInput::UserAnswer {
            request_id: pending.id.clone(),
            answer: serde_json::json!("continue the report"),
        },
    )?;
    while let Some(event) = resumed.next_event().await {
        println!("{}", serde_json::to_string(&event)?);
    }
    let done = resumed.take_result().expect("drained operation result")?;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.llm_attempts, 2);
    Ok(())
}
