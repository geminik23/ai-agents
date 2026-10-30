// Embed a YAML support task using the same domain installation as task-domain-evals.
// Flow: YAML -> installed domain bindings -> actual question -> exact response -> verified ticket -> safe display.
// Run offline from the repository root: cargo run --manifest-path examples/Cargo.toml -p example-custom-llm --bin task-domain-host
// Replace the provider and Records implementation for your host; keep typed decisions and actual domain validation.

mod task_domain;
use ai_agents::autonomy::TaskResumeInput;
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::{Agent, AgentBuilder, TaskHostPrompt, TaskRunStatus};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// A finite offline queue tests the protocol; it is not a semantic decision rule or an application provider.
struct Provider(Mutex<VecDeque<&'static str>>);
#[async_trait]
impl LLMProvider for Provider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<LLMResponse, LLMError> {
        let response = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LLMError::Config("offline queue exhausted".into()))?;
        Ok(LLMResponse::new(response, FinishReason::Stop))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> Result<Box<dyn futures::Stream<Item = Result<LLMChunk, LLMError>> + Unpin + Send>, LLMError>
    {
        Err(LLMError::Config("unexpected token stream".into()))
    }
    fn provider_name(&self) -> &str {
        "offline-support"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = Provider(Mutex::new(VecDeque::from([
        r#"{"tool":"ask_user","arguments":{"question":"Confirm resolution","options":["yes"]}}"#,
        r#"{"tool":"domain","arguments":{"op":"resolve"}}"#,
        "The actual ticket record is resolved.",
    ])));
    // The integration is runtime-only; this application does not load an eval suite or use an eval-specific task controller.
    let builder = AgentBuilder::from_yaml(include_str!(
        "../../../yaml/autonomy/support_resolution.yaml"
    ))?
    .llm(Arc::new(provider))
    .auto_configure_features()?;
    let agent = task_domain::install_domain(builder)?
        .task_interaction_delivery(true)
        .build_task_host()
        .await?;
    let _response = agent.chat("Resolve the ticket.").await?;
    let paused = agent
        .automatic_task_result()
        .ok_or("missing actual task result")?;
    assert_eq!(paused.run.status, TaskRunStatus::Paused);
    let interaction = agent
        .task_interaction(&paused.run.key.run_id, paused.run.revision)
        .await?
        .ok_or("missing retained question")?;
    assert!(matches!(interaction.prompt, TaskHostPrompt::Question(_)));
    // A real application obtains this answer through its authenticated UI, not by parsing model text or display lines.
    let runner = agent
        .automatic_task_runner()
        .ok_or("missing same live runner")?;
    let done = runner
        .resume(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: interaction.request_id,
                answer: serde_json::json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await?;
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.llm_attempts, 3);
    println!(
        "{}",
        serde_json::to_string(&runner.task_display(&done.run.key.run_id).await?)?
    );
    Ok(())
}
