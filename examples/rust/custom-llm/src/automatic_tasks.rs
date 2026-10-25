// Development-only YAML state/skill task entry with an offline provider and no tools or external effects.
// Flow: ordinary input -> automatic shared controller -> retained question -> explicit host answer -> completion.
// Rust supplies the host lifetime and UI boundary; the YAML chooses objective, scope, limits and completion policy.

use ai_agents::autonomy::{TaskResumeInput, TaskRunEventKind, TaskRunStatus};
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::{AgentBuilder, AgentStreamEvent, Result};
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct OfflineProvider {
    router: bool,
    calls: AtomicUsize,
}
#[async_trait]
impl LLMProvider for OfflineProvider {
    // Fixed replies exercise structural control; replace these with a provider and domain validator for semantic quality.
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(LLMResponse::new(
            if self.router {
                "research"
            } else if call == 0 {
                "draft"
            } else {
                "verified report"
            },
            FinishReason::Stop,
        ))
    }
    // Automatic task lifecycle events do not require provider token streaming.
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
        "offline-automatic-task"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

const STATE: &str = r#"
name: OfflineStateTask
system_prompt: Produce a verified report.
tools: []
states:
  initial: reporting
  states:
    reporting:
      autonomy:
        enabled: true
        mode: state_until_exit
        max_turns: 3
        max_llm_calls: 3
        completion: {response_contains: verified report}
        premature_finish: {action: ask_user, on_unavailable: stop}
"#;

const SKILL: &str = r#"
name: OfflineSkillTask
system_prompt: Produce a verified report.
llm: {default: default, router: router}
tools: []
skills:
  - id: research
    description: Research a report
    trigger: The user requests a report
    autonomy:
      enabled: true
      mode: skill_until_complete
      objective_template: '{{ user_input }}'
      max_turns: 3
      max_llm_calls: 3
      completion: {response_contains: verified report}
      premature_finish: {action: ask_user, on_unavailable: stop}
    steps:
      - prompt: 'Prepare the report for {{ user_input }}'
"#;

#[tokio::main]
async fn main() -> Result<()> {
    for (scope, yaml) in [("state", STATE), ("skill", SKILL)] {
        let main = Arc::new(OfflineProvider {
            router: false,
            calls: AtomicUsize::new(0),
        });
        let router = Arc::new(OfflineProvider {
            router: true,
            calls: AtomicUsize::new(0),
        });
        // The same builder used for ordinary YAML installs retained volatile hosting only for enabled declarations.
        let agent = AgentBuilder::from_yaml(yaml)?
            .llm(main.clone())
            .llm_alias("router", router.clone())
            .build()?;
        let mut initial = agent.chat_stream_events("Prepare the report").await?;
        let mut saw_pause = false;
        while let Some(event) = initial.next().await {
            match event {
                AgentStreamEvent::Task(event) => {
                    saw_pause |= event.kind == TaskRunEventKind::TaskPaused;
                    println!("{}", serde_json::to_string(&event)?);
                }
                AgentStreamEvent::Final(response) => {
                    assert_eq!(
                        response.metadata.as_ref().unwrap()["task_run"]["status"],
                        "paused"
                    );
                }
                _ => {}
            }
        }
        assert!(saw_pause);
        let paused = agent
            .automatic_task_result()
            .expect("acknowledged operation result");
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        // Exact question details are trusted-host data for an authorized UI, not a display-event export.
        let pending = paused
            .run
            .pending
            .as_ref()
            .expect("retained controller question");
        let runner = agent.automatic_task_runner().expect("same live controller");
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
        let done = resumed.take_result().expect("drained exact result")?;
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(done.run.counters.llm_attempts, 2);
        assert_eq!(main.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            router.calls.load(Ordering::SeqCst),
            usize::from(scope == "skill")
        );
        assert_eq!(
            agent.automatic_task_result().unwrap().run.status,
            TaskRunStatus::Completed
        );
        println!("{scope}: acknowledged completion");
    }
    Ok(())
}
