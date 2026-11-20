// Public Rust consumer of the configured host with the same domain components used by application and eval examples.
// The model is deterministic; SQLite, native history, domain records and corpus reads are real isolated state.
#![cfg(any(target_os = "linux", target_os = "macos"))]
use super::task_domain;

use ai_agents::autonomy::*;
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::{Agent, AgentBuilder, AgentError, PreparedAgentSpec, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

struct Program(Arc<AtomicUsize>);
#[async_trait]
impl LLMProvider for Program {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let program = [
            r#"{"tool":"research_catalog","arguments":{"op":"read","id":"alerts-a"}}"#,
            r#"{"tool":"domain","arguments":{"op":"source","value":"ticket://original-context"}}"#,
            r#"{"tool":"ask_user","arguments":{"question":"Confirm resolution","options":["yes"],"allow_other":false}}"#,
            r#"{"tool":"domain","arguments":{"op":"resolve"}}"#,
            "The original ticket is resolved.",
        ];
        let next = self.0.fetch_add(1, Ordering::SeqCst);
        let text = program
            .get(next)
            .ok_or_else(|| LLMError::Config("unexpected replay".into()))?;
        Ok(LLMResponse::new(*text, FinishReason::Stop))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Err(LLMError::Config("unexpected stream".into()))
    }
    fn provider_name(&self) -> &str {
        "offline-restorable-domain"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}
struct State {
    domain: Arc<dyn TaskRestoreComponentState>,
    cursor: Arc<AtomicUsize>,
}
impl TaskRestoreComponentState for State {
    fn binding(&self) -> Result<TaskRestoreBinding> {
        Ok(TaskRestoreBinding {
            id: "example.offline-domain-host".into(),
            contract_version: 1,
            configuration: json!({"program":"support-and-corpus-v1"}),
            state: json!({"domain":self.domain.binding()?, "cursor":self.cursor.load(Ordering::SeqCst)}),
        })
    }
}
struct Factory;
impl TaskRestoreComponents for Factory {
    fn identity(&self) -> (String, u32) {
        ("example.offline-domain-host".into(), 1)
    }
    fn validate(&self, _: &PreparedAgentSpec, binding: &TaskRestoreBinding) -> Result<()> {
        if (binding.id.clone(), binding.contract_version) != self.identity()
            || binding.configuration != json!({"program":"support-and-corpus-v1"})
            || binding.state["cursor"]
                .as_u64()
                .is_none_or(|cursor| cursor > 5)
        {
            return Err(AgentError::Config("offline host differs".into()));
        }
        task_domain::validate_domain_binding(&serde_json::from_value(
            binding.state["domain"].clone(),
        )?)
    }
    fn install(
        &self,
        prepared: &PreparedAgentSpec,
        builder: AgentBuilder,
        saved: Option<&TaskRestoreBinding>,
        shared: Option<&ai_agents::llm::LLMRegistry>,
    ) -> Result<(AgentBuilder, Arc<dyn TaskRestoreComponentState>)> {
        if let Some(saved) = saved {
            self.validate(prepared, saved)?;
        }
        let cursor = Arc::new(AtomicUsize::new(saved.map_or(0, |binding| {
            binding.state["cursor"].as_u64().unwrap() as usize
        })));
        let builder = if shared.is_none() {
            builder.llm(Arc::new(Program(cursor.clone())))
        } else {
            builder
        };
        let domain: Option<TaskRestoreBinding> = saved
            .map(|binding| serde_json::from_value(binding.state["domain"].clone()))
            .transpose()?;
        let (builder, domain) =
            task_domain::install_domain_state(builder.auto_configure_features()?, domain.as_ref())?;
        Ok((builder, Arc::new(State { domain, cursor })))
    }
}

#[tokio::test]
async fn configured_public_consumer_restores_ticket_source_and_corpus_receipts() {
    let scratch = ai_agents_eval::fixtures::AttemptFixtureContext::create(None).unwrap();
    let mut source: serde_yaml::Value = serde_yaml::from_str(include_str!(
        "../../../../yaml/autonomy/support_resolution.yaml"
    ))
    .unwrap();
    source["tools"]
        .as_sequence_mut()
        .unwrap()
        .push(serde_yaml::Value::String("research_catalog".into()));
    source["storage"] = serde_yaml::to_value(
        json!({"type":"sqlite","path":scratch.workspace.join("tasks.sqlite")}),
    )
    .unwrap();
    source["autonomy"]["restoration_timeout_ms"] = 5000.into();
    source["autonomy"]["persistence"] = serde_yaml::to_value(json!({"enabled":true})).unwrap();
    let source = serde_yaml::to_string(&source).unwrap();
    let make_host = || {
        Arc::new(
            ConfiguredTaskRestoreHost::new(
                PreparedAgentSpec::from_yaml_strict(&source).unwrap(),
                scratch.workspace.clone(),
                None,
                Duration::from_secs(5),
            )
            .unwrap()
            .components(Arc::new(Factory))
            .unwrap(),
        )
    };
    let producer = make_host();
    let agent = producer.build().await.unwrap();
    agent
        .chat("Read the source and resolve the ticket after confirmation.")
        .await
        .unwrap();
    let saved = agent.automatic_task_result().unwrap();
    assert_eq!(saved.run.status, TaskRunStatus::Paused);
    assert_eq!(saved.run.counters.llm_attempts, 3);
    assert!(
        agent
            .automatic_task_runner()
            .unwrap()
            .prepared_restart_available(&saved.run.key.run_id, saved.run.revision)
            .await
            .unwrap()
    );
    drop(agent);
    drop(producer);
    let consumer = make_host();
    let (runner, restored) = consumer
        .restore(saved.run.key.run_id.clone(), saved.run.revision)
        .await
        .unwrap();
    let review = runner
        .restoration_review(&saved.run.key.run_id)
        .await
        .unwrap()
        .unwrap();
    let reviewed = runner
        .approve_restoration(
            &saved.run.key.run_id,
            restored.run.revision,
            &review.id,
            ai_agents::hitl::ApprovalResult::Approved,
        )
        .await
        .unwrap();
    let done = runner
        .resume(
            &saved.run.key.run_id,
            reviewed.run.revision,
            TaskResumeInput::UserAnswer {
                request_id: reviewed.run.pending.as_ref().unwrap().id.clone(),
                answer: json!({"answered":true,"selected":["yes"]}),
            },
        )
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.counters.llm_attempts, 5);
    let records: Vec<Value> = done
        .run
        .evidence
        .tool_calls
        .iter()
        .filter(|entry| entry.record.canonical_id == "domain" && entry.record.executed)
        .map(|entry| serde_json::from_str(&entry.record.output).unwrap())
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[1]["sources"], json!(["ticket://original-context"]));
    assert_eq!(records[1]["ticket"], "resolved");
    assert_eq!(done.run.evidence.tool_calls.iter().filter(|entry| entry.record.canonical_id == "research_catalog" && entry.record.executed).count(), 1);
    // The reconstructed application object contains the real read identity; a final citation string cannot manufacture it.
    let actual = consumer.runtime_binding(runner.runtime(), None).unwrap();
    let domain = &actual.state["components"]["state"]["domain"]["state"];
    assert_eq!(
        domain["records"]["sources"],
        json!(["ticket://original-context"])
    );
    assert_eq!(
        domain["corpus"]["read"]
            .as_object()
            .unwrap()
            .values()
            .collect::<Vec<_>>(),
        vec![&json!("alerts-a")]
    );
    drop(runner);
    drop(consumer);
    // Runtime pools can finish their own shutdown asynchronously; this fixture owns only its isolated directory.
    std::fs::remove_dir_all(&scratch.workspace).unwrap();
}
