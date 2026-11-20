// Run the prepared sample project through the existing task controller with a finite offline provider.
// Flow: initial failing check -> scripted read and edit -> managed command -> fresh acceptance.
// Rust binds the command runner and inspects actual execution records, not model completion text.
// The scripted edit demonstrates actual execution; it is not evidence of live model reasoning.

use crate::code_agent_workspace::PreparedConfiguration;
use ai_agents::autonomy::{
    AutonomyExtensions, AutonomyHostCeilings, AutonomyRunner, ScopedTaskRunStore, TaskRunStatus,
};
use ai_agents::llm::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents::tools::{ProcessCommandRunner, ToolCallSource};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

struct FixtureProvider(Mutex<VecDeque<String>>);

#[async_trait]
impl LLMProvider for FixtureProvider {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let response = self
            .0
            .lock()
            .map_err(|_| LLMError::Config("fixture provider lock poisoned".into()))?
            .pop_front()
            .ok_or_else(|| LLMError::Config("fixture provider responses exhausted".into()))?;
        Ok(LLMResponse::new(response, FinishReason::Stop))
    }

    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        Err(LLMError::Config("fixture does not stream".into()))
    }

    fn provider_name(&self) -> &str {
        "offline-code-fixture"
    }

    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}

// Keep the original runtime, executor and task owner; only this disposable sample receives scripted input.
// Initial and final independent checks are host-owned; the task validation command uses the task ledger.
pub async fn run(setup: &PreparedConfiguration) -> Result<(), Box<dyn std::error::Error>> {
    let initial = setup.verify_fixture()?;
    if initial.passed {
        return Err("sample project is already passing; prepare a new failing workspace".into());
    }
    let initial_output = std::fs::read_to_string(initial.scratch.join("stdout.txt"))?;
    if !initial_output.contains("test preserves_value_inside_bounds ... FAILED") {
        return Err(format!(
            "initial validation did not demonstrate the intended sample failure; evidence: {}",
            initial.scratch.display()
        )
        .into());
    }
    println!(
        "Initial sample-project acceptance failed as expected. Evidence: {}",
        initial.scratch.display()
    );

    let source = setup.source_file();
    let responses = VecDeque::from([
        serde_json::json!({"tool":"file_read","arguments":{"path":source}}).to_string(),
        serde_json::json!({"tool":"file_edit","arguments":{
            "path":source,
            "old_text":"value.max(lower).max(upper)",
            "new_text":"value.max(lower).min(upper)",
            "dry_run":false
        }})
        .to_string(),
        "The sample edit is ready for the required acceptance check.".to_string(),
    ]);
    let provider = Arc::new(FixtureProvider(Mutex::new(responses)));
    let agent = Arc::new(
        setup
            .builder()
            .llm(provider)
            .auto_configure_features()?
            .autonomy_extensions(AutonomyExtensions::builtins())
            .build()?,
    );
    // This runtime's built-ins share their actual command slot; no eval recording registry is replaced.
    agent.set_command_runner(Arc::new(ProcessCommandRunner));
    let store = Arc::new(ScopedTaskRunStore::in_memory(
        agent.info().id,
        None,
        "offline-code-agent-fixture.v1".into(),
    )?);
    let runner = AutonomyRunner::try_new(
        agent,
        setup.task_profile(),
        AutonomyHostCeilings::default(),
        store,
        "offline-code-agent-fixture.v1".into(),
    )?;
    let result = runner
        .run(
            "Fix the sample project's clamp function and validate it.",
            None,
        )
        .await?;
    if result.run.status != TaskRunStatus::Completed {
        return Err(format!("sample task did not complete: {:?}", result.run.status).into());
    }
    for id in ["file_read", "file_edit"] {
        let record = &result
            .run
            .evidence
            .tool_calls
            .iter()
            .find(|evidence| {
                evidence.record.canonical_id == id
                    && evidence.record.executed
                    && evidence.record.success
                    && matches!(&evidence.record.source, ToolCallSource::Model)
                    && evidence
                        .record
                        .executed_arguments
                        .get("path")
                        .and_then(|value| value.as_str())
                        == source.to_str()
            })
            .ok_or_else(|| format!("missing executed {id} evidence for the authorized source"))?
            .record;
        if id == "file_edit" {
            let output: serde_json::Value = serde_json::from_str(&record.output)?;
            if record.output_truncated
                || record
                    .executed_arguments
                    .get("dry_run")
                    .and_then(|value| value.as_bool())
                    != Some(false)
                || output["mutation_performed"] != true
            {
                return Err("file_edit did not report an actual completed mutation".into());
            }
        }
    }
    let command = result
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|evidence| {
            evidence.record.canonical_id == "command"
                && evidence.record.executed
                && matches!(&evidence.record.source, ToolCallSource::Task)
        })
        .ok_or("missing executed validation command")?;
    let command_result: serde_json::Value = serde_json::from_str(&command.record.output)?;
    if !command.record.success
        || command.record.output_truncated
        || command.record.timed_out
        || command.record.cancelled
        || command_result["exit_code"] != 0
        || command_result["termination"] != "exited"
    {
        return Err(
            "the actual validation command did not produce a complete passing result".into(),
        );
    }
    let final_check = setup.verify_fixture()?;
    if !final_check.passed {
        return Err(format!(
            "fresh independent sample acceptance did not pass; evidence: {}",
            final_check.scratch.display()
        )
        .into());
    }
    println!("Actual file_read and file_edit executed through the shared executor.");
    println!("The runtime command and a fresh independent acceptance both passed.");
    println!(
        "Fresh acceptance evidence: {}",
        final_check.scratch.display()
    );
    Ok(())
}
