//! Model-tool composition reuses existing algorithms under invocation-specific captured frame authority.

use crate::autonomy::{
    self,
    composition::{CompositionChild, scope_message_composition},
    message::{MessageResponse, MessageState},
};
use crate::spawner::AgentRegistry;
use ai_agents_core::{AgentError, AgentResponse, Result, ToolResult};
use ai_agents_llm::LLMRegistry;
use serde_json::{Value, json};
use std::sync::Arc;

/// Parses the existing public tool arguments before capturing a catalogue or admitting any child work.
fn members(args: &Value, kind: &MessageResponse) -> Result<(String, Vec<String>)> {
    let (key, input_key) = match kind {
        MessageResponse::Pipeline => ("stages", "input"),
        MessageResponse::Concurrent => ("agents", "question"),
        MessageResponse::Group => ("participants", "topic"),
        MessageResponse::Handoff => ("available_agents", "input"),
        _ => return Err(AgentError::Config("not a composition tool".into())),
    };
    let input = args
        .get(input_key)
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Config("missing composition input".into()))?;
    let mut ids: Vec<String> = args
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| AgentError::Config("missing composition agents".into()))?
        .iter()
        .map(|id| {
            id.as_str()
                .map(str::to_owned)
                .ok_or_else(|| AgentError::Config("composition agent ID must be text".into()))
        })
        .collect::<Result<_>>()?;
    if matches!(kind, MessageResponse::Handoff) {
        let initial = args
            .get("initial_agent")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError::Config("missing initial handoff agent".into()))?;
        if !ids.iter().any(|id| id == initial) {
            ids.insert(0, initial.into());
        }
    }
    if ids.is_empty() || ids.len() > autonomy::MAX_TASK_CHECKPOINT_RECORDS {
        return Err(AgentError::Config("empty or oversized composition".into()));
    }
    Ok((input.into(), ids))
}

/// Installation consumes the actual final-executor permit once and pins every selected registry object before execution.
pub(crate) async fn dispatch(
    registry: Arc<AgentRegistry>,
    llms: Option<Arc<LLMRegistry>>,
    args: Value,
    concurrent: bool,
) -> Result<ToolResult> {
    dispatch_kind(
        registry,
        llms,
        args,
        if concurrent {
            MessageResponse::Concurrent
        } else {
            MessageResponse::Pipeline
        },
    )
    .await
}

/// Every structured model tool uses its own captured algorithm and catalogue under the same shared execution permit.
pub(crate) async fn dispatch_kind(
    registry: Arc<AgentRegistry>,
    llms: Option<Arc<LLMRegistry>>,
    args: Value,
    kind: MessageResponse,
) -> Result<ToolResult> {
    let (input, ids) = members(&args, &kind)?;
    let Some(invocation) = autonomy::message::take_message() else {
        return run(&registry, llms.as_deref(), &args, &kind)
            .await
            .map(|response| ToolResult::ok(response.content));
    };
    let execution =
        autonomy::current_execution().ok_or(autonomy::TaskRunStorageError::InvalidCheckpoint)?;
    let _admission = execution.message_admission.lock().await;
    execution.check_message_slot(&invocation.frame)?;
    let mut frame = *invocation.frame;
    let mut state: MessageState = serde_json::from_value(frame.cursor.clone())?;
    state.response = kind.clone();
    let slots = if matches!(kind, MessageResponse::Handoff) {
        let count = usize::try_from(
            args.get("max_handoffs")
                .and_then(Value::as_u64)
                .unwrap_or(5),
        )
        .ok()
        .and_then(|max| max.checked_add(1))
        .filter(|count| {
            count
                .checked_mul(ids.len())
                .is_some_and(|total| total <= autonomy::MAX_TASK_CHECKPOINT_RECORDS)
        })
        .ok_or_else(|| AgentError::Config("handoff catalogue bound exceeded".into()))?;
        state.work = json!({"targets":ids});
        (0..count)
            .flat_map(|_| ids.iter().cloned())
            .collect::<Vec<_>>()
    } else {
        ids
    };
    frame.children = slots
        .iter()
        .enumerate()
        .map(|(index, id)| {
            let target = registry
                .get(id)
                .ok_or_else(|| AgentError::Config(format!("composition target missing: {id}")))?;
            target.preflight_standalone_autonomy()?;
            target.reject_task_ancestor()?;
            Ok(CompositionChild {
                registry_id: id.clone(),
                runtime_id: target.info().id,
                operation: format!("{}:{index}", frame.child_operation),
            })
        })
        .collect::<Result<_>>()?;
    frame.delegate_input = input;
    state.waiting_for = execution.composition_dependency(&frame)?;
    let deferred = state.waiting_for.is_some();
    frame.cursor = serde_json::to_value(state)?;
    execution
        .retain_delegate_frame(frame.clone(), registry.clone())
        .await?;
    execution.targets.bind_tool(&frame.id, invocation.tool)?;
    execution
        .targets
        .bind_composition(&frame.id, registry.clone(), llms.clone())?;
    drop(_admission);
    if deferred && execution.wait_message_dependency(&frame).await? {
        return Err(AgentError::TaskSuspended(frame.id));
    }
    scope_message_composition(
        frame.runtime_id,
        frame.id,
        Box::pin(run(&registry, llms.as_deref(), &args, &kind)),
    )
    .await
    .map(|response| ToolResult::ok(response.content))
}

/// Existing pipeline/concurrent algorithms own prepared input, actor progression, admission closure, draining and ordered results.
pub(crate) async fn run(
    registry: &AgentRegistry,
    llms: Option<&LLMRegistry>,
    args: &Value,
    kind: &MessageResponse,
) -> Result<AgentResponse> {
    let (input, ids) = members(args, kind)?;
    if matches!(kind, MessageResponse::Pipeline) {
        let templates = args.get("stage_inputs").and_then(Value::as_array);
        let stages = ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                let mut stage = super::types::PipelineStage::id(id);
                if let Some(template) = templates
                    .and_then(|values| values.get(index))
                    .and_then(Value::as_str)
                {
                    stage = stage.with_input(template);
                }
                stage
            })
            .collect::<Vec<_>>();
        let result = super::pipeline(registry, &input, &stages, None, None, None).await?;
        return Ok(AgentResponse::new(json!({"response":result.response.content,"stages":result.stage_outputs.iter().map(|stage| json!({"agent_id":stage.agent_id,"output":stage.output,"duration_ms":stage.duration_ms,"skipped":stage.skipped})).collect::<Vec<_>>()}).to_string()));
    }
    let llms =
        llms.ok_or_else(|| AgentError::Config("concurrent tool lacks captured providers".into()))?;
    if matches!(kind, MessageResponse::Group) {
        let rounds = u32::try_from(args.get("max_rounds").and_then(Value::as_u64).unwrap_or(3))
            .map_err(|_| AgentError::Config("group rounds overflow".into()))?;
        let config = ai_agents_state::GroupChatStateConfig {
            participants: ids
                .into_iter()
                .map(|id| ai_agents_state::ChatParticipant { id, role: None })
                .collect(),
            style: match args.get("style").and_then(Value::as_str) {
                Some("debate") => ai_agents_state::ChatStyle::Debate,
                Some("consensus") => ai_agents_state::ChatStyle::Consensus,
                _ => ai_agents_state::ChatStyle::Brainstorm,
            },
            max_rounds: rounds,
            manager: None,
            termination: ai_agents_state::TerminationConfig {
                method: ai_agents_state::TerminationMethod::MaxRounds,
                max_stall_rounds: 2,
            },
            debate: None,
            maker_checker: None,
            timeout_ms: None,
            input: None,
            context_mode: None,
        };
        let speaker =
            super::role_provider(llms, ai_agents_llm::LLMRole::OrchestrationSpeaker, None)?;
        let consensus =
            super::role_provider(llms, ai_agents_llm::LLMRole::OrchestrationConsensus, None)?;
        let result = super::group_chat_with_llms(
            registry,
            &input,
            &config,
            super::GroupLLMs {
                speaker: speaker.as_deref(),
                consensus: consensus.as_deref(),
            },
            None,
        )
        .await?;
        return Ok(AgentResponse::new(json!({"conclusion":result.response.content,"rounds":result.rounds_completed,"termination":result.termination_reason,
            "transcript":result.transcript.iter().map(|turn| json!({"speaker":turn.speaker,"round":turn.round,"content":turn.content})).collect::<Vec<_>>()}).to_string()));
    }
    if matches!(kind, MessageResponse::Handoff) {
        let initial = args
            .get("initial_agent")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError::Config("missing initial handoff agent".into()))?;
        let max = u32::try_from(
            args.get("max_handoffs")
                .and_then(Value::as_u64)
                .unwrap_or(5),
        )
        .map_err(|_| AgentError::Config("handoff bound overflow".into()))?;
        let provider =
            super::role_provider(llms, ai_agents_llm::LLMRole::OrchestrationHandoff, None)?
                .ok_or_else(|| AgentError::Config("no router LLM configured".into()))?;
        let result = super::handoff(
            registry,
            &input,
            initial,
            &ids,
            max,
            provider.as_ref(),
            None,
        )
        .await?;
        return Ok(AgentResponse::new(json!({"response":result.response.content,"final_agent":result.final_agent,
            "handoffs":result.handoff_chain.iter().map(|event| json!({"from":event.from_agent,"to":event.to_agent,"reason":event.reason})).collect::<Vec<_>>()}).to_string()));
    }
    let providers = super::AggregationProviders::resolve(llms, None)?;
    let aggregation = ai_agents_state::AggregationConfig {
        strategy: match args
            .get("aggregation")
            .and_then(Value::as_str)
            .unwrap_or("all")
        {
            "llm_synthesis" => ai_agents_state::AggregationStrategy::LlmSynthesis,
            "first_wins" => ai_agents_state::AggregationStrategy::FirstWins,
            "voting" => ai_agents_state::AggregationStrategy::Voting,
            _ => ai_agents_state::AggregationStrategy::All,
        },
        synthesizer_llm: if llms.router_roles().is_some() {
            None
        } else {
            Some("router".into())
        },
        synthesizer_prompt: None,
        vote: None,
    };
    let agents = ids
        .into_iter()
        .map(ai_agents_state::ConcurrentAgentRef::Id)
        .collect::<Vec<_>>();
    let result = super::concurrent_with_llms(
        registry,
        &input,
        &agents,
        &aggregation,
        providers.as_refs(),
        None,
        None,
        ai_agents_state::PartialFailureAction::ProceedWithAvailable,
        None,
    )
    .await?;
    Ok(AgentResponse::new(json!({"response":result.response.content,"agent_results":result.agent_results.iter().map(|agent| json!({"agent_id":agent.agent_id,"response":agent.response.as_ref().map(|r|r.content.as_str()),"success":agent.success})).collect::<Vec<_>>()}).to_string()))
}
