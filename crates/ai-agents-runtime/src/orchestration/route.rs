use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::{ChatMessage, LLMProvider};
use ai_agents_observability::{ObservationPurpose, with_observation_purpose};
use tracing::{debug, info};

use super::types::{RouteResult, RoutingMethod};
use crate::Agent;
use crate::spawner::AgentRegistry;
use crate::turn_context::current_turn_actor_context;

/// Routes ordinary input using the existing selection behavior and actor-scoped child entry.
pub async fn route(
    registry: &AgentRegistry,
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
    method: RoutingMethod,
    rr_counter: Option<&AtomicUsize>,
) -> Result<RouteResult> {
    let (selected, reason) = select_route(llm, input, candidates, method, rr_counter).await?;
    let agent = registry.get(&selected).ok_or_else(|| {
        AgentError::Other(format!("Routed agent not found in registry: {selected}"))
    })?;
    let response = if let Some(context) = current_turn_actor_context() {
        agent.chat_with_actor_context(input, context).await?
    } else {
        agent.chat(input).await?
    };
    info!(agent = %selected, "Route completed");
    Ok(RouteResult {
        response,
        selected_agent: selected,
        reason,
        confidence: None,
    })
}

/// Task selection completes once before an exact parent-child frame is installed.
/// Provider callbacks cannot consume the dispatch capability while the selector is running.
pub(crate) async fn route_task(
    registry: &Arc<AgentRegistry>,
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
    method: RoutingMethod,
    rr_counter: Option<&AtomicUsize>,
) -> Result<RouteResult> {
    let invocation = crate::autonomy::message::take_message();
    let (selected, reason) = crate::autonomy::message::scope_message(
        None,
        Box::pin(select_route(llm, input, candidates, method, rr_counter)),
    )
    .await?;

    let actor = current_turn_actor_context();
    let response = crate::autonomy::message::scope_message(
        invocation,
        Box::pin(registry.route_task_message(&selected, input, actor, &reason)),
    )
    .await?;
    Ok(RouteResult {
        response,
        selected_agent: selected,
        reason,
        confidence: None,
    })
}

/// Selection shares ordinary semantics, including the legacy candidate fallback, without invoking a child.
async fn select_route(
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
    method: RoutingMethod,
    rr_counter: Option<&AtomicUsize>,
) -> Result<(String, String)> {
    if candidates.is_empty() {
        return Err(AgentError::Config("No candidates for routing".into()));
    }
    if matches!(method, RoutingMethod::RoundRobin) {
        let index = rr_counter
            .map(|counter| counter.fetch_add(1, Ordering::Relaxed))
            .unwrap_or(0)
            % candidates.len();
        return Ok((candidates[index].clone(), "round_robin".into()));
    }
    let agent_list = candidates
        .iter()
        .enumerate()
        .map(|(index, id)| format!("{}. {}", index + 1, id))
        .collect::<Vec<_>>()
        .join("\n");
    let system = format!(
        "You are a routing assistant. Given a user message, select the best agent to handle it.\n\
         Available agents:\n{}\n\n\
         Respond with ONLY the agent ID (exact text) that best matches the user's request.",
        agent_list
    );
    let messages = vec![ChatMessage::system(&system), ChatMessage::user(input)];
    let response = with_observation_purpose(
        ObservationPurpose::OrchestrationRouting,
        Box::pin(ai_agents_llm::managed_completion(llm, &messages, None)),
    )
    .await
    .map_err(|error| AgentError::LLM(format!("Routing LLM failed: {error}")))?;
    let selected_raw = response.content.trim();
    let selected = candidates
        .iter()
        .find(|candidate| selected_raw.contains(candidate.as_str()))
        .cloned()
        .unwrap_or_else(|| candidates[0].clone());
    debug!(selected = %selected, "LLM routed to agent");
    Ok((selected, "LLM selected based on input analysis".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_round_robin_cycles_through_candidates() {
        let candidates: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let counter = AtomicUsize::new(0);

        let idx0 = counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
        assert_eq!(candidates[idx0], "a");

        let idx1 = counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
        assert_eq!(candidates[idx1], "b");

        let idx2 = counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
        assert_eq!(candidates[idx2], "c");

        let idx3 = counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
        assert_eq!(candidates[idx3], "a");
    }

    #[test]
    fn test_round_robin_no_counter_defaults_to_first() {
        let candidates: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let idx = None::<&AtomicUsize>
            .map(|c| c.fetch_add(1, Ordering::Relaxed))
            .unwrap_or(0)
            % candidates.len();
        assert_eq!(candidates[idx], "a");
    }
}
