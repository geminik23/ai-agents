use std::sync::atomic::{AtomicUsize, Ordering};

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::{ChatMessage, LLMProvider};
use tracing::{debug, info};

use super::types::{RouteResult, RoutingMethod};
use crate::Agent;
use crate::spawner::AgentRegistry;

/// Route input to the best agent from candidates.
pub async fn route(
    registry: &AgentRegistry,
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
    method: RoutingMethod,
    rr_counter: Option<&AtomicUsize>,
) -> Result<RouteResult> {
    if candidates.is_empty() {
        return Err(AgentError::Config("No candidates for routing".into()));
    }

    match method {
        RoutingMethod::Llm => route_via_llm(registry, llm, input, candidates).await,
        RoutingMethod::RoundRobin => {
            let idx = rr_counter
                .map(|c| c.fetch_add(1, Ordering::Relaxed))
                .unwrap_or(0)
                % candidates.len();
            let selected = &candidates[idx];
            let agent = registry.get(selected).ok_or_else(|| {
                AgentError::Other(format!("Agent not found in registry: {}", selected))
            })?;
            let response = agent.chat(input).await?;
            Ok(RouteResult {
                response,
                selected_agent: selected.clone(),
                reason: "round_robin".into(),
                confidence: None,
            })
        }
    }
}

/// Use LLM to pick the best candidate agent for the input.
async fn route_via_llm(
    registry: &AgentRegistry,
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
) -> Result<RouteResult> {
    let agent_list = candidates
        .iter()
        .enumerate()
        .map(|(i, id)| format!("{}. {}", i + 1, id))
        .collect::<Vec<_>>()
        .join("\n");

    let system_msg = format!(
        "You are a routing assistant. Given a user message, select the best agent to handle it.\n\
         Available agents:\n{}\n\n\
         Respond with ONLY the agent ID (exact text) that best matches the user's request.",
        agent_list
    );

    let messages = vec![ChatMessage::system(&system_msg), ChatMessage::user(input)];

    let llm_response = llm
        .complete(&messages, None)
        .await
        .map_err(|e| AgentError::LLM(format!("Routing LLM failed: {}", e)))?;

    let selected_raw = llm_response.content.trim().to_string();

    // Match the LLM output to one of the candidates.
    let selected = candidates
        .iter()
        .find(|c| selected_raw.contains(c.as_str()))
        .cloned()
        .unwrap_or_else(|| candidates[0].clone());

    debug!(selected = %selected, "LLM routed to agent");

    let agent = registry.get(&selected).ok_or_else(|| {
        AgentError::Other(format!("Routed agent not found in registry: {}", selected))
    })?;

    let response = agent.chat(input).await?;

    info!(agent = %selected, "Route completed");

    Ok(RouteResult {
        response,
        selected_agent: selected,
        reason: "LLM selected based on input analysis".into(),
        confidence: None,
    })
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
