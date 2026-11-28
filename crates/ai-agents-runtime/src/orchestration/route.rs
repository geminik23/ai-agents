use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ai_agents_core::{AgentError, ChoiceCandidate, ChoiceRequest, Result};
use ai_agents_llm::{ChatMessage, LLMProvider};
use ai_agents_observability::{ObservationPurpose, with_observation_purpose};
use tracing::{debug, info};

use super::types::{RouteResult, RoutingMethod};
use crate::Agent;
use crate::spawner::AgentRegistry;
use crate::turn_context::current_turn_actor_context;

/// Routes input using the configured selector before actor-scoped child entry.
/// Inherited task execution pins the selected registration before waiting for the child's root gate.
pub async fn route(
    registry: &AgentRegistry,
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
    method: RoutingMethod,
    rr_counter: Option<&AtomicUsize>,
) -> Result<RouteResult> {
    let (selected, reason) =
        select_route(registry, llm, input, candidates, method, rr_counter).await?;
    let agent = registry.get(&selected).ok_or_else(|| {
        AgentError::Other(format!("Routed agent not found in registry: {selected}"))
    })?;
    let _target_pin = registry.capture_task_target(&selected, &agent)?;
    // Retain this selected route while a hosted automatic child awaits its explicit host response.
    let response = crate::autonomy::automatic::scope_composing(async {
        if let Some(context) = current_turn_actor_context() {
            agent.chat_with_actor_context(input, context).await
        } else {
            agent.chat(input).await
        }
    })
    .await?;
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
        Box::pin(select_route(
            registry, llm, input, candidates, method, rr_counter,
        )),
    )
    .await?;

    let actor = current_turn_actor_context();
    let response = crate::autonomy::message::scope_message(
        invocation,
        registry.route_task_message(&selected, input, actor, &reason),
    )
    .await?;
    Ok(RouteResult {
        response,
        selected_agent: selected,
        reason,
        confidence: None,
    })
}

/// Selects a label without invoking a child and keeps the legacy text fallback local to its original mode.
async fn select_route(
    registry: &AgentRegistry,
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
    method: RoutingMethod,
    rr_counter: Option<&AtomicUsize>,
) -> Result<(String, String)> {
    if candidates.is_empty() {
        return Err(AgentError::Config("No candidates for routing".into()));
    }
    if llm.uses_choice_for_routing() {
        if matches!(method, RoutingMethod::RoundRobin) {
            return Err(AgentError::Config(
                "round robin cannot override a structured route".into(),
            ));
        }
        return select_structured_route(registry, llm, input, candidates).await;
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

// Validate candidate membership before paying for selection; the existing dispatch paths bind and check the actual child afterward.
async fn select_structured_route(
    registry: &AgentRegistry,
    llm: &dyn LLMProvider,
    input: &str,
    candidates: &[String],
) -> Result<(String, String)> {
    let mut unique = std::collections::HashSet::with_capacity(candidates.len());
    for id in candidates {
        if id.trim().is_empty() || !unique.insert(id.as_str()) || registry.get(id).is_none() {
            return Err(AgentError::Config(
                "structured routing requires distinct registered candidate IDs".into(),
            ));
        }
    }
    if let [only] = candidates {
        return Ok((only.clone(), "Only eligible candidate".into()));
    }
    let options: Vec<_> = candidates
        .iter()
        .map(|id| ChoiceCandidate {
            id,
            description: None,
        })
        .collect();
    let decision = with_observation_purpose(
        ObservationPurpose::OrchestrationRouting,
        ai_agents_llm::managed_choice(
            llm,
            &ChoiceRequest {
                input,
                question: "Which candidate should handle this request?",
                candidates: &options,
            },
        ),
    )
    .await
    .map_err(|error| AgentError::LLM(format!("Routing decision failed: {error}")))?;
    let selected = decision
        .selected
        .ok_or_else(|| AgentError::LLM("Routing decision refused".into()))?;
    if !unique.contains(selected.as_str()) {
        return Err(AgentError::LLM(
            "Routing decision returned an unknown candidate".into(),
        ));
    }
    Ok((
        selected,
        "Structured decision selected an exact candidate".into(),
    ))
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
