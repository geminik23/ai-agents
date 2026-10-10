use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_hooks::AgentHooks;
use ai_agents_llm::{ChatMessage, LLMProvider};
use ai_agents_observability::{ObservationPurpose, with_observation_purpose};
use serde_json::Value;
use tracing::{debug, info};

use super::types::{HandoffEvent, HandoffResult};
use crate::Agent;
use crate::spawner::AgentRegistry;
use crate::turn_context::current_turn_actor_context;

/// Structured decision from the handoff evaluator LLM.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct HandoffDecision {
    action: String,
    confidence: f32,
    reason: String,
}

/// Runs a handoff chain with an acknowledged task cursor for the exact child input, response and routing decision.
/// A parked child resumes its saved operation without repeating earlier turns, decisions or acknowledged hooks.
pub async fn handoff(
    registry: &AgentRegistry,
    input: &str,
    initial_agent: &str,
    available_agents: &[String],
    max_handoffs: u32,
    llm: &dyn LLMProvider,
    hooks: Option<&dyn AgentHooks>,
) -> Result<HandoffResult> {
    let authorized = crate::autonomy::composition::enter_composition_dispatch();
    crate::autonomy::composition::scope_dispatch_authority(
        authorized,
        Box::pin(handoff_dispatch(
            registry,
            input,
            initial_agent,
            available_agents,
            max_handoffs,
            llm,
            hooks,
        )),
    )
    .await
}

/// A nested public handoff cannot consume its caller's already selected cursor or routing decision.
async fn handoff_dispatch(
    registry: &AgentRegistry,
    input: &str,
    initial_agent: &str,
    available_agents: &[String],
    max_handoffs: u32,
    llm: &dyn LLMProvider,
    hooks: Option<&dyn AgentHooks>,
) -> Result<HandoffResult> {
    use crate::autonomy::composition;
    let execution = crate::autonomy::current_execution();
    let frame = execution
        .as_ref()
        .map(|execution| execution.composition_frame())
        .transpose()?
        .flatten();
    let saved = frame.as_ref().and_then(|frame| frame.cursor.get("handoff"));
    let mut cursor = saved
        .map(|value| serde_json::from_value::<TaskHandoffCursor>(value.clone()))
        .transpose()?
        .unwrap_or_else(|| TaskHandoffCursor {
            step: 0,
            current_agent: initial_agent.into(),
            current_input: input.into(),
            chain: Vec::new(),
            final_response: AgentResponse::new(""),
            actor: current_turn_actor_context(),
            response: None,
            decision: None,
            finished: false,
        });
    if cursor.step > max_handoffs.saturating_add(1) {
        return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
    }
    let mut current_agent = cursor.current_agent.clone();
    let mut current_input = cursor.current_input.clone();
    let mut chain = cursor.chain.clone();
    let mut final_response = cursor.final_response.clone();
    let mut actor_context = cursor.actor.clone();
    if saved.is_none() {
        if let Some(h) = hooks {
            h.on_handoff_start(initial_agent).await;
        }
        checkpoint_handoff_cursor(execution.as_deref(), frame.as_ref(), &cursor).await?;
    }

    for step in cursor.step..=max_handoffs {
        if cursor.finished {
            break;
        }
        if frame.as_ref().is_some_and(|frame| {
            frame
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        }) {
            if let Some(execution) = &execution {
                execution.stop("composition_timeout");
            }
            return Err(AgentError::HITLTimeout);
        }
        let agent = registry.get(&current_agent).ok_or_else(|| {
            AgentError::Other(format!("Handoff agent not found: {}", current_agent))
        })?;

        debug!(agent = %current_agent, "Handoff chain executing agent");

        let response = if let Some(response) = &cursor.response {
            response.clone()
        } else {
            let slot = if let Some(frame) = &frame {
                let targets: Vec<String> = serde_json::from_value(frame.cursor["targets"].clone())?;
                let target = targets
                    .iter()
                    .position(|id| id == &current_agent)
                    .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?;
                Some(
                    frame
                        .children
                        .get(step as usize * targets.len() + target)
                        .cloned()
                        .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)?,
                )
            } else {
                None
            };
            let response = composition::run_composition_child(
                composition::current_composition(),
                slot,
                composition::group_response_custody(),
                async {
                    if let Some(context) = actor_context.clone() {
                        agent.chat_with_actor_context(&current_input, context).await
                    } else {
                        agent.chat(&current_input).await
                    }
                },
            )
            .await?;
            cursor.response = Some(response.clone());
            cursor.final_response = response.clone();
            checkpoint_handoff_cursor(execution.as_deref(), frame.as_ref(), &cursor).await?;
            response
        };
        final_response = response.clone();
        if let Some(execution) = &execution
            && let Some(reason) = execution.stop_reason()
        {
            return Err(AgentError::Other(reason));
        }

        // Build the list of agents we could hand off to (excluding current).
        let candidates: Vec<&str> = available_agents
            .iter()
            .filter(|a| a.as_str() != current_agent.as_str())
            .map(|a| a.as_str())
            .collect();

        if candidates.is_empty() {
            debug!("No other agents available for handoff, ending chain");
            break;
        }

        let decision = if let Some(decision) = &cursor.decision {
            decision.clone()
        } else {
            let decision = evaluate_handoff(
                llm,
                &current_agent,
                &candidates,
                &current_input,
                &response.content,
            )
            .await?;
            cursor.decision = Some(decision.clone());
            checkpoint_handoff_cursor(execution.as_deref(), frame.as_ref(), &cursor).await?;
            decision
        };

        if decision.action != "stay" {
            let next_agent = decision.action.clone();

            if available_agents.contains(&next_agent) && registry.contains(&next_agent) {
                info!(
                    from = %current_agent,
                    to = %next_agent,
                    confidence = %decision.confidence,
                    reason = %decision.reason,
                    "Handoff"
                );

                chain.push(HandoffEvent {
                    from_agent: current_agent.clone(),
                    to_agent: next_agent.clone(),
                    reason: decision.reason,
                });

                if let Some(h) = hooks {
                    let last = chain.last().unwrap();
                    h.on_handoff(&last.from_agent, &last.to_agent, &last.reason)
                        .await;
                }

                current_input = format!(
                    "Continuing conversation from another agent. Previous context: {}",
                    response.content
                );
                if let Some(context) = actor_context.as_mut() {
                    *context = context.for_sender(current_agent.clone());
                }
                current_agent = next_agent;
                cursor.step = step + 1;
                cursor.current_agent = current_agent.clone();
                cursor.current_input = current_input.clone();
                cursor.chain = chain.clone();
                cursor.actor = actor_context.clone();
                cursor.response = None;
                cursor.decision = None;
                checkpoint_handoff_cursor(execution.as_deref(), frame.as_ref(), &cursor).await?;
                continue;
            }

            debug!(
                candidate = %next_agent,
                "LLM suggested handoff to unknown or unavailable agent, staying"
            );
        }

        // No handoff needed or candidate invalid.
        break;
    }

    cursor.finished = true;
    cursor.final_response = final_response.clone();
    cursor.current_agent = current_agent.clone();
    cursor.chain = chain.clone();
    checkpoint_handoff_cursor(execution.as_deref(), frame.as_ref(), &cursor).await?;
    info!(
        initial = %initial_agent,
        final_agent = %current_agent,
        handoffs = chain.len(),
        "Handoff chain completed"
    );

    Ok(HandoffResult {
        response: final_response,
        handoff_chain: chain,
        final_agent: current_agent,
    })
}

/// Exact handoff coordinates preserve settled child responses and model decisions separately.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskHandoffCursor {
    step: u32,
    current_agent: String,
    current_input: String,
    chain: Vec<HandoffEvent>,
    final_response: AgentResponse,
    actor: Option<crate::TurnActorContext>,
    response: Option<AgentResponse>,
    decision: Option<HandoffDecision>,
    finished: bool,
}

/// Cursor acknowledgement shares the run ledger; failure cannot publish a new routing location or release its owner.
async fn checkpoint_handoff_cursor(
    execution: Option<&crate::autonomy::RunExecution>,
    frame: Option<&crate::autonomy::DelegateFrame>,
    cursor: &TaskHandoffCursor,
) -> Result<()> {
    if let (Some(execution), Some(frame)) = (execution, frame) {
        let mut saved = frame.cursor.clone();
        saved["handoff"] = serde_json::to_value(cursor)?;
        Box::pin(execution.checkpoint_composition_cursor(&frame.runtime_id, saved)).await?;
    }
    Ok(())
}

/// Ask the LLM for a structured JSON handoff decision.
async fn evaluate_handoff(
    llm: &dyn LLMProvider,
    current_agent: &str,
    candidates: &[&str],
    user_input: &str,
    agent_response: &str,
) -> Result<HandoffDecision> {
    let agent_list = candidates.join(", ");

    let messages = vec![
        ChatMessage::system(format!(
            "You are evaluating whether a conversation should be handed off to a different specialist.\n\
             Current agent: {current_agent}\n\
             Available agents: {agent_list}\n\n\
             Respond with ONLY a JSON object in this exact format:\n\
             {{\"action\": \"stay\" or \"<agent_id>\", \"confidence\": 0.0 to 1.0, \"reason\": \"brief explanation\"}}\n\n\
             Set action to \"stay\" if the current agent is handling the conversation well.\n\
             Set action to one of the available agent IDs if a handoff is needed."
        )),
        ChatMessage::user(format!(
            "User message: {user_input}\n\nAgent response: {agent_response}"
        )),
    ];

    let response = with_observation_purpose(
        ObservationPurpose::OrchestrationRouting,
        ai_agents_llm::managed_completion(llm, &messages, None),
    )
    .await
    .map_err(|e| AgentError::LLM(format!("Handoff decision failed: {}", e)))?;

    let raw = response.content.trim();

    // Primary path: structured JSON parse.
    if let Some(decision) = try_parse_handoff_json(raw, candidates) {
        return Ok(decision);
    }

    // Fallback: fuzzy text matching.
    debug!("JSON parse failed, falling back to fuzzy parse");
    Ok(fuzzy_parse_handoff(raw, candidates))
}

/// Try to parse a structured JSON handoff decision from the LLM output.
/// Returns None if the JSON is missing or references an unknown agent.
fn try_parse_handoff_json(raw: &str, candidates: &[&str]) -> Option<HandoffDecision> {
    let value = extract_json_value(raw)?;

    let action_raw = value.get("action")?.as_str()?;
    let confidence = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let reason = value
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("no reason provided")
        .to_string();

    let action_lower = action_raw.trim().to_lowercase();

    if action_lower == "stay" {
        return Some(HandoffDecision {
            action: "stay".to_string(),
            confidence,
            reason,
        });
    }

    // Case-insensitive match against known candidates.
    let matched = candidates
        .iter()
        .find(|c| c.to_lowercase() == action_lower)?;

    Some(HandoffDecision {
        action: matched.to_string(),
        confidence,
        reason,
    })
}

/// Extract a JSON value from potentially noisy LLM output.
/// Tries raw parse, markdown code block extraction, then brace extraction.
fn extract_json_value(raw: &str) -> Option<Value> {
    // Try raw JSON parse first.
    if let Ok(v) = serde_json::from_str::<Value>(raw)
        && v.is_object()
    {
        return Some(v);
    }

    // Try extracting from a markdown code block.
    if let Some(start) = raw.find("```") {
        let after_fence = &raw[start + 3..];
        // Skip optional language tag on the same line.
        let body_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after_fence[body_start..];
        if let Some(end) = body.find("```") {
            let block = body[..end].trim();
            if let Ok(v) = serde_json::from_str::<Value>(block)
                && v.is_object()
            {
                return Some(v);
            }
        }
    }

    // Try first '{' to last '}'.
    let open = raw.find('{')?;
    let close = raw.rfind('}')?;
    if close > open {
        let slice = &raw[open..=close];
        if let Ok(v) = serde_json::from_str::<Value>(slice)
            && v.is_object()
        {
            return Some(v);
        }
    }

    None
}

/// Fuzzy fallback parser for when JSON parsing fails entirely.
/// Matches agent names in the raw text or defaults to "stay".
fn fuzzy_parse_handoff(raw: &str, candidates: &[&str]) -> HandoffDecision {
    let lower = raw.to_lowercase();
    let trimmed = lower.trim_start();

    // Explicit stay signals.
    if trimmed.starts_with("stay")
        || trimmed.starts_with("no handoff")
        || trimmed.starts_with("no,")
    {
        return HandoffDecision {
            action: "stay".to_string(),
            confidence: 0.5,
            reason: "fuzzy: detected stay signal".to_string(),
        };
    }

    // Sort candidates longest-first to avoid substring collisions.
    let mut sorted: Vec<&str> = candidates.to_vec();
    sorted.sort_by_key(|candidate| std::cmp::Reverse(candidate.len()));

    for candidate in &sorted {
        if lower.contains(&candidate.to_lowercase()) {
            return HandoffDecision {
                action: candidate.to_string(),
                confidence: 0.3,
                reason: "fuzzy: matched agent name in text".to_string(),
            };
        }
    }

    // Nothing matched, default to stay.
    HandoffDecision {
        action: "stay".to_string(),
        confidence: 0.1,
        reason: "fuzzy: no match found, defaulting to stay".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_parse_handoff_json_valid() {
        let raw = r#"{"action": "billing_agent", "confidence": 0.9, "reason": "user asked about billing"}"#;
        let candidates = &["billing_agent", "support_agent"];
        let decision = try_parse_handoff_json(raw, candidates).unwrap();
        assert_eq!(decision.action, "billing_agent");
        assert!((decision.confidence - 0.9).abs() < 0.01);
        assert_eq!(decision.reason, "user asked about billing");
    }

    #[test]
    fn test_try_parse_handoff_json_markdown_block() {
        let raw = "Here is my decision:\n```json\n{\"action\": \"stay\", \"confidence\": 0.8, \"reason\": \"handling well\"}\n```";
        let candidates = &["billing_agent"];
        let decision = try_parse_handoff_json(raw, candidates).unwrap();
        assert_eq!(decision.action, "stay");
        assert!((decision.confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_try_parse_handoff_json_with_preamble() {
        let raw = "I think we should hand off. {\"action\": \"support_agent\", \"confidence\": 0.7, \"reason\": \"needs help\"}";
        let candidates = &["support_agent", "billing_agent"];
        let decision = try_parse_handoff_json(raw, candidates).unwrap();
        assert_eq!(decision.action, "support_agent");
        assert!((decision.confidence - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_try_parse_handoff_json_case_insensitive() {
        let raw =
            r#"{"action": "Billing_Agent", "confidence": 0.85, "reason": "billing question"}"#;
        let candidates = &["billing_agent", "support_agent"];
        let decision = try_parse_handoff_json(raw, candidates).unwrap();
        assert_eq!(decision.action, "billing_agent");
    }

    #[test]
    fn test_try_parse_handoff_json_unknown_agent() {
        let raw = r#"{"action": "nonexistent_agent", "confidence": 0.9, "reason": "some reason"}"#;
        let candidates = &["billing_agent", "support_agent"];
        let result = try_parse_handoff_json(raw, candidates);
        assert!(result.is_none());
    }

    #[test]
    fn test_fuzzy_parse_stay() {
        let candidates = &["billing_agent", "support_agent"];

        let d1 = fuzzy_parse_handoff("Stay with current agent", candidates);
        assert_eq!(d1.action, "stay");

        let d2 = fuzzy_parse_handoff("No handoff is needed here", candidates);
        assert_eq!(d2.action, "stay");

        let d3 = fuzzy_parse_handoff("No, the agent is doing fine", candidates);
        assert_eq!(d3.action, "stay");
    }

    #[test]
    fn test_fuzzy_parse_agent_match() {
        let candidates = &["billing_agent", "support_agent"];
        let decision = fuzzy_parse_handoff(
            "I think we should transfer to billing_agent for this",
            candidates,
        );
        assert_eq!(decision.action, "billing_agent");
    }

    #[test]
    fn test_fuzzy_parse_longest_match_first() {
        let candidates = &["agent", "super_agent"];
        let decision = fuzzy_parse_handoff("Route to super_agent please", candidates);
        assert_eq!(decision.action, "super_agent");
    }

    #[test]
    fn test_fuzzy_parse_no_match() {
        let candidates = &["billing_agent", "support_agent"];
        let decision = fuzzy_parse_handoff("Something completely unrelated", candidates);
        assert_eq!(decision.action, "stay");
        assert!(decision.confidence < 0.2);
    }

    #[test]
    fn test_extract_json_value_raw() {
        let raw = r#"{"action": "stay", "confidence": 0.5, "reason": "ok"}"#;
        let value = extract_json_value(raw).unwrap();
        assert_eq!(value["action"], "stay");
    }

    #[test]
    fn test_extract_json_value_no_json() {
        let raw = "This is just plain text with no JSON at all.";
        assert!(extract_json_value(raw).is_none());
    }
}
