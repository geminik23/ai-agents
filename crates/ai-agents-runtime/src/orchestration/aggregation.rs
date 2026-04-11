use std::collections::HashMap;

use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_llm::{ChatMessage, LLMProvider};
use ai_agents_state::{
    AggregationConfig, AggregationStrategy, TiebreakerStrategy, VoteConfig, VoteMethod,
};
use rand::seq::SliceRandom;
use tracing::debug;

use super::types::AgentResult;

/// Aggregate multiple agent results into a single response.
pub async fn aggregate(
    results: &[AgentResult],
    config: &AggregationConfig,
    llm: Option<&dyn LLMProvider>,
    agent_weights: &HashMap<String, f64>,
) -> Result<AgentResponse> {
    let successful: Vec<&AgentResult> = results.iter().filter(|r| r.success).collect();

    if successful.is_empty() {
        return Err(AgentError::Other("All agents failed".into()));
    }

    match config.strategy {
        AggregationStrategy::FirstWins => {
            debug!("Aggregation strategy: first_wins");
            let first = &successful[0];
            Ok(first
                .response
                .clone()
                .unwrap_or_else(|| AgentResponse::new("")))
        }
        AggregationStrategy::All => {
            debug!(count = successful.len(), "Aggregation strategy: all");
            let content = successful
                .iter()
                .filter_map(|r| {
                    r.response
                        .as_ref()
                        .map(|resp| format!("**{}**:\n{}", r.agent_id, resp.content))
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            Ok(AgentResponse::new(content))
        }
        AggregationStrategy::LlmSynthesis => {
            debug!("Aggregation strategy: llm_synthesis");
            let llm = llm.ok_or_else(|| {
                AgentError::Config("LLM required for llm_synthesis aggregation".into())
            })?;
            synthesize_with_llm(llm, &successful, config.synthesizer_prompt.as_deref()).await
        }
        AggregationStrategy::Voting => {
            debug!("Aggregation strategy: voting");
            let llm = llm
                .ok_or_else(|| AgentError::Config("LLM required for voting aggregation".into()))?;
            let vote_config = config.vote.as_ref();
            vote_with_llm(llm, results, vote_config, agent_weights).await
        }
    }
}

/// Use an LLM to synthesize multiple responses into one coherent answer.
async fn synthesize_with_llm(
    llm: &dyn LLMProvider,
    results: &[&AgentResult],
    custom_prompt: Option<&str>,
) -> Result<AgentResponse> {
    let agent_responses = results
        .iter()
        .filter_map(|r| {
            r.response
                .as_ref()
                .map(|resp| format!("[Agent: {}]\n{}", r.agent_id, resp.content))
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let system = custom_prompt.unwrap_or(
        "You are a synthesis assistant. \
         Multiple agents have provided their analysis. \
         Combine their insights into a single coherent response. \
         Include the key points from each perspective.",
    );

    let messages = vec![
        ChatMessage::system(system),
        ChatMessage::user(&format!(
            "Synthesize these responses:\n\n{}",
            agent_responses
        )),
    ];

    let response = llm
        .complete(&messages, None)
        .await
        .map_err(|e| AgentError::LLM(format!("Synthesis LLM failed: {}", e)))?;

    Ok(AgentResponse::new(response.content))
}

/// Extract votes from agent responses via LLM and tally them.
async fn vote_with_llm(
    llm: &dyn LLMProvider,
    results: &[AgentResult],
    vote_config: Option<&VoteConfig>,
    agent_weights: &HashMap<String, f64>,
) -> Result<AgentResponse> {
    let vote_prompt = vote_config
        .and_then(|v| v.vote_prompt.as_deref())
        .unwrap_or(
            "Extract the main recommendation or decision from this response as a single short phrase.",
        );

    let method = vote_config.map(|v| v.method.clone()).unwrap_or_default();

    let tiebreaker = vote_config
        .map(|v| v.tiebreaker.clone())
        .unwrap_or_default();

    // Phase 1: Extract votes from each agent response.
    let mut votes: Vec<(String, String, f64)> = Vec::new();

    for result in results.iter().filter(|r| r.success) {
        if let Some(ref resp) = result.response {
            let messages = vec![
                ChatMessage::system(vote_prompt),
                ChatMessage::user(&resp.content),
            ];

            let extraction = llm
                .complete(&messages, None)
                .await
                .map_err(|e| AgentError::LLM(format!("Vote extraction failed: {}", e)))?;

            let weight = match method {
                VoteMethod::Weighted => agent_weights.get(&result.agent_id).copied().unwrap_or(1.0),
                _ => 1.0,
            };

            votes.push((
                result.agent_id.clone(),
                extraction.content.trim().to_string(),
                weight,
            ));
        }
    }

    if votes.is_empty() {
        return Err(AgentError::Other("No votes extracted".into()));
    }

    // Phase 2: Check unanimous agreement if required.
    if matches!(method, VoteMethod::Unanimous) {
        let first_vote = votes[0].1.to_lowercase();
        let all_agree = votes.iter().all(|(_, v, _)| v.to_lowercase() == first_vote);
        if !all_agree {
            let vote_lines = votes
                .iter()
                .map(|(id, v, _)| format!("- {}: {}", id, v))
                .collect::<Vec<_>>()
                .join("\n");
            return Err(AgentError::Other(format!(
                "Unanimous vote failed: agents did not agree\n\nVotes:\n{}",
                vote_lines
            )));
        }
        return Ok(AgentResponse::new(format!(
            "Unanimous decision: {}",
            votes[0].1
        )));
    }

    // Phase 3: Tally votes.
    let mut tally: HashMap<String, f64> = HashMap::new();
    for (_, vote, weight) in &votes {
        *tally.entry(vote.clone()).or_default() += weight;
    }

    // Phase 4: Find winner with tiebreaker.
    let max_score = tally.values().cloned().fold(f64::NEG_INFINITY, f64::max);

    let tied: Vec<String> = tally
        .iter()
        .filter(|(_, v)| (**v - max_score).abs() < f64::EPSILON)
        .map(|(k, _)| k.clone())
        .collect();

    let winner = if tied.len() == 1 {
        tied[0].clone()
    } else {
        match tiebreaker {
            TiebreakerStrategy::First => {
                // Pick the choice cast by the earliest agent in results order.
                votes
                    .iter()
                    .find(|(_, choice, _)| tied.contains(choice))
                    .map(|(_, choice, _)| choice.clone())
                    .unwrap_or_else(|| tied[0].clone())
            }
            TiebreakerStrategy::Random => {
                let mut rng = rand::thread_rng();
                tied.choose(&mut rng)
                    .cloned()
                    .unwrap_or_else(|| tied[0].clone())
            }
            TiebreakerStrategy::RouterDecides => resolve_tie_with_llm(llm, &tied).await?,
        }
    };

    let vote_lines = votes
        .iter()
        .map(|(id, v, _)| format!("- {}: {}", id, v))
        .collect::<Vec<_>>()
        .join("\n");

    let summary = format!("Vote result: {}\n\nVotes:\n{}", winner, vote_lines);

    debug!(winner = %winner, total_votes = votes.len(), "Vote aggregation complete");

    Ok(AgentResponse::new(summary))
}

/// Ask the LLM to break a vote tie.
async fn resolve_tie_with_llm(llm: &dyn LLMProvider, tied_choices: &[String]) -> Result<String> {
    let choices_list = tied_choices
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, c))
        .collect::<Vec<_>>()
        .join("\n");

    let messages = vec![
        ChatMessage::system(
            "You are a tiebreaker. Multiple options received equal votes. \
             Pick the single best option. Respond with ONLY the option text.",
        ),
        ChatMessage::user(&format!(
            "These options are tied:\n{}\n\nPick one.",
            choices_list
        )),
    ];

    let response = llm
        .complete(&messages, None)
        .await
        .map_err(|e| AgentError::LLM(format!("Tiebreaker LLM failed: {}", e)))?;

    let raw = response.content.trim().to_string();

    // Try to match the LLM output to one of the tied choices.
    for choice in tied_choices {
        if raw.contains(choice.as_str()) || choice.contains(raw.as_str()) {
            return Ok(choice.clone());
        }
    }

    // Fallback: return the first tied choice.
    Ok(tied_choices[0].clone())
}
