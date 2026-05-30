use std::collections::HashMap;

use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_llm::{ChatMessage, LLMProvider};
use ai_agents_observability::{ObservationPurpose, with_observation_purpose};
use ai_agents_state::{
    AggregationConfig, AggregationStrategy, TiebreakerStrategy, VoteConfig, VoteMethod,
};
use futures::future::join_all;
use rand::seq::SliceRandom;
use tracing::debug;

use super::types::AgentResult;

/// Aggregate multiple agent results into a single response.
pub async fn aggregate(
    results: &[AgentResult],
    config: &AggregationConfig,
    llm: Option<&dyn LLMProvider>,
    agent_weights: &HashMap<String, f64>,
    vote_parallelism: Option<usize>,
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
            vote_with_llm(llm, results, vote_config, agent_weights, vote_parallelism).await
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

    let response = with_observation_purpose(
        ObservationPurpose::OrchestrationAggregation,
        llm.complete(&messages, None),
    )
    .await
    .map_err(|e| AgentError::LLM(format!("Synthesis LLM failed: {}", e)))?;

    Ok(AgentResponse::new(response.content))
}

async fn extract_vote(
    llm: &dyn LLMProvider,
    result: &AgentResult,
    vote_prompt: &str,
    method: &VoteMethod,
    agent_weights: &HashMap<String, f64>,
) -> Result<(usize, String, String, f64)> {
    let response = result
        .response
        .as_ref()
        .ok_or_else(|| AgentError::Other(format!("Agent {} has no response", result.agent_id)))?;
    let messages = vec![
        ChatMessage::system(vote_prompt),
        ChatMessage::user(&response.content),
    ];

    let extraction = with_observation_purpose(
        ObservationPurpose::OrchestrationAggregation,
        llm.complete(&messages, None),
    )
    .await
    .map_err(|e| AgentError::LLM(format!("Vote extraction failed: {}", e)))?;

    let weight = match method {
        VoteMethod::Weighted => agent_weights.get(&result.agent_id).copied().unwrap_or(1.0),
        _ => 1.0,
    };

    Ok((
        result.agent_index,
        result.agent_id.clone(),
        extraction.content.trim().to_string(),
        weight,
    ))
}

/// Extract votes from agent responses via LLM and tally them.
async fn vote_with_llm(
    llm: &dyn LLMProvider,
    results: &[AgentResult],
    vote_config: Option<&VoteConfig>,
    agent_weights: &HashMap<String, f64>,
    vote_parallelism: Option<usize>,
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

    let successful: Vec<&AgentResult> = results
        .iter()
        .filter(|result| result.success && result.response.is_some())
        .collect();

    let mut votes: Vec<(usize, String, String, f64)> = Vec::new();
    if let Some(limit) = vote_parallelism.filter(|limit| *limit > 1) {
        for chunk in successful.chunks(limit) {
            let extraction_futures = chunk.iter().copied().map(|result| {
                let method_for_task = method.clone();
                async move {
                    let response = result.response.as_ref().ok_or_else(|| {
                        AgentError::Other(format!("Agent {} has no response", result.agent_id))
                    })?;
                    let messages = vec![
                        ChatMessage::system(vote_prompt),
                        ChatMessage::user(&response.content),
                    ];

                    let extraction = with_observation_purpose(
                        ObservationPurpose::OrchestrationAggregation,
                        llm.complete(&messages, None),
                    )
                    .await
                    .map_err(|e| AgentError::LLM(format!("Vote extraction failed: {}", e)))?;

                    let weight = match method_for_task {
                        VoteMethod::Weighted => {
                            agent_weights.get(&result.agent_id).copied().unwrap_or(1.0)
                        }
                        _ => 1.0,
                    };

                    Ok::<(usize, String, String, f64), AgentError>((
                        result.agent_index,
                        result.agent_id.clone(),
                        extraction.content.trim().to_string(),
                        weight,
                    ))
                }
            });
            for extraction in join_all(extraction_futures).await {
                votes.push(extraction?);
            }
        }
    } else {
        for result in successful {
            votes.push(extract_vote(llm, result, vote_prompt, &method, agent_weights).await?);
        }
    }
    votes.sort_by_key(|(agent_index, _, _, _)| *agent_index);

    if votes.is_empty() {
        return Err(AgentError::Other("No votes extracted".into()));
    }

    // Phase 2: Check unanimous agreement if required.
    if matches!(method, VoteMethod::Unanimous) {
        let first_vote = votes[0].2.to_lowercase();
        let all_agree = votes
            .iter()
            .all(|(_, _, v, _)| v.to_lowercase() == first_vote);
        if !all_agree {
            let vote_lines = votes
                .iter()
                .map(|(_, id, v, _)| format!("- {}: {}", id, v))
                .collect::<Vec<_>>()
                .join("\n");
            return Err(AgentError::Other(format!(
                "Unanimous vote failed: agents did not agree\n\nVotes:\n{}",
                vote_lines
            )));
        }
        return Ok(AgentResponse::new(format!(
            "Unanimous decision: {}",
            votes[0].2
        )));
    }

    // Phase 3: Tally votes.
    let mut tally: HashMap<String, f64> = HashMap::new();
    for (_, _, vote, weight) in &votes {
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
                // Pick the choice cast by the earliest agent in declaration order.
                votes
                    .iter()
                    .find(|(_, _, choice, _)| tied.contains(choice))
                    .map(|(_, _, choice, _)| choice.clone())
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
        .map(|(_, id, v, _)| format!("- {}: {}", id, v))
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

    let response = with_observation_purpose(
        ObservationPurpose::OrchestrationAggregation,
        llm.complete(&messages, None),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::AgentResponse;
    use ai_agents_llm::mock::MockLLMProvider;
    use ai_agents_state::AggregationStrategy;
    use std::time::Instant;

    fn agent_result(index: usize, id: &str, content: &str) -> AgentResult {
        AgentResult {
            agent_index: index,
            agent_id: id.to_string(),
            response: Some(AgentResponse::new(content)),
            duration_ms: 0,
            success: true,
            error: None,
        }
    }

    #[tokio::test]
    async fn vote_extraction_is_serial_without_parallelism() {
        let mut llm = MockLLMProvider::new("votes");
        llm.set_responses(
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            false,
        );
        llm.set_latency(25);
        let results = vec![
            agent_result(0, "a", "first"),
            agent_result(1, "b", "second"),
            agent_result(2, "c", "third"),
        ];
        let config = AggregationConfig {
            strategy: AggregationStrategy::Voting,
            synthesizer_llm: None,
            synthesizer_prompt: None,
            vote: None,
        };
        let started = Instant::now();
        let _ = aggregate(&results, &config, Some(&llm), &HashMap::new(), None)
            .await
            .unwrap();
        assert!(started.elapsed() >= std::time::Duration::from_millis(60));
        assert_eq!(llm.call_count(), 3);
    }

    #[tokio::test]
    async fn vote_extraction_uses_bounded_parallelism_when_enabled() {
        let mut llm = MockLLMProvider::new("votes");
        llm.set_responses(
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            false,
        );
        llm.set_latency(50);
        let results = vec![
            agent_result(0, "a", "first"),
            agent_result(1, "b", "second"),
            agent_result(2, "c", "third"),
        ];
        let config = AggregationConfig {
            strategy: AggregationStrategy::Voting,
            synthesizer_llm: None,
            synthesizer_prompt: None,
            vote: None,
        };
        let started = Instant::now();
        let _ = aggregate(&results, &config, Some(&llm), &HashMap::new(), Some(2))
            .await
            .unwrap();
        assert!(started.elapsed() < std::time::Duration::from_millis(140));
        assert_eq!(llm.call_count(), 3);
    }

    #[tokio::test]
    async fn vote_tiebreaker_first_uses_declaration_order() {
        let mut llm = MockLLMProvider::new("votes");
        llm.set_responses(vec!["B".to_string(), "A".to_string()], false);
        let results = vec![
            agent_result(1, "b", "second"),
            agent_result(0, "a", "first"),
        ];
        let config = AggregationConfig {
            strategy: AggregationStrategy::Voting,
            synthesizer_llm: None,
            synthesizer_prompt: None,
            vote: Some(VoteConfig {
                method: VoteMethod::Majority,
                tiebreaker: TiebreakerStrategy::First,
                vote_prompt: None,
            }),
        };

        let response = aggregate(&results, &config, Some(&llm), &HashMap::new(), None)
            .await
            .unwrap();

        assert!(response.content.starts_with("Vote result: A"));
    }
}
