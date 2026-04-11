use std::collections::HashMap;
use std::time::Instant;

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::LLMProvider;
use ai_agents_state::{AggregationConfig, ConcurrentAgentRef, PartialFailureAction};
use tokio::task::JoinSet;
use tracing::{info, warn};

use super::aggregation;
use super::types::{AgentResult, ConcurrentResult};
use crate::Agent;
use crate::spawner::AgentRegistry;

/// Run multiple agents in parallel and aggregate results.
pub async fn concurrent(
    registry: &AgentRegistry,
    input: &str,
    agents: &[ConcurrentAgentRef],
    aggregation_config: &AggregationConfig,
    llm: Option<&dyn LLMProvider>,
    min_required: Option<usize>,
    timeout_ms: Option<u64>,
    on_partial_failure: PartialFailureAction,
) -> Result<ConcurrentResult> {
    if agents.is_empty() {
        return Err(AgentError::Config(
            "No agents for concurrent execution".into(),
        ));
    }

    let start = Instant::now();
    let mut join_set = JoinSet::new();

    for agent_ref in agents {
        let agent_id = agent_ref.id().to_string();
        let agent = registry.get(&agent_id).ok_or_else(|| {
            AgentError::Other(format!("Agent not found in registry: {}", agent_id))
        })?;
        let input_owned = input.to_string();
        let timeout = timeout_ms;

        join_set.spawn(async move {
            let agent_start = Instant::now();
            let result = if let Some(t) = timeout {
                match tokio::time::timeout(
                    tokio::time::Duration::from_millis(t),
                    agent.chat(&input_owned),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_) => Err(AgentError::Other(format!(
                        "Agent {} timed out after {}ms",
                        agent_id, t
                    ))),
                }
            } else {
                agent.chat(&input_owned).await
            };

            let duration_ms = agent_start.elapsed().as_millis() as u64;
            match result {
                Ok(response) => AgentResult {
                    agent_id,
                    response: Some(response),
                    duration_ms,
                    success: true,
                    error: None,
                },
                Err(e) => AgentResult {
                    agent_id,
                    response: None,
                    duration_ms,
                    success: false,
                    error: Some(e.to_string()),
                },
            }
        });
    }

    let mut results = Vec::with_capacity(agents.len());
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(agent_result) => results.push(agent_result),
            Err(e) => {
                warn!(error = %e, "Concurrent task panicked");
            }
        }
    }

    let success_count = results.iter().filter(|r| r.success).count();
    let failed_count = results.len() - success_count;

    // Abort on any failure if configured.
    if failed_count > 0 && matches!(on_partial_failure, PartialFailureAction::Abort) {
        let failed_agents: Vec<_> = results
            .iter()
            .filter(|r| !r.success)
            .map(|r| r.agent_id.as_str())
            .collect();
        return Err(AgentError::Other(format!(
            "Concurrent execution aborted: {} agent(s) failed [{}]",
            failed_count,
            failed_agents.join(", ")
        )));
    }

    // Check minimum required successes.
    if let Some(min) = min_required {
        if success_count < min {
            return Err(AgentError::Other(format!(
                "Only {} of {} required agents succeeded",
                success_count, min
            )));
        }
    }

    // Build agent weight map for voting aggregation.
    let agent_weights: HashMap<String, f64> = agents
        .iter()
        .map(|a| (a.id().to_string(), a.weight()))
        .collect();

    let strategy_name = format!("{:?}", aggregation_config.strategy);
    let response =
        aggregation::aggregate(&results, aggregation_config, llm, &agent_weights).await?;

    info!(
        agents = results.len(),
        successes = success_count,
        duration_ms = start.elapsed().as_millis() as u64,
        "Concurrent execution completed"
    );

    Ok(ConcurrentResult {
        response,
        agent_results: results,
        aggregation_strategy: strategy_name,
    })
}
