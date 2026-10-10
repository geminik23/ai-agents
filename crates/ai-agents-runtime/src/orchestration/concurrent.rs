use std::collections::HashMap;
use std::time::Instant;

use ai_agents_core::{AgentError, Result};
use ai_agents_llm::LLMProvider;
use ai_agents_observability::{current_observation_context, with_observation_context};
use ai_agents_state::{AggregationConfig, ConcurrentAgentRef, PartialFailureAction};
use tokio::task::JoinSet;
use tracing::{info, warn};

use super::aggregation;
use super::types::{AgentResult, ConcurrentResult};
use crate::Agent;
use crate::runtime::{current_runtime_gate_identity_stack, scope_runtime_gate_identity_stack};
use crate::spawner::AgentRegistry;
use crate::turn_context::current_turn_actor_context;

/// Run multiple agents in parallel and aggregate results.
/// The explicit arguments preserve the public orchestration call contract and its independent controls.
/// The complete immutable root-gate ancestry is captured before spawning so every child preserves cycle detection.
#[allow(clippy::too_many_arguments)]
pub async fn concurrent(
    registry: &AgentRegistry,
    input: &str,
    agents: &[ConcurrentAgentRef],
    aggregation_config: &AggregationConfig,
    llm: Option<&dyn LLMProvider>,
    min_required: Option<usize>,
    timeout_ms: Option<u64>,
    on_partial_failure: PartialFailureAction,
    vote_parallelism: Option<usize>,
) -> Result<ConcurrentResult> {
    Box::pin(concurrent_with_llms(
        registry,
        input,
        agents,
        aggregation_config,
        super::AggregationLLMs::shared(llm),
        min_required,
        timeout_ms,
        on_partial_failure,
        vote_parallelism,
    ))
    .await
}

/// Runs participant turns with unchanged ownership and uses captured role providers only for aggregation.
/// Private slot, response and barrier custody cross JoinSet boundaries; suspension is propagated only after admitted siblings drain and settle.
#[allow(clippy::too_many_arguments)]
pub async fn concurrent_with_llms(
    registry: &AgentRegistry,
    input: &str,
    agents: &[ConcurrentAgentRef],
    aggregation_config: &AggregationConfig,
    llms: super::AggregationLLMs<'_>,
    min_required: Option<usize>,
    timeout_ms: Option<u64>,
    on_partial_failure: PartialFailureAction,
    vote_parallelism: Option<usize>,
) -> Result<ConcurrentResult> {
    let authorized = crate::autonomy::composition::enter_composition_dispatch();
    crate::autonomy::composition::scope_dispatch_authority(
        authorized,
        Box::pin(concurrent_dispatch(
            registry,
            input,
            agents,
            aggregation_config,
            llms,
            min_required,
            timeout_ms,
            on_partial_failure,
            vote_parallelism,
        )),
    )
    .await
}

/// Only the owning public entry receives parent cursor authority; nested callbacks retain ordinary orchestration semantics.
#[allow(clippy::too_many_arguments)]
async fn concurrent_dispatch(
    registry: &AgentRegistry,
    input: &str,
    agents: &[ConcurrentAgentRef],
    aggregation_config: &AggregationConfig,
    llms: super::AggregationLLMs<'_>,
    min_required: Option<usize>,
    timeout_ms: Option<u64>,
    on_partial_failure: PartialFailureAction,
    vote_parallelism: Option<usize>,
) -> Result<ConcurrentResult> {
    if agents.is_empty() {
        return Err(AgentError::Config(
            "No agents for concurrent execution".into(),
        ));
    }

    use crate::autonomy::composition;

    let start = Instant::now();
    let execution = crate::autonomy::current_execution();
    let frame = execution
        .as_ref()
        .map(|execution| execution.composition_frame())
        .transpose()?
        .flatten();
    let scope = composition::current_composition();
    let response_custody = composition::group_response_custody();
    let mut results: Vec<AgentResult> = frame
        .as_ref()
        .and_then(|frame| frame.cursor.get("results"))
        .map(|results| serde_json::from_value(results.clone()))
        .transpose()?
        .unwrap_or_default();
    let targets = agents
        .iter()
        .map(|agent_ref| {
            registry.get(agent_ref.id()).ok_or_else(|| {
                AgentError::Other(format!("Agent not found in registry: {}", agent_ref.id()))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let deadlines: Vec<Option<chrono::DateTime<chrono::Utc>>> = if let Some(value) = frame
        .as_ref()
        .and_then(|frame| frame.cursor.get("deadlines"))
    {
        serde_json::from_value(value.clone())?
    } else {
        let expiry = if let Some(frame) = &frame {
            frame.expires_at
        } else {
            timeout_ms
                .map(|millis| {
                    let millis = i64::try_from(millis)
                        .map_err(|_| AgentError::Config("concurrent deadline overflow".into()))?;
                    chrono::Utc::now()
                        .checked_add_signed(chrono::Duration::milliseconds(millis))
                        .ok_or_else(|| AgentError::Config("concurrent deadline overflow".into()))
                })
                .transpose()?
        };
        vec![expiry; agents.len()]
    };
    if deadlines.len() != agents.len() {
        return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
    }
    if let (Some(execution), Some(frame)) = (&execution, &frame) {
        Box::pin(execution.checkpoint_composition_cursor(
            &frame.runtime_id,
            serde_json::json!({"results":results,"deadlines":deadlines}),
        ))
        .await?;
    }

    let mut join_set = JoinSet::new();
    //
    // Tokio task-locals do not cross JoinSet::spawn, so all children inherit one immutable snapshot of the caller's complete root ownership chain.
    //
    let gate_identity_stack = current_runtime_gate_identity_stack();

    for (agent_index, agent_ref) in agents.iter().enumerate() {
        if results
            .iter()
            .any(|result| result.agent_index == agent_index)
        {
            continue;
        }
        let agent_id = agent_ref.id().to_string();
        let agent = targets[agent_index].clone();
        let slot = frame
            .as_ref()
            .map(|frame| {
                frame
                    .children
                    .get(agent_index)
                    .cloned()
                    .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)
            })
            .transpose()?;
        let scope = scope.clone();
        let response_custody = response_custody.clone();
        let input_owned = input.to_string();
        let timeout = if frame.is_some() {
            deadlines[agent_index]
                .map(|expiry| (expiry - chrono::Utc::now()).to_std().unwrap_or_default())
        } else {
            timeout_ms.map(std::time::Duration::from_millis)
        };
        let actor_context = current_turn_actor_context();
        let observation_context = current_observation_context();
        let gate_identity_stack = gate_identity_stack.clone();
        let execution = crate::autonomy::current_execution();
        let required = crate::autonomy::child_required()
            && matches!(on_partial_failure, PartialFailureAction::Abort);

        join_set.spawn(Box::pin(async move {
            crate::autonomy::scope_child_requirement(
                required,
                crate::autonomy::scope_inherited_execution(
                    execution,
                    Box::pin(scope_runtime_gate_identity_stack(
                        &gate_identity_stack,
                        async move {
                            let agent_start = Instant::now();
                            let run = async {
                                if let Some(context) = actor_context {
                                    agent.chat_with_actor_context(&input_owned, context).await
                                } else {
                                    agent.chat(&input_owned).await
                                }
                            };
                            let run = composition::run_composition_child(
                                scope,
                                slot,
                                response_custody,
                                run,
                            );
                            let result = if let Some(t) = timeout {
                                match tokio::time::timeout(t, async {
                                    if let Some(context) = observation_context.clone() {
                                        with_observation_context(context, run).await
                                    } else {
                                        run.await
                                    }
                                })
                                .await
                                {
                                    Ok(r) => r,
                                    Err(_) => Err(AgentError::Other(format!(
                                        "Agent {} timed out after {}ms",
                                        agent_id,
                                        t.as_millis()
                                    ))),
                                }
                            } else if let Some(context) = observation_context {
                                with_observation_context(context, run).await
                            } else {
                                run.await
                            };

                            let duration_ms = agent_start.elapsed().as_millis() as u64;
                            match result {
                                Err(error @ AgentError::TaskSuspended(_)) => Err(error),
                                Ok(response) => Ok(AgentResult {
                                    agent_index,
                                    agent_id,
                                    response: Some(response),
                                    duration_ms,
                                    success: true,
                                    error: None,
                                }),
                                Err(e) => Ok(AgentResult {
                                    agent_index,
                                    agent_id,
                                    response: None,
                                    duration_ms,
                                    success: false,
                                    error: Some(e.to_string()),
                                }),
                            }
                        },
                    )),
                ),
            )
            .await
        }));
    }

    let mut suspension = None;
    let mut join_failed = false;
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok(Ok(agent_result)) => results.push(agent_result),
            Ok(Err(error @ AgentError::TaskSuspended(_))) => suspension = Some(error),
            Ok(Err(error)) => return Err(error),
            Err(e) => {
                join_failed = true;
                warn!(error = %e, "Concurrent task panicked");
            }
        }
    }

    results.sort_by_key(|result| result.agent_index);
    if let (Some(execution), Some(frame)) = (&execution, &frame) {
        if join_failed {
            execution.stop("uncertain_child");
            return Err(AgentError::Other(
                "concurrent child requires recovery".into(),
            ));
        }
        Box::pin(execution.checkpoint_composition_cursor(
            &frame.runtime_id,
            serde_json::json!({"results":results,"deadlines":deadlines}),
        ))
        .await?;
    }
    if let Some(suspension) = suspension {
        return Err(suspension);
    }

    let success_count = results.iter().filter(|r| r.success).count();
    let failed_count = results.len() - success_count;

    // Abort on any failure if configured.
    if failed_count > 0 && matches!(on_partial_failure, PartialFailureAction::Abort) {
        if crate::autonomy::child_required()
            && let Some(execution) = crate::autonomy::current_execution()
        {
            execution.stop("required_child_failure");
        }
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
    if let Some(min) = min_required
        && success_count < min
    {
        if crate::autonomy::child_required()
            && let Some(execution) = crate::autonomy::current_execution()
        {
            execution.stop("required_child_failure");
        }
        return Err(AgentError::Other(format!(
            "Only {} of {} required agents succeeded",
            success_count, min
        )));
    }

    // Build agent weight map for voting aggregation.
    let agent_weights: HashMap<String, f64> = agents
        .iter()
        .map(|a| (a.id().to_string(), a.weight()))
        .collect();

    let strategy_name = format!("{:?}", aggregation_config.strategy);
    let response = aggregation::aggregate_with_llms(
        &results,
        aggregation_config,
        llms,
        &agent_weights,
        vote_parallelism,
    )
    .await?;

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
