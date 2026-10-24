//! Acknowledged numeric telemetry for gate input, separate from authoritative resource admission.

use super::*;
use ai_agents_core::{AgentError, Result};
use ai_agents_observability::manager::ObservabilityCursor;
use ai_agents_observability::{EventType, ObservabilityManager, ObservationEvent};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
#[path = "rg_telemetry_tests.rs"]
mod tests;

const ADAPTER: &str = "runtime.telemetry";

struct Source {
    manager: Arc<ObservabilityManager>,
    cursor: ObservabilityCursor,
}

/// Live collectors are retained by actual object identity; their cursors are never reconstructed from serialized labels.
#[derive(Default)]
pub(crate) struct TaskTelemetry {
    sources: parking_lot::Mutex<Vec<Source>>,
    missing: std::sync::atomic::AtomicBool,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericEvent {
    id: String,
    llm: bool,
    tool: bool,
    error: bool,
    purpose: ai_agents_observability::ObservationPurpose,
    duration_ms: u64,
    tokens: Option<ai_agents_observability::ObservationTokenUsage>,
    cost: Option<ai_agents_observability::CostEstimate>,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Journal {
    complete: bool,
    tokens_complete: bool,
    cost_complete: bool,
    events: BTreeMap<String, NumericEvent>,
}

type Cut = Vec<(Arc<ObservabilityManager>, ObservabilityCursor)>;

impl TaskTelemetry {
    /// Permanently records an unobserved interval, including legacy images without a numeric journal.
    pub(crate) fn mark_missing(&self) {
        self.missing
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Registers each actual collector once, excluding events from earlier ordinary work; a missing participant poisons completeness.
    pub(crate) fn register(&self, manager: Option<Arc<ObservabilityManager>>) {
        let Some(manager) = manager else {
            self.missing
                .store(true, std::sync::atomic::Ordering::Release);
            return;
        };
        let mut sources = self.sources.lock();
        if !sources
            .iter()
            .any(|source| Arc::ptr_eq(&source.manager, &manager))
        {
            sources.push(Source {
                cursor: manager.event_cursor(),
                manager,
            });
        }
    }

    /// Projects only run-labelled completed events and persists numeric facts, never prompt, tool output or actor payloads.
    /// Cursors remain unchanged until the enclosing checkpoint ACK, making lost ACK inspection and repeat captures idempotent.
    pub(crate) fn prepare(&self, payload: &mut TaskCheckpointPayload, run_id: &str) -> Result<Cut> {
        let sources = self.sources.lock();

        let existing = payload
            .adapters
            .iter()
            .position(|adapter| adapter.id == ADAPTER);
        let mut journal: Journal = if let Some(index) = existing {
            serde_json::from_value(payload.adapters[index].state.clone())?
        } else {
            Journal {
                complete: !sources.is_empty(),
                tokens_complete: true,
                cost_complete: true,
                ..Default::default()
            }
        };
        journal.complete &= !self.missing.load(std::sync::atomic::Ordering::Acquire);
        let mut cuts = Vec::new();
        for source in sources.iter() {
            let (cursor, events, complete) = source.manager.checkpoint_since(source.cursor);
            let config = source.manager.config();
            journal.complete &= complete
                && config.latency.track_llm
                && config.latency.track_tools
                && config.latency.track_skills
                && config.latency.track_orchestration
                && config.latency.track_hitl;
            journal.tokens_complete &= config.tokens.count_input && config.tokens.count_output;
            journal.cost_complete &=
                config.cost.enabled && config.tokens.count_input && config.tokens.count_output;
            for event in events.into_iter().filter(|event| {
                event
                    .dimensions
                    .get("task_run")
                    .is_some_and(|id| id == run_id)
            }) {
                let numeric = NumericEvent {
                    id: event.span_id.clone(),
                    llm: matches!(event.event_type, EventType::LlmCall { .. }),
                    tool: matches!(event.event_type, EventType::ToolCall { .. }),
                    error: event.status == ai_agents_observability::EventStatus::Error,
                    purpose: match event.purpose {
                        ai_agents_observability::ObservationPurpose::Other(_) => {
                            ai_agents_observability::ObservationPurpose::Other(String::new())
                        }
                        purpose => purpose,
                    },
                    duration_ms: event.duration_ms,
                    tokens: event.tokens,
                    cost: event.cost,
                };
                journal.tokens_complete &= !numeric.llm
                    || numeric.tokens.as_ref().is_some_and(|tokens| {
                        tokens.source != ai_agents_observability::TokenUsageSource::Missing
                    });
                journal.cost_complete &= !numeric.llm
                    || (numeric.tokens.as_ref().is_some_and(|tokens| {
                        tokens.source != ai_agents_observability::TokenUsageSource::Missing
                    }) && numeric.cost.as_ref().is_some_and(|cost| {
                        cost.source != ai_agents_observability::CostSource::Unknown
                    }));
                // Numeric telemetry may become unavailable, but saturation must never prevent cancellation or authoritative cleanup journals.
                if journal.events.len() >= 4096 && !journal.events.contains_key(&numeric.id) {
                    journal.complete = false;
                    continue;
                }
                if let Some(old) = journal.events.insert(numeric.id.clone(), numeric.clone())
                    && serde_json::to_value(old)? != serde_json::to_value(numeric)?
                {
                    return Err(AgentError::Config(
                        "conflicting task telemetry event".into(),
                    ));
                }
            }
            cuts.push((source.manager.clone(), cursor));
        }

        let state = serde_json::to_value(journal)?;
        let adapter = TaskAdapterCheckpoint {
            id: ADAPTER.into(),
            adapter: ADAPTER.into(),
            contract_version: 1,
            config: json!({"contract":"run_numeric_events_v1"}),
            state,
        };
        if let Some(index) = existing {
            payload.adapters[index] = adapter;
        } else {
            payload.adapters.push(adapter);
        }
        Ok(cuts)
    }

    /// Advances only the exact live collector cuts whose containing task write was acknowledged.
    pub(crate) fn acknowledge(&self, cuts: Cut) {
        let mut sources = self.sources.lock();
        for (manager, cursor) in cuts {
            if let Some(source) = sources
                .iter_mut()
                .find(|source| Arc::ptr_eq(&source.manager, &manager))
            {
                source.cursor = cursor;
            }
        }
    }
}

/// Missing measurements remain unknown even for exists/negation or a matcher on their containing object.
pub(crate) fn metric_available(value: &Value, path: &str) -> bool {
    value
        .get("_unavailable")
        .and_then(Value::as_array)
        .is_none_or(|items| {
            !items.iter().any(|item| {
                item.as_str().is_none_or(|missing| {
                    missing == path
                        || path.starts_with(&format!("{missing}."))
                        || missing.starts_with(&format!("{path}."))
                })
            })
        })
}

/// Rebuilds the existing report schema from acknowledged numeric events; missing or lossy telemetry cannot pass a gate.
pub(crate) fn observation(
    payload: &TaskCheckpointPayload,
    identity: EvidenceIdentity,
) -> Result<Option<ScopedObservation<Value>>> {
    let Some(adapter) = payload
        .adapters
        .iter()
        .find(|adapter| adapter.id == ADAPTER)
    else {
        return Ok(None);
    };
    let journal: Journal = serde_json::from_value(adapter.state.clone())?;
    if adapter.contract_version != 1 || journal.events.len() > 4096 {
        return Err(AgentError::Config("invalid task telemetry journal".into()));
    }
    let events = journal
        .events
        .into_values()
        .map(|event| ObservationEvent {
            trace_id: String::new(),
            span_id: event.id,
            parent_span_id: None,
            turn_id: String::new(),
            agent_id: String::new(),
            actor_id: None,
            session_id: None,
            event_type: if event.llm {
                EventType::LlmCall {
                    provider: String::new(),
                    model: String::new(),
                    alias: None,
                    streaming: false,
                }
            } else if event.tool {
                EventType::ToolCall {
                    tool_id: String::new(),
                }
            } else {
                EventType::MemoryOperation {
                    operation: String::new(),
                }
            },
            purpose: event.purpose,
            status: if event.error {
                ai_agents_observability::EventStatus::Error
            } else {
                ai_agents_observability::EventStatus::Success
            },
            timestamp: chrono::DateTime::UNIX_EPOCH,
            duration_ms: event.duration_ms,
            tokens: event.tokens,
            cost: event.cost,
            error: None,
            dimensions: Default::default(),
            tags: Default::default(),
            payload: None,
        })
        .collect::<Vec<_>>();
    let report = ai_agents_observability::report::generate_report(&events, Vec::new(), 0);
    // An enabled collector is not proof that every actual provider/tool was wrapped; missing or duplicated instrumentation cannot prove a zero/upper bound.
    // This uses the ledger only to verify observation coverage, never telemetry to authorize or refund an invocation.
    let complete = journal.complete
        && report.summary.total_llm_calls == payload.counters.llm_attempts
        && report.summary.total_tool_calls == payload.counters.tool_attempts;
    let mut value = json!({"summary": report.summary, "token_breakdown":report.token_breakdown, "cost_breakdown":report.cost_breakdown});
    // Unmeasured token/cost values are absent, never a manufactured zero that can satisfy an upper-bound gate.
    let mut unavailable = Vec::new();
    if !journal.tokens_complete {
        unavailable.extend(["summary.total_tokens", "token_breakdown"]);
        value["summary"]
            .as_object_mut()
            .unwrap()
            .remove("total_tokens");
        value.as_object_mut().unwrap().remove("token_breakdown");
    }
    if !journal.cost_complete {
        unavailable.extend([
            "summary.total_cost_usd",
            "summary.main_llm_cost_usd",
            "summary.router_cost_usd",
            "cost_breakdown",
        ]);
        for field in ["total_cost_usd", "main_llm_cost_usd", "router_cost_usd"] {
            value["summary"].as_object_mut().unwrap().remove(field);
        }
        value.as_object_mut().unwrap().remove("cost_breakdown");
    }
    value["_unavailable"] = json!(unavailable);
    Ok(Some(ScopedObservation {
        identity,
        complete,
        value,
    }))
}
