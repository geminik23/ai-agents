use std::collections::HashMap;

use ai_agents_observability::ObservabilityReport;
use ai_agents_runtime::RuntimeAgent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::fixtures::RecordingToolLog;

/// Source category for a recorded tool execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionSource {
    Llm,
    Skill,
    StateAction,
    OnEnter,
    OnExit,
    PostTransition,
    Spawner,
    Orchestration,
    Mock,
}

/// Structured record for one tool execution observed during eval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    /// Unique ID for this recorded tool call.
    pub call_id: String,
    /// Canonical tool ID executed by the registry.
    pub tool_id: String,
    /// Tool name requested by the model or runtime.
    pub requested_name: String,
    /// Source category assigned to this execution.
    pub source: ToolExecutionSource,
    /// Current or expected state name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Actor ID associated with this evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    /// Original tool arguments before execution.
    pub arguments_original: Value,
    /// Arguments passed to the wrapped tool.
    pub arguments_executed: Value,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Directory where output artifacts are written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Error text for failed execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Optional response or tool metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// UTC timestamp when execution started.
    pub started_at: DateTime<Utc>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Optional observability span ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observability_span_id: Option<String>,
}

/// Skill routing evidence inferred or reported for a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillEvidence {
    /// Skill ID selected by routing, if available.
    pub selected_skill_id: Option<String>,
    /// Skill ID actually executed, if available.
    pub executed_skill_id: Option<String>,
    /// Whether routing found no matching skill.
    pub no_match: bool,
    /// Whether clarification was requested.
    pub clarification_requested: bool,
}

/// Normalized status values for disambiguation evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DisambiguationStatus {
    Clear,
    Skipped,
    Triggered,
    Clarified,
    BestGuess,
    Abandoned,
    GiveUp,
    Escalated,
}

/// Disambiguation evidence inferred or reported for a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisambiguationEvidence {
    /// Final or normalized status value.
    pub status: DisambiguationStatus,
    /// Ambiguity type reported by detection, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ambiguity_type: Option<String>,
    /// Detection confidence reported by the system.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Resolved payload when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved: Option<Value>,
}

/// Actor fact evidence captured around one turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactsEvidence {
    /// Actor ID associated with this evidence.
    pub actor_id: Option<String>,
    /// Serialized fact records visible after the turn.
    pub facts: Vec<Value>,
    /// Number of facts before the turn when known.
    pub before_count: Option<usize>,
    /// Number of facts after the turn when known.
    pub after_count: Option<usize>,
}

/// Relationship memory evidence captured around one turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEvidence {
    /// Actor ID associated with this evidence.
    pub actor_id: Option<String>,
    /// Model or relationship model name.
    pub model: Option<String>,
    /// Perspectives available for assertions.
    pub available_perspectives: Vec<String>,
    /// Current serialized relationship state.
    pub current: Option<Value>,
    /// State before the turn when available.
    pub before: Option<Value>,
    /// State after the turn when available.
    pub after: Option<Value>,
}

/// Persona reveal and evolution evidence for one turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaEvidence {
    /// Whether any persona secret is currently revealed.
    pub secret_revealed: bool,
    /// IDs of revealed secrets when stable IDs are available.
    pub revealed_secret_ids: Vec<String>,
    /// Number of revealed secrets.
    pub revealed_secret_count: usize,
    /// Number of persona evolution events recorded.
    pub evolution_events: usize,
}

/// Observability evidence attached to one evaluated turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnObservabilityEvidence {
    /// Trace ID associated with the turn when available.
    pub trace_id: Option<String>,
    /// Span IDs observed during the turn.
    pub span_ids: Vec<String>,
    /// Observability report snapshot generated after the turn.
    pub report: Option<ObservabilityReport>,
}

/// Full assertion-time evidence collected after a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvidence {
    /// Response metadata produced by the runtime.
    pub response_metadata: Option<Value>,
    /// Current or expected state name.
    pub state: Option<String>,
    /// State transition history observed by the runtime.
    pub state_history: Vec<ai_agents_core::StateTransitionEvent>,
    /// Runtime or fixture context value.
    pub context: Value,
    /// Tool calls recorded during this turn.
    pub tool_executions: Vec<ToolExecutionRecord>,
    /// Skill evidence for this turn, if available.
    pub skill: Option<SkillEvidence>,
    /// Expected disambiguation status or evidence.
    pub disambiguation: Option<DisambiguationEvidence>,
    /// Serialized fact records visible after the turn.
    pub facts: Option<FactsEvidence>,
    /// Relationship memory assertion or evidence.
    pub relationship: Option<RelationshipEvidence>,
    /// persona value for TurnEvidence.
    pub persona: Option<PersonaEvidence>,
    /// Orchestration metadata assertion or evidence.
    pub orchestration: Option<Value>,
    /// Observability assertion, setting, or report value.
    pub observability: Option<TurnObservabilityEvidence>,
}

pub fn collect_turn_evidence(
    agent: &RuntimeAgent,
    response_metadata: Option<HashMap<String, Value>>,
    tool_log: &RecordingToolLog,
    tool_start_index: usize,
    before_relationship: Option<Value>,
) -> TurnEvidence {
    let context_map = agent.get_context();
    let context = serde_json::to_value(&context_map).unwrap_or(Value::Null);
    let metadata_value = response_metadata
        .clone()
        .and_then(|metadata| serde_json::to_value(metadata).ok());
    let orchestration = metadata_value
        .as_ref()
        .and_then(|metadata| metadata.get("orchestration").cloned())
        .or_else(|| context.get("orchestration").cloned());
    let disambiguation = infer_disambiguation(metadata_value.as_ref(), &context);
    let skill = infer_skill(metadata_value.as_ref(), disambiguation.as_ref());
    let actor_id = agent.actor_id();
    let facts = Some(FactsEvidence {
        actor_id: actor_id.clone(),
        facts: agent
            .actor_facts()
            .into_iter()
            .filter_map(|fact| serde_json::to_value(fact).ok())
            .collect(),
        before_count: None,
        after_count: Some(agent.actor_facts().len()),
    });
    let relationship = collect_relationship(agent, actor_id.clone(), before_relationship);
    let persona = collect_persona(agent, &context_map);
    let observability = agent.observability().map(|manager| {
        let report = manager.generate_report();
        let raw_events = manager.raw_events();
        TurnObservabilityEvidence {
            trace_id: raw_events.last().map(|event| event.trace_id.clone()),
            span_ids: raw_events
                .iter()
                .map(|event| event.span_id.clone())
                .collect(),
            report: Some(report),
        }
    });

    TurnEvidence {
        response_metadata: metadata_value,
        state: agent.current_state(),
        state_history: agent.state_history(),
        context,
        tool_executions: tool_log.records_since(tool_start_index),
        skill,
        disambiguation,
        facts,
        relationship,
        persona,
        orchestration,
        observability,
    }
}

pub fn relationship_snapshot(agent: &RuntimeAgent) -> Option<Value> {
    let actor_id = agent.actor_id()?;
    let manager = agent.relationship_manager()?;
    manager.relationship_as_value(&actor_id).ok().flatten()
}

fn infer_disambiguation(
    metadata: Option<&Value>,
    context: &Value,
) -> Option<DisambiguationEvidence> {
    if let Some(disambiguation) = metadata.and_then(|m| m.get("disambiguation")) {
        let status = match disambiguation
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("triggered")
        {
            "awaiting_clarification" => DisambiguationStatus::Triggered,
            "clarified" => DisambiguationStatus::Clarified,
            "best_guess" => DisambiguationStatus::BestGuess,
            "abandoned" => DisambiguationStatus::Abandoned,
            "give_up" => DisambiguationStatus::GiveUp,
            "escalated" => DisambiguationStatus::Escalated,
            "skipped" => DisambiguationStatus::Skipped,
            "clear" => DisambiguationStatus::Clear,
            _ => DisambiguationStatus::Triggered,
        };
        let detection = disambiguation.get("detection");
        return Some(DisambiguationEvidence {
            status,
            ambiguity_type: detection.and_then(|d| d.get("type")).map(|v| v.to_string()),
            confidence: detection
                .and_then(|d| d.get("confidence"))
                .and_then(Value::as_f64)
                .map(|v| v as f32),
            resolved: disambiguation.get("resolved").cloned(),
        });
    }

    if context
        .pointer("/disambiguation/resolved")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some(DisambiguationEvidence {
            status: DisambiguationStatus::Clarified,
            ambiguity_type: None,
            confidence: None,
            resolved: context.get("disambiguation").cloned(),
        });
    }

    None
}

fn infer_skill(
    metadata: Option<&Value>,
    disambiguation: Option<&DisambiguationEvidence>,
) -> Option<SkillEvidence> {
    let skill_id = metadata
        .and_then(|m| m.get("skill_id"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            metadata
                .and_then(|m| m.get("disambiguation"))
                .and_then(|d| d.get("skill_id"))
                .and_then(Value::as_str)
                .map(str::to_string)
        });

    if skill_id.is_none() && disambiguation.is_none() {
        return None;
    }

    Some(SkillEvidence {
        selected_skill_id: skill_id.clone(),
        executed_skill_id: skill_id,
        no_match: false,
        clarification_requested: disambiguation
            .map(|d| d.status == DisambiguationStatus::Triggered)
            .unwrap_or(false),
    })
}

fn collect_relationship(
    agent: &RuntimeAgent,
    actor_id: Option<String>,
    before: Option<Value>,
) -> Option<RelationshipEvidence> {
    let actor_id = actor_id?;
    let manager = agent.relationship_manager()?;
    let current = manager.relationship_as_value(&actor_id).ok().flatten();
    let model = current
        .as_ref()
        .and_then(|value| value.get("model"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut available = vec!["agent_to_actor".to_string(), "mutual".to_string()];
    if model.as_deref() == Some("two_sided") {
        available.push("perceived_actor_to_agent".to_string());
    }
    Some(RelationshipEvidence {
        actor_id: Some(actor_id),
        model,
        available_perspectives: available,
        before,
        after: current.clone(),
        current,
    })
}

fn collect_persona(
    agent: &RuntimeAgent,
    context_map: &HashMap<String, Value>,
) -> Option<PersonaEvidence> {
    let manager = agent.persona_manager()?;
    let revealed_count = manager.revealed_secrets(context_map).len();
    Some(PersonaEvidence {
        secret_revealed: revealed_count > 0,
        revealed_secret_ids: Vec::new(),
        revealed_secret_count: revealed_count,
        evolution_events: manager.history().len(),
    })
}
