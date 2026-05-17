use std::collections::HashMap;

use ai_agents_observability::ObservabilityReport;
use ai_agents_runtime::RuntimeAgent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::fixtures::RecordingToolLog;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    pub call_id: String,
    pub tool_id: String,
    pub requested_name: String,
    pub source: ToolExecutionSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    pub arguments_original: Value,
    pub arguments_executed: Value,
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    pub started_at: DateTime<Utc>,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observability_span_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillEvidence {
    pub selected_skill_id: Option<String>,
    pub executed_skill_id: Option<String>,
    pub no_match: bool,
    pub clarification_requested: bool,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisambiguationEvidence {
    pub status: DisambiguationStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ambiguity_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactsEvidence {
    pub actor_id: Option<String>,
    pub facts: Vec<Value>,
    pub before_count: Option<usize>,
    pub after_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEvidence {
    pub actor_id: Option<String>,
    pub model: Option<String>,
    pub available_perspectives: Vec<String>,
    pub current: Option<Value>,
    pub before: Option<Value>,
    pub after: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaEvidence {
    pub secret_revealed: bool,
    pub revealed_secret_ids: Vec<String>,
    pub revealed_secret_count: usize,
    pub evolution_events: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnObservabilityEvidence {
    pub trace_id: Option<String>,
    pub span_ids: Vec<String>,
    pub report: Option<ObservabilityReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvidence {
    pub response_metadata: Option<Value>,
    pub state: Option<String>,
    pub state_history: Vec<ai_agents_core::StateTransitionEvent>,
    pub context: Value,
    pub tool_executions: Vec<ToolExecutionRecord>,
    pub skill: Option<SkillEvidence>,
    pub disambiguation: Option<DisambiguationEvidence>,
    pub facts: Option<FactsEvidence>,
    pub relationship: Option<RelationshipEvidence>,
    pub persona: Option<PersonaEvidence>,
    pub orchestration: Option<Value>,
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
