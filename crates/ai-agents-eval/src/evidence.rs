use std::collections::HashMap;

use ai_agents_core::{ChatMessage, Role};
use ai_agents_hitl::{ApprovalRequest, ApprovalResolvedOutcome, ApprovalResult, ApprovalTrigger};
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
    /// Whether the wrapped tool implementation was invoked.
    #[serde(default = "default_executed_true")]
    pub executed: bool,
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

fn default_executed_true() -> bool {
    true
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

/// Normalized approval decision used by eval evidence and assertions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approved,
    Rejected,
    Modified,
    Timeout,
    Error,
}

/// Normalized trigger that caused an approval request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApprovalTriggerEvidence {
    Tool { name: String },
    Condition { name: String, matched: String },
    State { from: Option<String>, to: String },
}

impl From<&ApprovalTrigger> for ApprovalTriggerEvidence {
    fn from(trigger: &ApprovalTrigger) -> Self {
        match trigger {
            ApprovalTrigger::Tool { name, .. } => Self::Tool { name: name.clone() },
            ApprovalTrigger::Condition { name, matched } => Self::Condition {
                name: name.clone(),
                matched: matched.clone(),
            },
            ApprovalTrigger::State { from, to } => Self::State {
                from: from.clone(),
                to: to.clone(),
            },
        }
    }
}

/// In-memory evidence for one fully resolved approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalEvidence {
    /// Stable request ID assigned by the HITL runtime.
    pub request_id: String,
    /// Normalized request trigger without tool argument values.
    pub trigger: ApprovalTriggerEvidence,
    /// Decision returned directly by the approval handler.
    pub raw_decision: ApprovalDecision,
    /// Decision after runtime timeout and error resolution.
    pub effective_decision: ApprovalDecision,
    /// Original tool arguments supplied with the request.
    #[serde(default, skip_serializing)]
    pub original_args: Option<Value>,
    /// Complete tool arguments after an effective modification.
    #[serde(default, skip_serializing)]
    pub modified_args: Option<Value>,
    /// Tool arguments that would be executed after the effective decision.
    #[serde(default, skip_serializing)]
    pub effective_args: Option<Value>,
    /// Localized message shown to the approval handler.
    #[serde(default, skip_serializing)]
    pub message: String,
    /// Rejection reason from the effective result, when present.
    #[serde(default, skip_serializing)]
    pub rejection_reason: Option<String>,
    /// Resolution error from the effective result, when present.
    #[serde(default, skip_serializing)]
    pub error: Option<String>,
}

impl ApprovalEvidence {
    /// Normalize one approval hook resolution into assertion-time evidence.
    pub fn from_resolution(
        request: &ApprovalRequest,
        raw_result: &ApprovalResult,
        effective_result: &ApprovalResolvedOutcome,
    ) -> Self {
        let original_args = match &request.trigger {
            ApprovalTrigger::Tool { args, .. } => Some(args.clone()),
            _ => None,
        };
        let (effective_decision, changes, rejection_reason, error) = match effective_result {
            ApprovalResolvedOutcome::Approved => (ApprovalDecision::Approved, None, None, None),
            ApprovalResolvedOutcome::Rejected { reason } => {
                (ApprovalDecision::Rejected, None, reason.clone(), None)
            }
            ApprovalResolvedOutcome::Modified { changes } => {
                (ApprovalDecision::Modified, Some(changes), None, None)
            }
            ApprovalResolvedOutcome::Error { message } => {
                (ApprovalDecision::Error, None, None, Some(message.clone()))
            }
        };
        let modified_args = changes.and_then(|changes| {
            original_args
                .as_ref()
                .map(|original| apply_argument_changes(original, changes))
        });
        let effective_args = match effective_decision {
            ApprovalDecision::Approved => original_args.clone(),
            ApprovalDecision::Modified => modified_args.clone(),
            ApprovalDecision::Rejected | ApprovalDecision::Timeout | ApprovalDecision::Error => {
                None
            }
        };

        Self {
            request_id: request.id.clone(),
            trigger: ApprovalTriggerEvidence::from(&request.trigger),
            raw_decision: approval_result_decision(raw_result),
            effective_decision,
            original_args,
            modified_args,
            effective_args,
            message: request.message.clone(),
            rejection_reason,
            error,
        }
    }
}

fn approval_result_decision(result: &ApprovalResult) -> ApprovalDecision {
    match result {
        ApprovalResult::Approved => ApprovalDecision::Approved,
        ApprovalResult::Rejected { .. } => ApprovalDecision::Rejected,
        ApprovalResult::Modified { .. } => ApprovalDecision::Modified,
        ApprovalResult::Timeout => ApprovalDecision::Timeout,
    }
}

fn apply_argument_changes(original: &Value, changes: &HashMap<String, Value>) -> Value {
    let mut modified = original.clone();
    if let Value::Object(arguments) = &mut modified {
        for (key, value) in changes {
            arguments.insert(key.clone(), value.clone());
        }
        modified
    } else {
        Value::Object(changes.clone().into_iter().collect())
    }
}

/// One message sent as part of an LLM request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessageEvidence {
    /// Role assigned to the message.
    pub role: Role,
    /// Complete message content retained only for in-memory assertions.
    #[serde(default, skip_serializing)]
    pub content: String,
}

/// One complete message list supplied to an LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequestEvidence {
    /// Messages captured together by one LLM start hook invocation.
    pub messages: Vec<LlmMessageEvidence>,
}

impl LlmRequestEvidence {
    /// Copy a hook message slice while preserving its request boundary.
    pub fn from_messages(messages: &[ChatMessage]) -> Self {
        Self {
            messages: messages
                .iter()
                .map(|message| LlmMessageEvidence {
                    role: message.role,
                    content: message.content.clone(),
                })
                .collect(),
        }
    }
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
    /// Fully resolved approval requests recorded during this turn.
    #[serde(default)]
    pub approvals: Vec<ApprovalEvidence>,
    /// Complete LLM requests retained only for in-memory assertions.
    #[serde(default, skip_serializing)]
    pub llm_requests: Vec<LlmRequestEvidence>,
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
        approvals: Vec::new(),
        llm_requests: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn llm_request_preserves_roles_and_keeps_content_in_memory() {
        let evidence = LlmRequestEvidence::from_messages(&[
            ChatMessage::system("persona and reasoning prompt"),
            ChatMessage::user("private user history"),
            ChatMessage::assistant("private assistant history"),
        ]);

        assert_eq!(evidence.messages.len(), 3);
        assert_eq!(evidence.messages[0].role, Role::System);
        assert_eq!(evidence.messages[1].content, "private user history");

        let serialized = serde_json::to_string(&evidence).unwrap();
        assert!(!serialized.contains("persona and reasoning prompt"));
        assert!(!serialized.contains("private user history"));
        assert!(!serialized.contains("private assistant history"));
    }

    #[test]
    fn normalizes_modified_approval_and_keeps_sensitive_values_in_memory() {
        let request = ApprovalRequest::new(
            ApprovalTrigger::tool("transfer", json!({"amount": 100, "currency": "USD"})),
            "Approve this transfer?",
        );
        let mut changes = HashMap::new();
        changes.insert("amount".to_string(), json!(25));
        let evidence = ApprovalEvidence::from_resolution(
            &request,
            &ApprovalResult::Modified {
                changes: changes.clone(),
            },
            &ApprovalResolvedOutcome::Modified { changes },
        );

        assert_eq!(evidence.raw_decision, ApprovalDecision::Modified);
        assert_eq!(evidence.effective_decision, ApprovalDecision::Modified);
        assert_eq!(
            evidence.original_args,
            Some(json!({"amount": 100, "currency": "USD"}))
        );
        assert_eq!(
            evidence.modified_args,
            Some(json!({"amount": 25, "currency": "USD"}))
        );
        assert_eq!(evidence.effective_args, evidence.modified_args);
        assert_eq!(evidence.message, "Approve this transfer?");

        let serialized = serde_json::to_value(&evidence).unwrap();
        assert!(serialized.get("original_args").is_none());
        assert!(serialized.get("modified_args").is_none());
        assert!(serialized.get("effective_args").is_none());
        assert!(serialized.get("message").is_none());
    }

    #[test]
    fn normalizes_timeout_to_effective_error_without_executable_arguments() {
        let request = ApprovalRequest::new(
            ApprovalTrigger::tool("delete", json!({"path": "/private"})),
            "Delete file?",
        );
        let evidence = ApprovalEvidence::from_resolution(
            &request,
            &ApprovalResult::Timeout,
            &ApprovalResolvedOutcome::Error {
                message: "approval timed out".to_string(),
            },
        );

        assert_eq!(evidence.raw_decision, ApprovalDecision::Timeout);
        assert_eq!(evidence.effective_decision, ApprovalDecision::Error);
        assert_eq!(evidence.error.as_deref(), Some("approval timed out"));
        assert!(evidence.effective_args.is_none());
    }
}
