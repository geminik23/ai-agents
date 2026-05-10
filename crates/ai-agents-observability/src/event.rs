use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationEvent {
    pub trace_id: String,
    pub span_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    pub turn_id: String,
    pub agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub event_type: EventType,
    pub purpose: ObservationPurpose,
    pub status: EventStatus,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<ObservationTokenUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostEstimate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ObservationError>,
    #[serde(default)]
    pub dimensions: HashMap<String, String>,
    #[serde(default)]
    pub tags: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    LlmCall {
        provider: String,
        model: String,
        alias: Option<String>,
        streaming: bool,
    },
    ToolCall {
        tool_id: String,
    },
    SkillExecution {
        skill_id: String,
    },
    SkillStep {
        skill_id: String,
        step_index: usize,
        step_type: String,
    },
    StateTransition {
        from: Option<String>,
        to: String,
    },
    Orchestration {
        pattern: String,
    },
    HitlApproval {
        trigger: String,
    },
    MemoryOperation {
        operation: String,
    },
    PersonaEvent {
        event: String,
    },
    FactsEvent {
        event: String,
    },
    RelationshipEvent {
        event: String,
    },
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPurpose {
    MainResponse,
    Router,
    SkillRouting,
    SkillPrompt,
    ProcessDetect,
    ProcessExtract,
    ProcessValidate,
    ProcessTransform,
    DisambiguationDetection,
    DisambiguationClarification,
    StateTransitionEvaluation,
    StateAction,
    ContextExtraction,
    Summarization,
    ReflectionDecision,
    ReflectionEvaluation,
    PlanGeneration,
    PlanStep,
    FactsExtraction,
    RelationshipUpdate,
    HitlLocalization,
    OrchestrationRouting,
    OrchestrationAggregation,
    OrchestrationConversation,
    EvaluationJudge,
    Other(String),
}

impl ObservationPurpose {
    pub fn as_label(&self) -> String {
        match self {
            Self::MainResponse => "main_response".to_string(),
            Self::Router => "router".to_string(),
            Self::SkillRouting => "skill_routing".to_string(),
            Self::SkillPrompt => "skill_prompt".to_string(),
            Self::ProcessDetect => "process_detect".to_string(),
            Self::ProcessExtract => "process_extract".to_string(),
            Self::ProcessValidate => "process_validate".to_string(),
            Self::ProcessTransform => "process_transform".to_string(),
            Self::DisambiguationDetection => "disambiguation_detection".to_string(),
            Self::DisambiguationClarification => "disambiguation_clarification".to_string(),
            Self::StateTransitionEvaluation => "state_transition_evaluation".to_string(),
            Self::StateAction => "state_action".to_string(),
            Self::ContextExtraction => "context_extraction".to_string(),
            Self::Summarization => "summarization".to_string(),
            Self::ReflectionDecision => "reflection_decision".to_string(),
            Self::ReflectionEvaluation => "reflection_evaluation".to_string(),
            Self::PlanGeneration => "plan_generation".to_string(),
            Self::PlanStep => "plan_step".to_string(),
            Self::FactsExtraction => "facts_extraction".to_string(),
            Self::RelationshipUpdate => "relationship_update".to_string(),
            Self::HitlLocalization => "hitl_localization".to_string(),
            Self::OrchestrationRouting => "orchestration_routing".to_string(),
            Self::OrchestrationAggregation => "orchestration_aggregation".to_string(),
            Self::OrchestrationConversation => "orchestration_conversation".to_string(),
            Self::EvaluationJudge => "evaluation_judge".to_string(),
            Self::Other(value) => value.clone(),
        }
    }
}

impl Default for ObservationPurpose {
    fn default() -> Self {
        Self::Other("other".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventStatus {
    Success,
    Error,
    Cancelled,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationTokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub source: TokenUsageSource,
}

impl ObservationTokenUsage {
    pub fn new(input_tokens: u64, output_tokens: u64, source: TokenUsageSource) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            source,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenUsageSource {
    Provider,
    StreamFinalChunk,
    Estimated,
    Missing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub input_usd: f64,
    pub output_usd: f64,
    pub total_usd: f64,
    pub source: CostSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    Configured,
    BuiltIn,
    Unknown,
    Estimated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationError {
    pub kind: String,
    pub message: String,
}

impl ObservationError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            message: message.into(),
        }
    }
}
