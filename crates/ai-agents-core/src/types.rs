use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// High-level operation category used for tool policy and scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOperationKind {
    Read,
    Write,
    Edit,
    Delete,
    Patch,
    VcsInspect,
    Diagnostics,
    Command,
    Network,
    Interactive,
    Compute,
    Wait,
}

/// Side-effect level used to decide approval and concurrency behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSideEffectLevel {
    None,
    LocalWrite,
    ExternalRead,
    ExternalWrite,
    Destructive,
}

/// Permission decision returned before tool execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionOutcome {
    Allow,
    Deny,
    RequiresApproval,
    Unavailable,
}

/// Static safety metadata attached to a tool registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSafetyMetadata {
    /// True when the tool does not mutate local or external state.
    pub read_only: bool,
    /// True when calls can run concurrently without observable races.
    pub concurrency_safe: bool,
    /// Default operation category for this tool.
    pub operation: ToolOperationKind,
    /// Default side-effect level for this tool.
    pub side_effect_level: ToolSideEffectLevel,
    /// True when the tool can access the network.
    pub requires_network: bool,
    /// True when the tool can destroy or remove state.
    pub destructive: bool,
    /// True when the tool can reach open-ended external resources.
    pub open_world: bool,
    /// True when the tool requires a host-provided service.
    pub host_dependent: bool,
    /// True when execution requires user interaction.
    pub requires_user_interaction: bool,
    /// True when the executor can cancel the tool cooperatively.
    pub supports_cancellation: bool,
    /// True when policy should require approval unless explicitly allowed.
    pub default_requires_approval: bool,
    /// True when large schemas should be loaded lazily in future registry flows.
    pub should_defer_schema: bool,
    /// Default maximum model-facing output characters.
    pub max_output_chars: Option<usize>,
    /// Default maximum stored result characters.
    pub max_result_size_chars: Option<usize>,
}

impl Default for ToolSafetyMetadata {
    fn default() -> Self {
        Self::conservative_unknown()
    }
}

impl ToolSafetyMetadata {
    /// Returns fail-closed metadata for tools that do not declare safety details.
    pub fn conservative_unknown() -> Self {
        Self {
            read_only: false,
            concurrency_safe: false,
            operation: ToolOperationKind::Compute,
            side_effect_level: ToolSideEffectLevel::LocalWrite,
            requires_network: false,
            destructive: false,
            open_world: false,
            host_dependent: false,
            requires_user_interaction: false,
            supports_cancellation: false,
            default_requires_approval: true,
            should_defer_schema: false,
            max_output_chars: Some(20_000),
            max_result_size_chars: Some(20_000),
        }
    }

    /// Returns metadata for a read-only tool with the given operation kind.
    pub fn read_only(operation: ToolOperationKind) -> Self {
        Self {
            read_only: true,
            concurrency_safe: true,
            operation,
            side_effect_level: ToolSideEffectLevel::None,
            requires_network: false,
            destructive: false,
            open_world: false,
            host_dependent: false,
            requires_user_interaction: false,
            supports_cancellation: false,
            default_requires_approval: false,
            should_defer_schema: false,
            max_output_chars: Some(20_000),
            max_result_size_chars: Some(20_000),
        }
    }

    /// Returns metadata for deterministic or local compute tools.
    pub fn compute() -> Self {
        Self::read_only(ToolOperationKind::Compute)
    }
}

/// Call-level safety classification after arguments are known.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallClassification {
    /// Operation kind for this specific call.
    pub operation: ToolOperationKind,
    /// Side-effect level for this specific call.
    pub side_effect_level: ToolSideEffectLevel,
    /// True when this call is read-only.
    pub read_only: bool,
    /// True when this call may safely run in parallel.
    pub concurrency_safe: bool,
    /// True when this call can destroy or remove state.
    pub destructive: bool,
    /// True when this call can access the network.
    pub requires_network: bool,
    /// True when this call should ask for approval by default.
    pub requires_approval: bool,
    /// Optional timeout override in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Optional output cap for this call.
    pub max_output_chars: Option<usize>,
}

impl ToolCallClassification {
    pub fn from_metadata(metadata: &ToolSafetyMetadata) -> Self {
        Self {
            operation: metadata.operation,
            side_effect_level: metadata.side_effect_level,
            read_only: metadata.read_only,
            concurrency_safe: metadata.concurrency_safe,
            destructive: metadata.destructive,
            requires_network: metadata.requires_network,
            requires_approval: metadata.default_requires_approval,
            timeout_ms: None,
            max_output_chars: metadata.max_output_chars,
        }
    }
}

/// Runtime source that requested a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum ToolCallSource {
    Model,
    Skill {
        skill_id: String,
        step_index: usize,
    },
    StateAction {
        state: Option<String>,
        action_index: usize,
    },
    Plan {
        step_index: usize,
    },
    Orchestration,
    Spawner,
    Fallback {
        original_tool: String,
    },
    Task,
    Manual,
    EvalFixture,
}

/// Request passed into the shared tool execution boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRequest {
    /// Stable call ID used in records and history.
    pub call_id: String,
    /// Tool name, alias, or display name requested by the caller.
    pub requested_name: String,
    /// Original call arguments before approval or policy mutation.
    pub arguments: Value,
    /// Runtime path that requested this tool call.
    pub source: ToolCallSource,
}

impl ToolExecutionRequest {
    pub fn new(
        call_id: impl Into<String>,
        requested_name: impl Into<String>,
        arguments: Value,
        source: ToolCallSource,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            requested_name: requested_name.into(),
            arguments,
            source,
        }
    }
}

/// Policy decision stored with every tool execution record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicyDecisionRecord {
    pub outcome: PermissionOutcome,
    pub reason: Option<String>,
    pub requires_user_action: bool,
    pub retryable: bool,
}

impl ToolPolicyDecisionRecord {
    pub fn allow() -> Self {
        Self {
            outcome: PermissionOutcome::Allow,
            reason: None,
            requires_user_action: false,
            retryable: false,
        }
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            outcome: PermissionOutcome::Deny,
            reason: Some(reason.into()),
            requires_user_action: true,
            retryable: false,
        }
    }

    pub fn approval(reason: impl Into<String>) -> Self {
        Self {
            outcome: PermissionOutcome::RequiresApproval,
            reason: Some(reason.into()),
            requires_user_action: true,
            retryable: false,
        }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            outcome: PermissionOutcome::Unavailable,
            reason: Some(reason.into()),
            requires_user_action: true,
            retryable: false,
        }
    }
}

/// Approval status attached to a tool execution record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolApprovalStatus {
    NotRequired,
    Approved,
    Modified,
    Rejected,
    Timeout,
    Unavailable,
}

/// Human approval evidence for a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolApprovalRecord {
    pub status: ToolApprovalStatus,
    pub reason: Option<String>,
    pub modified_arguments: Option<Value>,
}

/// Structured internal result for a tool request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    /// Stable call ID copied from the request.
    pub call_id: String,
    /// Original name, display name, or alias requested by the caller.
    pub requested_name: String,
    /// Canonical registry ID used for policy and evidence.
    pub canonical_id: String,
    /// Runtime path that requested this call.
    pub source: ToolCallSource,
    /// Original call arguments.
    pub arguments: Value,
    /// Arguments actually used after approval changes.
    pub executed_arguments: Value,
    /// Policy snapshot version used for the decision.
    pub policy_version: u64,
    /// Registry snapshot version used for canonical resolution.
    pub registry_version: u64,
    /// Runtime-control snapshot version used for execution.
    pub runtime_config_version: u64,
    /// True only when the tool implementation was invoked.
    pub executed: bool,
    /// True when the model-facing result is successful.
    pub success: bool,
    /// Model-facing output or structured error text.
    pub output: String,
    /// Preserved tool metadata and executor annotations.
    pub metadata: HashMap<String, Value>,
    /// Permission decision for this call.
    pub policy: ToolPolicyDecisionRecord,
    /// Approval evidence when approval was checked.
    pub approval: Option<ToolApprovalRecord>,
    /// Start timestamp for observability and eval evidence.
    pub started_at: DateTime<Utc>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// True when execution exceeded the configured timeout.
    pub timed_out: bool,
    /// True when execution was cancelled by runtime control.
    pub cancelled: bool,
    /// Human-readable cancellation reason.
    pub cancellation_reason: Option<String>,
    /// True when model-facing output was truncated.
    pub output_truncated: bool,
}

impl ToolExecutionRecord {
    /// Returns the value that should be appended to model-visible tool history.
    pub fn model_output_value(&self) -> Value {
        if self.success {
            serde_json::from_str(&self.output)
                .unwrap_or_else(|_| Value::String(self.output.clone()))
        } else {
            serde_json::json!({
                "success": false,
                "error": {
                    "kind": match self.policy.outcome {
                        PermissionOutcome::Allow => "tool_error",
                        PermissionOutcome::Deny => "permission_denied",
                        PermissionOutcome::RequiresApproval => "approval_unavailable",
                        PermissionOutcome::Unavailable => "tool_unavailable",
                    },
                    "reason": self.policy.reason.clone().unwrap_or_else(|| self.output.clone()),
                    "retryable": self.policy.retryable,
                    "requires_user_action": self.policy.requires_user_action
                }
            })
        }
    }

    /// Returns the string form delivered to existing runtime callers.
    pub fn model_output_string(&self) -> String {
        if self.success {
            self.output.clone()
        } else {
            self.model_output_value().to_string()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub finish_reason: FinishReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LLMResponse {
    pub fn new(content: impl Into<String>, finish_reason: FinishReason) -> Self {
        Self {
            content: content.into(),
            finish_reason,
            usage: None,
            model: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    UserStopped,
    Error,
    Other,
}

impl FinishReason {
    pub fn is_complete(&self) -> bool {
        matches!(self, FinishReason::Stop | FinishReason::ToolCall)
    }

    pub fn is_error(&self) -> bool {
        matches!(self, FinishReason::Error | FinishReason::ContentFilter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    pub fn from_total(total_tokens: u32) -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMChunk {
    pub delta: String,
    pub is_final: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl LLMChunk {
    pub fn new(delta: impl Into<String>, is_final: bool) -> Self {
        Self {
            delta: delta.into(),
            is_final,
            finish_reason: None,
            usage: None,
        }
    }

    pub fn final_chunk(
        delta: impl Into<String>,
        finish_reason: FinishReason,
        usage: Option<TokenUsage>,
    ) -> Self {
        Self {
            delta: delta.into(),
            is_final: true,
            finish_reason: Some(finish_reason),
            usage,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_budget_tokens: Option<u32>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            max_tokens: Some(2048),
            top_p: Some(0.9),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            timeout_seconds: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
            extra: HashMap::new(),
        }
    }
}

impl LLMConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    pub fn with_timeout_seconds(mut self, timeout: u64) -> Self {
        self.timeout_seconds = Some(timeout);
        self
    }

    pub fn with_reasoning(mut self, enabled: bool) -> Self {
        self.reasoning = Some(enabled);
        self
    }

    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn with_reasoning_budget_tokens(mut self, budget: u32) -> Self {
        self.reasoning_budget_tokens = Some(budget);
        self
    }

    pub fn merge(mut self, other: &LLMConfig) -> Self {
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }
        if other.top_p.is_some() {
            self.top_p = other.top_p;
        }
        if other.top_k.is_some() {
            self.top_k = other.top_k;
        }
        if other.frequency_penalty.is_some() {
            self.frequency_penalty = other.frequency_penalty;
        }
        if other.presence_penalty.is_some() {
            self.presence_penalty = other.presence_penalty;
        }
        if other.stop_sequences.is_some() {
            self.stop_sequences = other.stop_sequences.clone();
        }
        if other.timeout_seconds.is_some() {
            self.timeout_seconds = other.timeout_seconds;
        }
        if other.reasoning.is_some() {
            self.reasoning = other.reasoning;
        }
        if other.reasoning_effort.is_some() {
            self.reasoning_effort = other.reasoning_effort.clone();
        }
        if other.reasoning_budget_tokens.is_some() {
            self.reasoning_budget_tokens = other.reasoning_budget_tokens;
        }
        for (k, v) in &other.extra {
            self.extra.insert(k.clone(), v.clone());
        }
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LLMFeature {
    Streaming,
    FunctionCalling,
    Vision,
    JsonMode,
    SystemMessages,
    BatchProcessing,
    FineTuning,
    Embeddings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl AgentInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: version.into(),
            description: None,
            capabilities: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl AgentResponse {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: None,
            tool_calls: None,
        }
    }

    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(calls);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let metadata = self.metadata.get_or_insert_with(HashMap::new);
        metadata.insert(key.into(), value);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionEvent {
    pub from: String,
    pub to: String,
    pub reason: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachineSnapshot {
    pub current_state: String,
    pub previous_state: Option<String>,
    pub turn_count: u32,
    pub no_transition_count: u32,
    pub history: Vec<StateTransitionEvent>,
}

//
// Key Facts types for session management and actor memory.
//

/// A single extracted fact about an actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFact {
    pub id: String,
    /// Which actor this fact is about. None means a general fact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    pub category: FactCategory,
    /// Fact content in natural language (always English for cross-language dedup).
    pub content: String,
    /// Extraction confidence from 0.0 to 1.0.
    pub confidence: f32,
    /// Importance score from 0.0 to 1.0. Reserved for time-based decay algorithms.
    #[serde(default = "default_salience")]
    pub salience: f32,
    pub extracted_at: DateTime<Utc>,
    /// Last time this fact was injected into context. Reserved for recency tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_accessed: Option<DateTime<Utc>>,
    /// Which message triggered this extraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_message_id: Option<String>,
    /// Language of the original conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_language: Option<String>,
}

fn default_salience() -> f32 {
    1.0
}

impl KeyFact {
    /// Priority score used for ranking and eviction.
    pub fn priority(&self) -> f32 {
        self.salience * self.confidence
    }
}

/// Built-in categories plus extensible custom categories.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FactCategory {
    UserPreference,
    UserContext,
    Decision,
    Agreement,
    #[serde(untagged)]
    Custom(String),
}

impl std::fmt::Display for FactCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactCategory::UserPreference => write!(f, "preference"),
            FactCategory::UserContext => write!(f, "context"),
            FactCategory::Decision => write!(f, "decision"),
            FactCategory::Agreement => write!(f, "agreement"),
            FactCategory::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// Filter for querying facts.
#[derive(Debug, Clone, Default)]
pub struct FactFilter {
    pub actor_id: Option<String>,
    pub category: Option<FactCategory>,
    pub min_confidence: Option<f32>,
    pub min_salience: Option<f32>,
    pub limit: Option<usize>,
}

//
// Session metadata types for actor memory and session lifecycle.
//

/// Metadata attached to a session for filtering, TTL, and actor tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Primary actor interacting in this session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    /// All actors that participated in this session.
    #[serde(default)]
    pub actors: Vec<String>,
    /// Freeform tags for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Custom metadata.
    #[serde(default)]
    pub custom: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub message_count: usize,
    /// Session TTL in seconds. None means no expiry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

impl Default for SessionMetadata {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            actor_id: None,
            actors: vec![],
            tags: vec![],
            custom: HashMap::new(),
            created_at: now,
            last_active: now,
            message_count: 0,
            ttl_seconds: None,
        }
    }
}

/// Filter for listing sessions.
#[derive(Debug, Clone, Default)]
pub struct SessionFilter {
    pub actor_id: Option<String>,
    pub tags: Option<Vec<String>>,
    pub agent_id: Option<String>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

/// Compact summary returned by list operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub agent_id: String,
    pub actor_id: Option<String>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub message_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_llm_config_merge() {
        let config1 = LLMConfig::new().with_temperature(0.5);
        let config2 = LLMConfig {
            temperature: None,
            max_tokens: Some(1000),
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            timeout_seconds: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
            extra: HashMap::new(),
        };
        let merged = config1.merge(&config2);
        assert_eq!(merged.temperature, Some(0.5));
        assert_eq!(merged.max_tokens, Some(1000));
    }

    #[test]
    fn test_llm_config_merge_reasoning_fields() {
        let base = LLMConfig::default().with_timeout_seconds(60);

        let overlay = LLMConfig {
            timeout_seconds: Some(120),
            reasoning: Some(true),
            reasoning_effort: Some("high".to_string()),
            reasoning_budget_tokens: Some(16384),
            ..LLMConfig::default()
        };

        let merged = base.merge(&overlay);
        assert_eq!(merged.timeout_seconds, Some(120));
        assert_eq!(merged.reasoning, Some(true));
        assert_eq!(merged.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(merged.reasoning_budget_tokens, Some(16384));
    }
}
