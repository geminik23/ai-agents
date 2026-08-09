use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    /// Optional timeout override in milliseconds for each invocation attempt.
    pub timeout_ms: Option<u64>,
    /// Optional output cap for this call.
    pub max_output_chars: Option<usize>,
    /// True when retrying the same call cannot repeat side effects.
    pub safely_retryable: bool,
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
            safely_retryable: metadata.read_only,
        }
    }
}

/// Effective tool limits derived from runtime defaults, policy, safety metadata, and call classification.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolExecutionLimits {
    /// Maximum wall-clock time in milliseconds for each invocation attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Maximum model-facing output characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_chars: Option<usize>,
    /// Maximum stored result characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_result_chars: Option<usize>,
    /// Maximum rows, matches, entries, or items returned by list-like tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
    /// Maximum local file bytes that a tool may read or inspect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_file_size_bytes: Option<u64>,
    /// Maximum response bytes accepted from network tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_response_bytes: Option<usize>,
    /// Maximum redirect hops for network tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_redirects: Option<usize>,
    /// Maximum exact replacements a mutation tool may perform.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_replacements: Option<usize>,
    /// Maximum files a future mutation tool may change in one call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_changed_files: Option<usize>,
    /// Maximum changed lines a future mutation tool may produce in one call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_changed_lines: Option<usize>,
}

impl ToolExecutionLimits {
    /// Returns a copy with each optional cap lowered by the matching cap from `other`.
    pub fn lowered_by(mut self, other: &Self) -> Self {
        self.timeout_ms = min_optional_u64(self.timeout_ms, other.timeout_ms);
        self.max_output_chars = min_optional_usize(self.max_output_chars, other.max_output_chars);
        self.max_result_chars = min_optional_usize(self.max_result_chars, other.max_result_chars);
        self.max_results = min_optional_usize(self.max_results, other.max_results);
        self.max_file_size_bytes =
            min_optional_u64(self.max_file_size_bytes, other.max_file_size_bytes);
        self.max_response_bytes =
            min_optional_usize(self.max_response_bytes, other.max_response_bytes);
        self.max_redirects = min_optional_usize(self.max_redirects, other.max_redirects);
        self.max_replacements = min_optional_usize(self.max_replacements, other.max_replacements);
        self.max_changed_files =
            min_optional_usize(self.max_changed_files, other.max_changed_files);
        self.max_changed_lines =
            min_optional_usize(self.max_changed_lines, other.max_changed_lines);
        self
    }
}

/// Turn actor and sender identity forwarded into tool execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolActorContext {
    /// Effective actor ID used for memory and user-scoped evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    /// Original user, customer, player, or top-level actor for the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_actor_id: Option<String>,
    /// Immediate agent sender for inter-agent hops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sender_agent_id: Option<String>,
}

/// Cooperative cancellation observer shared with a tool call.
#[derive(Clone)]
pub struct ToolCancellationToken {
    cancelled: Arc<AtomicBool>,
    reason: Option<String>,
}

impl std::fmt::Debug for ToolCancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCancellationToken")
            .field("cancelled", &self.is_cancelled())
            .field("reason", &self.reason)
            .finish()
    }
}

impl Default for ToolCancellationToken {
    fn default() -> Self {
        Self::new(Arc::new(AtomicBool::new(false)), None)
    }
}

impl ToolCancellationToken {
    /// Create a cancellation observer from shared runtime state.
    pub fn new(cancelled: Arc<AtomicBool>, reason: Option<String>) -> Self {
        Self { cancelled, reason }
    }

    /// Returns true when the runtime has requested cooperative cancellation.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Returns the current cancellation reason when one was provided.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

/// Context passed from the shared executor into one tool invocation attempt.
///
/// Retries clone the request-level fields, then refresh [`Self::deadline`] immediately before each `Tool::execute` invocation.
#[derive(Debug, Clone)]
pub struct ToolExecutionContext {
    /// Tool name, alias, or display name requested by the caller.
    pub requested_name: String,
    /// Canonical tool ID used for policy, HITL, recovery, and evidence.
    pub canonical_id: String,
    /// Display name resolved from the registry.
    pub display_name: String,
    /// Provider ID for provider-backed tools.
    pub provider_id: Option<String>,
    /// Registry snapshot version used for this call.
    pub registry_version: u64,
    /// Policy snapshot version used for this call.
    pub policy_version: u64,
    /// Runtime-control snapshot version used for this call.
    pub runtime_control_version: u64,
    /// Stable call ID used in records and history.
    pub call_id: String,
    /// Runtime path that requested this call.
    pub source: ToolCallSource,
    /// Actor and sender identity for the current turn.
    pub actor: ToolActorContext,
    /// Cooperative cancellation observer.
    pub cancellation: ToolCancellationToken,
    /// UTC time at which shared executor handling started across all retry attempts.
    pub started_at: DateTime<Utc>,
    /// UTC deadline for the current invocation attempt, refreshed before each retry.
    pub deadline: Option<DateTime<Utc>>,
    /// Permission decision that allowed this call to reach the tool.
    pub permission: ToolPolicyDecisionRecord,
    /// HITL decision evidence when approval was checked.
    pub approval: Option<ToolApprovalRecord>,
    /// Call-level safety classification after arguments are known.
    pub classification: ToolCallClassification,
    /// Static tool safety metadata from the resolved registration.
    pub safety: ToolSafetyMetadata,
    /// Effective limits that tools must treat as upper bounds.
    pub limits: ToolExecutionLimits,
    /// Raw per-tool policy snapshot when the runtime is allowed to expose it.
    pub policy_snapshot: Value,
    /// Custom settings from tool_security.tools.<tool_id>.config.
    pub custom_config: Value,
}

impl ToolExecutionContext {
    /// Builds a minimal context for unit tests and direct examples.
    pub fn test(tool_id: impl Into<String>) -> Self {
        let tool_id = tool_id.into();
        let started_at = Utc::now();
        Self {
            requested_name: tool_id.clone(),
            canonical_id: tool_id.clone(),
            display_name: tool_id.clone(),
            provider_id: None,
            registry_version: 0,
            policy_version: 0,
            runtime_control_version: 0,
            call_id: "test-call".to_string(),
            source: ToolCallSource::Manual,
            actor: ToolActorContext::default(),
            cancellation: ToolCancellationToken::default(),
            started_at,
            deadline: None,
            permission: ToolPolicyDecisionRecord::allow(),
            approval: None,
            classification: ToolCallClassification::from_metadata(&ToolSafetyMetadata::compute()),
            safety: ToolSafetyMetadata::compute(),
            limits: ToolExecutionLimits::default(),
            policy_snapshot: Value::Null,
            custom_config: Value::Null,
        }
    }
}

/// Path access mode declared by a tool policy binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathAccessMode {
    Read,
    Write,
    ReadWrite,
}

/// Path argument role declared by a tool policy binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathBindingKind {
    Path,
    BasePath,
    Cwd,
    PatchBase,
    MultiPath,
}

/// Declarative mapping from an input field to path policy checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathPolicyBinding {
    /// Dot path to the argument field containing the path value.
    pub field: String,
    /// Access mode requested for the field.
    pub mode: PathAccessMode,
    /// Role of the path in the tool request.
    pub kind: PathBindingKind,
    /// Default value used when the field is omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_path: Option<String>,
}

impl PathPolicyBinding {
    /// Declare a read-only path field.
    pub fn read(field: impl Into<String>) -> Self {
        Self::new(field, PathAccessMode::Read, PathBindingKind::Path)
    }

    /// Declare a write path field.
    pub fn write(field: impl Into<String>) -> Self {
        Self::new(field, PathAccessMode::Write, PathBindingKind::Path)
    }

    /// Declare a read-write path field.
    pub fn read_write(field: impl Into<String>) -> Self {
        Self::new(field, PathAccessMode::ReadWrite, PathBindingKind::Path)
    }

    /// Declare a path field with an explicit binding kind.
    pub fn new(field: impl Into<String>, mode: PathAccessMode, kind: PathBindingKind) -> Self {
        Self {
            field: field.into(),
            mode,
            kind,
            default_path: None,
        }
    }

    /// Attach a default path used for policy checks when the field is absent.
    pub fn with_default_path(mut self, default_path: impl Into<String>) -> Self {
        self.default_path = Some(default_path.into());
        self
    }
}

/// Declarative mapping from an input field to URL or domain policy checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainPolicyBinding {
    /// Dot path to the argument field containing a URL or host.
    pub field: String,
    /// True when the field is a full URL instead of a bare host.
    pub is_url: bool,
}

impl DomainPolicyBinding {
    /// Declare a URL field such as `url`.
    pub fn url(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            is_url: true,
        }
    }

    /// Declare a bare host or domain field.
    pub fn host(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            is_url: false,
        }
    }
}

/// Command argument role declared by a tool policy binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandBindingKind {
    CommandString,
    Argv,
    Cwd,
    Env,
    TemplateVariable,
}

/// Declarative mapping from an input field to command policy checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandPolicyBinding {
    /// Dot path to the argument field containing command data.
    pub field: String,
    /// Role of the command data in the request.
    pub kind: CommandBindingKind,
}

impl CommandPolicyBinding {
    /// Declare a command string field.
    pub fn command(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            kind: CommandBindingKind::CommandString,
        }
    }

    /// Declare an argv array field.
    pub fn argv(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            kind: CommandBindingKind::Argv,
        }
    }

    /// Declare a command working-directory field.
    pub fn cwd(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            kind: CommandBindingKind::Cwd,
        }
    }

    /// Declare a command environment object field.
    pub fn env(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            kind: CommandBindingKind::Env,
        }
    }
}

/// Limit kind declared by a result-limit policy binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultLimitKind {
    MaxResults,
    MaxLines,
    MaxOutputChars,
    MaxFileSizeBytes,
    MaxResponseBytes,
    MaxRedirects,
    MaxReplacements,
    MaxChangedFiles,
    MaxChangedLines,
    Pagination,
}

/// Declarative mapping from an input field to a common limit cap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultLimitBinding {
    /// Dot path to the argument field containing a numeric limit.
    pub field: String,
    /// Common limit applied to this field.
    pub kind: ResultLimitKind,
}

impl ResultLimitBinding {
    /// Declare a numeric cap field.
    pub fn new(field: impl Into<String>, kind: ResultLimitKind) -> Self {
        Self {
            field: field.into(),
            kind,
        }
    }
}

/// Tool-declared policy bindings used by the shared executor.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicyBindings {
    /// Path fields subject to path allow, deny, and approval policy.
    #[serde(default)]
    pub path_fields: Vec<PathPolicyBinding>,
    /// URL or host fields subject to domain policy.
    #[serde(default)]
    pub domain_fields: Vec<DomainPolicyBinding>,
    /// Command fields subject to command policy.
    #[serde(default)]
    pub command_fields: Vec<CommandPolicyBinding>,
    /// Operation selector fields subject to operation policy.
    #[serde(default)]
    pub operation_fields: Vec<String>,
    /// Request limit fields that should be capped by effective policy.
    #[serde(default)]
    pub result_limit_fields: Vec<ResultLimitBinding>,
}

impl ToolPolicyBindings {
    /// Returns true when the tool exposes any path binding.
    pub fn has_path_bindings(&self) -> bool {
        !self.path_fields.is_empty()
    }

    /// Returns true when the tool exposes any URL or domain binding.
    pub fn has_domain_bindings(&self) -> bool {
        !self.domain_fields.is_empty()
    }

    /// Returns true when the tool exposes any command binding.
    pub fn has_command_bindings(&self) -> bool {
        !self.command_fields.is_empty()
    }

    /// Returns true when the tool exposes any operation selector binding.
    pub fn has_operation_bindings(&self) -> bool {
        !self.operation_fields.is_empty()
    }

    /// Returns true when the tool exposes any request limit binding.
    pub fn has_result_limit_bindings(&self) -> bool {
        !self.result_limit_fields.is_empty()
    }
}

fn min_optional_usize(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn min_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
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

/// Structured internal result for one logical shared-executor request.
///
/// Retry invocations are folded into this one request-level record. A configured fallback finalizes the failed original record, then produces a separate record with [`ToolCallSource::Fallback`].
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
    /// Wall-clock duration for the logical request, including all retry attempts.
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

/// Provider-neutral policy for native tool selection.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ToolChoice {
    Auto,
    Required,
    Specific(String),
    None,
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::None => serializer.serialize_str("none"),
            Self::Specific(tool_id) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("specific", tool_id)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(value) => match value.as_str() {
                "auto" => Ok(Self::Auto),
                "required" => Ok(Self::Required),
                "none" => Ok(Self::None),
                _ => Err(serde::de::Error::custom(
                    "tool_choice must be auto, required, none, or { specific: <canonical_id> }",
                )),
            },
            serde_json::Value::Object(mut map) if map.len() == 1 => {
                let tool_id = map
                    .remove("specific")
                    .and_then(|value| value.as_str().map(str::to_string))
                    .filter(|tool_id| !tool_id.is_empty())
                    .ok_or_else(|| {
                        serde::de::Error::custom(
                            "specific tool_choice requires a non-empty canonical tool ID",
                        )
                    })?;
                Ok(Self::Specific(tool_id))
            }
            _ => Err(serde::de::Error::custom(
                "tool_choice must be auto, required, none, or { specific: <canonical_id> }",
            )),
        }
    }
}

/// Provider-neutral function tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Native tool completion request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMToolRequest {
    pub tools: Vec<LLMToolDefinition>,
    pub choice: ToolChoice,
}

const LLM_TOOL_CALLS_METADATA_KEY: &str = "_ai_agents_native_tool_calls";

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

    /// Stores normalized native tool calls under the framework-reserved metadata key.
    pub fn set_tool_calls(
        &mut self,
        calls: Vec<ToolCall>,
    ) -> Result<(), crate::traits::llm::LLMError> {
        let value = serde_json::to_value(calls).map_err(crate::traits::llm::LLMError::from)?;
        self.metadata
            .insert(LLM_TOOL_CALLS_METADATA_KEY.to_string(), value);
        Ok(())
    }

    /// Stores normalized native tool calls and returns the response.
    pub fn with_tool_calls(
        mut self,
        calls: Vec<ToolCall>,
    ) -> Result<Self, crate::traits::llm::LLMError> {
        self.set_tool_calls(calls)?;
        Ok(self)
    }

    /// Reads normalized native tool calls from framework metadata.
    pub fn tool_calls(&self) -> Result<Option<Vec<ToolCall>>, crate::traits::llm::LLMError> {
        let Some(value) = self.metadata.get(LLM_TOOL_CALLS_METADATA_KEY) else {
            return Ok(None);
        };

        serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|error| {
                crate::traits::llm::LLMError::Serialization(format!(
                    "invalid native tool call metadata: {error}"
                ))
            })
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    fn test_llm_response_tool_calls_round_trip() {
        let calls = vec![ToolCall {
            id: "call-1".to_string(),
            name: "calculator".to_string(),
            arguments: serde_json::json!({"expression": "2 + 2"}),
        }];
        let response = LLMResponse::new("", FinishReason::ToolCall)
            .with_tool_calls(calls.clone())
            .unwrap();

        assert_eq!(response.tool_calls().unwrap(), Some(calls));
    }

    #[test]
    fn test_llm_response_rejects_corrupt_tool_call_metadata() {
        let mut response = LLMResponse::new("", FinishReason::ToolCall);
        response.metadata.insert(
            LLM_TOOL_CALLS_METADATA_KEY.to_string(),
            serde_json::json!({"not": "an array"}),
        );

        let error = response.tool_calls().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid native tool call metadata")
        );
    }

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
