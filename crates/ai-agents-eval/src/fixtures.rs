use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use ai_agents_core::{
    ChatMessage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse, Tool,
    ToolExecutionContext, ToolPolicyBindings, ToolResult,
};
use ai_agents_hitl::{ApprovalHandler, ApprovalRequest, ApprovalResult, ApprovalTrigger};
use ai_agents_llm::providers::{ProviderType, UnifiedLLMProvider};
use ai_agents_llm::{FinishReason, LLMRegistry};
use ai_agents_runtime::spec::AgentSpec;
use ai_agents_tools::{
    CommandResponse, DiagnosticItem, DiagnosticsProvider, StaticCommandRunner,
    StaticDiagnosticsProvider, StaticWebSearchProvider, ToolRegistry,
    UnavailableDiagnosticsProvider, UnavailableWebSearchProvider, WebFetchResolver, WebFetchTool,
    WebFetchTransport, WebFetchTransportRequest, WebFetchTransportResponse, WebSearchProvider,
    WebSearchResponse, create_builtin_registry,
};
use async_trait::async_trait;
use futures::Stream;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::evidence::{ToolExecutionRecord, ToolExecutionSource};
use crate::{EvalError, Result};

/// Fixture configuration used to replace external dependencies during eval.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FixturesConfig {
    /// Runtime or fixture context value.
    #[serde(default)]
    pub context: Option<Value>,
    /// JSON context file resolved relative to the suite file.
    #[serde(default)]
    pub context_file: Option<PathBuf>,
    /// Mock tool definitions keyed by tool ID.
    #[serde(default)]
    pub tools: HashMap<String, ToolMockConfig>,
    /// Optional LLM alias or provider used for judge calls.
    #[serde(default)]
    pub llm: LlmFixtureConfig,
    /// Optional local HTTP mock server configuration.
    #[serde(default)]
    pub mock_server: Option<MockServerConfig>,
    /// Optional mocked diagnostics returned by the diagnostics tool.
    #[serde(default)]
    pub diagnostics: Option<DiagnosticsFixtureConfig>,
    /// Optional mocked command-runner responses for the command tool.
    #[serde(default)]
    pub commands: Option<CommandsFixtureConfig>,
    /// Optional mocked web-search responses for the web_search tool.
    #[serde(default)]
    pub web_search: Option<WebSearchFixtureConfig>,
    /// Optional exact-URL in-memory transport for the real web_fetch tool.
    #[serde(default)]
    pub web_fetch_transport: Option<WebFetchTransportFixtureConfig>,
    /// Deterministic approval responses for human-in-the-loop requests.
    #[serde(default)]
    pub approvals: Option<ApprovalFixtureConfig>,
    /// Narrow per-attempt workspace additions for existing tool policies.
    #[serde(default)]
    pub workspace_policy: Option<WorkspacePolicyFixtureConfig>,
}

/// Existing tool policies that receive the isolated attempt workspace.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct WorkspacePolicyFixtureConfig {
    /// Tool policies that receive the workspace in read_paths.
    #[serde(default)]
    pub read_tools: Vec<String>,
    /// Tool policies that receive the workspace in write_paths.
    #[serde(default)]
    pub write_tools: Vec<String>,
}

/// Deterministic approval behavior for one eval handler instance.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApprovalFixtureConfig {
    /// Rules evaluated in declaration order.
    #[serde(default)]
    pub rules: Vec<ApprovalFixtureRule>,
    /// Result returned when no rule matches.
    #[serde(default)]
    pub default: ApprovalFixtureOutcome,
    /// Preferred language advertised to the HITL message resolver.
    #[serde(default)]
    pub preferred_language: Option<String>,
    /// Supported languages advertised to the HITL message resolver.
    #[serde(default)]
    pub supported_languages: Option<Vec<String>>,
}

/// One ordered approval rule with an optional 1-based occurrence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalFixtureRule {
    /// Trigger fields matched against the approval request.
    pub trigger: ApprovalFixtureTrigger,
    /// Match only this occurrence of the trigger predicate.
    #[serde(default)]
    pub occurrence: Option<NonZeroUsize>,
    /// Result returned when the rule matches.
    #[serde(flatten)]
    pub outcome: ApprovalFixtureOutcome,
}

/// Approval trigger predicate used by an eval fixture rule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApprovalFixtureTrigger {
    Tool {
        name: String,
        #[serde(default)]
        args: Option<Value>,
    },
    Condition {
        name: String,
        #[serde(default)]
        matched: Option<String>,
    },
    StateTransition {
        #[serde(default)]
        from: Option<String>,
        to: String,
    },
    DisambiguationEscalation {
        #[serde(default)]
        reason: Option<String>,
    },
}

/// Approval result produced by a fixture rule or fallback.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ApprovalFixtureOutcome {
    Approve,
    Reject {
        #[serde(default)]
        reason: Option<String>,
    },
    Modify {
        #[serde(default)]
        changes: HashMap<String, Value>,
    },
    Timeout,
    #[default]
    Unavailable,
}

/// Eval approval handler with isolated, concurrency-safe occurrence counters.
pub struct FixtureApprovalHandler {
    config: ApprovalFixtureConfig,
    occurrences: Mutex<Vec<usize>>,
}

impl FixtureApprovalHandler {
    pub fn new(config: ApprovalFixtureConfig) -> Self {
        let rule_count = config.rules.len();
        Self {
            config,
            occurrences: Mutex::new(vec![0; rule_count]),
        }
    }

    fn select_outcome(&self, request: &ApprovalRequest) -> ApprovalFixtureOutcome {
        let mut occurrences = self.occurrences.lock();
        for (index, rule) in self.config.rules.iter().enumerate() {
            if !rule.trigger.matches(&request.trigger) {
                continue;
            }
            occurrences[index] += 1;
            if rule
                .occurrence
                .is_none_or(|occurrence| occurrence.get() == occurrences[index])
            {
                return rule.outcome.clone();
            }
        }
        self.config.default.clone()
    }
}

impl ApprovalFixtureTrigger {
    fn matches(&self, trigger: &ApprovalTrigger) -> bool {
        match (self, trigger) {
            (
                Self::Tool { name, args },
                ApprovalTrigger::Tool {
                    name: actual_name,
                    args: actual_args,
                },
            ) => name == actual_name && args.as_ref().is_none_or(|args| args == actual_args),
            (
                Self::Condition { name, matched },
                ApprovalTrigger::Condition {
                    name: actual_name,
                    matched: actual_matched,
                },
            ) => {
                name != "disambiguation_escalation"
                    && name == actual_name
                    && matched
                        .as_ref()
                        .is_none_or(|matched| matched == actual_matched)
            }
            (
                Self::StateTransition { from, to },
                ApprovalTrigger::State {
                    from: actual_from,
                    to: actual_to,
                },
            ) => from == actual_from && to == actual_to,
            (
                Self::DisambiguationEscalation { reason },
                ApprovalTrigger::Condition {
                    name,
                    matched: actual_reason,
                },
            ) => {
                name == "disambiguation_escalation"
                    && reason.as_ref().is_none_or(|reason| reason == actual_reason)
            }
            _ => false,
        }
    }
}

impl From<ApprovalFixtureOutcome> for ApprovalResult {
    fn from(outcome: ApprovalFixtureOutcome) -> Self {
        match outcome {
            ApprovalFixtureOutcome::Approve => Self::approved(),
            ApprovalFixtureOutcome::Reject { reason } => Self::rejected(reason),
            ApprovalFixtureOutcome::Modify { changes } => Self::modified(changes),
            ApprovalFixtureOutcome::Timeout => Self::timeout(),
            ApprovalFixtureOutcome::Unavailable => {
                Self::rejected_with_reason("Approval fixture unavailable")
            }
        }
    }
}

#[async_trait]
impl ApprovalHandler for FixtureApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> ApprovalResult {
        self.select_outcome(&request).into()
    }

    fn preferred_language(&self) -> Option<String> {
        self.config.preferred_language.clone()
    }

    fn supported_languages(&self) -> Option<Vec<String>> {
        self.config.supported_languages.clone()
    }
}

/// Build a fresh approval handler with state isolated to this invocation.
pub fn build_approval_handler(config: &ApprovalFixtureConfig) -> Arc<dyn ApprovalHandler> {
    Arc::new(FixtureApprovalHandler::new(config.clone()))
}

/// Diagnostics fixture configuration for host-backed diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsFixtureConfig {
    /// Whether the provider is available.
    #[serde(default = "default_true")]
    pub available: bool,
    /// Deterministic diagnostics returned by the provider.
    #[serde(default)]
    pub items: Vec<DiagnosticItem>,
}

/// Command-runner fixture configuration for process-backed validation tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandsFixtureConfig {
    /// Whether the command runner is available.
    #[serde(default = "default_true")]
    pub available: bool,
    /// Deterministic exact-argv command responses.
    #[serde(default)]
    pub entries: Vec<CommandFixtureEntry>,
}

/// One exact-argv mocked command response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandFixtureEntry {
    /// Full argv vector used for exact matching.
    #[serde(default)]
    pub argv: Vec<String>,
    /// Mocked command response returned to the tool.
    #[serde(default)]
    pub response: CommandResponse,
}

/// One exact-query mocked web-search response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchFixtureEntry {
    /// Query string matched exactly by the static provider.
    pub query: String,
    /// Mocked search response returned to the tool.
    #[serde(default)]
    pub response: WebSearchResponse,
}

/// Web-search fixture configuration for provider-neutral search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchFixtureConfig {
    /// Whether the provider is available.
    #[serde(default = "default_true")]
    pub available: bool,
    /// Deterministic exact-query search responses.
    #[serde(default)]
    pub entries: Vec<WebSearchFixtureEntry>,
}

/// Exact-URL responses served in memory to the real web_fetch tool.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebFetchTransportFixtureConfig {
    /// Routes matched against the normalized URL requested by web_fetch.
    #[serde(default)]
    pub routes: Vec<WebFetchTransportFixtureRoute>,
}

/// One no-socket web_fetch transport response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFetchTransportFixtureRoute {
    /// Absolute URL matched exactly after URL normalization.
    pub url: String,
    /// HTTP status returned by the transport.
    #[serde(default = "default_status")]
    pub status: u16,
    /// Response headers. Content-Type and Location retain their web semantics.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// String bodies are returned verbatim; other values are encoded as JSON.
    #[serde(default)]
    pub body: Value,
}

/// Static output configuration for an eval mock tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMockConfig {
    /// Whether the operation succeeded.
    #[serde(default = "default_true")]
    pub success: bool,
    /// Directory where output artifacts are written.
    #[serde(default)]
    pub output: Value,
}

impl Default for ToolMockConfig {
    fn default() -> Self {
        Self {
            success: true,
            output: Value::Null,
        }
    }
}

/// LLM fixture mode and data used by the eval runner.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmFixtureConfig {
    /// LLM fixture mode used for configured aliases.
    #[serde(default)]
    pub mode: LlmFixtureMode,
    /// Optional cassette JSONL file for replay or record mode.
    #[serde(default)]
    pub cassette: Option<PathBuf>,
    /// Ordered text responses used by mock mode and fallback replay.
    #[serde(default)]
    pub responses: Vec<String>,
    /// Per-LLM alias ordered responses for deterministic multi-branch evals.
    #[serde(default)]
    pub responses_by_alias: HashMap<String, Vec<String>>,
    /// Per-LLM alias errors for deterministic failure-path evals.
    #[serde(default)]
    pub errors_by_alias: HashMap<String, String>,
    /// Per-LLM alias ordered response and error outcomes.
    #[serde(default)]
    pub outcomes_by_alias: HashMap<String, Vec<LlmFixtureOutcome>>,
    /// Per-LLM alias delay in milliseconds for deterministic branch ordering.
    #[serde(default)]
    pub delays_by_alias: HashMap<String, u64>,
}

/// One response or error in an ordered mock LLM sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LlmFixtureOutcome {
    Response {
        content: String,
    },
    Error {
        message: String,
        #[serde(default)]
        status: Option<u16>,
    },
}

/// LLM fixture strategy used while building eval providers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LlmFixtureMode {
    #[default]
    Real,
    Mock,
    Replay,
    Record,
}

/// Local HTTP mock server configuration for eval scenarios.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MockServerConfig {
    /// Whether this feature is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Optional fixed port. Zero or none requests a dynamic port.
    #[serde(default)]
    pub port: Option<u16>,
    /// Route definitions served by the mock server.
    #[serde(default)]
    pub routes: Vec<Value>,
}

/// One route served by the lightweight eval mock server.
#[derive(Debug, Clone, Deserialize)]
struct MockRoute {
    /// HTTP method matched by this route.
    method: String,
    /// Path used for file lookup, HTTP routing, or dot-path checks.
    path: String,
    /// Final or normalized status value.
    #[serde(default = "default_status")]
    status: u16,
    /// Extra response headers returned by this route.
    #[serde(default)]
    headers: HashMap<String, String>,
    /// JSON or string body returned by this route.
    #[serde(default)]
    body: Value,
}

/// Running mock server handle that stops the server on drop.
pub struct MockServerHandle {
    /// Base URL injected into eval context.
    base_url: String,
    /// Background accept loop for the mock server.
    task: JoinHandle<()>,
}

impl MockServerHandle {
    pub fn context(&self) -> HashMap<String, Value> {
        let mut context = HashMap::new();
        context.insert(
            "mock_server".to_string(),
            serde_json::json!({"base_url": self.base_url}),
        );
        context
    }
}

impl Drop for MockServerHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Generated values shared by every agent build and reset in one attempt.
#[derive(Debug, PartialEq, Eq)]
pub struct AttemptFixtureContext {
    /// Opaque identifier used to isolate non-file persistence.
    pub isolation_id: String,
    /// Absolute per-attempt workspace exposed as eval.workspace.
    pub workspace: PathBuf,
    /// Optional server URL exposed as mock_server.base_url.
    pub mock_server_base_url: Option<String>,
}

impl AttemptFixtureContext {
    pub fn create(mock_server: Option<&MockServerHandle>) -> Result<Self> {
        let isolation_id = uuid::Uuid::new_v4().to_string();
        let workspace = std::env::temp_dir().join(format!("ai_agents_eval_{}", isolation_id));
        std::fs::create_dir_all(&workspace)?;
        let workspace = workspace.canonicalize()?;
        Ok(Self {
            isolation_id,
            workspace,
            mock_server_base_url: mock_server.map(|server| server.base_url.clone()),
        })
    }

    pub fn runtime_context(&self) -> HashMap<String, Value> {
        let mut context = HashMap::from([(
            "eval".to_string(),
            json!({"workspace": self.workspace.display().to_string()}),
        )]);
        if let Some(base_url) = &self.mock_server_base_url {
            context.insert("mock_server".to_string(), json!({"base_url": base_url}));
        }
        context
    }

    pub fn interpolate_llm_fixture(&self, config: &LlmFixtureConfig) -> Result<LlmFixtureConfig> {
        let mut rewritten = config.clone();
        for response in &mut rewritten.responses {
            *response = self.interpolate_response(response)?;
        }
        for responses in rewritten.responses_by_alias.values_mut() {
            for response in responses {
                *response = self.interpolate_response(response)?;
            }
        }
        for outcomes in rewritten.outcomes_by_alias.values_mut() {
            for outcome in outcomes {
                if let LlmFixtureOutcome::Response { content } = outcome {
                    *content = self.interpolate_response(content)?;
                }
            }
        }
        Ok(rewritten)
    }

    fn interpolate_response(&self, response: &str) -> Result<String> {
        const WORKSPACE_TOKEN: &str = "{{ eval.workspace }}";
        const MOCK_SERVER_TOKEN: &str = "{{ mock_server.base_url }}";
        if !response.contains(WORKSPACE_TOKEN) && !response.contains(MOCK_SERVER_TOKEN) {
            return Ok(response.to_string());
        }
        if response.contains(MOCK_SERVER_TOKEN) && self.mock_server_base_url.is_none() {
            return Err(EvalError::Config(format!(
                "mock LLM response uses {} without an enabled fixtures.mock_server",
                MOCK_SERVER_TOKEN
            )));
        }
        let workspace = self.workspace.display().to_string();
        let mock_server = self.mock_server_base_url.as_deref();
        if let Ok(mut value) = serde_json::from_str::<Value>(response) {
            interpolate_json_strings(&mut value, &workspace, mock_server);
            return serde_json::to_string(&value).map_err(EvalError::from);
        }
        Ok(interpolate_fixture_string(
            response,
            &workspace,
            mock_server,
        ))
    }
}

fn interpolate_json_strings(value: &mut Value, workspace: &str, mock_server: Option<&str>) {
    match value {
        Value::String(text) => *text = interpolate_fixture_string(text, workspace, mock_server),
        Value::Array(values) => {
            for value in values {
                interpolate_json_strings(value, workspace, mock_server);
            }
        }
        Value::Object(values) => {
            let previous = std::mem::take(values);
            for (key, mut value) in previous {
                interpolate_json_strings(&mut value, workspace, mock_server);
                values.insert(
                    interpolate_fixture_string(&key, workspace, mock_server),
                    value,
                );
            }
        }
        _ => {}
    }
}

fn interpolate_fixture_string(value: &str, workspace: &str, mock_server: Option<&str>) -> String {
    let value = value.replace("{{ eval.workspace }}", workspace);
    match mock_server {
        Some(base_url) => value.replace("{{ mock_server.base_url }}", base_url),
        None => value,
    }
}

/// Shared in-memory log of tool executions for one attempt.
#[derive(Clone, Default)]
pub struct RecordingToolLog {
    /// Wrapped implementation or shared storage.
    inner: Arc<Mutex<Vec<ToolExecutionRecord>>>,
}

impl From<&ai_agents_core::ToolCallSource> for ToolExecutionSource {
    fn from(source: &ai_agents_core::ToolCallSource) -> Self {
        match source {
            ai_agents_core::ToolCallSource::Model => Self::Llm,
            ai_agents_core::ToolCallSource::Skill { .. } => Self::Skill,
            ai_agents_core::ToolCallSource::Plan { .. } => Self::Plan,
            ai_agents_core::ToolCallSource::StateAction { .. } => Self::StateAction,
            ai_agents_core::ToolCallSource::Orchestration => Self::Orchestration,
            ai_agents_core::ToolCallSource::Spawner => Self::Spawner,
            ai_agents_core::ToolCallSource::EvalFixture => Self::Mock,
            ai_agents_core::ToolCallSource::Fallback { .. }
            | ai_agents_core::ToolCallSource::Task
            | ai_agents_core::ToolCallSource::Manual => Self::Llm,
        }
    }
}

fn state_from_source(source: &ai_agents_core::ToolCallSource) -> Option<String> {
    match source {
        ai_agents_core::ToolCallSource::StateAction { state, .. } => state.clone(),
        _ => None,
    }
}

impl RecordingToolLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn push(&self, record: ToolExecutionRecord) {
        self.inner.lock().push(record);
    }

    /// Record evidence produced by the shared runtime executor.
    pub fn push_executor_record(&self, record: &ai_agents_core::ToolExecutionRecord) {
        let output = record.success.then(|| {
            serde_json::from_str(&record.output).unwrap_or(Value::String(record.output.clone()))
        });
        let mut metadata = record.metadata.clone();
        metadata.insert("executed".to_string(), Value::Bool(record.executed));
        metadata.insert(
            "policy".to_string(),
            serde_json::to_value(&record.policy).unwrap_or(Value::Null),
        );
        metadata.insert(
            "approval".to_string(),
            serde_json::to_value(&record.approval).unwrap_or(Value::Null),
        );
        metadata.insert("timed_out".to_string(), Value::Bool(record.timed_out));
        metadata.insert("cancelled".to_string(), Value::Bool(record.cancelled));
        self.push(ToolExecutionRecord {
            call_id: record.call_id.clone(),
            tool_id: record.canonical_id.clone(),
            requested_name: record.requested_name.clone(),
            source: ToolExecutionSource::from(&record.source),
            state: state_from_source(&record.source),
            actor_id: None,
            arguments_original: record.arguments.clone(),
            arguments_executed: record.executed_arguments.clone(),
            executed: record.executed,
            success: record.success,
            output,
            error: (!record.success).then_some(record.output.clone()),
            metadata: Some(serde_json::to_value(metadata).unwrap_or(Value::Null)),
            started_at: record.started_at,
            duration_ms: record.duration_ms,
            observability_span_id: None,
        });
    }

    pub fn records_since(&self, index: usize) -> Vec<ToolExecutionRecord> {
        self.inner.lock().iter().skip(index).cloned().collect()
    }
}

pub fn resolve_fixture_context(
    config: &FixturesConfig,
    base_dir: &Path,
) -> Result<HashMap<String, Value>> {
    let mut result = HashMap::new();
    if let Some(path) = &config.context_file {
        let resolved = resolve_path(base_dir, path);
        let content = std::fs::read_to_string(&resolved).map_err(|error| {
            EvalError::Config(format!(
                "failed to read context_file '{}': {}",
                resolved.display(),
                error
            ))
        })?;
        let value: Value = serde_json::from_str(&content).map_err(|error| {
            EvalError::Config(format!(
                "failed to parse context_file '{}': {}",
                resolved.display(),
                error
            ))
        })?;
        merge_object_into_map(&mut result, value)?;
    }
    if let Some(value) = &config.context {
        merge_object_into_map(&mut result, value.clone())?;
    }
    Ok(result)
}

fn merge_object_into_map(target: &mut HashMap<String, Value>, value: Value) -> Result<()> {
    let Value::Object(map) = value else {
        return Err(EvalError::Config(
            "fixture context must be a JSON object".into(),
        ));
    };
    for (key, value) in map {
        target.insert(key, value);
    }
    Ok(())
}

fn resolve_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

pub async fn start_mock_server(
    config: Option<&MockServerConfig>,
) -> Result<Option<MockServerHandle>> {
    let Some(config) = config else {
        return Ok(None);
    };
    if !config.enabled {
        return Ok(None);
    }
    let routes = config
        .routes
        .iter()
        .cloned()
        .map(serde_json::from_value::<MockRoute>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let port = config.port.unwrap_or(0);
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .map_err(|error| EvalError::Runtime(format!("failed to start mock server: {}", error)))?;
    let addr = listener.local_addr().map_err(|error| {
        EvalError::Runtime(format!("failed to read mock server addr: {}", error))
    })?;
    let base_url = format!("http://{}", addr);
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let routes = routes.clone();
            tokio::spawn(async move {
                let _ = handle_mock_connection(stream, routes).await;
            });
        }
    });
    Ok(Some(MockServerHandle { base_url, task }))
}

async fn handle_mock_connection(
    mut stream: tokio::net::TcpStream,
    routes: Vec<MockRoute>,
) -> std::io::Result<()> {
    let mut buffer = vec![0_u8; 8192];
    let read = stream.read(&mut buffer).await?;
    let request = String::from_utf8_lossy(&buffer[..read]);
    let first_line = request.lines().next().unwrap_or_default();
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();
    let route = routes
        .iter()
        .find(|route| route.method.eq_ignore_ascii_case(method) && route.path == path);
    let (status, headers, body) = if let Some(route) = route {
        (
            route.status,
            route.headers.clone(),
            mock_body_to_string(&route.body),
        )
    } else {
        (
            404,
            HashMap::new(),
            serde_json::json!({"error":"not found"}).to_string(),
        )
    };
    let reason = match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let mut response = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n",
        status,
        reason,
        body.len()
    );
    for (key, value) in headers {
        response.push_str(&format!("{}: {}\r\n", key, value));
    }
    response.push_str("\r\n");
    response.push_str(&body);
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

fn mock_body_to_string(body: &Value) -> String {
    if let Some(text) = body.as_str() {
        text.to_string()
    } else {
        serde_json::to_string(body).unwrap_or_else(|_| "null".to_string())
    }
}

pub fn build_tool_registry(
    fixtures: &FixturesConfig,
    log: RecordingToolLog,
) -> Result<ToolRegistry> {
    let builtin = create_builtin_registry();
    let mut registry = ToolRegistry::new();
    let provider: Arc<dyn DiagnosticsProvider> = if let Some(diagnostics) = &fixtures.diagnostics {
        Arc::new(StaticDiagnosticsProvider::with_availability(
            diagnostics.items.clone(),
            diagnostics.available,
        ))
    } else {
        Arc::new(UnavailableDiagnosticsProvider)
    };
    builtin.set_diagnostics_provider(provider.clone());
    registry.set_diagnostics_provider(provider);

    if let Some(commands) = &fixtures.commands {
        let responses = commands
            .entries
            .iter()
            .map(|entry| (entry.argv.clone(), entry.response.clone()))
            .collect();
        let runner = Arc::new(StaticCommandRunner::with_availability(
            responses,
            commands.available,
        ));
        builtin.set_command_runner(runner.clone());
        registry.set_command_runner(runner);
    }

    let search_provider: Arc<dyn WebSearchProvider> = if let Some(web_search) = &fixtures.web_search
    {
        let responses = web_search
            .entries
            .iter()
            .map(|entry| (entry.query.clone(), entry.response.clone()))
            .collect();
        Arc::new(StaticWebSearchProvider::with_availability(
            responses,
            web_search.available,
        ))
    } else {
        Arc::new(UnavailableWebSearchProvider)
    };
    builtin.set_web_search_provider(search_provider.clone());
    registry.set_web_search_provider(search_provider);

    for (id, mock) in &fixtures.tools {
        let contract = builtin.get(id);
        registry
            .register(Arc::new(RecordingTool::new(
                Arc::new(MockTool::new(id.clone(), mock.clone(), contract)),
                log.clone(),
                ToolExecutionSource::Mock,
            )))
            .map_err(|error| EvalError::Config(error.to_string()))?;
    }

    let web_fetch = fixtures
        .web_fetch_transport
        .as_ref()
        .map(build_web_fetch_fixture_tool)
        .transpose()?;
    for id in builtin.list_ids() {
        if fixtures.tools.contains_key(&id) {
            continue;
        }
        let tool = if id == "web_fetch" {
            web_fetch.clone().or_else(|| builtin.get(&id))
        } else {
            builtin.get(&id)
        };
        if let Some(tool) = tool {
            registry
                .register(Arc::new(RecordingTool::new(
                    tool,
                    log.clone(),
                    ToolExecutionSource::Llm,
                )))
                .map_err(|error| EvalError::Config(error.to_string()))?;
        }
    }

    Ok(registry)
}

fn build_web_fetch_fixture_tool(config: &WebFetchTransportFixtureConfig) -> Result<Arc<dyn Tool>> {
    let mut routes = HashMap::new();
    for route in &config.routes {
        let normalized_url = reqwest::Url::parse(&route.url)
            .map_err(|error| {
                EvalError::Config(format!(
                    "invalid fixtures.web_fetch_transport route URL '{}': {}",
                    route.url, error
                ))
            })?
            .to_string();
        let content_type = header_value(&route.headers, "content-type").map(str::to_string);
        let location = header_value(&route.headers, "location").map(str::to_string);
        let response = WebFetchTransportResponse {
            status: route.status,
            content_type,
            location,
            body: mock_body_to_string(&route.body).into_bytes(),
        };
        if routes.insert(normalized_url.clone(), response).is_some() {
            return Err(EvalError::Config(format!(
                "duplicate fixtures.web_fetch_transport route URL '{}'",
                normalized_url
            )));
        }
    }
    Ok(Arc::new(WebFetchTool::with_transport_and_resolver(
        Arc::new(FixtureWebFetchTransport { routes }),
        Arc::new(FixtureWebFetchResolver),
    )))
}

fn header_value<'a>(headers: &'a HashMap<String, String>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

struct FixtureWebFetchTransport {
    routes: HashMap<String, WebFetchTransportResponse>,
}

#[async_trait]
impl WebFetchTransport for FixtureWebFetchTransport {
    async fn send(
        &self,
        request: WebFetchTransportRequest,
    ) -> std::result::Result<WebFetchTransportResponse, String> {
        self.routes
            .get(&request.url)
            .cloned()
            .ok_or_else(|| format!("No fixtures.web_fetch_transport route for {}", request.url))
    }
}

struct FixtureWebFetchResolver;

#[async_trait]
impl WebFetchResolver for FixtureWebFetchResolver {
    async fn resolve(&self, _host: &str, _port: u16) -> std::result::Result<Vec<IpAddr>, String> {
        Ok(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))])
    }
}

/// Tool implementation returning a configured eval fixture result.
struct MockTool {
    /// Stable identifier for this item.
    id: String,
    /// Configuration used by this component.
    config: ToolMockConfig,
    /// Built-in contract preserved while fixture execution replaces the implementation.
    contract: Option<Arc<dyn Tool>>,
}

impl MockTool {
    fn new(id: String, config: ToolMockConfig, contract: Option<Arc<dyn Tool>>) -> Self {
        Self {
            id,
            config,
            contract,
        }
    }
}

#[async_trait]
impl Tool for MockTool {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        self.contract
            .as_ref()
            .map_or(self.id.as_str(), |tool| tool.name())
    }

    fn description(&self) -> &str {
        self.contract
            .as_ref()
            .map_or("Evaluation mock tool", |tool| tool.description())
    }

    fn input_schema(&self) -> Value {
        self.contract.as_ref().map_or_else(
            || serde_json::json!({"type": "object"}),
            |tool| tool.input_schema(),
        )
    }

    fn safety_metadata(&self) -> ai_agents_core::ToolSafetyMetadata {
        self.contract.as_ref().map_or_else(
            ai_agents_core::ToolSafetyMetadata::conservative_unknown,
            |tool| tool.safety_metadata(),
        )
    }

    fn classify_call(&self, args: &Value) -> ai_agents_core::ToolCallClassification {
        self.contract.as_ref().map_or_else(
            || ai_agents_core::ToolCallClassification::from_metadata(&self.safety_metadata()),
            |tool| tool.classify_call(args),
        )
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        self.contract
            .as_ref()
            .map_or_else(ToolPolicyBindings::default, |tool| tool.policy_bindings())
    }

    async fn execute(
        &self,
        _args: Value,
        _ctx: ai_agents_core::ToolExecutionContext,
    ) -> ToolResult {
        let output = if self.config.output.is_string() {
            self.config.output.as_str().unwrap_or_default().to_string()
        } else {
            serde_json::to_string(&self.config.output).unwrap_or_else(|_| "null".to_string())
        };
        ToolResult {
            success: self.config.success,
            output,
            metadata: None,
        }
    }
}

/// Tool wrapper that records calls before returning the inner result.
struct RecordingTool {
    /// Wrapped implementation or shared storage.
    inner: Arc<dyn Tool>,
}

impl RecordingTool {
    fn new(inner: Arc<dyn Tool>, _log: RecordingToolLog, _source: ToolExecutionSource) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Tool for RecordingTool {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    fn safety_metadata(&self) -> ai_agents_core::ToolSafetyMetadata {
        self.inner.safety_metadata()
    }

    fn classify_call(&self, args: &Value) -> ai_agents_core::ToolCallClassification {
        self.inner.classify_call(args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        self.inner.policy_bindings()
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        self.inner.execute(args, ctx).await
    }
}

pub fn build_llm_registry(
    spec: &AgentSpec,
    fixtures: &LlmFixtureConfig,
    base_dir: &Path,
) -> Result<(LLMRegistry, Option<Arc<dyn LLMProvider>>)> {
    let mut registry = LLMRegistry::new();
    let aliases = if spec.llms.is_empty() {
        vec![(
            "default".to_string(),
            spec.llm.as_config().cloned().unwrap_or_default(),
        )]
    } else {
        spec.llms
            .iter()
            .map(|(alias, config)| (alias.clone(), config.clone()))
            .collect()
    };

    let cassette_records = load_cassette_records(fixtures, base_dir)?;
    let mut judge_provider = None;

    for (alias, config) in aliases {
        let fixture_responses = load_fixture_responses_for_alias(fixtures, base_dir, &alias)?;
        let fixture_outcomes =
            load_fixture_outcomes_for_alias(fixtures, &alias, &fixture_responses);
        let fixture_delay_ms = fixtures.delays_by_alias.get(&alias).copied().unwrap_or(0);
        let provider = match fixtures.mode {
            LlmFixtureMode::Mock => {
                Arc::new(SequenceLLMProvider::new(fixture_outcomes, fixture_delay_ms))
                    as Arc<dyn LLMProvider>
            }
            LlmFixtureMode::Replay => Arc::new(ReplayLLMProvider::new(
                alias.clone(),
                config.model.clone(),
                cassette_records.clone(),
                fixture_responses.clone(),
                fixture_delay_ms,
            )) as Arc<dyn LLMProvider>,
            LlmFixtureMode::Real => build_real_provider(&config)?,
            LlmFixtureMode::Record => {
                let inner = build_real_provider(&config)?;
                let path = fixtures
                    .cassette
                    .as_ref()
                    .map(|p| resolve_path(base_dir, p))
                    .unwrap_or_else(|| base_dir.join("llm_cassette.jsonl"));
                Arc::new(RecordingLLMProvider::new(
                    inner,
                    alias.clone(),
                    config.model.clone(),
                    path,
                )) as Arc<dyn LLMProvider>
            }
        };
        if judge_provider.is_none() {
            judge_provider = Some(provider.clone());
        }
        registry.register(alias, provider);
    }

    let default_alias = spec.llm.get_default_alias();
    registry.set_default(default_alias);
    if let Some(router) = spec.llm.get_router_alias() {
        registry.set_router(router);
    }

    Ok((registry, judge_provider))
}

fn build_real_provider(
    config: &ai_agents_runtime::spec::LLMConfig,
) -> Result<Arc<dyn LLMProvider>> {
    use std::str::FromStr;
    let provider_type = ProviderType::from_str(&config.provider)
        .map_err(|error| EvalError::Config(error.to_string()))?;
    let core_config = ai_agents_core::LLMConfig {
        temperature: Some(config.temperature),
        max_tokens: Some(config.max_tokens),
        top_p: config.top_p,
        top_k: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop_sequences: None,
        timeout_seconds: config.timeout_seconds,
        reasoning: config.reasoning,
        reasoning_effort: config.reasoning_effort.clone(),
        reasoning_budget_tokens: config.reasoning_budget_tokens,
        extra: config.extra.clone(),
    };
    let base_url = config.base_url.clone().or_else(|| {
        config
            .extra
            .get("base_url")
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let api_key = config
        .api_key_env
        .as_ref()
        .and_then(|env| std::env::var(env).ok());
    let mut provider = UnifiedLLMProvider::from_spec_config(
        provider_type,
        &config.model,
        api_key,
        base_url,
        core_config,
    )
    .map_err(|error| EvalError::Runtime(error.to_string()))?;
    if let Some(value) = config.function_calling {
        provider = provider.with_feature_override(LLMFeature::FunctionCalling, value);
    }
    if let Some(value) = config.vision {
        provider = provider.with_feature_override(LLMFeature::Vision, value);
    }
    if let Some(value) = config.json_mode {
        provider = provider.with_feature_override(LLMFeature::JsonMode, value);
    }
    Ok(Arc::new(provider))
}

fn load_fixture_responses_for_alias(
    config: &LlmFixtureConfig,
    base_dir: &Path,
    alias: &str,
) -> Result<Vec<LLMResponse>> {
    let configured = config
        .responses_by_alias
        .get(alias)
        .unwrap_or(&config.responses);
    let mut responses = Vec::new();
    for content in configured {
        responses.push(LLMResponse::new(content.clone(), FinishReason::Stop));
    }
    for record in load_cassette_records(config, base_dir)? {
        if record.alias == alias {
            responses.push(record.response);
        }
    }
    if responses.is_empty() {
        responses.push(LLMResponse::new("Mock response", FinishReason::Stop));
    }
    Ok(responses)
}

fn load_fixture_outcomes_for_alias(
    config: &LlmFixtureConfig,
    alias: &str,
    responses: &[LLMResponse],
) -> Vec<SequenceOutcome> {
    if let Some(outcomes) = config.outcomes_by_alias.get(alias)
        && !outcomes.is_empty()
    {
        return outcomes
            .iter()
            .map(|outcome| match outcome {
                LlmFixtureOutcome::Response { content } => {
                    SequenceOutcome::Response(LLMResponse::new(content.clone(), FinishReason::Stop))
                }
                LlmFixtureOutcome::Error { message, status } => SequenceOutcome::Error {
                    message: message.clone(),
                    status: *status,
                },
            })
            .collect();
    }
    if let Some(message) = config.errors_by_alias.get(alias) {
        return vec![SequenceOutcome::Error {
            message: message.clone(),
            status: None,
        }];
    }
    responses
        .iter()
        .cloned()
        .map(SequenceOutcome::Response)
        .collect()
}

fn load_cassette_records(
    config: &LlmFixtureConfig,
    base_dir: &Path,
) -> Result<Vec<CassetteRecord>> {
    let mut records = Vec::new();
    if let Some(path) = &config.cassette {
        let resolved = resolve_path(base_dir, path);
        if resolved.exists() {
            let content = std::fs::read_to_string(&resolved)?;
            for line in content.lines().filter(|line| !line.trim().is_empty()) {
                records.push(serde_json::from_str(line)?);
            }
        }
    }
    Ok(records)
}

/// One recorded LLM response used by replay and record modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CassetteRecord {
    /// LLM alias used for this provider or cassette record.
    alias: String,
    /// Model or relationship model name.
    model: String,
    /// Stable hash of messages and request config.
    request_hash: String,
    /// Hash format version.
    #[serde(default)]
    request_hash_version: Option<String>,
    /// Assistant response text or redacted output value.
    response: LLMResponse,
}

#[derive(Clone)]
enum SequenceOutcome {
    Response(LLMResponse),
    Error {
        message: String,
        status: Option<u16>,
    },
}

/// Deterministic LLM provider returning fixture outcomes in order.
struct SequenceLLMProvider {
    /// Ordered response and error outcomes used by mock mode.
    outcomes: Arc<Vec<SequenceOutcome>>,
    /// Zero-based turn index within the scenario.
    index: Mutex<usize>,
    /// Fixed delay before returning a response.
    delay_ms: u64,
}

/// LLM provider replaying cassette records by request hash.
struct ReplayLLMProvider {
    /// LLM alias used for this provider or cassette record.
    alias: String,
    /// Model or relationship model name.
    model: String,
    /// Cassette records available for hash matching.
    records: Arc<Vec<CassetteRecord>>,
    /// Ordered text responses used by mock mode and fallback replay.
    responses: SequenceLLMProvider,
    /// Fixed delay before returning a replayed response.
    delay_ms: u64,
}

impl SequenceLLMProvider {
    fn new(outcomes: Vec<SequenceOutcome>, delay_ms: u64) -> Self {
        Self {
            outcomes: Arc::new(outcomes),
            index: Mutex::new(0),
            delay_ms,
        }
    }

    async fn wait_if_configured(&self) {
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
    }

    fn next_outcome(&self) -> std::result::Result<LLMResponse, LLMError> {
        let mut index = self.index.lock();
        let outcome = self.outcomes.get(*index).or_else(|| self.outcomes.last());
        if *index + 1 < self.outcomes.len() {
            *index += 1;
        }
        match outcome {
            Some(SequenceOutcome::Response(response)) => Ok(response.clone()),
            Some(SequenceOutcome::Error { message, status }) => Err(LLMError::API {
                message: message.clone(),
                status: *status,
            }),
            None => Ok(LLMResponse::new("Mock response", FinishReason::Stop)),
        }
    }
}

impl ReplayLLMProvider {
    fn new(
        alias: String,
        model: String,
        records: Vec<CassetteRecord>,
        fallback: Vec<LLMResponse>,
        delay_ms: u64,
    ) -> Self {
        Self {
            alias,
            model,
            records: Arc::new(records),
            responses: SequenceLLMProvider::new(
                fallback
                    .into_iter()
                    .map(SequenceOutcome::Response)
                    .collect(),
                delay_ms,
            ),
            delay_ms,
        }
    }

    async fn wait_if_configured(&self) {
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
    }

    fn response_for(&self, messages: &[ChatMessage], config: Option<&LLMConfig>) -> LLMResponse {
        let request_hash = hash_request(messages, config);
        if let Some(record) = self.records.iter().find(|record| {
            record.alias == self.alias
                && record.request_hash == request_hash
                && (record.model == self.model || record.model.is_empty())
        }) {
            return record.response.clone();
        }
        self.responses
            .next_outcome()
            .unwrap_or_else(|_| LLMResponse::new("Mock response", FinishReason::Stop))
    }
}

#[async_trait]
impl LLMProvider for ReplayLLMProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.wait_if_configured().await;
        Ok(self.response_for(messages, config))
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        self.wait_if_configured().await;
        let response = self.response_for(messages, config);
        Ok(Box::new(futures::stream::iter(chunks_from_response(
            response,
        ))))
    }

    fn provider_name(&self) -> &str {
        "eval-replay"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::Streaming | LLMFeature::SystemMessages)
    }
}

#[async_trait]
impl LLMProvider for SequenceLLMProvider {
    async fn complete(
        &self,
        _messages: &[ChatMessage],
        _config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.wait_if_configured().await;
        self.next_outcome()
    }

    async fn complete_stream(
        &self,
        _messages: &[ChatMessage],
        _config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        self.wait_if_configured().await;
        let response = self.next_outcome()?;
        Ok(Box::new(futures::stream::iter(chunks_from_response(
            response,
        ))))
    }

    fn provider_name(&self) -> &str {
        "eval-sequence"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::Streaming | LLMFeature::SystemMessages)
    }
}

fn chunks_from_response(response: LLMResponse) -> Vec<std::result::Result<LLMChunk, LLMError>> {
    let deltas = split_stream_content(&response.content);
    if deltas.is_empty() {
        return vec![Ok(LLMChunk::final_chunk(
            "",
            response.finish_reason,
            response.usage,
        ))];
    }
    let last_index = deltas.len() - 1;
    deltas
        .into_iter()
        .enumerate()
        .map(|(index, delta)| {
            if index == last_index {
                Ok(LLMChunk::final_chunk(
                    delta,
                    response.finish_reason.clone(),
                    response.usage.clone(),
                ))
            } else {
                Ok(LLMChunk::new(delta, false))
            }
        })
        .collect()
}

fn split_stream_content(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.len() > 1 {
        return words
            .into_iter()
            .enumerate()
            .map(|(index, word)| {
                if index == 0 {
                    word.to_string()
                } else {
                    format!(" {}", word)
                }
            })
            .collect();
    }
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= 1 {
        return vec![content.to_string()];
    }
    let chunk_size = 4.min(chars.len().div_ceil(2));
    chars
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect()
}

/// LLM provider wrapper appending responses to a cassette file.
struct RecordingLLMProvider {
    /// Wrapped implementation or shared storage.
    inner: Arc<dyn LLMProvider>,
    /// LLM alias used for this provider or cassette record.
    alias: String,
    /// Model or relationship model name.
    model: String,
    /// Path used for file lookup, HTTP routing, or dot-path checks.
    path: PathBuf,
}

impl RecordingLLMProvider {
    fn new(inner: Arc<dyn LLMProvider>, alias: String, model: String, path: PathBuf) -> Self {
        Self {
            inner,
            alias,
            model,
            path,
        }
    }
}

#[async_trait]
impl LLMProvider for RecordingLLMProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        let response = self.inner.complete(messages, config).await?;
        let record = CassetteRecord {
            alias: self.alias.clone(),
            model: self.model.clone(),
            request_hash: hash_request(messages, config),
            request_hash_version: Some("sha256-v1".to_string()),
            response: response.clone(),
        };
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(
                file,
                "{}",
                serde_json::to_string(&record).unwrap_or_default()
            );
        }
        Ok(response)
    }

    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn Stream<Item = std::result::Result<LLMChunk, LLMError>> + Unpin + Send>,
        LLMError,
    > {
        self.inner.complete_stream(messages, config).await
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        self.inner.supports(feature)
    }
}

fn hash_request(messages: &[ChatMessage], config: Option<&LLMConfig>) -> String {
    let canonical_messages: Vec<Value> = messages
        .iter()
        .map(|message| {
            json!({
                "role": format!("{:?}", message.role),
                "content": message.content,
                "name": message.name,
            })
        })
        .collect();
    let canonical = json!({
        "version": "sha256-v1",
        "messages": canonical_messages,
        "config": config,
    });
    let encoded = serde_json::to_vec(&canonical).unwrap_or_default();
    let digest = Sha256::digest(encoded);
    format!("sha256-v1:{:x}", digest)
}

fn default_status() -> u16 {
    200
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_tool_sources_preserve_existing_labels_and_distinguish_plan() {
        use ai_agents_core::ToolCallSource;

        let mappings = [
            (ToolCallSource::Model, ToolExecutionSource::Llm),
            (
                ToolCallSource::Skill {
                    skill_id: "skill".to_string(),
                    step_index: 0,
                },
                ToolExecutionSource::Skill,
            ),
            (
                ToolCallSource::Plan { step_index: 0 },
                ToolExecutionSource::Plan,
            ),
            (
                ToolCallSource::StateAction {
                    state: Some("ready".to_string()),
                    action_index: 0,
                },
                ToolExecutionSource::StateAction,
            ),
            (
                ToolCallSource::Orchestration,
                ToolExecutionSource::Orchestration,
            ),
            (ToolCallSource::Spawner, ToolExecutionSource::Spawner),
            (
                ToolCallSource::Fallback {
                    original_tool: "missing".to_string(),
                },
                ToolExecutionSource::Llm,
            ),
            (ToolCallSource::Task, ToolExecutionSource::Llm),
            (ToolCallSource::Manual, ToolExecutionSource::Llm),
            (ToolCallSource::EvalFixture, ToolExecutionSource::Mock),
        ];

        for (source, expected) in mappings {
            assert_eq!(ToolExecutionSource::from(&source), expected);
        }
        assert_eq!(
            serde_json::to_value(ToolExecutionSource::Plan).unwrap(),
            json!("plan")
        );
    }

    #[test]
    fn llm_fixture_parses_tagged_outcomes_by_alias() {
        let config: FixturesConfig = serde_yaml::from_str(
            r#"
llm:
  mode: mock
  outcomes_by_alias:
    worker:
      - type: error
        message: transient overload
      - type: response
        content: recovered
"#,
        )
        .unwrap();

        assert_eq!(
            config.llm.outcomes_by_alias["worker"],
            vec![
                LlmFixtureOutcome::Error {
                    message: "transient overload".to_string(),
                    status: None,
                },
                LlmFixtureOutcome::Response {
                    content: "recovered".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn sequence_provider_mixes_outcomes_with_independent_alias_cursors() {
        let config: LlmFixtureConfig = serde_yaml::from_str(
            r#"
mode: mock
outcomes_by_alias:
  alpha:
    - type: error
      message: retry alpha
    - type: response
      content: alpha recovered
  beta:
    - type: response
      content: beta first
    - type: response
      content: beta second
"#,
        )
        .unwrap();
        let fallback = vec![LLMResponse::new("fallback", FinishReason::Stop)];
        let alpha = SequenceLLMProvider::new(
            load_fixture_outcomes_for_alias(&config, "alpha", &fallback),
            0,
        );
        let beta = SequenceLLMProvider::new(
            load_fixture_outcomes_for_alias(&config, "beta", &fallback),
            0,
        );

        assert!(matches!(
            alpha.complete(&[], None).await,
            Err(LLMError::API { message, status: None }) if message == "retry alpha"
        ));
        assert_eq!(
            beta.complete(&[], None).await.unwrap().content,
            "beta first"
        );
        assert_eq!(
            alpha.complete(&[], None).await.unwrap().content,
            "alpha recovered"
        );
        assert_eq!(
            beta.complete(&[], None).await.unwrap().content,
            "beta second"
        );
        assert_eq!(
            alpha.complete(&[], None).await.unwrap().content,
            "alpha recovered"
        );
        assert_eq!(
            beta.complete(&[], None).await.unwrap().content,
            "beta second"
        );
    }

    #[tokio::test]
    async fn sequence_provider_stream_errors_advance_to_recovery() {
        let provider = SequenceLLMProvider::new(
            vec![
                SequenceOutcome::Error {
                    message: "stream retry".to_string(),
                    status: None,
                },
                SequenceOutcome::Response(LLMResponse::new("stream recovered", FinishReason::Stop)),
            ],
            0,
        );

        assert!(matches!(
            provider.complete_stream(&[], None).await,
            Err(LLMError::API { message, status: None }) if message == "stream retry"
        ));
        assert_eq!(
            provider.complete(&[], None).await.unwrap().content,
            "stream recovered"
        );
    }

    #[tokio::test]
    async fn sequence_provider_preserves_legacy_responses_and_errors() {
        let config = LlmFixtureConfig {
            mode: LlmFixtureMode::Mock,
            responses: vec!["global".to_string()],
            responses_by_alias: HashMap::from([(
                "worker".to_string(),
                vec!["first".to_string(), "second".to_string()],
            )]),
            errors_by_alias: HashMap::from([(
                "failing".to_string(),
                "permanent failure".to_string(),
            )]),
            ..Default::default()
        };
        let worker_responses =
            load_fixture_responses_for_alias(&config, Path::new("."), "worker").unwrap();
        let failing_responses =
            load_fixture_responses_for_alias(&config, Path::new("."), "failing").unwrap();
        let worker = SequenceLLMProvider::new(
            load_fixture_outcomes_for_alias(&config, "worker", &worker_responses),
            0,
        );
        let failing = SequenceLLMProvider::new(
            load_fixture_outcomes_for_alias(&config, "failing", &failing_responses),
            0,
        );

        assert_eq!(worker.complete(&[], None).await.unwrap().content, "first");
        assert_eq!(worker.complete(&[], None).await.unwrap().content, "second");
        assert_eq!(worker.complete(&[], None).await.unwrap().content, "second");
        for _ in 0..2 {
            assert!(matches!(
                failing.complete(&[], None).await,
                Err(LLMError::API { message, status: None }) if message == "permanent failure"
            ));
        }
    }

    #[test]
    fn approval_fixture_parses_all_triggers_and_outcomes() {
        let config: FixturesConfig = serde_yaml::from_str(
            r#"
approvals:
  preferred_language: en
  supported_languages: [en, fr]
  rules:
    - trigger: { type: tool, name: transfer, args: { amount: 10 } }
      occurrence: 2
      outcome: approve
    - trigger: { type: condition, name: high_value, matched: "amount > 100" }
      outcome: reject
      reason: too expensive
    - trigger: { type: state_transition, from: review, to: complete }
      outcome: modify
      changes: { amount: 5 }
    - trigger: { type: disambiguation_escalation, reason: unclear intent }
      outcome: timeout
  default:
    outcome: unavailable
"#,
        )
        .unwrap();
        let approvals = config.approvals.unwrap();

        assert_eq!(approvals.rules.len(), 4);
        assert_eq!(approvals.rules[0].occurrence.unwrap().get(), 2);
        assert_eq!(approvals.preferred_language.as_deref(), Some("en"));
        assert_eq!(
            approvals.supported_languages,
            Some(vec!["en".to_string(), "fr".to_string()])
        );
        assert_eq!(approvals.default, ApprovalFixtureOutcome::Unavailable);
    }

    #[test]
    fn approval_fixture_rejects_zero_occurrence() {
        let error = serde_yaml::from_str::<ApprovalFixtureConfig>(
            r#"
rules:
  - trigger: { type: tool, name: transfer }
    occurrence: 0
    outcome: approve
"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("nonzero"));
    }

    #[tokio::test]
    async fn approval_handler_matches_order_occurrence_and_default() {
        let config = ApprovalFixtureConfig {
            rules: vec![
                ApprovalFixtureRule {
                    trigger: ApprovalFixtureTrigger::Tool {
                        name: "transfer".to_string(),
                        args: Some(json!({"amount": 10})),
                    },
                    occurrence: NonZeroUsize::new(2),
                    outcome: ApprovalFixtureOutcome::Modify {
                        changes: HashMap::from([("amount".to_string(), json!(5))]),
                    },
                },
                ApprovalFixtureRule {
                    trigger: ApprovalFixtureTrigger::Tool {
                        name: "transfer".to_string(),
                        args: None,
                    },
                    occurrence: None,
                    outcome: ApprovalFixtureOutcome::Reject {
                        reason: Some("fallback rule".to_string()),
                    },
                },
            ],
            default: ApprovalFixtureOutcome::Timeout,
            preferred_language: Some("ja".to_string()),
            supported_languages: Some(vec!["ja".to_string(), "en".to_string()]),
        };
        let handler = FixtureApprovalHandler::new(config);
        let request = || {
            ApprovalRequest::new(
                ApprovalTrigger::tool("transfer", json!({"amount": 10})),
                "Approve transfer?",
            )
        };

        assert!(matches!(
            handler.request_approval(request()).await,
            ApprovalResult::Rejected { reason: Some(reason) } if reason == "fallback rule"
        ));
        assert!(matches!(
            handler.request_approval(request()).await,
            ApprovalResult::Modified { changes } if changes.get("amount") == Some(&json!(5))
        ));
        assert!(matches!(
            handler
                .request_approval(ApprovalRequest::new(
                    ApprovalTrigger::tool("email", json!({})),
                    "Approve email?",
                ))
                .await,
            ApprovalResult::Timeout
        ));
        assert_eq!(handler.preferred_language(), Some("ja".to_string()));
        assert_eq!(
            handler.supported_languages(),
            Some(vec!["ja".to_string(), "en".to_string()])
        );
    }

    #[tokio::test]
    async fn approval_handler_matches_condition_state_and_escalation() {
        let config: ApprovalFixtureConfig = serde_yaml::from_str(
            r#"
rules:
  - trigger: { type: condition, name: high_value, matched: threshold }
    outcome: approve
  - trigger: { type: state_transition, from: review, to: complete }
    outcome: reject
  - trigger: { type: disambiguation_escalation, reason: unclear }
    outcome: timeout
default:
  outcome: unavailable
"#,
        )
        .unwrap();
        let handler = FixtureApprovalHandler::new(config);

        let condition = handler
            .request_approval(ApprovalRequest::new(
                ApprovalTrigger::condition("high_value", "threshold"),
                "condition",
            ))
            .await;
        let state = handler
            .request_approval(ApprovalRequest::new(
                ApprovalTrigger::state(Some("review".to_string()), "complete"),
                "state",
            ))
            .await;
        let escalation = handler
            .request_approval(ApprovalRequest::new(
                ApprovalTrigger::condition("disambiguation_escalation", "unclear"),
                "escalation",
            ))
            .await;
        let unavailable = handler
            .request_approval(ApprovalRequest::new(
                ApprovalTrigger::condition("disambiguation_escalation", "other"),
                "other escalation",
            ))
            .await;

        assert!(matches!(condition, ApprovalResult::Approved));
        assert!(matches!(state, ApprovalResult::Rejected { reason: None }));
        assert!(matches!(escalation, ApprovalResult::Timeout));
        assert!(matches!(
            unavailable,
            ApprovalResult::Rejected { reason: Some(reason) }
                if reason == "Approval fixture unavailable"
        ));
    }

    #[tokio::test]
    async fn approval_handler_instances_have_fresh_concurrent_state() {
        let config: ApprovalFixtureConfig = serde_yaml::from_str(
            r#"
rules:
  - trigger: { type: tool, name: transfer }
    occurrence: 2
    outcome: approve
default:
  outcome: reject
"#,
        )
        .unwrap();
        let first = build_approval_handler(&config);
        let second = build_approval_handler(&config);
        let request =
            || ApprovalRequest::new(ApprovalTrigger::tool("transfer", json!({})), "transfer");

        let (first_result, second_result) = tokio::join!(
            first.request_approval(request()),
            first.request_approval(request())
        );
        let approved =
            usize::from(first_result.is_approved()) + usize::from(second_result.is_approved());
        assert_eq!(approved, 1);
        assert!(second.request_approval(request()).await.is_rejected());
        assert!(second.request_approval(request()).await.is_approved());
    }

    #[test]
    fn workspace_policy_is_optional_and_deserializes_narrow_tool_lists() {
        let absent: FixturesConfig = serde_yaml::from_str("llm: { mode: mock }").unwrap();
        assert_eq!(absent.workspace_policy, None);

        let configured: FixturesConfig = serde_yaml::from_str(
            r#"
workspace_policy:
  read_tools: [file_read]
  write_tools: [file_write]
"#,
        )
        .unwrap();
        assert_eq!(
            configured.workspace_policy,
            Some(WorkspacePolicyFixtureConfig {
                read_tools: vec!["file_read".to_string()],
                write_tools: vec!["file_write".to_string()],
            })
        );
    }

    #[test]
    fn attempt_context_uses_opaque_absolute_unique_workspaces() {
        let scenario_id = "customer-refund-scenario";
        let first = AttemptFixtureContext::create(None).unwrap();
        let second = AttemptFixtureContext::create(None).unwrap();

        assert!(first.workspace.is_absolute());
        assert!(second.workspace.is_absolute());
        assert_ne!(first.workspace, second.workspace);
        assert!(!first.workspace.display().to_string().contains(scenario_id));
        let suffix = first
            .workspace
            .file_name()
            .unwrap()
            .to_string_lossy()
            .trim_start_matches("ai_agents_eval_")
            .to_string();
        assert_eq!(uuid::Uuid::parse_str(&suffix).unwrap().to_string(), suffix);
        assert_eq!(
            first.runtime_context()["eval"]["workspace"],
            json!(first.workspace.display().to_string())
        );
    }

    #[test]
    fn attempt_context_interpolates_only_exact_tokens_json_safely() {
        let context = AttemptFixtureContext {
            isolation_id: "attempt-one".to_string(),
            workspace: PathBuf::from(r"C:\eval\attempt"),
            mock_server_base_url: Some("http://127.0.0.1:43123".to_string()),
        };
        let config = LlmFixtureConfig {
            mode: LlmFixtureMode::Mock,
            responses: vec![
                r#"{"{{ eval.workspace }}":{"workspace":"{{ eval.workspace }}"},"nested":{"workspace":"{{ eval.workspace }}"},"items":["{{ mock_server.base_url }}/ok",7]}"#.to_string(),
                "keep {{ unrelated.template }} and {{eval.workspace}}".to_string(),
            ],
            responses_by_alias: HashMap::from([(
                "worker".to_string(),
                vec!["write to {{ eval.workspace }}".to_string()],
            )]),
            outcomes_by_alias: HashMap::from([(
                "reviewer".to_string(),
                vec![
                    LlmFixtureOutcome::Error {
                        message: "keep {{ eval.workspace }} literal".to_string(),
                        status: None,
                    },
                    LlmFixtureOutcome::Response {
                        content: "review {{ eval.workspace }}".to_string(),
                    },
                ],
            )]),
            ..Default::default()
        };

        let rewritten = context.interpolate_llm_fixture(&config).unwrap();
        let json: Value = serde_json::from_str(&rewritten.responses[0]).unwrap();
        assert_eq!(json["nested"]["workspace"], json!(r"C:\eval\attempt"));
        assert!(json.get(r"C:\eval\attempt").is_some());
        assert_eq!(json["items"][0], json!("http://127.0.0.1:43123/ok"));
        assert_eq!(
            rewritten.responses[1],
            "keep {{ unrelated.template }} and {{eval.workspace}}"
        );
        assert_eq!(
            rewritten.responses_by_alias["worker"][0],
            r"write to C:\eval\attempt"
        );
        assert_eq!(
            rewritten.outcomes_by_alias["reviewer"][0],
            LlmFixtureOutcome::Error {
                message: "keep {{ eval.workspace }} literal".to_string(),
                status: None,
            }
        );
        assert_eq!(
            rewritten.outcomes_by_alias["reviewer"][1],
            LlmFixtureOutcome::Response {
                content: r"review C:\eval\attempt".to_string(),
            }
        );
    }

    #[test]
    fn attempt_context_rejects_mock_server_token_without_server() {
        let context = AttemptFixtureContext {
            isolation_id: "attempt-one".to_string(),
            workspace: PathBuf::from("/tmp/attempt-one"),
            mock_server_base_url: None,
        };
        let config = LlmFixtureConfig {
            mode: LlmFixtureMode::Mock,
            responses: vec!["{{ mock_server.base_url }}/missing".to_string()],
            ..Default::default()
        };

        let error = context.interpolate_llm_fixture(&config).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without an enabled fixtures.mock_server")
        );
    }

    #[test]
    fn context_file_and_inline_context_merge() {
        let dir = std::env::temp_dir().join(format!(
            "ai_agents_eval_fixture_test_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let context_path = dir.join("context.json");
        std::fs::write(
            &context_path,
            r#"{"user":{"tier":"basic"},"channel":"file"}"#,
        )
        .unwrap();
        let config = FixturesConfig {
            context: Some(serde_json::json!({"channel":"inline","feature":true})),
            context_file: Some(PathBuf::from("context.json")),
            ..Default::default()
        };
        let context = resolve_fixture_context(&config, &dir).unwrap();
        assert_eq!(context.get("channel"), Some(&serde_json::json!("inline")));
        assert_eq!(context.get("feature"), Some(&serde_json::json!(true)));
        assert!(context.get("user").is_some());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn web_fetch_transport_uses_real_tool_without_sockets() {
        let fixtures: FixturesConfig = serde_yaml::from_str(
            r#"
web_fetch_transport:
  routes:
    - url: https://fixture.example/data
      status: 200
      headers:
        Content-Type: application/json
      body: { ok: true }
"#,
        )
        .unwrap();
        let registry = build_tool_registry(&fixtures, RecordingToolLog::new()).unwrap();
        let tool = registry.get("web_fetch").unwrap();
        let result = tool
            .execute(
                json!({"url":"https://fixture.example/data","cache_ttl_seconds":0}),
                ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(result.success, "{}", result.output);
        let output: Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["status"], 200);
        assert_eq!(output["content"], "{\"ok\":true}");

        let missing = tool
            .execute(
                json!({"url":"https://fixture.example/missing","cache_ttl_seconds":0}),
                ToolExecutionContext::test("web_fetch"),
            )
            .await;
        assert!(!missing.success);
        assert!(
            missing
                .output
                .contains("No fixtures.web_fetch_transport route")
        );
    }

    #[tokio::test]
    async fn web_fetch_transport_preserves_redirect_policy_checks() {
        let fixtures: FixturesConfig = serde_yaml::from_str(
            r#"
web_fetch_transport:
  routes:
    - url: https://fixture.example/start
      status: 302
      headers:
        Location: https://blocked.example/secret
    - url: https://blocked.example/secret
      status: 200
      body: should-not-be-returned
"#,
        )
        .unwrap();
        let registry = build_tool_registry(&fixtures, RecordingToolLog::new()).unwrap();
        let tool = registry.get("web_fetch").unwrap();
        let mut execution_context = ToolExecutionContext::test("web_fetch");
        execution_context.policy_snapshot = json!({"blocked_domains":["blocked.example"]});
        let result = tool
            .execute(
                json!({"url":"https://fixture.example/start","cache_ttl_seconds":0}),
                execution_context,
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("blocked by policy"));
    }

    #[test]
    fn mocked_builtin_preserves_tool_contract() {
        let fixtures = FixturesConfig {
            tools: HashMap::from([(
                "web_fetch".to_string(),
                ToolMockConfig {
                    success: true,
                    output: json!({"status": 200}),
                },
            )]),
            ..Default::default()
        };
        let registry = build_tool_registry(&fixtures, RecordingToolLog::new()).unwrap();
        let mocked = registry.get("web_fetch").unwrap();
        let builtin = ai_agents_tools::builtin::get_builtin_tool("web_fetch").unwrap();

        assert_eq!(mocked.name(), builtin.name());
        assert_eq!(mocked.input_schema(), builtin.input_schema());
        assert_eq!(
            serde_json::to_value(mocked.safety_metadata()).unwrap(),
            serde_json::to_value(builtin.safety_metadata()).unwrap()
        );
        assert_eq!(mocked.policy_bindings(), builtin.policy_bindings());
        assert!(!mocked.policy_bindings().domain_fields.is_empty());
    }

    #[test]
    fn mock_streaming_splits_response_into_multiple_chunks() {
        let response = LLMResponse::new(
            "Streaming hello from the mocked provider.".to_string(),
            FinishReason::Stop,
        );
        let chunks = chunks_from_response(response);
        assert!(chunks.len() > 1);
        let mut reconstructed = String::new();
        for (index, chunk) in chunks.into_iter().enumerate() {
            let chunk = chunk.unwrap();
            if index == 0 {
                assert!(!chunk.is_final);
            }
            reconstructed.push_str(&chunk.delta);
            if chunk.is_final {
                assert!(chunk.finish_reason.is_some());
            }
        }
        assert_eq!(reconstructed, "Streaming hello from the mocked provider.");
    }

    #[test]
    fn mock_streaming_splits_single_word_response() {
        let chunks = split_stream_content("Hello");
        assert!(chunks.len() > 1);
        assert_eq!(chunks.join(""), "Hello");
    }

    #[tokio::test]
    async fn mock_server_serves_configured_route() {
        let config = MockServerConfig {
            enabled: true,
            port: None,
            routes: vec![serde_json::json!({
                "method":"GET",
                "path":"/ok",
                "status":200,
                "body":{"ok":true}
            })],
        };
        let server = start_mock_server(Some(&config)).await.unwrap().unwrap();
        let context = server.context();
        let base_url = context
            .get("mock_server")
            .and_then(|value| value.get("base_url"))
            .and_then(Value::as_str)
            .unwrap()
            .trim_start_matches("http://")
            .to_string();
        let mut stream = tokio::net::TcpStream::connect(base_url).await.unwrap();
        stream
            .write_all(b"GET /ok HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        assert!(response.contains("200 OK"));
        assert!(response.contains("{\"ok\":true}"));
    }
}
