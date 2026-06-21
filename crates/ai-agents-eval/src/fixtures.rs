use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
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
use ai_agents_llm::providers::{ProviderType, UnifiedLLMProvider};
use ai_agents_llm::{FinishReason, LLMRegistry};
use ai_agents_runtime::spec::AgentSpec;
use ai_agents_tools::{
    CommandResponse, DiagnosticItem, DiagnosticsProvider, StaticCommandRunner,
    StaticDiagnosticsProvider, ToolRegistry, UnavailableDiagnosticsProvider,
    create_builtin_registry,
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
    /// Per-LLM alias delay in milliseconds for deterministic branch ordering.
    #[serde(default)]
    pub delays_by_alias: HashMap<String, u64>,
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
            ai_agents_core::ToolCallSource::StateAction { .. } => Self::StateAction,
            ai_agents_core::ToolCallSource::Orchestration => Self::Orchestration,
            ai_agents_core::ToolCallSource::Spawner => Self::Spawner,
            ai_agents_core::ToolCallSource::EvalFixture => Self::Mock,
            ai_agents_core::ToolCallSource::Plan { .. }
            | ai_agents_core::ToolCallSource::Fallback { .. }
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

    for (id, mock) in &fixtures.tools {
        registry
            .register(Arc::new(RecordingTool::new(
                Arc::new(MockTool::new(id.clone(), mock.clone())),
                log.clone(),
                ToolExecutionSource::Mock,
            )))
            .map_err(|error| EvalError::Config(error.to_string()))?;
    }

    for id in builtin.list_ids() {
        if fixtures.tools.contains_key(&id) {
            continue;
        }
        if let Some(tool) = builtin.get(&id) {
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

/// Tool implementation returning a configured eval fixture result.
struct MockTool {
    /// Stable identifier for this item.
    id: String,
    /// Configuration used by this component.
    config: ToolMockConfig,
}

impl MockTool {
    fn new(id: String, config: ToolMockConfig) -> Self {
        Self { id, config }
    }
}

#[async_trait]
impl Tool for MockTool {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        "Evaluation mock tool"
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
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
        let fixture_error = fixtures.errors_by_alias.get(&alias).cloned();
        let fixture_delay_ms = fixtures.delays_by_alias.get(&alias).copied().unwrap_or(0);
        let provider = match fixtures.mode {
            LlmFixtureMode::Mock => Arc::new(SequenceLLMProvider::new(
                alias.clone(),
                fixture_responses.clone(),
                fixture_error,
                fixture_delay_ms,
            )) as Arc<dyn LLMProvider>,
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

/// Deterministic LLM provider returning fixture responses in order.
struct SequenceLLMProvider {
    /// Ordered text responses used by mock mode and fallback replay.
    responses: Arc<Mutex<Vec<LLMResponse>>>,
    /// Zero-based turn index within the scenario.
    index: Arc<Mutex<usize>>,
    /// Optional deterministic provider error.
    error: Option<String>,
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
    fn new(
        _name: String,
        responses: Vec<LLMResponse>,
        error: Option<String>,
        delay_ms: u64,
    ) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            index: Arc::new(Mutex::new(0)),
            error,
            delay_ms,
        }
    }

    async fn wait_if_configured(&self) {
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
    }

    fn next_response(&self) -> LLMResponse {
        let responses = self.responses.lock();
        let mut index = self.index.lock();
        let response = responses
            .get(*index)
            .cloned()
            .or_else(|| responses.last().cloned())
            .unwrap_or_else(|| LLMResponse::new("Mock response", FinishReason::Stop));
        if *index + 1 < responses.len() {
            *index += 1;
        }
        response
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
                "replay-fallback".to_string(),
                fallback,
                None,
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
        self.responses.next_response()
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
        if let Some(error) = &self.error {
            return Err(LLMError::API {
                message: error.clone(),
                status: None,
            });
        }
        Ok(self.next_response())
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
        if let Some(error) = &self.error {
            return Err(LLMError::API {
                message: error.clone(),
                status: None,
            });
        }
        let response = self.next_response();
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
