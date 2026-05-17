use std::collections::HashMap;
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use ai_agents_core::{
    ChatMessage, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse, Tool,
    ToolResult,
};
use ai_agents_llm::providers::{ProviderType, UnifiedLLMProvider};
use ai_agents_llm::{FinishReason, LLMRegistry};
use ai_agents_runtime::spec::AgentSpec;
use ai_agents_tools::{ToolRegistry, create_builtin_registry};
use async_trait::async_trait;
use futures::Stream;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::evidence::{ToolExecutionRecord, ToolExecutionSource};
use crate::{EvalError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FixturesConfig {
    #[serde(default)]
    pub context: Option<Value>,
    #[serde(default)]
    pub context_file: Option<PathBuf>,
    #[serde(default)]
    pub tools: HashMap<String, ToolMockConfig>,
    #[serde(default)]
    pub llm: LlmFixtureConfig,
    #[serde(default)]
    pub mock_server: Option<MockServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMockConfig {
    #[serde(default = "default_true")]
    pub success: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmFixtureConfig {
    #[serde(default)]
    pub mode: LlmFixtureMode,
    #[serde(default)]
    pub cassette: Option<PathBuf>,
    #[serde(default)]
    pub responses: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LlmFixtureMode {
    #[default]
    Real,
    Mock,
    Replay,
    Record,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MockServerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub routes: Vec<Value>,
}

#[derive(Clone, Default)]
pub struct RecordingToolLog {
    inner: Arc<Mutex<Vec<ToolExecutionRecord>>>,
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

pub fn build_tool_registry(
    fixtures: &FixturesConfig,
    log: RecordingToolLog,
) -> Result<ToolRegistry> {
    let builtin = create_builtin_registry();
    let mut registry = ToolRegistry::new();

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

struct MockTool {
    id: String,
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

    async fn execute(&self, _args: Value) -> ToolResult {
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

struct RecordingTool {
    inner: Arc<dyn Tool>,
    log: RecordingToolLog,
    source: ToolExecutionSource,
}

impl RecordingTool {
    fn new(inner: Arc<dyn Tool>, log: RecordingToolLog, source: ToolExecutionSource) -> Self {
        Self { inner, log, source }
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

    async fn execute(&self, args: Value) -> ToolResult {
        let started_at = chrono::Utc::now();
        let start = Instant::now();
        let result = self.inner.execute(args.clone()).await;
        let duration_ms = start.elapsed().as_millis() as u64;
        let output =
            serde_json::from_str(&result.output).unwrap_or(Value::String(result.output.clone()));
        self.log.push(ToolExecutionRecord {
            call_id: uuid::Uuid::new_v4().to_string(),
            tool_id: self.inner.id().to_string(),
            requested_name: self.inner.name().to_string(),
            source: self.source.clone(),
            state: None,
            actor_id: None,
            arguments_original: args.clone(),
            arguments_executed: args,
            success: result.success,
            output: result.success.then_some(output),
            error: (!result.success).then_some(result.output.clone()),
            metadata: result
                .metadata
                .clone()
                .map(|m| serde_json::to_value(m).unwrap_or(Value::Null)),
            started_at,
            duration_ms,
            observability_span_id: None,
        });
        result
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

    let fixture_responses = load_fixture_responses(fixtures, base_dir)?;
    let mut judge_provider = None;

    for (alias, config) in aliases {
        let provider = match fixtures.mode {
            LlmFixtureMode::Mock | LlmFixtureMode::Replay => Arc::new(SequenceLLMProvider::new(
                alias.clone(),
                fixture_responses.clone(),
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

fn load_fixture_responses(config: &LlmFixtureConfig, base_dir: &Path) -> Result<Vec<LLMResponse>> {
    let mut responses = Vec::new();
    for content in &config.responses {
        responses.push(LLMResponse::new(content.clone(), FinishReason::Stop));
    }
    if let Some(path) = &config.cassette {
        let resolved = resolve_path(base_dir, path);
        if resolved.exists() {
            let content = std::fs::read_to_string(&resolved)?;
            for line in content.lines().filter(|line| !line.trim().is_empty()) {
                let record: CassetteRecord = serde_json::from_str(line)?;
                responses.push(record.response);
            }
        }
    }
    if responses.is_empty() {
        responses.push(LLMResponse::new("Mock response", FinishReason::Stop));
    }
    Ok(responses)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CassetteRecord {
    alias: String,
    model: String,
    request_hash: String,
    response: LLMResponse,
}

struct SequenceLLMProvider {
    responses: Arc<Mutex<Vec<LLMResponse>>>,
    index: Arc<Mutex<usize>>,
}

impl SequenceLLMProvider {
    fn new(_name: String, responses: Vec<LLMResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            index: Arc::new(Mutex::new(0)),
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

#[async_trait]
impl LLMProvider for SequenceLLMProvider {
    async fn complete(
        &self,
        _messages: &[ChatMessage],
        _config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
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
        let response = self.next_response();
        let chunk = LLMChunk::final_chunk(response.content, response.finish_reason, response.usage);
        Ok(Box::new(futures::stream::iter(vec![Ok(chunk)])))
    }

    fn provider_name(&self) -> &str {
        "eval-sequence"
    }

    fn supports(&self, feature: LLMFeature) -> bool {
        matches!(feature, LLMFeature::Streaming | LLMFeature::SystemMessages)
    }
}

struct RecordingLLMProvider {
    inner: Arc<dyn LLMProvider>,
    alias: String,
    model: String,
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
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for message in messages {
        format!("{:?}", message.role).hash(&mut hasher);
        message.content.hash(&mut hasher);
    }
    if let Some(config) = config {
        serde_json::to_string(config)
            .unwrap_or_default()
            .hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn default_true() -> bool {
    true
}
