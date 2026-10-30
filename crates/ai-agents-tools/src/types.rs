use async_trait::async_trait;
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolProviderType {
    #[default]
    Builtin,
    Yaml,
    Process,
    Mcp,
    Wasm,
    Http,
    Custom,
}

impl ToolProviderType {
    pub fn default_trust_level(&self) -> TrustLevel {
        match self {
            ToolProviderType::Builtin => TrustLevel::Full,
            ToolProviderType::Yaml => TrustLevel::High,
            ToolProviderType::Process => TrustLevel::Medium,
            ToolProviderType::Mcp => TrustLevel::Medium,
            ToolProviderType::Custom => TrustLevel::Medium,
            ToolProviderType::Wasm => TrustLevel::Sandboxed,
            ToolProviderType::Http => TrustLevel::Low,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    Low,
    Sandboxed,
    #[default]
    Medium,
    High,
    Full,
}

impl PartialOrd for TrustLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TrustLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_u8().cmp(&other.as_u8())
    }
}

impl TrustLevel {
    fn as_u8(&self) -> u8 {
        match self {
            TrustLevel::Low => 0,
            TrustLevel::Sandboxed => 1,
            TrustLevel::Medium => 2,
            TrustLevel::High => 3,
            TrustLevel::Full => 4,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolAliases {
    #[serde(default)]
    pub names: HashMap<String, String>,
    #[serde(default)]
    pub descriptions: HashMap<String, String>,
    #[serde(default)]
    pub parameter_aliases: HashMap<String, HashMap<String, String>>,
}

impl ToolAliases {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, lang: impl Into<String>, name: impl Into<String>) -> Self {
        self.names.insert(lang.into(), name.into());
        self
    }

    pub fn with_description(mut self, lang: impl Into<String>, desc: impl Into<String>) -> Self {
        self.descriptions.insert(lang.into(), desc.into());
        self
    }

    pub fn get_name(&self, lang: &str) -> Option<&str> {
        self.names.get(lang).map(|s| s.as_str())
    }

    pub fn get_description(&self, lang: &str) -> Option<&str> {
        self.descriptions.get(lang).map(|s| s.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty() && self.descriptions.is_empty() && self.parameter_aliases.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMetadata {
    #[serde(default)]
    pub tags: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_duration_ms: Option<u64>,

    #[serde(default)]
    pub has_side_effects: bool,

    #[serde(default)]
    pub requires_network: bool,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, Value>,
}

impl ToolMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_side_effects(mut self) -> Self {
        self.has_side_effects = true;
        self
    }

    pub fn with_network(mut self) -> Self {
        self.requires_network = true;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub state_name: Option<String>,
    pub language: Option<String>,
    pub extra: HashMap<String, Value>,
}

impl ToolContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_state(mut self, state_name: impl Into<String>) -> Self {
        self.state_name = Some(state_name.into());
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

/// Version evidence captured when file contents are read.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct FileVersionEvidence {
    /// Normalized file path used as the version key.
    pub path: String,
    /// SHA-256 hash of the observed file bytes.
    pub sha256: String,
    /// File size in bytes at observation time.
    pub size_bytes: u64,
    /// Modified timestamp as milliseconds since Unix epoch when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_unix_ms: Option<u128>,
}

/// Session-local file version store used for read-before-write checks.
#[derive(Clone, Default)]
pub struct FileVersionStore {
    inner: Arc<RwLock<HashMap<String, FileVersionEvidence>>>,
}

impl FileVersionStore {
    /// Record a file version under its normalized path key.
    pub fn record(&self, evidence: FileVersionEvidence) {
        self.inner.write().insert(evidence.path.clone(), evidence);
    }

    /// Return the version evidence stored for a path.
    pub fn get(&self, path: impl AsRef<Path>) -> Option<FileVersionEvidence> {
        let key = normalize_version_path(path.as_ref());
        self.inner.read().get(&key).cloned()
    }

    /// Return true when the stored version matches the supplied evidence.
    pub fn matches(&self, evidence: &FileVersionEvidence) -> bool {
        self.inner
            .read()
            .get(&evidence.path)
            .is_some_and(|stored| stored == evidence)
    }
}

/// Build file version evidence from observed bytes and metadata.
pub fn file_version_evidence(
    path: impl AsRef<Path>,
    bytes: &[u8],
) -> std::io::Result<FileVersionEvidence> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path)?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis());
    let mut hasher = sha2::Sha256::new();
    sha2::Digest::update(&mut hasher, bytes);
    let hash = sha2::Digest::finalize(hasher);
    Ok(FileVersionEvidence {
        path: normalize_version_path(path),
        sha256: format!("{:x}", hash),
        size_bytes: metadata.len(),
        modified_unix_ms,
    })
}

fn normalize_version_path(path: &Path) -> String {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    path.components()
        .collect::<PathBuf>()
        .to_string_lossy()
        .to_string()
}

/// Shared slot used by `ask_user` to find the active host handler.
pub type QuestionHandlerSlot = Arc<RwLock<Option<Arc<dyn QuestionHandler>>>>;

/// Structured question sent from a tool call to the host UI.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct QuestionRequest {
    /// Text shown to the user.
    pub question: String,
    /// Selectable choices shown by CLI and TUI hosts.
    #[serde(default)]
    pub options: Vec<String>,
    /// Allows more than one selected option.
    #[serde(default)]
    pub multi_select: bool,
    /// Allows free text when the host supports it.
    #[serde(default = "default_true")]
    pub allow_other: bool,
    /// Fallback value used by non-interactive hosts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// Max seconds to wait before using fallback behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
}

/// Answer returned by an interactive or fallback question handler.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct QuestionResponse {
    /// Whether the request received a usable answer.
    pub answered: bool,
    /// Selected option labels, empty when free text was used.
    #[serde(default)]
    pub selected: Vec<String>,
    /// Free text answer when the host allowed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_text: Option<String>,
    /// Whether the handler timed out.
    #[serde(default)]
    pub timed_out: bool,
    /// Whether no interactive host was available.
    #[serde(default)]
    pub unavailable: bool,
}

/// Host bridge for asking preference or clarification questions.
#[async_trait]
pub trait QuestionHandler: Send + Sync {
    /// Bounds all logical writes made while serving this exact question; unsupported is not inferred from UI labels.
    fn declared_write_footprint(
        &self,
        _request: &QuestionRequest,
        _ctx: &ai_agents_core::ToolExecutionContext,
        _max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(None)
    }
    /// Ask a structured question and return a bounded answer.
    async fn ask_question(&self, request: QuestionRequest) -> QuestionResponse;
}

/// Severity assigned to a compiler, linter, or editor diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

/// Query sent to a diagnostics provider.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct DiagnosticsRequest {
    /// Optional file or directory path filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Optional severity filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<DiagnosticSeverity>,
    /// Maximum returned diagnostics.
    #[serde(default)]
    pub max_results: Option<usize>,
}

/// One diagnostic item returned by a host provider.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DiagnosticItem {
    /// File path associated with the diagnostic.
    pub path: String,
    /// One-based line number when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// One-based column number when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
    /// Severity reported by the host.
    pub severity: DiagnosticSeverity,
    /// Tool, compiler, linter, or LSP source name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Optional provider-specific error or warning code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Diagnostics returned by a host provider.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct DiagnosticsResponse {
    /// Whether a provider was available for this request.
    pub available: bool,
    /// Diagnostics matching the request.
    #[serde(default)]
    pub diagnostics: Vec<DiagnosticItem>,
    /// Optional message for unavailable or partial results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Provider hook for compiler, linter, or editor diagnostics.
#[async_trait]
pub trait DiagnosticsProvider: Send + Sync {
    /// Declares implementation-owned logical writes; diagnostic collection is not assumed side-effect-free.
    fn declared_write_footprint(
        &self,
        _request: &DiagnosticsRequest,
        _ctx: &ai_agents_core::ToolExecutionContext,
        _max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(None)
    }
    /// Return whether this provider can serve diagnostics now.
    fn is_available(&self) -> bool {
        true
    }

    /// Return diagnostics for the supplied request.
    async fn diagnostics(&self, request: DiagnosticsRequest) -> DiagnosticsResponse;
}

/// Diagnostics provider used when the host has not installed one.
#[derive(Debug, Default)]
pub struct UnavailableDiagnosticsProvider;

#[async_trait]
impl DiagnosticsProvider for UnavailableDiagnosticsProvider {
    fn declared_write_footprint(
        &self,
        _: &DiagnosticsRequest,
        _: &ai_agents_core::ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            "host.unavailable.diagnostics.v1",
        )))
    }
    fn is_available(&self) -> bool {
        false
    }

    async fn diagnostics(&self, _request: DiagnosticsRequest) -> DiagnosticsResponse {
        DiagnosticsResponse {
            available: false,
            diagnostics: Vec::new(),
            message: Some("diagnostics provider is unavailable".to_string()),
        }
    }
}

/// Static diagnostics provider used by tests and eval fixtures.
#[derive(Debug, Clone)]
pub struct StaticDiagnosticsProvider {
    diagnostics: Vec<DiagnosticItem>,
    available: bool,
}

impl StaticDiagnosticsProvider {
    /// Create a provider that returns a deterministic diagnostics list.
    pub fn new(diagnostics: Vec<DiagnosticItem>) -> Self {
        Self {
            diagnostics,
            available: true,
        }
    }

    /// Create a provider with explicit availability.
    pub fn with_availability(diagnostics: Vec<DiagnosticItem>, available: bool) -> Self {
        Self {
            diagnostics,
            available,
        }
    }
}

#[async_trait]
impl DiagnosticsProvider for StaticDiagnosticsProvider {
    fn declared_write_footprint(
        &self,
        _: &DiagnosticsRequest,
        _: &ai_agents_core::ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            "host.static.diagnostics.v1",
        )))
    }
    fn is_available(&self) -> bool {
        self.available
    }

    async fn diagnostics(&self, request: DiagnosticsRequest) -> DiagnosticsResponse {
        if !self.available {
            return DiagnosticsResponse {
                available: false,
                diagnostics: Vec::new(),
                message: Some("diagnostics provider is unavailable".to_string()),
            };
        }

        let mut diagnostics: Vec<DiagnosticItem> = self
            .diagnostics
            .iter()
            .filter(|item| {
                request
                    .path
                    .as_ref()
                    .is_none_or(|path| item.path.starts_with(path))
            })
            .filter(|item| {
                request
                    .severity
                    .as_ref()
                    .is_none_or(|severity| &item.severity == severity)
            })
            .cloned()
            .collect();

        if let Some(max_results) = request.max_results {
            diagnostics.truncate(max_results);
        }

        DiagnosticsResponse {
            available: true,
            diagnostics,
            message: None,
        }
    }
}

/// Shared slot used by `diagnostics` to find the active provider.
pub type DiagnosticsProviderSlot = Arc<RwLock<Arc<dyn DiagnosticsProvider>>>;

/// Safe-search preference for provider-neutral web search.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchSafeSearch {
    Off,
    #[default]
    Moderate,
    Strict,
}

/// Search request sent to a host web-search provider.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub struct WebSearchRequest {
    /// Search query sent to the provider.
    pub query: String,
    /// Maximum result count requested by the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
    /// Optional domain filters requested by the model or eval fixture.
    #[serde(default)]
    pub include_domains: Vec<String>,
    /// Optional language hint such as en or ja.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional region hint such as US or JP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Optional provider safe-search preference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safe_search: Option<WebSearchSafeSearch>,
}

/// One normalized search result returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq)]
pub struct WebSearchResultItem {
    /// Result title shown to the user.
    pub title: String,
    /// Canonical result URL when the provider supplies one.
    pub url: String,
    /// Short provider snippet or summary.
    #[serde(default)]
    pub snippet: String,
    /// Optional provider or source label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Optional publication timestamp or date string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
}

/// Normalized response returned by a web-search provider.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct WebSearchResponse {
    /// Whether a provider was available for this request.
    pub available: bool,
    /// Provider name or fixture label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Search results after provider normalization.
    #[serde(default)]
    pub results: Vec<WebSearchResultItem>,
    /// Whether returned results were truncated by policy or provider limits.
    #[serde(default)]
    pub truncated: bool,
    /// Optional unavailable or partial-result message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Host bridge for provider-neutral public web search.
#[async_trait]
pub trait WebSearchProvider: Send + Sync {
    /// Covers host-local logical writes, not the search service's invoice or unrelated remote storage.
    fn declared_write_footprint(
        &self,
        _request: &WebSearchRequest,
        _ctx: &ai_agents_core::ToolExecutionContext,
        _max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(None)
    }
    /// Return whether this provider can serve search requests now.
    fn is_available(&self) -> bool {
        true
    }

    /// Search with a bounded provider-neutral request.
    async fn search(&self, request: WebSearchRequest) -> WebSearchResponse;
}

/// Web-search provider used when no host installed search.
#[derive(Debug, Default)]
pub struct UnavailableWebSearchProvider;

#[async_trait]
impl WebSearchProvider for UnavailableWebSearchProvider {
    fn declared_write_footprint(
        &self,
        _: &WebSearchRequest,
        _: &ai_agents_core::ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            "host.unavailable.search.v1",
        )))
    }
    fn is_available(&self) -> bool {
        false
    }

    async fn search(&self, _request: WebSearchRequest) -> WebSearchResponse {
        WebSearchResponse {
            available: false,
            results: Vec::new(),
            message: Some("web search provider is unavailable".to_string()),
            ..WebSearchResponse::default()
        }
    }
}

/// Static web-search provider used by tests and eval fixtures.
#[derive(Debug, Clone, Default)]
pub struct StaticWebSearchProvider {
    responses: HashMap<String, WebSearchResponse>,
    available: bool,
}

impl StaticWebSearchProvider {
    /// Create a deterministic provider from exact-query responses.
    pub fn new(responses: HashMap<String, WebSearchResponse>) -> Self {
        Self {
            responses,
            available: true,
        }
    }

    /// Create a deterministic provider with explicit availability.
    pub fn with_availability(
        responses: HashMap<String, WebSearchResponse>,
        available: bool,
    ) -> Self {
        Self {
            responses,
            available,
        }
    }
}

#[async_trait]
impl WebSearchProvider for StaticWebSearchProvider {
    fn declared_write_footprint(
        &self,
        _: &WebSearchRequest,
        _: &ai_agents_core::ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            "host.static.search.v1",
        )))
    }
    fn is_available(&self) -> bool {
        self.available
    }

    async fn search(&self, request: WebSearchRequest) -> WebSearchResponse {
        if !self.available {
            return WebSearchResponse {
                available: false,
                message: Some("web search provider is unavailable".to_string()),
                ..WebSearchResponse::default()
            };
        }

        let mut response = self
            .responses
            .get(&request.query)
            .cloned()
            .unwrap_or_else(|| WebSearchResponse {
                available: true,
                provider: Some("static".to_string()),
                results: Vec::new(),
                message: Some("no fixture search results matched the query".to_string()),
                ..WebSearchResponse::default()
            });
        response.available = true;
        if response.provider.is_none() {
            response.provider = Some("static".to_string());
        }
        if !request.include_domains.is_empty() {
            let before = response.results.len();
            response
                .results
                .retain(|item| result_matches_domains(&item.url, &request.include_domains));
            response.truncated |= response.results.len() != before;
        }
        if let Some(max_results) = request.max_results
            && response.results.len() > max_results
        {
            response.results.truncate(max_results);
            response.truncated = true;
        }
        response
    }
}

/// Shared slot used by `web_search` to find the active host provider.
pub type WebSearchProviderSlot = Arc<RwLock<Arc<dyn WebSearchProvider>>>;

fn result_matches_domains(url: &str, domains: &[String]) -> bool {
    let host = reqwest::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_ascii_lowercase))
        .unwrap_or_else(|| url.to_ascii_lowercase());
    domains.iter().any(|domain| {
        let domain = domain.trim().to_ascii_lowercase();
        host == domain || host.ends_with(&format!(".{}", domain))
    })
}

/// Request sent to a host command runner.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct CommandRequest {
    /// Full argv vector, including executable name.
    pub argv: Vec<String>,
    /// Working directory for process execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Environment variables explicitly supplied by policy.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Max runtime in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Max combined output characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_chars: Option<usize>,
    /// User-visible reason for the command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Result returned by a host command runner.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct CommandResponse {
    /// Whether the process completed with exit code 0.
    pub success: bool,
    /// Process exit code when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Termination reason: exited, timeout, cancelled, unavailable, or error.
    pub termination: String,
    /// Captured stdout, preserving bounded beginning and ending text when truncated.
    #[serde(default)]
    pub stdout: String,
    /// Captured stderr, preserving bounded beginning and ending text when truncated.
    #[serde(default)]
    pub stderr: String,
    /// Captured stdout followed by stderr, not their chronological interleaving.
    #[serde(default)]
    pub combined_output: String,
    /// True when output was truncated.
    #[serde(default)]
    pub truncated: bool,
    /// True when the timeout ended the command.
    #[serde(default)]
    pub timed_out: bool,
    /// Working directory used for execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Redacted argv evidence.
    #[serde(default)]
    pub argv_redacted: Vec<String>,
}

/// Host bridge for controlled command execution.
#[async_trait]
pub trait CommandRunner: Send + Sync {
    /// Trusted hosts may declare an enforced complete logical footprint for the actual policy-normalized request.
    /// Process runners remain unsupported by default; a command name or argv allowlist is not write containment.
    fn declared_write_footprint(
        &self,
        _request: &CommandRequest,
        _ctx: &ai_agents_core::ToolExecutionContext,
        _max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(None)
    }
    /// Return whether this runner can execute commands now.
    fn is_available(&self) -> bool {
        true
    }

    /// Run a command request and return bounded output evidence.
    async fn run_command(
        &self,
        request: CommandRequest,
        ctx: ai_agents_core::ToolExecutionContext,
    ) -> CommandResponse;
}

/// Command runner used when no host installed process execution.
#[derive(Debug, Default)]
pub struct UnavailableCommandRunner;

#[async_trait]
impl CommandRunner for UnavailableCommandRunner {
    fn declared_write_footprint(
        &self,
        _: &CommandRequest,
        _: &ai_agents_core::ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            "host.unavailable.command.v1",
        )))
    }
    fn is_available(&self) -> bool {
        false
    }

    async fn run_command(
        &self,
        request: CommandRequest,
        _ctx: ai_agents_core::ToolExecutionContext,
    ) -> CommandResponse {
        CommandResponse {
            success: false,
            termination: "unavailable".to_string(),
            cwd: request.cwd,
            argv_redacted: redact_argv(&request.argv),
            ..CommandResponse::default()
        }
    }
}

/// Static command runner used by tests and eval fixtures.
#[derive(Debug, Clone, Default)]
pub struct StaticCommandRunner {
    responses: HashMap<Vec<String>, CommandResponse>,
    available: bool,
}

impl StaticCommandRunner {
    /// Create a deterministic runner from exact argv responses.
    pub fn new(responses: HashMap<Vec<String>, CommandResponse>) -> Self {
        Self {
            responses,
            available: true,
        }
    }

    /// Create a deterministic runner with explicit availability.
    pub fn with_availability(
        responses: HashMap<Vec<String>, CommandResponse>,
        available: bool,
    ) -> Self {
        Self {
            responses,
            available,
        }
    }
}

#[async_trait]
impl CommandRunner for StaticCommandRunner {
    fn declared_write_footprint(
        &self,
        _: &CommandRequest,
        _: &ai_agents_core::ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            "host.static.command.v1",
        )))
    }
    fn is_available(&self) -> bool {
        self.available
    }

    async fn run_command(
        &self,
        request: CommandRequest,
        _ctx: ai_agents_core::ToolExecutionContext,
    ) -> CommandResponse {
        if !self.available {
            return CommandResponse {
                success: false,
                termination: "unavailable".to_string(),
                cwd: request.cwd,
                argv_redacted: redact_argv(&request.argv),
                ..CommandResponse::default()
            };
        }
        self.responses
            .get(&request.argv)
            .cloned()
            .unwrap_or_else(|| CommandResponse {
                success: false,
                exit_code: Some(127),
                termination: "not_found".to_string(),
                stderr: "mock command not found".to_string(),
                combined_output: "mock command not found".to_string(),
                cwd: request.cwd,
                argv_redacted: redact_argv(&request.argv),
                ..CommandResponse::default()
            })
    }
}

/// Process-backed command runner for CLI and trusted hosts.
#[derive(Debug, Clone, Default)]
pub struct ProcessCommandRunner;

#[async_trait]
impl CommandRunner for ProcessCommandRunner {
    async fn run_command(
        &self,
        request: CommandRequest,
        ctx: ai_agents_core::ToolExecutionContext,
    ) -> CommandResponse {
        if request.argv.is_empty() {
            return CommandResponse {
                success: false,
                termination: "error".to_string(),
                stderr: "argv must not be empty".to_string(),
                combined_output: "argv must not be empty".to_string(),
                cwd: request.cwd,
                argv_redacted: Vec::new(),
                ..CommandResponse::default()
            };
        }
        let mut command = tokio::process::Command::new(&request.argv[0]);
        command.args(&request.argv[1..]);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear();
        command.kill_on_drop(true);
        if let Some(cwd) = &request.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &request.env {
            command.env(key, value);
        }
        let timeout_ms = request
            .timeout_ms
            .or(ctx.limits.timeout_ms)
            .unwrap_or(30_000);
        let max_output_chars = request
            .max_output_chars
            .or(ctx.limits.max_output_chars)
            .unwrap_or(20_000);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                return CommandResponse {
                    success: false,
                    termination: "error".to_string(),
                    stderr: error.to_string(),
                    combined_output: error.to_string(),
                    cwd: request.cwd,
                    argv_redacted: redact_argv(&request.argv),
                    ..CommandResponse::default()
                };
            }
        };
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_task = tokio::spawn(read_pipe_bounded(stdout, max_output_chars));
        let stderr_task = tokio::spawn(read_pipe_bounded(stderr, max_output_chars));
        let status = tokio::time::timeout(Duration::from_millis(timeout_ms), child.wait()).await;
        match status {
            Ok(Ok(status)) => {
                let stdout = stdout_task.await.unwrap_or_else(|_| {
                    let mut captured = BoundedProcessOutput::new(max_output_chars);
                    captured.read_failed = true;
                    captured
                });
                let stderr = stderr_task.await.unwrap_or_else(|_| {
                    let mut captured = BoundedProcessOutput::new(max_output_chars);
                    captured.read_failed = true;
                    captured
                });
                command_output_response(status, stdout, stderr, request, max_output_chars)
            }
            Ok(Err(error)) => CommandResponse {
                success: false,
                termination: "error".to_string(),
                stderr: error.to_string(),
                combined_output: error.to_string(),
                cwd: request.cwd,
                argv_redacted: redact_argv(&request.argv),
                ..CommandResponse::default()
            },
            Err(_) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                stdout_task.abort();
                stderr_task.abort();
                let message = "command timed out; process cleanup requested".to_string();
                CommandResponse {
                    success: false,
                    termination: "timeout".to_string(),
                    stderr: message.clone(),
                    combined_output: message,
                    truncated: true,
                    timed_out: true,
                    cwd: request.cwd,
                    argv_redacted: redact_argv(&request.argv),
                    ..CommandResponse::default()
                }
            }
        }
    }
}

/// Shared slot used by `command` to find the active host runner.
pub type CommandRunnerSlot = Arc<RwLock<Arc<dyn CommandRunner>>>;

// Retain both ends while draining the entire pipe; the configured character bound does not grow with process output.
struct BoundedProcessOutput {
    head: String,
    tail: VecDeque<char>,
    count: usize,
    limit: usize,
    read_failed: bool,
}

impl BoundedProcessOutput {
    fn new(limit: usize) -> Self {
        Self {
            head: String::new(),
            tail: VecDeque::new(),
            count: 0,
            limit,
            read_failed: false,
        }
    }

    fn push(&mut self, character: char) {
        if self.count < self.limit {
            self.head.push(character);
        }
        if self.limit > 0 {
            if self.tail.len() == self.limit {
                self.tail.pop_front();
            }
            self.tail.push_back(character);
        }
        self.count = self.count.saturating_add(1);
    }

    fn tail_string(&self) -> String {
        self.tail.iter().collect()
    }

    fn render(&self) -> (String, bool) {
        render_head_tail(&self.head, &self.tail_string(), self.count, self.limit)
    }
}

const OUTPUT_TRUNCATION_MARKER: &str = "\n...[truncated]...\n";

// Preserve the original string when it fits and reserve space for the marker within the same character limit.
fn render_head_tail(head: &str, tail: &str, count: usize, max_chars: usize) -> (String, bool) {
    if count <= max_chars {
        return (head.to_string(), false);
    }
    let marker_chars = OUTPUT_TRUNCATION_MARKER.chars().count();
    const MIN_ERROR_TAIL_CHARS: usize = 32;
    if max_chars <= marker_chars + MIN_ERROR_TAIL_CHARS {
        let mut suffix: Vec<_> = tail.chars().rev().take(max_chars).collect();
        suffix.reverse();
        return (suffix.into_iter().collect(), true);
    }
    let available = max_chars - marker_chars;
    let tail_count = available.div_ceil(2).max(MIN_ERROR_TAIL_CHARS);
    let head_count = available - tail_count;
    let mut suffix: Vec<_> = tail.chars().rev().take(tail_count).collect();
    suffix.reverse();
    let mut output: String = head.chars().take(head_count).collect();
    output.push_str(OUTPUT_TRUNCATION_MARKER);
    output.extend(suffix);
    (output, true)
}

// Serialized envelopes carry their own truncation flag, so shortening them needs no additional marker or non-monotonic escape cost.
pub(crate) fn truncate_head_tail(value: &str, max_chars: usize) -> (String, bool) {
    if value.chars().count() <= max_chars {
        return (value.to_string(), false);
    }
    let head_count = max_chars / 2;
    let tail_count = max_chars - head_count;
    let mut suffix: Vec<_> = value.chars().rev().take(tail_count).collect();
    suffix.reverse();
    let mut output: String = value.chars().take(head_count).collect();
    output.extend(suffix);
    (output, true)
}

// Decode incrementally so a UTF-8 character split across reads is not replaced merely because it crossed a pipe boundary.
fn decode_process_output(output: &mut BoundedProcessOutput, pending: &mut Vec<u8>, finished: bool) {
    let mut consumed = 0;
    while consumed < pending.len() {
        match std::str::from_utf8(&pending[consumed..]) {
            Ok(text) => {
                for character in text.chars() {
                    output.push(character);
                }
                consumed = pending.len();
            }
            Err(error) => {
                let valid_end = consumed + error.valid_up_to();
                let valid = std::str::from_utf8(&pending[consumed..valid_end])
                    .expect("UTF-8 decoder returned an invalid prefix");
                for character in valid.chars() {
                    output.push(character);
                }
                consumed = valid_end;
                match error.error_len() {
                    Some(length) => {
                        output.push('\u{fffd}');
                        consumed += length;
                    }
                    None => break,
                }
            }
        }
    }
    pending.drain(..consumed);
    if finished {
        for character in String::from_utf8_lossy(pending).chars() {
            output.push(character);
        }
        pending.clear();
    }
}

async fn read_pipe_bounded<R>(pipe: Option<R>, max_chars: usize) -> BoundedProcessOutput
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut output = BoundedProcessOutput::new(max_chars);
    let Some(mut pipe) = pipe else {
        return output;
    };
    let mut pending = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = match pipe.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => read,
            Err(_) => {
                output.read_failed = true;
                break;
            }
        };
        pending.extend_from_slice(&buffer[..read]);
        decode_process_output(&mut output, &mut pending, false);
    }
    decode_process_output(&mut output, &mut pending, true);
    output
}

fn command_output_response(
    status: std::process::ExitStatus,
    stdout: BoundedProcessOutput,
    stderr: BoundedProcessOutput,
    request: CommandRequest,
    max_output_chars: usize,
) -> CommandResponse {
    let separator = if stdout.count > 0 && stderr.count > 0 {
        "\n"
    } else {
        ""
    };
    // Combined output preserves the existing stdout-then-stderr order, not the time order of pipe writes.
    let combined_head = format!("{}{}{}", stdout.head, separator, stderr.head);
    let combined_tail = format!(
        "{}{}{}",
        stdout.tail_string(),
        separator,
        stderr.tail_string()
    );
    let combined_count = stdout
        .count
        .saturating_add(stderr.count)
        .saturating_add(separator.len());
    let (combined_output, combined_truncated) = render_head_tail(
        &combined_head,
        &combined_tail,
        combined_count,
        max_output_chars,
    );
    let (stdout_text, stdout_truncated) = stdout.render();
    let (stderr_text, stderr_truncated) = stderr.render();
    CommandResponse {
        success: status.success(),
        exit_code: status.code(),
        termination: "exited".to_string(),
        stdout: stdout_text,
        stderr: stderr_text,
        combined_output,
        truncated: stdout.read_failed
            || stderr.read_failed
            || stdout_truncated
            || stderr_truncated
            || combined_truncated,
        timed_out: false,
        cwd: request.cwd,
        argv_redacted: redact_argv(&request.argv),
    }
}

fn redact_argv(argv: &[String]) -> Vec<String> {
    let mut redacted = Vec::with_capacity(argv.len());
    let mut redact_next = false;
    for arg in argv {
        let lower = arg.to_ascii_lowercase();
        let sensitive = lower.contains("token")
            || lower.contains("secret")
            || lower.contains("password")
            || lower.contains("apikey")
            || lower.contains("api-key");
        if redact_next || sensitive {
            redacted.push("[redacted]".to_string());
        } else {
            redacted.push(arg.clone());
        }
        redact_next = matches!(
            lower.as_str(),
            "--token" | "--secret" | "--password" | "--api-key"
        );
    }
    redacted
}

/// Status value for one session-local todo item.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

/// One structured task tracked by the session-local todo tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TodoItem {
    /// Stable task ID chosen by the model or host.
    pub id: String,
    /// Task description shown in summaries and UIs.
    pub content: String,
    /// Present-continuous phrase for active status displays.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_form: Option<String>,
    /// Current task state.
    #[serde(default)]
    pub status: TodoStatus,
}

/// Identifies the sole run owner of a canonical todo list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TodoRunBinding {
    #[serde(default)]
    pub objective_revision: u64,
    pub agent_id: String,
    pub run_id: String,
    pub token: String,
}

#[derive(Default)]
struct TodoStoreState {
    items: Vec<TodoItem>,
    binding: Option<TodoRunBinding>,
    max_open: Option<u32>,
}

/// Session-local storage shared by the tool and runtime, optionally bound to one task run.
#[derive(Clone, Default)]
pub struct TodoStore {
    inner: Arc<RwLock<TodoStoreState>>,
}

tokio::task_local! { static TASK_TODO_AUTHORITY: (TodoStore, TodoRunBinding); }

/// Installs the coordinating canonical store without replacing a child's prepared tool handle.
pub async fn scope_task_todos<F: std::future::Future>(
    store: TodoStore,
    binding: TodoRunBinding,
    future: F,
) -> F::Output {
    TASK_TODO_AUTHORITY.scope((store, binding), future).await
}

/// A stale scoped binding fails closed instead of falling back to a participant's session-local list.
pub fn current_task_todo_store() -> ai_agents_core::Result<Option<TodoStore>> {
    TASK_TODO_AUTHORITY
        .try_with(|(store, binding)| {
            if store.list_for_run(binding).is_some() {
                Ok(Some(store.clone()))
            } else {
                Err(ai_agents_core::autonomy::TaskRunStorageError::Conflict.into())
            }
        })
        .unwrap_or(Ok(None))
}

impl TodoStore {
    /// Return a snapshot of all todo items.
    pub fn list(&self) -> Vec<TodoItem> {
        self.inner.read().items.clone()
    }

    /// Replace the full todo list.
    pub fn set(&self, items: Vec<TodoItem>) {
        let _ = self.try_set(items);
    }

    /// Applies the run's open-item bound atomically without replacing the sole canonical list on rejection.
    pub fn try_set(&self, items: Vec<TodoItem>) -> bool {
        let mut state = self.inner.write();
        if state
            .max_open
            .is_some_and(|limit| open_todos(&items) > limit as usize)
        {
            return false;
        }
        state.items = items;
        true
    }

    /// Installs a bound only for the current run owner; ordinary session todo behavior is unchanged.
    pub fn set_run_open_limit(&self, binding: &TodoRunBinding, limit: Option<u32>) -> bool {
        let mut state = self.inner.write();
        if state.binding.as_ref() != Some(binding)
            || limit == Some(0)
            || limit.is_some_and(|limit| open_todos(&state.items) > limit as usize)
        {
            return false;
        }
        state.max_open = limit;
        true
    }

    /// Update one item by ID and return whether it existed.
    pub fn update(
        &self,
        id: &str,
        content: Option<String>,
        active_form: Option<String>,
        status: Option<TodoStatus>,
    ) -> bool {
        let mut state = self.inner.write();
        let Some(index) = state.items.iter().position(|item| item.id == id) else {
            return false;
        };
        let was_open = matches!(
            state.items[index].status,
            TodoStatus::Pending | TodoStatus::InProgress
        );
        let becomes_open = status
            .as_ref()
            .is_some_and(|status| matches!(status, TodoStatus::Pending | TodoStatus::InProgress));
        if !was_open
            && becomes_open
            && state
                .max_open
                .is_some_and(|limit| open_todos(&state.items) >= limit as usize)
        {
            return false;
        }
        let item = &mut state.items[index];
        if let Some(content) = content {
            item.content = content;
        }
        if let Some(active_form) = active_form {
            item.active_form = Some(active_form);
        }
        if let Some(status) = status {
            item.status = status;
        }
        true
    }

    /// Remove every todo item.
    pub fn clear(&self) {
        self.inner.write().items.clear();
    }

    /// Confirms a host alias observes the same canonical authority rather than an unrelated JSON-returning tool.
    pub fn shares_store(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Binds a new run atomically and discards only the prior unbound session list.
    pub fn bind_run(&self, binding: TodoRunBinding, items: Vec<TodoItem>) -> bool {
        let mut state = self.inner.write();
        if state.binding.is_some() {
            return false;
        }
        state.binding = Some(binding);
        state.items = items;
        true
    }

    /// Replaces one objective binding and its list under the same lock without exposing an unbound interval.
    /// The caller must retain the old list for audit and acknowledge the matching durable handoff before execution.
    pub fn rebind_run(&self, previous: &TodoRunBinding, next: TodoRunBinding) -> bool {
        let mut state = self.inner.write();
        if state.binding.as_ref() != Some(previous)
            || next.agent_id != previous.agent_id
            || next.run_id != previous.run_id
            || previous.objective_revision.checked_add(1) != Some(next.objective_revision)
            || next.token.is_empty()
            || next.token == previous.token
        {
            return false;
        }
        state.binding = Some(next);
        state.items.clear();
        true
    }

    /// Returns one authoritative list only when the run binding still matches.
    pub fn list_for_run(&self, binding: &TodoRunBinding) -> Option<Vec<TodoItem>> {
        let state = self.inner.read();
        (state.binding.as_ref() == Some(binding)).then(|| state.items.clone())
    }

    /// Releases a matching run; a stale adapter cannot detach a newer owner.
    pub fn release_run(&self, binding: &TodoRunBinding) -> bool {
        let mut state = self.inner.write();
        if state.binding.as_ref() != Some(binding) {
            return false;
        }
        state.binding = None;
        state.max_open = None;
        true
    }
}

// Completed and cancelled items are visible history, not open work.
fn open_todos(items: &[TodoItem]) -> usize {
    items
        .iter()
        .filter(|item| matches!(item.status, TodoStatus::Pending | TodoStatus::InProgress))
        .count()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "subprocess helper"]
    fn output_and_edit_regression_process_output_helper() {
        if std::env::var("AI_AGENTS_PROCESS_OUTPUT_HELPER").as_deref() != Ok("1") {
            return;
        }
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        let mut stderr = std::io::stderr().lock();
        write!(stdout, "START-OUT {} END-OUT", "가".repeat(4_000)).unwrap();
        write!(stderr, "START-ERR {} 끝-오류", "z".repeat(12_000)).unwrap();
        stdout.flush().unwrap();
        stderr.flush().unwrap();
        // Exit the helper before the test harness appends its own summary to stdout.
        let code = if std::env::var("AI_AGENTS_PROCESS_OUTPUT_FAIL").as_deref() == Ok("1") {
            1
        } else {
            0
        };
        std::process::exit(code);
    }

    #[tokio::test]
    async fn output_and_edit_regression_process_runner_keeps_head_and_error_tail() {
        let executable = std::env::current_exe().unwrap();
        let request = CommandRequest {
            argv: vec![
                executable.to_string_lossy().into_owned(),
                "types::tests::output_and_edit_regression_process_output_helper".to_string(),
                "--exact".to_string(),
                "--ignored".to_string(),
                "--nocapture".to_string(),
            ],
            env: HashMap::from([
                ("AI_AGENTS_PROCESS_OUTPUT_HELPER".into(), "1".into()),
                ("AI_AGENTS_PROCESS_OUTPUT_FAIL".into(), "1".into()),
            ]),
            timeout_ms: Some(5_000),
            max_output_chars: Some(160),
            ..CommandRequest::default()
        };
        let response = ProcessCommandRunner
            .run_command(
                request,
                ai_agents_core::ToolExecutionContext::test("command"),
            )
            .await;

        assert_eq!(response.termination, "exited");
        assert_eq!(response.exit_code, Some(1));
        assert!(!response.success);
        assert!(response.truncated);
        assert!(response.stdout.contains("START-OUT"));
        assert!(response.stdout.contains("END-OUT"));
        assert!(response.stderr.contains("START-ERR"));
        assert!(response.stderr.contains("끝-오류"));
        assert!(response.combined_output.contains("끝-오류"));
        for value in [
            &response.stdout,
            &response.stderr,
            &response.combined_output,
        ] {
            assert!(value.chars().count() <= 160);
            assert!(!value.contains('\u{fffd}'));
        }
    }

    #[test]
    fn output_and_edit_regression_small_cap_keeps_the_final_error() {
        let text = format!("{}FINAL-ERROR", "x".repeat(100));
        let (output, truncated) = render_head_tail(&text, &text, text.chars().count(), 20);
        assert!(truncated);
        assert!(output.contains("FINAL-ERROR"));
        assert!(output.chars().count() <= 20);
    }

    #[tokio::test]
    async fn output_and_edit_regression_split_utf8_and_tiny_output_limits() {
        use tokio::io::AsyncWriteExt;
        let text = "A가🦀BéZ";
        for limit in [0, 1, 5, 6, 20] {
            let (reader, mut writer) = tokio::io::duplex(1);
            let bytes = text.as_bytes().to_vec();
            let sending = tokio::spawn(async move {
                for byte in bytes {
                    writer.write_all(&[byte]).await.unwrap();
                }
            });
            let captured = read_pipe_bounded(Some(reader), limit).await;
            sending.await.unwrap();
            let (output, truncated) = captured.render();
            assert_eq!(truncated, text.chars().count() > limit);
            assert!(output.chars().count() <= limit);
            assert!(!output.contains('\u{fffd}'));
            if text.chars().count() <= limit {
                assert_eq!(output, text);
            } else if limit > 0 {
                assert!(output.ends_with('Z'));
            }
        }
    }

    #[tokio::test]
    async fn output_and_edit_regression_invalid_utf8_and_zero_limit_still_drain() {
        use tokio::io::AsyncWriteExt;
        let bytes = vec![b'a', 0xff, 0xf0, 0x9f];
        let (reader, mut writer) = tokio::io::duplex(1);
        let sent = bytes.clone();
        let sending = tokio::spawn(async move {
            for byte in sent {
                writer.write_all(&[byte]).await.unwrap();
            }
        });
        let captured = read_pipe_bounded(Some(reader), 20).await;
        sending.await.unwrap();
        assert_eq!(captured.render().0, String::from_utf8_lossy(&bytes));
        assert!(!captured.render().1);

        let (reader, mut writer) = tokio::io::duplex(1);
        let sending = tokio::spawn(async move {
            writer.write_all(&vec![b'x'; 10_000]).await.unwrap();
        });
        let captured =
            tokio::time::timeout(Duration::from_secs(5), read_pipe_bounded(Some(reader), 0))
                .await
                .expect("the complete pipe must be drained");
        sending.await.unwrap();
        let (output, truncated) = captured.render();
        assert!(output.is_empty());
        assert!(truncated);
    }

    #[test]
    fn process_command_runner_timeout_helper() {
        if std::env::var("AI_AGENTS_PROCESS_TIMEOUT_HELPER").as_deref() != Ok("1") {
            return;
        }
        let Ok(started_path) = std::env::var("AI_AGENTS_PROCESS_TIMEOUT_STARTED") else {
            return;
        };
        let Ok(completed_path) = std::env::var("AI_AGENTS_PROCESS_TIMEOUT_COMPLETED") else {
            return;
        };
        std::fs::write(started_path, b"started").unwrap();
        std::thread::sleep(Duration::from_secs(10));
        std::fs::write(completed_path, b"completed").unwrap();
    }

    #[tokio::test]
    async fn process_command_runner_timeout_kills_direct_child() {
        let directory = tempfile::tempdir().unwrap();
        let started_path = directory.path().join("started");
        let completed_path = directory.path().join("completed");
        let executable = std::env::current_exe().unwrap();
        let request = CommandRequest {
            argv: vec![
                executable.to_string_lossy().into_owned(),
                "types::tests::process_command_runner_timeout_helper".to_string(),
                "--exact".to_string(),
            ],
            env: HashMap::from([
                (
                    "AI_AGENTS_PROCESS_TIMEOUT_HELPER".to_string(),
                    "1".to_string(),
                ),
                (
                    "AI_AGENTS_PROCESS_TIMEOUT_STARTED".to_string(),
                    started_path.to_string_lossy().into_owned(),
                ),
                (
                    "AI_AGENTS_PROCESS_TIMEOUT_COMPLETED".to_string(),
                    completed_path.to_string_lossy().into_owned(),
                ),
            ]),
            timeout_ms: Some(2_000),
            ..CommandRequest::default()
        };

        let response = ProcessCommandRunner
            .run_command(
                request,
                ai_agents_core::ToolExecutionContext::test("command"),
            )
            .await;

        assert_eq!(response.termination, "timeout");
        assert!(response.timed_out);
        assert!(
            response.truncated,
            "timeout abandons any partially captured output"
        );
        assert!(started_path.exists());
        assert!(!completed_path.exists());
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(!completed_path.exists());
    }

    #[test]
    fn test_provider_type_default() {
        let pt = ToolProviderType::default();
        assert_eq!(pt, ToolProviderType::Builtin);
    }

    #[test]
    fn test_provider_type_trust_levels() {
        assert_eq!(
            ToolProviderType::Builtin.default_trust_level(),
            TrustLevel::Full
        );
        assert_eq!(
            ToolProviderType::Yaml.default_trust_level(),
            TrustLevel::High
        );
        assert_eq!(
            ToolProviderType::Wasm.default_trust_level(),
            TrustLevel::Sandboxed
        );
        assert_eq!(
            ToolProviderType::Http.default_trust_level(),
            TrustLevel::Low
        );
    }

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Full > TrustLevel::High);
        assert!(TrustLevel::High > TrustLevel::Medium);
        assert!(TrustLevel::Medium > TrustLevel::Sandboxed);
        assert!(TrustLevel::Sandboxed > TrustLevel::Low);
    }

    #[test]
    fn test_tool_aliases() {
        let aliases = ToolAliases::new()
            .with_name("ko", "웹검색")
            .with_name("ja", "ウェブ検索")
            .with_description("ko", "웹에서 정보 검색");

        assert_eq!(aliases.get_name("ko"), Some("웹검색"));
        assert_eq!(aliases.get_name("ja"), Some("ウェブ検索"));
        assert_eq!(aliases.get_name("en"), None);
        assert_eq!(aliases.get_description("ko"), Some("웹에서 정보 검색"));
        assert!(!aliases.is_empty());
    }

    #[test]
    fn test_tool_metadata() {
        let metadata = ToolMetadata::new()
            .with_tags(vec!["network".to_string(), "api".to_string()])
            .with_side_effects()
            .with_network();

        assert_eq!(metadata.tags.len(), 2);
        assert!(metadata.has_side_effects);
        assert!(metadata.requires_network);
    }

    #[test]
    fn test_tool_context() {
        let ctx = ToolContext::new()
            .with_session("session123")
            .with_user("user456")
            .with_state("greeting")
            .with_language("ko");

        assert_eq!(ctx.session_id, Some("session123".to_string()));
        assert_eq!(ctx.user_id, Some("user456".to_string()));
        assert_eq!(ctx.state_name, Some("greeting".to_string()));
        assert_eq!(ctx.language, Some("ko".to_string()));
    }

    #[test]
    fn test_provider_type_serde() {
        let json = serde_json::to_string(&ToolProviderType::Builtin).unwrap();
        assert_eq!(json, "\"builtin\"");

        let pt: ToolProviderType = serde_json::from_str("\"yaml\"").unwrap();
        assert_eq!(pt, ToolProviderType::Yaml);
    }
}
