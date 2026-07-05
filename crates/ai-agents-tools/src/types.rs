use async_trait::async_trait;
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;
use std::collections::HashMap;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    Low,
    Sandboxed,
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

impl Default for TrustLevel {
    fn default() -> Self {
        TrustLevel::Medium
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchSafeSearch {
    Off,
    Moderate,
    Strict,
}

impl Default for WebSearchSafeSearch {
    fn default() -> Self {
        Self::Moderate
    }
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
        if let Some(max_results) = request.max_results {
            if response.results.len() > max_results {
                response.results.truncate(max_results);
                response.truncated = true;
            }
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
    /// Captured stdout after truncation.
    #[serde(default)]
    pub stdout: String,
    /// Captured stderr after truncation.
    #[serde(default)]
    pub stderr: String,
    /// Captured stdout and stderr after truncation.
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
        let output_byte_cap = max_output_chars.saturating_mul(4).max(1);
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_task = tokio::spawn(read_pipe_bounded(stdout, output_byte_cap));
        let stderr_task = tokio::spawn(read_pipe_bounded(stderr, output_byte_cap));
        let status = tokio::time::timeout(Duration::from_millis(timeout_ms), child.wait()).await;
        match status {
            Ok(Ok(status)) => {
                let (stdout, stdout_truncated) = stdout_task.await.unwrap_or_default();
                let (stderr, stderr_truncated) = stderr_task.await.unwrap_or_default();
                command_output_response(
                    status,
                    stdout,
                    stderr,
                    stdout_truncated || stderr_truncated,
                    request,
                    max_output_chars,
                )
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

async fn read_pipe_bounded<R>(pipe: Option<R>, max_bytes: usize) -> (Vec<u8>, bool)
where
    R: tokio::io::AsyncRead + Unpin,
{
    let Some(mut pipe) = pipe else {
        return (Vec::new(), false);
    };
    let mut output = Vec::new();
    let mut truncated = false;
    let mut buffer = vec![0u8; 8192];
    loop {
        let read = match pipe.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => read,
            Err(_) => break,
        };
        let remaining = max_bytes.saturating_sub(output.len());
        if remaining > 0 {
            output.extend_from_slice(&buffer[..read.min(remaining)]);
        }
        if read > remaining {
            truncated = true;
        }
    }
    (output, truncated)
}

fn command_output_response(
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    pre_truncated: bool,
    request: CommandRequest,
    max_output_chars: usize,
) -> CommandResponse {
    let stdout = String::from_utf8_lossy(&stdout).to_string();
    let stderr = String::from_utf8_lossy(&stderr).to_string();
    let combined = if stderr.is_empty() {
        stdout.clone()
    } else if stdout.is_empty() {
        stderr.clone()
    } else {
        format!("{}\n{}", stdout, stderr)
    };
    let (stdout, stdout_truncated) = truncate_chars(stdout, max_output_chars);
    let (stderr, stderr_truncated) = truncate_chars(stderr, max_output_chars);
    let (combined_output, combined_truncated) = truncate_chars(combined, max_output_chars);
    CommandResponse {
        success: status.success(),
        exit_code: status.code(),
        termination: "exited".to_string(),
        stdout,
        stderr,
        combined_output,
        truncated: pre_truncated || stdout_truncated || stderr_truncated || combined_truncated,
        timed_out: false,
        cwd: request.cwd,
        argv_redacted: redact_argv(&request.argv),
    }
}

fn truncate_chars(value: String, max_chars: usize) -> (String, bool) {
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        (truncated, true)
    } else {
        (value, false)
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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

impl Default for TodoStatus {
    fn default() -> Self {
        Self::Pending
    }
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

/// Session-local todo storage shared by the registry and runtime accessor.
#[derive(Clone, Default)]
pub struct TodoStore {
    inner: Arc<RwLock<Vec<TodoItem>>>,
}

impl TodoStore {
    /// Return a snapshot of all todo items.
    pub fn list(&self) -> Vec<TodoItem> {
        self.inner.read().clone()
    }

    /// Replace the full todo list.
    pub fn set(&self, items: Vec<TodoItem>) {
        *self.inner.write() = items;
    }

    /// Update one item by ID and return whether it existed.
    pub fn update(
        &self,
        id: &str,
        content: Option<String>,
        active_form: Option<String>,
        status: Option<TodoStatus>,
    ) -> bool {
        let mut items = self.inner.write();
        let Some(item) = items.iter_mut().find(|item| item.id == id) else {
            return false;
        };
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
        self.inner.write().clear();
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

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
