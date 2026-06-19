use async_trait::async_trait;
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

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
