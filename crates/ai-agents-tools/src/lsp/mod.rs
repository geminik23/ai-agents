//! Opt-in document diagnostics from a host-approved stdio language server.

mod protocol;
mod rust_analyzer;

use crate::{
    DiagnosticItem, DiagnosticSeverity, DiagnosticsProvider, DiagnosticsRequest,
    DiagnosticsResponse,
};
use async_trait::async_trait;
use protocol::{Session, invalid};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::AsyncReadExt,
    process::Command,
    sync::{Mutex, oneshot},
};

const SNAPSHOT_LIMIT: u64 = 4 * 1024 * 1024;

/// Trusted host registration for a single language server and explicitly selected documents.
/// Project inputs must cover the selected server's analysis scope and must be controlled by the host.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LspDiagnosticsConfig {
    /// Canonical root of the original documents.
    pub root: PathBuf,
    /// Absolute executable path selected by the host.
    pub program: PathBuf,
    /// Fixed server arguments, with no shell evaluation.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment explicitly supplied by the host. The calling environment is cleared.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Documents eligible for a full document diagnostic observation, relative to the root.
    pub documents: Vec<PathBuf>,
    /// Additional root-relative inputs, such as manifests, module files and linter settings.
    #[serde(default)]
    pub inputs: Vec<PathBuf>,
    /// LSP language identifier used when opening a document.
    pub language_id: String,
    /// Server initialization settings for this registration.
    #[serde(default)]
    pub initialization_options: Value,
    /// Time allowed for snapshot, startup, collection and graceful server shutdown.
    pub timeout_ms: u64,
}

#[async_trait]
trait DocumentReadiness: Send + Sync {
    fn initialize(&self, _params: &mut Value) -> io::Result<()> {
        Ok(())
    }

    fn supports(&self, diagnostic_provider: &Value) -> bool;

    async fn ready(&self, _session: &mut Session, _uri: &Url) -> io::Result<()> {
        Ok(())
    }
}

struct IndependentDocuments;

#[async_trait]
impl DocumentReadiness for IndependentDocuments {
    fn supports(&self, provider: &Value) -> bool {
        provider["interFileDependencies"] == false
    }
}

/// Provides current document diagnostics from a private snapshot of host-selected project inputs.
///
/// The host must control all writers and return a non-reused revision for every relevant
/// change. A private copy avoids concurrent mutation of server inputs but does not sandbox
/// the executable or its subprocesses. Only register trusted binaries and settings.
#[derive(Clone)]
pub struct LspDiagnostics {
    config: Arc<LspDiagnosticsConfig>,
    revision: Arc<dyn Fn() -> Option<String> + Send + Sync>,
    readiness: Arc<dyn DocumentReadiness>,
    session: Arc<Mutex<()>>,
}

impl LspDiagnostics {
    /// Registers a server declaring independent document pull diagnostics.
    /// Project-dependent analysis needs a readiness check specific to the server.
    pub fn new(
        config: LspDiagnosticsConfig,
        revision: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> io::Result<Self> {
        Self::register(config, revision, Arc::new(IndependentDocuments))
    }

    fn register(
        mut config: LspDiagnosticsConfig,
        revision: impl Fn() -> Option<String> + Send + Sync + 'static,
        readiness: Arc<dyn DocumentReadiness>,
    ) -> io::Result<Self> {
        if !config.root.is_absolute()
            || !config.program.is_absolute()
            || config.root.canonicalize()? != config.root
            || !config.root.is_dir()
            || config.timeout_ms == 0
            || config.language_id.is_empty()
            || config.documents.is_empty()
            || config
                .documents
                .iter()
                .chain(&config.inputs)
                .any(|path| !relative_file(path))
        {
            return Err(invalid("invalid document diagnostics registration"));
        }
        config.inputs.extend(config.documents.iter().cloned());
        config.inputs.sort();
        config.inputs.dedup();
        Ok(Self {
            config: Arc::new(config),
            revision: Arc::new(revision),
            readiness,
            session: Arc::new(Mutex::new(())),
        })
    }

    fn document(&self, path: Option<&str>) -> io::Result<&Path> {
        let path = Path::new(path.ok_or_else(|| invalid("document path required"))?);
        let path = if path.is_absolute() {
            path.strip_prefix(&self.config.root)
                .map_err(|_| invalid("document outside workspace"))?
        } else {
            path
        };
        self.config
            .documents
            .iter()
            .find(|document| *document == path)
            .map(PathBuf::as_path)
            .ok_or_else(|| invalid("document is outside the registered diagnostic scope"))
    }

    async fn snapshot(&self, destination: &Path) -> io::Result<()> {
        let mut remaining = SNAPSHOT_LIMIT;
        for input in &self.config.inputs {
            let source = self.config.root.join(input);
            if tokio::fs::canonicalize(&source).await? != source
                || !tokio::fs::metadata(&source).await?.is_file()
            {
                return Err(invalid("invalid diagnostic project input"));
            }
            let mut data = Vec::new();
            tokio::fs::File::open(source)
                .await?
                .take(remaining + 1)
                .read_to_end(&mut data)
                .await?;
            remaining = remaining
                .checked_sub(data.len() as u64)
                .ok_or_else(|| invalid("diagnostic project inputs exceed the snapshot limit"))?;
            let target = destination.join(input);
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(target, data).await?;
        }
        Ok(())
    }

    async fn pull(
        &self,
        session: &mut Session,
        root: &Path,
        document: &Path,
    ) -> io::Result<Vec<DiagnosticItem>> {
        let uri = Url::from_file_path(root.join(document))
            .map_err(|_| invalid("invalid LSP document URI"))?;
        let root_uri =
            Url::from_directory_path(root).map_err(|_| invalid("invalid LSP workspace URI"))?;
        let mut params = json!({
            "processId":std::process::id(),
            "rootUri":root_uri.as_str(),
            "workspaceFolders":[{"uri":root_uri.as_str(),"name":"diagnostics"}],
            "capabilities":{"textDocument":{"diagnostic":{}}},
            "initializationOptions":self.config.initialization_options
        });
        self.readiness.initialize(&mut params)?;
        let initialized = session.request("initialize", params).await?;
        let provider = &initialized["capabilities"]["diagnosticProvider"];
        if !provider.is_object() || !self.readiness.supports(provider) {
            return Err(invalid(
                "server lacks the required document diagnostic capability",
            ));
        }
        session.notify("initialized", json!({})).await?;
        session
            .notify(
                "textDocument/didOpen",
                json!({"textDocument":{
                    "uri":uri.as_str(), "languageId":self.config.language_id, "version":1,
                    "text":tokio::fs::read_to_string(root.join(document)).await?
                }}),
            )
            .await?;
        self.readiness.ready(session, &uri).await?;
        // One snapshot, one document revision and one full response. No previous-result or partial token is sent.
        let response = session
            .request(
                "textDocument/diagnostic",
                json!({
                    "textDocument":{"uri":uri.as_str()}
                }),
            )
            .await?;
        if response["kind"] != "full" || response.get("relatedDocuments").is_some() {
            return Err(invalid(
                "incomplete or out-of-scope LSP diagnostic response",
            ));
        }
        let items = response["items"]
            .as_array()
            .ok_or_else(|| invalid("missing diagnostic items"))?
            .iter()
            .map(|item| diagnostic(document, item))
            .collect::<io::Result<Vec<_>>>()?;
        session
            .notify(
                "textDocument/didClose",
                json!({"textDocument":{"uri":uri.as_str()}}),
            )
            .await?;
        session.request("shutdown", Value::Null).await?;
        session.notify("exit", Value::Null).await?;
        if !session.child.wait().await?.success() {
            return Err(invalid("language server exited unsuccessfully"));
        }
        Ok(items)
    }

    async fn collect(
        &self,
        path: Option<&str>,
        result: &mut oneshot::Sender<io::Result<Vec<DiagnosticItem>>>,
    ) -> io::Result<Vec<DiagnosticItem>> {
        let document = self.document(path)?;
        let deadline = tokio::time::Instant::now()
            .checked_add(Duration::from_millis(self.config.timeout_ms))
            .ok_or_else(|| invalid("invalid diagnostic collection deadline"))?;
        let _exclusive = tokio::select! {
            _ = result.closed() => return Err(invalid("diagnostic request cancelled")),
            lock = tokio::time::timeout_at(deadline, self.session.lock()) => lock?,
        };
        let revision =
            (self.revision)().ok_or_else(|| invalid("workspace revision unavailable"))?;
        let scratch = tempfile::tempdir()?;
        tokio::time::timeout_at(deadline, self.snapshot(scratch.path())).await??;
        if result.is_closed() || (self.revision)().as_ref() != Some(&revision) {
            return Err(invalid("diagnostic input changed or request was cancelled"));
        }
        let mut command = Command::new(&self.config.program);
        command
            .args(&self.config.args)
            .env_clear()
            .envs(&self.config.env)
            .current_dir(scratch.path());
        let mut session = Session::start(&mut command)?;
        let items = tokio::select! {
            _ = result.closed() => Err(invalid("diagnostic request cancelled")),
            value = tokio::time::timeout_at(deadline, self.pull(&mut session, scratch.path(), document)) =>
                value.unwrap_or_else(|_| Err(invalid("language server diagnostic deadline elapsed"))),
        };
        // Dropping the caller only closes the channel. The worker still owns and reaps the child
        // before releasing the scratch directory and admitting another server for this binding.
        if items.is_err() {
            session.child.kill().await?;
        }
        if (self.revision)().as_ref() != Some(&revision) {
            return Err(invalid("diagnostic input changed during collection"));
        }
        items
    }
}

#[async_trait]
impl DiagnosticsProvider for LspDiagnostics {
    async fn diagnostics(&self, request: DiagnosticsRequest) -> DiagnosticsResponse {
        let (mut send, receive) = oneshot::channel();
        let host = self.clone();
        let path = request.path;
        tokio::spawn(async move {
            let collected = host.collect(path.as_deref(), &mut send).await;
            let _ = send.send(collected);
        });
        let mut diagnostics = match receive.await {
            Ok(Ok(items)) => items,
            Ok(Err(error)) => {
                // Transport errors can contain local paths. Report only controlled failure messages.
                let message = if error.kind() == io::ErrorKind::InvalidData {
                    error.to_string()
                } else {
                    "language server or diagnostic inputs unavailable".into()
                };
                return DiagnosticsResponse {
                    message: Some(message),
                    ..Default::default()
                };
            }
            Err(_) => {
                return DiagnosticsResponse {
                    message: Some("diagnostic worker unavailable".into()),
                    ..Default::default()
                };
            }
        };
        if let Some(severity) = request.severity {
            diagnostics.retain(|item| item.severity == severity);
        }
        let message = request
            .max_results
            .filter(|limit| diagnostics.len() > *limit)
            .map(|limit| {
                diagnostics.truncate(limit);
                "language server diagnostics exceeded the result limit".into()
            });
        DiagnosticsResponse {
            available: true,
            diagnostics,
            message,
        }
    }
}

/// Routes document diagnostics to explicitly assigned servers within a single host workspace.
pub struct LspDiagnosticsHost {
    servers: Vec<LspDiagnostics>,
}

impl LspDiagnosticsHost {
    /// Installs multiple registrations for one root. Each document must have exactly one owner.
    pub fn new(servers: Vec<LspDiagnostics>) -> io::Result<Self> {
        let first = servers
            .first()
            .ok_or_else(|| invalid("missing language server registration"))?;
        let mut documents = BTreeSet::new();
        for server in &servers {
            if server.config.root != first.config.root
                || !server
                    .config
                    .documents
                    .iter()
                    .all(|document| documents.insert(document))
            {
                return Err(invalid(
                    "language server registrations have conflicting roots or documents",
                ));
            }
        }
        Ok(Self { servers })
    }
}

#[async_trait]
impl DiagnosticsProvider for LspDiagnosticsHost {
    async fn diagnostics(&self, request: DiagnosticsRequest) -> DiagnosticsResponse {
        if let Some(server) = self
            .servers
            .iter()
            .find(|server| server.document(request.path.as_deref()).is_ok())
        {
            server.diagnostics(request).await
        } else {
            DiagnosticsResponse {
                message: Some("document is outside the registered diagnostic scope".into()),
                ..Default::default()
            }
        }
    }
}

fn relative_file(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

fn diagnostic(document: &Path, item: &Value) -> io::Result<DiagnosticItem> {
    let start = &item["range"]["start"];
    let coordinate = |name: &str| {
        start[name]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok().and_then(|n| n.checked_add(1)))
            .ok_or_else(|| invalid("invalid LSP position"))
    };
    let severity = match item["severity"].as_u64() {
        Some(1) => DiagnosticSeverity::Error,
        Some(2) => DiagnosticSeverity::Warning,
        Some(3) => DiagnosticSeverity::Info,
        Some(4) => DiagnosticSeverity::Hint,
        // LSP permits omitted severity. Treat it conservatively as an error.
        None if item.get("severity").is_none() => DiagnosticSeverity::Error,
        _ => return Err(invalid("invalid LSP diagnostic severity")),
    };
    let code = match &item["code"] {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    };
    Ok(DiagnosticItem {
        path: document.to_string_lossy().into_owned(),
        line: Some(coordinate("line")?),
        column: Some(coordinate("character")?),
        severity,
        source: item["source"].as_str().map(str::to_owned),
        message: item["message"]
            .as_str()
            .ok_or_else(|| invalid("missing LSP diagnostic message"))?
            .into(),
        code,
    })
}
