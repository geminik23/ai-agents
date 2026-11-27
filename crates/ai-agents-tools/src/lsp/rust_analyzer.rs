//! rust-analyzer's experimental readiness contract for native document diagnostics.

use super::protocol::{Session, invalid};
use super::{DocumentReadiness, LspDiagnostics, LspDiagnosticsConfig};
use async_trait::async_trait;
use reqwest::Url;
use serde_json::{Value, json};
use std::{collections::BTreeSet, io, path::Path, sync::Arc};

struct RustAnalyzer;

impl LspDiagnostics {
    /// Registers rust-analyzer's native diagnostics with project readiness and membership checks.
    pub fn rust_analyzer(
        config: LspDiagnosticsConfig,
        revision: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> io::Result<Self> {
        Self::register(config, revision, Arc::new(RustAnalyzer))
    }
}

#[async_trait]
impl DocumentReadiness for RustAnalyzer {
    fn supports(&self, provider: &Value) -> bool {
        provider["interFileDependencies"] == true
    }

    fn initialize(&self, params: &mut Value) -> io::Result<()> {
        params["capabilities"]["experimental"]["serverStatusNotification"] = json!(true);
        let options = &mut params["initializationOptions"];
        if !options.is_null() && !options.is_object() {
            return Err(invalid("invalid rust-analyzer initialization options"));
        }
        apply_native_scope(options)
    }

    fn certifies_coverage(&self) -> bool {
        true
    }

    async fn validate_snapshot(
        &self,
        config: &LspDiagnosticsConfig,
        scratch: &Path,
    ) -> io::Result<()> {
        // Project and crate settings take precedence over client initialization. Verify the
        // actual copied inputs; a missing declared override must not turn disabled diagnostics on.
        let directories: BTreeSet<_> = config
            .inputs
            .iter()
            .flat_map(|input| input.parent().into_iter().flat_map(Path::ancestors))
            .collect();
        for directory in directories {
            for name in ["rust-analyzer.toml", ".rust-analyzer.toml"] {
                let relative = directory.join(name);
                if tokio::fs::try_exists(config.root.join(&relative)).await? {
                    let text = tokio::fs::read_to_string(scratch.join(&relative))
                        .await
                        .map_err(|_| {
                            invalid("Rust project settings missing from diagnostic inputs")
                        })?;
                    let mut settings: Value = toml::from_str(&text)
                        .map_err(|_| invalid("invalid Rust project settings"))?;
                    apply_native_scope(&mut settings)?;
                }
            }
        }
        Ok(())
    }

    async fn ready(&self, session: &mut Session, uri: &Url) -> io::Result<()> {
        loop {
            let status = session.notification("experimental/serverStatus").await?;
            if status["quiescent"] == true {
                if status["health"] != "ok" {
                    return Err(invalid("rust-analyzer project setup is unhealthy"));
                }
                break;
            }
        }
        // Quiescence alone also occurs when a document has no usable project. Confirm membership
        // through the server's status request and fail closed if the format or membership is unknown.
        let response = session
            .request(
                "rust-analyzer/analyzerStatus",
                json!({
                    "textDocument":{"uri":uri.as_str()}
                }),
            )
            .await?;
        let status = response
            .as_str()
            .ok_or_else(|| invalid("invalid Rust project status"))?;
        let (workspace, analysis) = status
            .split_once("\nAnalysis:\n")
            .ok_or_else(|| invalid("unrecognized Rust project status"))?;
        let mut lines = analysis.lines();
        if !workspace.starts_with("Workspaces:\nLoaded ")
            || !lines
                .next()
                .is_some_and(|line| line.starts_with("Crates for file "))
            || !lines.next().is_some_and(|line| line.starts_with("Crate: "))
        {
            return Err(invalid("Rust document is not in a loaded crate"));
        }
        Ok(())
    }
}

// Use the same required leaves when constructing client settings and checking higher-priority project settings.
fn apply_native_scope(options: &mut Value) -> io::Result<()> {
    merge_required(
        options,
        &json!({
            "checkOnSave":false,
            "cargo":{"buildScripts":{"enable":false},"extraEnv":{"CARGO_NET_OFFLINE":"true"}},
            "procMacro":{"enable":false},
            "diagnostics":{"enable":true}
        }),
    )
}

// Insert missing contract fields without replacing host options at adjacent leaves.
fn merge_required(options: &mut Value, required: &Value) -> io::Result<()> {
    if let Some(fields) = required.as_object() {
        if options.is_null() {
            *options = json!({});
        }
        let target = options
            .as_object_mut()
            .ok_or_else(|| invalid("invalid rust-analyzer option group"))?;
        for (key, value) in fields {
            if let Some(existing) = target.get_mut(key) {
                merge_required(existing, value)?;
            } else {
                target.insert(key.clone(), value.clone());
            }
        }
    } else if options != required {
        return Err(invalid(
            "settings conflict with the native Rust diagnostic scope",
        ));
    }
    Ok(())
}
