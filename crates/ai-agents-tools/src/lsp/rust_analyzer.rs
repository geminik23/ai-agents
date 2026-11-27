//! rust-analyzer's experimental readiness contract for native document diagnostics.

use super::protocol::{Session, invalid};
use super::{DocumentReadiness, LspDiagnostics, LspDiagnosticsConfig};
use async_trait::async_trait;
use reqwest::Url;
use serde_json::{Value, json};
use std::{io, sync::Arc};

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
        for (key, required) in [
            ("checkOnSave", json!(false)),
            (
                "cargo",
                json!({"buildScripts":{"enable":false},"offline":true}),
            ),
            ("procMacro", json!({"enable":false})),
            ("diagnostics", json!({"enable":true})),
        ] {
            if options.get(key).is_some_and(|value| *value != required) {
                return Err(invalid(
                    "settings conflict with the native Rust diagnostic scope",
                ));
            }
            options[key] = required;
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
