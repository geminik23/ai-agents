//! Small test-owned business functions served by the registry rmcp implementation.

use std::{
    fs::OpenOptions,
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt, model::*, service::RequestContext};
use serde::Deserialize;
use serde_json::json;
use tokio::io::AsyncReadExt;

#[derive(Clone)]
struct TestServer {
    state: Arc<Mutex<(i64, std::fs::File)>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Increment {
    delta: i64,
}

impl TestServer {
    // Append independent server-side evidence only when the corresponding handler actually runs.
    fn record(&self, event: serde_json::Value) -> Result<(), ErrorData> {
        let mut state = self.state.lock().unwrap();
        writeln!(state.1, "{event}").map_err(|e| ErrorData::internal_error(e.to_string(), None))
    }

    // The catalog contains a blocked function so client-side filtering has an observable negative case.
    fn tools() -> Vec<Tool> {
        let increment = json!({"type":"object","properties":{"delta":{"type":"integer"}},"required":["delta"],"additionalProperties":false});
        let empty = json!({"type":"object","properties":{},"additionalProperties":false});
        vec![
            Tool::new(
                "accumulate",
                "Add delta to the server-owned counter",
                increment.as_object().unwrap().clone(),
            ),
            Tool::new(
                "fail",
                "Return an intentional remote tool error",
                empty.as_object().unwrap().clone(),
            ),
            Tool::new(
                "blocked",
                "Must never execute through the configured framework policy",
                empty.as_object().unwrap().clone(),
            ),
        ]
    }
}

impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        self.record(json!({"kind":"discovery"}))?;
        Ok(ListToolsResult {
            tools: Self::tools(),
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        match request.name.as_ref() {
            "accumulate" => {
                let args: Increment =
                    serde_json::from_value(json!(request.arguments.unwrap_or_default()))
                        .map_err(|e| ErrorData::invalid_params(e.to_string(), None))?;
                let mut state = self.state.lock().unwrap();
                state.0 = state
                    .0
                    .checked_add(args.delta)
                    .ok_or_else(|| ErrorData::invalid_params("counter overflow", None))?;
                let total = state.0;
                writeln!(
                    state.1,
                    "{}",
                    json!({"kind":"accumulate","delta":args.delta,"total":total})
                )
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    json!({"total":total}).to_string(),
                )])
                .into())
            }
            "fail" => {
                self.record(json!({"kind":"fail"}))?;
                Ok(
                    CallToolResult::error(vec![ContentBlock::text("intentional Rust MCP error")])
                        .into(),
                )
            }
            "blocked" => {
                self.record(json!({"kind":"blocked"}))?;
                Ok(
                    CallToolResult::success(vec![ContentBlock::text("unexpected execution")])
                        .into(),
                )
            }
            _ => Err(ErrorData::invalid_params("unknown function", None)),
        }
    }
}

// Publish process identity and the actual endpoint atomically, without using protocol stdout.
fn ready(dir: &Path, url: Option<String>) -> Result<()> {
    std::fs::write(
        dir.join("ready.tmp"),
        serde_json::to_vec(&json!({"pid":std::process::id(),"url":url}))?,
    )?;
    std::fs::rename(dir.join("ready.tmp"), dir.join("ready.json"))?;
    Ok(())
}

// HTTP is owned by the test driver through a control pipe; stdio lifetime is owned by the SDK client.
pub(super) async fn serve(transport: &str, dir: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dir.join("events.jsonl"))?;
    let server = TestServer {
        state: Arc::new(Mutex::new((0, file))),
    };
    match transport {
        "stdio" => {
            ready(dir, None)?;
            super::wait_until(|| Ok(dir.join("observed").is_file())).await?;
            server
                .serve(rmcp::transport::stdio())
                .await
                .context("stdio handshake")?
                .waiting()
                .await?;
        }
        "http" => {
            use rmcp::transport::streamable_http_server::{
                StreamableHttpServerConfig, StreamableHttpService,
                session::local::LocalSessionManager,
            };
            let cancellation = tokio_util::sync::CancellationToken::new();
            let service: StreamableHttpService<TestServer, LocalSessionManager> =
                StreamableHttpService::new(
                    move || Ok(server.clone()),
                    Default::default(),
                    StreamableHttpServerConfig::default()
                        .with_cancellation_token(cancellation.clone()),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            ready(dir, Some(format!("http://{}/mcp", listener.local_addr()?)))?;
            axum::serve(listener, axum::Router::new().nest_service("/mcp", service))
                .with_graceful_shutdown(async move {
                    let _ = tokio::io::stdin().read_to_end(&mut Vec::new()).await;
                    cancellation.cancel();
                })
                .await?;
        }
        _ => anyhow::bail!("expected stdio or http"),
    }
    std::fs::write(dir.join("closed"), "acknowledged")?;
    Ok(())
}
