//! Exercise the real SDK transports against bounded local fixtures, not external MCP services.

use super::*;
use crate::mcp::MCPViewTool;
use std::{process::Child, sync::Arc, time::Duration};

const SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/mcp/fixtures/transport.py");
const WAIT: Duration = Duration::from_secs(10);

struct Fixture {
    dir: tempfile::TempDir,
    child: Option<Child>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
            child: None,
        }
    }

    async fn http(mode: &str) -> Self {
        let mut fixture = Self::new();
        fixture.child = Some(
            std::process::Command::new("python3")
                .args(["-u", SCRIPT])
                .env("MCP_FIXTURE_DIR", fixture.dir.path())
                .env("MCP_FIXTURE_MODE", mode)
                .spawn()
                .expect("python3 is required for local MCP transport tests"),
        );
        fixture.ready().await;
        fixture
    }

    async fn ready(&self) -> Value {
        tokio::time::timeout(WAIT, async {
            loop {
                if let Ok(bytes) = std::fs::read(self.dir.path().join("ready.json")) {
                    return serde_json::from_slice(&bytes).unwrap();
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("fixture failed to start")
    }

    fn config(&self, transport: MCPWrapperTransport) -> MCPWrapperConfig {
        MCPWrapperConfig {
            name: "fixture".into(),
            transport,
            env: HashMap::new(),
            startup_timeout_ms: 2000,
            security: MCPWrapperSecurity {
                blocked_functions: vec!["blocked".into()],
                hitl_functions: vec!["approval".into()],
            },
            description: None,
            views: HashMap::new(),
        }
    }

    async fn http_config(&self, headers: bool) -> MCPWrapperConfig {
        self.config(MCPWrapperTransport::Http {
            url: self.ready().await["url"].as_str().unwrap().into(),
            headers: if headers {
                HashMap::from([("Authorization".into(), "Bearer fixture-token".into())])
            } else {
                HashMap::new()
            },
        })
    }

    fn stdio_config(&self, mode: &str) -> MCPWrapperConfig {
        let mut config = self.config(MCPWrapperTransport::Stdio {
            command: "python3".into(),
            args: vec!["-u".into(), SCRIPT.into()],
        });
        config.env = HashMap::from([
            (
                "MCP_FIXTURE_DIR".into(),
                self.dir.path().to_str().unwrap().into(),
            ),
            ("MCP_FIXTURE_MODE".into(), mode.into()),
        ]);
        config
    }

    fn requests(&self) -> Vec<Value> {
        std::fs::read_to_string(self.dir.path().join("requests.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    async fn exited(&self) {
        let pid = self.ready().await["pid"].as_u64().unwrap().to_string();
        tokio::time::timeout(WAIT, async {
            loop {
                let status = std::process::Command::new("kill")
                    .args(["-0", &pid])
                    .stderr(std::process::Stdio::null())
                    .status()
                    .unwrap();
                if !status.success() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("SDK child was not reaped");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

// Verify discovery, text projection, errors and policy filtering through the public Tool surface.
async fn exercise(tool: MCPWrapperTool) {
    let tool = Arc::new(tool);
    assert_eq!(tool.function_names(), ["echo", "fail", "approval"]);
    let view = MCPViewTool::new("read".into(), tool.clone(), vec!["echo".into()], None).unwrap();
    let result = view
        .execute(
            json!({"function":"echo","params":{"text":"hello"}}),
            ToolExecutionContext::test("read"),
        )
        .await;
    assert!(result.success);
    assert_eq!(result.output, "prefix\nhello");
    let failed = tool.call_function("fail", json!({})).await;
    assert!(!failed.success);
    assert_eq!(failed.output, "prefix\nremote-error");
    assert!(!tool.call_function("blocked", json!({})).await.success);
    assert!(
        !view
            .execute(
                json!({"function":"fail"}),
                ToolExecutionContext::test("read")
            )
            .await
            .success
    );
    let approval = tool
        .execute(
            json!({"function":"approval"}),
            ToolExecutionContext::test("fixture"),
        )
        .await;
    assert_eq!(approval.metadata.unwrap()["_hitl_required"], true);
    tokio::time::timeout(WAIT, tool.shutdown()).await.unwrap();
    assert!(
        !tool
            .call_function("echo", json!({"text":"after shutdown"}))
            .await
            .success
    );
}

fn assert_calls(requests: &[Value]) {
    let names: Vec<_> = requests
        .iter()
        .filter(|r| r["method"] == "tools/call")
        .map(|r| r["params"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["echo", "fail"]);
}

#[tokio::test]
async fn http_and_sse_discovery_calls_headers_views_and_shutdown() {
    for (mode, headers) in [("http", false), ("http", true), ("sse", false)] {
        let fixture = Fixture::http(mode).await;
        let mut config = fixture.http_config(headers).await;
        if mode == "sse" {
            let MCPWrapperTransport::Http { url, headers } = config.transport else {
                unreachable!()
            };
            config.transport = MCPWrapperTransport::Sse { url, headers };
        }
        let tool = tokio::time::timeout(WAIT, MCPWrapperTool::new(config).initialized())
            .await
            .unwrap()
            .unwrap();
        exercise(tool).await;
        let requests = fixture.requests();
        assert_calls(&requests);
        let actual = requests
            .iter()
            .filter_map(|r| r.get("headers"))
            .collect::<Vec<_>>();
        assert!(!actual.is_empty());
        for request in actual {
            let auth = request
                .as_object()
                .unwrap()
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
                .map(|(_, v)| v);
            assert_eq!(auth, headers.then_some(&json!("Bearer fixture-token")));
        }
    }
}

#[tokio::test]
async fn http_rejects_unfinished_oversized_success_and_error_bodies() {
    for mode in ["oversize-success", "oversize-error"] {
        for headers in [false, true] {
            let fixture = Fixture::http(mode).await;
            let result = tokio::time::timeout(
                WAIT,
                MCPWrapperTool::new(fixture.http_config(headers).await).initialized(),
            )
            .await
            .expect("unbounded response body was not rejected");
            let error = result.err().expect("oversized response was accepted");
            if mode == "oversize-success" {
                assert!(
                    error.contains("exceeded the maximum size of 16777216 bytes"),
                    "{error}"
                );
            } else {
                assert!(
                    error.contains("500"),
                    "HTTP status must survive size rejection: {error}"
                );
            }
        }
    }
}

#[tokio::test]
async fn stdio_discovery_calls_and_shutdown_reap_child() {
    let fixture = Fixture::new();
    let tool = tokio::time::timeout(
        WAIT,
        MCPWrapperTool::new(fixture.stdio_config("stdio")).initialized(),
    )
    .await
    .unwrap()
    .unwrap();
    exercise(tool).await;
    fixture.exited().await;
    assert_eq!(
        std::fs::read_to_string(fixture.dir.path().join("closed")).unwrap(),
        "stdin-eof"
    );
    assert_calls(&fixture.requests());
}

#[tokio::test]
async fn cancelled_stdio_handshake_reaps_child() {
    let fixture = Fixture::new();
    let work = tokio::spawn(MCPWrapperTool::new(fixture.stdio_config("stdio-stall")).initialized());
    fixture.ready().await;
    work.abort();
    assert!(matches!(work.await, Err(error) if error.is_cancelled()));
    fixture.exited().await;
}

#[tokio::test]
async fn timed_out_stdio_handshake_reaps_child() {
    let fixture = Fixture::new();
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        MCPWrapperTool::new(fixture.stdio_config("stdio-stall")).initialized(),
    )
    .await;
    assert!(result.is_err());
    fixture.exited().await;
}

// OpenSSL's trust environment is isolated to a subprocess; parallel tests never change process-wide trust.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn https_verifies_certificates_and_connects_with_an_explicit_test_ca() {
    let fixture = Fixture::http("https").await;
    let url = fixture.ready().await["url"].as_str().unwrap().to_owned();
    let result = tokio::time::timeout(
        WAIT,
        MCPWrapperTool::new(fixture.http_config(false).await).initialized(),
    )
    .await
    .unwrap();
    assert!(
        result.is_err(),
        "an untrusted self-signed certificate must be rejected"
    );
    for headers in [false, true] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "mcp::wrapper::transport_tests::https_trusted_child",
                "--ignored",
                "--nocapture",
            ])
            .env("MCP_TEST_URL", &url)
            .env("MCP_TEST_HEADERS", if headers { "yes" } else { "no" })
            .env("SSL_CERT_FILE", fixture.dir.path().join("cert.pem"))
            .env("SSL_CERT_DIR", fixture.dir.path())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let wait = tokio::time::timeout(WAIT, async {
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        if wait.is_err() {
            let _ = child.kill();
            let _ = child.wait();
        }
        wait.expect("HTTPS child timed out");
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("https_trusted_child ... ok"), "{stdout}");
        assert!(stdout.contains("1 passed; 0 failed"), "{stdout}");
    }
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "invoked only by the isolated HTTPS trust test"]
fn https_trusted_child() {
    assert!(std::path::Path::new(&std::env::var("SSL_CERT_FILE").unwrap()).is_file());
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let config = MCPWrapperConfig {
            name: "fixture".into(),
            transport: MCPWrapperTransport::Http {
                url: std::env::var("MCP_TEST_URL").unwrap(),
                headers: if std::env::var("MCP_TEST_HEADERS").unwrap() == "yes" {
                    HashMap::from([("Authorization".into(), "Bearer fixture-token".into())])
                } else {
                    HashMap::new()
                },
            },
            env: HashMap::new(),
            startup_timeout_ms: 2000,
            security: MCPWrapperSecurity {
                blocked_functions: vec!["blocked".into()],
                hitl_functions: vec!["approval".into()],
            },
            description: None,
            views: HashMap::new(),
        };
        let tool = tokio::time::timeout(WAIT, MCPWrapperTool::new(config).initialized())
            .await
            .unwrap()
            .unwrap();
        exercise(tool).await;
    });
}
