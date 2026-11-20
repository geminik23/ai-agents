//! Explicit YAML-to-runtime integration against a separately running Rust MCP server.

mod server;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const WAIT: Duration = Duration::from_secs(30);

/// Run the test-only MCP server in the current executable's child process.
pub fn serve(transport: &str, dir: &Path) -> Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(server::serve(transport, dir))
}

/// Build real YAML agents and compare runtime evidence with independent server effects.
pub fn run(root: &Path) -> Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(run_async(root))
}

struct OwnedServer(Child);
impl Drop for OwnedServer {
    // Failure cleanup only touches the directly owned child; the caller still reports missing acknowledgement.
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

// Wait for atomic publication or acknowledged termination; a deadline failure never counts as cleanup.
async fn wait_until(mut predicate: impl FnMut() -> Result<bool>) -> Result<()> {
    tokio::time::timeout(WAIT, async {
        loop {
            if predicate()? {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("MCP test deadline exceeded")?
}

// Process observation is limited to Unix, matching the currently enabled native CI matrix.
fn process_exists(pid: u64) -> Result<bool> {
    ensure!(
        cfg!(unix),
        "MCP integration process observation currently requires Unix"
    );
    Ok(Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(Stdio::null())
        .status()?
        .success())
}

// Only the model is replaced; EvalRunner retains the normal parser, MCP installer and shared tool executor.
async fn run_async(root: &Path) -> Result<()> {
    let run = root.join("target/mcp-integration").join(format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir_all(&run)?;
    let executable = std::env::current_exe()?;
    for transport in ["stdio", "http"] {
        let dir = run.join(transport);
        std::fs::create_dir(&dir)?;
        let mut owned = if transport == "http" {
            Some(OwnedServer(
                Command::new(&executable)
                    .args(["mcp-test-server", transport])
                    .arg(&dir)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::null())
                    .stderr(Stdio::inherit())
                    .spawn()?,
            ))
        } else {
            None
        };
        let mut spec: serde_yaml::Value =
            serde_yaml::from_str(include_str!("../tests/mcp/agent.yaml"))?;
        let config = spec["tools"][0].as_mapping_mut().unwrap();
        if let Some(server) = &mut owned {
            wait_until(|| {
                ensure!(
                    server.0.try_wait()?.is_none(),
                    "HTTP MCP server exited before readiness"
                );
                Ok(dir.join("ready.json").is_file())
            })
            .await?;
            let ready: Value = serde_json::from_slice(&std::fs::read(dir.join("ready.json"))?)?;
            config.insert("transport".into(), "http".into());
            config.insert("url".into(), ready["url"].as_str().unwrap().into());
        } else {
            config.insert(
                "command".into(),
                executable
                    .to_str()
                    .context("test executable path is not UTF-8")?
                    .into(),
            );
            config.insert(
                "args".into(),
                serde_yaml::to_value(["mcp-test-server", transport, dir.to_str().unwrap()])?,
            );
        }
        std::fs::write(dir.join("agent.yaml"), serde_yaml::to_string(&spec)?)?;
        std::fs::write(
            dir.join("suite.yaml"),
            include_str!("../tests/mcp/suite.yaml"),
        )?;
        let runner = ai_agents_eval::EvalRunner::from_file(
            dir.join("suite.yaml"),
            ai_agents_eval::EvalRunnerOptions {
                output: dir.join("report"),
                ..Default::default()
            },
        )?;
        let result = runner.run().await?;
        ai_agents_eval::write_outputs(&result, &dir.join("report"), true)?;
        ensure!(
            result.total == 1 && result.passed == 1 && result.failed == 0,
            "{transport} YAML eval failed: {}/{}; inspect {}",
            result.passed,
            result.total,
            dir.display()
        );
        let records: Vec<Value> = std::fs::read_to_string(dir.join("events.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            records.iter().any(|r| r["kind"] == "discovery"),
            "no actual SDK tool discovery"
        );
        let effects: Vec<_> = records
            .into_iter()
            .filter(|r| r["kind"] != "discovery")
            .collect();
        ensure!(
            effects
                == [
                    json!({"kind":"accumulate","delta":7,"total":7}),
                    json!({"kind":"fail"}),
                    json!({"kind":"accumulate","delta":11,"total":18})
                ],
            "unexpected server effects: {effects:?}"
        );
        let ready: Value = serde_json::from_slice(&std::fs::read(dir.join("ready.json"))?)?;
        if let Some(server) = &mut owned {
            drop(server.0.stdin.take());
            wait_until(|| Ok(server.0.try_wait()?.is_some())).await?;
            ensure!(
                server.0.wait()?.success(),
                "HTTP MCP server shutdown failed"
            );
        }
        wait_until(|| Ok(!process_exists(ready["pid"].as_u64().unwrap())?)).await?;
        ensure!(
            dir.join("closed").is_file(),
            "server exit had no orderly closure acknowledgement"
        );
        println!(
            "MCP integration {transport}: passed (YAML runtime, server effects and acknowledged exit)"
        );
    }
    println!("MCP integration reports: {}", run.display());
    Ok(())
}
