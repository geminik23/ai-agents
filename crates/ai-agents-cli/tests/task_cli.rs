//! Credential-free subprocess acceptance uses the production CLI/parser/provider with an isolated loopback HTTP fixture.

use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
#[cfg(unix)]
fn run_tui_pty(agent: &std::path::Path, initial: &str, expected: &str, answer: &str, mode: &str) {
    let out = Command::new("python3")
        // Match the isolated working directory used by the non-PTY producer and its relative resource bindings.
        .current_dir(agent.parent().unwrap())
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tui_pty.py"))
        .args([
            env!("CARGO_BIN_EXE_ai-agents-cli"),
            agent.to_str().unwrap(),
            initial,
            expected,
            answer,
            mode,
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "PTY failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// The supervisor must outlive the foreground CLI without masking a missing terminal restoration.
#[cfg(unix)]
#[test]
fn tui_pty_harness_preserves_terminal_and_detects_missing_restoration() {
    let out = Command::new("python3")
        .args(["-B", "-m", "unittest", "discover", "-s"])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))
        .args(["-p", "test_tui_pty.py", "-v"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "PTY harness regression failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(unix)]
#[test]
fn tui_pty_retained_question_and_terminal_restoration() {
    let fixture = Fixture::new(&["draft", "done"]);
    let agent = fixture.agent(&format!(
        "{TASK}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    run_tui_pty(
        &agent,
        "/task run 보고서 작성",
        "Agent: done",
        "continue",
        "complete",
    );
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 2);
}

#[cfg(unix)]
#[test]
fn tui_pty_quit_from_a_volatile_pause_settles_without_replay() {
    let fixture = Fixture::new(&["draft", "must not execute"]);
    let agent = fixture.agent(&format!(
        "{TASK}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    run_tui_pty(&agent, "/task run Research", "", "", "cancel");
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
}

#[cfg(unix)]
#[test]
fn tui_pty_ordinary_parent_services_independent_children_before_final() {
    let first = Fixture::new(&["draft", "done-first"]);
    let second = Fixture::new(&["draft", "done-second"]);
    let root = Fixture::new(&[]);
    let extra = "tools: []\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        max_turns: 3\n        max_llm_calls: 3\n        completion: {response_contains: done}\n        premature_finish: {action: ask_user, on_unavailable: stop}\n";
    let a = first.agent(extra);
    let b = second.agent(extra);
    let parent=root.agent(&format!("tools: []\nspawner:\n  shared_llms: false\n  auto_spawn:\n    - {{id: first, agent: '{}' }}\n    - {{id: second, agent: '{}' }}\nstates:\n  initial: collect\n  states:\n    collect:\n      concurrent:\n        agents: [first, second]\n        aggregation: {{strategy: all}}\n",a.display(),b.display()));
    run_tui_pty(
        &parent,
        "research",
        "done-first|done-second",
        "continue",
        "complete",
    );
    assert_eq!(first.calls.load(Ordering::SeqCst), 2);
    assert_eq!(second.calls.load(Ordering::SeqCst), 2);
    assert_eq!(root.calls.load(Ordering::SeqCst), 0);
}

#[cfg(unix)]
#[test]
fn tui_pty_report_requires_an_actual_isolated_file_effect() {
    let fixture = Fixture::new(&[]);
    let path = fixture.dir.join("report.txt");
    let call = json!({"tool":"file_write","arguments":{"path":path,"content":"validated report"}})
        .to_string();
    let provider = Fixture::new(&[&call, "done"]);
    let config=TASK.replace("tools: []","tools: [file_write]").replace("completion: {response_contains: done}","completion: {all: [{response_contains: done}, {tool_called: {id: file_write, count: 1, executed: true, success: true}}]}");
    let agent=provider.agent(&format!("{config}  hitl: {{on_approval_required: pause_run}}\nhitl: {{default_timeout_seconds: 60, on_timeout: reject}}\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_write:\n      write_paths: ['{}']\n      overwrite_existing: false\n",fixture.dir.display()));
    run_tui_pty(&agent, "/task run report", "Agent: done", "yes", "complete");
    assert_eq!(std::fs::read_to_string(path).unwrap(), "validated report");
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
}

#[cfg(unix)]
#[test]
fn tui_pty_task_owned_child_questions_preserve_completed_prefixes() {
    let reply =
        r#"{"tool":"ask_user","arguments":{"question":"Confirm source","options":["yes"]}}"#;
    let first = Fixture::new(&[reply, "done-first"]);
    let second = Fixture::new(&[reply, "done-second"]);
    let root = Fixture::new(&[]);
    let a = first.agent("tools: [ask_user]");
    let b = second.agent("tools: [ask_user]");
    std::fs::write(
        &a,
        std::fs::read_to_string(&a)
            .unwrap()
            .replace("name: Fixture", "name: FirstWorker"),
    )
    .unwrap();
    std::fs::write(
        &b,
        std::fs::read_to_string(&b)
            .unwrap()
            .replace("name: Fixture", "name: SecondWorker"),
    )
    .unwrap();
    let task = TASK.replace("tools: []", "tools: [ask_user]");
    let parent=root.agent(&format!("{task}  hitl: {{on_user_question: pause_run}}\nspawner:\n  shared_llms: false\n  allowed_tools: [ask_user]\n  auto_spawn:\n    - {{id: first, agent: '{}' }}\n    - {{id: second, agent: '{}' }}\nstates:\n  initial: collect\n  states:\n    collect:\n      concurrent:\n        agents: [first, second]\n        aggregation: {{strategy: all}}\n",a.display(),b.display()));
    run_tui_pty(
        &parent,
        "/task run research",
        "done-first|done-second",
        "yes",
        "complete",
    );
    assert_eq!(first.calls.load(Ordering::SeqCst), 2);
    assert_eq!(second.calls.load(Ordering::SeqCst), 2);
    assert_eq!(root.calls.load(Ordering::SeqCst), 0);
}

// Provider-free store commands retain exact access to legacy finite runs without reinterpreting their limits.
#[test]
fn task_store_commands_manage_legacy_finite_namespace() {
    use ai_agents::autonomy::{
        AutonomyConfig, AutonomyHostCeilings, AutonomyScope, TASK_RUN_SCHEMA_VERSION,
        TaskCheckpointPayload, TaskRunKey, TaskRunSnapshot, TaskRunStatus, TaskRuntimeCheckpoint,
        resolve_profile,
    };
    use ai_agents::{AgentSnapshot, AgentStorage, PreparedAgentSpec, SqliteStorage, TaskHostScope};
    let dir = std::env::temp_dir().join(format!(
        "task-cli-legacy-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let db = dir.join("tasks.db");
    let agent = dir.join("agent.yaml");
    let source = format!(
        "name: LegacyTask\nsystem_prompt: test\nstorage: {{type: sqlite, path: '{}'}}\n",
        db.display()
    );
    std::fs::write(&agent, &source).unwrap();
    let prepared = PreparedAgentSpec::from_yaml_strict(&source).unwrap();
    let legacy = TaskHostScope::from_spec(prepared.spec(), &dir, None).unwrap();
    let current = TaskHostScope::from_prepared_spec(&prepared, &dir, None).unwrap();
    assert_ne!(legacy.identity(), current.identity());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let storage = SqliteStorage::new(db.to_str().unwrap()).await.unwrap();
        let host = AutonomyHostCeilings {
            max_turns: Some(30),
            max_active_time_seconds: Some(900),
            max_llm_calls: Some(60),
            max_tool_calls: Some(120),
            max_command_calls: Some(10),
            ..Default::default()
        };
        let profile = resolve_profile(
            &AutonomyConfig::default(),
            AutonomyScope::Task,
            None,
            None,
            None,
            None,
            &host,
        )
        .unwrap();
        let mut payload = TaskCheckpointPayload::new(
            "legacy objective".into(),
            legacy.identity().into(),
            &profile,
            TaskRuntimeCheckpoint::between_turns(AgentSnapshot::new("LegacyTask".into())).unwrap(),
        );
        payload.version = 1;
        let now = chrono::Utc::now();
        let snapshot = payload
            .bind(TaskRunSnapshot {
                schema_version: TASK_RUN_SCHEMA_VERSION,
                key: TaskRunKey {
                    agent_id: "LegacyTask".into(),
                    run_id: "legacy-run".into(),
                },
                actor_id: None,
                revision: 0,
                status: TaskRunStatus::Paused,
                owner_token: None,
                cancel_requested: false,
                created_at: now,
                updated_at: now,
                payload: json!({}),
            })
            .unwrap();
        storage.create_task_run(&snapshot).await.unwrap();
        storage.close().await;
    });
    let run = |args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_ai-agents-cli"))
            .env_clear()
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "CLI failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<Value>(&output.stdout).unwrap()
    };
    let path = agent.to_str().unwrap();
    let rows = run(&["task", "list", path, "--json"]);
    assert_eq!(rows.as_array().unwrap().len(), 1);
    assert_eq!(rows[0]["key"]["run_id"], "legacy-run");
    let view = run(&["task", "show", path, "legacy-run", "--json"]);
    assert_eq!(view["limits"]["max_llm_calls"], 60);
    let receipt = run(&[
        "task",
        "cancel",
        path,
        "legacy-run",
        "--revision",
        "0",
        "--json",
    ]);
    assert_eq!(receipt["disposition"], "requested");
    let cancelled = run(&["task", "show", path, "legacy-run", "--json"]);
    assert_eq!(cancelled["summary"]["cancel_requested"], true);
    assert_eq!(cancelled["limits"]["max_llm_calls"], 60);
    std::fs::remove_dir_all(dir).unwrap();
}

#[derive(Debug, PartialEq)]
enum FixtureRequest {
    Generation(Value),
    Get,
}

// Accepted sockets have platform-dependent modes; request parsing requires bounded blocking reads on every platform.
// EOF/reset ends only this connection, while timeouts and malformed requests remain visible test failures.
fn read_fixture_request(
    socket: &mut std::net::TcpStream,
) -> std::io::Result<Option<FixtureRequest>> {
    use std::io::{Error, ErrorKind};

    socket.set_nonblocking(false)?;
    socket.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut read_more = |bytes: &mut Vec<u8>| -> std::io::Result<bool> {
        let mut buffer = [0; 4096];
        loop {
            match socket.read(&mut buffer) {
                Ok(0) => return Ok(false),
                Ok(n) => {
                    bytes.extend_from_slice(&buffer[..n]);
                    return Ok(true);
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error)
                    if matches!(
                        error.kind(),
                        ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted
                    ) =>
                {
                    return Ok(false);
                }
                Err(error) => return Err(error),
            }
        }
    };
    let mut bytes = Vec::new();
    let end = loop {
        if !read_more(&mut bytes)? {
            return Ok(None);
        }
        if let Some(i) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let headers = std::str::from_utf8(&bytes[..end])
        .map_err(|error| Error::new(ErrorKind::InvalidData, error))?;
    let mut lines = headers.lines();
    if lines.next().is_some_and(|line| line.starts_with("GET ")) {
        return Ok(Some(FixtureRequest::Get));
    }
    let len = lines
        .find_map(|line| {
            line.split_once(':')
                .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim().parse::<usize>())
        })
        .transpose()
        .map_err(|error| Error::new(ErrorKind::InvalidData, error))?
        .unwrap_or(0);
    while bytes.len() - end < len {
        if !read_more(&mut bytes)? {
            return Ok(None);
        }
    }
    serde_json::from_slice(&bytes[end..end + len])
        .map(|body| Some(FixtureRequest::Generation(body)))
        .map_err(|error| Error::new(ErrorKind::InvalidData, error))
}

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    dir: PathBuf,
    url: String,
    calls: Arc<AtomicUsize>,
    get_calls: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    gate: Arc<AtomicBool>,
    slow_output: bool,
    worker: Option<thread::JoinHandle<()>>,
}
impl Fixture {
    fn new(replies: &[&str]) -> Self {
        Self::observe_requests(replies, |_| {})
    }

    fn observe_requests(replies: &[&str], observe: impl Fn(&Value) + Send + 'static) -> Self {
        Self::serve(replies, observe, |body, _, reply| {
            let streaming = body["stream"] == true;
            Some(if streaming {
                (
                    "text/event-stream",
                    format!(
                        "data: {}\n\ndata: [DONE]\n\n",
                        json!({"choices":[{"delta":{"content":reply},"finish_reason":"stop"}]})
                    ),
                )
            } else {
                ("application/json",json!({"id":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":reply},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}).to_string())
            })
        })
    }

    fn serve(
        replies: &[&str],
        observe: impl Fn(&Value) + Send + 'static,
        response_body: impl Fn(&Value, usize, &str) -> Option<(&'static str, String)> + Send + 'static,
    ) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "task-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1/", listener.local_addr().unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let get_calls = Arc::new(AtomicUsize::new(0));
        let get_count = get_calls.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let gate = Arc::new(AtomicBool::new(false));
        let held = gate.clone();
        let replies: Vec<String> = replies.iter().map(|v| v.to_string()).collect();
        let worker = thread::spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let (mut socket, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => panic!("fixture accept failed: {error}"),
                };
                let Some(request) =
                    read_fixture_request(&mut socket).expect("fixture request read failed")
                else {
                    // A client may disconnect before sending a request; it does not stop the listener or consume a reply.
                    continue;
                };
                let FixtureRequest::Generation(body) = request else {
                    // Discovery is unavailable on this generation fixture.
                    get_count.fetch_add(1, Ordering::SeqCst);
                    let _ = socket.write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                };
                observe(&body);
                let index = count.fetch_add(1, Ordering::SeqCst);
                let reply = replies.get(index).cloned().unwrap_or_else(|| "done".into());
                while held.load(Ordering::Acquire) && !stopped.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(2));
                }
                let Some((kind, response)) = response_body(&body, index, &reply) else {
                    continue;
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",
                    response.len()
                );
                let _ = socket.write_all(response.as_bytes());
            }
        });
        Self {
            dir,
            url,
            calls,
            get_calls,
            stop,
            gate,
            slow_output: false,
            worker: Some(worker),
        }
    }
    fn agent(&self, extra: &str) -> PathBuf {
        let path = self.dir.join("agent.yaml");
        let yaml = format!(
            "name: Fixture\nsystem_prompt: Complete the declared objective.\nllm:\n  provider: openai-compatible\n  model: fixture\n  api_key_env: TASK_FIXTURE_TOKEN\n  base_url: {}\n{extra}\n",
            self.url
        );
        std::fs::write(&path, yaml).unwrap();
        path
    }
    fn interact(
        &self,
        args: &[&str],
        mut react: impl FnMut(&str, u32) -> Option<String>,
    ) -> Output {
        let mut child = Command::new(env!("CARGO_BIN_EXE_ai-agents-cli"))
            .args(args)
            .current_dir(&self.dir)
            .env("RUST_LOG", "off")
            .env("TASK_FIXTURE_TOKEN", "isolated-test-placeholder")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let errors = Arc::new(Mutex::new(Vec::<u8>::new()));
        let o = output.clone();
        let e = errors.clone();
        let a = thread::spawn(move || {
            let mut reader = stdout;
            let mut bytes = [0; 4096];
            while let Ok(n) = reader.read(&mut bytes) {
                if n == 0 {
                    break;
                }
                o.lock().unwrap().extend_from_slice(&bytes[..n]);
            }
        });
        let b = thread::spawn(move || {
            let mut reader = stderr;
            let mut bytes = [0; 4096];
            while let Ok(n) = reader.read(&mut bytes) {
                if n == 0 {
                    break;
                }
                e.lock().unwrap().extend_from_slice(&bytes[..n]);
            }
        });
        let started = Instant::now();
        let status = loop {
            let text = String::from_utf8_lossy(&output.lock().unwrap()).to_string();
            if let Some(reply) = react(&text, child.id()) {
                let _ = input.write_all(reply.as_bytes());
            }
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > Duration::from_secs(25) {
                self.gate.store(false, Ordering::Release);
                let _ = child.kill();
                let _ = child.wait();
                a.join().unwrap();
                b.join().unwrap();
                panic!(
                    "interactive CLI timeout\nstdout={text}\nstderr={}",
                    String::from_utf8_lossy(&errors.lock().unwrap())
                );
            }
            thread::sleep(Duration::from_millis(2));
        };
        a.join().unwrap();
        b.join().unwrap();
        let stdout = output.lock().unwrap().clone();
        let stderr = errors.lock().unwrap().clone();
        Output {
            status,
            stdout,
            stderr,
        }
    }

    fn run(&self, args: &[&str], input: &str) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ai-agents-cli"));
        command
            .args(args)
            .current_dir(&self.dir)
            .env("RUST_LOG", "off")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if args.first() == Some(&"run")
            || (args.first() == Some(&"task") && matches!(args.get(1), Some(&"run" | &"resume")))
        {
            command.env("TASK_FIXTURE_TOKEN", "isolated-test-placeholder");
        } else {
            command.env_remove("TASK_FIXTURE_TOKEN");
        }
        let mut child = command.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let err = Arc::new(Mutex::new(Vec::new()));
        let o = out.clone();
        let e = err.clone();
        let slow_output = self.slow_output;
        let a = thread::spawn(move || {
            let mut reader = stdout;
            let mut b = [0; 512];
            while let Ok(n) = reader.read(&mut b) {
                if n == 0 {
                    break;
                }
                o.lock().unwrap().extend_from_slice(&b[..n]);
                if slow_output {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        });
        let b = thread::spawn(move || {
            let mut reader = stderr;
            let mut b = [0; 4096];
            while let Ok(n) = reader.read(&mut b) {
                if n == 0 {
                    break;
                }
                e.lock().unwrap().extend_from_slice(&b[..n]);
            }
        });
        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if start.elapsed() > Duration::from_secs(25) {
                let _ = child.kill();
                let _ = child.wait();
                a.join().unwrap();
                b.join().unwrap();
                panic!(
                    "CLI timeout\nstdout={}\nstderr={}",
                    String::from_utf8_lossy(&out.lock().unwrap()),
                    String::from_utf8_lossy(&err.lock().unwrap())
                );
            }
            thread::sleep(Duration::from_millis(5));
        };
        a.join().unwrap();
        b.join().unwrap();
        let stdout = out.lock().unwrap().clone();
        let stderr = err.lock().unwrap().clone();
        Output {
            status,
            stdout,
            stderr,
        }
    }
}

#[test]
fn chat_sampling_respects_the_yaml_provider_model_and_option_presence() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = requests.clone();
    let fixture = Fixture::observe_requests(&[], move |body| {
        captured.lock().unwrap().push(body.clone());
    });
    let agent = fixture.agent("");
    let path = agent.to_str().unwrap();
    let run = |provider: &str, model: &str, options: &str, streaming| {
        std::fs::write(
            &agent,
            format!(
                "name: Fixture\nsystem_prompt: Test\nllm:\n  provider: {provider}\n  model: {model}\n  api_key_env: TASK_FIXTURE_TOKEN\n  base_url: {}\n{options}",
                fixture.url
            ),
        )
        .unwrap();
        let mut args = vec!["run", path, "--plain"];
        if streaming {
            args.push("--stream");
        }
        fixture.run(&args, "hello\n")
    };

    for (model, streaming) in [
        ("gpt-6-astra", false),
        ("gpt-6.1-sol", true),
        ("gpt-6.2-astra", false),
        ("gpt-6.2-sol", true),
        ("unregistered-model", false),
    ] {
        let output = run("openai", model, "  reasoning_effort: high\n", streaming);
        assert_exit(&output, 0);
        let body = requests.lock().unwrap().pop().unwrap();
        assert_eq!(body["model"], model);
        assert_eq!(body["stream"], streaming);
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["max_completion_tokens"], 2000);
        for field in ["temperature", "top_p", "logprobs", "top_logprobs"] {
            assert!(body.get(field).is_none(), "unexpected {field} in {body}");
        }
    }

    for (options, field) in [
        ("  temperature: 0.7\n", "temperature"),
        ("  top_p: 0.9\n", "top_p"),
        ("  provider_options: {logprobs: false}\n", "logprobs"),
        ("  extra_body: {top_logprobs: 0}\n", "top_logprobs"),
        (
            "  extra_body: {temperature: 0.7}\n  resilient: true\n",
            "temperature",
        ),
    ] {
        let output = run("openai", "gpt-6.2-sol", options, false);
        let error = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            error.contains(&format!("does not support {field}"))
                || stdout.contains(&format!("does not support {field}")),
            "stdout: {stdout}\nstderr: {error}"
        );
        assert!(requests.lock().unwrap().is_empty());
    }

    for (model, effort, allowed) in [
        ("gpt-6-astra", "low", false),
        ("gpt-6.1-sol", "low", false),
        ("gpt-6.1-terra", "none", true),
        ("gpt-6-luna", "none", true),
        ("gpt-6-sol", "high", false),
        ("gpt-5.4", "none", true),
        ("gpt-5.4", "high", false),
        ("unregistered-model", "future-effort", true),
    ] {
        let options =
            format!("  temperature: 0.3\n  provider_options: {{reasoning_effort: {effort}}}\n");
        let output = run("openai", model, &options, false);
        if allowed {
            assert_exit(&output, 0);
            let body = requests.lock().unwrap().pop().unwrap();
            assert_eq!(body["temperature"], 0.3);
            assert_eq!(body["reasoning_effort"], effort);
        } else {
            let diagnostic = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout)
            );
            assert!(
                diagnostic.contains("does not support temperature"),
                "{model}: {diagnostic}"
            );
            assert!(requests.lock().unwrap().is_empty());
        }
    }
    let output = run("openai-compatible", "gpt-6.2-sol", "  top_p: 0.8\n", false);
    assert_exit(&output, 0);
    let body = requests.lock().unwrap().pop().unwrap();
    assert_eq!(body["temperature"], 0.7);
    assert_eq!(body["top_p"], 0.8);
    assert!(requests.lock().unwrap().is_empty());
    assert_eq!(fixture.get_calls.load(Ordering::SeqCst), 0);
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let worker_result = self.worker.take().map(|worker| worker.join());
        let _ = std::fs::remove_dir_all(&self.dir);
        if let Some(Err(error)) = worker_result {
            // Keep worker failures fatal, but preserve the primary failure when the test is already unwinding.
            if thread::panicking() {
                eprintln!("fixture worker also failed during test cleanup");
            } else {
                std::panic::resume_unwind(error);
            }
        }
    }
}
#[test]
fn fixture_request_waits_for_fragmented_data_on_a_nonblocking_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (mut socket, _) = listener.accept().unwrap();
    // Reproduce the accepted-socket mode independently of the host operating system.
    socket.set_nonblocking(true).unwrap();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    let worker = thread::spawn(move || {
        ready_tx.send(()).unwrap();
        result_tx.send(read_fixture_request(&mut socket)).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_millis(25)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    client.write_all(b"POST / HTTP/1.1\r\nContent-Len").unwrap();
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_millis(25)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    client.write_all(b"gth: 2\r\n\r\n{").unwrap();
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_millis(25)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    client.write_all(b"}").unwrap();
    assert_eq!(
        result_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap(),
        Some(FixtureRequest::Generation(json!({})))
    );
    worker.join().unwrap();
}

#[test]
fn fixture_incomplete_connection_does_not_stop_server_or_consume_reply() {
    let fixture = Fixture::new(&["first-reply"]);
    let address = fixture
        .url
        .trim_start_matches("http://")
        .trim_end_matches("/v1/");
    let mut abandoned = std::net::TcpStream::connect(address).unwrap();
    abandoned
        .write_all(b"POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n{")
        .unwrap();
    drop(abandoned);

    let mut client = std::net::TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client
        .write_all(b"POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}")
        .unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(response.contains("first-reply"), "{response}");
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn fixture_worker_failure_is_reported_without_replacing_primary_panic() {
    for primary_panic in [false, true] {
        let fixture = Fixture::new(&[]);
        let address = fixture
            .url
            .trim_start_matches("http://")
            .trim_end_matches("/v1/");
        let mut client = std::net::TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        client
            .write_all(b"POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\nxx")
            .unwrap();
        // EOF acknowledges the worker has failed and closed this connection before testing cleanup.
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _fixture = fixture;
            if primary_panic {
                panic!("primary fixture test failure");
            }
        }));
        let error = result.expect_err("worker failure must not be swallowed");
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap();
        if primary_panic {
            assert_eq!(message, "primary fixture test failure");
        } else {
            assert!(message.contains("fixture request read failed"), "{message}");
        }
    }
}

const TASK: &str = "tools: []\nautonomy:\n  enabled: true\n  mode: until_complete\n  max_turns: 4\n  max_llm_calls: 4\n  completion: {response_contains: done}\n";

// A host without synchronized kernel time must prove safe refusal through the production binary.
// Acceptance hosts can require the positive continuation with a test-only setting.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn assert_configured_seed(
    fixture: &Fixture,
    output: &Output,
    expected: i32,
    agent: &std::path::Path,
    db: &std::path::Path,
) -> bool {
    if output.status.code() == Some(expected) {
        println!("native restoration: prepared producer accepted");
        return true;
    }
    const CLOCK_REFUSAL: &str =
        "task restoration requires synchronized OS time with a valid error bound";
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    if std::env::var_os("AI_AGENTS_REQUIRE_NATIVE_RESTORE_CLOCK").as_deref()
        != Some(std::ffi::OsStr::new("1"))
        && diagnostic.contains(CLOCK_REFUSAL)
        && ai_agents::autonomy::SystemTaskClock::sample().is_err()
    {
        let automatic = expected == 0;
        assert_exit(output, if automatic { 1 } else { 2 });
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 0);
        if !automatic {
            assert!(
                !db.exists(),
                "task run must refuse before opening the task journal"
            );
        }
        if db.exists() {
            let list = fixture.run(&["task", "list", agent.to_str().unwrap(), "--json"], "");
            assert_exit(&list, 0);
            let rows: Value = serde_json::from_slice(&list.stdout).unwrap();
            assert_eq!(rows, json!([]), "refusal must leave no task run");
        }
        println!("native restoration: refused before model calls and task creation: {diagnostic}");
        return false;
    }
    assert_exit(output, expected);
    unreachable!("the exit assertion reports any unclassified failure");
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// Scope flags are shared between producer, inspection and consumer without changing the supplied revision.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn scoped_args<'a>(mut args: Vec<&'a str>, actor: Option<&'a str>) -> Vec<&'a str> {
    if let Some(actor) = actor {
        args.extend(["--actor", actor]);
    }
    args
}

// Two real CLI processes use the configured factory and surviving SQLite checkpoint, never a test restoration host.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn task_subprocess_configured_restart_preserves_attempts_and_requires_fresh_review() {
    for actor in [None, Some("alice")] {
        let f = Fixture::new(&["draft", "done"]);
        let db = f.dir.join("restart.sqlite");
        let agent = f.agent(&format!(
            "{TASK}  restoration_timeout_ms: 5000\n  persistence: {{enabled: true}}\n  premature_finish: {{action: ask_user, on_unavailable: stop}}\nstorage: {{type: sqlite, path: '{}' }}\n",
            db.display()
        ));
        let first = f.run(
            &scoped_args(
                vec![
                    "task",
                    "run",
                    agent.to_str().unwrap(),
                    "finish the report",
                    "--json",
                    "--non-interactive",
                ],
                actor,
            ),
            "",
        );
        if !assert_configured_seed(&f, &first, 3, &agent, &db) {
            continue;
        }
        let saved: Value = serde_json::from_str(
            String::from_utf8_lossy(&first.stdout)
                .lines()
                .last()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(saved["status"], "paused");
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        let id = saved["run_id"].as_str().unwrap();
        let revision = saved["revision"].as_u64().unwrap().to_string();
        let show = || {
            let out = f.run(
                &scoped_args(
                    vec!["task", "show", agent.to_str().unwrap(), id, "--json"],
                    actor,
                ),
                "",
            );
            assert_exit(&out, 0);
            serde_json::from_slice::<Value>(&out.stdout).unwrap()
        };
        let before = show();
        let stale = (saved["revision"].as_u64().unwrap() - 1).to_string();
        let rejected = f.run(
            &scoped_args(
                vec![
                    "task",
                    "resume",
                    agent.to_str().unwrap(),
                    id,
                    "--revision",
                    &stale,
                    "--json",
                ],
                actor,
            ),
            "yes\ncontinue\n",
        );
        assert_exit(&rejected, 2);
        assert_eq!(show(), before);
        let wrong_actor = f.run(
            &[
                "task",
                "resume",
                agent.to_str().unwrap(),
                id,
                "--revision",
                &revision,
                "--actor",
                "someone-else",
            ],
            "yes\ncontinue\n",
        );
        assert_exit(&wrong_actor, 2);
        let original = std::fs::read_to_string(&agent).unwrap();
        std::fs::write(
            &agent,
            original.replace(
                "Complete the declared objective.",
                "A different configured objective.",
            ),
        )
        .unwrap();
        let changed = f.run(
            &scoped_args(
                vec![
                    "task",
                    "resume",
                    agent.to_str().unwrap(),
                    id,
                    "--revision",
                    &revision,
                ],
                actor,
            ),
            "yes\ncontinue\n",
        );
        assert_exit(&changed, 2);
        std::fs::write(&agent, original).unwrap();
        assert_eq!(show(), before);
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        let resumed = f.run(
            &scoped_args(
                vec![
                    "task",
                    "resume",
                    agent.to_str().unwrap(),
                    id,
                    "--revision",
                    &revision,
                    "--json",
                ],
                actor,
            ),
            "yes\ncontinue\n",
        );
        assert_exit(&resumed, 0);
        let done: Value = serde_json::from_str(
            String::from_utf8_lossy(&resumed.stdout)
                .lines()
                .last()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(done["status"], "completed");
        assert_eq!(done["run_id"], saved["run_id"]);
        assert_eq!(done["counters"]["llm_attempts"], 2);
        assert_eq!(f.calls.load(Ordering::SeqCst), 2);
        assert!(String::from_utf8_lossy(&resumed.stderr).contains("Restoration"));
    }
}

#[test]
fn task_subprocess_profiles_are_selected_without_implicitly_enabling_disabled_work() {
    let f = Fixture::new(&["done"]);
    let agent = f.agent(&format!(
        "{TASK}  profiles:\n    enabled: {{enabled: true}}\n    disabled: {{enabled: false}}\n"
    ));
    let denied = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "objective",
            "--profile",
            "disabled",
        ],
        "",
    );
    assert_exit(&denied, 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    let done = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "objective",
            "--profile",
            "enabled",
            "--non-interactive",
        ],
        "",
    );
    assert_exit(&done, 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn task_subprocess_durable_request_rejects_before_storage_or_provider_work() {
    let f = Fixture::new(&[]);
    let db = f.dir.join("must-not-create.sqlite");
    let agent = f.agent(&format!(
        "{TASK}  persistence: {{enabled: true}}\nstorage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let out = f.run(&["task", "run", agent.to_str().unwrap(), "objective"], "");
    assert_exit(&out, 2);
    assert!(!db.exists());
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn task_subprocess_retained_stage_review_and_cancel_preserve_exact_results() {
    for answer in ["yes\n", "cancel\n"] {
        let f = Fixture::new(&["done"]);
        let agent=f.agent(&format!("{TASK}  lifecycle: [{{id: research, instruction: Collect findings}}]\n  supervision: {{stage_changes: require_approval}}\n"));
        let out = f.run(
            &["task", "run", agent.to_str().unwrap(), "research"],
            answer,
        );
        assert_exit(&out, if answer.starts_with("yes") { 0 } else { 1 });
        assert!(String::from_utf8_lossy(&out.stdout).contains("Stage review:"));
        assert!(
            String::from_utf8_lossy(&out.stdout).contains(if answer.starts_with("yes") {
                ": Completed"
            } else {
                ": Cancelled"
            })
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn task_subprocess_tool_question_multiselect_has_one_executed_prefix() {
    let f = Fixture::new(&[
        r#"{"tool":"ask_user","arguments":{"question":"Choose sources","options":["A","B"],"multi_select":true}}"#,
        "done",
    ]);
    let yaml = TASK.replace("tools: []", "tools: [ask_user]");
    let agent = f.agent(&format!("{yaml}  hitl: {{on_user_question: pause_run}}\n"));
    let out = f.run(
        &["task", "run", agent.to_str().unwrap(), "research"],
        "1,2\n",
    );
    assert_exit(&out, 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn task_subprocess_ordinary_question_and_stream_flags_remain_usable() {
    let f = Fixture::new(&[
        r#"{"tool":"ask_user","arguments":{"question":"Choose source","options":["A","B"]}}"#,
        "done",
    ]);
    let agent = f.agent("tools: [ask_user]");
    let out = f.run(
        &["run", agent.to_str().unwrap(), "--plain", "--show-tools"],
        "research\n1\n",
    );
    assert_exit(&out, 0);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("ordinary question"));
    assert!(text.contains("done"));
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    let f = Fixture::new(&["streamed answer"]);
    let agent = f.agent("tools: []");
    let out = f.run(
        &["run", agent.to_str().unwrap(), "--plain", "--stream"],
        "research\n",
    );
    assert_exit(&out, 0);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout)
            .matches("streamed answer")
            .count(),
        1
    );
}

#[test]
fn task_subprocess_state_and_selected_skill_automatic_entry_reuse_the_controller() {
    for skill in [false, true] {
        let f = Fixture::new(if skill {
            &["research", "draft", "done"]
        } else {
            &["draft", "done"]
        });
        let extra = if skill {
            "tools: []\nskills:\n  - id: research\n    description: Research the request\n    trigger: User requests research\n    autonomy:\n      enabled: true\n      mode: skill_until_complete\n      max_turns: 4\n      max_llm_calls: 4\n      completion: {response_contains: done}\n      premature_finish: {action: ask_user, on_unavailable: stop}\n    steps:\n      - prompt: 'Research {{ user_input }}'\n"
        } else {
            "tools: []\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        max_turns: 4\n        max_llm_calls: 4\n        completion: {response_contains: done}\n        premature_finish: {action: ask_user, on_unavailable: stop}\n"
        };
        let agent = f.agent(extra);
        if skill {
            let yaml = std::fs::read_to_string(&agent).unwrap();
            let prefix = yaml.split("tools: []").next().unwrap();
            let provider = prefix.split("llm:\n").nth(1).unwrap();
            let yaml = format!(
                "name: Fixture\nsystem_prompt: Research\nllm: {{default: default, router: router}}\nllms:\n  default:\n{}\n  router:\n{}\n{}",
                provider
                    .lines()
                    .map(|s| format!("  {s}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                provider
                    .lines()
                    .map(|s| format!("  {s}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                extra
            );
            std::fs::write(&agent, yaml).unwrap();
        }
        let out = f.run(
            &["run", agent.to_str().unwrap(), "--plain", "--stream"],
            "Research the sources\ncontinue research\n",
        );
        assert_exit(&out, 0);
        assert!(String::from_utf8_lossy(&out.stdout).contains("Completed"));
        assert_eq!(f.calls.load(Ordering::SeqCst), if skill { 3 } else { 2 });
    }
}

#[test]
fn task_subprocess_ordinary_parent_services_two_independent_child_questions_before_final() {
    let first = Fixture::new(&["draft", "done-first"]);
    let second = Fixture::new(&["draft", "done-second"]);
    let root = Fixture::new(&[]);
    let extra = "tools: []\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        max_turns: 3\n        max_llm_calls: 3\n        completion: {response_contains: done}\n        premature_finish: {action: ask_user, on_unavailable: stop}\n";
    let a = first.agent(extra);
    let b = second.agent(extra);
    let parent=root.agent(&format!("tools: []\nspawner:\n  shared_llms: false\n  auto_spawn:\n    - {{id: first, agent: '{}' }}\n    - {{id: second, agent: '{}' }}\nstates:\n  initial: collect\n  states:\n    collect:\n      concurrent:\n        agents: [first, second]\n        aggregation: {{strategy: all}}\n",a.display(),b.display()));
    let out = root.run(
        &["run", parent.to_str().unwrap(), "--plain"],
        "research\ncontinue first\ncontinue second\n",
    );
    assert_exit(&out, 0);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("done-first"));
    assert!(text.contains("done-second"));
    assert_eq!(first.calls.load(Ordering::SeqCst), 2);
    assert_eq!(second.calls.load(Ordering::SeqCst), 2);
    assert_eq!(root.calls.load(Ordering::SeqCst), 0);
}

fn pending_header(text: &str) -> Option<(String, u64, String)> {
    let line = text.lines().rev().find(|line| line.contains(" request "))?;
    let words: Vec<_> = line.trim_matches(['[', ']']).split_whitespace().collect();
    Some((
        words.get(1)?.to_string(),
        words.get(3)?.parse().ok()?,
        words.get(5)?.to_string(),
    ))
}

#[test]
fn task_subprocess_stale_response_is_not_rebound_and_explicit_resume_works() {
    let f = Fixture::new(&["draft", "done"]);
    let agent = f.agent(&format!(
        "{TASK}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    let mut step = 0;
    let out = f.interact(
        &["task", "run", agent.to_str().unwrap(), "research"],
        |text, _| {
            if step == 0 {
                if let Some((id, rev, request)) = pending_header(text) {
                    step = 1;
                    return Some(format!(
                        "/task resume {id} {} {request} continue\n",
                        rev + 1
                    ));
                }
            } else if step == 1 && text.contains("control rejected") {
                let (id, rev, request) = pending_header(text).unwrap();
                step = 2;
                return Some(format!("/task resume {id} {rev} {request} continue\n"));
            }
            None
        },
    );
    assert_exit(&out, 0);
    assert_eq!(step, 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn task_subprocess_objective_revision_requires_explicit_continue_and_preserves_usage() {
    let f = Fixture::new(&["draft", "done"]);
    let agent = f.agent(&format!(
        "{TASK}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    let mut step = 0;
    let out = f.interact(
        &["task", "run", agent.to_str().unwrap(), "research"],
        |text, _| {
            if step == 0 {
                if let Some((id, rev, _)) = pending_header(text) {
                    step = 1;
                    return Some(format!("/task objective {id} {rev} revised research\n"));
                }
            } else if step == 1 && text.contains("Host continuation:") {
                step = 2;
                return Some("continue\n".into());
            }
            None
        },
    );
    assert_exit(&out, 0);
    assert_eq!(step, 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn task_subprocess_remote_cancel_is_a_request_until_original_operation_settles() {
    remote_cancel_case(false);
}

#[test]
fn task_overrides_remote_cancel_without_original_file_preserves_settlement() {
    remote_cancel_case(true);
}

// The live owner settles the original operation; journal management must not reread invocation input or create providers.
fn remote_cancel_case(with_overrides: bool) {
    let f = Fixture::new(&["done"]);
    f.gate.store(true, Ordering::Release);
    let db = f.dir.join("tasks.sqlite");
    let agent = f.agent(&format!(
        "{TASK}storage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let overlay = f.dir.join("invocation.yaml");
    let mut args = vec![
        "task",
        "run",
        agent.to_str().unwrap(),
        "operations",
        "--json",
    ];
    if with_overrides {
        std::fs::write(&overlay, "max_llm_calls: 1\n").unwrap();
        args.extend(["--overrides", overlay.to_str().unwrap()]);
    }
    let mut requested = false;
    let out = f.interact(&args, |text, _| {
        if !requested && f.calls.load(Ordering::SeqCst) > 0 {
            let id = text
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .find_map(|v| v["run_id"].as_str().map(str::to_owned));
            if let Some(id) = id {
                if with_overrides {
                    std::fs::remove_file(&overlay).unwrap();
                }
                let view = f.run(
                    &["task", "show", agent.to_str().unwrap(), &id, "--json"],
                    "",
                );
                assert_exit(&view, 0);
                let view: Value = serde_json::from_slice(&view.stdout).unwrap();
                let rev = view["summary"]["revision"].as_u64().unwrap().to_string();
                let receipt = f.run(
                    &[
                        "task",
                        "cancel",
                        agent.to_str().unwrap(),
                        &id,
                        "--revision",
                        &rev,
                        "--json",
                    ],
                    "",
                );
                assert_exit(&receipt, 0);
                let receipt: Value = serde_json::from_slice(&receipt.stdout).unwrap();
                assert_eq!(receipt["disposition"], "requested");
                assert_eq!(receipt["locally_retained"], Value::Null);
                requested = true;
                f.gate.store(false, Ordering::Release);
            }
        }
        None
    });
    assert_exit(&out, 1);
    assert!(requested);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    let final_result = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| value["type"] == "result")
        .expect("the original foreground operation must acknowledge a final result");
    assert_eq!(final_result["status"], "cancelled");
}

#[cfg(unix)]
#[test]
fn task_subprocess_signal_keeps_original_foreground_polling() {
    let f = Fixture::new(&["done"]);
    f.gate.store(true, Ordering::Release);
    let db = f.dir.join("signal.sqlite");
    let agent = f.agent(&format!(
        "{TASK}storage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let mut sent = false;
    let mut accepted = false;
    let out = f.interact(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "operations",
            "--json",
        ],
        |text, pid| {
            if !sent && f.calls.load(Ordering::SeqCst) > 0 {
                sent = true;
                assert!(
                    Command::new("kill")
                        .args(["-INT", &pid.to_string()])
                        .status()
                        .unwrap()
                        .success()
                );
            }
            if sent
                && !accepted
                && let Some(id) = text
                    .lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .find_map(|v| v["run_id"].as_str().map(str::to_owned))
            {
                let view = f.run(
                    &["task", "show", agent.to_str().unwrap(), &id, "--json"],
                    "",
                );
                assert_exit(&view, 0);
                let view: Value = serde_json::from_slice(&view.stdout).unwrap();
                if view["summary"]["cancel_requested"] == true {
                    accepted = true;
                    f.gate.store(false, Ordering::Release);
                }
            }
            None
        },
    );
    assert_exit(&out, 1);
    assert!(sent && accepted);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert!(!String::from_utf8_lossy(&out.stderr).contains("Forced exit"));
}

#[test]
fn task_subprocess_task_owned_child_questions_keep_one_coordinating_request() {
    let reply =
        r#"{"tool":"ask_user","arguments":{"question":"Confirm source","options":["yes"]}}"#;
    let first = Fixture::new(&[reply, "done-first"]);
    let second = Fixture::new(&[reply, "done-second"]);
    let root = Fixture::new(&[]);
    let a = first.agent("tools: [ask_user]");
    let b = second.agent("tools: [ask_user]");
    std::fs::write(
        &a,
        std::fs::read_to_string(&a)
            .unwrap()
            .replace("name: Fixture", "name: FirstWorker"),
    )
    .unwrap();
    std::fs::write(
        &b,
        std::fs::read_to_string(&b)
            .unwrap()
            .replace("name: Fixture", "name: SecondWorker"),
    )
    .unwrap();
    let task = TASK.replace("tools: []", "tools: [ask_user]");
    let parent=root.agent(&format!("{task}  hitl: {{on_user_question: pause_run}}\nspawner:\n  shared_llms: false\n  allowed_tools: [ask_user]\n  auto_spawn:\n    - {{id: first, agent: '{}' }}\n    - {{id: second, agent: '{}' }}\nstates:\n  initial: collect\n  states:\n    collect:\n      concurrent:\n        agents: [first, second]\n        aggregation: {{strategy: all}}\n",a.display(),b.display()));
    let out = root.run(
        &["task", "run", parent.to_str().unwrap(), "research"],
        "1\n1\n",
    );
    assert_exit(&out, 0);
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(text.matches("Question: Confirm source").count(), 2);
    assert_eq!(first.calls.load(Ordering::SeqCst), 2);
    assert_eq!(second.calls.load(Ordering::SeqCst), 2);
    assert_eq!(root.calls.load(Ordering::SeqCst), 0);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn decisions_task_restores_selected_child_without_reselection() {
    let call = json!({"tool":"route_to_agent","arguments":{
        "input":"Investigate the failed build","candidates":["ops","devops"],"method":"llm"
    }});
    let root = Fixture::new(&[&call.to_string(), "done"]);
    let ops = Fixture::new(&[]);
    let child_call =
        r#"{"tool":"ask_user","arguments":{"question":"Confirm the build log","options":["yes"]}}"#;
    let devops = Fixture::new(&[child_call, "done-devops"]);
    let router = Fixture::serve(
        &[],
        |body| {
            assert_eq!(body["model"], "fixture-decision");
            assert_eq!(body["input"], "Investigate the failed build");
            assert_eq!(
                body["questions"][0]["choices"],
                json!([{"value":"ops"},{"value":"devops"}])
            );
            assert!(body.get("max_tokens").is_none() && body.get("temperature").is_none());
        },
        |body, index, _| {
            assert_eq!(index, 0, "decision must not replay");
            Some(("application/json", json!({
                "answers":[{"type":"choice","name":body["questions"][0]["name"],"choice":"devops"}],
                "model":"fixture-decision","usage":{"input_tokens":19,"output_tokens":1,"total_tokens":20}
            }).to_string()))
        },
    );
    let child = |fixture: &Fixture, name: &str| {
        let path = fixture.agent("tools: [ask_user]");
        let yaml = std::fs::read_to_string(&path)
            .unwrap()
            .replace("name: Fixture", name);
        std::fs::write(&path, yaml).unwrap();
        path
    };
    let a = child(&ops, "name: Ops");
    let b = child(&devops, "name: DevOps");
    let db = root.dir.join("decisions.sqlite");
    let task = TASK
        .replace("tools: []", "tools: [route_to_agent, ask_user]")
        .replace("max_llm_calls: 4", "max_llm_calls: 5");
    let agent = root.dir.join("agent.yaml");
    let yaml = format!(
        r#"
name: DecisionRoot
system_prompt: Route to the right team.
llm:
  default: main
  router:
    orchestration:
      routing: work_queue
llms:
  main:
    provider: openai-compatible
    model: fixture
    base_url: {main}
    api_key_env: TASK_FIXTURE_TOKEN
  work_queue:
    provider: openai
    protocol: decisions
    model: fixture-decision
    base_url: {router}
    api_key_env: TASK_FIXTURE_TOKEN
{task}  max_wall_time_seconds: 600
  restoration_timeout_ms: 5000
  persistence: {{enabled: true}}
  hitl: {{on_user_question: pause_run}}
storage: {{type: sqlite, path: '{db}'}}
spawner:
  shared_llms: false
  orchestration_tools: [route_to_agent]
  allowed_tools: [ask_user]
  auto_spawn:
    - {{id: ops, agent: '{a}'}}
    - {{id: devops, agent: '{b}'}}
"#,
        main = root.url,
        router = router.url,
        db = db.display(),
        a = a.display(),
        b = b.display()
    );
    std::fs::write(&agent, yaml).unwrap();
    let path = agent.to_str().unwrap();
    let seed = root.run(
        &[
            "task",
            "run",
            path,
            "build help",
            "--non-interactive",
            "--json",
        ],
        "",
    );
    if !assert_configured_seed(&root, &seed, 3, &agent, &db) {
        assert_eq!(router.calls.load(Ordering::SeqCst), 0);
        return;
    }
    let result = |out: &Output| {
        serde_json::from_str::<Value>(String::from_utf8_lossy(&out.stdout).lines().last().unwrap())
            .unwrap()
    };
    let saved = result(&seed);
    assert_eq!(saved["status"], "paused");
    assert_eq!(saved["counters"]["llm_attempts"], 3);
    assert_eq!(router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(ops.calls.load(Ordering::SeqCst), 0);
    assert_eq!(devops.calls.load(Ordering::SeqCst), 1);
    let resumed = root.run(
        &[
            "task",
            "resume",
            path,
            saved["run_id"].as_str().unwrap(),
            "--revision",
            &saved["revision"].to_string(),
            "--json",
        ],
        // Restoration review and the original child question are distinct host decisions.
        "yes\n1\n",
    );
    assert_exit(&resumed, 0);
    let done = result(&resumed);
    assert_eq!(done["status"], "completed");
    assert_eq!(done["run_id"], saved["run_id"]);
    assert_eq!(done["counters"]["llm_attempts"], 5);
    assert_eq!(router.calls.load(Ordering::SeqCst), 1);
    assert_eq!(ops.calls.load(Ordering::SeqCst), 0);
    assert_eq!(devops.calls.load(Ordering::SeqCst), 2);
    assert_eq!(root.calls.load(Ordering::SeqCst), 2);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            use ai_agents::persistence::{AgentStorage, SqliteStorage};
            let store = SqliteStorage::new(db.to_str().unwrap()).await.unwrap();
            let key = ai_agents::autonomy::TaskRunKey {
                agent_id: "DecisionRoot".into(),
                run_id: saved["run_id"].as_str().unwrap().into(),
            };
            let snapshot = store.load_task_run(&key).await.unwrap().unwrap();
            let choices: Vec<_> = snapshot.payload["reservations"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|attempt| {
                    let result = &attempt["result"];
                    (result["choice_completed"] == true).then_some(&result["usage"])
                })
                .collect();
            assert_eq!(
                choices,
                [&json!({"prompt_tokens":19,"completion_tokens":1,"total_tokens":20})]
            );
        });
}

#[test]
fn ollama_cloud_yaml_streams_without_catalog_and_keeps_legacy_local_ollama() {
    for cloud in [true, false] {
        let fixture = Fixture::serve(
            &[],
            |_| {},
            move |body, _, _| {
                assert_eq!(body["model"], "fixture:chosen");
                assert_eq!(body["stream"], cloud);
                // Exercise local Ollama's existing blocking contract and Cloud's native NDJSON path.
                if !cloud {
                    return Some((
                        "application/json",
                        json!({
                            "model":"fixture:chosen","created_at":"2026-11-22T00:00:00Z",
                            "message":{"role":"assistant","content":"verified reply"},"done":true
                        })
                        .to_string(),
                    ));
                }
                assert!(body.get("options").is_none());
                assert!(body.get("tools").is_none());
                Some((
                    "application/x-ndjson",
                    format!(
                        "{}\n{}\n",
                        json!({"model":"fixture:chosen","created_at":"2026-11-22T00:00:00Z","message":{"role":"assistant","content":"verified reply","thinking":"private text"},"done":false}),
                        json!({"model":"fixture:chosen","created_at":"2026-11-22T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}),
                    ),
                ))
            },
        );
        let agent = fixture.agent("");
        let source = std::fs::read_to_string(&agent)
            .unwrap()
            .replace("model: fixture", "model: fixture:chosen")
            .replace(
                "provider: openai-compatible",
                if cloud {
                    "provider: ollama-cloud"
                } else {
                    "provider: ollama"
                },
            );
        let source = if cloud {
            source
        } else {
            source
                .replace("  api_key_env: TASK_FIXTURE_TOKEN\n", "")
                .replace(&fixture.url, fixture.url.trim_end_matches("/v1/"))
        };
        std::fs::write(&agent, source).unwrap();
        let mut args = vec!["run", agent.to_str().unwrap(), "--plain"];
        if cloud {
            args.push("--stream");
        }
        let out = fixture.run(&args, "hello\n");
        assert_exit(&out, 0);
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("verified reply"),
            "{out:?}"
        );
        assert!(!String::from_utf8_lossy(&out.stdout).contains("private text"));
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
        assert_eq!(fixture.get_calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn ollama_cloud_task_omitted_tool_choice_keeps_granted_tools_out_of_the_prompt() {
    let fixture = Fixture::serve(
        &[],
        |_| {},
        |body, _, _| {
            assert!(body.get("tools").is_none());
            for message in body["messages"].as_array().unwrap() {
                assert!(!message["content"].as_str().unwrap().contains("file_write"));
            }
            Some(("application/json", json!({"message":{"role":"assistant","content":"done"},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":1}).to_string()))
        },
    );
    let agent = fixture.agent(&TASK.replace("tools: []", "tools: [file_write]"));
    let source = std::fs::read_to_string(&agent)
        .unwrap()
        .replace("provider: openai-compatible", "provider: ollama-cloud");
    std::fs::write(&agent, source).unwrap();
    let out = fixture.run(
        &["task", "run", agent.to_str().unwrap(), "answer", "--json"],
        "",
    );
    assert_exit(&out, 0);
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn ollama_cloud_malformed_or_rejected_tool_turns_have_no_effect() {
    for failure in [
        "malformed",
        "unknown",
        "multiple",
        "prompt-only",
        "rejected",
        "disconnect",
    ] {
        let files = Fixture::new(&[]);
        let path = files.dir.join("result.txt");
        let target = path.clone();
        let provider = Fixture::serve(
            &[],
            |_| {},
            move |_, index, _| {
                if index != 0 {
                    assert_eq!(failure, "disconnect");
                    return None;
                }
                let args = if failure == "malformed" {
                    json!("not an object")
                } else {
                    json!({"path":target,"content":"approved content"})
                };
                let call = json!({"function":{"name":if failure == "unknown" {"missing_tool"} else {"file_write"}, "arguments":args}});
                let message = if failure == "prompt-only" {
                    json!({"role":"assistant","content":json!({"tool":"file_write","arguments":{"path":target,"content":"forged"}}).to_string()})
                } else {
                    let calls = if failure == "multiple" {
                        vec![call.clone(), call]
                    } else {
                        vec![call]
                    };
                    json!({"role":"assistant","content":"","tool_calls":calls})
                };
                Some((
                    "application/json",
                    json!({"message":message,"done":true,"done_reason":"stop"}).to_string(),
                ))
            },
        );
        let task = TASK.replace("tools: []", "tools: [file_write]");
        let task = if failure == "prompt-only" {
            task.replace("response_contains: done", "response_contains: forged")
        } else {
            task
        };
        let agent=provider.agent(&format!("{task}  hitl: {{on_approval_required: pause_run}}\nhitl: {{default_timeout_seconds: 60, on_timeout: reject}}\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_write:\n      write_paths: ['{}']\n",files.dir.display()));
        let source = std::fs::read_to_string(&agent).unwrap().replace(
            "provider: openai-compatible",
            "provider: ollama-cloud\n  tool_choice: auto",
        );
        std::fs::write(&agent, source).unwrap();
        let args = [
            "task",
            "run",
            agent.to_str().unwrap(),
            "write report",
            "--json",
        ];
        let needs_approval = matches!(failure, "rejected" | "disconnect");
        let mut answered = false;
        let result = if needs_approval {
            provider.interact(&args, |events, _| {
                // Keep stdin open until the CLI acknowledges the pending approval.
                if !answered && events.contains("approval_required") {
                    answered = true;
                    return Some(
                        if failure == "disconnect" {
                            "yes\n"
                        } else {
                            "no\n"
                        }
                        .into(),
                    );
                }
                None
            })
        } else {
            provider.run(&args, "")
        };
        assert_eq!(answered, needs_approval, "{failure}: {result:?}");
        let events = String::from_utf8_lossy(&result.stdout);
        assert_eq!(
            path.exists(),
            failure == "disconnect",
            "{failure}: {result:?}"
        );
        assert_eq!(
            events.contains("approval_required"),
            needs_approval,
            "{failure}: {result:?}"
        );
        if failure == "disconnect" {
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "approved content");
            assert!(!result.status.success(), "{result:?}");
        }
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            if failure == "disconnect" { 2 } else { 1 }
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn responses_task_approval_survives_process_restart_and_replays_only_the_result() {
    signed_tool_task_survives_restarts(ReplayProtocol::Responses);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn messages_task_approval_survives_process_restart_and_replays_only_the_result() {
    signed_tool_task_survives_restarts(ReplayProtocol::Messages);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn messages_compatible_task_approval_preserves_signed_replay() {
    signed_tool_task_survives_restarts(ReplayProtocol::CompatibleMessages);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn ollama_cloud_task_approval_and_tool_result_survive_process_restarts() {
    for with_wire_id in [false, true] {
        signed_tool_task_survives_restarts(ReplayProtocol::OllamaCloud { with_wire_id });
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Clone, Copy)]
enum ReplayProtocol {
    Responses,
    Messages,
    CompatibleMessages,
    OllamaCloud { with_wire_id: bool },
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn signed_tool_task_survives_restarts(protocol: ReplayProtocol) {
    let files = Fixture::new(&[]);
    let path = files.dir.join("report.txt");
    let target = path.clone();
    let provider = Fixture::serve(
        &[],
        |_| {},
        move |body, index, _| {
            if let ReplayProtocol::OllamaCloud { with_wire_id } = protocol {
                assert_eq!(body["model"], "fixture");
                assert_eq!(body["stream"], false);
                assert!(body.get("options").is_none());
                assert_eq!(body["tools"][0]["function"]["name"], "file_write");
                let mut call = json!({"type":"function","function":{
                    "index":0,"name":"file_write","arguments":{"path":target,"content":"verified"}
                }});
                if with_wire_id {
                    call["id"] = json!("cloud_wire_id");
                }
                let first = json!({"role":"assistant","content":"","thinking":"fixture-private-thinking","tool_calls":[call]});
                let message = if index == 0 {
                    assert!(!target.exists());
                    first
                } else {
                    assert!(index <= 2);
                    let turns = body["messages"].as_array().unwrap();
                    let at = turns
                        .iter()
                        .position(|message| message["tool_calls"].is_array())
                        .unwrap();
                    assert_eq!(turns[at], first);
                    let result = &turns[at + 1];
                    assert_eq!(result["role"], "tool");
                    assert_eq!(result["tool_name"], "file_write");
                    assert!(
                        result["content"]
                            .as_str()
                            .is_some_and(|content| !content.is_empty())
                    );
                    if with_wire_id {
                        assert_eq!(result["tool_call_id"], "cloud_wire_id");
                    } else {
                        assert!(result.get("tool_call_id").is_none());
                    }
                    assert_eq!(std::fs::read_to_string(&target).unwrap(), "verified");
                    json!({"role":"assistant","content":if index == 1 {"draft"} else {"done"}})
                };
                return Some(("application/json", json!({
                    "done":true,"done_reason":"stop","message":message,"prompt_eval_count":12,"eval_count":4
                }).to_string()));
            }
            if !matches!(protocol, ReplayProtocol::Responses) {
                assert_eq!(body["model"], "claude-sonnet-4-6");
                assert_eq!(body["max_tokens"], 256);
                assert!(body.get("temperature").is_none());
                assert_eq!(body["thinking"]["type"], "adaptive");
                let signed = json!({"type":"thinking","thinking":"fixture-private-thinking","signature":"private-replay-fixture"});
                let redacted =
                    json!({"type":"redacted_thinking","data":"private-redacted-fixture"});
                let call = json!({
                    "type":"tool_use","id":"call_original","name":"file_write",
                    "input":{"path":target,"content":"verified"}
                });
                let (content, stop_reason) = if index == 0 {
                    assert!(!target.exists());
                    assert_eq!(body["tool_choice"]["type"], "auto");
                    assert!(
                        body["tools"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|tool| tool["name"] == "file_write")
                    );
                    (json!([signed, redacted, call]), "tool_use")
                } else {
                    assert!(index <= 2, "unexpected provider replay");
                    let turns = body["messages"].as_array().unwrap();
                    let replay = turns
                        .iter()
                        .position(|turn| {
                            turn["role"] == "assistant" && turn["content"][0] == signed
                        })
                        .unwrap();
                    assert_eq!(turns[replay]["content"], json!([signed, redacted, call]));
                    assert_eq!(turns[replay + 1]["role"], "user");
                    assert_eq!(turns[replay + 1]["content"][0]["type"], "tool_result");
                    assert_eq!(
                        turns[replay + 1]["content"][0]["tool_use_id"],
                        "call_original"
                    );
                    assert_eq!(std::fs::read_to_string(&target).unwrap(), "verified");
                    (
                        json!([{"type":"text","text":if index==1 {"draft"} else {"done"}}]),
                        "end_turn",
                    )
                };
                let reply = json!({"id":format!("msg_{index}"),"role":"assistant","stop_reason":stop_reason,
                    "content":content,"usage":{"input_tokens":12,"output_tokens":4}});
                return Some(("application/json", reply.to_string()));
            }
            assert_eq!(body["store"], false);
            assert!(body.get("max_output_tokens").is_none());
            assert_eq!(body["model"], "gpt-6.2-sol");
            let output = match index {
                0 => {
                    assert!(!target.exists());
                    assert!(
                        body["tools"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|t| t["name"] == "file_write")
                    );
                    json!([{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"private-replay-fixture"},{"type":"function_call","id":"fc_item","call_id":"call_original","name":"file_write","arguments":json!({"path":target,"content":"verified"}).to_string(),"status":"completed"}])
                }
                1 | 2 => {
                    let items = body["input"].as_array().unwrap();
                    let start = items.iter().position(|x| x["id"] == "rs_1").unwrap();
                    assert_eq!(items[start + 1]["id"], "fc_item");
                    assert_eq!(items[start + 2]["type"], "function_call_output");
                    assert_eq!(items[start + 2]["call_id"], "call_original");
                    assert_eq!(std::fs::read_to_string(&target).unwrap(), "verified");
                    json!([{"type":"message","id":format!("msg_{index}"),"role":"assistant","content":[{"type":"output_text","text":if index==1 {"draft"} else {"done"}}]}])
                }
                _ => panic!("unexpected provider replay"),
            };
            Some(("application/json",json!({"id":format!("resp_{index}"),"status":"completed","output":output,"usage":{"input_tokens":12,"output_tokens":4,"total_tokens":16}}).to_string()))
        },
    );
    let db = provider.dir.join("responses.sqlite");
    let task=TASK.replace("tools: []","tools: [file_write]").replace("completion: {response_contains: done}","completion: {all: [{response_contains: done}, {tool_called: {id: file_write, count: 1, executed: true, success: true}}]}");
    let agent=provider.agent(&format!("{task}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n  max_wall_time_seconds: 600\n  restoration_timeout_ms: 5000\n  persistence: {{enabled: true}}\n  hitl: {{on_approval_required: pause_run}}\nstorage: {{type: sqlite, path: '{}'}}\nhitl: {{default_timeout_seconds: 60, on_timeout: reject}}\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_write:\n      write_paths: ['{}']\n      overwrite_existing: false\n",db.display(), files.dir.display()));
    let (preset, model) = match protocol {
        ReplayProtocol::OllamaCloud { .. } => {
            ("provider: ollama-cloud\n  tool_choice: auto", "fixture")
        }
        ReplayProtocol::Messages => (
            "provider: anthropic\n  protocol: messages\n  tool_choice: auto\n  max_tokens: 256\n  provider_options: {thinking: {type: adaptive}}",
            "claude-sonnet-4-6",
        ),
        ReplayProtocol::CompatibleMessages => (
            "provider: anthropic-compatible\n  function_calling: true\n  tool_choice: auto\n  max_tokens: 256\n  provider_options: {thinking: {type: adaptive}}",
            "claude-sonnet-4-6",
        ),
        ReplayProtocol::Responses => (
            "provider: openai\n  protocol: responses\n  tool_choice: auto",
            "gpt-6.2-sol",
        ),
    };
    let source = std::fs::read_to_string(&agent)
        .unwrap()
        .replace("provider: openai-compatible", preset)
        .replace("model: fixture", &format!("model: {model}"));
    std::fs::write(&agent, source).unwrap();
    let agent_path = agent.to_str().unwrap();
    let seed = provider.run(
        &[
            "task",
            "run",
            agent_path,
            "write report",
            "--non-interactive",
            "--json",
        ],
        "",
    );
    if !assert_configured_seed(&provider, &seed, 3, &agent, &db) {
        return;
    }
    let saved: Value = serde_json::from_str(
        String::from_utf8_lossy(&seed.stdout)
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(saved["status"], "paused");
    assert_eq!(saved["counters"]["llm_attempts"], 1);
    assert!(!path.exists());
    assert!(!String::from_utf8_lossy(&seed.stdout).contains("private-replay-fixture"));
    let id = saved["run_id"].as_str().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let read_usage = || {
        runtime.block_on(async {
            use ai_agents::persistence::{AgentStorage, SqliteStorage};
            let store = SqliteStorage::new(db.to_str().unwrap()).await.unwrap();
            let key = ai_agents::autonomy::TaskRunKey {
                agent_id: "Fixture".into(),
                run_id: id.to_owned(),
            };
            let snapshot = store.load_task_run(&key).await.unwrap().unwrap();
            snapshot.payload["reservations"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|attempt| {
                    attempt
                        .pointer("/result/response/usage")
                        .map(|usage| usage["total_tokens"].as_u64().unwrap())
                })
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(read_usage(), vec![16]);
    let revision = saved["revision"].as_u64().unwrap().to_string();
    let stale = (saved["revision"].as_u64().unwrap() - 1).to_string();
    let rejected = provider.run(
        &[
            "task",
            "resume",
            agent_path,
            id,
            "--revision",
            &stale,
            "--json",
        ],
        "yes\nyes\n",
    );
    assert_exit(&rejected, 2);
    assert!(!path.exists());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let resume = |rev: &str, input: &str| {
        provider.run(
            &[
                "task",
                "resume",
                agent_path,
                id,
                "--revision",
                rev,
                "--json",
            ],
            input,
        )
    };
    let between = resume(&revision, "yes\nyes\n");
    assert_exit(&between, 3);
    let last_result = |out: &Output| {
        serde_json::from_str::<Value>(String::from_utf8_lossy(&out.stdout).lines().last().unwrap())
            .unwrap()
    };
    let paused = last_result(&between);
    assert_eq!(paused["status"], "paused");
    assert_eq!(paused["counters"]["llm_attempts"], 2);
    assert_eq!(paused["counters"]["tool_attempts"], 1);
    assert_eq!(read_usage(), vec![16, 16]);
    let expiry = |out: &Output| {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .find(|v| v["kind"] == "task_paused")
            .unwrap()["data"]["clocks"]["expires_at"]
            .clone()
    };
    assert!(expiry(&seed).is_string());
    assert_eq!(expiry(&seed), expiry(&between));
    let result = resume(&paused["revision"].to_string(), "yes\ncontinue\n");
    assert_exit(&result, 0);
    let done: Value = serde_json::from_str(
        String::from_utf8_lossy(&result.stdout)
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(done["run_id"], id);
    assert_eq!(done["status"], "completed");
    assert_eq!(done["counters"]["llm_attempts"], 3);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    assert_eq!(done["counters"]["tool_attempts"], 1);
    for output in [&seed, &between, &result] {
        for private in [
            "private-replay-fixture",
            "fixture-private-thinking",
            "private-redacted-fixture",
        ] {
            assert!(!String::from_utf8_lossy(&output.stdout).contains(private));
            assert!(!String::from_utf8_lossy(&output.stderr).contains(private));
        }
    }
}
#[test]
fn responses_task_failure_respects_tool_effects_and_never_replays() {
    signed_tool_task_failure_respects_effects(false);
}

#[test]
fn messages_task_failure_respects_tool_effects_and_never_replays() {
    signed_tool_task_failure_respects_effects(true);
}

fn signed_tool_task_failure_respects_effects(messages: bool) {
    for failure in ["malformed", "rejected", "disconnect"] {
        let files = Fixture::new(&[]);
        let path = files.dir.join("result.txt");
        let target = path.clone();
        let provider = Fixture::serve(
            &[],
            |_| {},
            move |_, index, _| {
                if index > 0 && failure == "disconnect" {
                    assert_eq!(
                        std::fs::read_to_string(&target).unwrap(),
                        "approved content"
                    );
                    return None; // Drop the HTTP connection after the tool effect.
                }
                if messages {
                    let (content, stop_reason) = if index == 0 {
                        let input = if failure == "malformed" {
                            json!("{")
                        } else {
                            json!({"path":target,"content":"approved content"})
                        };
                        (
                            json!([{"type":"tool_use","id":"call","name":"file_write","input":input}]),
                            "tool_use",
                        )
                    } else {
                        (json!([{"type":"text","text":"done"}]), "end_turn")
                    };
                    let response = json!({"id":format!("m{index}"),"role":"assistant","content":content,"stop_reason":stop_reason});
                    return Some(("application/json", response.to_string()));
                }
                let output = if index == 0 {
                    let arguments = if failure == "malformed" {
                        "{".to_owned()
                    } else {
                        json!({"path":target,"content":"approved content"}).to_string()
                    };
                    json!([{"type":"function_call","id":"item","call_id":"call","name":"file_write","arguments":arguments}])
                } else {
                    json!([{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"done"}]}])
                };
                Some((
                    "application/json",
                    json!({"id":format!("r{index}"),"status":"completed","output":output})
                        .to_string(),
                ))
            },
        );
        let task = TASK.replace("tools: []", "tools: [file_write]");
        let agent=provider.agent(&format!("{task}  hitl: {{on_approval_required: pause_run}}\nhitl: {{default_timeout_seconds: 60, on_timeout: reject}}\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_write:\n      write_paths: ['{}']\n",files.dir.display()));
        let (preset, model) = if messages {
            (
                "provider: anthropic\n  protocol: messages\n  tool_choice: auto\n  max_tokens: 256",
                "claude-sonnet-4-6",
            )
        } else {
            (
                "provider: openai\n  protocol: responses\n  tool_choice: auto",
                "gpt-6.2-astra",
            )
        };
        let source = std::fs::read_to_string(&agent)
            .unwrap()
            .replace("provider: openai-compatible", preset)
            .replace("model: fixture", &format!("model: {model}"));
        let db = provider.dir.join("journal.sqlite");
        std::fs::write(
            &agent,
            format!(
                "{source}\nstorage: {{type: sqlite, path: '{}'}}\n",
                db.display()
            ),
        )
        .unwrap();
        let result = provider.run(
            &[
                "task",
                "run",
                agent.to_str().unwrap(),
                "write report",
                "--json",
            ],
            if failure == "disconnect" {
                "yes\n"
            } else {
                "no\n"
            },
        );
        assert_eq!(path.exists(), failure == "disconnect");
        let events = String::from_utf8_lossy(&result.stdout);
        if failure == "disconnect" {
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            let listed = provider.run(&["task", "list", agent.to_str().unwrap(), "--json"], "");
            assert_exit(&listed, 0);
            let runs: Value = serde_json::from_slice(&listed.stdout).unwrap();
            let id = runs[0]["key"]["run_id"].as_str().unwrap();
            let show = provider.run(&["task", "show", agent.to_str().unwrap(), id, "--json"], "");
            assert_exit(&show, 0);
            let view: Value = serde_json::from_slice(&show.stdout).unwrap();
            assert_eq!(view["counters"]["tool_attempts"], 1);
            assert_eq!(view["counters"]["llm_attempts"], 2);
            assert_ne!(view["summary"]["status"], "completed");
        } else if failure == "malformed" {
            assert!(!events.contains("approval_required"));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(events.contains("approval_required"), "{events}");
        }
    }
}

#[test]
fn task_subprocess_report_requires_real_file_effect_and_policy_denial_cannot_be_approved() {
    for deny in [false, true] {
        let f = Fixture::new(&[]);
        let path = f.dir.join("report.txt");
        // This fixture selects an actual isolated write, not a model claim that an artifact exists.
        let call =
            json!({"tool":"file_write","arguments":{"path":path,"content":"validated report"}})
                .to_string();
        let provider = Fixture::new(&[&call, "done"]);
        let config=TASK.replace("tools: []","tools: [file_write]").replace("completion: {response_contains: done}","completion: {all: [{response_contains: done}, {tool_called: {id: file_write, count: 1, executed: true, success: true}}]}");
        let denied = if deny {
            format!("      blocked_paths: ['{}']\n", path.display())
        } else {
            String::new()
        };
        let agent=provider.agent(&format!("{config}  hitl: {{on_approval_required: pause_run}}\nhitl: {{default_timeout_seconds: 60, on_timeout: reject}}\ntool_security:\n  enabled: true\n  fail_closed: true\n  tools:\n    file_write:\n      write_paths: ['{}']\n      overwrite_existing: false\n{denied}",f.dir.display()));
        let out = provider.run(&["task", "run", agent.to_str().unwrap(), "report"], "yes\n");
        assert_exit(&out, if deny { 1 } else { 0 });
        if deny {
            assert!(!path.exists());
        } else {
            assert_eq!(std::fs::read_to_string(path).unwrap(), "validated report");
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        }
    }
}

#[test]
fn task_subprocess_limit_revision_never_refills_consumed_calls() {
    let f = Fixture::new(&["draft", "done"]);
    let config = TASK
        .replace("max_turns: 4", "max_turns: 1")
        .replace("max_llm_calls: 4", "max_llm_calls: 1");
    let agent = f.agent(&format!(
        "{config}  on_limit: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    let mut step = 0;
    let out = f.interact(
        &["task", "run", agent.to_str().unwrap(), "analysis"],
        |text, _| {
            if step == 0 {
                if let Some((id, _, _)) = pending_header(text) {
                    step = 1;
                    return Some(format!("/task show {id}\n"));
                }
            } else if step == 1 {
                if let Some(view) = text
                    .lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .find(|v| v.get("limits").is_some())
                {
                    let id = view["summary"]["key"]["run_id"].as_str().unwrap();
                    let rev = view["summary"]["revision"].as_u64().unwrap();
                    let mut limits = view["limits"].clone();
                    limits["max_turns"] = json!(4);
                    limits["max_llm_calls"] = json!(4);
                    step = 2;
                    return Some(format!("/task limits {id} {rev} {limits}\n"));
                }
            } else if step == 2 && text.contains("HostControlApplied") {
                step = 3;
                return Some("continue work\n".into());
            }
            None
        },
    );
    assert_exit(&out, 0);
    assert_eq!(step, 3);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn task_subprocess_json_exports_remain_redacted_while_local_questions_are_usable() {
    let f = Fixture::new(&[
        r#"{"tool":"ask_user","arguments":{"question":"fixture-private-question","options":["private-choice"]}}"#,
        "done",
    ]);
    let config = TASK.replace("tools: []", "tools: [ask_user]");
    let agent = f.agent(&format!(
        "{config}  hitl: {{on_user_question: pause_run}}\n"
    ));
    let out = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "private-objective",
            "--json",
        ],
        "1\n",
    );
    assert_exit(&out, 0);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(!text.contains("fixture-private-question"));
    assert!(!text.contains("private-choice"));
    assert!(!text.contains("private-objective"));
    for line in text.lines() {
        serde_json::from_str::<Value>(line).unwrap();
    }
    assert!(String::from_utf8_lossy(&out.stderr).contains("fixture-private-question"));
}

#[test]
fn task_subprocess_slow_output_preserves_the_complete_authoritative_suffix() {
    let mut replies = vec!["draft"; 40];
    replies.push("done");
    let mut f = Fixture::new(&replies);
    f.slow_output = true;
    let config = TASK
        .replace("max_turns: 4", "max_turns: 60")
        .replace("max_llm_calls: 4", "max_llm_calls: 60");
    let agent = f.agent(&format!(
        "{config}  premature_finish: {{action: continue, max_continuations: 60}}\n"
    ));
    let out = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "analysis",
            "--json",
            "--non-interactive",
        ],
        "",
    );
    assert_exit(&out, 0);
    let rows: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(rows.iter().any(|v| v["kind"] == "task_completed"));
    assert_eq!(rows.last().unwrap()["status"], "completed");
    assert_eq!(rows.last().unwrap()["counters"]["llm_attempts"], 41);
    assert_eq!(f.calls.load(Ordering::SeqCst), 41);
}

#[test]
fn task_subprocess_broken_pipe_reports_io_failure_without_panicking_or_replaying() {
    let f = Fixture::new(&["done"]);
    let agent = f.agent(TASK);
    let mut child = Command::new(env!("CARGO_BIN_EXE_ai-agents-cli"))
        .args([
            "task",
            "run",
            agent.to_str().unwrap(),
            "analysis",
            "--json",
            "--non-interactive",
        ])
        .env("TASK_FIXTURE_TOKEN", "isolated-test-placeholder")
        .env("RUST_LOG", "off")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            panic!("broken output prevented foreground termination");
        }
        thread::sleep(Duration::from_millis(2));
    };
    let mut error = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut error)
        .unwrap();
    assert_eq!(status.code(), Some(2), "{error}");
    assert!(!error.contains("panicked"));
    assert!(f.calls.load(Ordering::SeqCst) <= 1);
}

#[test]
fn repl_subprocess_ordinary_recoverable_error_keeps_context_commands_usable() {
    let f = Fixture::new(&["ordinary answer"]);
    let agent = f.agent("tools: []\ncontext:\n  user: {type: runtime, required: true}\n");
    let out = f.run(
        &["run", agent.to_str().unwrap(), "--plain"],
        "hello\n/context set user Alice\nhello again\n",
    );
    assert_exit(&out, 0);
    assert!(String::from_utf8_lossy(&out.stderr).contains("[Error]"));
    assert!(String::from_utf8_lossy(&out.stdout).contains("ordinary answer"));
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn task_subprocess_input_syntax_errors_do_not_stop_or_rebind_the_paused_run() {
    let f = Fixture::new(&["draft", "done"]);
    let agent = f.agent(&format!(
        "{TASK}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    let mut step = 0;
    let out = f.interact(
        &["task", "run", agent.to_str().unwrap(), "research"],
        |text, _| {
            let errors =
                text.matches("[input error]").count() + text.matches("[control rejected]").count();
            if let Some((id, rev, request)) = pending_header(text) {
                if step == 0 {
                    step = 1;
                    return Some(format!("/task resume {id} {rev} {request}\n"));
                }
                if step == 1 && errors >= 1 {
                    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
                    step = 2;
                    return Some(format!("/task mystery {id} {rev}\n"));
                }
                if step == 2 && errors >= 2 {
                    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
                    step = 3;
                    return Some(format!("/task limits {id} {rev} not-json\n"));
                }
                if step == 3 && errors >= 3 {
                    step = 4;
                    return Some(format!("/task resume {id} {rev} {request} continue\n"));
                }
            }
            None
        },
    );
    assert_exit(&out, 0);
    assert_eq!(step, 4);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn repl_subprocess_lost_output_rejects_an_outstanding_ordinary_callback() {
    let f = Fixture::new(&[
        r#"{"tool":"ask_user","arguments":{"question":"ordinary preference"}}"#,
        "done",
    ]);
    f.gate.store(true, Ordering::Release);
    let agent = f.agent("tools: [ask_user]");
    let mut child = Command::new(env!("CARGO_BIN_EXE_ai-agents-cli"))
        .args(["run", agent.to_str().unwrap(), "--plain"])
        .env("TASK_FIXTURE_TOKEN", "isolated-test-placeholder")
        .env("RUST_LOG", "off")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input.write_all(b"research\n").unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut initial = String::new();
    while !initial.contains("You > ") {
        let mut bytes = [0; 256];
        let n = stdout.read(&mut bytes).unwrap();
        assert!(n > 0);
        initial.push_str(&String::from_utf8_lossy(&bytes[..n]));
    }
    drop(stdout);
    f.gate.store(false, Ordering::Release);
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("ordinary callback survived confirmed output loss");
        }
        thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(status.code(), Some(1));
    assert!(f.calls.load(Ordering::SeqCst) <= 2);
    drop(input);
}

#[test]
fn task_subprocess_explicit_invalid_approval_is_a_response_error_not_a_write_failure() {
    let f = Fixture::new(&["done"]);
    let agent=f.agent(&format!("{TASK}  lifecycle: [{{id: research, instruction: Collect findings}}]\n  supervision: {{stage_changes: require_approval}}\n"));
    let mut step = 0;
    let out = f.interact(
        &["task", "run", agent.to_str().unwrap(), "research"],
        |text, _| {
            if let Some((id, rev, request)) = pending_header(text) {
                if step == 0 {
                    step = 1;
                    return Some(format!("/task resume {id} {rev} {request} maybe\n"));
                }
                if step == 1 && text.contains("control rejected") {
                    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
                    step = 2;
                    return Some(format!("/task resume {id} {rev} {request} yes\n"));
                }
            }
            None
        },
    );
    assert_exit(&out, 0);
    assert_eq!(step, 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn task_subprocess_no_storage_completes_and_emits_safe_json() {
    let f = Fixture::new(&["done"]);
    let agent = f.agent(TASK);
    let out = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "한국어 분석",
            "--json",
            "--non-interactive",
        ],
        "",
    );
    assert_exit(&out, 0);
    let rows: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.last().unwrap()["status"], "completed");
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn task_subprocess_unavailable_question_stops_without_replay() {
    let f = Fixture::new(&["draft", "done"]);
    let agent = f.agent(&format!(
        "{TASK}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    let out = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "research",
            "--non-interactive",
        ],
        "",
    );
    assert_exit(&out, 1);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn task_subprocess_retained_question_answers_same_run() {
    let f = Fixture::new(&["draft", "done"]);
    let agent = f.agent(&format!(
        "{TASK}  premature_finish: {{action: ask_user, on_unavailable: stop}}\n"
    ));
    let out = f.run(
        &["task", "run", agent.to_str().unwrap(), "research"],
        "continue the research\n",
    );
    assert_exit(&out, 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    assert!(String::from_utf8_lossy(&out.stdout).contains("Question:"));
}

#[test]
fn task_subprocess_usage_and_unsupported_resume_do_not_call_provider() {
    let f = Fixture::new(&[]);
    let agent = f.agent(TASK);
    for args in [
        vec!["task", "resume", agent.to_str().unwrap(), "absent"],
        vec!["task", "cancel", agent.to_str().unwrap(), "absent"],
        vec![
            "task",
            "run",
            agent.to_str().unwrap(),
            "objective",
            "--profile",
            "missing",
        ],
    ] {
        let out = f.run(&args, "");
        assert_exit(&out, 2);
    }
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn task_subprocess_sqlite_inspection_is_provider_free_and_scoped() {
    let f = Fixture::new(&["done"]);
    let db = f.dir.join("tasks.sqlite");
    let agent = f.agent(&format!(
        "{TASK}storage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let run = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "report",
            "--json",
            "--non-interactive",
            "--actor",
            "alice",
        ],
        "",
    );
    assert_exit(&run, 0);
    let result: Value =
        serde_json::from_str(String::from_utf8_lossy(&run.stdout).lines().last().unwrap()).unwrap();
    let id = result["run_id"].as_str().unwrap();
    let list = f.run(
        &[
            "task",
            "list",
            agent.to_str().unwrap(),
            "--json",
            "--actor",
            "alice",
        ],
        "",
    );
    assert_exit(&list, 0);
    let rows: Vec<Value> = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(rows.len(), 1);
    let show = f.run(
        &[
            "task",
            "show",
            agent.to_str().unwrap(),
            id,
            "--actor",
            "alice",
        ],
        "",
    );
    assert_exit(&show, 0);
    let wrong = f.run(
        &[
            "task",
            "show",
            agent.to_str().unwrap(),
            id,
            "--actor",
            "bob",
        ],
        "",
    );
    assert_exit(&wrong, 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn repl_subprocess_eof_and_ordinary_chat_are_not_a_busy_loop() {
    let f = Fixture::new(&["ordinary answer"]);
    let agent = f.agent("tools: []");
    let out = f.run(&["run", agent.to_str().unwrap(), "--plain"], "hello\n");
    assert_exit(&out, 0);
    assert!(String::from_utf8_lossy(&out.stdout).contains("ordinary answer"));
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn task_overrides_preflight_rejects_before_provider_and_storage() {
    let f = Fixture::new(&[]);
    let db = f.dir.join("override-must-not-create.sqlite");
    let agent = f.agent(&format!(
        "{TASK}\nstorage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let path = f.dir.join("override.yaml");
    for yaml in [
        "persistence: {enabled: true}",
        "objective: one\nobjective_template: two",
        "max_llm_calls: 0",
        "completion: {check: nonexistent}",
        "unknown: true",
    ] {
        std::fs::write(&path, yaml).unwrap();
        let output = f.run(
            &[
                "task",
                "run",
                agent.to_str().unwrap(),
                "work",
                "--overrides",
                path.to_str().unwrap(),
                "--non-interactive",
            ],
            "",
        );
        assert_exit(&output, 2);
        assert!(!db.exists(), "{yaml}");
        assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn task_overrides_use_one_namespace_and_management_needs_no_overlay_file() {
    let f = Fixture::new(&["accepted-one", "accepted-two", "done"]);
    let db = f.dir.join("override-journal.sqlite");
    let agent = f.agent(&format!(
        "{TASK}\nstorage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let path = f.dir.join("override.yaml");
    for answer in ["accepted-one", "accepted-two"] {
        std::fs::write(
            &path,
            format!("completion: {{response_contains: {answer}}}\nmax_llm_calls: 1\n"),
        )
        .unwrap();
        let output = f.run(
            &[
                "task",
                "run",
                agent.to_str().unwrap(),
                "work",
                "--overrides",
                path.to_str().unwrap(),
                "--non-interactive",
                "--json",
            ],
            "",
        );
        assert_exit(&output, 0);
    }
    std::fs::remove_file(path).unwrap();
    assert_exit(
        &f.run(
            &[
                "task",
                "run",
                agent.to_str().unwrap(),
                "work",
                "--non-interactive",
            ],
            "",
        ),
        0,
    );
    let list = f.run(&["task", "list", agent.to_str().unwrap(), "--json"], "");
    assert_exit(&list, 0);
    let rows: Value = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 3);
    for row in rows.as_array().unwrap() {
        let id = row["key"]["run_id"].as_str().unwrap();
        assert_exit(
            &f.run(&["task", "show", agent.to_str().unwrap(), id, "--json"], ""),
            0,
        );
    }
    assert_eq!(f.calls.load(Ordering::SeqCst), 3);
}

#[test]
fn task_overrides_preflight_precedes_provider_credentials() {
    let f = Fixture::new(&[]);
    let db = f.dir.join("no-preparation.sqlite");
    let agent = f.agent(&format!(
        "{TASK}\nstorage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let yaml = std::fs::read_to_string(&agent)
        .unwrap()
        .replace("provider: openai-compatible", "provider: openai")
        .replace("TASK_FIXTURE_TOKEN", "TASK_OVERRIDE_MISSING_KEY");
    std::fs::write(&agent, yaml).unwrap();
    let path = f.dir.join("override.yaml");
    std::fs::write(&path, "persistence: {enabled: true}").unwrap();
    let command = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_ai-agents-cli"));
        cmd.args([
            "task",
            "run",
            agent.to_str().unwrap(),
            "work",
            "--non-interactive",
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("TASK_OVERRIDE_MISSING_KEY");
        cmd
    };
    let rejected = command()
        .args(["--overrides", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_exit(&rejected, 2);
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("durable task execution requires"));
    let normal = command().output().unwrap();
    assert_exit(&normal, 2);
    assert!(String::from_utf8_lossy(&normal.stderr).contains("API key not found"));
    assert!(!db.exists());
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
}

// Startup selects the saved task before ordinary input or auto-spawn construction, for explicit and automatic-state entry alike.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn task_subprocess_configured_startup_resumes_same_task_before_new_chat() {
    for automatic in [false, true] {
        let f = Fixture::new(&["draft", "done"]);
        let db = f.dir.join("startup.sqlite");
        let extra = if automatic {
            format!(
                "tools: []\nautonomy:\n  restoration_timeout_ms: 5000\n  persistence: {{enabled: true, resume_on_start: true}}\nstates:\n  initial: work\n  states:\n    work:\n      autonomy:\n        enabled: true\n        mode: state_until_exit\n        completion: {{response_contains: done}}\n        premature_finish: {{action: ask_user, on_unavailable: stop}}\nstorage: {{type: sqlite, path: '{}' }}\n",
                db.display()
            )
        } else {
            format!(
                "{TASK}  restoration_timeout_ms: 5000\n  persistence: {{enabled: true, resume_on_start: true}}\n  premature_finish: {{action: ask_user, on_unavailable: stop}}\nstorage: {{type: sqlite, path: '{}' }}\n",
                db.display()
            )
        };
        let agent = f.agent(&extra);
        let first = if automatic {
            f.run(
                &["run", agent.to_str().unwrap(), "--plain"],
                "finish the report\n",
            )
        } else {
            f.run(
                &[
                    "task",
                    "run",
                    agent.to_str().unwrap(),
                    "finish the report",
                    "--non-interactive",
                ],
                "",
            )
        };
        if !assert_configured_seed(&f, &first, if automatic { 0 } else { 3 }, &agent, &db) {
            continue;
        }
        assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        let list = f.run(&["task", "list", agent.to_str().unwrap(), "--json"], "");
        assert_exit(&list, 0);
        let rows: Value = serde_json::from_slice(&list.stdout).unwrap();
        assert_eq!(rows.as_array().unwrap().len(), 1);
        assert_eq!(rows[0]["status"], "paused");
        let id = rows[0]["key"]["run_id"].as_str().unwrap();
        let startup = f.run(
            &["run", agent.to_str().unwrap(), "--plain"],
            "yes\ncontinue\n",
        );
        assert_exit(&startup, 0);
        assert!(
            String::from_utf8_lossy(&startup.stdout).contains("Restoration review"),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&startup.stdout),
            String::from_utf8_lossy(&startup.stderr)
        );
        assert_eq!(f.calls.load(Ordering::SeqCst), 2);
        let show = f.run(&["task", "show", agent.to_str().unwrap(), id, "--json"], "");
        assert_exit(&show, 0);
        let done: Value = serde_json::from_slice(&show.stdout).unwrap();
        assert_eq!(done["summary"]["status"], "completed");
        assert_eq!(done["counters"]["llm_attempts"], 2);
    }
}

// The real TUI services restoration review and the original question, then restores the same live terminal on exit.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn tui_pty_configured_startup_restores_task_and_terminal() {
    let question = r#"{"tool":"ask_user","arguments":{"question":"Confirm saved source","options":["yes"],"allow_other":false}}"#;
    let f = Fixture::new(&[question, "done"]);
    let db = f.dir.join("tui-restart.sqlite");
    let task = TASK.replace("tools: []", "tools: [ask_user]");
    let agent = f.agent(&format!(
        "{task}  restoration_timeout_ms: 5000\n  persistence: {{enabled: true, resume_on_start: true}}\n  hitl: {{on_user_question: pause_run}}\nstorage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let seed = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "confirm source",
            "--non-interactive",
        ],
        "",
    );
    if !assert_configured_seed(&f, &seed, 3, &agent, &db) {
        return;
    }
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    run_tui_pty(&agent, "", "Agent: done", "yes", "complete");
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    let list = f.run(&["task", "list", agent.to_str().unwrap(), "--json"], "");
    assert_exit(&list, 0);
    let rows: Value = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 1);
    assert_eq!(rows[0]["status"], "completed");
}

// A selected skill keeps its activation and cursor across processes instead of repeating routing or its completed prefix.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn task_subprocess_configured_skill_startup_preserves_activation() {
    let f = Fixture::new(&["research", "draft", "done"]);
    let db = f.dir.join("skill-restart.sqlite");
    let skill = f.dir.join("research.skill.yaml");
    let skill_source = "id: research\ndescription: Research the request\ntrigger: User requests research\nautonomy:\n  enabled: true\n  mode: skill_until_complete\n  completion: {response_contains: done}\n  premature_finish: {action: ask_user, on_unavailable: stop}\nsteps:\n  - prompt: 'Research {{ user_input }}'\n";
    std::fs::write(&skill, skill_source).unwrap();
    let extra = format!(
        "tools: []\nautonomy:\n  restoration_timeout_ms: 5000\n  persistence: {{enabled: true, resume_on_start: true}}\nskills: [{{file: research.skill.yaml}}]\nstorage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    );
    let agent = f.agent(&extra);
    let yaml = std::fs::read_to_string(&agent).unwrap();
    let prefix = yaml.split("tools: []").next().unwrap();
    let provider = prefix.split("llm:\n").nth(1).unwrap();
    let provider = provider
        .lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&agent, format!("name: Fixture\nsystem_prompt: Research\nllm: {{default: default, router: router}}\nllms:\n  default:\n{provider}\n  router:\n{provider}\n{extra}")).unwrap();
    let seed = f.run(
        &["run", agent.to_str().unwrap(), "--plain"],
        "Research the sources\n",
    );
    if !assert_configured_seed(&f, &seed, 0, &agent, &db) {
        return;
    }
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    let list = f.run(&["task", "list", agent.to_str().unwrap(), "--json"], "");
    assert_exit(&list, 0);
    let rows: Value = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 1);
    assert_eq!(rows[0]["status"], "paused");
    let id = rows[0]["key"]["run_id"].as_str().unwrap();
    // The loaded definition belongs to the construction identity even though the root YAML did not change.
    std::fs::write(
        &skill,
        skill_source.replace("Research {{ user_input }}", "Changed instruction"),
    )
    .unwrap();
    let revision = rows[0]["revision"].as_u64().unwrap().to_string();
    let changed = f.run(
        &[
            "task",
            "resume",
            agent.to_str().unwrap(),
            id,
            "--revision",
            &revision,
        ],
        "yes\ncontinue\n",
    );
    assert!(!changed.status.success());
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
    std::fs::write(&skill, skill_source).unwrap();
    let resumed = f.run(
        &["run", agent.to_str().unwrap(), "--plain"],
        "yes\ncontinue\n",
    );
    assert_exit(&resumed, 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 3);
    let show = f.run(&["task", "show", agent.to_str().unwrap(), id, "--json"], "");
    assert_exit(&show, 0);
    let done: Value = serde_json::from_slice(&show.stdout).unwrap();
    assert_eq!(done["summary"]["status"], "completed");
    // Initial routing precedes task activation, so only the two task-owned calls belong to this ledger.
    assert_eq!(done["counters"]["llm_attempts"], 2);
}

// A coordinating task restores fresh declared child objects and original shared accounting instead of re-sending its completed prefix.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn task_subprocess_configured_child_restart_preserves_topology_and_exact_call_limit() {
    let question = r#"{"tool":"ask_user","arguments":{"question":"Confirm child result","options":["yes"],"allow_other":false}}"#;
    let child = Fixture::new(&[question, "done-child"]);
    let child_yaml = child.agent("tools: [ask_user]\n");
    let text = std::fs::read_to_string(&child_yaml).unwrap();
    std::fs::write(
        &child_yaml,
        text.replacen("name: Fixture", "name: Worker", 1),
    )
    .unwrap();
    let message = r#"{"tool":"send_agent_message","arguments":{"to":"worker","message":"verify the child result"}}"#;
    let root = Fixture::new(&[message, "done"]);
    let db = root.dir.join("children.sqlite");
    let agent = root.agent(&format!(
        "{TASK}  restoration_timeout_ms: 5000\n  persistence: {{enabled: true}}\n  hitl: {{on_user_question: pause_run, on_approval_required: pause_run}}\nspawner:\n  shared_llms: false\n  management_tools: [send_agent_message]\n  auto_spawn: [{{id: worker, agent: '{}' }}]\nstorage: {{type: sqlite, path: '{}' }}\n", child_yaml.display(), db.display()
    ));
    let first = root.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "verify",
            "--non-interactive",
            "--json",
        ],
        "",
    );
    if !assert_configured_seed(&root, &first, 3, &agent, &db) {
        assert_eq!(child.calls.load(Ordering::SeqCst), 0);
        return;
    }
    let saved: Value = serde_json::from_str(
        String::from_utf8_lossy(&first.stdout)
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(saved["status"], "paused");
    assert_eq!(root.calls.load(Ordering::SeqCst), 1);
    assert_eq!(child.calls.load(Ordering::SeqCst), 1);
    let revision = saved["revision"].as_u64().unwrap().to_string();
    let resumed = root.run(
        &[
            "task",
            "resume",
            agent.to_str().unwrap(),
            saved["run_id"].as_str().unwrap(),
            "--revision",
            &revision,
            "--json",
        ],
        "yes\nyes\n",
    );
    assert_exit(&resumed, 0);
    let done: Value = serde_json::from_str(
        String::from_utf8_lossy(&resumed.stdout)
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(done["status"], "completed");
    assert_eq!(done["run_id"], saved["run_id"]);
    assert_eq!(done["counters"]["llm_attempts"], 4);
    assert_eq!(root.calls.load(Ordering::SeqCst), 2);
    assert_eq!(child.calls.load(Ordering::SeqCst), 2);
}

// A real configuration read blocks only after the restore work journal is committed; killing the process tests actual CLI recovery without a product fault switch.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn task_subprocess_configured_constructor_crash_recovers_without_replay() {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::process::ExitStatusExt;
    struct ChildGuard(Option<std::process::Child>);
    impl Drop for ChildGuard {
        // A failing assertion must not leave a blocked fixture process or its lease behind.
        fn drop(&mut self) {
            if let Some(child) = self.0.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let f = Fixture::new(&["draft", "done"]);
    let db = f.dir.join("crash.sqlite");
    let pricing = f.dir.join("pricing.json");
    std::fs::write(&pricing, "{}").unwrap();
    let agent = f.agent(&format!(
        "{TASK}  restoration_timeout_ms: 5000\n  persistence: {{enabled: true}}\n  premature_finish: {{action: ask_user, on_unavailable: stop}}\nstorage: {{type: sqlite, path: '{}' }}\nobservability:\n  enabled: true\n  cost:\n    pricing_file: '{}'\n", db.display(), pricing.display()
    ));
    let first = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "report",
            "--non-interactive",
            "--json",
        ],
        "",
    );
    if !assert_configured_seed(&f, &first, 3, &agent, &db) {
        return;
    }
    let saved: Value = serde_json::from_str(
        String::from_utf8_lossy(&first.stdout)
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(saved["status"], "paused");
    let id = saved["run_id"].as_str().unwrap();
    let revision = saved["revision"].as_u64().unwrap().to_string();
    std::fs::remove_file(&pricing).unwrap();
    let fifo = std::ffi::CString::new(pricing.as_os_str().as_bytes()).unwrap();
    // SAFETY: fifo is a live NUL-terminated path in the isolated fixture directory; only this process owns its creation.
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    let mut child = ChildGuard(Some(
        Command::new(env!("CARGO_BIN_EXE_ai-agents-cli"))
            .args([
                "task",
                "resume",
                agent.to_str().unwrap(),
                id,
                "--revision",
                &revision,
                "--json",
            ])
            .current_dir(&f.dir)
            .env("RUST_LOG", "off")
            .env("TASK_FIXTURE_TOKEN", "isolated-test-placeholder")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));
    let started = Instant::now();
    let gate = loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&pricing)
        {
            Ok(gate) => break gate,
            Err(error) if error.raw_os_error() == Some(libc::ENXIO) => {}
            Err(error) => panic!("pricing gate: {error}"),
        }
        if child.0.as_mut().unwrap().try_wait().unwrap().is_some() {
            let out = child.0.take().unwrap().wait_with_output().unwrap();
            panic!(
                "constructor returned before the read gate: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        assert!(
            started.elapsed() < Duration::from_secs(25),
            "constructor did not reach its real read boundary"
        );
        thread::sleep(Duration::from_millis(5));
    };
    let show = f.run(&["task", "show", agent.to_str().unwrap(), id, "--json"], "");
    assert_exit(&show, 0);
    let running: Value = serde_json::from_slice(&show.stdout).unwrap();
    assert_eq!(running["summary"]["status"], "running");
    assert_eq!(running["counters"]["llm_attempts"], 1);
    let revision = running["summary"]["revision"].as_u64().unwrap().to_string();
    let args = [
        "task",
        "recover",
        agent.to_str().unwrap(),
        id,
        "--revision",
        &revision,
        "--action",
        "resume-prepared",
        "--reason",
        "interrupted configured preparation",
        "--json",
    ];
    let held = f.run(&args, "");
    assert_exit(&held, 2);
    assert!(
        String::from_utf8_lossy(&held.stderr).contains("active owner"),
        "{}",
        String::from_utf8_lossy(&held.stderr)
    );
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    child.0.as_mut().unwrap().kill().unwrap();
    let stopped = child.0.take().unwrap().wait_with_output().unwrap();
    assert_eq!(stopped.status.signal(), Some(libc::SIGKILL));
    drop(gate);
    std::fs::remove_file(&pricing).unwrap();
    std::fs::write(&pricing, "{}").unwrap();
    let recovery = f.run(&args, "");
    assert_exit(&recovery, 0);
    let ready: Value = serde_json::from_slice(&recovery.stdout).unwrap();
    assert_eq!(ready["status"], "paused");
    let revision = ready["revision"].as_u64().unwrap().to_string();
    let done = f.run(
        &[
            "task",
            "resume",
            agent.to_str().unwrap(),
            id,
            "--revision",
            &revision,
            "--json",
        ],
        "yes\ncontinue\n",
    );
    assert_exit(&done, 0);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

// A provider-free cancellation request blocks execution; explicit configured cleanup acknowledges retirement without consuming the pending question.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn task_subprocess_configured_cancelled_pause_cleans_up_without_model_work() {
    let f = Fixture::new(&["draft", "must not execute"]);
    let db = f.dir.join("cancelled.sqlite");
    let agent = f.agent(&format!("{TASK}  restoration_timeout_ms: 5000\n  persistence: {{enabled: true}}\n  premature_finish: {{action: ask_user, on_unavailable: stop}}\nstorage: {{type: sqlite, path: '{}' }}\n", db.display()));
    let first = f.run(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "report",
            "--non-interactive",
            "--json",
        ],
        "",
    );
    if !assert_configured_seed(&f, &first, 3, &agent, &db) {
        return;
    }
    let saved: Value = serde_json::from_str(
        String::from_utf8_lossy(&first.stdout)
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    let id = saved["run_id"].as_str().unwrap();
    let revision = saved["revision"].as_u64().unwrap().to_string();
    let cancel = f.run(
        &[
            "task",
            "cancel",
            agent.to_str().unwrap(),
            id,
            "--revision",
            &revision,
            "--json",
        ],
        "",
    );
    assert_exit(&cancel, 0);
    let requested: Value = serde_json::from_slice(&cancel.stdout).unwrap();
    assert_eq!(requested["disposition"], "requested");
    let current = requested["run"]["revision"].as_u64().unwrap().to_string();
    let resume = f.run(
        &[
            "task",
            "resume",
            agent.to_str().unwrap(),
            id,
            "--revision",
            &current,
        ],
        "yes\ncontinue\n",
    );
    assert_exit(&resume, 2);
    let stale = f.run(
        &[
            "task",
            "cleanup",
            agent.to_str().unwrap(),
            id,
            "--revision",
            &revision,
        ],
        "",
    );
    assert_exit(&stale, 2);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    let cleanup = f.run(
        &[
            "task",
            "cleanup",
            agent.to_str().unwrap(),
            id,
            "--revision",
            &current,
            "--json",
        ],
        "",
    );
    assert_exit(&cleanup, 0);
    let done: Value = serde_json::from_slice(&cleanup.stdout).unwrap();
    assert_eq!(done["status"], "cancelled");
    assert_eq!(done["counters"]["llm_attempts"], 1);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}
