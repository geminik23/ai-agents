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
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    dir: PathBuf,
    url: String,
    calls: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    gate: Arc<AtomicBool>,
    slow_output: bool,
    worker: Option<thread::JoinHandle<()>>,
}
impl Fixture {
    fn new(replies: &[&str]) -> Self {
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
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let gate = Arc::new(AtomicBool::new(false));
        let held = gate.clone();
        let replies: Vec<String> = replies.iter().map(|v| v.to_string()).collect();
        let worker = thread::spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let (mut socket, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(_) => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                };
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = vec![];
                let end = loop {
                    let mut buf = [0; 4096];
                    let n = socket.read(&mut buf).unwrap();
                    if n == 0 {
                        return;
                    }
                    bytes.extend_from_slice(&buf[..n]);
                    if let Some(i) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let len = std::str::from_utf8(&bytes[..end])
                    .unwrap()
                    .lines()
                    .find_map(|line| {
                        line.split_once(':')
                            .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                            .map(|(_, v)| v.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                while bytes.len() < end + len {
                    let mut buf = [0; 4096];
                    let n = socket.read(&mut buf).unwrap();
                    if n == 0 {
                        return;
                    }
                    bytes.extend_from_slice(&buf[..n]);
                }
                let body: Value = serde_json::from_slice(&bytes[end..end + len]).unwrap();
                let index = count.fetch_add(1, Ordering::SeqCst);
                let reply = replies.get(index).cloned().unwrap_or_else(|| "done".into());
                while held.load(Ordering::Acquire) && !stopped.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(2));
                }
                let streaming = body["stream"] == true;
                let (kind, response) = if streaming {
                    (
                        "text/event-stream",
                        format!(
                            "data: {}\n\ndata: [DONE]\n\n",
                            json!({"choices":[{"delta":{"content":reply},"finish_reason":null}]})
                        ),
                    )
                } else {
                    ("application/json",json!({"id":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":reply},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}).to_string())
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
            || (args.first() == Some(&"task") && args.get(1) == Some(&"run"))
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
impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
const TASK: &str = "tools: []\nautonomy:\n  enabled: true\n  mode: until_complete\n  max_turns: 4\n  max_llm_calls: 4\n  completion: {response_contains: done}\n";
fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
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
    let f = Fixture::new(&["done"]);
    f.gate.store(true, Ordering::Release);
    let db = f.dir.join("tasks.sqlite");
    let agent = f.agent(&format!(
        "{TASK}storage: {{type: sqlite, path: '{}' }}\n",
        db.display()
    ));
    let mut requested = false;
    let out = f.interact(
        &[
            "task",
            "run",
            agent.to_str().unwrap(),
            "operations",
            "--json",
        ],
        |text, _| {
            if !requested && f.calls.load(Ordering::SeqCst) > 0 {
                let id = text
                    .lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .find_map(|v| v["run_id"].as_str().map(str::to_owned));
                if let Some(id) = id {
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
        },
    );
    assert_exit(&out, 1);
    assert!(requested);
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert!(!String::from_utf8_lossy(&out.stdout).contains("\"status\":\"completed\""));
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
