// Exercise live declarations with deterministic providers before spending any provider budget.
// The same application-owned corpus and detailed result checker are used during real evaluation.
use ai_agents::AgentBuilder;
use ai_agents_eval::fixtures::AttemptFixtureContext;
use ai_agents_eval::{EvalHost, EvalResult, EvalRunner, EvalRunnerOptions};
use async_trait::async_trait;
use example_custom_llm::{task_domain, task_live, task_live_acceptance};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
struct Host;
#[async_trait]
impl EvalHost for Host {
    fn configure(
        &self,
        b: AgentBuilder,
        _: &AttemptFixtureContext,
    ) -> ai_agents_eval::Result<AgentBuilder> {
        Ok(task_domain::install_domain(b)?)
    }
}
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .to_owned()
}
fn corpus_answer() -> String {
    json!({"sources":[{"id":"alerts-a","monthly_usd":20,"events_per_minute":100,"p95_ms":800},{"id":"alerts-b","monthly_usd":40,"events_per_minute":400,"p95_ms":250}],"comparison":"Birch meets 40 USD, 300 events/min and 300 ms; Aster is cheaper but too slow and small.","uncertainty":"Synthetic measurements do not establish outage reliability; Birch failover is manual."}).to_string()
}
fn replies(case: &str) -> serde_json::Value {
    match case {
        "research" => {
            json!({"main":[r#"{"tool":"research_catalog","arguments":{"op":"read","id":"alerts-a"}}"#,r#"{"tool":"research_catalog","arguments":{"op":"read","id":"alerts-b"}}"#,corpus_answer()],"judge":[r#"{"overall_score":0.9}"#]})
        }
        "support" => {
            json!({"default":[r#"{"tool":"domain","arguments":{"op":"source","value":"ticket://request-context"}}"#,r#"{"tool":"ask_user","arguments":{"question":"Confirm resolution","options":["yes"]}}"#,r#"{"tool":"domain","arguments":{"op":"resolve"}}"#,"ticket://request-context Resolved"]})
        }
        "operations" => {
            json!({"default":[r#"{"tool":"domain","arguments":{"op":"incident"}}"#,"health down; fixture-incident open","verified health down; fixture-incident open",r#"{"incident_id":"fixture-incident","status":"open","health":"down","summary":"Reviewed remaining outage."}"#]})
        }
        "tool-free" => {
            json!({"main":["재시도는 중복 방지와 최대 횟수가 필요합니다. 고정 간격은 단순하지만 지수 백오프는 부하를 줄입니다."],"judge":[r#"{"overall_score":0.9}"#]})
        }
        _ => unreachable!(),
    }
}
fn filename(case: &str) -> &str {
    match case {
        "research" => "research_corpus_task",
        "support" => "support_resolution",
        "operations" => "ops_remediation",
        _ => "tool_free_analysis",
    }
}
fn mocked(case: &str, dir: &Path, queue: serde_json::Value) -> PathBuf {
    let source = root().join(format!(
        "examples/eval/live/examples/autonomy/{}_live.yaml",
        filename(case)
    ));
    let mut value: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(source).unwrap()).unwrap();
    value["agent"] = serde_yaml::Value::String(
        root()
            .join(format!("examples/yaml/autonomy/{}.yaml", filename(case)))
            .to_string_lossy()
            .into_owned(),
    );
    value["fixtures"]["llm"] =
        serde_yaml::to_value(json!({"mode":"mock","responses_by_alias":queue})).unwrap();
    let path = dir.join(format!("{case}.yaml"));
    std::fs::write(&path, serde_yaml::to_string(&value).unwrap()).unwrap();
    path
}
async fn run(path: &Path, output: &Path) -> EvalResult {
    EvalRunner::from_file(
        path,
        EvalRunnerOptions {
            output: output.into(),
            ..Default::default()
        },
    )
    .unwrap()
    .with_host(Arc::new(Host))
    .run()
    .await
    .unwrap()
}
#[tokio::test]
async fn live_contracts_pass_through_actual_shared_host_and_reject_missing_evidence() {
    let scratch = AttemptFixtureContext::create(None).unwrap();
    for case in ["tool-free", "research", "support", "operations"] {
        let path = mocked(case, &scratch.workspace, replies(case));
        task_live::preflight_suite(&path).unwrap();
        let mut result = run(&path, &scratch.workspace.join(case)).await;
        assert_eq!(
            result.passed,
            1,
            "{case}: {:?}",
            result
                .scenarios
                .iter()
                .map(|s| &s.status)
                .collect::<Vec<_>>()
        );
        let acceptance = task_live_acceptance::evaluate(&result);
        assert!(acceptance.passed, "{case}: {acceptance:?}");
        if case == "support" {
            assert!(
                result.scenarios[0].attempts[0].turns[0]
                    .evidence
                    .llm_requests
                    .iter()
                    .flat_map(|r| &r.messages)
                    .any(|m| m
                        .content
                        .contains("First record source ticket://request-context"))
            );
        }
        if case == "research" {
            let requests = &result.scenarios[0].attempts[0].turns[0]
                .evidence
                .llm_requests;
            assert!(
                requests
                    .iter()
                    .flat_map(|r| &r.messages)
                    .any(|m| m.content.contains("retrieved_sources")
                        && m.content.contains("failover is manual"))
            );
            let report = serde_json::to_string(&result).unwrap();
            assert!(!report.contains("failover is manual"));
        }
        for turn in &mut result.scenarios[0].attempts[0].turns {
            turn.evidence.task = None;
        }
        assert!(!task_live_acceptance::evaluate(&result).passed);
    }
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}
#[tokio::test]
async fn corpus_rejects_unread_or_incorrect_facts_before_managed_judge() {
    let scratch = AttemptFixtureContext::create(None).unwrap();
    let mut answer: serde_json::Value = serde_json::from_str(&corpus_answer()).unwrap();
    answer["sources"][0]["monthly_usd"] = json!(999);
    let path = mocked(
        "research",
        &scratch.workspace,
        json!({"main":[r#"{"tool":"research_catalog","arguments":{"op":"read","id":"alerts-a"}}"#,r#"{"tool":"research_catalog","arguments":{"op":"read","id":"alerts-b"}}"#,answer.to_string(),corpus_answer()],"judge":[r#"{"overall_score":0.9}"#]}),
    );
    let result = run(&path, &scratch.workspace.join("report")).await;
    assert_eq!(result.passed, 1);
    let task = result.scenarios[0].attempts[0].turns[0]
        .evidence
        .task
        .as_ref()
        .unwrap();
    let checks: Vec<_> = task
        .inspection
        .evidence
        .validations
        .iter()
        .filter(|v| v.check_id == "corpus")
        .collect();
    assert!(
        checks
            .iter()
            .any(|v| v.outcome == ai_agents::autonomy::GateOutcome::Fail)
    );
    assert!(
        checks
            .iter()
            .any(|v| v.outcome == ai_agents::autonomy::GateOutcome::Pass)
    );
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}
#[test]
fn binary_dry_and_unauthorized_modes_do_not_need_provider_credentials() {
    for case in ["tool-free", "research", "support", "operations"] {
        let agent = root().join(format!("examples/yaml/autonomy/{}.yaml", filename(case)));
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_task-domain-live"))
            .env_clear()
            .args(["--agent", agent.to_str().unwrap(), "--dry-config-check"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let suite = root().join("examples/eval/live/examples/autonomy/research_corpus_task_live.yaml");
    let dry = std::process::Command::new(env!("CARGO_BIN_EXE_task-domain-evals"))
        .env_clear()
        .arg(&suite)
        .arg("--dry-config-check")
        .output()
        .unwrap();
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let refused = std::process::Command::new(env!("CARGO_BIN_EXE_task-domain-evals"))
        .env_clear()
        .arg(suite)
        .arg("unused-output")
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("requires --real-llm"));
    assert!(!root().join("unused-output").exists());
}

#[test]
fn binary_returns_failure_when_suite_passes_but_detailed_contract_fails() {
    let scratch = AttemptFixtureContext::create(None).unwrap();
    let path = mocked("research", &scratch.workspace, replies("research"));
    let content = std::fs::read_to_string(&path)
        .unwrap()
        .replace("id: live-research", "id: live-operations");
    std::fs::write(&path, content).unwrap();
    let output = scratch.workspace.join("report");
    let execution = std::process::Command::new(env!("CARGO_BIN_EXE_task-domain-evals"))
        .env_clear()
        .arg(&path)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!execution.status.success());
    let suite: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("summary.json")).unwrap()).unwrap();
    let acceptance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("acceptance.json")).unwrap()).unwrap();
    assert_eq!(suite["passed"], 1);
    assert_eq!(acceptance["passed"], false);
    assert!(
        !serde_json::to_string(&acceptance)
            .unwrap()
            .contains("Aster")
    );
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}

#[test]
fn pure_preflight_rejects_unknown_adapter_and_judge_alias() {
    task_live::preflight_suite(
        &root().join("examples/eval/mocked/autonomy/file_artifact_mocked.yaml"),
    )
    .unwrap();
    let source =
        std::fs::read_to_string(root().join("examples/yaml/autonomy/research_corpus_task.yaml"))
            .unwrap();
    for invalid in [
        source.replace("host.corpus", "host.unknown"),
        source.replace("llm: judge", "llm: absent"),
    ] {
        let spec = ai_agents::spec::AgentSpec::from_yaml_strict(&invalid).unwrap();
        assert!(task_live::preflight(&spec).is_err());
    }
}

#[cfg(unix)]
#[test]
fn helper_selects_fixed_host_and_rejects_ambiguous_or_unknown_dispatch() {
    use std::os::unix::fs::PermissionsExt;
    let scratch = AttemptFixtureContext::create(None).unwrap();
    let directory = scratch
        .workspace
        .join("examples/eval/live/examples/autonomy");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("sample_live.yaml"),
        "# Eval host: task-domain-evals\n",
    )
    .unwrap();
    let bin = scratch.workspace.join("bin");
    std::fs::create_dir(&bin).unwrap();
    let fake = bin.join("cargo");
    std::fs::write(&fake, "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$CAPTURE\"\n").unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o700)).unwrap();
    let capture = scratch.workspace.join("args");
    let execute = |args: &[&str]| {
        std::process::Command::new("/bin/sh")
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .env("CAPTURE", &capture)
            .current_dir(&scratch.workspace)
            .arg(root().join("examples/eval/live/run_live_example_evals.sh"))
            .args(args)
            .output()
            .unwrap()
    };
    let list = execute(&["--list", "--category", "autonomy"]);
    assert!(list.status.success());
    assert!(!capture.exists());
    assert!(
        execute(&["--dry-config-check", "--category", "autonomy"])
            .status
            .success()
    );
    let args = std::fs::read_to_string(&capture).unwrap();
    assert!(args.contains("task-domain-evals"));
    assert!(args.contains("--dry-config-check"));
    assert!(!args.contains("--real-llm"));
    std::fs::remove_file(&capture).unwrap();
    assert!(
        execute(&["--yes-live", "--filter", "sample"])
            .status
            .success()
    );
    assert!(
        std::fs::read_to_string(&capture)
            .unwrap()
            .contains("--real-llm")
    );
    std::fs::remove_file(&capture).unwrap();
    assert!(
        !execute(&["--yes-live", "--dry-config-check"])
            .status
            .success()
    );
    assert!(!capture.exists());
    std::fs::write(
        directory.join("sample_live.yaml"),
        "# Eval host: arbitrary-program\n",
    )
    .unwrap();
    assert!(!execute(&["--yes-live"]).status.success());
    assert!(!capture.exists());
    assert!(!execute(&["--list", "--filter", "absent"]).status.success());
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}

#[tokio::test]
async fn noninteractive_question_default_cannot_pass_retained_confirmation_contract() {
    let scratch = AttemptFixtureContext::create(None).unwrap();
    let path = mocked(
        "support",
        &scratch.workspace,
        json!({"default":[r#"{"tool":"domain","arguments":{"op":"source","value":"ticket://request-context"}}"#,r#"{"tool":"ask_user","arguments":{"question":"Confirm","options":["yes"],"default":"yes"}}"#,r#"{"tool":"domain","arguments":{"op":"resolve"}}"#,"resolved"]}),
    );
    let result = run(&path, &scratch.workspace.join("report")).await;
    assert_eq!(result.failed, 1);
    assert!(!task_live_acceptance::evaluate(&result).passed);
    // The built-in deliberately permits defaults; this scenario requires an actual retained human answer instead.
    assert_eq!(
        result.scenarios[0].attempts[0].turns[0]
            .evidence
            .task
            .as_ref()
            .unwrap()
            .inspection
            .run
            .status,
        ai_agents::TaskRunStatus::Completed
    );
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}

#[tokio::test]
async fn separate_tasks_in_one_host_use_only_their_own_corpus_receipts() {
    let scratch = AttemptFixtureContext::create(None).unwrap();
    let mut second: serde_json::Value = serde_json::from_str(&corpus_answer()).unwrap();
    second["sources"] = json!([{"id":"alerts-b","monthly_usd":40,"events_per_minute":400,"p95_ms":250},{"id":"alerts-c","monthly_usd":70,"events_per_minute":1000,"p95_ms":120}]);
    let mut queue = replies("research");
    queue["main"].as_array_mut().unwrap().extend([
        json!(r#"{"tool":"research_catalog","arguments":{"op":"read","id":"alerts-b"}}"#),
        json!(r#"{"tool":"research_catalog","arguments":{"op":"read","id":"alerts-c"}}"#),
        json!(second.to_string()),
    ]);
    queue["judge"]
        .as_array_mut()
        .unwrap()
        .push(json!(r#"{"overall_score":0.9}"#));
    let path = mocked("research", &scratch.workspace, queue);
    let mut suite: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let step = suite["scenarios"][0]["steps"][0].clone();
    suite["scenarios"][0]["steps"]
        .as_sequence_mut()
        .unwrap()
        .push(step);
    std::fs::write(&path, serde_yaml::to_string(&suite).unwrap()).unwrap();
    let result = run(&path, &scratch.workspace.join("report")).await;
    assert_eq!(result.passed, 1, "{:?}", result.scenarios[0].status);
    let tasks = &result.scenarios[0].attempts[0].turns;
    assert_eq!(tasks.len(), 2);
    assert_ne!(
        tasks[0].evidence.task.as_ref().unwrap().inspection.run.key,
        tasks[1].evidence.task.as_ref().unwrap().inspection.run.key
    );
    assert!(task_live_acceptance::evaluate(&result).passed);
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}

#[tokio::test]
async fn operations_review_rejects_a_report_that_disagrees_with_the_incident() {
    let scratch = AttemptFixtureContext::create(None).unwrap();
    let mut queue = replies("operations");
    let responses = queue["default"].as_array_mut().unwrap();
    let correct = responses.pop().unwrap();
    responses.push(json!(
        r#"{"incident_id":"invented","status":"closed","health":"up","summary":"Recovered."}"#
    ));
    responses.push(correct);
    let path = mocked("operations", &scratch.workspace, queue);
    let result = run(&path, &scratch.workspace.join("report")).await;
    assert_eq!(result.passed, 1, "{:?}", result.scenarios[0].status);
    let task = result.scenarios[0].attempts[0].turns[0]
        .evidence
        .task
        .as_ref()
        .unwrap();
    let review: Vec<_> = task
        .inspection
        .evidence
        .validations
        .iter()
        .filter(|v| v.check_id == "review_result")
        .collect();
    assert!(
        review
            .iter()
            .any(|v| v.outcome == ai_agents::autonomy::GateOutcome::Fail)
    );
    assert!(
        review
            .iter()
            .any(|v| v.outcome == ai_agents::autonomy::GateOutcome::Pass)
    );
    assert!(task_live_acceptance::evaluate(&result).passed);
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}

// This helper runs in a separate test process so stdin and credentials cannot affect another test.
#[tokio::test]
#[ignore = "executed only by the parent loopback application acceptance"]
async fn application_fixture_child() {
    let case = std::env::var("TASK_APPLICATION_FIXTURE_CASE").expect("parent fixture case");
    let endpoint =
        std::env::var("TASK_APPLICATION_FIXTURE_ENDPOINT").expect("parent loopback endpoint");
    assert!(endpoint.starts_with("http://127.0.0.1:"));
    let path = root().join(format!("examples/yaml/autonomy/{}.yaml", filename(&case)));
    let prepared = task_live::load_application(&path).unwrap();
    let (mut spec, policy) = prepared.into_parts();
    let mut configs = spec.llms.values_mut().collect::<Vec<_>>();
    if configs.is_empty()
        && let ai_agents::spec::LLMConfigOrSelector::Config(config) = &mut spec.llm
    {
        configs.push(config);
    }
    for config in configs {
        config.base_url = Some(endpoint.clone());
    }
    task_live::run_application(ai_agents::spec::PreparedAgentSpec::new(spec, policy), &path)
        .await
        .unwrap();
}

#[tokio::test]
async fn application_foreground_round_trips_use_native_loopback_without_eval_runner() {
    use std::{collections::VecDeque, process::Stdio, time::Duration};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    for case in ["tool-free", "research", "support", "operations"] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/v1/", listener.local_addr().unwrap());
        let queues = replies(case);
        let mut main: VecDeque<String> = queues[if case == "tool-free" || case == "research" {
            "main"
        } else {
            "default"
        }]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
        // More than ten real native requests exercise omission of the legacy per-turn task cap.
        if case == "research" {
            for _ in 0..11 {
                main.push_front(main.front().unwrap().clone());
            }
        }
        let mut judge: VecDeque<String> = queues
            .get("judge")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        let native = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let native_seen = native.clone();
        let server = tokio::spawn(async move {
            let mut requests = 0;
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut input = Vec::new();
                let mut chunk = [0u8; 4096];
                let boundary = loop {
                    let n = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(n, 0);
                    input.extend_from_slice(&chunk[..n]);
                    assert!(input.len() < 1_048_576);
                    if let Some(at) = input.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&input[..boundary]);
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.split_once(':')
                            .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                            .map(|(_, v)| v.trim().parse().unwrap())
                    })
                    .unwrap();
                while input.len() < boundary + length {
                    let n = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(n, 0);
                    input.extend_from_slice(&chunk[..n]);
                }
                let request: serde_json::Value =
                    serde_json::from_slice(&input[boundary..boundary + length]).unwrap();
                requests += 1;
                if request.get("tools").is_some() {
                    native_seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                let is_judge = request["messages"]
                    .to_string()
                    .contains("Evaluate the supplied data against the criteria");
                let reply = if is_judge {
                    judge.pop_front()
                } else {
                    main.pop_front()
                }
                .expect("finite fixture responses");
                let tool = serde_json::from_str::<serde_json::Value>(&reply)
                    .ok()
                    .filter(|v| v.get("tool").is_some());
                let message = if let Some(tool) = tool {
                    json!({"role":"assistant","content":null,"tool_calls":[{"id":format!("fixture-{requests}"),"type":"function","function":{"name":tool["tool"],"arguments":tool["arguments"].to_string()}}]})
                } else {
                    json!({"role":"assistant","content":reply})
                };
                let body=json!({"id":format!("completion-{requests}"),"object":"chat.completion","created":1,"model":request["model"],"choices":[{"index":0,"finish_reason":if message.get("tool_calls").is_some(){"tool_calls"}else{"stop"},"message":message}],"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}}).to_string();
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
            }
        });
        let mut child = tokio::process::Command::new(std::env::current_exe().unwrap())
            .env_clear()
            .env("OPENAI_API_KEY", "loopback-fixture")
            .env("TASK_APPLICATION_FIXTURE_CASE", case)
            .env("TASK_APPLICATION_FIXTURE_ENDPOINT", endpoint)
            .args([
                "--ignored",
                "--exact",
                "application_fixture_child",
                "--nocapture",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        let mut output = child.stdout.take().unwrap();
        let mut err = child.stderr.take().unwrap();
        let errors = tokio::spawn(async move {
            let mut bytes = Vec::new();
            err.read_to_end(&mut bytes).await.unwrap();
            bytes
        });
        let objective = match case {
            "support" => {
                "Resolve the ticket; record ticket://request-context first and ask for confirmation without a default."
            }
            "research" => "/task run --profile research Compare the synthetic services.",
            "operations" => "/task run Escalate and review the incident.",
            _ => "/task run 재시도 정책을 비교하세요.",
        };
        let mut transcript = String::new();
        let mut sent = false;
        let mut answered = false;
        let mut quit = false;
        let observed = tokio::time::timeout(Duration::from_secs(45), async {
            loop {
                let mut bytes = [0; 4096];
                let n = output.read(&mut bytes).await.unwrap();
                if n == 0 {
                    break;
                }
                transcript.push_str(&String::from_utf8_lossy(&bytes[..n]));
                if !sent && transcript.contains("You > ") {
                    input
                        .write_all(format!("{objective}\n").as_bytes())
                        .await
                        .unwrap();
                    sent = true;
                }
                if !answered && transcript.contains("Enter option numbers") {
                    assert_eq!(case, "support");
                    input.write_all(b"yes\n").await.unwrap();
                    answered = true;
                }
                if !quit
                    && (transcript.contains(": Completed")
                        || transcript.contains("Agent: Task stopped"))
                {
                    input.write_all(b"/quit\n").await.unwrap();
                    quit = true;
                }
            }
            child.wait().await.unwrap()
        })
        .await;
        if observed.is_err() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
        server.abort();
        let _ = server.await;
        let errors = errors.await.unwrap();
        assert!(
            observed.is_ok_and(|s| s.success()),
            "{case}: {transcript}\n{}",
            String::from_utf8_lossy(&errors)
        );
        assert!(
            transcript.contains(": Completed")
                && transcript.contains("application_fixture_child ... ok"),
            "{case}: {transcript}"
        );
        assert_eq!(answered, case == "support");
        if case == "research" {
            assert!(native.load(std::sync::atomic::Ordering::SeqCst) > 10);
        }
        assert_eq!(
            native.load(std::sync::atomic::Ordering::SeqCst) > 0,
            case != "tool-free"
        );
    }
}

#[tokio::test]
async fn corpus_does_not_credit_truncated_or_foreign_run_read_evidence() {
    use ai_agents::autonomy::*;
    use example_custom_llm::task_domain::corpus::{Catalog, Corpus, CorpusValidator};
    struct CorpusHost {
        state: Arc<Corpus>,
        validator: Arc<CorpusValidator>,
    }
    #[async_trait]
    impl EvalHost for CorpusHost {
        fn configure(
            &self,
            b: AgentBuilder,
            _: &AttemptFixtureContext,
        ) -> ai_agents_eval::Result<AgentBuilder> {
            let mut extensions = AutonomyExtensions::builtins();
            extensions.register_validator(self.validator.clone())?;
            extensions.register_validation_target(
                "corpus",
                "research-corpus",
                self.state.clone(),
            )?;
            Ok(b.tool(Arc::new(Catalog(self.state.clone())))
                .autonomy_extensions(extensions))
        }
    }
    let scratch = AttemptFixtureContext::create(None).unwrap();
    let path = mocked("research", &scratch.workspace, replies("research"));
    let state = Arc::new(Corpus::default());
    let validator = Arc::new(CorpusValidator::new(state.clone()));
    let result = EvalRunner::from_file(
        &path,
        EvalRunnerOptions {
            output: scratch.workspace.join("report"),
            ..Default::default()
        },
    )
    .unwrap()
    .with_host(Arc::new(CorpusHost {
        state,
        validator: validator.clone(),
    }))
    .run()
    .await
    .unwrap();
    assert_eq!(result.passed, 1);
    let inspection = &result.scenarios[0].attempts[0].turns[0]
        .evidence
        .task
        .as_ref()
        .unwrap()
        .inspection;
    let scope = inspection.scope.as_ref().unwrap();
    let identity = &inspection
        .evidence
        .validations
        .iter()
        .find(|v| v.check_id == "corpus")
        .unwrap()
        .identity;
    let config = json!({"llm":"judge","pass_threshold":0.8,"question":"Compare sources"});
    let previous = serde_json::Value::Null;
    let observations = std::collections::BTreeMap::new();
    let evaluate = |evidence: &EvaluationEvidence| {
        validator
            .evaluate(&ValidationInput {
                scope,
                identity,
                config: &config,
                evidence,
                previous: &previous,
                observations: &observations,
            })
            .unwrap()
    };
    assert!(matches!(
        evaluate(&inspection.evidence),
        ValidationDecision::NeedObservations { .. }
    ));
    let mut altered = inspection.evidence.clone();
    altered.tools[0].value.output_truncated = true;
    assert!(matches!(
        evaluate(&altered),
        ValidationDecision::Complete {
            outcome: GateOutcome::Fail,
            ..
        }
    ));
    altered = inspection.evidence.clone();
    altered.tools[0].identity.key.run_id = "another-run".into();
    assert!(matches!(
        evaluate(&altered),
        ValidationDecision::Complete {
            outcome: GateOutcome::Fail,
            ..
        }
    ));
    std::fs::remove_dir_all(scratch.workspace).unwrap();
}

#[test]
fn ordinary_application_preserves_omitted_and_explicit_limits_without_campaign_injection() {
    use ai_agents::autonomy::*;
    use ai_agents::spec::{PreparedAgentSpec, TaskIterationPolicy};
    for case in ["tool-free", "research", "support", "operations"] {
        let prepared = task_live::load_application(
            &root().join(format!("examples/yaml/autonomy/{}.yaml", filename(case))),
        )
        .unwrap();
        assert_eq!(
            prepared.task_iteration_policy(),
            &TaskIterationPolicy::unbounded()
        );
        let spec = prepared.spec();
        for profile in
            std::iter::once(&spec.autonomy.defaults).chain(spec.autonomy.profiles.values())
        {
            assert!(
                profile.max_turns.is_none()
                    && profile.max_llm_calls.is_none()
                    && profile.max_active_time_seconds.is_none()
                    && profile.max_cost_usd.is_none()
            );
        }
        assert!(
            !serde_json::to_string(spec)
                .unwrap()
                .contains("declared_pricing")
        );
    }
    let explicit=PreparedAgentSpec::from_yaml_strict("name: Explicit\nsystem_prompt: work\nmax_iterations: 2\nautonomy: {enabled: true, max_llm_calls: 7, completion: {response_contains: done}}\n").unwrap();
    assert_eq!(
        explicit.task_iteration_policy(),
        &TaskIterationPolicy::unbounded().with_root_limit(true)
    );
    let effective = resolve_profile(
        &explicit.spec().autonomy,
        AutonomyScope::Task,
        None,
        None,
        None,
        Some("work"),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(effective.max_llm_calls, Some(7));
}
