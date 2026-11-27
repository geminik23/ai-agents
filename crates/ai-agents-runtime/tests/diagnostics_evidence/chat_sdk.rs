//! One priced YAML consumer across an approved native edit, SQLite restart and a fresh Rust check.
use super::*;
use ai_agents_core::{PermissionOutcome, ToolApprovalStatus};
use ai_agents_hitl::ApprovalResult;
use tokio::{io::AsyncWriteExt, net::TcpListener};

#[path = "../support/chat_http.rs"]
mod chat_http;

const MODEL: &str = "local-chat";
const INPUT_LIMIT: u64 = 65_536;
const OUTPUT_LIMIT: u64 = 512;
// One micro-USD per input token, two per output token and three per request.
const RESERVATION: u64 = INPUT_LIMIT + 2 * OUTPUT_LIMIT + 3;

// The declaration binds the Chat provider and pricing schedule. Only the document host is custom.
struct DocumentHost(Application);

impl TaskRestoreComponentState for DocumentHost {
    fn binding(&self) -> ai_agents_core::Result<TaskRestoreBinding> {
        let (id, contract_version) = self.identity();
        Ok(TaskRestoreBinding {
            id,
            contract_version,
            configuration: self.0.configuration(),
            state: Value::Null,
        })
    }
}

impl TaskRestoreComponents for DocumentHost {
    fn identity(&self) -> (String, u32) {
        ("priced-document-host".into(), 1)
    }

    fn validate(
        &self,
        _: &PreparedAgentSpec,
        binding: &TaskRestoreBinding,
    ) -> ai_agents_core::Result<()> {
        if binding.configuration != self.0.configuration() || !binding.state.is_null() {
            return Err(AgentError::Config("document host binding changed".into()));
        }
        Ok(())
    }

    fn install(
        &self,
        _: &PreparedAgentSpec,
        builder: AgentBuilder,
        _: Option<&TaskRestoreBinding>,
        shared: Option<&ai_agents_llm::LLMRegistry>,
    ) -> ai_agents_core::Result<(AgentBuilder, Arc<dyn TaskRestoreComponentState>)> {
        assert!(shared.is_none(), "the consumer is a single root task");
        let builder = document_builder(&self.0.directory, builder)?
            .auto_configure_llms()?
            .auto_configure_features()?;
        Ok((
            builder,
            Arc::new(DocumentHost(Application {
                directory: self.0.directory.clone(),
            })),
        ))
    }
}

fn task_yaml(root: &Path, endpoint: &str, attempts: usize) -> String {
    let file = root.join("project").join(PATH);
    format!(
        r#"
name: CombinedTask
system_prompt: Repair the selected Rust document.
llm:
  provider: openai
  model: {MODEL}
  api_key_env: AI_AGENTS_LOCAL_CHAT_KEY
  base_url: '{endpoint}'
  tool_choice: auto
  max_tokens: {OUTPUT_LIMIT}
  declared_pricing:
    schedule_id: document-test
    model: {MODEL}
    endpoint: '{endpoint}'
    server_input_token_limit: {INPUT_LIMIT}
    server_output_token_limit: {OUTPUT_LIMIT}
    input_micro_usd_per_million: 1000000
    output_micro_usd_per_million: 2000000
    request_fee_micro_usd: 3
    max_internal_attempts: 1
    complete_usage: true
tools: [file_read, file_edit, ask_user, diagnostics]
hitl: {{default_timeout_seconds: 60, on_timeout: reject}}
tool_security:
  enabled: true
  fail_closed: true
  tools:
    file_read: {{read_paths: ['{file}']}}
    file_edit: {{write_paths: ['{file}'], require_confirmation: true, require_read_before_write: true}}
    diagnostics: {{read_paths: ['{file}'], allow_without_confirmation: true}}
    ask_user: {{allow_without_confirmation: true}}
autonomy:
  enabled: true
  max_cost_usd: '0.1'
  max_llm_calls: {attempts}
  max_wall_time_seconds: 180
  restoration_timeout_ms: 20000
  persistence: {{enabled: true, resume_on_start: true}}
  hitl: {{on_user_question: pause_run, on_approval_required: pause_run}}
  validation:
    checks: [{{id: syntax, adapter: builtin.diagnostics, config: {{path: '{file}'}}}}]
  completion: {{all: [{{response_contains: done}}, {{validator_passed: syntax}}]}}
storage: {{type: sqlite, path: '{db}'}}
"#,
        file = file.display(),
        db = root.join("tasks.sqlite").display()
    )
}

fn tool_call(id: &str, name: &str, args: Value) -> Value {
    json!({"role":"assistant","content":null,"tool_calls":[{
        "id":id, "type":"function", "function":{"name":name,"arguments":args.to_string()}
    }]})
}

fn message(index: usize, file: &Path) -> Value {
    match index {
        0 => tool_call("before", "diagnostics", json!({"path":file})),
        1 => tool_call("read", "file_read", json!({"path":file})),
        2 => tool_call(
            "edit",
            "file_edit",
            json!({"path":file,"old_text":BROKEN_LINE,"new_text":FIX}),
        ),
        3 => tool_call("after", "diagnostics", json!({"path":file})),
        4 => tool_call(
            "wait",
            "ask_user",
            json!({"question":"Accept repair?","options":["yes"],"allow_other":false}),
        ),
        5 => json!({"role":"assistant","content":"done"}),
        _ => panic!("unexpected model request {index}"),
    }
}

async fn checkpoint(scope: &TaskHostScope) -> (TaskRunSnapshot, TaskCheckpointPayload) {
    let store = scope.open_store().await.unwrap();
    let runs = store.list().await.unwrap();
    assert_eq!(runs.len(), 1);
    let saved = store.load(&runs[0].key.run_id).await.unwrap().unwrap();
    let payload = serde_json::from_value(saved.payload.clone()).unwrap();
    (saved, payload)
}

struct Endpoint {
    listener: TcpListener,
    scope: TaskHostScope,
    // Actual durable reservation IDs sampled while each HTTP response was still withheld.
    charges: BTreeMap<String, u64>,
}

impl Endpoint {
    fn reconcile(&self, payload: &TaskCheckpointPayload) {
        let completed: BTreeMap<_, _> = payload
            .reservations
            .iter()
            .filter(|entry| {
                entry.reserved_micro_usd > 0 && entry.state == TaskEffectState::Completed
            })
            .map(|entry| (entry.id.clone(), entry.charged_micro_usd))
            .collect();
        assert_eq!(completed, self.charges);
        assert_eq!(
            payload.counters.charged_micro_usd,
            self.charges.values().sum::<u64>()
        );
    }

    // At a safe pause or after the final response, every HTTP reservation must be fully settled.
    fn assert_settled(&self, payload: &TaskCheckpointPayload) {
        self.reconcile(payload);
        assert_eq!(payload.counters.llm_attempts, self.charges.len() as u64);
        assert!(
            payload
                .reservations
                .iter()
                .filter(|r| r.reserved_micro_usd > 0)
                .all(|r| r.state == TaskEffectState::Completed
                    && r.reserved_micro_usd == RESERVATION)
        );
    }

    async fn respond(&mut self, mut socket: tokio::net::TcpStream, body: Value, root: &Path) {
        let index = self.charges.len();
        assert_eq!(body["model"], MODEL);
        assert!(!body["tools"].as_array().unwrap().is_empty());
        let messages = body["messages"].as_array().unwrap();
        if index > 0 {
            let prior_id = ["before", "read", "edit", "after", "wait"][index - 1];
            let result = messages.iter().rev().find(|m| m["role"] == "tool").unwrap();
            assert_eq!(result["tool_call_id"], prior_id);
            let content: Value = serde_json::from_str(result["content"].as_str().unwrap()).unwrap();
            match prior_id {
                "before" => assert!(
                    content["diagnostics"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|item| item["severity"] == "error" && item["code"] == "E0308")
                ),
                "read" => assert!(content["content"].as_str().unwrap().contains(BROKEN_LINE)),
                "edit" => assert_eq!(content["mutation_performed"], true),
                "after" => {
                    assert_eq!(content["available"], true);
                    assert_eq!(content["count"], 0);
                    assert!(content["message"].is_null());
                }
                "wait" => assert_eq!(content["selected"], json!(["yes"])),
                _ => unreachable!(),
            }
        }
        let (snapshot, payload) = checkpoint(&self.scope).await;
        assert!(snapshot.owner_token.is_some());
        self.reconcile(&payload);
        let mut active = payload
            .reservations
            .iter()
            .filter(|r| r.reserved_micro_usd > 0 && r.state == TaskEffectState::Dispatched);
        let attempt = active
            .next()
            .expect("priced reservation before HTTP response");
        assert!(active.next().is_none());
        assert_eq!(attempt.reserved_micro_usd, RESERVATION);
        assert_eq!(attempt.charged_micro_usd, 0);
        assert_eq!(payload.counters.llm_attempts, index as u64 + 1);
        let stage = fs::read_to_string(root.join("dispatch-stage")).unwrap();
        assert_eq!(
            stage,
            match index {
                0..=2 => "seed",
                3..=4 => "approved",
                5 => "resumed",
                _ => unreachable!(),
            }
        );
        let file = root.join("project").join(PATH);
        let response = message(index, &file);
        let reason = if response.get("tool_calls").is_some() {
            "tool_calls"
        } else {
            "stop"
        };
        // Synthetic metering at the HTTP boundary includes the schema and native history.
        let input = body.to_string().len() as u64;
        let output = response.to_string().len() as u64;
        assert!(input <= INPUT_LIMIT && output <= OUTPUT_LIMIT);
        let charge = input + output * 2 + 3;
        assert!(self.charges.insert(attempt.id.clone(), charge).is_none());
        let data = json!({"id":format!("chat-{index}"), "model":MODEL,
            "choices":[{"index":0,"message":response,"finish_reason":reason}],
            "usage":{"prompt_tokens":input,"completion_tokens":output,"total_tokens":input+output}})
        .to_string();
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}",data.len()).as_bytes()).await.unwrap();
    }

    async fn worker(&mut self, root: &Path, mode: &str) {
        fs::write(root.join("dispatch-stage"), "preparation").unwrap();
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "lsp_acceptance::chat_sdk::combined_worker",
                "--ignored",
                "--nocapture",
            ])
            .current_dir(root)
            .env("DOCUMENT_WORKER", mode)
            .env("AI_AGENTS_LOCAL_CHAT_KEY", "fixture")
            .kill_on_drop(true);
        let child = command.output();
        tokio::pin!(child);
        tokio::time::timeout(Duration::from_secs(100), async {
            loop {
                tokio::select! {
                    output = &mut child => {
                        let output = output.unwrap();
                        assert!(output.status.success(), "{mode}: {}", String::from_utf8_lossy(&output.stderr));
                        break;
                    },
                    request = chat_http::receive_path(&self.listener,"/v1/chat/completions") => {
                        let (socket,body,_) = request;
                        self.respond(socket,body,root).await;
                    }
                }
            }
        }).await.expect("combined task child exceeded its test deadline");
    }
}

#[tokio::test]
#[ignore = "child process driven by combined_chat_edit_diagnostics_restart"]
async fn combined_worker() {
    let Ok(mode) = std::env::var("DOCUMENT_WORKER") else {
        return;
    };
    let directory = std::env::current_dir().unwrap();
    let spec =
        PreparedAgentSpec::from_yaml_strict(&fs::read_to_string("agent.yaml").unwrap()).unwrap();
    let host = Arc::new(
        ConfiguredTaskRestoreHost::new(spec, directory.clone(), None, Duration::from_secs(20))
            .unwrap()
            .host_ceilings(AutonomyHostCeilings {
                max_cost_usd: Some(UsdAmount::parse("0.1").unwrap()),
                ..Default::default()
            })
            .components(Arc::new(DocumentHost(Application {
                directory: directory.clone(),
            })))
            .unwrap(),
    );
    if mode.starts_with("resume-") {
        resume_document(&directory, host, &mode, || {
            fs::write("dispatch-stage", "resumed").unwrap()
        })
        .await;
        return;
    }
    let agent = host.build().await.unwrap();
    fs::write("dispatch-stage", "seed").unwrap();
    let paused = agent
        .run_task("Fix this Rust document", None)
        .await
        .unwrap();
    assert_eq!(
        paused.run.status,
        TaskRunStatus::Paused,
        "{:?}",
        paused.run.stop_reason
    );
    let pending = paused.run.pending.as_ref().unwrap();
    assert!(
        matches!(pending.kind, TaskPendingKind::Approval),
        "{pending:?}"
    );
    assert_eq!(paused.run.counters.llm_attempts, 3);
    assert!(paused.run.counters.charged_micro_usd > 0);
    assert!(pending.issued_revision <= paused.run.revision);
    assert_eq!(pending.reviewed_action["call_id"], "edit");
    assert_eq!(pending.reviewed_action["request_id"], pending.id);
    let file = directory.join("project").join(PATH);
    let expected_args = json!({"path":file,"old_text":BROKEN_LINE,"new_text":FIX});
    assert_eq!(pending.reviewed_action["trigger"]["name"], "file_edit");
    assert_eq!(pending.reviewed_action["trigger"]["args"], expected_args);
    assert_eq!(fs::read_to_string(&file).unwrap(), ORIGINAL);
    assert!(
        !paused
            .run
            .evidence
            .tool_calls
            .iter()
            .any(|e| e.record.canonical_id == "file_edit" && e.record.executed)
    );
    let decision = match mode.as_str() {
        "reject" => ApprovalResult::rejected_with_reason("keep the original"),
        "outside" => {
            ApprovalResult::modified([("path".into(), json!(directory.join("outside.rs")))].into())
        }
        "seed" => ApprovalResult::Approved,
        other => panic!("unknown worker {other}"),
    };
    fs::write("dispatch-stage", "approved").unwrap();
    let result = agent
        .resume_task(
            &paused.run.key.run_id,
            paused.run.revision,
            TaskResumeInput::Approval {
                request_id: pending.id.clone(),
                result: decision,
            },
        )
        .await
        .unwrap();
    let record = &result
        .run
        .evidence
        .tool_calls
        .iter()
        .find(|e| e.record.canonical_id == "file_edit")
        .expect("reviewed edit evidence")
        .record;
    assert_eq!(record.call_id, "edit");
    assert_eq!(record.arguments, expected_args);
    if mode != "seed" {
        assert!(!record.executed && !record.success, "{record:?}");
        let approval = record.approval.as_ref().expect("reviewed edit decision");
        if mode == "outside" {
            assert!(
                matches!(approval.status, ToolApprovalStatus::Modified),
                "{record:?}"
            );
            assert_eq!(record.policy.outcome, PermissionOutcome::Deny);
            assert_eq!(
                record.executed_arguments["path"],
                json!(directory.join("outside.rs"))
            );
        } else {
            assert!(
                matches!(approval.status, ToolApprovalStatus::Rejected),
                "{record:?}"
            );
        }
        assert_eq!(fs::read_to_string(file).unwrap(), ORIGINAL);
        assert_eq!(
            fs::read_to_string(directory.join("outside.rs")).unwrap(),
            ORIGINAL
        );
        assert_ne!(result.run.status, TaskRunStatus::Completed);
        return;
    }
    assert!(record.executed && record.success);
    assert!(matches!(
        record.approval.as_ref().unwrap().status,
        ToolApprovalStatus::Approved
    ));
    assert_eq!(record.executed_arguments, expected_args);
    assert_eq!(
        serde_json::from_str::<Value>(&record.output).unwrap()["mutation_performed"],
        true
    );
    assert!(matches!(
        result.run.pending.as_ref().unwrap().kind,
        TaskPendingKind::UserQuestion
    ));
    save_after_edit(&directory, &result, 2);
}

#[tokio::test]
#[ignore = "requires rust-analyzer and native Linux restoration time evidence"]
async fn combined_chat_edit_diagnostics_restart() {
    for scenario in ["resume-current", "resume-changed", "reject", "outside"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let file = root.join("project").join(PATH);
        project(&root.join("project"));
        fs::write(root.join("outside.rs"), ORIGINAL).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1/", listener.local_addr().unwrap());
        let resume = scenario.starts_with("resume-");
        let yaml = task_yaml(root, &base, if resume { 6 } else { 3 });
        fs::write(root.join("agent.yaml"), &yaml).unwrap();
        let spec = PreparedAgentSpec::from_yaml_strict(&yaml).unwrap();
        let scope = TaskHostScope::from_prepared_spec(&spec, root, None).unwrap();
        let mut endpoint = Endpoint {
            listener,
            scope,
            charges: BTreeMap::new(),
        };
        endpoint
            .worker(root, if resume { "seed" } else { scenario })
            .await;
        let (saved, payload) = checkpoint(&endpoint.scope).await;
        endpoint.assert_settled(&payload);
        assert_eq!(endpoint.charges.len(), if resume { 5 } else { 3 });
        if resume {
            if scenario == "resume-changed" {
                fs::write(&file, ORIGINAL).unwrap();
            }
            let bytes = fs::read(&file).unwrap();
            let modified = fs::metadata(&file).unwrap().modified().unwrap();
            endpoint.worker(root, scenario).await;
            let (finished, payload) = checkpoint(&endpoint.scope).await;
            assert_eq!(finished.key, saved.key);
            assert!(finished.revision > saved.revision);
            assert_eq!(endpoint.charges.len(), 6);
            endpoint.assert_settled(&payload);
            let validation = payload
                .evidence
                .records
                .iter()
                .rev()
                .filter_map(|record| record.data["validations"].as_array())
                .flatten()
                .find(|result| result["check_id"] == "syntax")
                .expect("acknowledged task validation");
            assert_eq!(
                validation["outcome"],
                if scenario == "resume-current" {
                    "pass"
                } else {
                    "fail"
                }
            );
            assert_eq!(validation["identity"]["target"], "selected-document");
            assert!(
                validation["identity"]["target_revision"]
                    .as_str()
                    .is_some_and(|rev| !rev.is_empty())
            );
            assert_eq!(fs::read(&file).unwrap(), bytes);
            assert_eq!(fs::metadata(&file).unwrap().modified().unwrap(), modified);
        }
        println!(
            "{scenario}: {} HTTP receipts, {} micro-USD reconciled",
            endpoint.charges.len(),
            endpoint.charges.values().sum::<u64>()
        );
    }
}
