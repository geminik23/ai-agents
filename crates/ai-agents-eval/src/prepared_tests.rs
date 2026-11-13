//! A disposable producer/consumer fixture exercises typed eval against the installed prepared SDK, not a production clock host.
use ai_agents_core::{
    ChatMessage, FinishReason, LLMChunk, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse,
};
use ai_agents_runtime::autonomy::*;
use ai_agents_runtime::{AgentBuilder, RuntimeAgent};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicUsize, Ordering},
};

const AGENT: &str = "name: PreparedEval\nsystem_prompt: Produce a verified report.\ntools: [ticket]\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  max_active_time_seconds: 60\n  max_wall_time_seconds: 3600\n  validation: {checks: [{id: ticket, adapter: host.ticket, config: {}}]}\n  completion: {validator_passed: ticket}\n  premature_finish: {action: ask_user, on_unavailable: stop}\n";
// The application record is durable independently of the task journal; a provider's success claim cannot resolve it.
struct Ticket {
    path: PathBuf,
    status: Mutex<String>,
}
#[async_trait]
impl ai_agents_core::Tool for Ticket {
    fn id(&self) -> &str {
        "ticket"
    }
    fn name(&self) -> &str {
        "Disposable ticket"
    }
    fn description(&self) -> &str {
        "Resolve the isolated ticket record"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"resolve":{"type":"boolean"}},"required":["resolve"],"additionalProperties":false})
    }
    async fn execute(
        &self,
        args: Value,
        _: ai_agents_core::ToolExecutionContext,
    ) -> ai_agents_core::ToolResult {
        if args["resolve"] != true {
            return ai_agents_core::ToolResult::error("invalid ticket action");
        }
        if std::fs::write(&self.path, "resolved").is_err() {
            return ai_agents_core::ToolResult::error("ticket write failed");
        }
        *self.status.lock().unwrap() = "resolved".into();
        ai_agents_core::ToolResult::ok("resolved")
    }
}
struct TicketValidator {
    descriptor: AdapterDescriptor,
    ticket: Arc<Ticket>,
}
impl AutonomyValidator for TicketValidator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    fn validate_config(&self, config: &Value) -> ai_agents_core::Result<()> {
        if config.as_object().is_none_or(|object| !object.is_empty()) {
            return Err(ai_agents_core::AgentError::Config(
                "ticket config must be empty".into(),
            ));
        }
        Ok(())
    }
    fn evaluate(&self, _: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        Ok(ValidationDecision::Complete {
            outcome: if *self.ticket.status.lock().unwrap() == "resolved" {
                GateOutcome::Pass
            } else {
                GateOutcome::Fail
            },
            reason: "actual_ticket_state".into(),
            metrics: json!({}),
            evidence_refs: vec![],
        })
    }
}
fn install_ticket(
    builder: AgentBuilder,
    directory: &std::path::Path,
) -> ai_agents_core::Result<AgentBuilder> {
    let path = directory.join("ticket.txt");
    let status = std::fs::read_to_string(&path)
        .map_err(|error| ai_agents_core::AgentError::Config(error.to_string()))?;
    let ticket = Arc::new(Ticket {
        path,
        status: Mutex::new(status),
    });
    let mut extensions = AutonomyExtensions::builtins();
    extensions.register_validator(Arc::new(TicketValidator {
        descriptor: AdapterDescriptor {
            id: "host.ticket".into(),
            contract_version: 1,
            ..Default::default()
        },
        ticket: ticket.clone(),
    }))?;
    Ok(builder.tool(ticket).autonomy_extensions(extensions))
}

struct Queue {
    replies: Mutex<VecDeque<String>>,
    calls: AtomicUsize,
}
impl Queue {
    fn new(replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.iter().map(|s| (*s).into()).collect()),
            calls: AtomicUsize::new(0),
        })
    }
}
#[async_trait]
impl LLMProvider for Queue {
    async fn complete(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let reply = self.replies.lock().unwrap().pop_front().ok_or_else(|| {
            LLMError::Config("fixture queue exhausted; work may have replayed".into())
        })?;
        Ok(LLMResponse::new(reply, FinishReason::Stop))
    }
    async fn complete_stream(
        &self,
        _: &[ChatMessage],
        _: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<dyn futures::Stream<Item = std::result::Result<LLMChunk, LLMError>> + Send + Unpin>,
        LLMError,
    > {
        Err(LLMError::Config("unexpected provider token stream".into()))
    }
    fn provider_name(&self) -> &str {
        "prepared-eval-fixture"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
}
struct RestoreHost {
    providers: Mutex<BTreeMap<usize, Weak<Queue>>>,
    staged: Mutex<Option<Arc<RuntimeAgent>>>,
}
impl RestoreHost {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            providers: Default::default(),
            staged: Default::default(),
        })
    }
    fn register(&self, agent: &Arc<RuntimeAgent>, queue: &Arc<Queue>) {
        self.providers
            .lock()
            .unwrap()
            .insert(Arc::as_ptr(agent) as usize, Arc::downgrade(queue));
    }
}
#[async_trait]
impl TaskRestoreHost for RestoreHost {
    fn identity(&self) -> (String, u32) {
        ("prepared-eval-v1".into(), 1)
    }
    fn reconstruction_budget(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }
    // Only this controlled test fixture certifies the backend's UTC source; it is not the default production hosting contract.
    fn now(&self) -> ai_agents_core::Result<chrono::DateTime<chrono::Utc>> {
        Ok(chrono::Utc::now())
    }
    fn runtime_binding(
        &self,
        runtime: &RuntimeAgent,
        _: Option<&ai_agents_runtime::spec::AgentSpec>,
    ) -> ai_agents_core::Result<TaskRestoreBinding> {
        let provider = self
            .providers
            .lock()
            .unwrap()
            .get(&(runtime as *const RuntimeAgent as usize))
            .and_then(Weak::upgrade)
            .ok_or_else(|| {
                ai_agents_core::AgentError::Config("unknown actual fixture runtime".into())
            })?;
        let state = serde_json::to_value(provider.replies.lock().unwrap().clone())?;
        Ok(TaskRestoreBinding {
            id: self.identity().0,
            contract_version: 1,
            configuration: json!({"recipe":"prepared-eval-v1","provider":"queue-v1","hooks":"provider-free-caller-owned"}),
            state,
        })
    }
    fn provider_binding(
        &self,
        _: &ai_agents_llm::LLMRegistry,
    ) -> ai_agents_core::Result<TaskRestoreBinding> {
        // The audited constructor installs one actual queue; runtime_binding covers its complete hidden state, not its display name.
        Ok(TaskRestoreBinding {
            id: self.identity().0,
            contract_version: 1,
            configuration: json!({"providers":"queue-v1"}),
            state: Value::Null,
        })
    }
    fn validate_manifest(&self, manifest: &TaskRestoreManifest) -> ai_agents_core::Result<()> {
        if manifest.host != self.identity() || manifest.nodes.len()!=1 || manifest.nodes.values().any(|node|
            node.binding.id != self.identity().0 || node.binding.contract_version!=1 || node.binding.configuration !=
                json!({"recipe":"prepared-eval-v1","provider":"queue-v1","hooks":"provider-free-caller-owned"})) {
            return Err(ai_agents_core::AgentError::Config("incompatible controlled fixture recipe".into()));
        }
        Ok(())
    }
    async fn stage(
        &self,
        manifest: &TaskRestoreManifest,
    ) -> ai_agents_core::Result<TaskRestoreGraph> {
        self.validate_manifest(manifest)?;
        let agent =
            self.staged.lock().unwrap().take().ok_or_else(|| {
                ai_agents_core::AgentError::Config("no fresh fixture runtime".into())
            })?;
        let provider = self
            .providers
            .lock()
            .unwrap()
            .get(&(Arc::as_ptr(&agent) as usize))
            .and_then(Weak::upgrade)
            .unwrap();
        *provider.replies.lock().unwrap() =
            serde_json::from_value(manifest.nodes[&manifest.root].binding.state.clone())?;
        TaskRestoreGraph::from_root(agent)
    }
}
struct Consumer {
    directory: PathBuf,
    queue: Arc<Queue>,
    host: Arc<RestoreHost>,
}
#[async_trait]
impl crate::EvalHost for Consumer {
    fn configure(
        &self,
        builder: AgentBuilder,
        _: &crate::fixtures::AttemptFixtureContext,
    ) -> crate::Result<AgentBuilder> {
        // This explicit test supplies its own finite offline provider; no scenario provider budget is declared or replaced.
        let mut registry = ai_agents_llm::LLMRegistry::new();
        registry.register("default", self.queue.clone());
        registry.set_default("default");
        Ok(install_ticket(
            builder.llm_registry(registry),
            &self.directory,
        )?)
    }
    async fn prepare(
        &self,
        agent: Arc<RuntimeAgent>,
        _: &crate::fixtures::AttemptFixtureContext,
    ) -> crate::Result<Option<(Arc<AutonomyRunner>, TaskRunResult)>> {
        self.host.register(&agent, &self.queue);
        *self.host.staged.lock().unwrap() = Some(agent);
        let storage = Arc::new(
            ai_agents_storage::SqliteStorage::new(
                self.directory.join("tasks.sqlite").to_str().unwrap(),
            )
            .await?,
        );
        let store = Arc::new(ScopedTaskRunStore::new(
            storage,
            "PreparedEval".into(),
            None,
            "prepared-fixture.v1".into(),
        )?);
        let receipt: Value =
            serde_json::from_slice(&std::fs::read(self.directory.join("request.json"))?)?;
        let spec = ai_agents_runtime::spec::AgentSpec::from_yaml_strict(AGENT)?;
        let (runner, result) = AutonomyRunner::restore_paused(
            TaskRestoreRequest {
                run_id: receipt["id"].as_str().unwrap().into(),
                expected_revision: receipt["revision"].as_u64().unwrap(),
                config_identity: if std::env::var("AI_AGENTS_EVAL_PREPARED_CASE").as_deref()
                    == Ok("incompatible")
                {
                    "wrong-fixture.v1".into()
                } else {
                    "prepared-fixture.v1".into()
                },
                config: spec.autonomy,
                ceilings: Default::default(),
            },
            store,
            self.host.clone(),
        )
        .await?;
        assert_eq!(self.queue.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            std::fs::read_to_string(self.directory.join("ticket.txt"))?,
            "open"
        );
        assert_eq!(result.run.counters.llm_attempts, 1);
        assert_eq!(
            serde_json::to_value(result.run.clocks.expires_at)?,
            receipt["expires_at"]
        );
        Ok(Some((Arc::new(runner), result)))
    }
}

#[test]
fn prepared_process_result_is_evaluated_after_fresh_review_without_replaying_prefix() {
    for case in ["complete", "reject", "incompatible"] {
        prepared_case(case);
    }
}

// Each disposition gets an independently produced safe pause; terminal work is never copied or replayed for another case.
fn prepared_case(case: &str) {
    let directory = std::env::temp_dir().join(format!("eval-prepared-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("agent.yaml"), AGENT).unwrap();
    std::fs::write(directory.join("ticket.txt"), "open").unwrap();
    std::fs::write(directory.join("suite.yaml"),r#"
name: Prepared Support Continuation
agent: agent.yaml
settings: {retries: 0, timeout_per_scenario_ms: 20000}
fixtures: {llm: {mode: mock}}
scenarios:
  - id: prepared-reconstruction-review-is-not-the-original-answer
    steps:
      - !task
        operation: {action: inspect}
        assert: {task: {status: paused, counters: {path: llm_attempts, eq: 1}}}
      - !task
        operation: {action: restoration_review, decision: {status: approved}}
        assert: {task: {status: paused, counters: {path: llm_attempts, eq: 1}}}
      - !task
        operation: {action: resume, response: {kind: answer, value: continue}}
        assert:
          all:
            - task: {status: completed, same_run: true, counters: {path: llm_attempts, eq: 3}, checks: [{id: ticket, outcome: pass}], event_result_parity: true}
            - response_contains: verified report
            - tool_called: {id: ticket, source_in: [llm], count: 1, executed: true, success: true}
"#).unwrap();
    if case == "reject" {
        std::fs::write(directory.join("suite.yaml"), r#"
name: Rejected Prepared Support
agent: agent.yaml
settings: {retries: 0, timeout_per_scenario_ms: 20000}
fixtures: {llm: {mode: mock}}
scenarios:
  - id: fresh-review-rejection-never-resolves-the-ticket
    steps:
      - !task
        operation: {action: inspect}
        assert: {task: {status: paused, counters: {path: llm_attempts, eq: 1}}}
      - !task
        operation: {action: restoration_review, decision: {status: rejected, reason: Do not reconstruct semantic work.}}
        assert:
          all:
            - task: {status: cancelled, same_run: true, counters: {path: llm_attempts, eq: 1}}
            - tool_not_called: ticket
"#).unwrap();
    }
    for phase in ["producer", "consumer"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "prepared_tests::prepared_process_worker",
                "--ignored",
                "--nocapture",
            ])
            .env("AI_AGENTS_EVAL_PREPARED_CASE", case)
            .env("AI_AGENTS_EVAL_PREPARED_PHASE", phase)
            .env("AI_AGENTS_EVAL_PREPARED_DIRECTORY", &directory)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{phase}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let result: Value =
        serde_json::from_slice(&std::fs::read(directory.join("report/summary.json")).unwrap())
            .unwrap();
    assert_eq!(result["passed"], if case == "incompatible" { 0 } else { 1 });
    assert_eq!(
        result["metrics"]["observed_task_runs"],
        if case == "incompatible" { 0 } else { 1 }
    );
    assert_eq!(
        std::fs::read_to_string(directory.join("ticket.txt")).unwrap(),
        if case == "complete" {
            "resolved"
        } else {
            "open"
        }
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
#[ignore = "explicit producer/consumer subprocess worker"]
async fn prepared_process_worker() {
    let phase = std::env::var("AI_AGENTS_EVAL_PREPARED_PHASE").unwrap();
    let directory = PathBuf::from(std::env::var_os("AI_AGENTS_EVAL_PREPARED_DIRECTORY").unwrap());
    if phase == "producer" {
        let queue = Queue::new(&[
            "draft",
            "{\"tool\":\"ticket\",\"arguments\":{\"resolve\":true}}",
            "verified report",
        ]);
        let agent = Arc::new(
            install_ticket(
                AgentBuilder::from_yaml_file(directory.join("agent.yaml"))
                    .unwrap()
                    .llm(queue.clone()),
                &directory,
            )
            .unwrap()
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
        );
        let host = RestoreHost::new();
        host.register(&agent, &queue);
        let storage = Arc::new(
            ai_agents_storage::SqliteStorage::new(directory.join("tasks.sqlite").to_str().unwrap())
                .await
                .unwrap(),
        );
        let store = Arc::new(
            ScopedTaskRunStore::new(
                storage.clone(),
                agent.info().id,
                None,
                "prepared-fixture.v1".into(),
            )
            .unwrap(),
        );
        let spec = ai_agents_runtime::spec::AgentSpec::from_yaml_strict(AGENT).unwrap();
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            spec.autonomy,
            Default::default(),
            store,
            "prepared-fixture.v1".into(),
        )
        .unwrap();
        let paused = runner.run("prepare report", None).await.unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        assert_eq!(queue.calls.load(Ordering::SeqCst), 1);
        let prepared = runner
            .checkpoint_for_restore(&paused.run.key.run_id, paused.run.revision, host.as_ref())
            .await
            .unwrap();
        std::fs::write(directory.join("request.json"),serde_json::to_vec(&json!({"id":prepared.run.key.run_id,"revision":prepared.run.revision,"expires_at":prepared.run.clocks.expires_at})).unwrap()).unwrap();
        drop(runner);
        drop(agent);
        storage.close().await;
    } else {
        assert_eq!(phase, "consumer");
        let queue = Queue::new(&[]);
        let host = Arc::new(Consumer {
            directory: directory.clone(),
            queue: queue.clone(),
            host: RestoreHost::new(),
        });
        let runner = crate::EvalRunner::from_file(directory.join("suite.yaml"), Default::default())
            .unwrap()
            .with_host(host);
        let result = runner.run().await.unwrap();
        let details: Vec<_> = result
            .scenarios
            .iter()
            .flat_map(|s| &s.attempts)
            .flat_map(|a| &a.turns)
            .map(|t| {
                (
                    t.assertion_results
                        .iter()
                        .filter(|a| !a.passed)
                        .map(|a| a.assertion.as_str())
                        .collect::<Vec<_>>(),
                    t.evidence
                        .task
                        .as_ref()
                        .map(|e| (&e.inspection.run.status, &e.inspection.run.counters)),
                    t.evidence.tool_executions.clone(),
                )
            })
            .collect();
        let case = std::env::var("AI_AGENTS_EVAL_PREPARED_CASE").unwrap();
        assert_eq!(
            result.passed,
            if case == "incompatible" { 0 } else { 1 },
            "{details:?}"
        );
        assert_eq!(result.failed, if case == "incompatible" { 1 } else { 0 });
        assert_eq!(
            queue.calls.load(Ordering::SeqCst),
            if case == "complete" { 2 } else { 0 }
        );
        assert_eq!(
            std::fs::read_to_string(directory.join("ticket.txt")).unwrap(),
            if case == "complete" {
                "resolved"
            } else {
                "open"
            }
        );
        crate::write_outputs(&result, &directory.join("report"), true).unwrap();
    }
}
