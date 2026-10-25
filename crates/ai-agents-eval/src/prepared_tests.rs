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

const AGENT: &str = "name: PreparedEval\nsystem_prompt: Produce a verified report.\ntools: []\nautonomy:\n  enabled: true\n  max_turns: 4\n  max_llm_calls: 4\n  max_active_time_seconds: 60\n  max_wall_time_seconds: 3600\n  completion: {response_contains: verified report}\n  premature_finish: {action: ask_user, on_unavailable: stop}\n";
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
        Ok(builder.llm_registry(registry))
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
                config_identity: "prepared-fixture.v1".into(),
                config: spec.autonomy,
                ceilings: Default::default(),
            },
            store,
            self.host.clone(),
        )
        .await?;
        assert_eq!(self.queue.calls.load(Ordering::SeqCst), 0);
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
    let directory = std::env::temp_dir().join(format!("eval-prepared-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("agent.yaml"), AGENT).unwrap();
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
            - task: {status: completed, same_run: true, counters: {path: llm_attempts, eq: 2}, event_result_parity: true}
            - response_contains: verified report
"#).unwrap();
    for phase in ["producer", "consumer"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "prepared_tests::prepared_process_worker",
                "--ignored",
                "--nocapture",
            ])
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
    assert_eq!(result["passed"], 1);
    assert_eq!(result["metrics"]["observed_task_runs"], 1);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
#[ignore = "explicit producer/consumer subprocess worker"]
async fn prepared_process_worker() {
    let phase = std::env::var("AI_AGENTS_EVAL_PREPARED_PHASE").unwrap();
    let directory = PathBuf::from(std::env::var_os("AI_AGENTS_EVAL_PREPARED_DIRECTORY").unwrap());
    if phase == "producer" {
        let queue = Queue::new(&["draft", "verified report"]);
        let agent = Arc::new(
            AgentBuilder::from_yaml_file(directory.join("agent.yaml"))
                .unwrap()
                .llm(queue.clone())
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
        assert_eq!(result.passed, 1, "{:#?}", result.scenarios);
        assert_eq!(queue.calls.load(Ordering::SeqCst), 1);
        crate::write_outputs(&result, &directory.join("report"), true).unwrap();
    }
}
