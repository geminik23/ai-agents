//! Public-API resource acceptance with real first-party HTTP adapters and isolated Memory/SQLite stores.
use ai_agents_core::{AgentError, Result, ToolChoice};
use ai_agents_llm::providers::{DeclaredPricing, ProviderBuilder, ProviderType};
use ai_agents_runtime::{Agent, AgentBuilder, RuntimeAgent, autonomy::*};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn server(messages: Vec<Value>) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let handle = tokio::spawn(async move {
        for message in messages {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let end = loop {
                let mut buffer = [0; 4096];
                let count = socket.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(index) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let headers = std::str::from_utf8(&bytes[..end]).unwrap();
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .unwrap();
            while bytes.len() < end + length {
                let mut buffer = [0; 4096];
                let count = socket.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
            }
            let request: Value = serde_json::from_slice(&bytes[end..end + length]).unwrap();
            assert!(request.to_string().contains("work"));
            called.fetch_add(1, Ordering::SeqCst);
            let body=json!({"id":"resource-fixture","choices":[{"index":0,"message":message,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}).to_string();
            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    });
    (format!("http://{address}/v1/"), calls, handle)
}

fn declared_provider(endpoint: &str) -> ai_agents_llm::providers::UnifiedLLMProvider {
    let pricing = DeclaredPricing {
        schedule_id: "public.fixture.v1".into(),
        model: "fixture-model".into(),
        endpoint: endpoint.into(),
        server_input_token_limit: 32,
        server_output_token_limit: 2048,
        input_micro_usd_per_million: 1_000_000,
        output_micro_usd_per_million: 1_000_000,
        request_fee_micro_usd: 0,
        max_internal_attempts: 1,
        complete_usage: true,
    };
    ProviderBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("fixture-model")
        .api_key("public-isolated-fixture")
        .base_url(endpoint)
        .build()
        .unwrap()
        .with_declared_pricing(pricing)
        .unwrap()
        .with_tool_choice(ToolChoice::Auto)
}

fn agent(endpoint: &str, questions: bool) -> Arc<RuntimeAgent> {
    let provider = declared_provider(endpoint);
    Arc::new(
        AgentBuilder::from_yaml(if questions {
            "name: Resources\nsystem_prompt: work\ntools: [ask_user]\n"
        } else {
            "name: Resources\nsystem_prompt: work\n"
        })
        .unwrap()
        .llm(Arc::new(provider))
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap(),
    )
}

fn config(calls: u32) -> (AutonomyConfig, AutonomyHostCeilings) {
    let cost = UsdAmount::parse("0.002100").unwrap();
    (
        AutonomyConfig {
            defaults: AutonomyProfile {
                enabled: Some(true),
                max_turns: Some(4),
                max_llm_calls: Some(calls),
                max_cost_usd: Some(cost.clone()),
                max_declared_write_paths: Some(1),
                completion: Some(CompletionGate::ResponseContains("done".into())),
                hitl: Some(AutonomyHitlConfig {
                    on_user_question: Some(InteractionAction::PauseRun),
                    on_approval_required: None,
                }),
                ..Default::default()
            },
            ..Default::default()
        },
        AutonomyHostCeilings {
            max_cost_usd: Some(cost),
            max_declared_write_paths: Some(1),
            ..Default::default()
        },
    )
}

async fn store(agent: &RuntimeAgent, sqlite: bool) -> Arc<ScopedTaskRunStore> {
    #[cfg(feature = "sqlite")]
    if sqlite {
        return Arc::new(
            ScopedTaskRunStore::new(
                Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap()),
                agent.info().id,
                None,
                "public.resources.v1".into(),
            )
            .unwrap(),
        );
    }
    assert!(!sqlite);
    Arc::new(
        ScopedTaskRunStore::in_memory(agent.info().id, None, "public.resources.v1".into()).unwrap(),
    )
}
fn backends() -> &'static [bool] {
    if cfg!(feature = "sqlite") {
        &[false, true]
    } else {
        &[false]
    }
}

struct OrdinaryHandler;
#[async_trait]
impl ai_agents_tools::QuestionHandler for OrdinaryHandler {
    async fn ask_question(
        &self,
        _: ai_agents_tools::QuestionRequest,
    ) -> ai_agents_tools::QuestionResponse {
        panic!("resumed answers must not invoke the ordinary handler");
    }
}

#[tokio::test]
async fn rs_public_paid_native_question_resume_preserves_money_and_exact_calls() {
    for &sqlite in backends() {
        let (endpoint,calls,server)=server(vec![json!({"role":"assistant","content":"","tool_calls":[{"id":"ask","type":"function","function":{"name":"ask_user","arguments":"{\"question\":\"work?\",\"options\":[\"yes\"]}"}}]}),json!({"role":"assistant","content":"done"})]).await;
        let agent = agent(&endpoint, true);
        agent.set_question_handler(Some(Arc::new(OrdinaryHandler)));
        let store = store(&agent, sqlite).await;
        let (config, ceilings) = config(2);
        let runner = AutonomyRunner::try_new(
            agent,
            config,
            ceilings,
            store.clone(),
            "public.resources.v1".into(),
        )
        .unwrap();
        let paused = runner.run("work", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert_eq!(paused.run.counters.charged_micro_usd, 14);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let done = runner
            .resume(
                &paused.run.key.run_id,
                paused.run.revision,
                TaskResumeInput::UserAnswer {
                    request_id: paused.run.pending.unwrap().id,
                    answer: json!({"answered":true,"selected":["yes"]}),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(done.run.counters.llm_attempts, 2);
        assert_eq!(done.run.counters.charged_micro_usd, 28);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&done.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert!(payload.declared_write_targets.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }
}

// Public host-installed message tools retain real child admission under the parent's price and write ceilings.
#[tokio::test]
async fn rs_public_message_child_shares_resource_ledger_without_private_builder_handles() {
    use ai_agents_runtime::spawner::{AgentRegistry, SendMessageTool, SpawnedAgent};
    for &sqlite in backends() {
        let (child_endpoint, child_calls, child_server) =
            server(vec![json!({"role":"assistant","content":"child done"})]).await;
        let (parent_endpoint,parent_calls,parent_server)=server(vec![json!({"role":"assistant","content":"","tool_calls":[{"id":"send","type":"function","function":{"name":"send_agent_message","arguments":"{\"to\":\"worker\",\"message\":\"work\"}"}}]}),json!({"role":"assistant","content":"done"})]).await;
        let registry = Arc::new(AgentRegistry::new());
        let child = Arc::try_unwrap(agent(&child_endpoint, false)).unwrap();
        registry
            .register(SpawnedAgent::from_runtime(
                "worker".into(),
                child,
                Default::default(),
            ))
            .await
            .unwrap();
        let parent = Arc::new(
            AgentBuilder::from_yaml(
                "name: Parent\nsystem_prompt: work\ntools: [send_agent_message]\n",
            )
            .unwrap()
            .llm(Arc::new(declared_provider(&parent_endpoint)))
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(SendMessageTool::new(registry, "parent")))
            .build()
            .unwrap(),
        );
        let store = store(&parent, sqlite).await;
        let (mut config, mut ceilings) = config(3);
        config.defaults.max_cost_usd = Some(UsdAmount::parse("0.002200").unwrap());
        ceilings.max_cost_usd = config.defaults.max_cost_usd.clone();
        let runner = AutonomyRunner::try_new(
            parent,
            config,
            ceilings,
            store.clone(),
            "public.resources.v1".into(),
        )
        .unwrap();
        let result = runner.run("work", None).await.unwrap();
        assert_eq!(
            result.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            result.run.stop_reason
        );
        assert_eq!(result.run.counters.llm_attempts, 3);
        assert_eq!(result.run.counters.charged_micro_usd, 42);
        assert_eq!(parent_calls.load(Ordering::SeqCst), 2);
        assert_eq!(child_calls.load(Ordering::SeqCst), 1);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&result.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert_eq!(payload.children.len(), 1);
        assert!(payload.declared_write_targets.is_empty());
        parent_server.await.unwrap();
        child_server.await.unwrap();
    }
}

// A second alias/runtime for the same billing target cannot evade the already pinned schedule by changing its identity hash.
#[tokio::test]
async fn rs_public_child_cannot_replace_the_same_targets_pinned_pricing() {
    use ai_agents_runtime::spawner::{AgentRegistry, SendMessageTool, SpawnedAgent};
    let (endpoint,calls,server)=server(vec![json!({"role":"assistant","content":"","tool_calls":[{"id":"send","type":"function","function":{"name":"send_agent_message","arguments":"{\"to\":\"worker\",\"message\":\"work\"}"}}]})]).await;
    let changed = DeclaredPricing {
        schedule_id: "changed-price".into(),
        model: "fixture-model".into(),
        endpoint: endpoint.clone(),
        server_input_token_limit: 32,
        server_output_token_limit: 2048,
        input_micro_usd_per_million: 1_000_000,
        output_micro_usd_per_million: 1_000_000,
        request_fee_micro_usd: 1,
        max_internal_attempts: 1,
        complete_usage: true,
    };
    let child = AgentBuilder::new()
        .system_prompt("work")
        .llm(Arc::new(
            ProviderBuilder::new()
                .provider(ProviderType::OpenAICompatible)
                .model("fixture-model")
                .api_key("isolated-other-alias")
                .base_url(&endpoint)
                .build()
                .unwrap()
                .with_declared_pricing(changed)
                .unwrap(),
        ))
        .build()
        .unwrap();
    let registry = Arc::new(AgentRegistry::new());
    registry
        .register(SpawnedAgent::from_runtime(
            "worker".into(),
            child,
            Default::default(),
        ))
        .await
        .unwrap();
    let parent = Arc::new(
        AgentBuilder::from_yaml("name: Parent\nsystem_prompt: work\ntools: [send_agent_message]\n")
            .unwrap()
            .llm(Arc::new(declared_provider(&endpoint)))
            .auto_configure_features()
            .unwrap()
            .tool(Arc::new(SendMessageTool::new(registry, "parent")))
            .build()
            .unwrap(),
    );
    let store = store(&parent, false).await;
    let (config, ceilings) = config(3);
    let runner = AutonomyRunner::try_new(
        parent,
        config,
        ceilings,
        store.clone(),
        "public.resources.v1".into(),
    )
    .unwrap();
    let result = runner.run("work", None).await.unwrap();
    assert_ne!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.llm_attempts, 1);
    assert_eq!(result.run.counters.charged_micro_usd, 14);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let payload: TaskCheckpointPayload = serde_json::from_value(
        store
            .load(&result.run.key.run_id)
            .await
            .unwrap()
            .unwrap()
            .payload,
    )
    .unwrap();
    assert_eq!(
        payload
            .adapters
            .iter()
            .filter(|binding| binding.adapter == "runtime.priced_provider")
            .count(),
        1
    );
    server.await.unwrap();
}

struct PricedAdmission;
#[async_trait]
impl ai_agents_core::autonomy::InvocationAdmission for PricedAdmission {
    fn requires_priced_llm(&self) -> bool {
        true
    }
    async fn admit_llm(&self) -> std::result::Result<String, ai_agents_core::LLMError> {
        panic!("child admission must not call a provider");
    }
    async fn settle_llm(
        &self,
        _: &str,
        _: Value,
    ) -> std::result::Result<(), ai_agents_core::LLMError> {
        unreachable!()
    }
    fn abandon_llm(&self, _: &str) {
        unreachable!()
    }
    fn execution_stopped(&self) -> bool {
        false
    }
}

// Model-authored pricing is data, not a host capability; shared providers remain authoritative without constructing child credentials.
#[tokio::test]
async fn rs_public_generated_child_cannot_mint_a_free_pricing_contract() {
    use ai_agents_runtime::{spawner::AgentSpawner, spec::AgentSpec};
    let spec=AgentSpec::from_yaml_strict("name: Injected\nsystem_prompt: work\nllm:\n  provider: openai-compatible\n  model: injected-model\n  api_key_env: MUST_NOT_BE_READ_BY_CHILD_ADMISSION\n  base_url: https://example.invalid/v1/\n  declared_pricing:\n    schedule_id: model-claimed-free\n    model: injected-model\n    endpoint: https://example.invalid/v1/\n    server_input_token_limit: 1\n    server_output_token_limit: 1\n    input_micro_usd_per_million: 0\n    output_micro_usd_per_million: 0\n").unwrap();
    let local = AgentSpawner::new();
    local.validate_explicit_child("injected", &spec).unwrap();
    let mut registry = ai_agents_llm::LLMRegistry::new();
    registry.register(
        "default",
        Arc::new(declared_provider("http://127.0.0.1:9/v1/")),
    );
    registry.set_default("default");
    let shared = AgentSpawner::new().with_shared_llms(registry);
    ai_agents_llm::scope_invocation_admission(Arc::new(PricedAdmission), async {
        let denied = local
            .validate_explicit_child("injected", &spec)
            .unwrap_err();
        assert!(
            denied
                .to_string()
                .contains("host-installed authoritative provider bindings")
        );
        let child = Box::pin(shared.spawn_from_spec(spec)).await.unwrap();
        assert_eq!(child.agent.info().name, "Injected");
    })
    .await;
}

struct ReceiptFault {
    inner: Arc<ScopedTaskRunStore>,
    mode: &'static str,
    armed: AtomicBool,
    reached: tokio::sync::Semaphore,
}
#[async_trait]
impl TaskRunStore for ReceiptFault {
    async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(snapshot).await
    }
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, revision: u64) -> Result<()> {
        self.inner.delete(id, revision).await
    }
    async fn mutate(&self, id: &str, mutation: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        let selected = matches!(mutation,TaskRunMutation::Checkpoint {payload,..} if payload["reservations"].as_array().is_some_and(|rs|rs.iter().any(|r|r["reserved_micro_usd"]==2080 && r["charged_micro_usd"]==14)));
        if selected && self.armed.swap(false, Ordering::SeqCst) {
            if self.mode == "reject" {
                return Err(AgentError::Persistence(
                    "fixture rejected settlement".into(),
                ));
            }
            self.inner.mutate(id, mutation).await?;
            self.reached.add_permits(1);
            if self.mode == "drop" {
                std::future::pending::<()>().await;
            }
            return Err(AgentError::Persistence(
                "fixture lost settlement ACK".into(),
            ));
        }
        self.inner.mutate(id, mutation).await
    }
}

#[tokio::test]
async fn rs_public_paid_ack_faults_never_refund_or_replay_dispatched_work() {
    for &sqlite in backends() {
        for mode in ["reject", "lost", "drop"] {
            let (endpoint, calls, server) =
                server(vec![json!({"role":"assistant","content":"done"})]).await;
            let agent = agent(&endpoint, false);
            let store = store(&agent, sqlite).await;
            let fault = Arc::new(ReceiptFault {
                inner: store.clone(),
                mode,
                armed: AtomicBool::new(true),
                reached: tokio::sync::Semaphore::new(0),
            });
            let (config, ceilings) = config(1);
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                config,
                ceilings,
                fault.clone(),
                "public.resources.v1".into(),
            )
            .unwrap();
            if mode == "drop" {
                let mut work = Box::pin(runner.run("work", None));
                tokio::select! {result=&mut work=>panic!("expected held ACK: {result:?}"), permit=fault.reached.acquire()=>permit.unwrap().forget(), _=tokio::time::sleep(std::time::Duration::from_secs(15))=>panic!("fixture settlement not reached")}
                drop(work);
            } else {
                let result = runner.run("work", None).await;
                assert!(
                    !result
                        .as_ref()
                        .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
                );
            }
            assert!(!fault.armed.load(Ordering::SeqCst));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(agent.chat("foreign").await.is_err());
            assert!(runner.run("replay", None).await.is_err());
            let runs = store.list().await.unwrap();
            assert_eq!(runs.len(), 1);
            let payload: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&runs[0].key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            let paid = payload
                .reservations
                .iter()
                .find(|r| r.reserved_micro_usd != 0)
                .unwrap();
            assert_eq!(paid.reserved_micro_usd, 2080);
            assert_eq!(payload.counters.llm_attempts, 1);
            assert_eq!(
                paid.charged_micro_usd,
                if mode == "reject" { 0 } else { 14 }
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            server.await.unwrap();
        }
    }
}
