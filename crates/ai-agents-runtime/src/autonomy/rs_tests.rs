use super::*;
#[path = "rs_receipt_tests.rs"]
mod receipts;
use ai_agents_core::ToolCall;
use ai_agents_llm::providers::{DeclaredPricing, ProviderBuilder, ProviderType};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// Real conditional stores preserve charged, in-flight and uncertain commitments independently of result eligibility.
#[tokio::test]
async fn rs_settlement_backend_matrix_retains_siblings_and_rejects_duplicate_refunds() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let execution = ledger_backend(Some(2), true, sqlite).await;
        let bound = |request: &str, maximum| ProviderCostBound {
            provider_identity: "rs.provider".into(),
            pricing_identity: "rs.price.v1".into(),
            request_id: request.into(),
            max_micro_usd: maximum,
        };
        let first_bound = bound("first", 300_000);
        let first = execution
            .admit_priced_llm(Some(first_bound.clone()))
            .await
            .unwrap();
        let unused = execution
            .admit_priced_llm(Some(bound("unused", 300_000)))
            .await
            .unwrap();
        let outstanding = execution
            .admit_priced_llm(Some(bound("outstanding", 300_000)))
            .await
            .unwrap();
        execution
            .settle_priced_llm(
                &first,
                json!({"response":"settled"}),
                Some(ProviderCostSettlement {
                    bound: first_bound.clone(),
                    charged_micro_usd: 100_000,
                }),
            )
            .await
            .unwrap();
        execution
            .settle_llm(&unused, json!({"not_invoked":true}))
            .await
            .unwrap();
        assert!(
            execution
                .settle_llm(&unused, json!({"not_invoked":true}))
                .await
                .is_err()
        );
        execution
            .settle(&outstanding, json!({"unknown":true}), true)
            .await
            .unwrap();
        let snapshot = execution.load_owned().await.unwrap();
        let payload: TaskCheckpointPayload =
            serde_json::from_value(snapshot.payload.clone()).unwrap();
        assert_eq!(payload.counters.charged_micro_usd, 100_000);
        assert_eq!(payload.counters.llm_attempts, 2);
        assert_eq!(
            payload
                .reservations
                .iter()
                .find(|r| r.id == outstanding)
                .unwrap()
                .reserved_micro_usd,
            300_000
        );
        assert_eq!(
            payload
                .reservations
                .iter()
                .find(|r| r.id == outstanding)
                .unwrap()
                .state,
            TaskEffectState::Uncertain
        );
        payload.validate(&snapshot, "ledger-v1").unwrap();
        assert!(
            execution
                .admit_priced_llm(Some(bound("cannot-route-around-uncertainty", 1)))
                .await
                .is_err()
        );
    }
}

// The raw provider consumes a real HTTP request, while all fixture output and storage remain isolated and credential-free.
async fn paid_server(
    replies: Vec<Value>,
) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let task = tokio::spawn(async move {
        for message in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let end = loop {
                let mut buffer = [0; 4096];
                let count = socket.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
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
            called.fetch_add(1, Ordering::SeqCst);
            let payload = json!({"id":"rs-fixture","choices":[{"index":0,"message":message,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}).to_string();
            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    });
    (format!("http://{address}/v1/"), calls, task)
}

struct OrdinaryQuestionHandler;
#[async_trait]
impl ai_agents_tools::QuestionHandler for OrdinaryQuestionHandler {
    async fn ask_question(
        &self,
        _: ai_agents_tools::QuestionRequest,
    ) -> ai_agents_tools::QuestionResponse {
        panic!("a retained task answer must not call the ordinary UI handler");
    }
}

// A hard-priced parent can pause on a real native question and consume only the pending work without replenishing money or targets.
#[tokio::test]
async fn rs_paid_native_question_resume_keeps_exact_usage_on_memory_and_sqlite() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let (endpoint, calls, server) = paid_server(vec![
            json!({"role":"assistant","content":"","tool_calls":[{"id":"ask","type":"function","function":{"name":"ask_user","arguments":"{\"question\":\"continue?\",\"options\":[\"yes\"]}"}}]}),
            json!({"role":"assistant","content":"done"}),
        ]).await;
        let pricing = DeclaredPricing {
            schedule_id: "rs.fixture.v1".into(),
            model: "fixture-model".into(),
            endpoint: endpoint.clone(),
            server_input_token_limit: 32,
            server_output_token_limit: 2048,
            input_micro_usd_per_million: 1_000_000,
            output_micro_usd_per_million: 1_000_000,
            request_fee_micro_usd: 0,
            max_internal_attempts: 1,
            complete_usage: true,
        };
        let provider = ProviderBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("fixture-model")
            .api_key("isolated-fixture")
            .base_url(endpoint)
            .build()
            .unwrap()
            .with_declared_pricing(pricing)
            .unwrap()
            .with_tool_choice(ai_agents_core::ToolChoice::Auto);
        let agent = Arc::new(
            crate::AgentBuilder::from_yaml(
                "name: ResourceQuestion\nsystem_prompt: work\ntools: [ask_user]\n",
            )
            .unwrap()
            .llm(Arc::new(provider))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
        );
        let store = crate::autonomy::continuation_test_support::store(
            agent.info().id,
            "rs.question.v1",
            sqlite,
        )
        .await;
        let mut config = crate::autonomy::runner_tests::config(2);
        config.defaults.max_cost_usd = Some(UsdAmount::parse("0.002100").unwrap());
        config.defaults.max_declared_write_paths = Some(1);
        config.defaults.hitl = Some(AutonomyHitlConfig {
            on_user_question: Some(InteractionAction::PauseRun),
            on_approval_required: None,
        });
        let ceilings = AutonomyHostCeilings {
            max_cost_usd: config.defaults.max_cost_usd.clone(),
            max_declared_write_paths: Some(1),
            ..Default::default()
        };
        agent.set_question_handler(Some(Arc::new(OrdinaryQuestionHandler)));
        let runner = AutonomyRunner::try_new(
            agent,
            config,
            ceilings,
            store.clone(),
            "rs.question.v1".into(),
        )
        .unwrap();
        let paused = runner.run("obtain evidence", None).await.unwrap();
        assert_eq!(
            paused.run.status,
            TaskRunStatus::Paused,
            "{:?}",
            paused.run.stop_reason
        );
        assert_eq!(paused.run.counters.charged_micro_usd, 14);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let completed = runner
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
            completed.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            completed.run.stop_reason
        );
        assert_eq!(completed.run.counters.llm_attempts, 2);
        assert_eq!(completed.run.counters.charged_micro_usd, 28);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let payload: TaskCheckpointPayload = serde_json::from_value(
            store
                .load(&completed.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .payload,
        )
        .unwrap();
        assert!(payload.declared_write_targets.is_empty());
        assert_eq!(
            payload
                .reservations
                .iter()
                .filter(|r| r.reserved_micro_usd != 0)
                .count(),
            2
        );
        server.await.unwrap();
    }
}

// Built-in coordination gets an empty local footprint, but registered hook tails remain explicitly capability-gated.
#[tokio::test]
async fn rs_footprint_framework_management_and_actual_child_execution_share_cap() {
    use crate::spawner::tools::{ListAgentsTool, SendMessageTool};
    use crate::spawner::{AgentRegistry, SpawnedAgent};
    let registry = Arc::new(AgentRegistry::new());
    let mut mock = ai_agents_llm::mock::MockLLMProvider::new("child");
    mock.set_response("child done");
    let child = crate::AgentBuilder::new()
        .system_prompt("child")
        .llm(Arc::new(mock))
        .build()
        .unwrap();
    registry
        .register(SpawnedAgent::from_runtime(
            "worker".into(),
            child,
            crate::spec::AgentSpec::default(),
        ))
        .await
        .unwrap();
    let send = SendMessageTool::new(registry.clone(), "parent");
    let list = ListAgentsTool::new(registry.clone());
    assert!(
        send.declared_write_footprint(
            &json!({"to":"worker","message":"work"}),
            &ToolExecutionContext::test("send_agent_message"),
            0
        )
        .unwrap()
        .unwrap()
        .targets
        .is_empty()
    );
    assert!(
        list.declared_write_footprint(&json!({}), &ToolExecutionContext::test("list_agents"), 0)
            .unwrap()
            .unwrap()
            .targets
            .is_empty()
    );
    let encoded = ai_agents_core::encode_native_tool_call_markers(
        &[ToolCall {
            id: "send".into(),
            name: "send_agent_message".into(),
            arguments: json!({"to":"worker","message":"work"}),
        }],
        None,
    )
    .unwrap();
    let provider = crate::autonomy::runner_tests::RecordingProvider::new(&[&encoded, "done"]);
    let parent = Arc::new(
        crate::AgentBuilder::from_yaml(
            "name: ResourceParent\nsystem_prompt: work\ntools: [send_agent_message]\n",
        )
        .unwrap()
        .llm(provider)
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(send))
        .build()
        .unwrap()
        .with_spawner_handles(Arc::new(crate::spawner::AgentSpawner::new()), registry),
    );
    let store = Arc::new(
        ScopedTaskRunStore::in_memory(parent.info().id, None, "rs.child.v1".into()).unwrap(),
    );
    let mut config = crate::autonomy::runner_tests::config(3);
    config.defaults.max_declared_write_paths = Some(1);
    let runner = AutonomyRunner::try_new(
        parent,
        config,
        AutonomyHostCeilings {
            max_declared_write_paths: Some(1),
            ..Default::default()
        },
        store.clone(),
        "rs.child.v1".into(),
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
}
