use super::*;
use ai_agents_core::TokenUsage;
use ai_agents_core::autonomy::InvocationAdmission;
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Default)]
struct Ledger {
    bounds: Mutex<Vec<ProviderCostBound>>,
    settlements: Mutex<Vec<Option<ProviderCostSettlement>>>,
    abandoned: AtomicUsize,
}
#[async_trait]
impl InvocationAdmission for Ledger {
    async fn admit_llm(&self) -> Result<String, LLMError> {
        unreachable!()
    }
    fn requires_priced_llm(&self) -> bool {
        true
    }
    async fn admit_priced_llm(&self, bound: Option<ProviderCostBound>) -> Result<String, LLMError> {
        let bound = bound.ok_or_else(|| LLMError::Config("missing bound".into()))?;
        let id = bound.request_id.clone();
        self.bounds.lock().push(bound);
        Ok(id)
    }
    async fn settle_llm(&self, _: &str, _: serde_json::Value) -> Result<(), LLMError> {
        unreachable!()
    }
    async fn settle_priced_llm(
        &self,
        _: &str,
        _: serde_json::Value,
        cost: Option<ProviderCostSettlement>,
    ) -> Result<(), LLMError> {
        self.settlements.lock().push(cost);
        Ok(())
    }
    fn abandon_llm(&self, _: &str) {
        self.abandoned.fetch_add(1, Ordering::SeqCst);
    }
    fn execution_stopped(&self) -> bool {
        false
    }
}

fn schedule(endpoint: &str) -> DeclaredPricing {
    DeclaredPricing {
        schedule_id: "fixture-price.v1".into(),
        model: "fixture-model".into(),
        endpoint: endpoint.into(),
        server_input_token_limit: 4096,
        server_output_token_limit: 2048,
        input_micro_usd_per_million: 1_000_000,
        output_micro_usd_per_million: 1_000_000,
        request_fee_micro_usd: 2,
        max_internal_attempts: 1,
        complete_usage: true,
    }
}

// Actual adapters send HTTP to an isolated loopback server; no external endpoint, credentials or paid service is used.
type Fixture = (
    String,
    tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>,
);
async fn fixture(google: bool, responses: usize, retry_first: bool) -> Fixture {
    fixture_stream(google, responses, retry_first, None).await
}

async fn fixture_stream(
    google: bool,
    responses: usize,
    retry_first: bool,
    stream_payload: Option<String>,
) -> Fixture {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        for index in 0..responses {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let end = loop {
                let mut chunk = [0; 4096];
                let count = socket.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(pos) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let header = std::str::from_utf8(&bytes[..end]).unwrap();
            let streaming = header.contains("streamGenerateContent");
            let length: usize = header
                .lines()
                .find_map(|line| {
                    let (k, v) = line.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse().unwrap())
                })
                .unwrap();
            while bytes.len() < end + length {
                let mut chunk = [0; 4096];
                let count = socket.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
            }
            let body: serde_json::Value =
                serde_json::from_slice(&bytes[end..end + length]).unwrap();
            let streaming = streaming || body["stream"] == true;
            tx.send(body).unwrap();
            let google_body = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"thoughtsTokenCount":0,"totalTokenCount":14}}"#;
            let openai_body = r#"{"id":"fixture","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"done","tool_calls":[]},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}"#;
            let failed = retry_first && index == 0;
            let payload = if failed {
                "{\"error\":{\"message\":\"retry\"}}".into()
            } else if !streaming {
                if google {
                    google_body.into()
                } else {
                    openai_body.into()
                }
            } else if google {
                stream_payload
                    .clone()
                    .unwrap_or_else(|| format!("data: {google_body}\n\n"))
            } else {
                "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n"
                    .into()
            };
            let content_type = if streaming {
                "text/event-stream"
            } else {
                "application/json"
            };
            let status = if failed {
                "500 Internal Server Error"
            } else {
                "200 OK"
            };
            let reply = format!(
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    });
    (format!("http://{address}/v1/"), rx)
}

#[tokio::test]
async fn rs_paid_actual_dispatch_matrix_keeps_completion_native_and_stream_accounting() {
    for kind in [
        ProviderType::OpenAI,
        ProviderType::OpenAICompatible,
        ProviderType::Google,
        ProviderType::Groq,
        ProviderType::Cohere,
        ProviderType::Mistral,
        ProviderType::OpenRouter,
    ] {
        let native = matches!(
            kind,
            ProviderType::OpenAI
                | ProviderType::OpenAICompatible
                | ProviderType::Google
                | ProviderType::OpenRouter
        );
        let count = if native { 3 } else { 2 };
        let (endpoint, mut received) = fixture(kind == ProviderType::Google, count, false).await;
        let provider = ProviderBuilder::new()
            .provider(kind)
            .model("fixture-model")
            .api_key("isolated-fixture-key")
            .base_url(&endpoint)
            .build()
            .unwrap()
            .with_declared_pricing(schedule(&endpoint))
            .unwrap()
            .with_feature_overrides(HashMap::from([(LLMFeature::FunctionCalling, true)]));
        let provider = crate::managed_provider(Arc::new(provider));
        let ledger = Arc::new(Ledger::default());
        crate::scope_invocation_admission(ledger.clone(), async {
            let messages = [ChatMessage::system("role"), ChatMessage::user("evidence")];
            assert_eq!(
                provider.complete(&messages, None).await.unwrap().content,
                "done",
                "{kind:?}"
            );
            let tools = LLMToolRequest {
                choice: ToolChoice::Auto,
                tools: vec![LLMToolDefinition {
                    name: "lookup".into(),
                    description: "isolated definition".into(),
                    input_schema: serde_json::json!({"type":"object","properties":{}}),
                }],
            };
            if native {
                provider
                    .complete_with_tools(&messages, None, &tools)
                    .await
                    .unwrap();
            } else {
                assert!(
                    provider
                        .complete_with_tools(&messages, None, &tools)
                        .await
                        .is_err()
                );
            }
            let mut stream = provider.complete_stream(&messages, None).await.unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
        })
        .await;
        let requests: Vec<_> = (0..count).map(|_| received.try_recv().unwrap()).collect();
        assert!(requests[0].to_string().contains("evidence"), "{kind:?}");
        if native {
            assert!(requests[1].to_string().contains("lookup"), "{kind:?}");
        }
        let bounds = ledger.bounds.lock();
        assert_eq!(bounds.len(), count, "{kind:?}");
        assert!(bounds.iter().all(|b| b.max_micro_usd == 6146));
        assert_ne!(bounds[0].request_id, bounds[1].request_id);
        let settlements = ledger.settlements.lock();
        assert_eq!(
            settlements[0].as_ref().unwrap().charged_micro_usd,
            16,
            "{kind:?}"
        );
        if native {
            assert_eq!(settlements[1].as_ref().unwrap().charged_micro_usd, 16);
        }
        if kind == ProviderType::Google {
            assert_eq!(
                settlements[count - 1].as_ref().unwrap().charged_micro_usd,
                16
            );
        } else {
            assert!(
                settlements[count - 1].is_none(),
                "stream without a complete billing receipt retains the bound"
            );
        }
        assert_eq!(ledger.abandoned.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn rs_paid_internal_retry_retains_every_attempts_bound() {
    let (endpoint, mut received) = fixture(false, 2, true).await;
    let mut pricing = schedule(&endpoint);
    pricing.max_internal_attempts = 2;
    let mut config = LLMConfig::default();
    config
        .extra
        .insert("resilient".into(), serde_json::json!(true));
    config
        .extra
        .insert("resilient_attempts".into(), serde_json::json!(2));
    config
        .extra
        .insert("resilient_base_delay_ms".into(), serde_json::json!(1));
    let provider = UnifiedLLMProvider::from_spec_config(
        ProviderType::OpenAI,
        "fixture-model",
        Some("fixture".into()),
        Some(endpoint),
        config,
    )
    .unwrap()
    .with_declared_pricing(pricing)
    .unwrap();
    let provider = crate::managed_provider(Arc::new(provider));
    let ledger = Arc::new(Ledger::default());
    crate::scope_invocation_admission(
        ledger.clone(),
        provider.complete(&[ChatMessage::user("retry")], None),
    )
    .await
    .unwrap();
    assert!(received.try_recv().is_ok());
    assert!(received.try_recv().is_ok());
    assert_eq!(ledger.bounds.lock()[0].max_micro_usd, 12292);
    assert!(ledger.settlements.lock()[0].is_none());
}

#[test]
fn rs_paid_inventory_rejects_unaudited_upstream_and_unknown_request_options() {
    for kind in [
        ProviderType::Anthropic,
        ProviderType::DeepSeek,
        ProviderType::XAI,
        ProviderType::Phind,
        ProviderType::Ollama,
    ] {
        let provider = ProviderBuilder::new()
            .provider(kind)
            .model("fixture-model")
            .api_key("fixture")
            .base_url("http://127.0.0.1:9/")
            .build()
            .unwrap();
        assert!(
            provider
                .with_declared_pricing(schedule("http://127.0.0.1:9/"))
                .is_err(),
            "{kind:?}"
        );
    }
    let provider = ProviderBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("fixture-model")
        .api_key("fixture")
        .base_url("http://127.0.0.1:9/")
        .build()
        .unwrap()
        .with_declared_pricing(schedule("http://127.0.0.1:9/"))
        .unwrap();
    for (key, value) in [
        (
            "extra_body",
            serde_json::json!({"max_completion_tokens": 99999}),
        ),
        ("openai_enable_web_search", serde_json::json!(true)),
        ("resilient", serde_json::json!(true)),
        ("reasoning_budget_tokens", serde_json::json!(u64::MAX)),
    ] {
        let mut config = LLMConfig::default();
        config.extra.insert(key.into(), value);
        let messages = [ChatMessage::user("deny")];
        assert!(
            provider
                .request_cost_bound(&ProviderRequest {
                    request_id: "request",
                    messages: &messages,
                    config: Some(&config),
                    tools: None,
                    streaming: false
                })
                .is_err(),
            "{key}"
        );
    }
}

// Only a complete final billing report at acknowledged EOF can release the stream's bound, even outside the opening scope.
#[tokio::test]
async fn rs_paid_google_final_stream_receipts_reject_early_partial_mismatched_and_dropped_usage() {
    for mode in [
        "complete",
        "missing",
        "early",
        "later-partial",
        "mismatch",
        "drop",
    ] {
        let mut final_frame = serde_json::json!({"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":14}});
        if mode == "missing" {
            final_frame["usageMetadata"]
                .as_object_mut()
                .unwrap()
                .remove("totalTokenCount");
        }
        if mode == "mismatch" {
            final_frame["usageMetadata"]["totalTokenCount"] = serde_json::json!(15);
        }
        let payload = if mode == "early" {
            let mut early = final_frame.clone();
            early["candidates"][0]
                .as_object_mut()
                .unwrap()
                .remove("finishReason");
            final_frame.as_object_mut().unwrap().remove("usageMetadata");
            format!("data: {early}\n\ndata: {final_frame}\n\n")
        } else if mode == "later-partial" {
            format!(
                "data: {final_frame}\n\ndata: {{\"usageMetadata\":{{\"totalTokenCount\":14}}}}\n\n"
            )
        } else {
            format!("data: {final_frame}\n\n")
        };
        let (endpoint, _received) = fixture_stream(true, 1, false, Some(payload)).await;
        let provider = crate::managed_provider(Arc::new(
            ProviderBuilder::new()
                .provider(ProviderType::Google)
                .model("fixture-model")
                .api_key("isolated-fixture")
                .base_url(&endpoint)
                .build()
                .unwrap()
                .with_declared_pricing(schedule(&endpoint))
                .unwrap(),
        ));
        let ledger = Arc::new(Ledger::default());
        let messages = [ChatMessage::user("stream")];
        let mut stream = crate::scope_invocation_admission(
            ledger.clone(),
            provider.complete_stream(&messages, None),
        )
        .await
        .unwrap();
        if mode == "drop" {
            assert!(!stream.next().await.unwrap().unwrap().is_final);
            drop(stream);
            assert_eq!(ledger.abandoned.load(Ordering::SeqCst), 1);
            assert!(ledger.settlements.lock().is_empty());
            continue;
        }
        let mut failed = false;
        while let Some(chunk) = stream.next().await {
            failed |= chunk.is_err();
        }
        drop(stream);
        if mode == "mismatch" {
            assert!(failed);
            assert_eq!(ledger.abandoned.load(Ordering::SeqCst), 1);
            assert!(ledger.settlements.lock().is_empty());
        } else if mode == "complete" {
            assert!(!failed);
            assert_eq!(
                ledger.settlements.lock()[0]
                    .as_ref()
                    .unwrap()
                    .charged_micro_usd,
                16
            );
        } else {
            assert!(!failed);
            assert!(ledger.settlements.lock()[0].is_none(), "{mode}");
        }
    }
}

#[test]
fn rs_paid_endpoint_binding_matches_actual_routes_and_canonical_accounting_aliases() {
    let single = "https://example.test/v1/";
    let double = "https://example.test/v1//";
    let make = |kind, endpoint: &str| {
        ProviderBuilder::new()
            .provider(kind)
            .model("fixture-model")
            .api_key("isolated-fixture")
            .base_url(endpoint)
            .build()
            .unwrap()
    };
    assert!(
        make(ProviderType::Google, double)
            .with_declared_pricing(schedule(single))
            .is_err()
    );
    let google = make(ProviderType::Google, double)
        .with_declared_pricing(schedule(double))
        .unwrap();
    assert_eq!(google.priced_endpoint().unwrap(), double);
    let openai = make(ProviderType::OpenAI, double)
        .with_declared_pricing(schedule(single))
        .unwrap();
    assert_eq!(openai.priced_endpoint().unwrap(), single);
    let compatible = make(ProviderType::OpenAICompatible, single)
        .with_declared_pricing(schedule(single))
        .unwrap();
    let messages = [ChatMessage::user("binding")];
    let request = ProviderRequest {
        request_id: "binding",
        messages: &messages,
        config: None,
        tools: None,
        streaming: false,
    };
    let first = openai.request_cost_bound(&request).unwrap().unwrap();
    let second = compatible.request_cost_bound(&request).unwrap().unwrap();
    assert_eq!(first.provider_identity, second.provider_identity);
    assert_eq!(first.pricing_identity, second.pricing_identity);
    assert_ne!(openai.provider_name(), compatible.provider_name());
    let mut changed = schedule(single);
    changed.request_fee_micro_usd += 1;
    let other = make(ProviderType::OpenAICompatible, single)
        .with_declared_pricing(changed)
        .unwrap()
        .request_cost_bound(&request)
        .unwrap()
        .unwrap();
    assert_eq!(first.provider_identity, other.provider_identity);
    assert_ne!(first.pricing_identity, other.pricing_identity);
}

#[test]
fn rs_paid_checked_rounding_identity_and_partial_usage() {
    let mut pricing = schedule("https://EXAMPLE.test/v1");
    pricing
        .validate("fixture-model", "https://example.test/v1/")
        .unwrap();
    let identity = pricing.identity("openai");
    pricing.request_fee_micro_usd += 1;
    assert_ne!(identity, pricing.identity("openai"));
    pricing.input_micro_usd_per_million = 1;
    pricing.output_micro_usd_per_million = 1;
    pricing.request_fee_micro_usd = 0;
    let config = LLMConfig::default();
    let messages = [ChatMessage::user("cost")];
    let bound = pricing
        .quote(
            "openai",
            &ProviderRequest {
                request_id: "one",
                messages: &messages,
                config: None,
                tools: None,
                streaming: false,
            },
            &config,
        )
        .unwrap();
    assert_eq!(bound.max_micro_usd, 1);
    let mut response = LLMResponse::new("ok", FinishReason::Stop);
    assert!(
        pricing
            .settle("openai", &bound, &response)
            .unwrap()
            .is_none()
    );
    response.usage = Some(TokenUsage::new(1, 1));
    assert_eq!(
        pricing
            .settle("openai", &bound, &response)
            .unwrap()
            .unwrap()
            .charged_micro_usd,
        1
    );
    response.usage = Some(TokenUsage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 1,
    });
    assert!(pricing.settle("openai", &bound, &response).is_err());
    pricing.request_fee_micro_usd = u64::MAX;
    assert!(
        pricing
            .validate("fixture-model", "https://example.test/v1/")
            .is_err()
    );
}
