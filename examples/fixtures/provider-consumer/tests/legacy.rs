//! Downstream callers can name the upstream trait and use capabilities outside framework chat.
use ai_agents_llm::LLMConfig;
use ai_agents_llm::providers::{ProviderType, UnifiedLLMProvider};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

#[test]
fn existing_presets_expose_their_documented_low_level_builder() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    for (id, upstream) in [
        ("openai", true),
        ("openai-compatible", true),
        ("anthropic", true),
        ("anthropic-compatible", false),
        ("google", true),
        ("ollama", true),
        ("ollama-cloud", false),
        ("deepseek", true),
        ("xai", true),
        ("mistral", true),
        ("openrouter", true),
        ("groq", true),
        ("cohere", true),
        ("phind", true),
    ] {
        let provider = UnifiedLLMProvider::from_spec_config(
            id.parse().unwrap(),
            "fixture",
            Some("fixture-key".into()),
            Some(base.clone()),
            LLMConfig {
                temperature: None,
                top_p: None,
                ..Default::default()
            },
        )
        .unwrap();
        let raw: Result<Box<dyn llm::LLMProvider>, _> = provider.build_llm(None);
        if upstream {
            assert!(raw.is_ok(), "{id}: {:?}", raw.err());
        } else {
            assert!(
                matches!(raw, Err(ai_agents_llm::LLMError::Config(_))),
                "{id}"
            );
        }
    }
    for (preset, protocol, limit) in [
        (ProviderType::OpenAI, "responses", None),
        (ProviderType::OpenAI, "decisions", None),
        (ProviderType::Anthropic, "messages", Some(64)),
    ] {
        let config = LLMConfig {
            temperature: None,
            top_p: None,
            max_tokens: limit,
            ..Default::default()
        }
        .with_extra("protocol", protocol.into());
        let provider = UnifiedLLMProvider::from_spec_config(
            preset,
            "fixture",
            Some("fixture-key".into()),
            Some(base.clone()),
            config,
        )
        .unwrap();
        assert!(
            matches!(
                provider.build_llm(None),
                Err(ai_agents_llm::LLMError::Config(_))
            ),
            "{protocol}"
        );
    }
}

#[tokio::test]
async fn low_level_ollama_caller_can_list_models_without_an_agent() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider = UnifiedLLMProvider::new(
        ProviderType::Ollama,
        "fixture".into(),
        None,
        Some(format!("http://{}", listener.local_addr().unwrap())),
    )
    .unwrap();
    let raw: Box<dyn llm::LLMProvider> = provider.build_llm(None).unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut lines = BufReader::new(&mut socket).lines();
        assert_eq!(
            lines.next_line().await.unwrap().as_deref(),
            Some("GET /api/tags HTTP/1.1")
        );
        while !lines.next_line().await.unwrap().unwrap().is_empty() {}
        let body = r#"{"models":[{"name":"local-fixture"}]}"#;
        let reply = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(reply.as_bytes()).await.unwrap();
    });
    let models = raw.list_models(None).await.unwrap();
    assert_eq!(models.get_models(), ["local-fixture"]);
    server.await.unwrap();
}
