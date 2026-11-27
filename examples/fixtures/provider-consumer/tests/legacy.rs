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
            (id != "anthropic").then(|| base.clone()),
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
    let anthropic = UnifiedLLMProvider::new(
        ProviderType::Anthropic,
        "fixture".into(),
        Some("fixture-key".into()),
        None,
    )
    .unwrap();
    let default: Box<dyn llm::LLMProvider> = anthropic.build_llm(None).unwrap();
    drop(default);
    let sdk_only = LLMConfig::default().with_extra("anthropic_version", "2023-06-01".into());
    assert!(matches!(
        anthropic.build_llm(Some(&sdk_only)),
        Err(ai_agents_llm::LLMError::Config(_))
    ));
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

#[tokio::test]
async fn raw_openai_chat_keeps_model_listing_and_validates_its_effective_settings() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider = UnifiedLLMProvider::new(
        ProviderType::OpenAICompatible,
        "fixture".into(),
        None,
        Some(format!(
            "http://{}/tenant/v1/",
            listener.local_addr().unwrap()
        )),
    )
    .unwrap();
    let sdk_only = LLMConfig::default().with_extra("provider_options", "sdk extension".into());
    assert!(matches!(
        provider.build_llm(Some(&sdk_only)),
        Err(ai_agents_llm::LLMError::Config(_))
    ));
    let raw: Box<dyn llm::LLMProvider> = provider.build_llm(None).unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut lines = BufReader::new(&mut socket).lines();
        assert_eq!(
            lines.next_line().await.unwrap().as_deref(),
            Some("GET /tenant/v1/models HTTP/1.1")
        );
        let mut bearer = false;
        loop {
            let line = lines.next_line().await.unwrap().unwrap();
            if line.is_empty() {
                break;
            }
            bearer |= line.eq_ignore_ascii_case("authorization: Bearer no-key");
        }
        assert!(bearer);
        let body = r#"{"object":"list","data":[{"id":"raw-fixture","object":"model","created":1,"owned_by":"fixture"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    assert_eq!(
        raw.list_models(None).await.unwrap().get_models(),
        ["raw-fixture"]
    );
    server.await.unwrap();
}

#[tokio::test]
async fn raw_openai_structured_stream_preserves_normalize_response() {
    use tokio::io::AsyncReadExt;
    for (setting, expected) in [
        (None, vec!["{}"]),
        (Some(true), vec!["{}"]),
        (Some(false), vec!["{", "}"]),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let provider = UnifiedLLMProvider::new(
            ProviderType::OpenAICompatible,
            "fixture".into(),
            None,
            Some(format!(
                "http://{}/tenant/v1/",
                listener.local_addr().unwrap()
            )),
        )
        .unwrap();
        let mut cfg = LLMConfig::default();
        if let Some(value) = setting {
            cfg = cfg.with_extra("normalize_response", value.into());
        }
        let raw = provider.build_llm(Some(&cfg)).unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(&mut socket);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            assert_eq!(line, "POST /tenant/v1/chat/completions HTTP/1.1\r\n");
            let mut length = None;
            loop {
                line.clear();
                assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((key, value)) = line.split_once(':')
                    && key.eq_ignore_ascii_case("content-length")
                {
                    length = Some(value.trim().parse().unwrap());
                }
            }
            reader
                .read_exact(&mut vec![0_u8; length.unwrap()])
                .await
                .unwrap();
            let body = concat!(
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_fixture","type":"function","function":{"name":"lookup","arguments":"{"}}]}}]}"#,
                "\n\n",
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"}"}}]}}]}"#,
                "\n\n",
                "data: [DONE]\n\n",
            );
            reader.get_mut().write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
        });
        let messages = [llm::chat::ChatMessage::user().content("test").build()];
        let mut stream = raw.chat_stream_struct(&messages).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(item) = std::future::poll_fn(|cx| stream.as_mut().poll_next(cx)).await {
            for choice in item.unwrap().choices {
                for call in choice.delta.tool_calls.unwrap_or_default() {
                    chunks.push(call.function.arguments);
                }
            }
        }
        assert_eq!(chunks, expected);
        server.await.unwrap();
    }
}
