//! Loopback HTTP consumers of the public Chat SDK and the existing YAML framework entry.
use ai_agents_core::{Tool, ToolExecutionContext, ToolResult};
use ai_agents_hitl::{ApprovalRequest, ApprovalResult, CallbackHandler};
use ai_agents_llm::sdk::{
    AudioFormat, ChatClient, ChatRequest, Error, Event, Function, InputPart, Message,
};
use ai_agents_runtime::{Agent, AgentBuilder, AgentStreamEvent, StreamChunk};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
    sync::oneshot,
};

async fn fixture() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/tenant/v1", listener.local_addr().unwrap());
    (listener, base)
}

// Read exactly one complete HTTP request so malformed paths or missing bodies fail the test.
async fn receive(listener: &TcpListener) -> (TcpStream, Value, String) {
    receive_path(listener, "/tenant/v1/chat/completions").await
}

async fn receive_path(listener: &TcpListener, path: &str) -> (TcpStream, Value, String) {
    let (mut socket, _) = listener.accept().await.unwrap();
    let mut reader = BufReader::new(&mut socket);
    let mut head = String::new();
    let mut len = None;
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).await.unwrap() > 0);
        if line == "\r\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':')
            && k.eq_ignore_ascii_case("content-length")
        {
            len = Some(v.trim().parse().unwrap());
        }
        head.push_str(&line);
    }
    assert!(head.starts_with(&format!("POST {path} HTTP/1.1\r\n")));
    let mut body = vec![0; len.unwrap()];
    reader.read_exact(&mut body).await.unwrap();
    (
        socket,
        serde_json::from_slice(&body).unwrap(),
        head.to_ascii_lowercase(),
    )
}

async fn reply(socket: &mut TcpStream, msg: Value, reason: &str) {
    let data=json!({"choices":[{"message":msg,"finish_reason":reason}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}).to_string();
    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}",data.len()).as_bytes()).await.unwrap();
}

async fn open_stream(socket: &mut TcpStream) {
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        )
        .await
        .unwrap();
}
async fn frame(socket: &mut TcpStream, delta: Value, reason: Option<&str>) {
    let event = json!({"choices":[{"index":0,"delta":delta,"finish_reason":reason}]});
    socket
        .write_all(format!("data: {event}\r\n\r\n").as_bytes())
        .await
        .unwrap();
}

#[tokio::test]
async fn sdk_encodes_ordered_media_and_open_effort_without_agent() {
    let (listener, base) = fixture().await;
    let server = tokio::spawn(async move {
        let (mut socket, body, head) = receive(&listener).await;
        assert!(!head.contains("authorization:"));
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type":"text","text":"compare"},
                {"type":"image_url","image_url":{"url":"https://example.invalid/image.png"}},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}},
                {"type":"input_audio","input_audio":{"format":"wav","data":"BAU="}}
            ])
        );
        assert_eq!(body["reasoning_effort"], "FutureEffort");
        assert_eq!(body["stream"], false);
        assert!(body.get("max_completion_tokens").is_none());
        reply(
            &mut socket,
            json!({"role":"assistant","content":"ok"}),
            "stop",
        )
        .await;
    });
    let client = ChatClient::new(&base, None).unwrap();
    let mut request = ChatRequest::new(
        "fixture",
        vec![Message::UserParts(vec![
            InputPart::Text("compare".into()),
            InputPart::ImageUrl("https://example.invalid/image.png".into()),
            InputPart::ImageBytes {
                mime_type: "image/png".into(),
                data: vec![1, 2, 3],
            },
            InputPart::Audio {
                format: AudioFormat::Wav,
                data: vec![4, 5],
            },
        ])],
    );
    request.reasoning_effort = Some("FutureEffort".into());
    let result = client.complete(&request).await.unwrap();
    assert_eq!(result.text, "ok");
    assert_eq!(result.usage.unwrap().total_tokens, 5);
    server.await.unwrap();
}

#[tokio::test]
async fn sdk_assembles_interleaved_tool_arguments_only_at_terminal() {
    let (listener, base) = fixture().await;
    let server = tokio::spawn(async move {
        let (mut s, body, head) = receive(&listener).await;
        assert!(head.contains("authorization: bearer fixture-key"));
        assert_eq!(body["stream_options"], json!({"include_usage":true}));
        open_stream(&mut s).await;
        frame(&mut s,json!({"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"lookup","arguments":"{\"id\":"}},{"index":0,"id":"a","function":{"name":"lookup","arguments":"{\"id\":"}}]}),None).await;
        frame(&mut s,json!({"tool_calls":[{"index":0,"function":{"arguments":"1}"}},{"index":1,"function":{"arguments":"2}"}}]}),Some("tool_calls")).await;
        s.write_all(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\ndata: [DONE]\n\n").await.unwrap();
    });
    let client = ChatClient::new(&base, Some("fixture-key".into())).unwrap();
    let mut r = ChatRequest::new("fixture", vec![Message::User("lookup".into())]);
    r.functions.push(Function {
        name: "lookup".into(),
        description: "lookup".into(),
        parameters: json!({"type":"object"}),
    });
    let mut stream = client.stream(&r).await.unwrap();
    let Event::Finished {
        reason,
        tool_calls,
        usage,
    } = stream.next().await.unwrap().unwrap()
    else {
        panic!("expected complete calls")
    };
    assert_eq!(reason, "tool_calls");
    assert_eq!(
        (&tool_calls[0].id, tool_calls[0].arguments.clone()),
        (&"a".to_owned(), json!({"id":1}))
    );
    assert_eq!(
        (&tool_calls[1].id, tool_calls[1].arguments.clone()),
        (&"b".to_owned(), json!({"id":2}))
    );
    assert_eq!(usage.unwrap().total_tokens, 3);
    assert!(stream.next().await.is_none());
    server.await.unwrap();
}

#[tokio::test]
async fn conflicting_options_are_rejected_before_connecting() {
    let (listener, base) = fixture().await;
    let client = ChatClient::new(&base, None).unwrap();
    let mut r = ChatRequest::new("fixture", vec![Message::User("hi".into())]);
    r.reasoning_effort = Some("high".into());
    for key in [
        "reasoning_effort",
        "response_format",
        "messages",
        "max_tokens",
        "tools",
        "stream",
    ] {
        r.provider_options.insert(key.into(), json!("override"));
        assert!(matches!(client.complete(&r).await, Err(Error::Config(_))));
        r.provider_options.clear();
    }
    // A completed connect is visible immediately even if the fixture never accepted it.
    assert!(futures::poll!(Box::pin(listener.accept())).is_pending());
}

fn yaml(base: &str) -> String {
    format!(
        "name: Chat\nsystem_prompt: test\nllm:\n  provider: openai-compatible\n  protocol: chat-completions\n  model: fixture\n  base_url: {base}\n  reasoning_effort: NewValue\n  provider_options: {{logprobs: true}}\n"
    )
}

#[tokio::test]
async fn yaml_framework_emits_first_delta_before_eof_and_never_finalizes_truncation() {
    for truncate in [false, true] {
        let (listener, base) = fixture().await;
        let (release, wait) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut s, body, _) = receive(&listener).await;
            assert_eq!(body["reasoning_effort"], "NewValue");
            assert_eq!(body["logprobs"], true);
            assert_eq!(body["max_completion_tokens"], 2000);
            assert_eq!(body["temperature"], 0.7);
            open_stream(&mut s).await;
            frame(&mut s, json!({"content":"안녕"}), None).await;
            wait.await.unwrap();
            if !truncate {
                frame(&mut s, json!({}), Some("stop")).await;
                s.write_all(b"data: [DONE]\n\n").await.unwrap();
            }
        });
        let agent = AgentBuilder::from_yaml(&yaml(&base))
            .unwrap()
            .auto_configure_llms()
            .unwrap()
            .build()
            .unwrap();
        let mut events = agent.chat_stream_events("hello").await.unwrap();
        let mut release = Some(release);
        let mut finals = Vec::new();
        let mut errors = 0;
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while let Some(e) = events.next().await {
                match e {
                    AgentStreamEvent::Chunk(StreamChunk::Content { text }) if text == "안녕" => {
                        release.take().unwrap().send(()).unwrap();
                    }
                    AgentStreamEvent::Chunk(StreamChunk::Error { .. }) => errors += 1,
                    AgentStreamEvent::Final(r) => finals.push(r.content),
                    _ => {}
                }
            }
        })
        .await
        .expect("first delta must arrive while server waits");
        assert!(release.is_none());
        if truncate {
            assert_eq!(errors, 1);
            assert!(finals.is_empty());
        } else {
            assert_eq!(errors, 0);
            assert_eq!(finals, ["안녕"]);
        }
        server.await.unwrap();
    }
}

struct Probe(Arc<AtomicUsize>);
#[async_trait]
impl Tool for Probe {
    fn id(&self) -> &str {
        "probe"
    }
    fn name(&self) -> &str {
        "probe"
    }
    fn description(&self) -> &str {
        "Local counter"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"value":{"type":"integer"}}})
    }
    async fn execute(&self, args: Value, _: ToolExecutionContext) -> ToolResult {
        assert_eq!(args["value"], 7);
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok("recorded")
    }
}
#[tokio::test]
async fn yaml_tool_requires_review_and_replays_exact_call_id() {
    let (listener, base) = fixture().await;
    let count = Arc::new(AtomicUsize::new(0));
    let reviewed = count.clone();
    let approvals = Arc::new(AtomicUsize::new(0));
    let reviewed_calls = approvals.clone();
    let handler = CallbackHandler::new(move |_: ApprovalRequest| {
        assert_eq!(reviewed.load(Ordering::SeqCst), 0);
        reviewed_calls.fetch_add(1, Ordering::SeqCst);
        ApprovalResult::approved()
    });
    let server = tokio::spawn(async move {
        let (mut s, body, _) = receive(&listener).await;
        assert_eq!(body["tools"][0]["function"]["name"], "probe");
        reply(&mut s,json!({"role":"assistant","tool_calls":[{"id":"chosen","type":"function","function":{"name":"probe","arguments":"{\"value\":7}"}}]}),"tool_calls").await;
        drop(s);
        let (mut s, body, _) = receive(&listener).await;
        let ms = body["messages"].as_array().unwrap();
        assert!(
            ms.iter()
                .any(|m| m["role"] == "assistant" && m["tool_calls"][0]["id"] == "chosen")
        );
        assert!(ms.iter().any(|m| m["role"] == "tool"
            && m["tool_call_id"] == "chosen"
            && m["content"].as_str().unwrap().contains("recorded")));
        reply(
            &mut s,
            json!({"role":"assistant","content":"finished"}),
            "stop",
        )
        .await;
    });
    let config = format!(
        "{}  function_calling: true\n  tool_choice: auto\ntools: [probe]\nhitl:\n  tools:\n    probe: {{require_approval: true}}\n",
        yaml(&base)
    );
    let agent = AgentBuilder::from_yaml(&config)
        .unwrap()
        .auto_configure_llms()
        .unwrap()
        .auto_configure_features()
        .unwrap()
        .tool(Arc::new(Probe(count.clone())))
        .approval_handler(Arc::new(handler))
        .build()
        .unwrap();
    let result = agent.chat("please probe").await.unwrap();
    assert_eq!(result.content, "finished");
    assert_eq!(approvals.load(Ordering::SeqCst), 1);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    server.await.unwrap();
}

#[tokio::test]
async fn sdk_does_not_follow_a_redirect_with_endpoint_credentials() {
    let (source, base) = fixture().await;
    let (foreign, other) = fixture().await;
    let server = tokio::spawn(async move {
        let (mut conn, _, head) = receive(&source).await;
        assert!(head.contains("authorization: bearer only-for-source"));
        conn.write_all(format!("HTTP/1.1 307 Temporary Redirect\r\nLocation: {other}/chat/completions\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
    });
    let client = ChatClient::new(&base, Some("only-for-source".into())).unwrap();
    let request = ChatRequest::new("fixture", vec![Message::User("hello".into())]);
    assert!(matches!(
        client.complete(&request).await,
        Err(Error::Http(307))
    ));
    server.await.unwrap();
    assert!(futures::poll!(Box::pin(foreign.accept())).is_pending());
}

// Capture the provider receipt before the runtime decides whether to commit an assistant turn.
#[derive(Default)]
struct StreamReceipts(std::sync::Mutex<Vec<(ai_agents_core::FinishReason, Option<u32>)>>);

#[async_trait]
impl ai_agents_hooks::AgentHooks for StreamReceipts {
    async fn on_llm_complete(&self, response: &ai_agents_core::LLMResponse, _: u64) {
        self.0.lock().unwrap().push((
            response.finish_reason.clone(),
            response.usage.map(|u| u.total_tokens),
        ));
    }
}

#[tokio::test]
async fn review_incomplete_stream_never_commits_or_executes_in_either_path() {
    use ai_agents_core::FinishReason;
    for (reason, finish) in [
        ("length", FinishReason::Length),
        ("content_filter", FinishReason::ContentFilter),
    ] {
        for buffered in [false, true] {
            let (listener, base) = fixture().await;
            let text = r#"{"tool":"probe","arguments":{"value":7}}"#;
            let server = tokio::spawn(async move {
                let (mut socket, body, _) = receive(&listener).await;
                assert_eq!(body["stream"], true);
                open_stream(&mut socket).await;
                frame(&mut socket, json!({"content":text}), Some(reason)).await;
                socket.write_all(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n").await.unwrap();
            });
            let settings = if buffered {
                r#"
streaming:
  enabled: true
runtime:
  optimization:
    enabled: true
    max_speculative_llm_calls_per_turn: 2
    speculative_state_transitions: true
    streaming_policy: buffer_until_routing_done
    max_parallel_runtime_tasks: 2
states:
  initial: triage
  states:
    triage:
      transitions:
        - to: billing
          guard:
            context:
              route:
                eq: billing
          timing: parallel
    billing:
      prompt: Billing
"#
            } else {
                ""
            };
            let calls = Arc::new(AtomicUsize::new(0));
            let receipts = Arc::new(StreamReceipts::default());
            let config = format!("{}tools: [probe]\n{settings}", yaml(&base));
            let agent = AgentBuilder::from_yaml(&config)
                .unwrap()
                .auto_configure_llms()
                .unwrap()
                .tool(Arc::new(Probe(calls.clone())))
                .hooks(receipts.clone())
                .build()
                .unwrap();
            let events = tokio::time::timeout(std::time::Duration::from_secs(10), async {
                agent
                    .chat_stream_events("sample")
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            })
            .await
            .expect("one loopback request must settle the stream");
            assert!(events.iter().any(|event| matches!(event, AgentStreamEvent::Chunk(StreamChunk::Error { message }) if message.contains(reason))), "{reason} buffered={buffered}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, AgentStreamEvent::Final(_)))
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(events.iter().any(|event| matches!(event, AgentStreamEvent::Chunk(StreamChunk::Content { text: chunk }) if chunk == text)), !buffered);
            if !buffered {
                assert_eq!(*receipts.0.lock().unwrap(), [(finish.clone(), Some(5))]);
            }
            server.await.unwrap();
        }
    }
}

#[tokio::test]
async fn review_disabled_legacy_options_keep_the_same_chat_wire_request() {
    let (listener, base) = fixture().await;
    let server = tokio::spawn(async move {
        let (mut socket, before, _) = receive(&listener).await;
        reply(
            &mut socket,
            json!({"role":"assistant","content":"done"}),
            "stop",
        )
        .await;
        drop(socket);
        let (mut socket, after, _) = receive(&listener).await;
        assert_eq!(before, after);
        assert_eq!(after["reasoning_effort"], "NewValue");
        reply(
            &mut socket,
            json!({"role":"assistant","content":"done"}),
            "stop",
        )
        .await;
    });
    let existing_yaml = yaml(&base)
        .replace("  protocol: chat-completions\n", "")
        .replace("  provider_options: {logprobs: true}\n", "");
    for options in [
        "",
        "  resilient: false\n  resilient_attempts: 3\n  openai_enable_web_search: false\n  openai_web_search_context_size: medium\n  reasoning: false\n",
    ] {
        let agent = AgentBuilder::from_yaml(&format!("{existing_yaml}{options}"))
            .unwrap()
            .auto_configure_llms()
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(agent.chat("sample").await.unwrap().content, "done");
    }
    server.await.unwrap();
}

#[tokio::test]
async fn review_legacy_response_formats_reach_chat_wire_as_owned_fields() {
    let (listener, base) = fixture().await;
    let server = tokio::spawn(async move {
        for format in [
            json!({"type":"json_object"}),
            json!({"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","additionalProperties":false}}}),
        ] {
            let (mut socket, body, _) = receive(&listener).await;
            assert_eq!(body["response_format"], format);
            assert_eq!(body["reasoning_effort"], "NewValue");
            reply(
                &mut socket,
                json!({"role":"assistant","content":"{}"}),
                "stop",
            )
            .await;
        }
    });
    for format in [
        "{type: json_object}",
        "{type: json_schema, json_schema: {name: answer, strict: true, schema: {type: object, additionalProperties: false}}}",
    ] {
        let config = format!(
            "{}  extra_body: {{response_format: {format}}}\n",
            yaml(&base)
        );
        let agent = AgentBuilder::from_yaml(&config)
            .unwrap()
            .auto_configure_llms()
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(agent.chat("sample").await.unwrap().content, "{}");
    }
    server.await.unwrap();
}

#[tokio::test]
async fn responses_sdk_preserves_items_and_reasoning_options_for_both_models() {
    use ai_agents_llm::sdk::responses::{ResponsesClient, ResponsesRequest};
    for model in ["gpt-6.2-astra", "gpt-6.2-sol"] {
        let (listener, base) = fixture().await;
        let server = tokio::spawn(async move {
            let (mut s, body, head) = receive_path(&listener, "/tenant/v1/responses").await;
            assert!(head.contains("authorization: bearer fixture"));
            assert_eq!(body["model"], model);
            assert_eq!(
                body["reasoning"],
                json!({"mode":"standard","effort":"future-effort"})
            );
            assert_eq!(body["instructions"], "brief");
            assert_eq!(body["max_output_tokens"], 100);
            assert_eq!(body["store"], false);
            assert!(body.get("temperature").is_none());
            let raw=json!({"id":"resp","status":"completed","output":[{"type":"reasoning","id":"r","encrypted_content":"opaque","summary":[]},{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"visible"}]}]}).to_string();
            s.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{raw}",
                    raw.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        });
        let client = ResponsesClient::new(&base, Some("fixture".into())).unwrap();
        let mut r = ResponsesRequest::new(model, vec![json!({"role":"user","content":"hello"})]);
        r.reasoning_mode = Some("standard".into());
        r.reasoning_effort = Some("future-effort".into());
        r.instructions = Some("brief".into());
        r.max_output_tokens = Some(100);
        let out = client.complete(&r).await.unwrap();
        assert_eq!(out.text, "visible");
        assert_eq!(out.output[0]["encrypted_content"], "opaque");
        assert!(out.usage.is_none());
        server.await.unwrap();
    }
}

#[tokio::test]
async fn messages_compatible_yaml_stream_and_endpoint_credentials_are_explicit() {
    use ai_agents_llm::sdk::messages::{MessagesAuth, MessagesClient, MessagesRequest};
    let (listener, base) = fixture().await;
    let server = tokio::spawn(async move {
        let (mut socket, body, head) = receive_path(&listener, "/tenant/v1/messages").await;
        assert!(!head.contains("authorization:") && !head.contains("x-api-key:"));
        assert!(head.contains("anthropic-version: 2023-06-01"));
        assert!(body["system"].as_str().unwrap().starts_with("top level"));
        assert_eq!(body["max_tokens"], 80);
        assert!(body.get("temperature").is_none());
        assert!(body.get("tools").is_none());
        assert_eq!(body["stream"], true);
        let frames = [
            json!({"type":"message_start","message":{"id":"msg","role":"assistant","content":[],"usage":{"input_tokens":2}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ready"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
            json!({"type":"message_stop"}),
        ];
        open_stream(&mut socket).await;
        for event in frames {
            socket
                .write_all(format!("data: {event}\n\n").as_bytes())
                .await
                .unwrap();
        }
        drop(socket);
        let (mut socket, body, head) = receive_path(&listener, "/tenant/v1/messages").await;
        assert!(!head.contains("authorization:") && !head.contains("x-api-key:"));
        assert_eq!(body["stream"], false);
        assert!(body.get("tools").is_none());
        assert!(body["system"].as_str().unwrap().contains("echo"));
        let payload = json!({"id":"prompt-mode","role":"assistant","content":[{"type":"text","text":"prompt-mode"}],"stop_reason":"end_turn"}).to_string();
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        drop(socket);
        let (mut socket, _, head) = receive_path(&listener, "/tenant/v1/messages").await;
        assert!(head.contains("authorization: bearer local-fixture"));
        assert!(head.contains("anthropic-version: fixture-version"));
        let payload = json!({"id":"m","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}).to_string();
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    let source = format!(
        "name: Compatible\nsystem_prompt: top level\nllm:\n  provider: anthropic-compatible\n  model: local-fixture\n  base_url: {base}\n  max_tokens: 80\n"
    );
    let agent = AgentBuilder::from_yaml(&source)
        .unwrap()
        .auto_configure_llms()
        .unwrap()
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    let mut events = agent.chat_stream_events("hello").await.unwrap();
    let mut completed = false;
    while let Some(event) = events.next().await {
        match event {
            AgentStreamEvent::Final(result) => {
                assert_eq!(result.content, "ready");
                completed = true;
            }
            AgentStreamEvent::Chunk(StreamChunk::Error { message }) => panic!("{message}"),
            _ => {}
        }
    }
    assert!(completed);
    // An explicit tool choice uses the buffered framework path. Compatible tool support is opt-in.
    let fallback = format!("{source}  tool_choice: auto\ntools: [echo]\n");
    let agent = AgentBuilder::from_yaml(&fallback)
        .unwrap()
        .auto_configure_llms()
        .unwrap()
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(agent.chat("hello").await.unwrap().content, "prompt-mode");

    let client = MessagesClient::compatible(
        &base,
        MessagesAuth::Bearer("local-fixture".into()),
        "fixture-version",
    )
    .unwrap();
    let mut request = MessagesRequest::new(
        "local-fixture",
        vec![json!({"role":"user","content":"hi"})],
        5,
    );
    request.timeout = Some(std::time::Duration::from_secs(5));
    assert_eq!(client.complete(&request).await.unwrap().text, "ok");
    server.await.unwrap();
}

#[tokio::test]
async fn messages_custom_origin_and_missing_limit_fail_before_dispatch() {
    let (listener, base) = fixture().await;
    let source = format!(
        "name: Official\nsystem_prompt: hi\nllm:\n  provider: anthropic\n  protocol: messages\n  model: claude-sonnet-4-6\n  base_url: {base}\n  max_tokens: 2048\n"
    );
    let result = AgentBuilder::from_yaml(&source)
        .unwrap()
        .auto_configure_llms();
    assert!(
        matches!(result, Err(error) if error.to_string().contains("explicitly selected credentials"))
    );
    let compatible = "name: Compatible\nsystem_prompt: hi\nllm:\n  provider: anthropic-compatible\n  model: fixture\n";
    let result = AgentBuilder::from_yaml(compatible)
        .unwrap()
        .auto_configure_llms();
    assert!(matches!(result, Err(error) if error.to_string().contains("max_tokens")));
    assert!(futures::poll!(Box::pin(listener.accept())).is_pending());
}

#[tokio::test]
async fn messages_sdk_rejects_cross_origin_redirect_with_api_key() {
    use ai_agents_llm::sdk::messages::{
        API_VERSION, MessagesAuth, MessagesClient, MessagesRequest,
    };
    let (listener, base) = fixture().await;
    let (foreign, target) = fixture().await;
    let server = tokio::spawn(async move {
        let (mut socket, _, head) = receive_path(&listener, "/tenant/v1/messages").await;
        assert!(head.contains("x-api-key: endpoint-only"));
        socket.write_all(format!("HTTP/1.1 307 Temporary Redirect\r\nLocation: {target}/messages\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
    });
    let client = MessagesClient::compatible(
        &base,
        MessagesAuth::ApiKey("endpoint-only".into()),
        API_VERSION,
    )
    .unwrap();
    let request = MessagesRequest::new("fixture", vec![json!({"role":"user","content":"hi"})], 10);
    assert!(matches!(
        client.complete(&request).await,
        Err(Error::Http(307))
    ));
    server.await.unwrap();
    assert!(futures::poll!(Box::pin(foreign.accept())).is_pending());
}
