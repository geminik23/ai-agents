use super::*;
use futures::{StreamExt, stream};

fn request_with_tool() -> ChatRequest {
    let mut request = ChatRequest::new("chosen-model:tag", vec![Message::text("user", "input")]);
    request.functions.push(Function {
        name: "test_tool".into(),
        description: "fixture operation".into(),
        parameters: json!({"type":"object"}),
    });
    request
}

#[tokio::test]
async fn ollama_ndjson_preserves_split_utf8_terminal_usage_and_single_call() {
    let records = concat!(
        "{\"message\":{\"role\":\"assistant\",\"content\":\"한\",\"thinking\":\"private\"},\"done\":false}\r\n",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"글\",\"tool_calls\":[{\"function\":{\"index\":0,\"name\":\"test_tool\",\"arguments\":{\"x\":1}}}]},\"done\":false}\n",
        "{\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":12,\"eval_count\":4}"
    );
    let byte_split = records.find('한').unwrap() + 1;
    let mut events = decode(
        stream::iter([
            Ok::<_, ()>(&records.as_bytes()[..byte_split]),
            Ok(&records.as_bytes()[byte_split..]),
        ]),
        Assembly::new(&request_with_tool(), FinishPolicy::Require),
        Ok,
    );
    assert!(matches!(events.next().await, Some(Ok(LocalChatEvent::Text(text))) if text == "한"));
    assert!(matches!(events.next().await, Some(Ok(LocalChatEvent::Text(text))) if text == "글"));
    let Some(Ok(LocalChatEvent::Finished(output))) = events.next().await else {
        panic!("missing verified finish");
    };
    assert_eq!(output.finish, Some(Finish::ToolCall));
    assert_eq!(output.message.content, "한글");
    assert_eq!(output.message.thinking.as_deref(), Some("private"));
    assert_eq!(output.message.tool_calls[0].id, None);
    assert_eq!(output.usage.unwrap().total_tokens, 16);
    assert!(events.next().await.is_none());
}

#[tokio::test]
async fn ollama_ndjson_rejects_error_eof_bad_tools_and_post_terminal_data() {
    let pending = "{\"message\":{\"role\":\"assistant\",\"content\":\"visible\"},\"done\":false}\n";
    let terminal = "{\"done\":true,\"done_reason\":\"stop\"}\n";
    for bad in [
        pending.to_owned(),
        format!("{pending}{{\"error\":\"fixture failure\"}}"),
        format!("{pending}{terminal}{pending}"),
        json!({"done":true, "done_reason":"stop","message":{"role":"assistant","tool_calls":[
            {"function":{"name":"test_tool","arguments":{} }},{"function":{"name":"test_tool","arguments":{} }}
        ]}}).to_string(),
        json!({"done":true, "done_reason":"stop","message":{"role":"assistant","tool_calls":[
            {"function":{"name":"outside_grant","arguments":{} }}
        ]}}).to_string(),
        json!({"done":true, "done_reason":"stop","message":{"role":"assistant","tool_calls":[
            {"function":{"name":"test_tool","arguments":"not an object" }}
        ]}}).to_string(),
    ] {
        let events = decode(stream::iter([Ok::<_, ()>(bad)]), Assembly::new(&request_with_tool(), FinishPolicy::Require), Ok)
            .collect::<Vec<_>>().await;
        assert!(matches!(events.last(), Some(Err(Error::Protocol(_)))));
        assert!(events.iter().all(|event| !matches!(event, Ok(LocalChatEvent::Finished(_)))));
    }
}

#[tokio::test]
async fn ollama_stream_distinguishes_length_and_never_estimates_missing_usage() {
    let reply = json!({
        "done":true,"done_reason":"length","message":{"role":"assistant","content":"partial"},
        "eval_count":2
    })
    .to_string();
    let mut events = decode(
        stream::iter([Ok::<_, ()>(reply)]),
        Assembly::new(&request_with_tool(), FinishPolicy::Require),
        Ok,
    );
    assert!(matches!(
        events.next().await,
        Some(Ok(LocalChatEvent::Text(_)))
    ));
    let Some(Ok(LocalChatEvent::Finished(output))) = events.next().await else {
        panic!("missing verified finish");
    };
    assert_eq!(output.finish, Some(Finish::Length));
    assert!(output.usage.is_none());
}

#[test]
fn ollama_preserves_option_omission_and_passes_open_thinking_levels() {
    let mut request = ChatRequest::new("chosen-model:tag", vec![Message::text("user", "input")]);
    let default = request.body(false).unwrap();
    assert_eq!(default["model"], "chosen-model:tag");
    assert_eq!(default["stream"], false);
    assert!(default.get("options").is_none());
    assert!(default.get("think").is_none());
    assert!(default.get("tools").is_none());
    request.think = Some(Think::Level("future-level".into()));
    request.options.insert("num_predict".into(), json!(64));
    let request = request.body(true).unwrap();
    assert_eq!(request["think"], "future-level");
    assert_eq!(request["options"]["num_predict"], 64);
    assert_eq!(request["stream"], true);
}

#[tokio::test]
async fn ollama_legacy_local_terminal_without_reason_is_not_a_cloud_success() {
    let reply = json!({"model":"fixture:tag", "done":true,
        "message":{"role":"assistant", "content":"text"}});
    let mut events = decode(
        stream::iter([Ok::<_, ()>(reply.to_string())]),
        Assembly::new(&request_with_tool(), FinishPolicy::AllowOmitted),
        Ok,
    );
    assert!(matches!(
        events.next().await,
        Some(Ok(LocalChatEvent::Text(_)))
    ));
    let Some(Ok(LocalChatEvent::Finished(output))) = events.next().await else {
        panic!("missing local completion");
    };
    assert_eq!(output.model.as_deref(), Some("fixture:tag"));
    assert!(output.done_reason.is_none() && output.finish.is_none());
    let mut cloud = decode(
        stream::iter([Ok::<_, ()>(reply.to_string())]),
        Assembly::new(&request_with_tool(), FinishPolicy::Require),
        Ok,
    );
    assert!(matches!(cloud.next().await, Some(Err(Error::Protocol(_)))));
    assert!(cloud.next().await.is_none());
}

#[tokio::test]
async fn ollama_drop_releases_a_partial_stream_without_a_final_record() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Witness(Arc<AtomicBool>);
    impl Drop for Witness {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let dropped = Arc::new(AtomicBool::new(false));
    let witness = Witness(dropped.clone());
    let bytes = Box::pin(async_stream::stream! {
        let _witness = witness;
        yield Ok::<_, ()>(b"{\"message\":{\"role\":\"assistant\",\"content\":\"partial\"},\"done\":false}\n");
        std::future::pending::<()>().await;
    });
    let mut events = decode(
        bytes,
        Assembly::new(&request_with_tool(), FinishPolicy::Require),
        Ok,
    );
    assert!(matches!(
        events.next().await,
        Some(Ok(LocalChatEvent::Text(_)))
    ));
    assert!(!dropped.load(Ordering::SeqCst));
    drop(events);
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn ollama_local_native_option_types_and_extensions() {
    let mut request = LocalChatRequest::new(ChatRequest::new("fixture", Vec::new()));
    for (name, invalid) in [
        ("temperature", json!({"value": 1})),
        ("top_p", json!("0.9")),
        ("top_k", json!(0.5)),
        ("num_predict", json!(true)),
        ("num_ctx", json!("4096")),
        ("num_gpu", json!([1])),
        ("stop", json!(["end", 1])),
        ("seed", json!("1")),
    ] {
        request.chat.options.insert(name.into(), invalid);
        for stream in [false, true] {
            assert!(
                matches!(request.body(stream), Err(Error::Config(msg)) if msg.contains(name)),
                "{name}"
            );
        }
        request.chat.options.clear();
    }
    request.chat.options = json!({"num_predict": -1, "num_gpu": -1, "stop": ["end"],
        "top_p": null, "top_k": null, "future_option": {"level": "custom"}})
    .as_object()
    .unwrap()
    .clone();
    for stream in [false, true] {
        assert_eq!(
            request.body(stream).unwrap()["options"],
            json!(request.chat.options)
        );
    }
}
