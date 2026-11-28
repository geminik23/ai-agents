use ai_agents_llm::sdk::{Function, FunctionChoice, messages::*};
use futures::StreamExt;
use serde_json::json;
use std::time::Duration;

async fn finish_stream(client: &MessagesClient, request: &MessagesRequest) -> MessageOutput {
    let mut stream = client.stream(request).await.unwrap();
    let mut text = String::new();
    let mut output = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            MessageEvent::Text(delta) => text.push_str(&delta),
            MessageEvent::Finished(result) => {
                assert!(output.is_none());
                assert_eq!(result.text, text);
                output = Some(result);
            }
        }
    }
    output.expect("missing Messages terminal event")
}

#[tokio::test]
#[ignore = "explicitly authorized live Anthropic Messages smoke"]
async fn live_messages_function_round_trip() {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY required");
    let client = MessagesClient::new(key).unwrap();
    let mut request = MessagesRequest::new(
        "claude-sonnet-4-6",
        vec![
            json!({"role":"user","content":"Use get_label exactly once. After observing its result, answer only with the returned label."}),
        ],
        1024,
    );
    request.timeout = Some(Duration::from_secs(60));
    request.thinking = Some(Thinking::Adaptive);
    request.effort = Some("low".into());
    request.functions.push(Function {
        name: "get_label".into(),
        description: "Returns the synthetic label needed for this task.".into(),
        parameters: json!({"type":"object","properties":{},"additionalProperties":false}),
    });
    request.tool_choice = Some(FunctionChoice::Auto);
    let first = finish_stream(&client, &request).await;
    assert_eq!(first.tool_calls.len(), 1);
    let call = &first.tool_calls[0];
    assert_eq!(call.name, "get_label");
    assert_eq!(call.arguments, json!({}));
    let signed = first
        .blocks
        .iter()
        .filter(|block| block["type"] == "thinking")
        .count();
    request
        .messages
        .push(json!({"role":"assistant","content":first.blocks}));
    request.messages.push(json!({"role":"user","content":[
        {"type":"tool_result","tool_use_id":call.id,"content":"maple47"}
    ]}));
    let final_output = finish_stream(&client, &request).await;
    assert!(final_output.tool_calls.is_empty());
    assert_eq!(final_output.text.trim(), "maple47");
    let (input, output, _) = first.usage.unwrap().totals().unwrap();
    let (final_input, final_tokens, _) = final_output.usage.unwrap().totals().unwrap();
    eprintln!(
        "Sonnet 4.6, 2 HTTP requests, {} input / {} output tokens, {signed} signed thinking blocks replayed",
        input + final_input,
        output + final_tokens
    );
}
