//! Opt-in official OpenRouter round trip with two generation requests and synthetic tool data.
use ai_agents_llm::sdk::openrouter::{OpenRouterClient, Request, StreamEvent};
use ai_agents_llm::sdk::{ChatRequest, Function, FunctionChoice, Message};
use futures::StreamExt;
use serde_json::json;

#[tokio::test]
#[ignore = "requires approval for two billed requests and an OpenRouter credential"]
async fn official_openrouter_replays_streamed_reasoning() {
    let key = std::env::var("OPENROUTER_API_KEY").expect("OpenRouter credential required");
    let model = std::env::var("OPENROUTER_LIVE_MODEL").expect("approved model required");
    let route = std::env::var("OPENROUTER_LIVE_ROUTE").expect("approved route required");
    let client = OpenRouterClient::new(key).unwrap();
    let mut chat = ChatRequest::new(
        model,
        vec![Message::User("Call read_fixture exactly once with id inventory. When you receive the result, reply with its value and nothing else.".into())],
    );
    chat.functions.push(Function {
        name: "read_fixture".into(),
        description: "Read the requested synthetic fixture".into(),
        parameters: json!({"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}),
    });
    chat.tool_choice = Some(FunctionChoice::Auto);
    chat.max_completion_tokens = Some(2048);
    chat.provider_options.insert(
        "provider".into(),
        json!({
            "only":[route], "allow_fallbacks":false, "require_parameters":true,
            "max_price":{"prompt":3,"completion":10,"request":0}
        }),
    );
    chat.provider_options
        .insert("reasoning".into(), json!({"effort":"low"}));
    let mut request = Request::new(chat);
    let mut stream = client.stream(&request).await.unwrap();
    let first = loop {
        match stream.next().await.expect("expected final event").unwrap() {
            StreamEvent::Text(_) => {}
            StreamEvent::Finished(output) => break output,
        }
    };
    assert!(stream.next().await.is_none());
    assert!(first.assistant.has_reasoning());
    let [call] = first.tool_calls.as_slice() else {
        panic!("expected one tool call")
    };
    assert!(call.name == "read_fixture" && call.arguments == json!({"id":"inventory"}));
    request.push_assistant(&first.assistant).unwrap();
    request.chat.messages.push(Message::ToolResult {
        id: call.id.clone(),
        content: "inventory-fixture-773".into(),
    });
    request.chat.tool_choice = Some(FunctionChoice::None);
    let second = client.complete(&request).await.unwrap();
    assert!(second.text.contains("inventory-fixture-773"));
    assert!(second.tool_calls.is_empty());
    for output in [*first, second] {
        let usage = output.usage.expect("expected full usage");
        println!(
            "input={} output={} cost={}",
            usage.prompt_tokens,
            usage.completion_tokens,
            output.cost.unwrap_or_default()
        );
    }
}
