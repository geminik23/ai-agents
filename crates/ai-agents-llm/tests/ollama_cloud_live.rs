//! Opt-in text and streaming verification of the official native chat endpoint.

use ai_agents_llm::sdk::ollama::{
    ChatEvent, ChatRequest, Finish, Message, OllamaCloudClient, Think,
};
use futures::StreamExt;
use serde_json::json;
use std::time::Duration;

#[tokio::test]
#[ignore = "sends two authorized inference requests to Ollama Cloud"]
async fn live_ollama_cloud_text_and_stream() {
    let client = OllamaCloudClient::new(
        std::env::var("OLLAMA_API_KEY").expect("Ollama credential required"),
    )
    .unwrap();
    let model = std::env::var("OLLAMA_CLOUD_SDK_MODEL").expect("evaluation model required");
    let mut request = ChatRequest::new(
        model,
        vec![Message::text(
            "user",
            "Reply in one short English sentence. What color is a clear daytime sky?",
        )],
    );
    request.timeout = Some(Duration::from_secs(90));
    request.think = Some(Think::Enabled(false));
    request.options.insert("num_predict".into(), json!(1024));
    let output = client.complete(&request).await.unwrap();
    assert_eq!(output.finish, Finish::Stop);
    assert!(!output.message.content.trim().is_empty());
    let one = output.usage.expect("non-streamed usage required");
    println!(
        "SDK chat: input {}, output {}, stop {}",
        one.prompt_tokens, one.completion_tokens, output.done_reason
    );
    assert!(one.prompt_tokens < 8192 && one.completion_tokens <= 1024);

    let mut events = client.stream(&request).await.unwrap();
    let mut text = String::new();
    let mut terminal = None;
    while let Some(event) = events.next().await {
        match event.unwrap() {
            ChatEvent::Text(part) => text.push_str(&part),
            ChatEvent::Finished(output) => {
                assert!(terminal.replace(output).is_none());
            }
            _ => panic!("unexpected Ollama event"),
        }
    }
    let output = terminal.expect("a complete stream");
    assert_eq!(output.finish, Finish::Stop);
    assert!(!text.trim().is_empty());
    assert_eq!(text, output.message.content);
    let two = output.usage.expect("streamed usage required");
    println!(
        "SDK stream: input {}, output {}, stop {}",
        two.prompt_tokens, two.completion_tokens, output.done_reason
    );
    assert!(one.prompt_tokens + two.prompt_tokens <= 8192 && two.completion_tokens <= 1024);
}
