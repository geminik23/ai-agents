//! Direct SDK consumer. Live cases use a synthetic, side-effect-free local function.
use ai_agents_llm::sdk::{FunctionChoice, responses::*};
use futures::StreamExt;
use serde_json::json;
use std::time::Duration;

#[tokio::test]
#[ignore = "explicitly authorized live OpenAI Responses smoke"]
async fn live_responses_function_round_trip() {
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY required");
    let client = ResponsesClient::new("https://api.openai.com/v1", Some(key)).unwrap();
    for model in ["gpt-6.2-sol", "gpt-6.2-astra"] {
        let mut request = ResponsesRequest::new(
            model,
            vec![
                json!({"role":"user","content":"Call get_label once. Then reply only with its returned label."}),
            ],
        );
        request.functions.push(ResponseFunction {
            strict: Some(true),
            name: "get_label".into(),
            description: "Get a synthetic label.".into(),
            parameters: json!({"type":"object","properties":{},"additionalProperties":false}),
        });
        request.tool_choice = Some(FunctionChoice::Named("get_label".into()));
        request.reasoning_effort = Some("low".into());
        request.max_output_tokens = Some(768);
        request.timeout = Some(Duration::from_secs(60));
        let first = client.complete(&request).await.unwrap();
        assert_eq!(first.tool_calls.len(), 1);
        let call = &first.tool_calls[0];
        assert_eq!(call.name, "get_label");
        assert_eq!(call.arguments, json!({}));
        let usage = first.usage.unwrap();
        request.input.extend(first.output);
        request
            .input
            .push(json!({"type":"function_call_output","call_id":call.id,"output":"maple47"}));
        request.tool_choice = Some(FunctionChoice::None);
        let mut stream = client.stream(&request).await.unwrap();
        let mut text = String::new();
        let mut finished = false;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                ResponseEvent::Text(delta) => text.push_str(&delta),
                ResponseEvent::Finished(result) => {
                    assert_eq!(result.text.trim(), "maple47");
                    assert!(result.tool_calls.is_empty());
                    let final_usage = result.usage.unwrap();
                    eprintln!(
                        "{model} two requests: input {}, output {}",
                        usage.input_tokens + final_usage.input_tokens,
                        usage.output_tokens + final_usage.output_tokens
                    );
                    finished = true;
                }
            }
        }
        assert!(finished);
        assert_eq!(text.trim(), "maple47");
    }
}
