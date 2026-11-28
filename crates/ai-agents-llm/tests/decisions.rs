use ai_agents_llm::sdk::{
    Error,
    decisions::{ChoiceOption, ChoiceOutcome, ChoiceRequest, DecisionClient},
};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::TcpListener,
};

#[tokio::test]
async fn decisions_sdk_validates_complete_correlated_answers_over_http() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = DecisionClient::new(
        &format!("http://{}/v1", listener.local_addr().unwrap()),
        "fixture-token".into(),
    )
    .unwrap();
    let options = [
        ChoiceOption {
            id: "ops",
            description: Some("Office operations"),
        },
        ChoiceOption {
            id: "devops",
            description: Some("Deployment and builds"),
        },
    ];
    let mut request = ChoiceRequest {
        model: "fixture-decision-model",
        input: "The build cannot deploy.",
        question: "Which team should investigate?",
        options: &options[..1],
        timeout: Some(Duration::from_secs(5)),
    };
    assert!(matches!(
        client.choose(&request).await,
        Err(Error::Config(_))
    ));
    request.options = &options;

    let selected = json!({"type":"choice","name":"selection","choice":"devops","confidence":0.93});
    let replies = [
        json!([selected]),
        json!([{"type":"refusal","name":"selection"}]),
        json!([]),
        json!([{"type":"choice","name":"another_question","choice":"devops"}]),
        json!([{"type":"choice","name":"selection","choice":"devops or ops"}]),
        json!([{"type":"choice","name":"selection","choice":true}]),
        json!([selected, selected]),
    ];
    let server = tokio::spawn(async move {
        for answers in replies {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = BufReader::new(socket);
            let mut line = String::new();
            socket.read_line(&mut line).await.unwrap();
            assert_eq!(line.trim_end(), "POST /v1/decisions HTTP/1.1");
            let mut length = None;
            let mut bearer = false;
            loop {
                line.clear();
                assert_ne!(socket.read_line(&mut line).await.unwrap(), 0);
                let header = line.trim_end();
                if header.is_empty() {
                    break;
                }
                let (key, value) = header.split_once(':').unwrap();
                if key.eq_ignore_ascii_case("content-length") {
                    length = Some(value.trim().parse().unwrap());
                }
                if key.eq_ignore_ascii_case("authorization") {
                    bearer = value.trim() == "Bearer fixture-token";
                }
            }
            assert!(bearer);
            let mut body = vec![0; length.unwrap()];
            socket.read_exact(&mut body).await.unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body,
                json!({
                    "model":"fixture-decision-model",
                    "input":"The build cannot deploy.",
                    "questions":[{"type":"choice","name":"selection",
                        "instructions":"Which team should investigate?",
                        "choices":[{"value":"ops","description":"Office operations"},
                                   {"value":"devops","description":"Deployment and builds"}]}]
                })
            );
            let response = json!({"answers":answers,"model":"fixture-decision-model",
                                  "usage":{"input_tokens":19,"output_tokens":1,"total_tokens":20}});
            let response = response.to_string();
            socket.get_mut().write_all(format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{response}",
                response.len()
            ).as_bytes()).await.unwrap();
        }
    });
    let picked = client.choose(&request).await.unwrap();
    assert_eq!(
        picked.outcome,
        ChoiceOutcome::Selected {
            id: "devops".into(),
            confidence: Some(0.93)
        }
    );
    let refusal = client.choose(&request).await.unwrap();
    assert_eq!(refusal.outcome, ChoiceOutcome::Refused);
    assert_eq!(refusal.usage.unwrap().total_tokens, 20);
    for _ in 0..5 {
        assert!(matches!(
            client.choose(&request).await,
            Err(Error::Protocol(_))
        ));
    }
    server.await.unwrap();
}
