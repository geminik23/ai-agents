use super::*;
use serde_json::json;
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpListener,
    task::JoinHandle,
};

struct Exchange {
    path: &'static str,
    headers: &'static [(&'static str, &'static str)],
    response: Reply,
}

enum Reply {
    Json(Value),
    Status(u16),
    Redirect(String),
}

async fn serve(exchanges: Vec<Exchange>) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        for exchange in exchanges {
            let (socket, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
                .await
                .unwrap()
                .unwrap();
            let mut socket = BufReader::new(socket);
            let mut line = String::new();
            socket.read_line(&mut line).await.unwrap();
            assert_eq!(line.trim_end(), format!("GET {} HTTP/1.1", exchange.path));
            let mut headers = std::collections::HashMap::new();
            loop {
                line.clear();
                assert_ne!(socket.read_line(&mut line).await.unwrap(), 0);
                if line.trim_end().is_empty() {
                    break;
                }
                let (name, value) = line.split_once(':').unwrap();
                headers.insert(name.to_ascii_lowercase(), value.trim().to_owned());
            }
            assert!(!headers.contains_key("transfer-encoding"));
            assert!(headers.get("content-length").is_none_or(|len| len == "0"));
            for (name, value) in exchange.headers {
                assert_eq!(headers.get(*name).map(String::as_str), Some(*value));
            }
            if !exchange
                .headers
                .iter()
                .any(|(name, _)| *name == "authorization")
            {
                assert!(!headers.contains_key("authorization"));
            }
            let (status, extra, body) = match exchange.response {
                Reply::Json(value) => (200, String::new(), value.to_string()),
                Reply::Status(status) => (status, String::new(), String::new()),
                Reply::Redirect(location) => {
                    (302, format!("Location: {location}\r\n"), String::new())
                }
            };
            socket.get_mut().write_all(format!(
                "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n{body}",
                body.len()
            ).as_bytes()).await.unwrap();
        }
    });
    (url, task)
}

const BEARER: &[(&str, &str)] = &[("authorization", "Bearer test-key")];
const ANTHROPIC: &[(&str, &str)] = &[
    ("authorization", "Bearer test-key"),
    ("anthropic-version", "2026-10-21"),
    ("anthropic-workspace-id", "wrkspc_fixture"),
    ("content-type", "application/json"),
];

fn anthropic(path: &'static str, response: Reply) -> Exchange {
    Exchange {
        path,
        headers: ANTHROPIC,
        response,
    }
}

#[tokio::test]
async fn openai_models_preserve_reported_values_and_empty_lists() {
    let (base, server) = serve([
        json!({"data":[{"id":"exact-openai-id","shutdown_date":null,"owned_by":"fixture","future":{"enabled":true}},{"id":"another-id"}]}),
        json!({"data":[]}),
        json!({"unexpected":[]}),
        json!({"data":[{"id":" \t " }]}),
        json!({"data":[{"owned_by":"fixture"}]}),
    ].into_iter().map(|body| Exchange {
        path: "/v1/models", headers: BEARER, response: Reply::Json(body),
    }).collect()).await;
    let catalog = OpenAiCatalog::at_base(&format!("{base}/v1"), "test-key".into()).unwrap();
    let list = catalog.list_models().await.unwrap();
    assert_eq!(list.pages, 1);
    assert_eq!(list.models[0].id, "exact-openai-id");
    assert_eq!(
        list.models[0].metadata.get("shutdown_date"),
        Some(&Value::Null)
    );
    assert_eq!(list.models[0].metadata["future"], json!({"enabled":true}));
    assert!(!list.models[1].metadata.contains_key("shutdown_date"));
    assert!(catalog.list_models().await.unwrap().models.is_empty());
    for _ in 0..3 {
        assert!(matches!(
            catalog.list_models().await,
            Err(Error::Protocol(_))
        ));
    }
    server.await.unwrap();
}

#[tokio::test]
async fn anthropic_models_follow_after_id_with_same_headers() {
    let (base, server) = serve(vec![
        anthropic(
            "/v1/models",
            Reply::Json(json!({
                "data":[{"id":"first/+","max_input_tokens":null,"lifecycle":"future"}],
                "last_id":"first/+","has_more":true
            })),
        ),
        anthropic(
            "/v1/models?after_id=first%2F%2B",
            Reply::Json(json!({
                "data":[{"id":"second","capabilities":{"future":{"supported":true}}}],
                "last_id":null,"has_more":false
            })),
        ),
    ])
    .await;
    let catalog = AnthropicCatalog::at_base(
        &format!("{base}/v1"),
        "test-key".into(),
        Some("wrkspc_fixture"),
    )
    .unwrap();
    let list = catalog.list_models().await.unwrap();
    assert_eq!(list.pages, 2);
    assert_eq!(
        list.models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        ["first/+", "second"]
    );
    assert_eq!(
        list.models[0].metadata.get("max_input_tokens"),
        Some(&Value::Null)
    );
    assert_eq!(list.models[0].metadata["lifecycle"], "future");
    assert_eq!(
        list.models[1].metadata["capabilities"]["future"]["supported"],
        true
    );
    server.await.unwrap();
}

#[tokio::test]
async fn anthropic_models_reject_missing_repeated_or_failed_pages() {
    fn next(path: &'static str, cursor: &str) -> Exchange {
        anthropic(
            path,
            Reply::Json(json!({
                "data":[{"id":cursor}], "has_more":true, "last_id":cursor
            })),
        )
    }
    let (base, server) = serve(vec![
        anthropic(
            "/v1/models",
            Reply::Json(json!({"data":[],"has_more":false})),
        ),
        anthropic(
            "/v1/models",
            Reply::Json(json!({"data":[{"id":"missing"}],"has_more":true})),
        ),
        next("/v1/models", "a"),
        next("/v1/models?after_id=a", "b"),
        next("/v1/models?after_id=b", "a"),
        next("/v1/models", "c"),
        anthropic("/v1/models?after_id=c", Reply::Status(503)),
    ])
    .await;
    let catalog = AnthropicCatalog::at_base(
        &format!("{base}/v1"),
        "test-key".into(),
        Some("wrkspc_fixture"),
    )
    .unwrap();
    assert!(catalog.list_models().await.unwrap().models.is_empty());
    assert!(matches!(
        catalog.list_models().await,
        Err(Error::Protocol("missing Anthropic models cursor"))
    ));
    assert!(matches!(
        catalog.list_models().await,
        Err(Error::Protocol("repeated Anthropic models cursor"))
    ));
    assert!(matches!(catalog.list_models().await, Err(Error::Http(503))));
    server.await.unwrap();
}

#[tokio::test]
async fn ollama_cloud_models_keep_tag_and_reported_capabilities() {
    let (base, server) = serve(vec![Exchange {
        path: "/api/tags",
        headers: &[],
        response: Reply::Json(json!({"models":[{
            "model":"example:tag","name":"display",
            "details":{"families":null},"capabilities":["decision","unknown"]
        }]})),
    }])
    .await;
    let list = OllamaCloudCatalog::at_base(&format!("{base}/api"))
        .unwrap()
        .list_models()
        .await
        .unwrap();
    assert_eq!(list.pages, 1);
    let model = &list.models[0];
    assert_eq!(model.id, "example:tag");
    assert_eq!(model.metadata["name"], "display");
    assert_eq!(model.metadata["details"], json!({"families":null}));
    assert_eq!(
        model.metadata["capabilities"],
        json!(["decision", "unknown"])
    );
    server.await.unwrap();
}

#[tokio::test]
async fn catalog_redirect_does_not_reach_a_different_origin() {
    let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (base, server) = serve(vec![Exchange {
        path: "/v1/models",
        headers: BEARER,
        response: Reply::Redirect(format!("http://{}/models", target.local_addr().unwrap())),
    }])
    .await;
    let catalog = OpenAiCatalog::at_base(&format!("{base}/v1"), "test-key".into()).unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(5), catalog.list_models())
            .await
            .unwrap(),
        Err(Error::Http(302))
    ));
    server.await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(100), target.accept())
            .await
            .is_err()
    );
}
