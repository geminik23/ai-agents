//! Opt-in, read-only smoke tests of the three official catalog endpoints.

use ai_agents_llm::sdk::{
    Error,
    models::{AnthropicCatalog, ModelList, OllamaCloudCatalog, OpenAiCatalog},
};
use std::{future::Future, time::Duration};

async fn verify(
    service: &str,
    request: impl Future<Output = Result<ModelList, Error>>,
    fields: &[&str],
) {
    let list = tokio::time::timeout(Duration::from_secs(30), request)
        .await
        .expect("catalog lookup timed out")
        .expect("catalog lookup failed");
    let reported_fields: Vec<_> = fields
        .iter()
        .filter(|field| {
            list.models
                .iter()
                .any(|model| model.metadata.contains_key(**field))
        })
        .collect();
    println!(
        "{service}: {} page(s), {} models, observed fields {reported_fields:?}",
        list.pages,
        list.models.len()
    );
}

#[tokio::test]
#[ignore = "requires explicit authorization and an OpenAI credential"]
async fn live_openai_catalog() {
    let catalog =
        OpenAiCatalog::new(std::env::var("OPENAI_API_KEY").expect("OpenAI credential required"))
            .unwrap();
    verify(
        "OpenAI",
        catalog.list_models(),
        &["created", "owned_by", "shutdown_date"],
    )
    .await;
}

#[tokio::test]
#[ignore = "requires explicit authorization and an Anthropic credential"]
async fn live_anthropic_catalog() {
    let workspace = std::env::var("ANTHROPIC_WORKSPACE_ID").ok();
    let catalog = AnthropicCatalog::new(
        std::env::var("ANTHROPIC_API_KEY").expect("Anthropic credential required"),
        workspace.as_deref(),
    )
    .unwrap();
    verify(
        "Anthropic",
        catalog.list_models(),
        &[
            "display_name",
            "capabilities",
            "max_input_tokens",
            "max_tokens",
            "lifecycle",
        ],
    )
    .await;
}

#[tokio::test]
#[ignore = "calls the public Ollama Cloud endpoint"]
async fn live_ollama_cloud_catalog() {
    let catalog = OllamaCloudCatalog::new().unwrap();
    let list = tokio::time::timeout(Duration::from_secs(30), catalog.list_models())
        .await
        .expect("catalog lookup timed out")
        .expect("catalog lookup failed");
    let tagged = list
        .models
        .iter()
        .filter(|model| model.id.contains(':'))
        .count();
    let reported_fields: Vec<_> = [
        "name",
        "details",
        "size",
        "modified_at",
        "digest",
        "capabilities",
    ]
    .into_iter()
    .filter(|field| {
        list.models
            .iter()
            .any(|model| model.metadata.contains_key(*field))
    })
    .collect();
    println!(
        "Ollama Cloud: {} page(s), {} models ({tagged} tagged), observed fields {reported_fields:?}",
        list.pages,
        list.models.len()
    );
}
