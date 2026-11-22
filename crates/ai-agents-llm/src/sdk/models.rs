//! Direct model catalogs for the official OpenAI, Anthropic, and Ollama Cloud endpoints.
//! Catalog metadata describes what the service reports for this request.

use super::{Error, messages, openai, transport};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value};
use std::collections::HashSet;

/// A model identifier and the other fields reported for it by the selected service.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CatalogModel {
    /// The service's exact identifier, including an Ollama tag when present.
    pub id: String,
    /// Provider-specific fields retain their wire names and JSON values, including explicit nulls.
    #[serde(flatten)]
    pub metadata: Map<String, Value>,
}

/// The complete model list in server order, with the number of successful page requests.
#[derive(Debug)]
pub struct ModelList {
    pub models: Vec<CatalogModel>,
    pub pages: usize,
}

/// Lists models visible to the supplied OpenAI API credential.
#[derive(Clone)]
pub struct OpenAiCatalog {
    transport: transport::Transport,
}

impl OpenAiCatalog {
    /// Creates a catalog client bound to the official OpenAI endpoint.
    pub fn new(api_key: String) -> Result<Self, Error> {
        Self::at_base(openai::API_BASE, api_key)
    }

    fn at_base(base: &str, api_key: String) -> Result<Self, Error> {
        Ok(Self {
            transport: transport::Transport::new(base, "models", required_bearer(api_key)?)?,
        })
    }

    /// Reads the models exposed by the service for this credential.
    pub async fn list_models(&self) -> Result<ModelList, Error> {
        let response: OpenAiList = get_json(&self.transport, &[]).await?;
        checked_models(response.data, 1)
    }
}

/// API revision used by the Anthropic model catalog.
pub const ANTHROPIC_MODELS_API_VERSION: &str = "2026-10-21";

/// Lists Claude models visible to the supplied credential and optional workspace.
#[derive(Clone)]
pub struct AnthropicCatalog {
    transport: transport::Transport,
}

impl AnthropicCatalog {
    /// Creates a client for the official Models endpoint. Workspace-scoped keys can use `None`.
    pub fn new(api_key: String, workspace_id: Option<&str>) -> Result<Self, Error> {
        Self::at_base(messages::OFFICIAL_BASE, api_key, workspace_id)
    }

    fn at_base(base: &str, api_key: String, workspace_id: Option<&str>) -> Result<Self, Error> {
        let mut headers = required_bearer(api_key)?;
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_MODELS_API_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(workspace_id) = workspace_id {
            headers.insert(
                "anthropic-workspace-id",
                HeaderValue::from_str(workspace_id)
                    .map_err(|_| Error::Config("invalid Anthropic workspace header".into()))?,
            );
        }
        Ok(Self {
            transport: transport::Transport::new(base, "models", headers)?,
        })
    }

    /// Reads all forward pages using the service's default lifecycle filter.
    /// A failed page or non-advancing cursor fails the whole call.
    pub async fn list_models(&self) -> Result<ModelList, Error> {
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        let mut models = Vec::new();
        let mut pages = 0;
        loop {
            let query = cursor.as_deref().map(|id| ("after_id", id));
            let page: AnthropicPage = get_json(&self.transport, query.as_slice()).await?;
            validate_ids(&page.data)?;
            models.extend(page.data);
            pages += 1;
            if !page.has_more {
                return Ok(ModelList { models, pages });
            }
            let next = page
                .last_id
                .filter(|id| !id.trim().is_empty())
                .ok_or(Error::Protocol("missing Anthropic models cursor"))?;
            if !seen.insert(next.clone()) {
                return Err(Error::Protocol("repeated Anthropic models cursor"));
            }
            cursor = Some(next);
        }
    }
}

/// Lists models in the public Ollama Cloud catalog.
#[derive(Clone)]
pub struct OllamaCloudCatalog {
    transport: transport::Transport,
}

impl OllamaCloudCatalog {
    /// Creates a catalog client for the official public endpoint.
    pub fn new() -> Result<Self, Error> {
        Self::at_base("https://ollama.com/api")
    }

    fn at_base(base: &str) -> Result<Self, Error> {
        Ok(Self {
            transport: transport::Transport::new(base, "tags", HeaderMap::new())?,
        })
    }

    /// Reads the service's public list without sending a credential.
    pub async fn list_models(&self) -> Result<ModelList, Error> {
        let response: OllamaList = get_json(&self.transport, &[]).await?;
        let models = response
            .models
            .into_iter()
            .map(|model| CatalogModel {
                id: model.model,
                metadata: model.metadata,
            })
            .collect();
        checked_models(models, 1)
    }
}

#[derive(Deserialize)]
struct OpenAiList {
    data: Vec<CatalogModel>,
}

#[derive(Deserialize)]
struct AnthropicPage {
    data: Vec<CatalogModel>,
    has_more: bool,
    last_id: Option<String>,
}

#[derive(Deserialize)]
struct OllamaList {
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct OllamaModel {
    model: String,
    #[serde(flatten)]
    metadata: Map<String, Value>,
}

fn required_bearer(api_key: String) -> Result<HeaderMap, Error> {
    if api_key.trim().is_empty() {
        return Err(Error::Config(
            "model catalog requires a bearer credential".into(),
        ));
    }
    transport::bearer_headers(Some(api_key))
}

async fn get_json<T: DeserializeOwned>(
    transport: &transport::Transport,
    query: &[(&str, &str)],
) -> Result<T, Error> {
    transport.get(query).await?.json().await.map_err(|error| {
        if error.is_decode() {
            Error::Protocol("invalid model catalog response")
        } else {
            Error::Transport
        }
    })
}

fn validate_ids(models: &[CatalogModel]) -> Result<(), Error> {
    if models.iter().any(|model| model.id.trim().is_empty()) {
        return Err(Error::Protocol("empty model catalog ID"));
    }
    Ok(())
}

fn checked_models(models: Vec<CatalogModel>, pages: usize) -> Result<ModelList, Error> {
    validate_ids(&models)?;
    Ok(ModelList { models, pages })
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
