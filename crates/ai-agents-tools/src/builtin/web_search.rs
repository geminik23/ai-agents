use async_trait::async_trait;
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use ai_agents_core::{
    ResultLimitBinding, ResultLimitKind, Tool, ToolExecutionContext, ToolOperationKind,
    ToolPolicyBindings, ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;
use crate::types::{
    UnavailableWebSearchProvider, WebSearchProviderSlot, WebSearchRequest, WebSearchResponse,
    WebSearchSafeSearch,
};

const DEFAULT_MAX_RESULTS: usize = 5;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 12_000;

/// Searches public web indexes through a host-provided search provider.
pub struct WebSearchTool {
    provider: WebSearchProviderSlot,
}

impl WebSearchTool {
    /// Create a web search tool with an unavailable provider slot.
    pub fn new() -> Self {
        Self::with_provider_slot(Arc::new(RwLock::new(Arc::new(
            UnavailableWebSearchProvider,
        ))))
    }

    /// Create a web search tool backed by a shared provider slot.
    pub fn with_provider_slot(provider: WebSearchProviderSlot) -> Self {
        Self { provider }
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WebSearchInput {
    /// Search query sent to the host provider.
    query: String,
    /// Maximum result count requested by the model. Defaults to 5.
    #[serde(
        default,
        deserialize_with = "crate::deserialize_optional_positive_usize"
    )]
    #[schemars(range(min = 1))]
    max_results: Option<usize>,
    /// Optional result-domain filters such as docs.rs or rust-lang.org.
    #[serde(default)]
    include_domains: Vec<String>,
    /// Optional language hint such as en or ja.
    #[serde(default)]
    language: Option<String>,
    /// Optional region hint such as US or JP.
    #[serde(default)]
    region: Option<String>,
    /// Optional safe-search preference.
    #[serde(default)]
    safe_search: Option<WebSearchSafeSearch>,
}

#[derive(Debug, Serialize)]
struct WebSearchOutput {
    available: bool,
    query: String,
    provider: Option<String>,
    count: usize,
    truncated: bool,
    results: Vec<crate::types::WebSearchResultItem>,
    message: Option<String>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn id(&self) -> &str {
        "web_search"
    }

    fn name(&self) -> &str {
        "Web Search"
    }

    fn description(&self) -> &str {
        "Search public web indexes through a host-provided provider and return bounded cited results."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<WebSearchInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        ToolSafetyMetadata {
            read_only: true,
            concurrency_safe: true,
            operation: ToolOperationKind::Network,
            side_effect_level: ToolSideEffectLevel::ExternalRead,
            requires_network: true,
            destructive: false,
            open_world: true,
            host_dependent: true,
            requires_user_interaction: false,
            supports_cancellation: true,
            default_requires_approval: false,
            should_defer_schema: false,
            max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
            max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        }
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            result_limit_fields: vec![ResultLimitBinding::new(
                "max_results",
                ResultLimitKind::MaxResults,
            )],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: WebSearchInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        if let Err(error) = crate::validate_positive_max_results(ctx.limits.max_results) {
            return ToolResult::error(format!("Invalid result limit: {error}"));
        }
        let max_results = input
            .max_results
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .min(ctx.limits.max_results.unwrap_or(DEFAULT_MAX_RESULTS));
        let request = WebSearchRequest {
            query: input.query.clone(),
            max_results: Some(max_results),
            include_domains: input.include_domains,
            language: input.language,
            region: input.region,
            safe_search: input.safe_search,
        };
        let provider = self.provider.read().clone();
        let response = provider.search(request).await;
        result_from_response(input.query, response, max_results)
    }
}

fn result_from_response(
    query: String,
    mut response: WebSearchResponse,
    max_results: usize,
) -> ToolResult {
    let mut truncated = response.truncated;
    if response.results.len() > max_results {
        response.results.truncate(max_results);
        truncated = true;
    }
    let output = WebSearchOutput {
        available: response.available,
        query,
        provider: response.provider,
        count: response.results.len(),
        truncated,
        results: response.results,
        message: response.message,
    };
    let serialized = match serde_json::to_string(&output) {
        Ok(value) => value,
        Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
    };
    if output.available {
        ToolResult::ok(serialized)
    } else {
        ToolResult {
            success: false,
            output: serialized,
            metadata: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StaticWebSearchProvider, WebSearchProvider, WebSearchResultItem};
    use parking_lot::Mutex;
    use std::collections::HashMap;

    struct RecordingProvider {
        max_results: Mutex<Vec<usize>>,
    }

    #[async_trait::async_trait]
    impl WebSearchProvider for RecordingProvider {
        async fn search(&self, request: WebSearchRequest) -> WebSearchResponse {
            self.max_results
                .lock()
                .push(request.max_results.unwrap_or_default());
            WebSearchResponse {
                available: true,
                ..Default::default()
            }
        }
    }

    #[tokio::test]
    async fn max_results_rejects_zero_and_preserves_positive_caps() {
        let provider = Arc::new(RecordingProvider {
            max_results: Mutex::new(Vec::new()),
        });
        let slot = Arc::new(RwLock::new(
            Arc::clone(&provider) as Arc<dyn crate::types::WebSearchProvider>
        ));
        let tool = WebSearchTool::with_provider_slot(slot);

        let zero_request = tool
            .execute(
                serde_json::json!({"query": "rust", "max_results": 0}),
                ToolExecutionContext::test("web_search"),
            )
            .await;
        assert!(!zero_request.success);
        assert!(
            zero_request
                .output
                .contains("max_results must be greater than 0")
        );

        let mut zero_context = ToolExecutionContext::test("web_search");
        zero_context.limits.max_results = Some(0);
        let invalid_context = tool
            .execute(serde_json::json!({"query": "rust"}), zero_context)
            .await;
        assert!(!invalid_context.success);

        let mut capped_context = ToolExecutionContext::test("web_search");
        capped_context.limits.max_results = Some(2);
        assert!(
            tool.execute(
                serde_json::json!({"query": "rust", "max_results": 4}),
                capped_context.clone(),
            )
            .await
            .success
        );
        assert!(
            tool.execute(
                serde_json::json!({"query": "rust", "max_results": 1}),
                capped_context,
            )
            .await
            .success
        );

        assert_eq!(*provider.max_results.lock(), vec![2, 1]);
    }

    #[tokio::test]
    async fn static_provider_returns_bounded_results() {
        let mut responses = HashMap::new();
        responses.insert(
            "rust async".to_string(),
            WebSearchResponse {
                available: true,
                provider: Some("fixture".to_string()),
                results: vec![
                    WebSearchResultItem {
                        title: "one".to_string(),
                        url: "https://example.com/one".to_string(),
                        snippet: "first".to_string(),
                        source: None,
                        published_at: None,
                    },
                    WebSearchResultItem {
                        title: "two".to_string(),
                        url: "https://example.com/two".to_string(),
                        snippet: "second".to_string(),
                        source: None,
                        published_at: None,
                    },
                ],
                truncated: false,
                message: None,
            },
        );
        let provider = Arc::new(RwLock::new(
            Arc::new(StaticWebSearchProvider::new(responses))
                as Arc<dyn crate::types::WebSearchProvider>,
        ));
        let tool = WebSearchTool::with_provider_slot(provider);
        let result = tool
            .execute(
                serde_json::json!({"query":"rust async","max_results":1}),
                ToolExecutionContext::test("web_search"),
            )
            .await;
        assert!(result.success);
        let output: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["count"], 1);
        assert_eq!(output["truncated"], true);
    }
}
