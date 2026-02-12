use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

use crate::{Tool, ToolResult, generate_schema};

pub struct HttpTool {
    client: reqwest::Client,
    default_timeout: Duration,
}

impl HttpTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            default_timeout: Duration::from_secs(30),
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            client: reqwest::Client::new(),
            default_timeout: timeout,
        }
    }
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct HttpInput {
    /// HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD)
    method: String,
    /// URL to request
    url: String,
    /// Optional request headers as key-value pairs
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
    /// Optional request body (for POST/PUT/PATCH)
    #[serde(default)]
    body: Option<String>,
    /// Optional timeout in milliseconds (default: 30000)
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
struct HttpOutput {
    status: u16,
    status_text: String,
    headers: HashMap<String, String>,
    body: String,
}

#[async_trait]
impl Tool for HttpTool {
    fn id(&self) -> &str {
        "http"
    }

    fn name(&self) -> &str {
        "HTTP Client"
    }

    fn description(&self) -> &str {
        "Make HTTP requests to external APIs and websites. Supports GET, POST, PUT, DELETE, PATCH, and HEAD methods."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<HttpInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: HttpInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let timeout = input
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(self.default_timeout);

        let method = input.method.to_uppercase();
        let mut request = match method.as_str() {
            "GET" => self.client.get(&input.url),
            "POST" => self.client.post(&input.url),
            "PUT" => self.client.put(&input.url),
            "DELETE" => self.client.delete(&input.url),
            "PATCH" => self.client.patch(&input.url),
            "HEAD" => self.client.head(&input.url),
            _ => return ToolResult::error(format!("Invalid HTTP method: {}", method)),
        };

        request = request.timeout(timeout);

        if let Some(headers) = input.headers {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }

        if let Some(body) = input.body {
            request = request.body(body);
        }

        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let status_code = status.as_u16();
                let status_text = status.canonical_reason().unwrap_or("Unknown").to_string();

                let headers: HashMap<String, String> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();

                let body = response.text().await.unwrap_or_default();

                let output = HttpOutput {
                    status: status_code,
                    status_text,
                    headers,
                    body,
                };

                match serde_json::to_string(&output) {
                    Ok(json) => ToolResult::ok(json),
                    Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
                }
            }
            Err(e) => ToolResult::error(format!("Request failed: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_tool_creation() {
        let tool = HttpTool::new();
        assert_eq!(tool.id(), "http");
        assert_eq!(tool.name(), "HTTP Client");
    }

    #[test]
    fn test_http_tool_with_timeout() {
        let tool = HttpTool::with_timeout(Duration::from_secs(60));
        assert_eq!(tool.default_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_input_schema() {
        let tool = HttpTool::new();
        let schema = tool.input_schema();
        assert!(schema.is_object());
    }
}
