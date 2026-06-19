use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::RwLock;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ai_agents_core::{
    ChatMessage, LLMConfig, LLMProvider, Tool, ToolCallClassification, ToolOperationKind,
    ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;

const DEFAULT_MAX_RESPONSE_BYTES: usize = 1_048_576;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;
const DEFAULT_CACHE_TTL_SECONDS: u64 = 900;
const DEFAULT_MAX_REDIRECTS: usize = 5;
const DEFAULT_TIMEOUT_MS: u64 = 15_000;

/// Fetches public web content with SSRF-oriented network safety checks.
pub struct WebFetchTool {
    client: reqwest::Client,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    extractor: Arc<RwLock<Option<Arc<dyn LLMProvider>>>>,
}

impl WebFetchTool {
    /// Create a web fetch tool with redirects handled manually.
    pub fn new() -> Self {
        Self::with_extractor_slot(Arc::new(RwLock::new(None)))
    }

    /// Create a web fetch tool that can use a shared extraction LLM.
    pub fn with_extractor_slot(extractor: Arc<RwLock<Option<Arc<dyn LLMProvider>>>>) -> Self {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(DEFAULT_TIMEOUT_MS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            extractor,
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WebFetchInput {
    /// URL to fetch.
    url: String,
    /// Optional extraction prompt. Returned as evidence when no extractor is configured.
    #[serde(default)]
    prompt: Option<String>,
    /// Maximum output characters. Defaults to 20000.
    #[serde(default)]
    max_chars: Option<usize>,
    /// Cache TTL in seconds. Defaults to 900.
    #[serde(default)]
    cache_ttl_seconds: Option<u64>,
    /// Maximum response bytes. Defaults to 1 MiB.
    #[serde(default)]
    max_response_bytes: Option<usize>,
    /// Maximum redirects. Defaults to 5.
    #[serde(default)]
    max_redirects: Option<usize>,
    #[serde(default, rename = "__ai_agents_policy")]
    #[schemars(skip)]
    policy: Option<WebFetchPolicyInput>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct WebFetchPolicyInput {
    #[serde(default)]
    allowed_domains: Vec<String>,
    #[serde(default)]
    blocked_domains: Vec<String>,
    #[serde(default)]
    domain_allow: Vec<String>,
    #[serde(default)]
    domain_deny: Vec<String>,
    #[serde(default)]
    domain_requires_approval: Vec<String>,
    #[serde(default)]
    domain_unavailable: Vec<String>,
    #[serde(default)]
    allowed_schemes: Vec<String>,
    #[serde(default)]
    allowed_ports: Vec<u16>,
    #[serde(default = "default_true")]
    blocked_private_networks: bool,
    #[serde(default)]
    max_redirects: Option<usize>,
}

#[derive(Debug, Serialize, Clone)]
struct WebFetchOutput {
    url: String,
    final_url: String,
    status: u16,
    content_type: Option<String>,
    content: String,
    truncated: bool,
    from_cache: bool,
    redirects: Vec<String>,
    extraction_prompt_used: bool,
    extraction_available: bool,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    output: WebFetchOutput,
    expires_at: Instant,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn id(&self) -> &str {
        "web_fetch"
    }

    fn name(&self) -> &str {
        "Web Fetch"
    }

    fn description(&self) -> &str {
        "Fetch public web content with URL, redirect, DNS/IP, byte, and output safety checks."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<WebFetchInput>()
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
            host_dependent: false,
            requires_user_interaction: false,
            supports_cancellation: true,
            default_requires_approval: false,
            should_defer_schema: false,
            max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
            max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        }
    }

    fn classify_call(&self, _args: &Value) -> ToolCallClassification {
        ToolCallClassification::from_metadata(&self.safety_metadata())
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: WebFetchInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let max_output_chars = input.max_chars.unwrap_or(DEFAULT_MAX_OUTPUT_CHARS);
        let max_response_bytes = input
            .max_response_bytes
            .unwrap_or(DEFAULT_MAX_RESPONSE_BYTES)
            .min(DEFAULT_MAX_RESPONSE_BYTES * 8);
        let max_redirects = input
            .max_redirects
            .or_else(|| {
                input
                    .policy
                    .as_ref()
                    .and_then(|policy| policy.max_redirects)
            })
            .unwrap_or(DEFAULT_MAX_REDIRECTS)
            .min(DEFAULT_MAX_REDIRECTS * 4);
        let cache_ttl = input.cache_ttl_seconds.unwrap_or(DEFAULT_CACHE_TTL_SECONDS);
        let cache_key = format!(
            "{}|{}|{}|{}",
            input.url,
            input.prompt.as_deref().unwrap_or(""),
            max_output_chars,
            max_response_bytes
        );
        if cache_ttl > 0 {
            if let Some(entry) = self.cache.read().get(&cache_key).cloned() {
                if Instant::now() < entry.expires_at {
                    let mut output = entry.output;
                    output.from_cache = true;
                    return web_result(&output, output.truncated, max_output_chars, true, None);
                }
            }
        }

        let original_url = match reqwest::Url::parse(&input.url) {
            Ok(url) => url,
            Err(error) => return ToolResult::error(format!("Invalid URL: {}", error)),
        };
        let mut current_url = original_url.clone();
        let mut redirects = Vec::new();
        if let Err(result) = validate_url_with_policy(&current_url, input.policy.as_ref()).await {
            return result;
        }

        let response = loop {
            if let Err(result) = validate_url_with_policy(&current_url, input.policy.as_ref()).await
            {
                return result;
            }
            let response = match self.client.get(current_url.clone()).send().await {
                Ok(response) => response,
                Err(error) => return ToolResult::error(format!("Request failed: {}", error)),
            };
            if response.status().is_redirection() {
                if redirects.len() >= max_redirects {
                    return ToolResult::error("Redirect limit exceeded");
                }
                let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
                    return ToolResult::error("Redirect response missing Location header");
                };
                let location = match location.to_str() {
                    Ok(location) => location,
                    Err(_) => {
                        return ToolResult::error("Redirect Location header is not valid UTF-8");
                    }
                };
                let next_url = match current_url.join(location) {
                    Ok(url) => url,
                    Err(error) => {
                        return ToolResult::error(format!("Invalid redirect URL: {}", error));
                    }
                };
                if let Err(result) =
                    validate_url_with_policy(&next_url, input.policy.as_ref()).await
                {
                    return result;
                }
                redirects.push(next_url.to_string());
                current_url = next_url;
                continue;
            }
            break response;
        };

        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    return ToolResult::error(format!("Response stream failed: {}", error));
                }
            };
            if body.len().saturating_add(chunk.len()) > max_response_bytes {
                return ToolResult::error(format!(
                    "Response exceeded max_response_bytes {}",
                    max_response_bytes
                ));
            }
            body.extend_from_slice(&chunk);
        }
        let raw_text = match String::from_utf8(body) {
            Ok(text) => text,
            Err(_) => return ToolResult::error("Response body is not UTF-8 text"),
        };
        let converted = if content_type
            .as_deref()
            .is_some_and(|value| value.to_ascii_lowercase().contains("html"))
            || raw_text.trim_start().starts_with('<')
        {
            html_to_text(&raw_text)
        } else {
            raw_text
        };
        let (content, truncated) = truncate_chars(converted, max_output_chars);
        let mut extraction_available = false;
        let mut final_content = content;
        let mut extraction_error = None;
        if let Some(prompt) = input.prompt.as_deref() {
            if let Some(extractor) = { self.extractor.read().clone() } {
                match extract_with_llm(extractor, prompt, &final_content).await {
                    Ok(answer) => {
                        final_content = answer;
                        extraction_available = true;
                    }
                    Err(error) => {
                        extraction_error = Some(error);
                    }
                }
            }
        }
        let (final_content, extraction_truncated) = truncate_chars(final_content, max_output_chars);
        let output = WebFetchOutput {
            url: original_url.to_string(),
            final_url: current_url.to_string(),
            status,
            content_type,
            content: final_content,
            truncated: truncated || extraction_truncated,
            from_cache: false,
            redirects,
            extraction_prompt_used: input.prompt.is_some(),
            extraction_available,
        };
        if cache_ttl > 0 {
            self.cache.write().insert(
                cache_key,
                CacheEntry {
                    output: output.clone(),
                    expires_at: Instant::now() + Duration::from_secs(cache_ttl),
                },
            );
        }
        web_result(
            &output,
            output.truncated,
            max_output_chars,
            false,
            extraction_error,
        )
    }
}

async fn validate_url_with_policy(
    url: &reqwest::Url,
    policy: Option<&WebFetchPolicyInput>,
) -> Result<(), ToolResult> {
    validate_url(url).await?;
    if let Some(policy) = policy {
        check_configured_url_policy(url, policy)?;
    }
    Ok(())
}

async fn validate_url(url: &reqwest::Url) -> Result<(), ToolResult> {
    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(ToolResult::error(format!(
                "URL scheme '{}' is not allowed",
                other
            )));
        }
    }
    if url.username() != "" || url.password().is_some() {
        return Err(ToolResult::error(
            "URLs with embedded credentials are not allowed",
        ));
    }
    let Some(host) = url.host_str() else {
        return Err(ToolResult::error("URL host is required"));
    };
    if is_metadata_host(host) || is_localhost_name(host) {
        return Err(ToolResult::error(
            "Localhost and metadata-service hosts are blocked",
        ));
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        validate_ip(ip)?;
        return Ok(());
    }
    let port = url.port_or_known_default().unwrap_or(443);
    let addresses = tokio::net::lookup_host((host, port))
        .await
        .map_err(|error| ToolResult::error(format!("DNS lookup failed: {}", error)))?;
    for address in addresses {
        validate_socket_addr(address)?;
    }
    Ok(())
}

fn check_configured_url_policy(
    url: &reqwest::Url,
    policy: &WebFetchPolicyInput,
) -> Result<(), ToolResult> {
    let scheme = url.scheme();
    if !policy.allowed_schemes.is_empty()
        && !policy
            .allowed_schemes
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(scheme))
    {
        return Err(ToolResult::error(format!(
            "URL scheme '{}' is not allowed by policy",
            scheme
        )));
    }

    if !policy.allowed_ports.is_empty() {
        let port = url.port_or_known_default().unwrap_or(0);
        if !policy.allowed_ports.contains(&port) {
            return Err(ToolResult::error(format!(
                "URL port '{}' is not allowed by policy",
                port
            )));
        }
    }

    let Some(host) = url.host_str().map(normalize_host) else {
        return Err(ToolResult::error("URL host is required"));
    };
    if policy.blocked_private_networks && (is_metadata_host(&host) || is_localhost_name(&host)) {
        return Err(ToolResult::error(
            "Private, localhost, link-local, or metadata host is blocked by policy",
        ));
    }

    for pattern in policy
        .blocked_domains
        .iter()
        .chain(policy.domain_deny.iter())
    {
        if host_matches(pattern, &host) {
            return Err(ToolResult::error(format!(
                "Domain '{}' is blocked by policy",
                pattern
            )));
        }
    }

    for pattern in &policy.domain_unavailable {
        if host_matches(pattern, &host) {
            return Err(ToolResult::error(format!(
                "Domain '{}' is unavailable by policy",
                pattern
            )));
        }
    }

    for pattern in &policy.domain_requires_approval {
        if host_matches(pattern, &host) {
            return Err(ToolResult::error(format!(
                "Domain '{}' requires approval and cannot be reached by redirect",
                pattern
            )));
        }
    }

    let allowed: Vec<&String> = policy
        .allowed_domains
        .iter()
        .chain(policy.domain_allow.iter())
        .collect();
    if !allowed.is_empty() && !allowed.iter().any(|pattern| host_matches(pattern, &host)) {
        return Err(ToolResult::error(
            "URL domain is not in the configured allowlist",
        ));
    }

    Ok(())
}

fn validate_socket_addr(address: SocketAddr) -> Result<(), ToolResult> {
    validate_ip(address.ip())
}

fn validate_ip(ip: IpAddr) -> Result<(), ToolResult> {
    if is_blocked_ip(ip) {
        Err(ToolResult::error(
            "Private, localhost, link-local, multicast, documentation, and metadata IPs are blocked",
        ))
    } else {
        Ok(())
    }
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.is_documentation()
                || ip.octets() == [169, 254, 169, 254]
                || ip.octets() == [0, 0, 0, 0]
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.segments()[0] & 0xfe00 == 0xfc00
                || ip.segments()[0] & 0xffc0 == 0xfe80
        }
    }
}

fn normalize_host(host: &str) -> String {
    host.trim_end_matches('.').to_ascii_lowercase()
}

fn host_matches(pattern: &str, host: &str) -> bool {
    let pattern = normalize_host(pattern.trim_start_matches("*."));
    host == pattern || host.ends_with(&format!(".{}", pattern))
}

fn is_localhost_name(host: &str) -> bool {
    let host = normalize_host(host);
    host == "localhost" || host.ends_with(".localhost")
}

fn is_metadata_host(host: &str) -> bool {
    let host = normalize_host(host);
    matches!(
        host.as_str(),
        "metadata.google.internal" | "metadata" | "169.254.169.254" | "100.100.100.200"
    )
}

fn html_to_text(html: &str) -> String {
    let without_scripts = Regex::new(r"(?is)<(script|style)[^>]*>.*?</(script|style)>")
        .map(|regex| regex.replace_all(html, " ").to_string())
        .unwrap_or_else(|_| html.to_string());
    let with_breaks = Regex::new(r"(?i)<\s*(br|/p|/div|/h[1-6]|/li)\s*/?>")
        .map(|regex| regex.replace_all(&without_scripts, "\n").to_string())
        .unwrap_or(without_scripts);
    let without_tags = Regex::new(r"(?is)<[^>]+>")
        .map(|regex| regex.replace_all(&with_breaks, " ").to_string())
        .unwrap_or(with_breaks);
    decode_html_entities(&without_tags)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_html_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn default_true() -> bool {
    true
}

fn truncate_chars(text: String, max_chars: usize) -> (String, bool) {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        (truncated, true)
    } else {
        (text, false)
    }
}

async fn extract_with_llm(
    extractor: Arc<dyn LLMProvider>,
    prompt: &str,
    content: &str,
) -> Result<String, String> {
    let messages = vec![
        ChatMessage::system(
            "Extract only the information requested by the user from the provided web content. Return a concise answer.",
        ),
        ChatMessage::user(format!(
            "Extraction request:\n{}\n\nWeb content:\n{}",
            prompt, content
        )),
    ];
    let config = LLMConfig {
        max_tokens: Some(800),
        temperature: Some(0.0),
        ..LLMConfig::default()
    };
    extractor
        .complete(&messages, Some(&config))
        .await
        .map(|response| response.content)
        .map_err(|error| error.to_string())
}

fn web_result(
    output: &WebFetchOutput,
    truncated: bool,
    max_output_chars: usize,
    from_cache: bool,
    extraction_error: Option<String>,
) -> ToolResult {
    let json = match serde_json::to_string(output) {
        Ok(json) => json,
        Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
    };
    let mut metadata = HashMap::new();
    metadata.insert("truncated".to_string(), Value::Bool(truncated));
    metadata.insert(
        "max_output_chars".to_string(),
        Value::from(max_output_chars),
    );
    metadata.insert("from_cache".to_string(), Value::Bool(from_cache));
    if output.extraction_prompt_used {
        let status = if output.extraction_available {
            "executed".to_string()
        } else if extraction_error.is_some() {
            "failed".to_string()
        } else {
            "unavailable".to_string()
        };
        metadata.insert("nested_llm_extraction".to_string(), Value::String(status));
        if let Some(error) = extraction_error {
            metadata.insert("nested_llm_error".to_string(), Value::String(error));
        }
    }
    ToolResult::ok_with_metadata(json, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn blocks_localhost_urls() {
        let result = WebFetchTool::new()
            .execute(serde_json::json!({"url": "http://127.0.0.1:1234"}))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn blocks_metadata_hosts() {
        let result = WebFetchTool::new()
            .execute(serde_json::json!({"url": "http://169.254.169.254/latest"}))
            .await;
        assert!(!result.success);
    }

    #[test]
    fn configured_policy_blocks_redirect_targets_outside_allowlist() {
        let policy = WebFetchPolicyInput {
            domain_allow: vec!["docs.rs".to_string()],
            allowed_schemes: vec!["https".to_string()],
            allowed_ports: vec![443],
            ..WebFetchPolicyInput::default()
        };
        let allowed = reqwest::Url::parse("https://docs.rs/serde/latest/serde/").unwrap();
        let denied = reqwest::Url::parse("https://example.com/").unwrap();

        assert!(check_configured_url_policy(&allowed, &policy).is_ok());
        assert!(check_configured_url_policy(&denied, &policy).is_err());
    }

    #[test]
    fn converts_html_to_text() {
        let text = html_to_text("<html><body><h1>Title</h1><p>Hello &amp; bye</p></body></html>");
        assert!(text.contains("Title"));
        assert!(text.contains("Hello & bye"));
    }
}
