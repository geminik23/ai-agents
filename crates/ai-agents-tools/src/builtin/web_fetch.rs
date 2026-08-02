use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::RwLock;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ai_agents_core::{
    ChatMessage, DomainPolicyBinding, LLMConfig, LLMProvider, ResultLimitBinding, ResultLimitKind,
    Tool, ToolCallClassification, ToolExecutionContext, ToolOperationKind, ToolPolicyBindings,
    ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;

const DEFAULT_MAX_RESPONSE_BYTES: usize = 1_048_576;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;
const DEFAULT_CACHE_TTL_SECONDS: u64 = 900;
const DEFAULT_MAX_REDIRECTS: usize = 5;
const DEFAULT_TIMEOUT_MS: u64 = 15_000;

/// A single HTTP GET issued by the web fetch tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebFetchTransportRequest {
    pub url: String,
    pub max_response_bytes: usize,
}

/// The transport-level fields consumed by the web fetch tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebFetchTransportResponse {
    pub status: u16,
    pub content_type: Option<String>,
    pub location: Option<String>,
    pub body: Vec<u8>,
}

/// Sends HTTP requests without applying URL or redirect policy.
#[async_trait]
pub trait WebFetchTransport: Send + Sync {
    async fn send(
        &self,
        request: WebFetchTransportRequest,
    ) -> Result<WebFetchTransportResponse, String>;

    /// Sends a request using only the addresses approved by URL validation.
    /// The compatibility default delegates to `send`; socket-opening transports must override it to enforce address binding.
    async fn send_validated(
        &self,
        request: WebFetchTransportRequest,
        _addresses: &[SocketAddr],
    ) -> Result<WebFetchTransportResponse, String> {
        self.send(request).await
    }
}

/// Resolves hostnames for SSRF validation before each request.
#[async_trait]
pub trait WebFetchResolver: Send + Sync {
    async fn resolve(&self, host: &str, port: u16) -> Result<Vec<IpAddr>, String>;
}

struct ReqwestWebFetchTransport;

#[async_trait]
impl WebFetchTransport for ReqwestWebFetchTransport {
    async fn send(
        &self,
        _request: WebFetchTransportRequest,
    ) -> Result<WebFetchTransportResponse, String> {
        Err(
            "Validated network addresses are required for the default web fetch transport"
                .to_string(),
        )
    }

    async fn send_validated(
        &self,
        request: WebFetchTransportRequest,
        addresses: &[SocketAddr],
    ) -> Result<WebFetchTransportResponse, String> {
        let url = reqwest::Url::parse(&request.url)
            .map_err(|error| format!("Request URL is invalid: {}", error))?;
        let host = url
            .host_str()
            .ok_or_else(|| "Request URL host is required".to_string())?;
        let port = url.port_or_known_default().unwrap_or(443);
        if addresses.is_empty() || addresses.iter().any(|address| address.port() != port) {
            return Err("Validated network addresses do not match the request port".to_string());
        }
        if let Ok(ip) = host.parse::<IpAddr>()
            && addresses.iter().any(|address| address.ip() != ip)
        {
            return Err("Validated network addresses do not match the request host".to_string());
        }

        // Build a request-scoped client so DNS overrides cannot leak across concurrent hosts.
        // Proxies are disabled because a proxy could resolve the hostname outside this validated address set.
        let mut client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(DEFAULT_TIMEOUT_MS))
            .no_proxy();
        if host.parse::<IpAddr>().is_err() {
            client = client.resolve_to_addrs(host, addresses);
        }
        let client = client
            .build()
            .map_err(|error| format!("Request client construction failed: {}", error))?;
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|error| format!("Request failed: {}", error))?;
        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .map(|value| {
                value
                    .to_str()
                    .map(str::to_string)
                    .map_err(|_| "Redirect Location header is not valid UTF-8".to_string())
            })
            .transpose()?;
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| format!("Response stream failed: {}", error))?;
            if body.len().saturating_add(chunk.len()) > request.max_response_bytes {
                return Err(format!(
                    "Response exceeded max_response_bytes {}",
                    request.max_response_bytes
                ));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(WebFetchTransportResponse {
            status,
            content_type,
            location,
            body,
        })
    }
}

struct TokioWebFetchResolver;

#[async_trait]
impl WebFetchResolver for TokioWebFetchResolver {
    async fn resolve(&self, host: &str, port: u16) -> Result<Vec<IpAddr>, String> {
        tokio::net::lookup_host((host, port))
            .await
            .map(|addresses| addresses.map(|address| address.ip()).collect())
            .map_err(|error| format!("DNS lookup failed: {}", error))
    }
}

/// Fetches public web content with SSRF-oriented network safety checks.
pub struct WebFetchTool {
    transport: Arc<dyn WebFetchTransport>,
    resolver: Arc<dyn WebFetchResolver>,
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
        Self::with_extractor_slot_and_transport(
            extractor,
            Arc::new(ReqwestWebFetchTransport),
            Arc::new(TokioWebFetchResolver),
        )
    }

    /// Create a web fetch tool with injected HTTP and DNS implementations.
    pub fn with_transport_and_resolver(
        transport: Arc<dyn WebFetchTransport>,
        resolver: Arc<dyn WebFetchResolver>,
    ) -> Self {
        Self::with_extractor_slot_and_transport(Arc::new(RwLock::new(None)), transport, resolver)
    }

    /// Create a web fetch tool with extraction, HTTP, and DNS implementations.
    pub fn with_extractor_slot_and_transport(
        extractor: Arc<RwLock<Option<Arc<dyn LLMProvider>>>>,
        transport: Arc<dyn WebFetchTransport>,
        resolver: Arc<dyn WebFetchResolver>,
    ) -> Self {
        Self {
            transport,
            resolver,
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

impl WebFetchPolicyInput {
    /// Build redirect-time policy from the executor-provided policy snapshot.
    fn from_context(value: &Value) -> Self {
        let mut policy = serde_json::from_value::<Self>(value.clone()).unwrap_or_default();
        policy.domain_allow.extend(
            value
                .get("domains")
                .and_then(|domains| domains.get("allow"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string),
        );
        policy.domain_deny.extend(
            value
                .get("domains")
                .and_then(|domains| domains.get("deny"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string),
        );
        policy.domain_requires_approval.extend(
            value
                .get("domains")
                .and_then(|domains| domains.get("requires_approval"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string),
        );
        policy.domain_unavailable.extend(
            value
                .get("domains")
                .and_then(|domains| domains.get("unavailable"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string),
        );
        policy
    }
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

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            domain_fields: vec![DomainPolicyBinding::url("url")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_chars", ResultLimitKind::MaxOutputChars),
                ResultLimitBinding::new("max_response_bytes", ResultLimitKind::MaxResponseBytes),
                ResultLimitBinding::new("max_redirects", ResultLimitKind::MaxRedirects),
            ],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: WebFetchInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let policy = WebFetchPolicyInput::from_context(&ctx.policy_snapshot);
        let max_output_chars = input.max_chars.unwrap_or(DEFAULT_MAX_OUTPUT_CHARS).min(
            ctx.limits
                .max_output_chars
                .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS),
        );
        let max_response_bytes = input
            .max_response_bytes
            .unwrap_or(DEFAULT_MAX_RESPONSE_BYTES)
            .min(DEFAULT_MAX_RESPONSE_BYTES * 8)
            .min(
                ctx.limits
                    .max_response_bytes
                    .unwrap_or(DEFAULT_MAX_RESPONSE_BYTES * 8),
            );
        let max_redirects = input
            .max_redirects
            .or(policy.max_redirects)
            .unwrap_or(DEFAULT_MAX_REDIRECTS)
            .min(DEFAULT_MAX_REDIRECTS * 4)
            .min(
                ctx.limits
                    .max_redirects
                    .unwrap_or(DEFAULT_MAX_REDIRECTS * 4),
            );
        let cache_ttl = input.cache_ttl_seconds.unwrap_or(DEFAULT_CACHE_TTL_SECONDS);

        let original_url = match reqwest::Url::parse(&input.url) {
            Ok(url) => url,
            Err(error) => return ToolResult::error(format!("Invalid URL: {}", error)),
        };
        if let Err(result) =
            validate_url_with_policy(&original_url, Some(&policy), self.resolver.as_ref()).await
        {
            return result;
        }
        let cache_key = format!(
            "{}|{}|{}|{}|{}",
            input.url,
            input.prompt.as_deref().unwrap_or(""),
            max_output_chars,
            max_response_bytes,
            policy_cache_fingerprint(&policy)
        );
        if cache_ttl > 0 {
            let cached_entry = { self.cache.read().get(&cache_key).cloned() };
            if let Some(entry) = cached_entry
                && Instant::now() < entry.expires_at
            {
                if let Err(result) = validate_cached_output_with_policy(
                    &entry.output,
                    &policy,
                    self.resolver.as_ref(),
                )
                .await
                {
                    return result;
                }
                let mut output = entry.output;
                output.from_cache = true;
                return web_result(&output, output.truncated, max_output_chars, true, None);
            }
        }

        let mut current_url = original_url.clone();
        let mut redirects = Vec::new();

        let response = loop {
            if ctx.cancellation.is_cancelled() {
                return ToolResult::error(
                    ctx.cancellation
                        .reason()
                        .unwrap_or("Tool execution cancelled"),
                );
            }
            let validated_target =
                match validate_url_with_policy(&current_url, Some(&policy), self.resolver.as_ref())
                    .await
                {
                    Ok(target) => target,
                    Err(result) => return result,
                };
            let response = match self
                .transport
                .send_validated(
                    WebFetchTransportRequest {
                        url: current_url.to_string(),
                        max_response_bytes,
                    },
                    &validated_target.addresses,
                )
                .await
            {
                Ok(response) => response,
                Err(error) => return ToolResult::error(error),
            };
            if (300..400).contains(&response.status) {
                if redirects.len() >= max_redirects {
                    return ToolResult::error("Redirect limit exceeded");
                }
                let Some(location) = response.location.as_deref() else {
                    return ToolResult::error("Redirect response missing Location header");
                };
                let next_url = match current_url.join(location) {
                    Ok(url) => url,
                    Err(error) => {
                        return ToolResult::error(format!("Invalid redirect URL: {}", error));
                    }
                };
                redirects.push(next_url.to_string());
                current_url = next_url;
                continue;
            }
            break response;
        };

        let status = response.status;
        let content_type = response.content_type;
        if response.body.len() > max_response_bytes {
            return ToolResult::error(format!(
                "Response exceeded max_response_bytes {}",
                max_response_bytes
            ));
        }
        if ctx.cancellation.is_cancelled() {
            return ToolResult::error(
                ctx.cancellation
                    .reason()
                    .unwrap_or("Tool execution cancelled"),
            );
        }
        let raw_text = match String::from_utf8(response.body) {
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
        if let Some(prompt) = input.prompt.as_deref()
            && let Some(extractor) = { self.extractor.read().clone() }
        {
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

struct ValidatedWebTarget {
    addresses: Vec<SocketAddr>,
}

async fn validate_url_with_policy(
    url: &reqwest::Url,
    policy: Option<&WebFetchPolicyInput>,
    resolver: &dyn WebFetchResolver,
) -> Result<ValidatedWebTarget, ToolResult> {
    let target = validate_url(url, resolver).await?;
    if let Some(policy) = policy {
        check_configured_url_policy(url, policy)?;
    }
    Ok(target)
}

async fn validate_cached_output_with_policy(
    output: &WebFetchOutput,
    policy: &WebFetchPolicyInput,
    resolver: &dyn WebFetchResolver,
) -> Result<(), ToolResult> {
    let final_url = reqwest::Url::parse(&output.final_url)
        .map_err(|error| ToolResult::error(format!("Cached final URL is invalid: {}", error)))?;
    let _ = validate_url_with_policy(&final_url, Some(policy), resolver).await?;
    for redirect in &output.redirects {
        let url = reqwest::Url::parse(redirect).map_err(|error| {
            ToolResult::error(format!("Cached redirect URL is invalid: {}", error))
        })?;
        let _ = validate_url_with_policy(&url, Some(policy), resolver).await?;
    }
    Ok(())
}

fn policy_cache_fingerprint(policy: &WebFetchPolicyInput) -> String {
    serde_json::json!({
        "allowed_domains": policy.allowed_domains,
        "blocked_domains": policy.blocked_domains,
        "domain_allow": policy.domain_allow,
        "domain_deny": policy.domain_deny,
        "domain_requires_approval": policy.domain_requires_approval,
        "domain_unavailable": policy.domain_unavailable,
        "allowed_schemes": policy.allowed_schemes,
        "allowed_ports": policy.allowed_ports,
        "blocked_private_networks": policy.blocked_private_networks,
        "max_redirects": policy.max_redirects,
    })
    .to_string()
}

async fn validate_url(
    url: &reqwest::Url,
    resolver: &dyn WebFetchResolver,
) -> Result<ValidatedWebTarget, ToolResult> {
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
    let port = url.port_or_known_default().unwrap_or(443);
    if let Ok(ip) = host.parse::<IpAddr>() {
        validate_ip(ip)?;
        return Ok(ValidatedWebTarget {
            addresses: vec![SocketAddr::new(ip, port)],
        });
    }
    let addresses = resolver
        .resolve(host, port)
        .await
        .map_err(ToolResult::error)?;
    if addresses.is_empty() {
        return Err(ToolResult::error("DNS lookup returned no addresses"));
    }
    for address in &addresses {
        validate_ip(*address)?;
    }
    Ok(ValidatedWebTarget {
        addresses: addresses
            .into_iter()
            .map(|address| SocketAddr::new(address, port))
            .collect(),
    })
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

fn validate_ip(ip: IpAddr) -> Result<(), ToolResult> {
    if is_blocked_ip(ip) {
        Err(ToolResult::error(
            "Non-public and special-use IP addresses are blocked",
        ))
    } else {
        Ok(())
    }
}

//
// The stable MSRV lacks IpAddr::is_global, so this conservative list rejects non-public and special-use prefixes before transport binding.
//
fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => [
            (Ipv4Addr::new(0, 0, 0, 0), 8),
            (Ipv4Addr::new(10, 0, 0, 0), 8),
            (Ipv4Addr::new(100, 64, 0, 0), 10),
            (Ipv4Addr::new(127, 0, 0, 0), 8),
            (Ipv4Addr::new(169, 254, 0, 0), 16),
            (Ipv4Addr::new(172, 16, 0, 0), 12),
            (Ipv4Addr::new(192, 0, 0, 0), 24),
            (Ipv4Addr::new(192, 0, 2, 0), 24),
            (Ipv4Addr::new(192, 88, 99, 0), 24),
            (Ipv4Addr::new(192, 168, 0, 0), 16),
            (Ipv4Addr::new(198, 18, 0, 0), 15),
            (Ipv4Addr::new(198, 51, 100, 0), 24),
            (Ipv4Addr::new(203, 0, 113, 0), 24),
            (Ipv4Addr::new(224, 0, 0, 0), 4),
            (Ipv4Addr::new(240, 0, 0, 0), 4),
        ]
        .into_iter()
        .any(|(network, prefix)| ipv4_in_prefix(ip, network, prefix)),
        IpAddr::V6(ip) => [
            (Ipv6Addr::UNSPECIFIED, 96),
            (Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0, 0), 96),
            (Ipv6Addr::new(0x64, 0xff9b, 0, 0, 0, 0, 0, 0), 96),
            (Ipv6Addr::new(0x64, 0xff9b, 1, 0, 0, 0, 0, 0), 48),
            (Ipv6Addr::new(0x100, 0, 0, 0, 0, 0, 0, 0), 64),
            (Ipv6Addr::new(0x100, 0, 0, 1, 0, 0, 0, 0), 64),
            (Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 23),
            (Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, 0), 32),
            (Ipv6Addr::new(0x2002, 0, 0, 0, 0, 0, 0, 0), 16),
            (Ipv6Addr::new(0x3ffe, 0, 0, 0, 0, 0, 0, 0), 16),
            (Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0), 20),
            (Ipv6Addr::new(0x5f00, 0, 0, 0, 0, 0, 0, 0), 16),
            (Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0), 7),
            (Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0), 10),
            (Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 0), 10),
            (Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0), 8),
        ]
        .into_iter()
        .any(|(network, prefix)| ipv6_in_prefix(ip, network, prefix)),
    }
}

fn ipv4_in_prefix(ip: Ipv4Addr, network: Ipv4Addr, prefix: u8) -> bool {
    let shift = 32 - u32::from(prefix);
    u32::from(ip) >> shift == u32::from(network) >> shift
}

fn ipv6_in_prefix(ip: Ipv6Addr, network: Ipv6Addr, prefix: u8) -> bool {
    let shift = 128 - u32::from(prefix);
    u128::from_be_bytes(ip.octets()) >> shift == u128::from_be_bytes(network.octets()) >> shift
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Default)]
    struct FixtureTransport {
        routes: RwLock<HashMap<String, WebFetchTransportResponse>>,
        requests: RwLock<Vec<String>>,
        validated_addresses: RwLock<Vec<Vec<SocketAddr>>>,
    }

    impl FixtureTransport {
        fn route(&self, url: &str, response: WebFetchTransportResponse) {
            self.routes.write().insert(url.to_string(), response);
        }

        fn requests(&self) -> Vec<String> {
            self.requests.read().clone()
        }

        fn validated_addresses(&self) -> Vec<Vec<SocketAddr>> {
            self.validated_addresses.read().clone()
        }
    }

    #[async_trait]
    impl WebFetchTransport for FixtureTransport {
        async fn send(
            &self,
            request: WebFetchTransportRequest,
        ) -> Result<WebFetchTransportResponse, String> {
            self.requests.write().push(request.url.clone());
            self.routes
                .read()
                .get(&request.url)
                .cloned()
                .ok_or_else(|| format!("Unconfigured web fetch route: {}", request.url))
        }

        async fn send_validated(
            &self,
            request: WebFetchTransportRequest,
            addresses: &[SocketAddr],
        ) -> Result<WebFetchTransportResponse, String> {
            self.validated_addresses.write().push(addresses.to_vec());
            self.send(request).await
        }
    }

    struct FixtureResolver {
        addresses: HashMap<String, Vec<IpAddr>>,
        default: Vec<IpAddr>,
    }

    impl Default for FixtureResolver {
        fn default() -> Self {
            Self {
                addresses: HashMap::new(),
                default: vec![IpAddr::from([93, 184, 216, 34])],
            }
        }
    }

    #[async_trait]
    impl WebFetchResolver for FixtureResolver {
        async fn resolve(&self, host: &str, _port: u16) -> Result<Vec<IpAddr>, String> {
            Ok(self
                .addresses
                .get(host)
                .cloned()
                .unwrap_or_else(|| self.default.clone()))
        }
    }

    fn response(
        status: u16,
        content_type: Option<&str>,
        location: Option<&str>,
        body: &str,
    ) -> WebFetchTransportResponse {
        WebFetchTransportResponse {
            status,
            content_type: content_type.map(str::to_string),
            location: location.map(str::to_string),
            body: body.as_bytes().to_vec(),
        }
    }

    fn fixture_tool(transport: Arc<FixtureTransport>) -> WebFetchTool {
        WebFetchTool::with_transport_and_resolver(transport, Arc::new(FixtureResolver::default()))
    }

    fn output(result: &ToolResult) -> Value {
        serde_json::from_str(&result.output).expect("web fetch output should be JSON")
    }

    #[tokio::test]
    async fn blocks_non_public_ip_ranges_without_requesting() {
        for url in [
            "http://0.0.0.1/",
            "http://10.0.0.1/",
            "http://100.64.0.1/",
            "http://127.0.0.1/",
            "http://169.254.1.1/",
            "http://169.254.169.254/latest",
            "http://192.0.0.1/",
            "http://192.0.2.1/",
            "http://192.88.99.2/",
            "http://198.18.0.1/",
            "http://240.0.0.1/",
            "http://255.255.255.255/",
            "http://[::1]/",
            "http://[64:ff9b::1]/",
            "http://[64:ff9b:1::1]/",
            "http://[100::1]/",
            "http://[100:0:0:1::1]/",
            "http://[2001:2::1]/",
            "http://[2001:db8::1]/",
            "http://[2002::1]/",
            "http://[3fff::1]/",
            "http://[5f00::1]/",
            "http://[fe80::1]/",
            "http://[fec0::1]/",
            "http://[::ffff:10.0.0.1]/",
        ] {
            let result = WebFetchTool::new()
                .execute(
                    serde_json::json!({"url": url}),
                    ai_agents_core::ToolExecutionContext::test("web_fetch"),
                )
                .await;
            assert!(!result.success, "URL should be blocked: {}", url);
        }
    }

    #[tokio::test]
    async fn default_transport_connects_only_to_supplied_address() {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0u8; 4096];
            let read = stream.read(&mut request).await.unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\nConnection: close\r\n\r\nbound")
                .await
                .unwrap();
            String::from_utf8_lossy(&request[..read]).into_owned()
        });
        let url = format!("http://binding.invalid:{}/bound", address.port());

        let response = ReqwestWebFetchTransport
            .send_validated(
                WebFetchTransportRequest {
                    url,
                    max_response_bytes: 64,
                },
                &[address],
            )
            .await
            .unwrap();
        let request = server.await.unwrap().to_ascii_lowercase();

        assert_eq!(response.body, b"bound");
        assert!(request.contains(&format!("host: binding.invalid:{}", address.port())));
    }

    #[tokio::test]
    async fn validated_addresses_are_passed_to_transport() {
        let transport = Arc::new(FixtureTransport::default());
        transport.route(
            "https://public.test/page",
            response(200, Some("text/plain"), None, "bound"),
        );
        let mut resolver = FixtureResolver::default();
        resolver
            .addresses
            .insert("public.test".to_string(), vec![IpAddr::from([1, 1, 1, 1])]);
        let tool = WebFetchTool::with_transport_and_resolver(
            Arc::clone(&transport) as Arc<dyn WebFetchTransport>,
            Arc::new(resolver),
        );

        let result = tool
            .execute(
                serde_json::json!({"url": "https://public.test/page"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(result.success);
        assert_eq!(
            transport.validated_addresses(),
            vec![vec![SocketAddr::from(([1, 1, 1, 1], 443))]]
        );
    }

    #[tokio::test]
    async fn blocks_embedded_credentials_without_requesting() {
        let transport = Arc::new(FixtureTransport::default());
        let result = fixture_tool(Arc::clone(&transport))
            .execute(
                serde_json::json!({"url": "https://user:password@public.test/"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("embedded credentials"));
        assert!(transport.requests().is_empty());
        assert!(transport.validated_addresses().is_empty());
    }

    #[tokio::test]
    async fn dns_alias_to_metadata_address_is_blocked_before_transport() {
        let transport = Arc::new(FixtureTransport::default());
        let mut resolver = FixtureResolver::default();
        resolver.addresses.insert(
            "metadata-alias.test".to_string(),
            vec![IpAddr::from([100, 100, 100, 200])],
        );
        let tool = WebFetchTool::with_transport_and_resolver(
            Arc::clone(&transport) as Arc<dyn WebFetchTransport>,
            Arc::new(resolver),
        );

        let result = tool
            .execute(
                serde_json::json!({"url": "http://metadata-alias.test/latest"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("Non-public and special-use"));
        assert!(transport.requests().is_empty());
        assert!(transport.validated_addresses().is_empty());
    }

    #[test]
    fn public_address_examples_remain_allowed() {
        for address in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            let address = address.parse::<IpAddr>().unwrap();
            assert!(
                !is_blocked_ip(address),
                "address should remain allowed: {address}"
            );
        }
    }

    #[tokio::test]
    async fn in_memory_transport_fetches_html_and_text() {
        let transport = Arc::new(FixtureTransport::default());
        transport.route(
            "https://public.test/page",
            response(
                200,
                Some("text/html"),
                None,
                "<h1>Title</h1><p>Hello &amp; bye</p>",
            ),
        );
        transport.route(
            "https://public.test/plain",
            response(200, Some("text/plain"), None, "plain response"),
        );
        let tool = fixture_tool(transport);

        let html = tool
            .execute(
                serde_json::json!({"url": "https://public.test/page"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;
        let text = tool
            .execute(
                serde_json::json!({"url": "https://public.test/plain"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(html.success);
        assert_eq!(output(&html)["content"], "Title\nHello & bye");
        assert!(text.success);
        assert_eq!(output(&text)["content"], "plain response");
    }

    #[tokio::test]
    async fn in_memory_transport_follows_exact_route_redirects() {
        let transport = Arc::new(FixtureTransport::default());
        transport.route(
            "https://public.test/start",
            response(302, None, Some("/final"), ""),
        );
        transport.route(
            "https://public.test/final",
            response(200, Some("text/plain"), None, "done"),
        );
        let tool = fixture_tool(Arc::clone(&transport));

        let result = tool
            .execute(
                serde_json::json!({"url": "https://public.test/start"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(result.success);
        assert_eq!(output(&result)["final_url"], "https://public.test/final");
        assert_eq!(
            transport.requests(),
            vec![
                "https://public.test/start".to_string(),
                "https://public.test/final".to_string()
            ]
        );
        assert_eq!(transport.validated_addresses().len(), 2);
    }

    #[tokio::test]
    async fn blocks_private_redirect_before_second_request() {
        let transport = Arc::new(FixtureTransport::default());
        transport.route(
            "https://public.test/start",
            response(302, None, Some("http://private.test/secret"), ""),
        );
        let mut resolver = FixtureResolver::default();
        resolver.addresses.insert(
            "private.test".to_string(),
            vec![IpAddr::from([10, 0, 0, 1])],
        );
        let tool = WebFetchTool::with_transport_and_resolver(
            Arc::clone(&transport) as Arc<dyn WebFetchTransport>,
            Arc::new(resolver),
        );

        let result = tool
            .execute(
                serde_json::json!({"url": "https://public.test/start"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(!result.success);
        assert_eq!(transport.requests(), vec!["https://public.test/start"]);
        assert!(result.output.contains("Non-public and special-use"));
    }

    #[tokio::test]
    async fn enforces_byte_limit_on_injected_responses() {
        let transport = Arc::new(FixtureTransport::default());
        transport.route(
            "https://public.test/large",
            response(200, Some("text/plain"), None, "123456"),
        );
        let result = fixture_tool(transport)
            .execute(
                serde_json::json!({
                    "url": "https://public.test/large",
                    "max_response_bytes": 5
                }),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("max_response_bytes 5"));
    }

    #[tokio::test]
    async fn caches_injected_transport_responses() {
        let transport = Arc::new(FixtureTransport::default());
        transport.route(
            "https://public.test/cached",
            response(200, Some("text/plain"), None, "cached"),
        );
        let tool = fixture_tool(Arc::clone(&transport));
        let args = serde_json::json!({
            "url": "https://public.test/cached",
            "cache_ttl_seconds": 60
        });

        let first = tool
            .execute(
                args.clone(),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;
        let second = tool
            .execute(
                args,
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(first.success && second.success);
        assert_eq!(transport.requests().len(), 1);
        assert_eq!(output(&first)["from_cache"], false);
        assert_eq!(output(&second)["from_cache"], true);
    }

    #[tokio::test]
    async fn applies_policy_before_injected_transport() {
        let transport = Arc::new(FixtureTransport::default());
        transport.route(
            "https://blocked.test/page",
            response(200, Some("text/plain"), None, "not reached"),
        );
        let tool = fixture_tool(Arc::clone(&transport));
        let mut context = ai_agents_core::ToolExecutionContext::test("web_fetch");
        context.policy_snapshot = serde_json::json!({
            "blocked_domains": ["blocked.test"]
        });

        let result = tool
            .execute(
                serde_json::json!({"url": "https://blocked.test/page"}),
                context,
            )
            .await;

        assert!(!result.success);
        assert!(transport.requests().is_empty());
        assert!(result.output.contains("blocked by policy"));
    }

    #[tokio::test]
    async fn reports_unconfigured_exact_routes() {
        let transport = Arc::new(FixtureTransport::default());
        let result = fixture_tool(Arc::clone(&transport))
            .execute(
                serde_json::json!({"url": "https://public.test/missing"}),
                ai_agents_core::ToolExecutionContext::test("web_fetch"),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("Unconfigured web fetch route"));
        assert_eq!(transport.requests(), vec!["https://public.test/missing"]);
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
}
