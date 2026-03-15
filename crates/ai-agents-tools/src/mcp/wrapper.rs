//! MCP wrapper tool — presents an MCP server as a single builtin Tool.
//!
//! Each instance wraps one MCP server connection and presents ALL of the
//! server's functions through a single tool with a `function` discriminator
//! field, matching the pattern used by `datetime`, `math`, `json`, etc.

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

use rmcp::model as mcp_model;
use rmcp::service::{Peer, RunningService};
use rmcp::{RoleClient, ServiceExt};

use ai_agents_core::{Tool, ToolResult};

/// A discovered function from the MCP server.
#[derive(Debug, Clone)]
struct DiscoveredFunction {
    /// Original function name as reported by the MCP server.
    name: String,
    /// Human-readable description of the function.
    description: String,
    /// JSON Schema for the function's parameters.
    input_schema: Value,
}

/// Configuration for the MCP wrapper tool, deserialized from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPWrapperConfig {
    /// Display name for this tool (also used as the tool ID).
    pub name: String,

    /// Transport configuration (stdio, http, or sse).
    #[serde(flatten)]
    pub transport: MCPWrapperTransport,

    /// Environment variables passed to the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Startup timeout in milliseconds.
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_ms: u64,

    /// Security settings for function-level blocking and HITL.
    #[serde(default)]
    pub security: MCPWrapperSecurity,

    /// Optional custom description override.
    /// If not set, auto-generated from discovered functions.
    #[serde(default)]
    pub description: Option<String>,
}

fn default_startup_timeout() -> u64 {
    30_000
}

/// Transport configuration for connecting to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "lowercase")]
pub enum MCPWrapperTransport {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
    },
    Http {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
    #[serde(alias = "sse")]
    Sse {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

/// Security settings for the MCP wrapper tool.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MCPWrapperSecurity {
    /// Functions that should never be exposed to the LLM.
    #[serde(default)]
    pub blocked_functions: Vec<String>,

    /// Functions that require HITL approval before execution.
    #[serde(default)]
    pub hitl_functions: Vec<String>,
}

/// An MCP server exposed as a single builtin Tool.
///
/// Connects to an MCP server at initialization, discovers available functions
/// via `peer.list_tools()`, and builds a dynamic `input_schema()` with
/// `function` as an enum of discovered names and `params` as per-function
/// parameters. Uses two-phase construction: `new()` then `initialized()`.
pub struct MCPWrapperTool {
    config: MCPWrapperConfig,
    /// Immutable after `initialized()` — tool description shown to the LLM.
    description: String,
    /// Immutable after `initialized()` — JSON Schema for the tool's input.
    schema: Value,
    /// Running service handle — kept alive so the background task is not dropped.
    _running: RwLock<Option<RunningService<RoleClient, ()>>>,
    /// Peer for issuing MCP requests.
    peer: RwLock<Option<Peer<RoleClient>>>,
    /// Discovered functions from the MCP server (populated after init).
    functions: Vec<DiscoveredFunction>,
}

impl MCPWrapperTool {
    /// Create a new wrapper tool from configuration.
    /// The tool is NOT connected yet — call `initialized()` to connect and discover.
    pub fn new(config: MCPWrapperConfig) -> Self {
        let desc = config
            .description
            .clone()
            .unwrap_or_else(|| format!("{} operations via MCP", config.name));
        Self {
            config,
            description: desc,
            schema: json!({"type": "object"}),
            _running: RwLock::new(None),
            peer: RwLock::new(None),
            functions: Vec::new(),
        }
    }

    /// Connect to the MCP server, discover functions, and build schema/description.
    /// Returns a new `MCPWrapperTool` with the discovered state baked in.
    pub async fn initialized(mut self) -> Result<Self, String> {
        let running = match &self.config.transport {
            MCPWrapperTransport::Stdio { command, args } => {
                Self::connect_stdio(command, args, &self.config.env, &self.config.name).await?
            }
            MCPWrapperTransport::Http { url, headers }
            | MCPWrapperTransport::Sse { url, headers } => {
                Self::connect_http(url, headers, &self.config.name).await?
            }
        };

        let peer = running.peer().clone();

        // Discover functions from the MCP server
        let tool_list = peer
            .list_all_tools()
            .await
            .map_err(|e| format!("Failed to list tools from '{}': {}", self.config.name, e))?;

        let mut functions = Vec::new();
        for tool in &tool_list {
            let name = tool.name.to_string();

            // Skip blocked functions
            if self.config.security.blocked_functions.contains(&name) {
                tracing::debug!(
                    server = %self.config.name,
                    function = %name,
                    "Skipping blocked MCP function"
                );
                continue;
            }

            let description = tool
                .description
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default();

            let input_schema = Value::Object(tool.input_schema.as_ref().clone());

            functions.push(DiscoveredFunction {
                name,
                description,
                input_schema,
            });
        }

        tracing::info!(
            server = %self.config.name,
            functions = functions.len(),
            "MCP wrapper tool initialized"
        );

        // Build immutable schema and description
        self.schema = Self::build_schema(&self.config.name, &functions);
        self.description = Self::build_description(
            &self.config.name,
            self.config.description.as_deref(),
            &functions,
        );
        self.functions = functions;
        *self.peer.write() = Some(peer);
        *self._running.write() = Some(running);

        Ok(self)
    }

    /// Build the dynamic input schema from discovered functions.
    fn build_schema(server_name: &str, functions: &[DiscoveredFunction]) -> Value {
        let function_names: Vec<Value> = functions
            .iter()
            .map(|f| Value::String(f.name.clone()))
            .collect();

        // Build per-function parameter hints for the LLM
        let mut params_description =
            String::from("Parameters for the selected function. See function list for details.");

        if functions.len() <= 30 {
            params_description = String::from("Parameters for the selected function:\n");
            for f in functions {
                if let Some(props) = f.input_schema.get("properties") {
                    let prop_names: Vec<&str> = props
                        .as_object()
                        .map(|obj| obj.keys().map(|k| k.as_str()).collect())
                        .unwrap_or_default();
                    if !prop_names.is_empty() {
                        params_description.push_str(&format!(
                            "  - {}: {{{}}}\n",
                            f.name,
                            prop_names.join(", ")
                        ));
                    } else {
                        params_description.push_str(&format!("  - {}: (no parameters)\n", f.name));
                    }
                }
            }
        }

        json!({
            "type": "object",
            "required": ["function"],
            "properties": {
                "function": {
                    "type": "string",
                    "description": format!("The {} function to call", server_name),
                    "enum": function_names
                },
                "params": {
                    "type": "object",
                    "description": params_description,
                    "additionalProperties": true
                }
            }
        })
    }

    /// Build a rich description listing all available functions.
    fn build_description(
        server_name: &str,
        custom: Option<&str>,
        functions: &[DiscoveredFunction],
    ) -> String {
        let mut desc = match custom {
            Some(c) if !c.is_empty() => c.to_string(),
            _ => format!("{} operations via MCP.", server_name),
        };

        if !functions.is_empty() {
            desc.push_str(" Available functions: ");
            let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
            desc.push_str(&names.join(", "));
            desc.push('.');

            // Add per-function descriptions for smaller function sets
            if functions.len() <= 20 {
                desc.push_str("\n\nFunction details:");
                for f in functions {
                    if !f.description.is_empty() {
                        desc.push_str(&format!("\n- {}: {}", f.name, f.description));
                    } else {
                        desc.push_str(&format!("\n- {}", f.name));
                    }
                }
            }
        }

        desc
    }

    /// Execute a function call on the MCP server.
    async fn call_function(&self, function: &str, params: Value) -> ToolResult {
        // Validate that the function exists
        if !self.functions.iter().any(|f| f.name == function) {
            let available: Vec<&str> = self.functions.iter().map(|f| f.name.as_str()).collect();
            return ToolResult::error(format!(
                "Unknown function '{}'. Available functions: {}",
                function,
                available.join(", ")
            ));
        }

        let peer = {
            let peer_guard = self.peer.read();
            match peer_guard.as_ref() {
                Some(p) => p.clone(),
                None => {
                    return ToolResult::error(format!(
                        "MCP server '{}' not initialized",
                        self.config.name
                    ));
                }
            }
        };

        let mut call_params = mcp_model::CallToolRequestParams::new(function.to_string());
        if let Value::Object(map) = params {
            call_params.arguments = Some(map.into_iter().collect());
        }

        match peer.call_tool(call_params).await {
            Ok(result) => {
                let output = result
                    .content
                    .iter()
                    .filter_map(|c| match &c.raw {
                        mcp_model::RawContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if result.is_error.unwrap_or(false) {
                    ToolResult::error(output)
                } else {
                    ToolResult::ok(output)
                }
            }
            Err(e) => ToolResult::error(format!("MCP function '{}' failed: {}", function, e)),
        }
    }

    /// Connect to an MCP server via stdio transport.
    async fn connect_stdio(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        server_name: &str,
    ) -> Result<RunningService<RoleClient, ()>, String> {
        use rmcp::transport::TokioChildProcess;
        use tokio::process::Command;

        let mut cmd = Command::new(command);
        cmd.args(args);
        for (key, value) in env {
            cmd.env(key, value);
        }

        let transport = TokioChildProcess::new(cmd)
            .map_err(|e| format!("Failed to spawn '{}': {}", command, e))?;

        let running: RunningService<RoleClient, ()> = ()
            .serve(transport)
            .await
            .map_err(|e| format!("Failed MCP handshake with '{}': {}", server_name, e))?;

        Ok(running)
    }

    /// Connect to an MCP server via HTTP/SSE transport.
    async fn connect_http(
        url: &str,
        headers: &HashMap<String, String>,
        server_name: &str,
    ) -> Result<RunningService<RoleClient, ()>, String> {
        use rmcp::transport::streamable_http_client::{
            StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
        };

        if headers.is_empty() {
            let transport = StreamableHttpClientTransport::from_uri(url);
            let running: RunningService<RoleClient, ()> = ()
                .serve(transport)
                .await
                .map_err(|e| format!("Failed HTTP MCP connection to '{}': {}", server_name, e))?;
            Ok(running)
        } else {
            use reqwest::header::{HeaderName, HeaderValue};

            let mut custom_headers = HashMap::new();
            for (key, value) in headers {
                let header_name = HeaderName::try_from(key.as_str())
                    .map_err(|e| format!("Invalid header name '{}': {}", key, e))?;
                let header_value = HeaderValue::try_from(value.as_str())
                    .map_err(|e| format!("Invalid header value for '{}': {}", key, e))?;
                custom_headers.insert(header_name, header_value);
            }

            let config =
                StreamableHttpClientTransportConfig::with_uri(url).custom_headers(custom_headers);
            let transport = StreamableHttpClientTransport::from_config(config);

            let running: RunningService<RoleClient, ()> = ()
                .serve(transport)
                .await
                .map_err(|e| format!("Failed HTTP MCP connection to '{}': {}", server_name, e))?;
            Ok(running)
        }
    }

    /// Gracefully shut down the MCP server connection.
    pub async fn shutdown(&self) {
        let running = self._running.write().take();
        if let Some(r) = running {
            let _ = r.cancel().await;
        }
        self.peer.write().take();
    }

    /// Check if a specific function requires HITL approval.
    pub fn requires_hitl(&self, function_name: &str) -> bool {
        self.config
            .security
            .hitl_functions
            .iter()
            .any(|f| f == function_name)
    }

    /// Get the number of discovered functions.
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Get the list of discovered function names.
    pub fn function_names(&self) -> Vec<&str> {
        self.functions.iter().map(|f| f.name.as_str()).collect()
    }
}

#[async_trait]
impl Tool for MCPWrapperTool {
    fn id(&self) -> &str {
        &self.config.name
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        // Extract the `function` field from input
        let function = match args.get("function").and_then(|v| v.as_str()) {
            Some(f) => f.to_string(),
            None => {
                let available: Vec<&str> = self.functions.iter().map(|f| f.name.as_str()).collect();
                return ToolResult::error(format!(
                    "'function' is required. Available functions: {}",
                    available.join(", ")
                ));
            }
        };

        // Extract optional `params` field (defaults to empty object)
        let params = args.get("params").cloned().unwrap_or_else(|| json!({}));

        // Per-function HITL: signal the runtime via metadata if approval is needed.
        // The runtime's HITL engine sees the tool ID ("github"), not the function.
        // For per-function granularity, we return metadata that the runtime can inspect.
        if self.requires_hitl(&function) {
            return ToolResult::ok_with_metadata(
                format!(
                    "Function '{}' on MCP server '{}' requires approval before execution.",
                    function, self.config.name
                ),
                HashMap::from([
                    ("_hitl_required".to_string(), json!(true)),
                    ("_hitl_function".to_string(), json!(function)),
                    ("_hitl_params".to_string(), params.clone()),
                    ("_hitl_tool".to_string(), json!(self.config.name)),
                ]),
            );
        }

        self.call_function(&function, params).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_wrapper_config_deserialize_stdio() {
        let yaml = r#"
name: github
type: mcp
transport: stdio
command: npx
args: ["-y", "@modelcontextprotocol/server-github"]
env:
  GITHUB_TOKEN: "test-token"
startup_timeout_ms: 15000
security:
  blocked_functions: [delete_repo]
  hitl_functions: [create_issue]
"#;
        let config: MCPWrapperConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "github");
        assert_eq!(config.startup_timeout_ms, 15000);
        assert_eq!(config.security.blocked_functions, vec!["delete_repo"]);
        assert_eq!(config.security.hitl_functions, vec!["create_issue"]);
    }

    #[test]
    fn test_mcp_wrapper_config_deserialize_http() {
        let yaml = r#"
name: custom_api
type: mcp
transport: http
url: "http://localhost:3000/mcp"
headers:
  Authorization: "Bearer test"
"#;
        let config: MCPWrapperConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "custom_api");
    }

    #[test]
    fn test_build_schema() {
        let functions = vec![
            DiscoveredFunction {
                name: "create_issue".to_string(),
                description: "Create a new issue".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["repo", "title"]
                }),
            },
            DiscoveredFunction {
                name: "list_repos".to_string(),
                description: "List repositories".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "org": {"type": "string"}
                    }
                }),
            },
        ];

        let schema = MCPWrapperTool::build_schema("github", &functions);

        assert_eq!(schema["type"], "object");
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("function"))
        );
        let func_enum = &schema["properties"]["function"]["enum"];
        assert!(
            func_enum
                .as_array()
                .unwrap()
                .contains(&json!("create_issue"))
        );
        assert!(func_enum.as_array().unwrap().contains(&json!("list_repos")));
    }

    #[test]
    fn test_build_description() {
        let functions = vec![
            DiscoveredFunction {
                name: "create_issue".to_string(),
                description: "Create a new issue".to_string(),
                input_schema: json!({}),
            },
            DiscoveredFunction {
                name: "list_repos".to_string(),
                description: "List repositories".to_string(),
                input_schema: json!({}),
            },
        ];

        let desc = MCPWrapperTool::build_description("github", None, &functions);

        assert!(desc.contains("github operations via MCP"));
        assert!(desc.contains("create_issue"));
        assert!(desc.contains("list_repos"));
        assert!(desc.contains("Create a new issue"));
    }

    #[test]
    fn test_requires_hitl() {
        let config = MCPWrapperConfig {
            name: "github".to_string(),
            transport: MCPWrapperTransport::Stdio {
                command: "npx".to_string(),
                args: vec![],
            },
            env: HashMap::new(),
            startup_timeout_ms: 30000,
            security: MCPWrapperSecurity {
                blocked_functions: vec![],
                hitl_functions: vec!["create_issue".to_string()],
            },
            description: None,
        };
        let tool = MCPWrapperTool::new(config);

        assert!(tool.requires_hitl("create_issue"));
        assert!(!tool.requires_hitl("list_repos"));
    }

    #[test]
    fn test_default_description() {
        let config = MCPWrapperConfig {
            name: "github".to_string(),
            transport: MCPWrapperTransport::Stdio {
                command: "npx".to_string(),
                args: vec![],
            },
            env: HashMap::new(),
            startup_timeout_ms: 30000,
            security: MCPWrapperSecurity::default(),
            description: None,
        };
        let tool = MCPWrapperTool::new(config);
        assert_eq!(tool.description(), "github operations via MCP");
    }

    #[test]
    fn test_custom_description() {
        let config = MCPWrapperConfig {
            name: "github".to_string(),
            transport: MCPWrapperTransport::Stdio {
                command: "npx".to_string(),
                args: vec![],
            },
            env: HashMap::new(),
            startup_timeout_ms: 30000,
            security: MCPWrapperSecurity::default(),
            description: Some("GitHub integration for DevOps".to_string()),
        };
        let tool = MCPWrapperTool::new(config);
        assert_eq!(tool.description(), "GitHub integration for DevOps");
    }
}
