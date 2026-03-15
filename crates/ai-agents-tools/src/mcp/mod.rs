//! MCP (Model Context Protocol) integration module.
//!
//! Each MCP server is exposed as a single builtin `Tool` via `MCPWrapperTool`,
//! matching the dispatch pattern of other builtins (`datetime`, `http`, etc.).

pub mod wrapper;

pub use wrapper::{MCPWrapperConfig, MCPWrapperSecurity, MCPWrapperTool, MCPWrapperTransport};
