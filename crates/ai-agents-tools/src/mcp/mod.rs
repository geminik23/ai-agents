//! MCP (Model Context Protocol) integration module.
//!
//! Each MCP server is exposed as a single builtin `Tool` via `MCPWrapperTool`, matching the dispatch pattern of other builtins (`datetime`, `http`, etc.).
//!
//! **Views** (`MCPViewTool`) expose named subsets of a wrapper's functions as separate tools for state-level scoping.

pub mod view;
pub mod wrapper;

pub use view::MCPViewTool;
pub use wrapper::{
    MCPViewConfig, MCPWrapperConfig, MCPWrapperSecurity, MCPWrapperTool, MCPWrapperTransport,
};
