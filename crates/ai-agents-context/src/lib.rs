//! Dynamic context management for AI Agents framework

mod builtin;
mod manager;
mod provider;
mod render;
mod source;

pub use manager::ContextManager;
pub use provider::ContextProvider;
pub use render::TemplateRenderer;
pub use source::{BuiltinSource, ContextSource, RefreshPolicy};
