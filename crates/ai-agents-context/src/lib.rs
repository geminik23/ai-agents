//! Dynamic context management for AI Agents framework

mod builtin;
mod instructions;
mod manager;
mod provider;
mod render;
mod source;

pub use instructions::ScopedInstructionFiles;
pub use manager::ContextManager;
pub use provider::ContextProvider;
pub use render::TemplateRenderer;
pub use source::{BuiltinSource, ContextSource, RefreshPolicy};
