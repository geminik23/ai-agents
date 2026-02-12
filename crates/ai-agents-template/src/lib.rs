//! Template system for loading and processing agent specifications

mod inheritance;
mod loader;
mod renderer;

pub use inheritance::TemplateInheritance;
pub use loader::TemplateLoader;
pub use renderer::TemplateRenderer;
