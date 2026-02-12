mod calculator;
mod datetime;
mod echo;
mod file;
mod json;
mod math;
mod random;
mod template;
mod text;

#[cfg(feature = "http-tool")]
mod http;

pub use calculator::CalculatorTool;
pub use datetime::DateTimeTool;
pub use echo::EchoTool;
pub use file::FileTool;
pub use json::JsonTool;
pub use math::MathTool;
pub use random::RandomTool;
pub use template::TemplateTool;
pub use text::TextTool;

#[cfg(feature = "http-tool")]
pub use http::HttpTool;

use super::Tool;
use std::sync::Arc;

pub fn all_builtin_tools() -> Vec<Arc<dyn Tool>> {
    #[allow(unused_mut)]
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(CalculatorTool::new()),
        Arc::new(EchoTool::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(JsonTool::new()),
        Arc::new(RandomTool::new()),
        Arc::new(FileTool::new()),
        Arc::new(TextTool::new()),
        Arc::new(TemplateTool::new()),
        Arc::new(MathTool::new()),
    ];

    #[cfg(feature = "http-tool")]
    tools.push(Arc::new(HttpTool::new()));

    tools
}

pub fn get_builtin_tool(id: &str) -> Option<Arc<dyn Tool>> {
    match id {
        "calculator" => Some(Arc::new(CalculatorTool::new())),
        "echo" => Some(Arc::new(EchoTool::new())),
        "datetime" => Some(Arc::new(DateTimeTool::new())),
        "json" => Some(Arc::new(JsonTool::new())),
        "random" => Some(Arc::new(RandomTool::new())),
        "file" => Some(Arc::new(FileTool::new())),
        "text" => Some(Arc::new(TextTool::new())),
        "template" => Some(Arc::new(TemplateTool::new())),
        "math" => Some(Arc::new(MathTool::new())),
        #[cfg(feature = "http-tool")]
        "http" => Some(Arc::new(HttpTool::new())),
        _ => None,
    }
}
