mod calculator;
mod datetime;
mod echo;
mod file;
mod http;
mod json;
mod math;
mod random;
mod template;
mod text;

pub use calculator::CalculatorTool;
pub use datetime::DateTimeTool;
pub use echo::EchoTool;
pub use file::FileTool;
pub use http::HttpTool;
pub use json::JsonTool;
pub use math::MathTool;
pub use random::RandomTool;
pub use template::TemplateTool;
pub use text::TextTool;

use super::Tool;
use std::sync::Arc;

pub fn all_builtin_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(CalculatorTool::new()),
        Arc::new(EchoTool::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(JsonTool::new()),
        Arc::new(RandomTool::new()),
        Arc::new(FileTool::new()),
        Arc::new(TextTool::new()),
        Arc::new(TemplateTool::new()),
        Arc::new(MathTool::new()),
        Arc::new(HttpTool::new()),
    ]
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
        "http" => Some(Arc::new(HttpTool::new())),
        _ => None,
    }
}
