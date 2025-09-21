mod calculator;
mod datetime;
mod echo;
mod json;
mod random;

#[cfg(feature = "http-tool")]
mod http;

pub use calculator::CalculatorTool;
pub use datetime::DateTimeTool;
pub use echo::EchoTool;
pub use json::JsonTool;
pub use random::RandomTool;

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
        #[cfg(feature = "http-tool")]
        "http" => Some(Arc::new(HttpTool::new())),
        _ => None,
    }
}
