mod calculator;
mod echo;

pub use calculator::CalculatorTool;
pub use echo::EchoTool;

use std::sync::Arc;
use super::Tool;

pub fn all_builtin_tools() -> Vec<Arc<dyn Tool>> {
    vec![Arc::new(CalculatorTool::new()), Arc::new(EchoTool::new())]
}

pub fn get_builtin_tool(id: &str) -> Option<Arc<dyn Tool>> {
    match id {
        "calculator" => Some(Arc::new(CalculatorTool::new())),
        "echo" => Some(Arc::new(EchoTool::new())),
        _ => None,
    }
}
