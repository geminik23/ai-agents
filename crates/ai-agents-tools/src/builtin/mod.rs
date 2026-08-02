mod calculator;
mod command;
mod datetime;
mod echo;
mod file;
mod fs_mutation;
mod fs_readonly;
mod git;
mod host;
mod http;
mod json;
mod math;
mod random;
mod template;
mod text;
mod web_fetch;
mod web_search;

pub use calculator::CalculatorTool;
pub use command::CommandTool;
pub use datetime::DateTimeTool;
pub use echo::EchoTool;
pub use file::FileTool;
pub use fs_mutation::{
    CopyPathTool, DeletePathTool, FileEditTool, FileWriteTool, MovePathTool, PatchTool,
};
pub use fs_readonly::{FileInfoTool, FileListTool, FileReadTool, GlobTool, GrepTool};
pub use git::{GitDiffTool, GitStatusTool};
pub use host::{AskUserTool, DiagnosticsTool, SleepTool, TodoTool};
pub use http::HttpTool;
pub use json::JsonTool;
pub use math::MathTool;
pub use random::RandomTool;
pub use template::TemplateTool;
pub use text::TextTool;
pub use web_fetch::{
    WebFetchResolver, WebFetchTool, WebFetchTransport, WebFetchTransportRequest,
    WebFetchTransportResponse,
};
pub use web_search::WebSearchTool;

use super::Tool;
use crate::types::{TodoStore, UnavailableDiagnosticsProvider};
use parking_lot::RwLock;
use std::sync::Arc;

const BUILTIN_TOOL_IDS: [&str; 30] = [
    "calculator",
    "echo",
    "datetime",
    "json",
    "random",
    "file",
    "glob",
    "grep",
    "file_read",
    "file_write",
    "file_edit",
    "patch",
    "copy_path",
    "move_path",
    "delete_path",
    "file_list",
    "file_info",
    "git_status",
    "git_diff",
    "diagnostics",
    "ask_user",
    "todo",
    "sleep",
    "web_fetch",
    "web_search",
    "command",
    "text",
    "template",
    "math",
    "http",
];

pub fn all_builtin_tools() -> Vec<Arc<dyn Tool>> {
    let versions = crate::types::FileVersionStore::default();
    let command_runner = Arc::new(RwLock::new(
        Arc::new(crate::types::UnavailableCommandRunner) as Arc<dyn crate::types::CommandRunner>,
    ));
    vec![
        Arc::new(CalculatorTool::new()),
        Arc::new(EchoTool::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(JsonTool::new()),
        Arc::new(RandomTool::new()),
        Arc::new(FileTool::new()),
        Arc::new(GlobTool::new()),
        Arc::new(GrepTool::new()),
        Arc::new(FileReadTool::with_version_store(versions.clone())),
        Arc::new(FileWriteTool::with_version_store(versions.clone())),
        Arc::new(FileEditTool::with_version_store(versions.clone())),
        Arc::new(PatchTool::with_version_store(versions.clone())),
        Arc::new(CopyPathTool::new()),
        Arc::new(MovePathTool::new()),
        Arc::new(DeletePathTool::new()),
        Arc::new(FileListTool::new()),
        Arc::new(FileInfoTool::new()),
        Arc::new(GitStatusTool::new()),
        Arc::new(GitDiffTool::new()),
        Arc::new(DiagnosticsTool::new(Arc::new(RwLock::new(Arc::new(
            UnavailableDiagnosticsProvider,
        ))))),
        Arc::new(AskUserTool::new(Arc::new(RwLock::new(None)))),
        Arc::new(TodoTool::new(TodoStore::default())),
        Arc::new(SleepTool::new()),
        Arc::new(WebFetchTool::new()),
        Arc::new(WebSearchTool::new()),
        Arc::new(CommandTool::new(command_runner)),
        Arc::new(TextTool::new()),
        Arc::new(TemplateTool::new()),
        Arc::new(MathTool::new()),
        Arc::new(HttpTool::new()),
    ]
}

pub fn get_builtin_tool(id: &str) -> Option<Arc<dyn Tool>> {
    if !BUILTIN_TOOL_IDS.contains(&id) {
        return None;
    }

    match id {
        "calculator" => Some(Arc::new(CalculatorTool::new())),
        "echo" => Some(Arc::new(EchoTool::new())),
        "datetime" => Some(Arc::new(DateTimeTool::new())),
        "json" => Some(Arc::new(JsonTool::new())),
        "random" => Some(Arc::new(RandomTool::new())),
        "file" => Some(Arc::new(FileTool::new())),
        "glob" => Some(Arc::new(GlobTool::new())),
        "grep" => Some(Arc::new(GrepTool::new())),
        "file_read" => Some(Arc::new(FileReadTool::new())),
        "file_write" => Some(Arc::new(FileWriteTool::new())),
        "file_edit" => Some(Arc::new(FileEditTool::new())),
        "patch" => Some(Arc::new(PatchTool::new())),
        "copy_path" => Some(Arc::new(CopyPathTool::new())),
        "move_path" => Some(Arc::new(MovePathTool::new())),
        "delete_path" => Some(Arc::new(DeletePathTool::new())),
        "file_list" => Some(Arc::new(FileListTool::new())),
        "file_info" => Some(Arc::new(FileInfoTool::new())),
        "git_status" => Some(Arc::new(GitStatusTool::new())),
        "git_diff" => Some(Arc::new(GitDiffTool::new())),
        "diagnostics" => Some(Arc::new(DiagnosticsTool::new(Arc::new(RwLock::new(
            Arc::new(UnavailableDiagnosticsProvider),
        ))))),
        "ask_user" => Some(Arc::new(AskUserTool::new(Arc::new(RwLock::new(None))))),
        "todo" => Some(Arc::new(TodoTool::new(TodoStore::default()))),
        "sleep" => Some(Arc::new(SleepTool::new())),
        "web_fetch" => Some(Arc::new(WebFetchTool::new())),
        "web_search" => Some(Arc::new(WebSearchTool::new())),
        "command" => Some(Arc::new(CommandTool::new(Arc::new(RwLock::new(Arc::new(
            crate::types::UnavailableCommandRunner,
        )
            as Arc<dyn crate::types::CommandRunner>))))),
        "text" => Some(Arc::new(TextTool::new())),
        "template" => Some(Arc::new(TemplateTool::new())),
        "math" => Some(Arc::new(MathTool::new())),
        "http" => Some(Arc::new(HttpTool::new())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn builtin_inventory_and_lookup_stay_in_sync() {
        let tools = all_builtin_tools();
        let actual: BTreeSet<_> = tools.iter().map(|tool| tool.id()).collect();
        let expected: BTreeSet<_> = BUILTIN_TOOL_IDS.into_iter().collect();

        assert_eq!(tools.len(), BUILTIN_TOOL_IDS.len());
        assert_eq!(actual.len(), tools.len(), "built-in IDs must be unique");
        assert_eq!(actual, expected);

        for id in BUILTIN_TOOL_IDS {
            let tool =
                get_builtin_tool(id).unwrap_or_else(|| panic!("missing built-in lookup: {id}"));
            assert_eq!(tool.id(), id);
        }
    }
}
