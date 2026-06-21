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

pub use calculator::CalculatorTool;
pub use command::CommandTool;
pub use datetime::DateTimeTool;
pub use echo::EchoTool;
pub use file::FileTool;
pub use fs_mutation::{FileEditTool, FileWriteTool, PatchTool};
pub use fs_readonly::{FileInfoTool, FileListTool, FileReadTool, GlobTool, GrepTool};
pub use git::{GitDiffTool, GitStatusTool};
pub use host::{AskUserTool, DiagnosticsTool, SleepTool, TodoTool};
pub use http::HttpTool;
pub use json::JsonTool;
pub use math::MathTool;
pub use random::RandomTool;
pub use template::TemplateTool;
pub use text::TextTool;
pub use web_fetch::WebFetchTool;

use super::Tool;
use crate::types::{TodoStore, UnavailableDiagnosticsProvider};
use parking_lot::RwLock;
use std::sync::Arc;

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
        Arc::new(CommandTool::new(command_runner)),
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
        "glob" => Some(Arc::new(GlobTool::new())),
        "grep" => Some(Arc::new(GrepTool::new())),
        "file_read" => Some(Arc::new(FileReadTool::new())),
        "file_write" => Some(Arc::new(FileWriteTool::new())),
        "file_edit" => Some(Arc::new(FileEditTool::new())),
        "patch" => Some(Arc::new(PatchTool::new())),
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
