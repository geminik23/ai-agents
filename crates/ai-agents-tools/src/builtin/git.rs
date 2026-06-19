use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use ai_agents_core::{
    Tool, ToolOperationKind, ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;

const DEFAULT_MAX_RESULTS: usize = 200;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;

/// Inspects repository status through a fixed read-only git command.
pub struct GitStatusTool;

impl GitStatusTool {
    /// Create a read-only repository status tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GitStatusTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Inspects bounded repository diffs through fixed read-only git commands.
pub struct GitDiffTool;

impl GitDiffTool {
    /// Create a read-only repository diff tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GitDiffTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GitStatusInput {
    /// Repository root or subdirectory. Defaults to current directory.
    #[serde(default)]
    path: Option<String>,
    /// Include untracked files. Defaults to true.
    #[serde(default = "default_true")]
    include_untracked: bool,
    /// Maximum changed paths. Defaults to 200.
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Serialize)]
struct GitStatusOutput {
    branch: Option<String>,
    staged: Vec<GitStatusEntry>,
    unstaged: Vec<GitStatusEntry>,
    untracked: Vec<GitStatusEntry>,
    count: usize,
    truncated: bool,
}

#[derive(Debug, Serialize, Clone)]
struct GitStatusEntry {
    path: String,
    status: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GitDiffInput {
    /// Repository root or subdirectory. Defaults to current directory.
    #[serde(default)]
    path: Option<String>,
    /// Show staged diff instead of working tree diff. Defaults to false.
    #[serde(default)]
    staged: bool,
    /// Optional path filters.
    #[serde(default)]
    paths: Vec<String>,
    /// Maximum output characters. Defaults to 20000.
    #[serde(default)]
    max_output_chars: Option<usize>,
}

#[derive(Debug, Serialize)]
struct GitDiffOutput {
    staged: bool,
    paths: Vec<String>,
    summary: Vec<String>,
    diff: String,
    truncated: bool,
}

#[async_trait]
impl Tool for GitStatusTool {
    fn id(&self) -> &str {
        "git_status"
    }

    fn name(&self) -> &str {
        "Git Status"
    }

    fn description(&self) -> &str {
        "Inspect repository status using bounded read-only git status output."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GitStatusInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        vcs_metadata()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: GitStatusInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let cwd = PathBuf::from(input.path.unwrap_or_else(|| ".".to_string()));
        if let Err(reason) = validate_path(&cwd) {
            return ToolResult::error(reason);
        }
        let branch = run_git(&cwd, &["branch", "--show-current"])
            .ok()
            .map(|output| output.trim().to_string())
            .filter(|output| !output.is_empty());
        let mut args = vec!["status", "--porcelain=v1", "--branch"];
        if !input.include_untracked {
            args.push("--untracked-files=no");
        }
        let raw = match run_git(&cwd, &args) {
            Ok(raw) => raw,
            Err(error) => return ToolResult::error(error),
        };
        let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
        let mut staged = Vec::new();
        let mut unstaged = Vec::new();
        let mut untracked = Vec::new();
        for line in raw.lines() {
            if line.starts_with("##") || line.len() < 3 {
                continue;
            }
            let bytes = line.as_bytes();
            let x = bytes[0] as char;
            let y = bytes[1] as char;
            let path = line[3..].to_string();
            if path_contains_git(&path) {
                continue;
            }
            if x == '?' && y == '?' {
                untracked.push(GitStatusEntry {
                    path,
                    status: "untracked".to_string(),
                });
                continue;
            }
            if x != ' ' {
                staged.push(GitStatusEntry {
                    path: path.clone(),
                    status: x.to_string(),
                });
            }
            if y != ' ' {
                unstaged.push(GitStatusEntry {
                    path,
                    status: y.to_string(),
                });
            }
        }
        let total = staged.len() + unstaged.len() + untracked.len();
        truncate_status_entries(&mut staged, max_results);
        let remaining = max_results.saturating_sub(staged.len());
        truncate_status_entries(&mut unstaged, remaining);
        let remaining = max_results.saturating_sub(staged.len() + unstaged.len());
        truncate_status_entries(&mut untracked, remaining);
        let count = staged.len() + unstaged.len() + untracked.len();
        let output = GitStatusOutput {
            branch,
            staged,
            unstaged,
            untracked,
            count,
            truncated: count < total,
        };
        json_result(&output, output.truncated, None)
    }
}

#[async_trait]
impl Tool for GitDiffTool {
    fn id(&self) -> &str {
        "git_diff"
    }

    fn name(&self) -> &str {
        "Git Diff"
    }

    fn description(&self) -> &str {
        "Inspect bounded repository diffs using fixed read-only git diff commands."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GitDiffInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        vcs_metadata()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: GitDiffInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let cwd = PathBuf::from(input.path.clone().unwrap_or_else(|| ".".to_string()));
        if let Err(reason) = validate_path(&cwd) {
            return ToolResult::error(reason);
        }
        for path in &input.paths {
            if path_contains_git(path) {
                return ToolResult::error("Path filters cannot reference raw .git contents");
            }
        }
        let max_output_chars = input.max_output_chars.unwrap_or(DEFAULT_MAX_OUTPUT_CHARS);
        let mut diff_args = vec![
            "diff",
            "--no-ext-diff",
            "--src-prefix=a/",
            "--dst-prefix=b/",
        ];
        if input.staged {
            diff_args.push("--cached");
        }
        let mut summary_args = vec!["diff", "--name-status"];
        if input.staged {
            summary_args.push("--cached");
        }
        if !input.paths.is_empty() {
            diff_args.push("--");
            summary_args.push("--");
            for path in &input.paths {
                diff_args.push(path);
                summary_args.push(path);
            }
        }
        let summary_raw = run_git(&cwd, &summary_args).unwrap_or_default();
        let diff_raw = match run_git(&cwd, &diff_args) {
            Ok(diff) => diff,
            Err(error) => return ToolResult::error(error),
        };
        let (diff, truncated) = truncate_chars(diff_raw, max_output_chars);
        let output = GitDiffOutput {
            staged: input.staged,
            paths: input.paths,
            summary: summary_raw.lines().map(str::to_string).collect(),
            diff,
            truncated,
        };
        json_result(&output, truncated, Some(max_output_chars))
    }
}

fn vcs_metadata() -> ToolSafetyMetadata {
    ToolSafetyMetadata {
        read_only: true,
        concurrency_safe: true,
        operation: ToolOperationKind::VcsInspect,
        side_effect_level: ToolSideEffectLevel::None,
        requires_network: false,
        destructive: false,
        open_world: false,
        host_dependent: true,
        requires_user_interaction: false,
        supports_cancellation: false,
        default_requires_approval: false,
        should_defer_schema: false,
        max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
    }
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("Failed to run git: {}", error))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("git exited with status {}", output.status)
        } else {
            stderr
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn validate_path(path: &Path) -> Result<(), String> {
    for component in path.components() {
        let Component::Normal(value) = component else {
            continue;
        };
        if value.to_string_lossy() == ".git" {
            return Err(
                "VCS tools inspect repository metadata but do not expose raw .git paths"
                    .to_string(),
            );
        }
    }
    Ok(())
}

fn path_contains_git(path: &str) -> bool {
    Path::new(path).components().any(|component| {
        matches!(component, Component::Normal(value) if value.to_string_lossy() == ".git")
    })
}

fn truncate_status_entries(entries: &mut Vec<GitStatusEntry>, max_len: usize) {
    entries.truncate(max_len);
}

fn truncate_chars(text: String, max_chars: usize) -> (String, bool) {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        (truncated, true)
    } else {
        (text, false)
    }
}

fn json_result<T: Serialize>(
    output: &T,
    truncated: bool,
    max_output_chars: Option<usize>,
) -> ToolResult {
    let json = match serde_json::to_string(output) {
        Ok(json) => json,
        Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
    };
    let mut metadata = HashMap::new();
    metadata.insert("truncated".to_string(), Value::Bool(truncated));
    if let Some(max) = max_output_chars {
        metadata.insert("max_output_chars".to_string(), Value::from(max));
    }
    ToolResult::ok_with_metadata(json, metadata)
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn init_repo() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        run_git(dir.path(), &["init"]).unwrap();
        run_git(dir.path(), &["config", "user.email", "test@example.com"]).unwrap();
        run_git(dir.path(), &["config", "user.name", "Test User"]).unwrap();
        dir
    }

    #[tokio::test]
    async fn git_status_reports_untracked_files() {
        if !git_available() {
            return;
        }
        let dir = init_repo();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        let result = GitStatusTool::new()
            .execute(serde_json::json!({"path": dir.path()}))
            .await;
        assert!(result.success);
        assert!(result.output.contains("untracked"));
    }

    #[tokio::test]
    async fn git_diff_returns_bounded_diff() {
        if !git_available() {
            return;
        }
        let dir = init_repo();
        std::fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
        run_git(dir.path(), &["add", "a.txt"]).unwrap();
        run_git(dir.path(), &["commit", "-m", "initial"]).unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\n").unwrap();
        let result = GitDiffTool::new()
            .execute(serde_json::json!({"path": dir.path(), "max_output_chars": 20}))
            .await;
        assert!(result.success);
        assert!(result.output.contains("truncated"));
    }
}
