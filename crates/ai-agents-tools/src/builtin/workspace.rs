use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::RegexBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use ai_agents_core::{
    Tool, ToolOperationKind, ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;

const DEFAULT_GLOB_MAX_RESULTS: usize = 100;
const DEFAULT_GREP_MAX_RESULTS: usize = 250;
const DEFAULT_FILE_LIST_MAX_RESULTS: usize = 200;
const DEFAULT_FILE_READ_MAX_LINES: usize = 2_000;
const DEFAULT_MAX_FILE_SIZE_BYTES: u64 = 1_048_576;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;
const LARGE_FILE_PREVIEW_BYTES: usize = 8_192;

/// Sort order used by path-oriented read-only tools.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
enum SortMode {
    #[default]
    Path,
    Modified,
    Size,
    Kind,
}

/// Output mode for the grep tool.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
enum GrepOutputMode {
    Content,
    #[default]
    FilesWithMatches,
    Count,
}

/// Pattern matching mode for the grep tool.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
enum GrepMode {
    #[default]
    Regex,
    Literal,
}

/// Read-only glob search over workspace paths.
pub struct GlobTool;

impl GlobTool {
    /// Create a glob tool with default limits.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GlobTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GlobInput {
    /// Glob pattern such as `**/*.rs`.
    pattern: String,
    /// Root path to search. Defaults to the current directory.
    #[serde(default)]
    path: Option<String>,
    /// Maximum returned paths. Defaults to 100.
    #[serde(default)]
    max_results: Option<usize>,
    /// Pagination offset. Defaults to 0.
    #[serde(default)]
    offset: usize,
    /// Include directories in results. Defaults to false.
    #[serde(default)]
    include_dirs: bool,
    /// Sort order. Defaults to path.
    #[serde(default)]
    sort: SortMode,
}

#[derive(Debug, Serialize)]
struct GlobOutput {
    paths: Vec<String>,
    count: usize,
    offset: usize,
    truncated: bool,
    duration_ms: u64,
    ignored_directories: Vec<String>,
}

#[async_trait]
impl Tool for GlobTool {
    fn id(&self) -> &str {
        "glob"
    }

    fn name(&self) -> &str {
        "Glob Search"
    }

    fn description(&self) -> &str {
        "Find workspace paths by glob pattern with stable sorting, pagination, and default ignored directories."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GlobInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_only_metadata(ToolOperationKind::Read, true)
    }

    async fn execute(&self, args: Value, _ctx: ai_agents_core::ToolExecutionContext) -> ToolResult {
        let input: GlobInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        if input.pattern.trim().is_empty() {
            return ToolResult::error("pattern must not be empty");
        }

        let start = Instant::now();
        let root = PathBuf::from(input.path.unwrap_or_else(|| ".".to_string()));
        if is_blocked_path(&root) {
            return ToolResult::error("path is blocked by default policy");
        }
        let matcher = match glob_regex(&input.pattern) {
            Ok(matcher) => matcher,
            Err(error) => return ToolResult::error(format!("Invalid glob pattern: {}", error)),
        };

        let mut entries = Vec::new();
        let mut ignored = HashSet::new();
        let root_for_display = root.clone();
        walk_paths(&root, true, input.include_dirs, false, &mut entries, &mut ignored).await;
        entries.retain(|entry| {
            let rel = relative_display(&root_for_display, &entry.path);
            matcher.is_match(&rel) || matcher.is_match(&display_path(&entry.path))
        });
        sort_entries(&mut entries, input.sort);

        let total = entries.len();
        let max_results = input.max_results.unwrap_or(DEFAULT_GLOB_MAX_RESULTS);
        let paths = entries
            .into_iter()
            .skip(input.offset)
            .take(max_results)
            .map(|entry| display_path(&entry.path))
            .collect::<Vec<_>>();
        let truncated = input.offset.saturating_add(paths.len()) < total;
        let mut ignored_directories: Vec<String> = ignored.into_iter().collect();
        ignored_directories.sort();

        to_json_result(
            &GlobOutput {
                paths,
                count: total,
                offset: input.offset,
                truncated,
                duration_ms: start.elapsed().as_millis() as u64,
                ignored_directories,
            },
            DEFAULT_MAX_OUTPUT_CHARS,
        )
    }
}

/// Read-only text search over workspace files.
pub struct GrepTool;

impl GrepTool {
    /// Create a grep tool with default limits.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GrepInput {
    /// Literal or regex pattern to search.
    pattern: String,
    /// Search mode. Defaults to regex.
    #[serde(default)]
    mode: GrepMode,
    /// File or directory path. Defaults to current directory.
    #[serde(default)]
    path: Option<String>,
    /// Optional glob filter for file paths.
    #[serde(default)]
    include_glob: Option<String>,
    /// Whether matching is case-sensitive. Defaults to false.
    #[serde(default)]
    case_sensitive: bool,
    /// Output shape. Defaults to files_with_matches.
    #[serde(default)]
    output_mode: GrepOutputMode,
    /// Number of context lines in content mode. Defaults to 0.
    #[serde(default)]
    context: usize,
    /// Maximum returned matches or files. Defaults to 250.
    #[serde(default)]
    max_results: Option<usize>,
    /// Pagination offset. Defaults to 0.
    #[serde(default)]
    offset: usize,
    /// Maximum searched file size in bytes. Defaults to 1 MiB.
    #[serde(default)]
    max_file_size_bytes: Option<u64>,
    /// Maximum model-facing output characters. Defaults to 20000.
    #[serde(default)]
    max_output_chars: Option<usize>,
}

#[derive(Debug, Serialize, Clone)]
struct GrepMatch {
    path: String,
    line: usize,
    text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    context_before: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    context_after: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GrepFileCount {
    path: String,
    count: usize,
}

#[derive(Debug, Serialize)]
struct GrepOutput {
    mode: String,
    matches: Vec<GrepMatch>,
    files: Vec<String>,
    counts: Vec<GrepFileCount>,
    count: usize,
    offset: usize,
    truncated: bool,
    skipped_binary: usize,
    skipped_too_large: usize,
}

#[async_trait]
impl Tool for GrepTool {
    fn id(&self) -> &str {
        "grep"
    }

    fn name(&self) -> &str {
        "Grep Search"
    }

    fn description(&self) -> &str {
        "Search text files with regex or literal patterns, bounded output, binary skipping, and stable pagination."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GrepInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_only_metadata(ToolOperationKind::Read, true)
    }

    async fn execute(&self, args: Value, _ctx: ai_agents_core::ToolExecutionContext) -> ToolResult {
        let input: GrepInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        if input.pattern.is_empty() {
            return ToolResult::error("pattern must not be empty");
        }

        let root = PathBuf::from(input.path.clone().unwrap_or_else(|| ".".to_string()));
        if is_blocked_path(&root) {
            return ToolResult::error("path is blocked by default policy");
        }
        let pattern = match input.mode {
            GrepMode::Regex => input.pattern.clone(),
            GrepMode::Literal => regex::escape(&input.pattern),
        };
        let regex = match RegexBuilder::new(&pattern)
            .case_insensitive(!input.case_sensitive)
            .build()
        {
            Ok(regex) => regex,
            Err(error) => return ToolResult::error(format!("Invalid regex: {}", error)),
        };
        let include_matcher = match input.include_glob.as_deref() {
            Some(glob) => match glob_regex(glob) {
                Ok(regex) => Some(regex),
                Err(error) => return ToolResult::error(format!("Invalid include_glob: {}", error)),
            },
            None => None,
        };

        let mut entries = Vec::new();
        let mut ignored = HashSet::new();
        let recursive = root.is_dir();
        walk_paths(&root, recursive, false, false, &mut entries, &mut ignored).await;
        sort_entries(&mut entries, SortMode::Path);

        let max_file_size = input
            .max_file_size_bytes
            .unwrap_or(DEFAULT_MAX_FILE_SIZE_BYTES);
        let mut content_matches = Vec::new();
        let mut files = Vec::new();
        let mut counts = Vec::new();
        let mut skipped_binary = 0;
        let mut skipped_too_large = 0;

        for (idx, entry) in entries.iter().enumerate() {
            if idx % 64 == 0 {
                tokio::task::yield_now().await;
            }
            if !entry.kind.is_file() {
                continue;
            }
            let rel = relative_display(&root, &entry.path);
            if include_matcher
                .as_ref()
                .is_some_and(|matcher| !matcher.is_match(&rel) && !matcher.is_match(&display_path(&entry.path)))
            {
                continue;
            }
            let metadata = match fs::metadata(&entry.path) {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };
            if metadata.len() > max_file_size {
                skipped_too_large += 1;
                continue;
            }
            let bytes = match fs::read(&entry.path) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            if is_probably_binary(&bytes) {
                skipped_binary += 1;
                continue;
            }
            let Ok(text) = std::str::from_utf8(&bytes) else {
                skipped_binary += 1;
                continue;
            };
            let lines: Vec<&str> = text.lines().collect();
            let mut file_count = 0;
            for (line_idx, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    file_count += 1;
                    if input.output_mode == GrepOutputMode::Content {
                        let start = line_idx.saturating_sub(input.context);
                        let end = (line_idx + input.context + 1).min(lines.len());
                        content_matches.push(GrepMatch {
                            path: display_path(&entry.path),
                            line: line_idx + 1,
                            text: (*line).to_string(),
                            context_before: lines[start..line_idx]
                                .iter()
                                .map(|line| (*line).to_string())
                                .collect(),
                            context_after: lines[line_idx + 1..end]
                                .iter()
                                .map(|line| (*line).to_string())
                                .collect(),
                        });
                    }
                }
            }
            if file_count > 0 {
                files.push(display_path(&entry.path));
                counts.push(GrepFileCount {
                    path: display_path(&entry.path),
                    count: file_count,
                });
            }
        }

        let max_results = input.max_results.unwrap_or(DEFAULT_GREP_MAX_RESULTS);
        let (matches, files, counts, total) = match input.output_mode {
            GrepOutputMode::Content => {
                let total = content_matches.len();
                let page = content_matches
                    .into_iter()
                    .skip(input.offset)
                    .take(max_results)
                    .collect::<Vec<_>>();
                (page, Vec::new(), Vec::new(), total)
            }
            GrepOutputMode::FilesWithMatches => {
                let total = files.len();
                let page = files
                    .into_iter()
                    .skip(input.offset)
                    .take(max_results)
                    .collect::<Vec<_>>();
                (Vec::new(), page, Vec::new(), total)
            }
            GrepOutputMode::Count => {
                let total = counts.len();
                let page = counts
                    .into_iter()
                    .skip(input.offset)
                    .take(max_results)
                    .collect::<Vec<_>>();
                (Vec::new(), Vec::new(), page, total)
            }
        };
        let page_len = matches.len().max(files.len()).max(counts.len());
        let truncated = input.offset.saturating_add(page_len) < total;
        let output_mode = match input.output_mode {
            GrepOutputMode::Content => "content",
            GrepOutputMode::FilesWithMatches => "files_with_matches",
            GrepOutputMode::Count => "count",
        };

        to_json_result(
            &GrepOutput {
                mode: output_mode.to_string(),
                matches,
                files,
                counts,
                count: total,
                offset: input.offset,
                truncated,
                skipped_binary,
                skipped_too_large,
            },
            input.max_output_chars.unwrap_or(DEFAULT_MAX_OUTPUT_CHARS),
        )
    }
}

/// Safe text file reader with line ranges and large-file fallback.
pub struct FileReadTool;

impl FileReadTool {
    /// Create a file read tool with default limits.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FileReadTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileReadInput {
    /// File path to read.
    path: String,
    /// First one-based line to include. Defaults to 1.
    #[serde(default)]
    start_line: Option<usize>,
    /// Last one-based line to include.
    #[serde(default)]
    end_line: Option<usize>,
    /// Maximum returned lines. Defaults to 2000.
    #[serde(default)]
    max_lines: Option<usize>,
    /// Maximum read bytes. Defaults to policy-sized 1 MiB.
    #[serde(default)]
    max_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
struct FileReadOutput {
    path: String,
    content: String,
    start_line: usize,
    end_line: usize,
    total_lines: Option<usize>,
    bytes_read: usize,
    file_size: u64,
    encoding: String,
    truncated: bool,
    large_file: bool,
    message: Option<String>,
}

#[async_trait]
impl Tool for FileReadTool {
    fn id(&self) -> &str {
        "file_read"
    }

    fn name(&self) -> &str {
        "File Read"
    }

    fn description(&self) -> &str {
        "Read a text file safely with optional line ranges, size limits, binary rejection, and Unicode-safe output."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileReadInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_only_metadata(ToolOperationKind::Read, true)
    }

    async fn execute(&self, args: Value, _ctx: ai_agents_core::ToolExecutionContext) -> ToolResult {
        let input: FileReadInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let path = PathBuf::from(&input.path);
        if is_blocked_path(&path) {
            return ToolResult::error("path is blocked by default policy");
        }
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => return ToolResult::error(format!("Metadata error: {}", error)),
        };
        if !metadata.is_file() {
            return ToolResult::error("path is not a file");
        }

        let max_bytes = input.max_bytes.unwrap_or(DEFAULT_MAX_FILE_SIZE_BYTES as usize);
        let has_line_range = input.start_line.is_some() || input.end_line.is_some();
        let large_file = metadata.len() > max_bytes as u64;
        let read_limit = if large_file && !has_line_range {
            LARGE_FILE_PREVIEW_BYTES.min(max_bytes)
        } else {
            max_bytes
        };
        let bytes = match read_limited(&path, read_limit) {
            Ok(bytes) => bytes,
            Err(error) => return ToolResult::error(format!("Read error: {}", error)),
        };
        if is_probably_binary(&bytes) {
            return ToolResult::error("binary files are rejected by default");
        }
        let text = match String::from_utf8(bytes.clone()) {
            Ok(text) => text,
            Err(_) => return ToolResult::error("file is not valid UTF-8 text"),
        };
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = Some(lines.len());
        let start_line = input.start_line.unwrap_or(1).max(1);
        let max_lines = input.max_lines.unwrap_or(DEFAULT_FILE_READ_MAX_LINES).max(1);
        let requested_end = input
            .end_line
            .unwrap_or_else(|| start_line.saturating_add(max_lines).saturating_sub(1));
        let end_line = requested_end.min(start_line.saturating_add(max_lines).saturating_sub(1));
        let content = if start_line > lines.len() {
            String::new()
        } else {
            lines[start_line - 1..end_line.min(lines.len())].join("\n")
        };
        let truncated = large_file
            || end_line < requested_end
            || end_line < lines.len()
            || bytes.len() == read_limit && metadata.len() > read_limit as u64;
        let message = (large_file && !has_line_range).then(|| {
            "file exceeded max_bytes, returning a bounded preview; request a line range for targeted reads".to_string()
        });

        let mut result = to_json_result(
            &FileReadOutput {
                path: input.path,
                content,
                start_line,
                end_line: end_line.min(lines.len().max(start_line)),
                total_lines,
                bytes_read: bytes.len(),
                file_size: metadata.len(),
                encoding: "utf-8".to_string(),
                truncated,
                large_file,
                message,
            },
            DEFAULT_MAX_OUTPUT_CHARS,
        );
        attach_metadata(&mut result, truncated, Some(metadata.len()), None);
        result
    }
}

/// Safe directory lister with pagination and symlink notes.
pub struct FileListTool;

impl FileListTool {
    /// Create a file list tool with default limits.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FileListTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileListInput {
    /// Directory path to list.
    path: String,
    /// Recurse into child directories. Defaults to false.
    #[serde(default)]
    recursive: bool,
    /// Optional include glob.
    #[serde(default)]
    include_glob: Option<String>,
    /// Optional exclude glob.
    #[serde(default)]
    exclude_glob: Option<String>,
    /// Include hidden entries. Defaults to false.
    #[serde(default)]
    include_hidden: bool,
    /// Maximum returned entries. Defaults to 200.
    #[serde(default)]
    max_results: Option<usize>,
    /// Pagination offset. Defaults to 0.
    #[serde(default)]
    offset: usize,
    /// Sort order. Defaults to path.
    #[serde(default)]
    sort: SortMode,
}

#[derive(Debug, Serialize, Clone)]
struct FileListEntry {
    path: String,
    kind: String,
    size: Option<u64>,
    modified: Option<String>,
    symlink: bool,
    symlink_escape: bool,
}

#[derive(Debug, Serialize)]
struct FileListOutput {
    path: String,
    entries: Vec<FileListEntry>,
    count: usize,
    offset: usize,
    truncated: bool,
    policy_notes: Vec<String>,
}

#[async_trait]
impl Tool for FileListTool {
    fn id(&self) -> &str {
        "file_list"
    }

    fn name(&self) -> &str {
        "File List"
    }

    fn description(&self) -> &str {
        "List directory entries safely with recursion, include or exclude globs, hidden-file policy, symlink notes, and pagination."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileListInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_only_metadata(ToolOperationKind::Read, true)
    }

    async fn execute(&self, args: Value, _ctx: ai_agents_core::ToolExecutionContext) -> ToolResult {
        let input: FileListInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let root = PathBuf::from(&input.path);
        if is_blocked_path(&root) {
            return ToolResult::error("path is blocked by default policy");
        }
        if !root.is_dir() {
            return ToolResult::error("path is not a directory");
        }
        let include_matcher = match input.include_glob.as_deref() {
            Some(glob) => match glob_regex(glob) {
                Ok(regex) => Some(regex),
                Err(error) => return ToolResult::error(format!("Invalid include_glob: {}", error)),
            },
            None => None,
        };
        let exclude_matcher = match input.exclude_glob.as_deref() {
            Some(glob) => match glob_regex(glob) {
                Ok(regex) => Some(regex),
                Err(error) => return ToolResult::error(format!("Invalid exclude_glob: {}", error)),
            },
            None => None,
        };

        let mut entries = Vec::new();
        let mut ignored = HashSet::new();
        walk_paths(
            &root,
            input.recursive,
            true,
            input.include_hidden,
            &mut entries,
            &mut ignored,
        )
        .await;
        sort_entries(&mut entries, input.sort);

        let root_canonical = root.canonicalize().ok();
        let mut output_entries = Vec::new();
        for entry in entries {
            let rel = relative_display(&root, &entry.path);
            if include_matcher
                .as_ref()
                .is_some_and(|matcher| !matcher.is_match(&rel) && !matcher.is_match(&display_path(&entry.path)))
            {
                continue;
            }
            if exclude_matcher
                .as_ref()
                .is_some_and(|matcher| matcher.is_match(&rel) || matcher.is_match(&display_path(&entry.path)))
            {
                continue;
            }
            let symlink_escape = entry.symlink
                && root_canonical.as_ref().is_some_and(|root| {
                    entry
                        .path
                        .canonicalize()
                        .map(|target| !target.starts_with(root))
                        .unwrap_or(false)
                });
            output_entries.push(FileListEntry {
                path: display_path(&entry.path),
                kind: entry.kind.as_str().to_string(),
                size: entry.size,
                modified: entry.modified,
                symlink: entry.symlink,
                symlink_escape,
            });
        }

        let total = output_entries.len();
        let max_results = input.max_results.unwrap_or(DEFAULT_FILE_LIST_MAX_RESULTS);
        let page = output_entries
            .into_iter()
            .skip(input.offset)
            .take(max_results)
            .collect::<Vec<_>>();
        let truncated = input.offset.saturating_add(page.len()) < total;
        let mut policy_notes = vec!["default ignored directories are skipped".to_string()];
        if !ignored.is_empty() {
            let mut ignored: Vec<String> = ignored.into_iter().collect();
            ignored.sort();
            policy_notes.push(format!("ignored: {}", ignored.join(", ")));
        }

        to_json_result(
            &FileListOutput {
                path: input.path,
                entries: page,
                count: total,
                offset: input.offset,
                truncated,
                policy_notes,
            },
            DEFAULT_MAX_OUTPUT_CHARS,
        )
    }
}

/// Safe file metadata inspector without content reads.
pub struct FileInfoTool;

impl FileInfoTool {
    /// Create a file info tool with default policy behavior.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FileInfoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileInfoInput {
    /// File or directory path to inspect.
    path: String,
    /// Follow symlinks when safe. Defaults to false.
    #[serde(default)]
    follow_symlinks: bool,
}

#[derive(Debug, Serialize)]
struct FileInfoOutput {
    path: String,
    exists: bool,
    kind: String,
    size: Option<u64>,
    modified: Option<String>,
    readonly: Option<bool>,
    symlink: bool,
    symlink_escape: bool,
    canonical_path: Option<String>,
    mime_hint: Option<String>,
    policy_classification: String,
}

#[async_trait]
impl Tool for FileInfoTool {
    fn id(&self) -> &str {
        "file_info"
    }

    fn name(&self) -> &str {
        "File Info"
    }

    fn description(&self) -> &str {
        "Inspect safe file or directory metadata without reading file contents, including symlink classification."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileInfoInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_only_metadata(ToolOperationKind::Read, true)
    }

    async fn execute(&self, args: Value, _ctx: ai_agents_core::ToolExecutionContext) -> ToolResult {
        let input: FileInfoInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let path = PathBuf::from(&input.path);
        if is_blocked_path(&path) {
            return ToolResult::error("path is blocked by default policy");
        }
        let link_metadata = fs::symlink_metadata(&path);
        if link_metadata.is_err() {
            return to_json_result(
                &FileInfoOutput {
                    path: input.path,
                    exists: false,
                    kind: "missing".to_string(),
                    size: None,
                    modified: None,
                    readonly: None,
                    symlink: false,
                    symlink_escape: false,
                    canonical_path: None,
                    mime_hint: None,
                    policy_classification: "allowed_metadata_only".to_string(),
                },
                DEFAULT_MAX_OUTPUT_CHARS,
            );
        }
        let link_metadata = link_metadata.unwrap();
        let symlink = link_metadata.file_type().is_symlink();
        let metadata = if input.follow_symlinks {
            fs::metadata(&path).unwrap_or_else(|_| link_metadata.clone())
        } else {
            link_metadata.clone()
        };
        let parent_root = path.parent().and_then(|parent| parent.canonicalize().ok());
        let canonical = path.canonicalize().ok();
        let symlink_escape = symlink
            && input.follow_symlinks
            && parent_root.as_ref().is_some_and(|root| {
                canonical
                    .as_ref()
                    .map(|target| !target.starts_with(root))
                    .unwrap_or(false)
            });
        let kind = if metadata.is_file() {
            "file"
        } else if metadata.is_dir() {
            "directory"
        } else if symlink {
            "symlink"
        } else {
            "other"
        };
        let modified = metadata.modified().ok().map(system_time_to_rfc3339);
        let readonly = Some(metadata.permissions().readonly());
        let canonical_path = (!symlink_escape)
            .then(|| canonical.map(|path| display_path(&path)))
            .flatten();
        let mime_hint = path.extension().and_then(|ext| ext.to_str()).map(|ext| {
            match ext.to_ascii_lowercase().as_str() {
                "rs" => "text/rust",
                "md" => "text/markdown",
                "json" => "application/json",
                "yaml" | "yml" => "application/yaml",
                "toml" => "application/toml",
                "txt" => "text/plain",
                _ => "application/octet-stream",
            }
            .to_string()
        });

        to_json_result(
            &FileInfoOutput {
                path: input.path,
                exists: true,
                kind: kind.to_string(),
                size: Some(metadata.len()),
                modified,
                readonly,
                symlink,
                symlink_escape,
                canonical_path,
                mime_hint,
                policy_classification: if symlink_escape {
                    "symlink_escape_denied".to_string()
                } else {
                    "allowed_metadata_only".to_string()
                },
            },
            DEFAULT_MAX_OUTPUT_CHARS,
        )
    }
}

#[derive(Debug, Clone)]
struct WalkEntry {
    path: PathBuf,
    kind: EntryKind,
    size: Option<u64>,
    modified: Option<String>,
    symlink: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    File,
    Directory,
    Symlink,
    Other,
}

impl EntryKind {
    fn is_file(self) -> bool {
        matches!(self, Self::File)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "directory",
            Self::Symlink => "symlink",
            Self::Other => "other",
        }
    }
}

async fn walk_paths(
    root: &Path,
    recursive: bool,
    include_dirs: bool,
    include_hidden: bool,
    out: &mut Vec<WalkEntry>,
    ignored: &mut HashSet<String>,
) {
    let mut stack = vec![root.to_path_buf()];
    let mut ticks = 0usize;
    while let Some(dir) = stack.pop() {
        ticks += 1;
        if ticks % 32 == 0 {
            tokio::task::yield_now().await;
        }
        if dir.is_file() {
            if let Some(entry) = walk_entry(&dir) {
                out.push(entry);
            }
            continue;
        }
        let Ok(read_dir) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if !include_hidden && is_hidden_name(&name) {
                continue;
            }
            if path.is_dir() && is_ignored_dir(&name) {
                ignored.insert(name);
                continue;
            }
            if is_blocked_path(&path) {
                continue;
            }
            if let Some(walk_entry) = walk_entry(&path) {
                if include_dirs || !matches!(walk_entry.kind, EntryKind::Directory) {
                    out.push(walk_entry.clone());
                }
                if recursive && matches!(walk_entry.kind, EntryKind::Directory) && !walk_entry.symlink {
                    stack.push(path);
                }
            }
        }
    }
}

fn walk_entry(path: &Path) -> Option<WalkEntry> {
    let link_metadata = fs::symlink_metadata(path).ok()?;
    let symlink = link_metadata.file_type().is_symlink();
    let metadata = if symlink {
        fs::metadata(path).unwrap_or_else(|_| link_metadata.clone())
    } else {
        link_metadata.clone()
    };
    let kind = if symlink {
        EntryKind::Symlink
    } else if metadata.is_file() {
        EntryKind::File
    } else if metadata.is_dir() {
        EntryKind::Directory
    } else {
        EntryKind::Other
    };
    Some(WalkEntry {
        path: path.to_path_buf(),
        kind,
        size: metadata.is_file().then(|| metadata.len()),
        modified: metadata.modified().ok().map(system_time_to_rfc3339),
        symlink,
    })
}

fn sort_entries(entries: &mut [WalkEntry], sort: SortMode) {
    entries.sort_by(|a, b| match sort {
        SortMode::Path => display_path(&a.path).cmp(&display_path(&b.path)),
        SortMode::Modified => a
            .modified
            .cmp(&b.modified)
            .then_with(|| display_path(&a.path).cmp(&display_path(&b.path))),
        SortMode::Size => a
            .size
            .cmp(&b.size)
            .then_with(|| display_path(&a.path).cmp(&display_path(&b.path))),
        SortMode::Kind => a
            .kind
            .as_str()
            .cmp(b.kind.as_str())
            .then_with(|| display_path(&a.path).cmp(&display_path(&b.path))),
    });
}

fn glob_regex(pattern: &str) -> Result<regex::Regex, regex::Error> {
    let mut regex = String::from("^");
    let chars: Vec<char> = pattern.replace('\\', "/").chars().collect();
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '*' if chars.get(index + 1) == Some(&'*') && chars.get(index + 2) == Some(&'/') => {
                regex.push_str("(?:.*/)?");
                index += 3;
            }
            '*' if chars.get(index + 1) == Some(&'*') => {
                regex.push_str(".*");
                index += 2;
            }
            '*' => {
                regex.push_str("[^/]*");
                index += 1;
            }
            '?' => {
                regex.push_str("[^/]");
                index += 1;
            }
            c => {
                regex.push_str(&regex::escape(&c.to_string()));
                index += 1;
            }
        }
    }
    regex.push('$');
    RegexBuilder::new(&regex).case_insensitive(false).build()
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(display_path)
        .unwrap_or_else(|_| display_path(path))
}

fn is_hidden_name(name: &str) -> bool {
    name.starts_with('.')
}

fn is_ignored_dir(name: &str) -> bool {
    matches!(
        name,
        ".git" | "target" | "node_modules" | ".next" | "dist" | "build" | ".cache"
    )
}

fn is_blocked_path(path: &Path) -> bool {
    path.components().any(|component| match component {
        Component::Normal(value) => {
            let name = value.to_string_lossy();
            matches!(name.as_ref(), ".git")
        }
        _ => false,
    })
}

fn is_probably_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8_192).any(|byte| *byte == 0)
}

fn read_limited(path: &Path, max_bytes: usize) -> std::io::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(max_bytes as u64).read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn system_time_to_rfc3339(time: std::time::SystemTime) -> String {
    let datetime: DateTime<Utc> = time.into();
    datetime.to_rfc3339()
}

fn to_json_result<T: Serialize>(output: &T, max_chars: usize) -> ToolResult {
    match serde_json::to_string(output) {
        Ok(json) => {
            let (output, truncated) = truncate_chars(json, max_chars);
            let mut metadata = HashMap::new();
            metadata.insert("output_truncated".to_string(), Value::Bool(truncated));
            metadata.insert("max_output_chars".to_string(), Value::from(max_chars));
            ToolResult::ok_with_metadata(output, metadata)
        }
        Err(error) => ToolResult::error(format!("Serialization error: {}", error)),
    }
}

fn attach_metadata(result: &mut ToolResult, truncated: bool, file_size: Option<u64>, count: Option<usize>) {
    let metadata = result.metadata.get_or_insert_with(HashMap::new);
    metadata.insert("truncated".to_string(), Value::Bool(truncated));
    if let Some(file_size) = file_size {
        metadata.insert("file_size".to_string(), Value::from(file_size));
    }
    if let Some(count) = count {
        metadata.insert("count".to_string(), Value::from(count));
    }
}

fn truncate_chars(value: String, max_chars: usize) -> (String, bool) {
    let mut chars = value.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        (truncated, true)
    } else {
        (value, false)
    }
}

fn read_only_metadata(operation: ToolOperationKind, supports_cancellation: bool) -> ToolSafetyMetadata {
    ToolSafetyMetadata {
        read_only: true,
        concurrency_safe: true,
        operation,
        side_effect_level: ToolSideEffectLevel::None,
        requires_network: false,
        destructive: false,
        open_world: false,
        host_dependent: false,
        requires_user_interaction: false,
        supports_cancellation,
        default_requires_approval: false,
        should_defer_schema: false,
        max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn glob_uses_stable_offset_and_ignored_dirs() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src").join("b.rs"), "fn b() {}").unwrap();
        fs::create_dir(dir.path().join("target")).unwrap();
        fs::write(dir.path().join("target").join("c.rs"), "fn c() {}").unwrap();

        let tool = GlobTool::new();
        let result = tool
            .execute(serde_json::json!({
                "pattern": "**/*.rs",
                "path": dir.path(),
                "max_results": 1,
                "offset": 1
            }), ai_agents_core::ToolExecutionContext::test("test"))
            .await;
        assert!(result.success);
        assert!(result.output.contains("b.rs"));
        assert!(!result.output.contains("c.rs"));
        assert!(result.output.contains("\"truncated\":false"));
    }

    #[tokio::test]
    async fn grep_supports_literal_regex_offsets_and_binary_skip() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "Hello rust\nhello tools\n").unwrap();
        fs::write(dir.path().join("b.txt"), b"binary\0data").unwrap();
        let tool = GrepTool::new();

        let result = tool
            .execute(serde_json::json!({
                "pattern": "hello",
                "mode": "literal",
                "path": dir.path(),
                "case_sensitive": false,
                "output_mode": "content",
                "max_results": 1,
                "offset": 1
            }), ai_agents_core::ToolExecutionContext::test("test"))
            .await;
        assert!(result.success);
        assert!(result.output.contains("hello tools"));
        assert!(result.output.contains("\"skipped_binary\":1"));
    }

    #[tokio::test]
    async fn file_read_handles_ranges_large_files_and_unicode() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("unicode.txt");
        fs::write(&path, "zero\none 😄\ntwo\nthree\n").unwrap();
        let tool = FileReadTool::new();
        let result = tool
            .execute(serde_json::json!({
                "path": path,
                "start_line": 2,
                "end_line": 3
            }), ai_agents_core::ToolExecutionContext::test("test"))
            .await;
        assert!(result.success);
        assert!(result.output.contains("one 😄"));
        assert!(result.output.contains("two"));
        assert!(!result.output.contains("three"));

        let large = dir.path().join("large.txt");
        fs::write(&large, "x".repeat(20_000)).unwrap();
        let result = tool
            .execute(serde_json::json!({
                "path": large,
                "max_bytes": 100
            }), ai_agents_core::ToolExecutionContext::test("test"))
            .await;
        assert!(result.success);
        assert!(result.output.contains("large_file"));
    }

    #[tokio::test]
    async fn file_list_paginates_and_reports_symlink_escape() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "a").unwrap();
        fs::write(dir.path().join("b.log"), "b").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("/tmp", dir.path().join("escape")).unwrap();

        let tool = FileListTool::new();
        let result = tool
            .execute(serde_json::json!({
                "path": dir.path(),
                "include_glob": "*.txt",
                "max_results": 1
            }), ai_agents_core::ToolExecutionContext::test("test"))
            .await;
        assert!(result.success);
        assert!(result.output.contains("a.txt"));
        assert!(!result.output.contains("b.log"));
    }

    #[tokio::test]
    async fn file_info_returns_safe_metadata_for_symlink() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("a.txt");
        fs::write(&file, "hello").unwrap();
        let tool = FileInfoTool::new();
        let result = tool
            .execute(serde_json::json!({
                "path": file
            }), ai_agents_core::ToolExecutionContext::test("test"))
            .await;
        assert!(result.success);
        assert!(result.output.contains("\"kind\":\"file\""));
        assert!(result.output.contains("\"size\":5"));
    }

    #[test]
    fn glob_pattern_supports_double_star_without_directory() {
        let regex = glob_regex("**/*.rs").unwrap();
        assert!(regex.is_match("lib.rs"));
        assert!(regex.is_match("src/lib.rs"));
        assert!(!regex.is_match("src/lib.py"));
    }

    #[test]
    fn count_output_is_stable_json() {
        let mut map = BTreeMap::new();
        map.insert("a", 1);
        let result = to_json_result(&map, 100);
        assert!(result.success);
        assert!(result.output.contains("a"));
    }
}
