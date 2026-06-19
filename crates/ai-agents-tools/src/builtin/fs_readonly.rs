use async_trait::async_trait;
use regex::{Regex, RegexBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use ai_agents_core::{
    Tool, ToolOperationKind, ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;

const DEFAULT_MAX_RESULTS: usize = 200;
const DEFAULT_MAX_FILE_BYTES: u64 = 1_048_576;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;
const DEFAULT_FILE_READ_MAX_LINES: usize = 2_000;

const DEFAULT_IGNORED_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
];

/// Finds workspace paths that match a glob pattern.
pub struct GlobTool;

impl GlobTool {
    /// Create a glob discovery tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GlobTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Searches workspace file content with regex or literal matching.
pub struct GrepTool;

impl GrepTool {
    /// Create a bounded text search tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads bounded text slices from local files.
pub struct FileReadTool;

impl FileReadTool {
    /// Create a safe text file reader.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FileReadTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Lists directory entries with pagination and hidden-file control.
pub struct FileListTool;

impl FileListTool {
    /// Create a deterministic directory lister.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FileListTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Inspects safe file metadata without reading contents.
pub struct FileInfoTool;

impl FileInfoTool {
    /// Create a safe file metadata inspector.
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
struct GlobInput {
    /// Glob pattern such as **/*.rs.
    pattern: String,
    /// Root directory to search. Defaults to current directory.
    #[serde(default)]
    path: Option<String>,
    /// Maximum returned paths. Defaults to 100.
    #[serde(default)]
    max_results: Option<usize>,
    /// Result offset for pagination. Defaults to 0.
    #[serde(default)]
    offset: Option<usize>,
    /// Include directory matches. Defaults to false.
    #[serde(default)]
    include_dirs: bool,
    /// Sort order: path, modified, or size. Defaults to path.
    #[serde(default)]
    sort: Option<String>,
}

#[derive(Debug, Serialize)]
struct GlobOutput {
    paths: Vec<String>,
    count: usize,
    total_count: usize,
    offset: usize,
    truncated: bool,
    duration_ms: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum GrepMode {
    Regex,
    Literal,
}

impl Default for GrepMode {
    fn default() -> Self {
        Self::Regex
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum GrepOutputMode {
    Content,
    FilesWithMatches,
    Count,
}

impl Default for GrepOutputMode {
    fn default() -> Self {
        Self::FilesWithMatches
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GrepInput {
    /// Literal or regex pattern to search for.
    pattern: String,
    /// Search mode. Defaults to regex.
    #[serde(default)]
    mode: GrepMode,
    /// File or directory root. Defaults to current directory.
    #[serde(default)]
    path: Option<String>,
    /// Optional include glob for file paths.
    #[serde(default)]
    include_glob: Option<String>,
    /// Whether matching is case-sensitive. Defaults to false.
    #[serde(default)]
    case_sensitive: bool,
    /// Output mode. Defaults to files_with_matches.
    #[serde(default)]
    output_mode: GrepOutputMode,
    /// Context lines for content output. Defaults to 0.
    #[serde(default)]
    context: Option<usize>,
    /// Maximum result rows. Defaults to 250.
    #[serde(default)]
    max_results: Option<usize>,
    /// Result offset for pagination. Defaults to 0.
    #[serde(default)]
    offset: Option<usize>,
    /// Maximum file bytes to inspect. Defaults to 1 MiB.
    #[serde(default)]
    max_file_size_bytes: Option<u64>,
    /// Soft cap for collected text. Defaults to 20000 characters.
    #[serde(default)]
    max_output_chars: Option<usize>,
}

#[derive(Debug, Serialize)]
struct GrepOutput {
    mode: String,
    matches: Vec<GrepMatch>,
    files: Vec<String>,
    count: usize,
    total_count: usize,
    offset: usize,
    truncated: bool,
    skipped_binary: usize,
    skipped_large: usize,
}

#[derive(Debug, Serialize, Clone)]
struct GrepMatch {
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileReadInput {
    /// File path to read.
    path: String,
    /// One-based first line. Defaults to 1.
    #[serde(default)]
    start_line: Option<usize>,
    /// One-based last line.
    #[serde(default)]
    end_line: Option<usize>,
    /// Maximum lines to return. Defaults to 2000.
    #[serde(default)]
    max_lines: Option<usize>,
    /// Maximum bytes to return. Defaults to 1 MiB.
    #[serde(default)]
    max_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct FileReadOutput {
    path: String,
    content: String,
    start_line: usize,
    end_line: usize,
    total_lines: usize,
    bytes_read: usize,
    file_size: u64,
    truncated: bool,
    large_file: bool,
    encoding: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileListInput {
    /// Directory path to list.
    path: String,
    /// Recurse into child directories. Defaults to false.
    #[serde(default)]
    recursive: bool,
    /// Optional include glob for relative paths.
    #[serde(default)]
    include_glob: Option<String>,
    /// Optional exclude glob for relative paths.
    #[serde(default)]
    exclude_glob: Option<String>,
    /// Include hidden files. Defaults to false.
    #[serde(default)]
    include_hidden: bool,
    /// Maximum returned entries. Defaults to 200.
    #[serde(default)]
    max_results: Option<usize>,
    /// Result offset for pagination. Defaults to 0.
    #[serde(default)]
    offset: Option<usize>,
    /// Sort order: path, modified, size, or kind. Defaults to path.
    #[serde(default)]
    sort: Option<String>,
}

#[derive(Debug, Serialize)]
struct FileListOutput {
    path: String,
    entries: Vec<FileListEntry>,
    count: usize,
    total_count: usize,
    offset: usize,
    truncated: bool,
    policy_notes: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct FileListEntry {
    path: String,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modified: Option<String>,
    symlink: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileInfoInput {
    /// File or directory path to inspect.
    path: String,
    /// Follow symlink target when safe. Defaults to false.
    #[serde(default)]
    follow_symlinks: bool,
}

#[derive(Debug, Serialize)]
struct FileInfoOutput {
    path: String,
    exists: bool,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modified: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created: Option<String>,
    readonly: bool,
    symlink: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_hint: Option<String>,
    policy_classification: String,
}

#[async_trait]
impl Tool for GlobTool {
    fn id(&self) -> &str {
        "glob"
    }

    fn name(&self) -> &str {
        "Glob"
    }

    fn description(&self) -> &str {
        "Find file paths by glob pattern with deterministic sorting and pagination."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GlobInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_tool_metadata(ToolOperationKind::Read)
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let started = Instant::now();
        let input: GlobInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let root = PathBuf::from(input.path.unwrap_or_else(|| ".".to_string()));
        if let Err(reason) = ensure_safe_path(&root) {
            return ToolResult::error(reason);
        }
        let matcher = match GlobMatcher::new(&input.pattern) {
            Ok(matcher) => matcher,
            Err(error) => return ToolResult::error(format!("Invalid glob pattern: {}", error)),
        };

        let mut entries = Vec::new();
        let mut stack = vec![root.clone()];
        let mut visited = 0usize;
        while let Some(dir) = stack.pop() {
            let read_dir = match fs::read_dir(&dir) {
                Ok(read_dir) => read_dir,
                Err(_) => continue,
            };
            for entry in read_dir.flatten() {
                visited += 1;
                if visited % 128 == 0 {
                    tokio::task::yield_now().await;
                }
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();
                let metadata = match fs::symlink_metadata(&path) {
                    Ok(metadata) => metadata,
                    Err(_) => continue,
                };
                let is_dir = metadata.is_dir();
                if is_dir && is_default_ignored_dir(&file_name) {
                    continue;
                }
                if is_dir {
                    stack.push(path.clone());
                }
                if is_dir && !input.include_dirs {
                    continue;
                }
                let relative = relative_path(&root, &path);
                if matcher.matches(&relative) {
                    entries.push(SortablePath {
                        path: normalize_slashes(relative),
                        modified: metadata.modified().ok(),
                        size: metadata.len(),
                        kind: if is_dir { "dir" } else { "file" }.to_string(),
                    });
                }
            }
        }

        sort_paths(&mut entries, input.sort.as_deref().unwrap_or("path"));
        let total_count = entries.len();
        let offset = input.offset.unwrap_or(0);
        let max_results = input.max_results.unwrap_or(100).min(DEFAULT_MAX_RESULTS);
        let paths: Vec<String> = entries
            .into_iter()
            .skip(offset)
            .take(max_results)
            .map(|entry| entry.path)
            .collect();
        let output = GlobOutput {
            count: paths.len(),
            total_count,
            offset,
            truncated: offset.saturating_add(paths.len()) < total_count,
            duration_ms: started.elapsed().as_millis() as u64,
            paths,
        };
        json_result_with_caps(&output, false, None)
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn id(&self) -> &str {
        "grep"
    }

    fn name(&self) -> &str {
        "Grep"
    }

    fn description(&self) -> &str {
        "Search text files using regex or literal matching with bounded, paginated output."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GrepInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_tool_metadata(ToolOperationKind::Read)
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: GrepInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let root = PathBuf::from(input.path.clone().unwrap_or_else(|| ".".to_string()));
        if let Err(reason) = ensure_safe_path(&root) {
            return ToolResult::error(reason);
        }
        let include = match optional_matcher(input.include_glob.as_deref()) {
            Ok(matcher) => matcher,
            Err(error) => return ToolResult::error(format!("Invalid include_glob: {}", error)),
        };
        let regex = match build_search_regex(&input.pattern, &input.mode, input.case_sensitive) {
            Ok(regex) => regex,
            Err(error) => return ToolResult::error(format!("Invalid search pattern: {}", error)),
        };
        let max_results = input
            .max_results
            .unwrap_or(250)
            .min(DEFAULT_MAX_RESULTS * 2);
        let offset = input.offset.unwrap_or(0);
        let max_file_size = input.max_file_size_bytes.unwrap_or(DEFAULT_MAX_FILE_BYTES);
        let max_output_chars = input.max_output_chars.unwrap_or(DEFAULT_MAX_OUTPUT_CHARS);
        let context = input.context.unwrap_or(0).min(20);

        let files = collect_files(&root, include.as_ref()).await;
        let mut matches = Vec::new();
        let mut files_with_matches = BTreeSet::new();
        let mut skipped_binary = 0usize;
        let mut skipped_large = 0usize;
        let mut collected_chars = 0usize;
        let mut truncated = false;

        for file in files {
            let metadata = match fs::metadata(&file) {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };
            if metadata.len() > max_file_size {
                skipped_large += 1;
                continue;
            }
            let bytes = match fs::read(&file) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            if looks_binary(&bytes) {
                skipped_binary += 1;
                continue;
            }
            let text = match String::from_utf8(bytes) {
                Ok(text) => text,
                Err(_) => {
                    skipped_binary += 1;
                    continue;
                }
            };
            let relative = normalize_slashes(relative_path(&root, &file));
            let line_hits = line_matches(&text, &regex, context);
            if line_hits.is_empty() {
                continue;
            }
            files_with_matches.insert(relative.clone());
            match input.output_mode {
                GrepOutputMode::FilesWithMatches => {
                    matches.push(GrepMatch {
                        path: relative,
                        line: None,
                        text: None,
                        count: None,
                    });
                }
                GrepOutputMode::Count => {
                    matches.push(GrepMatch {
                        path: relative,
                        line: None,
                        text: None,
                        count: Some(line_hits.iter().filter(|hit| hit.is_match).count()),
                    });
                }
                GrepOutputMode::Content => {
                    for hit in line_hits {
                        let mut text = hit.text;
                        let (bounded, was_truncated) = truncate_chars(text, 1_000);
                        text = bounded;
                        truncated |= was_truncated;
                        collected_chars = collected_chars.saturating_add(text.chars().count());
                        if collected_chars > max_output_chars {
                            truncated = true;
                            break;
                        }
                        matches.push(GrepMatch {
                            path: relative.clone(),
                            line: Some(hit.line),
                            text: Some(text),
                            count: None,
                        });
                    }
                }
            }
            if matches.len() >= offset.saturating_add(max_results) || truncated {
                if matches.len() >= offset.saturating_add(max_results) {
                    truncated = true;
                }
                break;
            }
        }

        let total_count = matches.len();
        let matches: Vec<GrepMatch> = matches.into_iter().skip(offset).take(max_results).collect();
        let files: Vec<String> = files_with_matches.into_iter().collect();
        let output = GrepOutput {
            mode: output_mode_name(&input.output_mode).to_string(),
            count: matches.len(),
            total_count,
            offset,
            truncated: truncated || offset.saturating_add(matches.len()) < total_count,
            skipped_binary,
            skipped_large,
            matches,
            files,
        };
        json_result_with_caps(&output, output.truncated, Some(max_output_chars))
    }
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
        "Read a bounded UTF-8 text file range with line numbers and large-file protection."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileReadInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_tool_metadata(ToolOperationKind::Read)
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: FileReadInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let path = PathBuf::from(&input.path);
        if let Err(reason) = ensure_safe_path(&path) {
            return ToolResult::error(reason);
        }
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => return ToolResult::error(format!("Metadata error: {}", error)),
        };
        if !metadata.is_file() {
            return ToolResult::error(format!("Not a file: {}", input.path));
        }
        let max_bytes = input.max_bytes.unwrap_or(DEFAULT_MAX_FILE_BYTES);
        let max_lines = input.max_lines.unwrap_or(DEFAULT_FILE_READ_MAX_LINES);
        let has_range = input.start_line.is_some() || input.end_line.is_some();
        let large_file = metadata.len() > max_bytes;
        let start_line = input.start_line.unwrap_or(1).max(1);
        let requested_end = input.end_line.unwrap_or(usize::MAX);
        if requested_end < start_line {
            return ToolResult::error("end_line must be greater than or equal to start_line");
        }
        let effective_end =
            requested_end.min(start_line.saturating_add(max_lines).saturating_sub(1));
        if large_file && !has_range {
            let output = FileReadOutput {
                path: input.path,
                content: String::new(),
                start_line: 0,
                end_line: 0,
                total_lines: 0,
                bytes_read: 0,
                file_size: metadata.len(),
                truncated: true,
                large_file: true,
                encoding: "utf-8".to_string(),
            };
            return json_result_with_caps(&output, true, None);
        }
        let sample = match read_prefix(&path, 8_192) {
            Ok(sample) => sample,
            Err(error) => return ToolResult::error(format!("Read error: {}", error)),
        };
        if looks_binary(&sample) || std::str::from_utf8(&sample).is_err() {
            return ToolResult::error("Binary or non-UTF-8 files are not supported by file_read");
        }

        match read_text_range(&path, start_line, effective_end, max_lines, max_bytes).await {
            Ok(output) => json_result_with_caps(&output, output.truncated, None),
            Err(error) => ToolResult::error(error),
        }
    }
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
        "List directory entries with recursive, glob, hidden-file, symlink, and pagination controls."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileListInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_tool_metadata(ToolOperationKind::Read)
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: FileListInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let root = PathBuf::from(&input.path);
        if let Err(reason) = ensure_safe_path(&root) {
            return ToolResult::error(reason);
        }
        if !root.is_dir() {
            return ToolResult::error(format!("Not a directory: {}", input.path));
        }
        let include = match optional_matcher(input.include_glob.as_deref()) {
            Ok(matcher) => matcher,
            Err(error) => return ToolResult::error(format!("Invalid include_glob: {}", error)),
        };
        let exclude = match optional_matcher(input.exclude_glob.as_deref()) {
            Ok(matcher) => matcher,
            Err(error) => return ToolResult::error(format!("Invalid exclude_glob: {}", error)),
        };
        let canonical_root = fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
        let mut stack = vec![root.clone()];
        let mut entries = Vec::new();
        let mut notes = Vec::new();
        let mut visited = 0usize;

        while let Some(dir) = stack.pop() {
            let read_dir = match fs::read_dir(&dir) {
                Ok(read_dir) => read_dir,
                Err(_) => continue,
            };
            for entry in read_dir.flatten() {
                visited += 1;
                if visited % 128 == 0 {
                    tokio::task::yield_now().await;
                }
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();
                if !input.include_hidden && file_name.starts_with('.') {
                    if file_name == ".git" {
                        notes.push("blocked .git directory".to_string());
                    }
                    continue;
                }
                if is_default_ignored_dir(&file_name) {
                    notes.push(format!("ignored default directory: {}", file_name));
                    continue;
                }
                let metadata = match fs::symlink_metadata(&path) {
                    Ok(metadata) => metadata,
                    Err(_) => continue,
                };
                let relative = normalize_slashes(relative_path(&root, &path));
                if include
                    .as_ref()
                    .is_some_and(|matcher| !matcher.matches(&relative))
                {
                    continue;
                }
                if exclude
                    .as_ref()
                    .is_some_and(|matcher| matcher.matches(&relative))
                {
                    continue;
                }
                let symlink = metadata.file_type().is_symlink();
                let mut policy = None;
                let mut kind = kind_from_metadata(&metadata).to_string();
                let mut follow_for_recurse = metadata.is_dir();
                if symlink {
                    match fs::canonicalize(&path) {
                        Ok(target) if !target.starts_with(&canonical_root) => {
                            policy = Some("symlink_escape".to_string());
                            follow_for_recurse = false;
                            notes.push(format!("blocked symlink escape: {}", relative));
                        }
                        Ok(target) if target.is_dir() => {
                            kind = "symlink_dir".to_string();
                            follow_for_recurse = true;
                        }
                        Ok(_) => {
                            kind = "symlink_file".to_string();
                        }
                        Err(_) => {
                            policy = Some("broken_symlink".to_string());
                        }
                    }
                }
                entries.push(FileListEntry {
                    path: relative,
                    kind,
                    size: metadata.is_file().then(|| metadata.len()),
                    modified: metadata.modified().ok().map(system_time_rfc3339),
                    symlink,
                    policy,
                });
                if input.recursive && follow_for_recurse {
                    stack.push(path);
                }
            }
        }

        sort_list_entries(&mut entries, input.sort.as_deref().unwrap_or("path"));
        let total_count = entries.len();
        let offset = input.offset.unwrap_or(0);
        let max_results = input
            .max_results
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .min(DEFAULT_MAX_RESULTS * 5);
        let entries: Vec<FileListEntry> =
            entries.into_iter().skip(offset).take(max_results).collect();
        notes.sort();
        notes.dedup();
        let output = FileListOutput {
            path: input.path,
            count: entries.len(),
            total_count,
            offset,
            truncated: offset.saturating_add(entries.len()) < total_count,
            entries,
            policy_notes: notes,
        };
        json_result_with_caps(&output, output.truncated, None)
    }
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
        "Inspect safe file or directory metadata without reading file contents."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileInfoInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        read_tool_metadata(ToolOperationKind::Read)
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: FileInfoInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let path = PathBuf::from(&input.path);
        if let Err(reason) = ensure_safe_path_allow_missing(&path) {
            return ToolResult::error(reason);
        }
        let symlink_metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let output = FileInfoOutput {
                    path: input.path,
                    exists: false,
                    kind: "missing".to_string(),
                    size: None,
                    modified: None,
                    created: None,
                    readonly: false,
                    symlink: false,
                    canonical_path: None,
                    mime_hint: None,
                    policy_classification: "allowed".to_string(),
                };
                return json_result_with_caps(&output, false, None);
            }
            Err(error) => return ToolResult::error(format!("Metadata error: {}", error)),
        };
        let is_symlink = symlink_metadata.file_type().is_symlink();
        let mut policy = "allowed".to_string();
        let mut canonical_path = None;
        let metadata = if is_symlink && input.follow_symlinks {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let canonical_parent =
                fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
            match fs::canonicalize(&path) {
                Ok(target) if !target.starts_with(&canonical_parent) => {
                    policy = "symlink_escape".to_string();
                    symlink_metadata.clone()
                }
                Ok(target) => {
                    canonical_path = Some(target.to_string_lossy().to_string());
                    fs::metadata(&path).unwrap_or_else(|_| symlink_metadata.clone())
                }
                Err(_) => {
                    policy = "broken_symlink".to_string();
                    symlink_metadata.clone()
                }
            }
        } else {
            if let Ok(canonical) = fs::canonicalize(&path) {
                canonical_path = Some(canonical.to_string_lossy().to_string());
            }
            symlink_metadata.clone()
        };
        let output = FileInfoOutput {
            path: input.path,
            exists: true,
            kind: kind_from_metadata(&metadata).to_string(),
            size: metadata.is_file().then(|| metadata.len()),
            modified: metadata.modified().ok().map(system_time_rfc3339),
            created: metadata.created().ok().map(system_time_rfc3339),
            readonly: metadata.permissions().readonly(),
            symlink: is_symlink,
            canonical_path: if policy == "allowed" {
                canonical_path
            } else {
                None
            },
            mime_hint: mime_hint(&path),
            policy_classification: policy,
        };
        json_result_with_caps(&output, false, None)
    }
}

#[derive(Debug, Clone)]
struct SortablePath {
    path: String,
    modified: Option<std::time::SystemTime>,
    size: u64,
    kind: String,
}

#[derive(Debug)]
struct GlobMatcher {
    regex: Regex,
}

impl GlobMatcher {
    fn new(pattern: &str) -> Result<Self, regex::Error> {
        Regex::new(&glob_to_regex(pattern)).map(|regex| Self { regex })
    }

    fn matches(&self, path: &str) -> bool {
        self.regex.is_match(&normalize_slashes(path.to_string()))
    }
}

#[derive(Debug)]
struct LineHit {
    line: usize,
    text: String,
    is_match: bool,
}

fn read_tool_metadata(operation: ToolOperationKind) -> ToolSafetyMetadata {
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
        supports_cancellation: true,
        default_requires_approval: false,
        should_defer_schema: false,
        max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
    }
}

fn ensure_safe_path(path: &Path) -> Result<(), String> {
    ensure_safe_path_allow_missing(path)?;
    if path.exists() {
        if let Ok(canonical) = fs::canonicalize(path) {
            ensure_no_blocked_components(&canonical)?;
        }
    }
    Ok(())
}

fn ensure_safe_path_allow_missing(path: &Path) -> Result<(), String> {
    ensure_no_blocked_components(path)?;
    Ok(())
}

fn ensure_no_blocked_components(path: &Path) -> Result<(), String> {
    for component in path.components() {
        let Component::Normal(value) = component else {
            continue;
        };
        let value = value.to_string_lossy();
        if value == ".git" {
            return Err("Access to raw .git paths is blocked".to_string());
        }
    }
    Ok(())
}

fn is_default_ignored_dir(name: &str) -> bool {
    DEFAULT_IGNORED_DIRS.iter().any(|ignored| ignored == &name)
}

fn optional_matcher(pattern: Option<&str>) -> Result<Option<GlobMatcher>, regex::Error> {
    pattern.map(GlobMatcher::new).transpose()
}

fn glob_to_regex(pattern: &str) -> String {
    let mut regex = String::from("^");
    let chars: Vec<char> = pattern.chars().collect();
    let mut index = 0usize;
    while index < chars.len() {
        match chars[index] {
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
            '/' | '\\' => {
                regex.push('/');
                index += 1;
            }
            ch => {
                regex.push_str(&regex::escape(&ch.to_string()));
                index += 1;
            }
        }
    }
    regex.push('$');
    regex
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn normalize_slashes(path: String) -> String {
    path.replace('\\', "/")
}

fn sort_paths(entries: &mut [SortablePath], sort: &str) {
    match sort {
        "modified" => entries.sort_by(|a, b| {
            a.modified
                .cmp(&b.modified)
                .then_with(|| a.path.cmp(&b.path))
        }),
        "size" => entries.sort_by(|a, b| a.size.cmp(&b.size).then_with(|| a.path.cmp(&b.path))),
        "kind" => entries.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.path.cmp(&b.path))),
        _ => entries.sort_by(|a, b| a.path.cmp(&b.path)),
    }
}

fn sort_list_entries(entries: &mut [FileListEntry], sort: &str) {
    match sort {
        "modified" => entries.sort_by(|a, b| {
            a.modified
                .cmp(&b.modified)
                .then_with(|| a.path.cmp(&b.path))
        }),
        "size" => entries.sort_by(|a, b| a.size.cmp(&b.size).then_with(|| a.path.cmp(&b.path))),
        "kind" => entries.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.path.cmp(&b.path))),
        _ => entries.sort_by(|a, b| a.path.cmp(&b.path)),
    }
}

async fn collect_files(root: &Path, include: Option<&GlobMatcher>) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if root.is_file() {
        files.push(root.to_path_buf());
        return files;
    }
    let mut stack = vec![root.to_path_buf()];
    let mut visited = 0usize;
    while let Some(dir) = stack.pop() {
        let read_dir = match fs::read_dir(&dir) {
            Ok(read_dir) => read_dir,
            Err(_) => continue,
        };
        for entry in read_dir.flatten() {
            visited += 1;
            if visited % 128 == 0 {
                tokio::task::yield_now().await;
            }
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();
            if is_default_ignored_dir(&file_name) {
                continue;
            }
            let metadata = match entry.metadata() {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() {
                let relative = normalize_slashes(relative_path(root, &path));
                if include.is_none_or(|matcher| matcher.matches(&relative)) {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

fn build_search_regex(
    pattern: &str,
    mode: &GrepMode,
    case_sensitive: bool,
) -> Result<Regex, regex::Error> {
    let source = match mode {
        GrepMode::Regex => pattern.to_string(),
        GrepMode::Literal => regex::escape(pattern),
    };
    RegexBuilder::new(&source)
        .case_insensitive(!case_sensitive)
        .build()
}

fn line_matches(text: &str, regex: &Regex, context: usize) -> Vec<LineHit> {
    let lines: Vec<&str> = text.lines().collect();
    let mut included = BTreeSet::new();
    let mut matching = BTreeSet::new();
    for (index, line) in lines.iter().enumerate() {
        if regex.is_match(line) {
            matching.insert(index + 1);
            let start = index.saturating_sub(context);
            let end = (index + context).min(lines.len().saturating_sub(1));
            for ctx in start..=end {
                included.insert(ctx + 1);
            }
        }
    }
    included
        .into_iter()
        .filter_map(|line_number| {
            lines.get(line_number - 1).map(|line| LineHit {
                line: line_number,
                text: (*line).to_string(),
                is_match: matching.contains(&line_number),
            })
        })
        .collect()
}

fn output_mode_name(mode: &GrepOutputMode) -> &'static str {
    match mode {
        GrepOutputMode::Content => "content",
        GrepOutputMode::FilesWithMatches => "files_with_matches",
        GrepOutputMode::Count => "count",
    }
}

fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8_192).any(|byte| *byte == 0)
}

fn read_prefix(path: &Path, max_bytes: usize) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut buffer = vec![0u8; max_bytes];
    let read = file.read(&mut buffer)?;
    buffer.truncate(read);
    Ok(buffer)
}

async fn read_text_range(
    path: &Path,
    start_line: usize,
    end_line: usize,
    max_lines: usize,
    max_bytes: u64,
) -> Result<FileReadOutput, String> {
    let file = fs::File::open(path).map_err(|error| format!("Read error: {}", error))?;
    let file_size = file
        .metadata()
        .map_err(|error| format!("Metadata error: {}", error))?
        .len();
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_number = 0usize;
    let mut content = String::new();
    let mut bytes_read = 0usize;
    let mut lines_returned = 0usize;
    let mut truncated = false;

    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .map_err(|error| format!("Read error: {}", error))?;
        if read == 0 {
            break;
        }
        line_number += 1;
        if line_number < start_line {
            continue;
        }
        if line_number > end_line || lines_returned >= max_lines {
            truncated = true;
            break;
        }
        if bytes_read.saturating_add(read) > max_bytes as usize {
            truncated = true;
            break;
        }
        content.push_str(&line);
        bytes_read += read;
        lines_returned += 1;
        if line_number % 256 == 0 {
            tokio::task::yield_now().await;
        }
    }

    Ok(FileReadOutput {
        path: path.to_string_lossy().to_string(),
        content,
        start_line,
        end_line: if lines_returned == 0 {
            0
        } else {
            start_line + lines_returned - 1
        },
        total_lines: line_number,
        bytes_read,
        file_size,
        truncated,
        large_file: file_size > max_bytes,
        encoding: "utf-8".to_string(),
    })
}

fn kind_from_metadata(metadata: &fs::Metadata) -> &'static str {
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        "symlink"
    } else if metadata.is_dir() {
        "directory"
    } else if metadata.is_file() {
        "file"
    } else {
        "other"
    }
}

fn system_time_rfc3339(time: std::time::SystemTime) -> String {
    let datetime: chrono::DateTime<chrono::Utc> = time.into();
    datetime.to_rfc3339()
}

fn mime_hint(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    let mime = match ext.as_str() {
        "rs" | "toml" | "yaml" | "yml" | "json" | "md" | "txt" | "html" | "css" | "js" | "ts"
        | "tsx" | "jsx" | "py" | "sh" => "text/plain",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "pdf" => "application/pdf",
        _ => return None,
    };
    Some(mime.to_string())
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

fn json_result_with_caps<T: Serialize>(
    output: &T,
    truncated: bool,
    max_output_chars: Option<usize>,
) -> ToolResult {
    let json = match serde_json::to_string(output) {
        Ok(json) => json,
        Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
    };
    let (bounded, output_truncated) = if let Some(max) = max_output_chars {
        truncate_chars(json, max)
    } else {
        (json, false)
    };
    let mut metadata = HashMap::new();
    metadata.insert(
        "truncated".to_string(),
        Value::Bool(truncated || output_truncated),
    );
    if let Some(max) = max_output_chars {
        metadata.insert("max_output_chars".to_string(), Value::from(max));
    }
    ToolResult::ok_with_metadata(bounded, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn value(output: &str) -> Value {
        serde_json::from_str(output).unwrap()
    }

    #[tokio::test]
    async fn glob_sorts_and_offsets_results() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("b.rs"), "").unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::create_dir(dir.path().join("target")).unwrap();
        fs::write(dir.path().join("target/ignored.rs"), "").unwrap();
        let result = GlobTool::new()
            .execute(serde_json::json!({
                "pattern": "*.rs",
                "path": dir.path(),
                "max_results": 1,
                "offset": 1
            }))
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert_eq!(output["paths"][0], "b.rs");
        assert_eq!(output["total_count"], 2);
    }

    #[tokio::test]
    async fn grep_supports_literal_and_content_offsets() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("one.txt"), "alpha\nbeta\n").unwrap();
        fs::write(dir.path().join("two.txt"), "alpha\ngamma\n").unwrap();
        let result = GrepTool::new()
            .execute(serde_json::json!({
                "pattern": "alpha",
                "mode": "literal",
                "path": dir.path(),
                "output_mode": "content",
                "max_results": 1,
                "offset": 1
            }))
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert_eq!(output["matches"].as_array().unwrap().len(), 1);
        assert!(output["truncated"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn grep_skips_binary_files() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("bad.bin"), b"a\0b").unwrap();
        let result = GrepTool::new()
            .execute(serde_json::json!({
                "pattern": "a",
                "path": dir.path()
            }))
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert_eq!(output["skipped_binary"], 1);
    }

    #[tokio::test]
    async fn file_read_handles_line_ranges_and_unicode() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("unicode.txt");
        fs::write(&path, "one\n안녕\nthree\n").unwrap();
        let result = FileReadTool::new()
            .execute(serde_json::json!({
                "path": path,
                "start_line": 2,
                "end_line": 2
            }))
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert_eq!(output["content"], "안녕\n");
        assert_eq!(output["start_line"], 2);
    }

    #[tokio::test]
    async fn file_read_large_file_without_range_returns_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large.txt");
        fs::write(&path, "abcdef").unwrap();
        let result = FileReadTool::new()
            .execute(serde_json::json!({
                "path": path,
                "max_bytes": 2
            }))
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert!(output["large_file"].as_bool().unwrap());
        assert!(output["truncated"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn file_list_paginates_and_notes_symlink_escape() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "a").unwrap();
        fs::write(dir.path().join("b.txt"), "b").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("/", dir.path().join("escape")).unwrap();
        let result = FileListTool::new()
            .execute(serde_json::json!({
                "path": dir.path(),
                "recursive": true,
                "max_results": 1,
                "offset": 1
            }))
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert_eq!(output["entries"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn file_info_reports_symlink_policy() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("target.txt");
        fs::write(&target, "hello").unwrap();
        let link = dir.path().join("link.txt");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(&target, &link).unwrap();
        let result = FileInfoTool::new()
            .execute(serde_json::json!({
                "path": link,
                "follow_symlinks": true
            }))
            .await;
        assert!(result.success);
        let output = value(&result.output);
        assert!(output["symlink"].as_bool().unwrap());
        assert_eq!(output["policy_classification"], "allowed");
    }
}
