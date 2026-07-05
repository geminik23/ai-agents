use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

use ai_agents_core::{
    PathAccessMode, PathBindingKind, PathPolicyBinding, ResultLimitBinding, ResultLimitKind, Tool,
    ToolCallClassification, ToolExecutionContext, ToolOperationKind, ToolPolicyBindings,
    ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;
use crate::types::{FileVersionEvidence, FileVersionStore, file_version_evidence};

const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;
const DEFAULT_MAX_REPLACEMENTS: usize = 20;
const DEFAULT_MAX_CHANGED_FILES: usize = 10;
const DEFAULT_MAX_CHANGED_LINES: usize = 500;

/// Writes new files or policy-approved overwrites with atomic replacement.
pub struct FileWriteTool {
    versions: FileVersionStore,
}

impl FileWriteTool {
    /// Create a file-write tool with isolated version storage.
    pub fn new() -> Self {
        Self::with_version_store(FileVersionStore::default())
    }

    /// Create a file-write tool backed by shared read-version storage.
    pub fn with_version_store(versions: FileVersionStore) -> Self {
        Self { versions }
    }
}

impl Default for FileWriteTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Performs exact text replacement with dry-run diff output.
pub struct FileEditTool {
    versions: FileVersionStore,
}

impl FileEditTool {
    /// Create a file-edit tool with isolated version storage.
    pub fn new() -> Self {
        Self::with_version_store(FileVersionStore::default())
    }

    /// Create a file-edit tool backed by shared read-version storage.
    pub fn with_version_store(versions: FileVersionStore) -> Self {
        Self { versions }
    }
}

impl Default for FileEditTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Validates or applies bounded unified diffs after dry-run validation.
pub struct PatchTool {
    versions: FileVersionStore,
}

impl PatchTool {
    /// Create a patch tool with isolated version storage.
    pub fn new() -> Self {
        Self::with_version_store(FileVersionStore::default())
    }

    /// Create a patch tool backed by shared read-version storage.
    pub fn with_version_store(versions: FileVersionStore) -> Self {
        Self { versions }
    }
}

impl Default for PatchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileWriteInput {
    /// File path to create or overwrite.
    path: String,
    /// New file content.
    content: String,
    /// Allow overwriting an existing file.
    #[serde(default)]
    overwrite: bool,
    /// Create missing parent directories.
    #[serde(default)]
    create_parent_dirs: bool,
    /// Validate and return a summary without writing.
    #[serde(default)]
    dry_run: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileEditInput {
    /// File path to edit.
    path: String,
    /// Exact text to replace.
    old_text: String,
    /// Replacement text.
    new_text: String,
    /// Replace every occurrence. Defaults to requiring a unique match.
    #[serde(default)]
    replace_all: bool,
    /// Validate and return a diff without writing.
    #[serde(default)]
    dry_run: bool,
    /// Request-level replacement cap lowered by policy.
    #[serde(default)]
    max_replacements: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PatchInput {
    /// Unified diff text.
    patch: String,
    /// Base path for relative patch paths. Defaults to current directory.
    #[serde(default)]
    base_path: Option<String>,
    /// Validate and return a summary without applying. Defaults to true.
    #[serde(default = "default_true")]
    dry_run: bool,
    /// Permit creating new files when policy allows it.
    #[serde(default)]
    allow_new_files: Option<bool>,
    /// Permit deleting files. Defaults to false.
    #[serde(default)]
    allow_delete: bool,
}

#[derive(Debug, Serialize)]
struct MutationOutput {
    path: Option<String>,
    dry_run: bool,
    changed_files: usize,
    changed_lines: usize,
    replacements: usize,
    bytes_written: usize,
    created: bool,
    overwritten: bool,
    truncated: bool,
    approval_required: bool,
    diff_summary: String,
    changed_paths: Vec<String>,
    version: Option<FileVersionEvidence>,
    near_matches: Vec<String>,
}

#[derive(Debug, Default)]
struct MutationPolicySnapshot {
    write_paths: Vec<String>,
    allowed_paths: Vec<String>,
    blocked_paths: Vec<String>,
    overwrite_existing: bool,
    create_parent_dirs: bool,
    require_read_before_write: bool,
    no_write_policy: String,
    allow_without_confirmation: bool,
}

impl MutationPolicySnapshot {
    fn from_context(value: &Value) -> Self {
        let no_write_policy = value
            .get("no_write_policy")
            .and_then(Value::as_str)
            .unwrap_or("dry_run_only")
            .to_string();
        let mut snapshot = Self {
            write_paths: strings_at(value, "write_paths"),
            allowed_paths: strings_at(value, "allowed_paths"),
            blocked_paths: strings_at(value, "blocked_paths"),
            overwrite_existing: bool_at(value, "overwrite_existing"),
            create_parent_dirs: bool_at(value, "create_parent_dirs"),
            require_read_before_write: bool_at(value, "require_read_before_write"),
            no_write_policy,
            allow_without_confirmation: bool_at(value, "allow_without_confirmation"),
        };
        if let Some(paths) = value.get("paths") {
            snapshot.write_paths.extend(strings_at(paths, "allow"));
            snapshot.blocked_paths.extend(strings_at(paths, "deny"));
        }
        snapshot
    }

    fn has_write_policy(&self) -> bool {
        !self.write_paths.is_empty() || !self.allowed_paths.is_empty()
    }

    fn approval_required(&self) -> bool {
        !self.allow_without_confirmation
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn id(&self) -> &str {
        "file_write"
    }

    fn name(&self) -> &str {
        "File Write"
    }

    fn description(&self) -> &str {
        "Create or overwrite a file with policy-gated atomic writes and dry-run summaries."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileWriteInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        mutation_metadata(ToolOperationKind::Write)
    }

    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        mutation_classification(&self.safety_metadata(), args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::write("path")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_changed_files", ResultLimitKind::MaxChangedFiles),
                ResultLimitBinding::new("max_changed_lines", ResultLimitKind::MaxChangedLines),
            ],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: FileWriteInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let path = PathBuf::from(&input.path);
        if let Err(reason) = validate_safe_target(&path) {
            return ToolResult::error(reason);
        }
        let policy = MutationPolicySnapshot::from_context(&ctx.policy_snapshot);
        if let Err(reason) = ensure_write_allowed(&path, input.dry_run, &policy) {
            return ToolResult::error(reason);
        }
        let exists = path.exists();
        if exists && !input.overwrite {
            return ToolResult::error("overwrite must be true to replace an existing file");
        }
        if exists && !policy.overwrite_existing && !input.dry_run {
            return ToolResult::error("overwrite_existing policy is false for this target");
        }
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if !(input.create_parent_dirs && policy.create_parent_dirs) {
                    return ToolResult::error(
                        "parent directory does not exist or create_parent_dirs is not allowed",
                    );
                }
                if !input.dry_run {
                    if let Err(error) = fs::create_dir_all(parent) {
                        return ToolResult::error(format!(
                            "Create parent directory error: {}",
                            error
                        ));
                    }
                }
            }
        }
        if exists && !input.dry_run {
            if let Err(reason) = enforce_read_before_write(&self.versions, &path, &policy) {
                return ToolResult::error(reason);
            }
        }
        let changed_lines = input
            .content
            .lines()
            .count()
            .max(usize::from(!input.content.is_empty()));
        if exceeds(ctx.limits.max_changed_files, 1)
            || exceeds(ctx.limits.max_changed_lines, changed_lines)
        {
            return ToolResult::error(
                "mutation exceeds configured changed-file or changed-line cap",
            );
        }
        let diff_summary = if exists {
            format!(
                "overwrite {} with {} bytes",
                input.path,
                input.content.len()
            )
        } else {
            format!("create {} with {} bytes", input.path, input.content.len())
        };
        let version = if input.dry_run {
            None
        } else {
            if let Err(error) = atomic_write(&path, input.content.as_bytes()) {
                return ToolResult::error(format!("Write error: {}", error));
            }
            match file_version_evidence(&path, input.content.as_bytes()) {
                Ok(version) => {
                    self.versions.record(version.clone());
                    Some(version)
                }
                Err(_) => None,
            }
        };
        json_result(&MutationOutput {
            path: Some(input.path.clone()),
            dry_run: input.dry_run,
            changed_files: 1,
            changed_lines,
            replacements: 0,
            bytes_written: if input.dry_run {
                0
            } else {
                input.content.len()
            },
            created: !exists,
            overwritten: exists,
            truncated: false,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary,
            changed_paths: vec![input.path],
            version,
            near_matches: Vec::new(),
        })
    }
}

#[async_trait]
impl Tool for FileEditTool {
    fn id(&self) -> &str {
        "file_edit"
    }

    fn name(&self) -> &str {
        "File Edit"
    }

    fn description(&self) -> &str {
        "Replace exact text in a file with uniqueness checks, dry-run diff summaries, and policy-gated writes."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileEditInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        mutation_metadata(ToolOperationKind::Edit)
    }

    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        mutation_classification(&self.safety_metadata(), args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::write("path")],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_replacements", ResultLimitKind::MaxReplacements),
                ResultLimitBinding::new("max_changed_lines", ResultLimitKind::MaxChangedLines),
            ],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: FileEditInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        if input.old_text == input.new_text {
            return ToolResult::error("old_text and new_text must differ");
        }
        let path = PathBuf::from(&input.path);
        if let Err(reason) = validate_safe_target(&path) {
            return ToolResult::error(reason);
        }
        let policy = MutationPolicySnapshot::from_context(&ctx.policy_snapshot);
        if let Err(reason) = ensure_write_allowed(&path, input.dry_run, &policy) {
            return ToolResult::error(reason);
        }
        let original = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) => return ToolResult::error(format!("Read error: {}", error)),
        };
        let matches: Vec<_> = original.match_indices(&input.old_text).collect();
        if matches.is_empty() {
            let output = MutationOutput {
                path: Some(input.path),
                dry_run: input.dry_run,
                changed_files: 0,
                changed_lines: 0,
                replacements: 0,
                bytes_written: 0,
                created: false,
                overwritten: false,
                truncated: false,
                approval_required: false,
                diff_summary: "old_text was not found".to_string(),
                changed_paths: Vec::new(),
                version: None,
                near_matches: near_matches(&original, &input.old_text),
            };
            return match serde_json::to_string(&output) {
                Ok(json) => ToolResult {
                    success: false,
                    output: json,
                    metadata: None,
                },
                Err(error) => ToolResult::error(format!("Serialization error: {}", error)),
            };
        }
        if matches.len() > 1 && !input.replace_all {
            return ToolResult::error(
                "old_text appears multiple times; set replace_all to true to replace all occurrences",
            );
        }
        let replacements = if input.replace_all { matches.len() } else { 1 };
        let max_replacements = input
            .max_replacements
            .unwrap_or(DEFAULT_MAX_REPLACEMENTS)
            .min(
                ctx.limits
                    .max_replacements
                    .unwrap_or(DEFAULT_MAX_REPLACEMENTS),
            );
        if replacements > max_replacements {
            return ToolResult::error("replacement count exceeds configured max_replacements");
        }
        if !input.dry_run {
            if let Err(reason) = enforce_read_before_write(&self.versions, &path, &policy) {
                return ToolResult::error(reason);
            }
        }
        let edited = if input.replace_all {
            original.replace(&input.old_text, &input.new_text)
        } else {
            original.replacen(&input.old_text, &input.new_text, 1)
        };
        let changed_lines = changed_line_count(&original, &edited);
        if exceeds(ctx.limits.max_changed_lines, changed_lines) {
            return ToolResult::error("edit exceeds configured max_changed_lines");
        }
        let diff_summary = preview_diff(&original, &edited, DEFAULT_MAX_OUTPUT_CHARS);
        let version = if input.dry_run {
            None
        } else {
            if let Err(error) = atomic_write(&path, edited.as_bytes()) {
                return ToolResult::error(format!("Write error: {}", error));
            }
            match file_version_evidence(&path, edited.as_bytes()) {
                Ok(version) => {
                    self.versions.record(version.clone());
                    Some(version)
                }
                Err(_) => None,
            }
        };
        json_result(&MutationOutput {
            path: Some(input.path.clone()),
            dry_run: input.dry_run,
            changed_files: 1,
            changed_lines,
            replacements,
            bytes_written: if input.dry_run { 0 } else { edited.len() },
            created: false,
            overwritten: true,
            truncated: diff_summary.chars().count() >= DEFAULT_MAX_OUTPUT_CHARS,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary,
            changed_paths: vec![input.path],
            version,
            near_matches: Vec::new(),
        })
    }
}

#[async_trait]
impl Tool for PatchTool {
    fn id(&self) -> &str {
        "patch"
    }

    fn name(&self) -> &str {
        "Patch"
    }

    fn description(&self) -> &str {
        "Validate or apply bounded unified diffs with per-file write policy checks."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<PatchInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        mutation_metadata(ToolOperationKind::Patch)
    }

    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        mutation_classification(&self.safety_metadata(), args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            path_fields: vec![
                PathPolicyBinding::new(
                    "base_path",
                    PathAccessMode::Write,
                    PathBindingKind::PatchBase,
                )
                .with_default_path("."),
            ],
            result_limit_fields: vec![
                ResultLimitBinding::new("max_changed_files", ResultLimitKind::MaxChangedFiles),
                ResultLimitBinding::new("max_changed_lines", ResultLimitKind::MaxChangedLines),
            ],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: PatchInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let base_path = PathBuf::from(input.base_path.unwrap_or_else(|| ".".to_string()));
        if let Err(reason) = validate_safe_target(&base_path) {
            return ToolResult::error(reason);
        }
        let policy = MutationPolicySnapshot::from_context(&ctx.policy_snapshot);
        let files = match parse_unified_diff(&input.patch) {
            Ok(files) => files,
            Err(error) => return ToolResult::error(error),
        };
        if files.is_empty() {
            return ToolResult::error("patch contains no changed files");
        }
        if exceeds(ctx.limits.max_changed_files, files.len())
            || files.len() > DEFAULT_MAX_CHANGED_FILES
        {
            return ToolResult::error("patch exceeds configured max_changed_files");
        }
        let mut changed_lines = 0usize;
        let mut changed_paths = Vec::new();
        let mut outputs = Vec::new();
        for file in &files {
            changed_lines += file.changed_lines();
            if changed_lines
                > ctx
                    .limits
                    .max_changed_lines
                    .unwrap_or(DEFAULT_MAX_CHANGED_LINES)
            {
                return ToolResult::error("patch exceeds configured max_changed_lines");
            }
            if file.is_delete() && !input.allow_delete {
                return ToolResult::error("patch deletes a file but allow_delete is false");
            }
            let path = base_path.join(strip_patch_prefix(file.target_path()));
            if let Err(reason) = validate_safe_target(&path) {
                return ToolResult::error(reason);
            }
            if let Err(reason) = ensure_write_allowed(&path, input.dry_run, &policy) {
                return ToolResult::error(reason);
            }
            let exists = path.exists();
            if file.is_delete() && !exists {
                return ToolResult::error("patch deletes a file that does not exist");
            }
            if file.is_new_file() && !input.allow_new_files.unwrap_or(false) {
                return ToolResult::error("patch creates a file but allow_new_files is false");
            }
            if let Some(parent) = path.parent() {
                if !parent.exists() && !policy.create_parent_dirs && !input.dry_run {
                    return ToolResult::error(
                        "patch parent directory creation is not allowed by policy",
                    );
                }
            }
            if exists && !input.dry_run {
                if let Err(reason) = enforce_read_before_write(&self.versions, &path, &policy) {
                    return ToolResult::error(reason);
                }
            }
            let original = if exists {
                match fs::read_to_string(&path) {
                    Ok(content) => content,
                    Err(error) => return ToolResult::error(format!("Read error: {}", error)),
                }
            } else {
                String::new()
            };
            let edited = match apply_file_patch(&original, file) {
                Ok(edited) => edited,
                Err(error) => return ToolResult::error(error),
            };
            changed_paths.push(path.to_string_lossy().to_string());
            if file.is_delete() {
                outputs.push(PatchApply::Delete(path));
            } else {
                outputs.push(PatchApply::Write(path, edited));
            }
        }
        if !input.dry_run {
            for output in &outputs {
                match output {
                    PatchApply::Delete(path) => {
                        if let Err(error) = fs::remove_file(path) {
                            return ToolResult::error(format!("Patch delete error: {}", error));
                        }
                    }
                    PatchApply::Write(path, content) => {
                        if let Some(parent) = path.parent() {
                            if !parent.exists() {
                                if !policy.create_parent_dirs {
                                    return ToolResult::error(
                                        "patch parent directory creation is not allowed by policy",
                                    );
                                }
                                if let Err(error) = fs::create_dir_all(parent) {
                                    return ToolResult::error(format!(
                                        "Create parent directory error: {}",
                                        error
                                    ));
                                }
                            }
                        }
                        if let Err(error) = atomic_write(path, content.as_bytes()) {
                            return ToolResult::error(format!("Patch write error: {}", error));
                        }
                        if let Ok(version) = file_version_evidence(path, content.as_bytes()) {
                            self.versions.record(version);
                        }
                    }
                }
            }
        }
        let diff_summary = summarize_patch(&files, DEFAULT_MAX_OUTPUT_CHARS);
        json_result(&MutationOutput {
            path: Some(base_path.to_string_lossy().to_string()),
            dry_run: input.dry_run,
            changed_files: files.len(),
            changed_lines,
            replacements: 0,
            bytes_written: if input.dry_run {
                0
            } else {
                outputs
                    .iter()
                    .map(|output| match output {
                        PatchApply::Write(_, content) => content.len(),
                        PatchApply::Delete(_) => 0,
                    })
                    .sum()
            },
            created: false,
            overwritten: !outputs.is_empty(),
            truncated: diff_summary.chars().count() >= DEFAULT_MAX_OUTPUT_CHARS,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary,
            changed_paths,
            version: None,
            near_matches: Vec::new(),
        })
    }
}

#[derive(Debug, Clone)]
struct PatchFile {
    old_path: String,
    new_path: String,
    hunks: Vec<PatchHunk>,
}

impl PatchFile {
    fn target_path(&self) -> &str {
        if self.new_path == "/dev/null" {
            &self.old_path
        } else {
            &self.new_path
        }
    }

    fn is_delete(&self) -> bool {
        self.new_path == "/dev/null"
    }

    fn is_new_file(&self) -> bool {
        self.old_path == "/dev/null"
    }

    fn changed_lines(&self) -> usize {
        self.hunks
            .iter()
            .flat_map(|hunk| hunk.lines.iter())
            .filter(|line| matches!(line.kind, PatchLineKind::Add | PatchLineKind::Remove))
            .count()
    }
}

#[derive(Debug, Clone)]
enum PatchApply {
    Write(PathBuf, String),
    Delete(PathBuf),
}

#[derive(Debug, Clone)]
struct PatchHunk {
    old_start: usize,
    lines: Vec<PatchLine>,
}

#[derive(Debug, Clone)]
struct PatchLine {
    kind: PatchLineKind,
    text: String,
}

#[derive(Debug, Clone, Copy)]
enum PatchLineKind {
    Context,
    Add,
    Remove,
}

fn mutation_metadata(operation: ToolOperationKind) -> ToolSafetyMetadata {
    ToolSafetyMetadata {
        read_only: false,
        concurrency_safe: false,
        operation,
        side_effect_level: ToolSideEffectLevel::LocalWrite,
        requires_network: false,
        destructive: false,
        open_world: false,
        host_dependent: false,
        requires_user_interaction: false,
        supports_cancellation: true,
        default_requires_approval: true,
        should_defer_schema: false,
        max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
    }
}

fn mutation_classification(metadata: &ToolSafetyMetadata, args: &Value) -> ToolCallClassification {
    let default_dry_run = matches!(metadata.operation, ToolOperationKind::Patch);
    let dry_run = args
        .get("dry_run")
        .and_then(Value::as_bool)
        .unwrap_or(default_dry_run);
    let mut classification = ToolCallClassification::from_metadata(metadata);
    classification.safely_retryable = dry_run;
    if dry_run {
        classification.read_only = true;
        classification.concurrency_safe = true;
        classification.side_effect_level = ToolSideEffectLevel::None;
        classification.requires_approval = false;
    }
    classification
}

fn ensure_write_allowed(
    path: &Path,
    dry_run: bool,
    policy: &MutationPolicySnapshot,
) -> Result<(), String> {
    let normalized = normalize_path(path);
    let resolved = resolve_existing_or_parent(path)?;
    for blocked in &policy.blocked_paths {
        if path_matches_policy(blocked, &normalized, &resolved)? {
            return Err("path is blocked by policy".to_string());
        }
    }
    if !policy.has_write_policy() {
        if policy.no_write_policy == "deny" || !dry_run {
            return Err("actual mutation requires explicit write_paths policy".to_string());
        }
        return Ok(());
    }
    if !policy
        .write_paths
        .iter()
        .chain(policy.allowed_paths.iter())
        .any(|allowed| path_matches_policy(allowed, &normalized, &resolved).unwrap_or(false))
    {
        return Err("path is not under an allowed write root".to_string());
    }
    Ok(())
}

fn enforce_read_before_write(
    versions: &FileVersionStore,
    path: &Path,
    policy: &MutationPolicySnapshot,
) -> Result<(), String> {
    if !policy.require_read_before_write {
        return Ok(());
    }
    let bytes =
        fs::read(path).map_err(|error| format!("Read-before-write check failed: {}", error))?;
    let current = file_version_evidence(path, &bytes)
        .map_err(|error| format!("Read-before-write version failed: {}", error))?;
    if versions.matches(&current) {
        Ok(())
    } else {
        Err(
            "file must be read with file_read before mutation and must not change before write"
                .to_string(),
        )
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let temp = parent.join(format!(".{}.tmp", Uuid::new_v4()));
    {
        let mut file = fs::File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(temp, path)
}

fn validate_safe_target(path: &Path) -> Result<(), String> {
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err("paths containing parent-directory traversal are not allowed".to_string());
        }
    }
    if path
        .components()
        .any(|component| component.as_os_str() == ".git")
    {
        return Err("raw .git paths are not allowed".to_string());
    }
    Ok(())
}

fn changed_line_count(before: &str, after: &str) -> usize {
    let before_lines: Vec<_> = before.lines().collect();
    let after_lines: Vec<_> = after.lines().collect();
    before_lines
        .iter()
        .zip(after_lines.iter())
        .filter(|(left, right)| left != right)
        .count()
        + before_lines.len().abs_diff(after_lines.len())
}

fn preview_diff(before: &str, after: &str, max_chars: usize) -> String {
    let before_lines: Vec<_> = before.lines().collect();
    let after_lines: Vec<_> = after.lines().collect();
    let mut output = String::new();
    for (index, (left, right)) in before_lines.iter().zip(after_lines.iter()).enumerate() {
        if left != right {
            output.push_str(&format!(
                "-{}:{}\n+{}:{}\n",
                index + 1,
                left,
                index + 1,
                right
            ));
        }
        if output.chars().count() >= max_chars {
            return output.chars().take(max_chars).collect();
        }
    }
    if before_lines.len() != after_lines.len() {
        output.push_str(&format!(
            "line count changed from {} to {}\n",
            before_lines.len(),
            after_lines.len()
        ));
    }
    output.chars().take(max_chars).collect()
}

fn near_matches(content: &str, needle: &str) -> Vec<String> {
    let prefix: String = needle.chars().take(16).collect();
    if prefix.is_empty() {
        return Vec::new();
    }
    content
        .lines()
        .filter(|line| line.contains(&prefix))
        .take(3)
        .map(str::to_string)
        .collect()
}

fn parse_unified_diff(input: &str) -> Result<Vec<PatchFile>, String> {
    let mut files = Vec::new();
    let mut current: Option<PatchFile> = None;
    let mut current_hunk: Option<PatchHunk> = None;
    for line in input.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            if let Some(hunk) = current_hunk.take() {
                if let Some(file) = current.as_mut() {
                    file.hunks.push(hunk);
                }
            }
            if let Some(file) = current.take() {
                files.push(file);
            }
            current = Some(PatchFile {
                old_path: clean_patch_path(rest),
                new_path: String::new(),
                hunks: Vec::new(),
            });
        } else if let Some(rest) = line.strip_prefix("+++ ") {
            let Some(file) = current.as_mut() else {
                return Err("patch has +++ before ---".to_string());
            };
            file.new_path = clean_patch_path(rest);
        } else if line.starts_with("@@") {
            if let Some(hunk) = current_hunk.take() {
                if let Some(file) = current.as_mut() {
                    file.hunks.push(hunk);
                }
            }
            current_hunk = Some(PatchHunk {
                old_start: parse_hunk_old_start(line)?,
                lines: Vec::new(),
            });
        } else if let Some(hunk) = current_hunk.as_mut() {
            if let Some(text) = line.strip_prefix(' ') {
                hunk.lines.push(PatchLine {
                    kind: PatchLineKind::Context,
                    text: text.to_string(),
                });
            } else if let Some(text) = line.strip_prefix('+') {
                hunk.lines.push(PatchLine {
                    kind: PatchLineKind::Add,
                    text: text.to_string(),
                });
            } else if let Some(text) = line.strip_prefix('-') {
                hunk.lines.push(PatchLine {
                    kind: PatchLineKind::Remove,
                    text: text.to_string(),
                });
            }
        }
    }
    if let Some(hunk) = current_hunk {
        if let Some(file) = current.as_mut() {
            file.hunks.push(hunk);
        }
    }
    if let Some(file) = current {
        files.push(file);
    }
    for file in &files {
        if file.new_path.is_empty() || file.hunks.is_empty() {
            return Err("patch file is missing target path or hunks".to_string());
        }
    }
    Ok(files)
}

fn apply_file_patch(original: &str, patch: &PatchFile) -> Result<String, String> {
    let newline = if original.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let original_lines: Vec<String> = if original.is_empty() {
        Vec::new()
    } else {
        original.lines().map(str::to_string).collect()
    };
    let mut output = Vec::new();
    let mut cursor = 0usize;
    for hunk in &patch.hunks {
        let start = hunk.old_start.saturating_sub(1);
        if start < cursor || start > original_lines.len() {
            return Err("patch hunk does not align with file contents".to_string());
        }
        output.extend_from_slice(&original_lines[cursor..start]);
        cursor = start;
        for line in &hunk.lines {
            match line.kind {
                PatchLineKind::Context => {
                    if original_lines.get(cursor).map(String::as_str) != Some(line.text.as_str()) {
                        return Err("patch context does not match file contents".to_string());
                    }
                    output.push(line.text.clone());
                    cursor += 1;
                }
                PatchLineKind::Remove => {
                    if original_lines.get(cursor).map(String::as_str) != Some(line.text.as_str()) {
                        return Err("patch removal does not match file contents".to_string());
                    }
                    cursor += 1;
                }
                PatchLineKind::Add => output.push(line.text.clone()),
            }
        }
    }
    output.extend_from_slice(&original_lines[cursor..]);
    let mut joined = output.join(newline);
    if original.ends_with('\n') || !joined.is_empty() {
        joined.push_str(newline);
    }
    Ok(joined)
}

fn summarize_patch(files: &[PatchFile], max_chars: usize) -> String {
    let mut output = String::new();
    for file in files {
        output.push_str(&format!(
            "{} -> {} ({} changed lines)\n",
            file.old_path,
            file.new_path,
            file.changed_lines()
        ));
    }
    output.chars().take(max_chars).collect()
}

fn parse_hunk_old_start(line: &str) -> Result<usize, String> {
    let start = line
        .split_whitespace()
        .find(|part| part.starts_with('-'))
        .ok_or_else(|| "invalid hunk header".to_string())?;
    let start = start
        .trim_start_matches('-')
        .split(',')
        .next()
        .unwrap_or("1");
    start
        .parse::<usize>()
        .map_err(|_| "invalid hunk start".to_string())
}

fn clean_patch_path(path: &str) -> String {
    path.split_whitespace().next().unwrap_or(path).to_string()
}

fn strip_patch_prefix(path: &str) -> &str {
    path.strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path)
}

fn strings_at(value: &Value, field: &str) -> Vec<String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn bool_at(value: &Value, field: &str) -> bool {
    value.get(field).and_then(Value::as_bool).unwrap_or(false)
}

fn normalize_path(path: &Path) -> PathBuf {
    let base = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    base.components().collect()
}

fn resolve_existing_or_parent(path: &Path) -> Result<PathBuf, String> {
    if path.exists() {
        return path
            .canonicalize()
            .map_err(|error| format!("Path canonicalization failed: {}", error));
    }
    let normalized = normalize_path(path);
    let mut ancestor = normalized.as_path();
    let mut missing = Vec::new();
    while !ancestor.exists() {
        let Some(name) = ancestor.file_name() else {
            break;
        };
        missing.push(name.to_os_string());
        ancestor = ancestor.parent().unwrap_or_else(|| Path::new("."));
    }
    let mut resolved = ancestor
        .canonicalize()
        .map_err(|error| format!("Path parent canonicalization failed: {}", error))?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved.components().collect())
}

fn path_matches_policy(pattern: &str, normalized: &Path, resolved: &Path) -> Result<bool, String> {
    let pattern_path = Path::new(pattern);
    let normalized_pattern = normalize_path(pattern_path);
    if !normalized.starts_with(&normalized_pattern) {
        return Ok(false);
    }
    if pattern_path.exists() {
        let resolved_pattern = pattern_path
            .canonicalize()
            .map_err(|error| format!("Policy path canonicalization failed: {}", error))?;
        return Ok(resolved.starts_with(resolved_pattern.components().collect::<PathBuf>()));
    }
    Ok(true)
}

fn exceeds(limit: Option<usize>, value: usize) -> bool {
    limit.is_some_and(|limit| value > limit)
}

fn default_true() -> bool {
    true
}

fn json_result<T: Serialize>(output: &T) -> ToolResult {
    match serde_json::to_string(output) {
        Ok(json) => ToolResult::ok(json),
        Err(error) => ToolResult::error(format!("Serialization error: {}", error)),
    }
}

/// Copies a file or directory tree with policy-gated dry-run previews.
pub struct CopyPathTool;

impl CopyPathTool {
    /// Create a copy tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CopyPathTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Moves a file or directory with policy-gated dry-run previews.
pub struct MovePathTool;

impl MovePathTool {
    /// Create a move tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MovePathTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Deletes a file or directory with policy-gated dry-run previews.
pub struct DeletePathTool;

impl DeletePathTool {
    /// Create a delete tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DeletePathTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CopyPathInput {
    /// Source path to copy from.
    source_path: String,
    /// Destination path to copy to.
    destination_path: String,
    /// Allow replacing an existing destination.
    #[serde(default)]
    overwrite: bool,
    /// Create missing parent directories for the destination.
    #[serde(default)]
    create_parent_dirs: bool,
    /// Validate and return a summary without copying.
    #[serde(default = "default_true")]
    dry_run: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MovePathInput {
    /// Source path to move from.
    source_path: String,
    /// Destination path to move to.
    destination_path: String,
    /// Allow replacing an existing destination.
    #[serde(default)]
    overwrite: bool,
    /// Create missing parent directories for the destination.
    #[serde(default)]
    create_parent_dirs: bool,
    /// Validate and return a summary without moving.
    #[serde(default = "default_true")]
    dry_run: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DeletePathInput {
    /// Path to delete.
    path: String,
    /// Remove a directory and its contents. Required for directories.
    #[serde(default)]
    recursive: bool,
    /// Validate and return a summary without deleting.
    #[serde(default = "default_true")]
    dry_run: bool,
}

#[derive(Debug, Serialize)]
struct PathMutationOutput {
    source_path: Option<String>,
    destination_path: Option<String>,
    path: Option<String>,
    dry_run: bool,
    copied: bool,
    moved: bool,
    deleted: bool,
    recursive: bool,
    overwritten: bool,
    bytes_affected: usize,
    items_affected: usize,
    approval_required: bool,
    diff_summary: String,
}

#[async_trait]
impl Tool for CopyPathTool {
    fn id(&self) -> &str {
        "copy_path"
    }

    fn name(&self) -> &str {
        "Copy Path"
    }

    fn description(&self) -> &str {
        "Copy a file or directory tree with source and destination policy checks and dry-run previews."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<CopyPathInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        path_mutation_metadata(ToolOperationKind::Write)
    }

    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        path_mutation_classification(args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            path_fields: vec![
                PathPolicyBinding::read("source_path"),
                PathPolicyBinding::write("destination_path"),
            ],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: CopyPathInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let source = PathBuf::from(&input.source_path);
        let destination = PathBuf::from(&input.destination_path);
        if let Err(reason) =
            validate_safe_target(&source).and_then(|_| validate_safe_target(&destination))
        {
            return ToolResult::error(reason);
        }
        let policy = MutationPolicySnapshot::from_context(&ctx.policy_snapshot);
        if let Err(reason) = ensure_read_allowed(&source, &policy) {
            return ToolResult::error(reason);
        }
        if let Err(reason) = ensure_write_allowed(&destination, input.dry_run, &policy) {
            return ToolResult::error(reason);
        }
        if !source.exists() {
            return ToolResult::error(format!("source path does not exist: {}", input.source_path));
        }
        let destination_exists = destination.exists();
        if destination_exists && !input.overwrite {
            return ToolResult::error("overwrite must be true to replace an existing destination");
        }
        if destination_exists && !policy.overwrite_existing && !input.dry_run {
            return ToolResult::error("overwrite_existing policy is false for this destination");
        }
        if let Some(parent) = destination.parent() {
            if !parent.exists() {
                if !(input.create_parent_dirs && policy.create_parent_dirs) {
                    return ToolResult::error(
                        "destination parent directory does not exist or create_parent_dirs is not allowed",
                    );
                }
                if !input.dry_run {
                    if let Err(error) = fs::create_dir_all(parent) {
                        return ToolResult::error(format!(
                            "Create parent directory error: {}",
                            error
                        ));
                    }
                }
            }
        }
        let bytes_affected = path_size(&source).unwrap_or(0);
        let items_affected = path_item_count(&source).unwrap_or(1);
        if !input.dry_run {
            if let Err(error) = copy_path(&source, &destination) {
                return ToolResult::error(format!("Copy error: {}", error));
            }
        }
        json_result(&PathMutationOutput {
            source_path: Some(input.source_path.clone()),
            destination_path: Some(input.destination_path.clone()),
            path: None,
            dry_run: input.dry_run,
            copied: true,
            moved: false,
            deleted: false,
            recursive: source.is_dir(),
            overwritten: destination_exists,
            bytes_affected,
            items_affected,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary: format!(
                "copy {} -> {} ({} bytes)",
                input.source_path, input.destination_path, bytes_affected
            ),
        })
    }
}

#[async_trait]
impl Tool for MovePathTool {
    fn id(&self) -> &str {
        "move_path"
    }

    fn name(&self) -> &str {
        "Move Path"
    }

    fn description(&self) -> &str {
        "Move or rename a file or directory with source and destination policy checks and dry-run previews."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<MovePathInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        path_mutation_metadata(ToolOperationKind::Write)
    }

    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        path_mutation_classification(args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            path_fields: vec![
                PathPolicyBinding::read_write("source_path"),
                PathPolicyBinding::write("destination_path"),
            ],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: MovePathInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let source = PathBuf::from(&input.source_path);
        let destination = PathBuf::from(&input.destination_path);
        if let Err(reason) =
            validate_safe_target(&source).and_then(|_| validate_safe_target(&destination))
        {
            return ToolResult::error(reason);
        }
        let policy = MutationPolicySnapshot::from_context(&ctx.policy_snapshot);
        if let Err(reason) = ensure_write_allowed(&source, input.dry_run, &policy) {
            return ToolResult::error(reason);
        }
        if let Err(reason) = ensure_write_allowed(&destination, input.dry_run, &policy) {
            return ToolResult::error(reason);
        }
        if !source.exists() {
            return ToolResult::error(format!("source path does not exist: {}", input.source_path));
        }
        let destination_exists = destination.exists();
        if destination_exists && !input.overwrite {
            return ToolResult::error("overwrite must be true to replace an existing destination");
        }
        if destination_exists && !policy.overwrite_existing && !input.dry_run {
            return ToolResult::error("overwrite_existing policy is false for this destination");
        }
        if let Some(parent) = destination.parent() {
            if !parent.exists() {
                if !(input.create_parent_dirs && policy.create_parent_dirs) {
                    return ToolResult::error(
                        "destination parent directory does not exist or create_parent_dirs is not allowed",
                    );
                }
                if !input.dry_run {
                    if let Err(error) = fs::create_dir_all(parent) {
                        return ToolResult::error(format!(
                            "Create parent directory error: {}",
                            error
                        ));
                    }
                }
            }
        }
        let bytes_affected = path_size(&source).unwrap_or(0);
        let items_affected = path_item_count(&source).unwrap_or(1);
        if !input.dry_run {
            if let Err(error) = fs::rename(&source, &destination) {
                return ToolResult::error(format!("Move error: {}", error));
            }
        }
        json_result(&PathMutationOutput {
            source_path: Some(input.source_path.clone()),
            destination_path: Some(input.destination_path.clone()),
            path: None,
            dry_run: input.dry_run,
            copied: false,
            moved: true,
            deleted: false,
            recursive: source.is_dir(),
            overwritten: destination_exists,
            bytes_affected,
            items_affected,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary: format!(
                "move {} -> {} ({} bytes)",
                input.source_path, input.destination_path, bytes_affected
            ),
        })
    }
}

#[async_trait]
impl Tool for DeletePathTool {
    fn id(&self) -> &str {
        "delete_path"
    }

    fn name(&self) -> &str {
        "Delete Path"
    }

    fn description(&self) -> &str {
        "Delete a file or directory with explicit policy checks, recursive-delete gating, and dry-run previews."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<DeletePathInput>()
    }

    fn safety_metadata(&self) -> ToolSafetyMetadata {
        path_mutation_metadata(ToolOperationKind::Delete)
    }

    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        path_mutation_classification(args)
    }

    fn policy_bindings(&self) -> ToolPolicyBindings {
        ToolPolicyBindings {
            path_fields: vec![PathPolicyBinding::write("path")],
            ..Default::default()
        }
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        let input: DeletePathInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(error) => return ToolResult::error(format!("Invalid input: {}", error)),
        };
        let path = PathBuf::from(&input.path);
        if let Err(reason) = validate_safe_target(&path) {
            return ToolResult::error(reason);
        }
        let policy = MutationPolicySnapshot::from_context(&ctx.policy_snapshot);
        if let Err(reason) = ensure_write_allowed(&path, input.dry_run, &policy) {
            return ToolResult::error(reason);
        }
        if !path.exists() {
            return ToolResult::error(format!("path does not exist: {}", input.path));
        }
        if path.is_dir() && !input.recursive {
            return ToolResult::error("recursive must be true to delete a directory");
        }
        let bytes_affected = path_size(&path).unwrap_or(0);
        let items_affected = path_item_count(&path).unwrap_or(1);
        if !input.dry_run {
            let result = if path.is_dir() {
                fs::remove_dir_all(&path)
            } else {
                fs::remove_file(&path)
            };
            if let Err(error) = result {
                return ToolResult::error(format!("Delete error: {}", error));
            }
        }
        json_result(&PathMutationOutput {
            source_path: None,
            destination_path: None,
            path: Some(input.path.clone()),
            dry_run: input.dry_run,
            copied: false,
            moved: false,
            deleted: true,
            recursive: path.is_dir(),
            overwritten: false,
            bytes_affected,
            items_affected,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary: format!(
                "delete {} ({} bytes, {} items)",
                input.path, bytes_affected, items_affected
            ),
        })
    }
}

fn path_mutation_metadata(operation: ToolOperationKind) -> ToolSafetyMetadata {
    ToolSafetyMetadata {
        read_only: false,
        concurrency_safe: false,
        operation,
        side_effect_level: if matches!(operation, ToolOperationKind::Delete) {
            ToolSideEffectLevel::Destructive
        } else {
            ToolSideEffectLevel::LocalWrite
        },
        requires_network: false,
        destructive: matches!(operation, ToolOperationKind::Delete),
        open_world: false,
        host_dependent: false,
        requires_user_interaction: false,
        supports_cancellation: true,
        default_requires_approval: true,
        should_defer_schema: false,
        max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
    }
}

fn path_mutation_classification(args: &Value) -> ToolCallClassification {
    let dry_run = args.get("dry_run").and_then(Value::as_bool).unwrap_or(true);
    let mut classification =
        ToolCallClassification::from_metadata(&path_mutation_metadata(ToolOperationKind::Write));
    classification.safely_retryable = dry_run;
    if dry_run {
        classification.read_only = true;
        classification.concurrency_safe = true;
        classification.side_effect_level = ToolSideEffectLevel::None;
        classification.requires_approval = false;
    }
    classification
}

fn ensure_read_allowed(path: &Path, policy: &MutationPolicySnapshot) -> Result<(), String> {
    let normalized = normalize_path(path);
    let resolved = resolve_existing_or_parent(path)?;
    for blocked in &policy.blocked_paths {
        if path_matches_policy(blocked, &normalized, &resolved)? {
            return Err("source path is blocked by policy".to_string());
        }
    }
    Ok(())
}

fn copy_path(source: &Path, destination: &Path) -> std::io::Result<()> {
    if source.is_dir() {
        copy_directory(source, destination)
    } else {
        if let Some(parent) = destination.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::copy(source, destination)?;
        Ok(())
    }
}

fn copy_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        if from.is_dir() {
            copy_directory(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn path_size(path: &Path) -> std::io::Result<usize> {
    if path.is_dir() {
        let mut total = 0usize;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            total += path_size(&entry.path())?;
        }
        Ok(total)
    } else {
        Ok(fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0))
    }
}

fn path_item_count(path: &Path) -> std::io::Result<usize> {
    if path.is_dir() {
        let mut total = 1usize;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            total += path_item_count(&entry.path())?;
        }
        Ok(total)
    } else {
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn file_edit_dry_run_requires_unique_match() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello\nhello\n").unwrap();
        let tool = FileEditTool::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "hello",
                    "new_text": "hi",
                    "dry_run": true
                }),
                ToolExecutionContext::test("file_edit"),
            )
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn file_edit_missing_old_text_fails_with_near_matches() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "Status: draft\n").unwrap();
        let tool = FileEditTool::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "Status: ready",
                    "new_text": "Status: reviewed",
                    "dry_run": true
                }),
                ToolExecutionContext::test("file_edit"),
            )
            .await;
        assert!(!result.success);
        let output: Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["diff_summary"], "old_text was not found");
    }

    #[tokio::test]
    async fn file_write_denies_actual_without_write_policy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let tool = FileWriteTool::new();
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "content": "hello"}),
                ToolExecutionContext::test("file_write"),
            )
            .await;
        assert!(!result.success);
    }

    #[test]
    fn patch_omitted_dry_run_classifies_as_read_only() {
        let tool = PatchTool::new();
        let classification =
            tool.classify_call(&serde_json::json!({"patch": "--- a/a\n+++ b/a\n"}));
        assert!(classification.read_only);
        assert!(classification.concurrency_safe);
        assert!(!classification.requires_approval);
    }

    #[tokio::test]
    async fn patch_delete_requires_allow_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delete.txt");
        fs::write(&path, "old\n").unwrap();
        let tool = PatchTool::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "base_path": dir.path().to_string_lossy(),
                    "dry_run": true,
                    "patch": "--- a/delete.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-old\n"
                }),
                ToolExecutionContext::test("patch"),
            )
            .await;
        assert!(!result.success);
        assert!(result.output.contains("allow_delete is false"));
    }

    #[test]
    fn parses_simple_patch() {
        let patch = "--- a/test.txt\n+++ b/test.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n";
        let files = parse_unified_diff(patch).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].changed_lines(), 2);
    }

    #[tokio::test]
    async fn delete_path_dry_run_does_not_remove_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("gone.txt");
        fs::write(&path, "bye\n").unwrap();
        let tool = DeletePathTool::new();
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "dry_run": true}),
                ToolExecutionContext::test("delete_path"),
            )
            .await;
        assert!(result.success);
        assert!(path.exists(), "dry-run must not remove the file");
        let output: Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["deleted"], true);
        assert_eq!(output["dry_run"], true);
    }

    #[tokio::test]
    async fn delete_path_requires_recursive_for_directory() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested");
        fs::create_dir_all(&path).unwrap();
        let tool = DeletePathTool::new();
        let result = tool
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "recursive": false, "dry_run": true}),
                ToolExecutionContext::test("delete_path"),
            )
            .await;
        assert!(!result.success);
        assert!(result.output.contains("recursive must be true"));
    }

    #[tokio::test]
    async fn copy_path_dry_run_does_not_create_destination() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("destination.txt");
        fs::write(&source, "hello").unwrap();
        let tool = CopyPathTool::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": true
                }),
                ToolExecutionContext::test("copy_path"),
            )
            .await;
        assert!(result.success);
        assert!(
            !destination.exists(),
            "dry-run must not create the destination"
        );
        let output: Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(output["copied"], true);
        assert_eq!(output["dry_run"], true);
    }
}
