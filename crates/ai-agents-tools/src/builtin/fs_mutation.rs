use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

use ai_agents_core::{
    PathAccessMode, PathBindingKind, PathPolicyBinding, ResultLimitBinding, ResultLimitKind, Tool,
    ToolCallClassification, ToolExecutionContext, ToolOperationKind, ToolPolicyBindings,
    ToolResult, ToolSafetyMetadata, ToolSideEffectLevel,
};

use crate::generate_schema;
use crate::security::path::PathPolicyResolver;
use crate::types::{
    FileVersionEvidence, FileVersionStore, file_version_evidence, truncate_head_tail,
};

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
    /// Validate and return a summary without applying.
    #[serde(default)]
    dry_run: bool,
    /// Permit creating new files when policy allows it.
    #[serde(default)]
    allow_new_files: Option<bool>,
    /// Permit deleting files. Defaults to false.
    #[serde(default)]
    allow_delete: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct MutationOutput {
    path: Option<String>,
    dry_run: bool,
    mutation_performed: bool,
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
    read_paths: Vec<String>,
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
            read_paths: strings_at(value, "read_paths"),
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
            snapshot.read_paths.extend(strings_at(paths, "allow"));
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
    fn declared_write_footprint(
        &self,
        args: &Value,
        _ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        let input: FileWriteInput = serde_json::from_value(args.clone())?;
        footprint_single(
            self.id(),
            Path::new(&input.path),
            input.dry_run,
            false,
            input.create_parent_dirs,
            max_targets,
        )
        .map(Some)
    }
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
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            if !(input.create_parent_dirs && policy.create_parent_dirs) {
                return ToolResult::error(
                    "parent directory does not exist or create_parent_dirs is not allowed",
                );
            }
            if !input.dry_run
                && let Err(error) = fs::create_dir_all(parent)
            {
                return ToolResult::error(format!("Create parent directory error: {}", error));
            }
        }
        if exists
            && !input.dry_run
            && let Err(reason) = enforce_read_before_write(&self.versions, &path, &policy)
        {
            return ToolResult::error(reason);
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
        let action = match (input.dry_run, exists) {
            (true, true) => "plan to overwrite",
            (true, false) => "plan to create",
            (false, true) => "overwrote",
            (false, false) => "created",
        };
        let diff_summary = format!(
            "{} {} with {} bytes",
            action,
            input.path,
            input.content.len()
        );
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
            mutation_performed: !input.dry_run,
            changed_files: 1,
            changed_lines,
            replacements: 0,
            bytes_written: if input.dry_run {
                0
            } else {
                input.content.len()
            },
            created: !input.dry_run && !exists,
            overwritten: !input.dry_run && exists,
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
    fn declared_write_footprint(
        &self,
        args: &Value,
        _ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        let input: FileEditInput = serde_json::from_value(args.clone())?;
        footprint_single(
            self.id(),
            Path::new(&input.path),
            input.dry_run,
            false,
            false,
            max_targets,
        )
        .map(Some)
    }
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
        let show_file_context = edit_context_is_readable(&path, &policy);
        let mut matches = original.match_indices(&input.old_text);
        let first_match = matches.next();
        if first_match.is_none() {
            let output = MutationOutput {
                path: Some(input.path),
                dry_run: input.dry_run,
                mutation_performed: false,
                changed_files: 0,
                changed_lines: 0,
                replacements: 0,
                bytes_written: 0,
                created: false,
                overwritten: false,
                truncated: false,
                approval_required: false,
                diff_summary: "old_text was not found; re-read the file and use exact text"
                    .to_string(),
                changed_paths: Vec::new(),
                version: None,
                near_matches: if show_file_context {
                    near_matches(&original, &input.old_text)
                } else {
                    Vec::new()
                },
            };
            return file_edit_result(output, false, ctx.limits.max_output_chars);
        }
        if !input.replace_all
            && let Some(second) = matches.next()
        {
            let locations = if show_file_context {
                let offsets = [
                    first_match.map(|(offset, _)| offset),
                    Some(second.0),
                    matches.next().map(|(offset, _)| offset),
                ];
                let lines: Vec<_> = offsets
                    .into_iter()
                    .flatten()
                    .map(|offset| {
                        original[..offset]
                            .bytes()
                            .filter(|byte| *byte == b'\n')
                            .count()
                            + 1
                    })
                    .collect();
                format!(" at lines {lines:?}")
            } else {
                String::new()
            };
            return ToolResult::error(format!(
                "old_text appears multiple times{locations}; re-read the file or set replace_all to true to replace all occurrences"
            ));
        }
        let max_replacements = input
            .max_replacements
            .unwrap_or(DEFAULT_MAX_REPLACEMENTS)
            .min(
                ctx.limits
                    .max_replacements
                    .unwrap_or(DEFAULT_MAX_REPLACEMENTS),
            );
        // Stop after one more than the permitted count instead of retaining every match in a large file.
        let replacements = if input.replace_all {
            1 + matches.take(max_replacements).count()
        } else {
            1
        };
        if replacements > max_replacements {
            return ToolResult::error("replacement count exceeds configured max_replacements");
        }
        if !input.dry_run
            && let Err(reason) = enforce_read_before_write(&self.versions, &path, &policy)
        {
            return ToolResult::error(reason);
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
        // A completed write needs room for an explicit effect receipt; never mutate when the configured output cannot carry it.
        let minimum_receipt = serde_json::json!({"mutation_performed": true, "truncated": true})
            .to_string()
            .chars()
            .count();
        if !input.dry_run
            && ctx
                .limits
                .max_output_chars
                .is_some_and(|limit| limit < minimum_receipt)
        {
            return ToolResult::error("edit not applied: output limit cannot report the effect");
        }
        let (diff_summary, diff_truncated) = if show_file_context {
            preview_diff(
                &original,
                &edited,
                &input.old_text,
                &input.new_text,
                replacements,
                DEFAULT_MAX_OUTPUT_CHARS,
            )
        } else {
            preview_requested_edit(&input.old_text, &input.new_text, replacements)
        };
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
        file_edit_result(
            MutationOutput {
                path: Some(input.path.clone()),
                dry_run: input.dry_run,
                mutation_performed: !input.dry_run,
                changed_files: 1,
                changed_lines,
                replacements,
                bytes_written: if input.dry_run { 0 } else { edited.len() },
                created: false,
                overwritten: !input.dry_run,
                truncated: diff_truncated,
                approval_required: !input.dry_run && policy.approval_required(),
                diff_summary,
                changed_paths: vec![input.path],
                version,
                near_matches: Vec::new(),
            },
            true,
            ctx.limits.max_output_chars,
        )
    }
}

#[async_trait]
impl Tool for PatchTool {
    fn declared_write_footprint(
        &self,
        args: &Value,
        ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        let input: PatchInput = serde_json::from_value(args.clone())?;
        if input.dry_run {
            return Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint::empty(
                "builtin.patch.v1",
            )));
        }
        let files = parse_unified_diff(&input.patch).map_err(ai_agents_core::AgentError::Tool)?;
        let base = PathBuf::from(input.base_path.unwrap_or_else(|| ".".into()));
        let mut targets = std::collections::BTreeSet::new();
        for file in files {
            let path = base.join(strip_patch_prefix(file.target_path()));
            add_footprint_entry(&path, &mut targets, max_targets)?;
            if MutationPolicySnapshot::from_context(&ctx.policy_snapshot).create_parent_dirs {
                add_footprint_parents(&path, &mut targets, max_targets)?;
            }
        }
        Ok(Some(ai_agents_core::autonomy::ToolWriteFootprint {
            binding_identity: "builtin.patch.v1".into(),
            targets: targets.into_iter().collect(),
        }))
    }
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
        let mut target_paths = HashSet::new();
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
            if !target_paths.insert(normalize_path(&path)) {
                return ToolResult::error("patch contains duplicate target paths");
            }
            let exists = path.exists();
            if exists {
                match fs::symlink_metadata(&path) {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        return ToolResult::error("patch targets must not be symbolic links");
                    }
                    Ok(_) => {}
                    Err(error) => {
                        return ToolResult::error(format!("Patch metadata error: {}", error));
                    }
                }
            }
            if file.is_delete() && !exists {
                return ToolResult::error("patch deletes a file that does not exist");
            }
            if file.is_new_file() {
                if !input.allow_new_files.unwrap_or(false) {
                    return ToolResult::error("patch creates a file but allow_new_files is false");
                }
                if exists {
                    return ToolResult::error("new-file patch target already exists");
                }
            } else if !file.is_delete() && !exists {
                return ToolResult::error("patch updates a file that does not exist");
            }
            if let Some(parent) = path.parent() {
                if parent.exists() && !parent.is_dir() {
                    return ToolResult::error("patch target parent is not a directory");
                }
                if !parent.exists() && !policy.create_parent_dirs {
                    return ToolResult::error(
                        "patch parent directory creation is not allowed by policy",
                    );
                }
            }
            if exists
                && !input.dry_run
                && let Err(reason) = enforce_read_before_write(&self.versions, &path, &policy)
            {
                return ToolResult::error(reason);
            }
            let (original, expected) = if exists {
                let snapshot = match read_file_snapshot(&path) {
                    Ok(snapshot) => snapshot,
                    Err(error) => return ToolResult::error(format!("Read error: {}", error)),
                };
                if !input.dry_run
                    && let Err(error) = validate_replacement_permissions(&snapshot.permissions)
                {
                    return ToolResult::error(format!("Patch preflight error: {}", error));
                }
                let content = match String::from_utf8(snapshot.bytes.clone()) {
                    Ok(content) => content,
                    Err(error) => {
                        return ToolResult::error(format!("Read error: {}", error));
                    }
                };
                (content, ExpectedPathState::Present(snapshot))
            } else {
                (String::new(), ExpectedPathState::Absent)
            };
            let edited = match apply_file_patch(&original, file) {
                Ok(edited) => edited,
                Err(error) => return ToolResult::error(error),
            };
            changed_paths.push(path.to_string_lossy().to_string());
            if file.is_delete() {
                let ExpectedPathState::Present(expected) = expected else {
                    return ToolResult::error("patch delete target disappeared during preflight");
                };
                outputs.push(PatchApply::Delete { path, expected });
            } else {
                outputs.push(PatchApply::Write {
                    path,
                    content: edited,
                    expected,
                });
            }
        }
        if !input.dry_run {
            if let Err(error) = apply_patch_transaction(&outputs) {
                return ToolResult::error(error);
            }
            for output in &outputs {
                if let Some((path, content)) = output.written_file()
                    && let Ok(version) = file_version_evidence(path, content.as_bytes())
                {
                    self.versions.record(version);
                }
            }
        }
        let diff_summary = summarize_patch(&files, DEFAULT_MAX_OUTPUT_CHARS);
        json_result(&MutationOutput {
            path: Some(base_path.to_string_lossy().to_string()),
            dry_run: input.dry_run,
            mutation_performed: !input.dry_run && !files.is_empty(),
            changed_files: files.len(),
            changed_lines,
            replacements: 0,
            bytes_written: if input.dry_run {
                0
            } else {
                outputs.iter().map(PatchApply::bytes_written).sum()
            },
            created: !input.dry_run && files.iter().any(PatchFile::is_new_file),
            overwritten: !input.dry_run
                && files
                    .iter()
                    .any(|file| !file.is_new_file() && !file.is_delete()),
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
struct FileSnapshot {
    bytes: Vec<u8>,
    permissions: fs::Permissions,
}

#[derive(Debug, Clone)]
enum ExpectedPathState {
    Absent,
    Present(FileSnapshot),
}

#[derive(Debug, Clone)]
enum PatchApply {
    Write {
        path: PathBuf,
        content: String,
        expected: ExpectedPathState,
    },
    Delete {
        path: PathBuf,
        expected: FileSnapshot,
    },
}

impl PatchApply {
    fn bytes_written(&self) -> usize {
        match self {
            Self::Write { content, .. } => content.len(),
            Self::Delete { .. } => 0,
        }
    }

    fn written_file(&self) -> Option<(&Path, &str)> {
        match self {
            Self::Write { path, content, .. } => Some((path, content)),
            Self::Delete { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
enum PatchRollback {
    RestoreWrite {
        path: PathBuf,
        original: FileSnapshot,
        written: Vec<u8>,
        written_permissions: fs::Permissions,
    },
    RestoreDelete {
        path: PathBuf,
        original: FileSnapshot,
    },
    RemoveCreated {
        path: PathBuf,
        written: Vec<u8>,
        written_metadata: fs::Metadata,
    },
}

fn apply_patch_transaction(outputs: &[PatchApply]) -> Result<(), String> {
    let mut rollbacks = Vec::new();
    let mut created_directories = Vec::new();
    for output in outputs {
        match output {
            PatchApply::Delete { path, expected } => {
                if let Err(error) = verify_present_snapshot(path, expected, "pre-apply") {
                    return fail_patch_transaction(error, &rollbacks, &created_directories);
                }
                if let Err(error) = fs::remove_file(path) {
                    return fail_patch_transaction(
                        format!("Patch delete error: {}", error),
                        &rollbacks,
                        &created_directories,
                    );
                }
                rollbacks.push(PatchRollback::RestoreDelete {
                    path: path.clone(),
                    original: expected.clone(),
                });
            }
            PatchApply::Write {
                path,
                content,
                expected,
            } => {
                if let Some(parent) = path.parent()
                    && !parent.exists()
                {
                    let missing = missing_directories(parent);
                    if let Err(error) = fs::create_dir_all(parent) {
                        let cleanup_error = cleanup_created_directories(&missing).err();
                        let error = match cleanup_error {
                            Some(cleanup_error) => format!(
                                "Create parent directory error: {}; partial directory creation may remain: {}",
                                error, cleanup_error
                            ),
                            None => format!("Create parent directory error: {}", error),
                        };
                        return fail_patch_transaction(error, &rollbacks, &created_directories);
                    }
                    created_directories.extend(missing);
                }
                if let Err(error) = verify_expected_state(path, expected, "pre-apply") {
                    return fail_patch_transaction(error, &rollbacks, &created_directories);
                }
                let written_metadata = match atomic_write(path, content.as_bytes()) {
                    Ok(metadata) => metadata,
                    Err(error) => {
                        return fail_patch_transaction(
                            format!("Patch write error: {}", error),
                            &rollbacks,
                            &created_directories,
                        );
                    }
                };
                let written = content.as_bytes().to_vec();
                match expected {
                    ExpectedPathState::Present(original) => {
                        let rollback = PatchRollback::RestoreWrite {
                            path: path.clone(),
                            original: original.clone(),
                            written,
                            written_permissions: original.permissions.clone(),
                        };
                        if let Err(error) = fs::set_permissions(path, original.permissions.clone())
                        {
                            rollbacks.push(rollback);
                            return fail_patch_transaction(
                                format!("Patch permission restore error: {}", error),
                                &rollbacks,
                                &created_directories,
                            );
                        }
                        rollbacks.push(rollback);
                    }
                    ExpectedPathState::Absent => {
                        rollbacks.push(PatchRollback::RemoveCreated {
                            path: path.clone(),
                            written,
                            written_metadata,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

fn fail_patch_transaction(
    error: String,
    rollbacks: &[PatchRollback],
    created_directories: &[PathBuf],
) -> Result<(), String> {
    match rollback_patch(rollbacks, created_directories) {
        Ok(()) if rollbacks.is_empty() && created_directories.is_empty() => Err(format!(
            "{}; no target changes required rollback; any reported temporary artifacts may remain",
            error
        )),
        Ok(()) => Err(format!(
            "{}; patch rollback restored file contents and permissions where applicable; other metadata may have changed",
            error
        )),
        Err(rollback_error) => Err(format!(
            "{}; partial patch application may remain because rollback failed: {}",
            error, rollback_error
        )),
    }
}

fn rollback_patch(
    rollbacks: &[PatchRollback],
    created_directories: &[PathBuf],
) -> Result<(), String> {
    let mut errors = Vec::new();
    for rollback in rollbacks.iter().rev() {
        let result = match rollback {
            PatchRollback::RestoreWrite {
                path,
                original,
                written,
                written_permissions,
            } => verify_present_content_and_permissions(
                path,
                written,
                written_permissions,
                "rollback",
            )
            .and_then(|_| restore_file_snapshot(path, original)),
            PatchRollback::RestoreDelete { path, original } => {
                verify_absent(path, "rollback").and_then(|_| restore_file_snapshot(path, original))
            }
            PatchRollback::RemoveCreated {
                path,
                written,
                written_metadata,
            } => verify_present_content_and_permissions(
                path,
                written,
                &written_metadata.permissions(),
                "rollback",
            )
            .and_then(|_| verify_created_identity(path, written_metadata))
            .and_then(|_| {
                fs::remove_file(path).map_err(|error| {
                    format!(
                        "Patch rollback remove error for {}: {}",
                        path.display(),
                        error
                    )
                })
            }),
        };
        if let Err(error) = result {
            errors.push(error);
        }
    }
    if let Err(error) = cleanup_created_directories(created_directories) {
        errors.push(error);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

// Check created-file identity where available before rollback removes the path.
fn verify_created_identity(path: &Path, expected: &fs::Metadata) -> Result<(), String> {
    let current = fs::symlink_metadata(path).map_err(|error| {
        format!(
            "Patch rollback conflict for {}: created file metadata is unavailable: {}",
            path.display(),
            error
        )
    })?;
    let same = current.file_type().is_file();
    #[cfg(unix)]
    let same = {
        use std::os::unix::fs::MetadataExt;
        same && current.dev() == expected.dev() && current.ino() == expected.ino()
    };
    #[cfg(not(unix))]
    let _ = expected;
    if same {
        Ok(())
    } else {
        Err(format!(
            "Patch rollback conflict for {}: created file was replaced",
            path.display()
        ))
    }
}

fn read_file_snapshot(path: &Path) -> std::io::Result<FileSnapshot> {
    let mut file = fs::File::open(path)?;
    let permissions = file.metadata()?.permissions();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(FileSnapshot { bytes, permissions })
}

fn verify_expected_state(
    path: &Path,
    expected: &ExpectedPathState,
    stage: &str,
) -> Result<(), String> {
    match expected {
        ExpectedPathState::Absent => verify_absent(path, stage),
        ExpectedPathState::Present(expected) => verify_present_snapshot(path, expected, stage),
    }
}

fn verify_absent(path: &Path, stage: &str) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err(format!(
            "Patch {} conflict for {}: expected path to be absent",
            stage,
            path.display()
        )),
        Err(error) => Err(format!(
            "Patch {} conflict for {}: absence could not be verified: {}",
            stage,
            path.display(),
            error
        )),
    }
}

fn verify_present_snapshot(
    path: &Path,
    expected: &FileSnapshot,
    stage: &str,
) -> Result<(), String> {
    let current = read_file_snapshot(path).map_err(|error| {
        format!(
            "Patch {} conflict for {}: expected file could not be read: {}",
            stage,
            path.display(),
            error
        )
    })?;
    if current.bytes != expected.bytes
        || !permissions_match(&current.permissions, &expected.permissions)
    {
        return Err(format!(
            "Patch {} conflict for {}: file changed since the expected state",
            stage,
            path.display()
        ));
    }
    Ok(())
}

fn verify_present_content(path: &Path, expected: &[u8], stage: &str) -> Result<(), String> {
    let current = fs::read(path).map_err(|error| {
        format!(
            "Patch {} conflict for {}: expected file could not be read: {}",
            stage,
            path.display(),
            error
        )
    })?;
    if current != expected {
        return Err(format!(
            "Patch {} conflict for {}: file content changed since the transaction write",
            stage,
            path.display()
        ));
    }
    Ok(())
}

fn verify_present_content_and_permissions(
    path: &Path,
    expected: &[u8],
    expected_permissions: &fs::Permissions,
    stage: &str,
) -> Result<(), String> {
    verify_present_content(path, expected, stage)?;
    let current = fs::metadata(path).map_err(|error| {
        format!(
            "Patch {} conflict for {}: expected file metadata could not be read: {}",
            stage,
            path.display(),
            error
        )
    })?;
    if !permissions_match(&current.permissions(), expected_permissions) {
        return Err(format!(
            "Patch {} conflict for {}: file permissions changed since the transaction write",
            stage,
            path.display()
        ));
    }
    Ok(())
}

fn restore_file_snapshot(path: &Path, snapshot: &FileSnapshot) -> Result<(), String> {
    atomic_write_with_permissions(path, &snapshot.bytes, Some(&snapshot.permissions))
        .map(|_| ())
        .map_err(|error| {
            format!(
                "Patch rollback restore error for {}: {}",
                path.display(),
                error
            )
        })
}

fn permissions_match(left: &fs::Permissions, right: &fs::Permissions) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        left.mode() == right.mode()
    }
    #[cfg(not(unix))]
    {
        left.readonly() == right.readonly()
    }
}

fn missing_directories(path: &Path) -> Vec<PathBuf> {
    let mut missing = Vec::new();
    let mut current = path;
    while !current.exists() {
        missing.push(current.to_path_buf());
        let Some(parent) = current.parent() else {
            break;
        };
        current = parent;
    }
    missing.reverse();
    missing
}

fn cleanup_created_directories(paths: &[PathBuf]) -> Result<(), String> {
    let mut errors = Vec::new();
    for path in paths.iter().rev() {
        if let Err(error) = fs::remove_dir(path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            errors.push(format!(
                "Patch rollback directory cleanup conflict for {}: {}",
                path.display(),
                error
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
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
        supports_cancellation: false,
        default_requires_approval: true,
        should_defer_schema: false,
        max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
    }
}

fn mutation_classification(metadata: &ToolSafetyMetadata, args: &Value) -> ToolCallClassification {
    let dry_run = args
        .get("dry_run")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut classification = ToolCallClassification::from_metadata(metadata);
    classification.safely_retryable = dry_run;
    if dry_run {
        classification.read_only = true;
        classification.concurrency_safe = true;
        classification.destructive = false;
        classification.side_effect_level = ToolSideEffectLevel::None;
        classification.requires_approval = false;
    }
    classification
}

// Mutation tools repeat only the final local path guard; the shared executor remains authoritative for complete policy and approval evaluation.
fn ensure_write_allowed(
    path: &Path,
    dry_run: bool,
    policy: &MutationPolicySnapshot,
) -> Result<(), String> {
    let resolver = mutation_path_resolver()?;
    resolver
        .resolve_path(path)
        .map_err(|error| format!("Path policy resolution failed: {}", error))?;
    for blocked in &policy.blocked_paths {
        if resolver
            .matches_restriction(path, Path::new(blocked))
            .map_err(|error| format!("Path policy resolution failed: {}", error))?
        {
            return Err("path is blocked by policy".to_string());
        }
    }
    if !policy.has_write_policy() {
        if policy.no_write_policy == "deny" || !dry_run {
            return Err("actual mutation requires explicit write_paths policy".to_string());
        }
        return Ok(());
    }
    let mut matches_allowed = false;
    for allowed in policy.write_paths.iter().chain(policy.allowed_paths.iter()) {
        if resolver
            .is_allowed(path, Path::new(allowed))
            .map_err(|error| format!("Path policy resolution failed: {}", error))?
        {
            matches_allowed = true;
            break;
        }
    }
    if !matches_allowed {
        return Err("path is not under an allowed write root".to_string());
    }
    Ok(())
}

// File-edit context is model-visible data; a write-only grant must not become authority to read neighboring file content.
fn edit_context_is_readable(path: &Path, policy: &MutationPolicySnapshot) -> bool {
    let Ok(resolver) = mutation_path_resolver() else {
        return false;
    };
    for root in policy.read_paths.iter().chain(policy.allowed_paths.iter()) {
        match resolver.is_allowed(path, Path::new(root)) {
            Ok(true) => return true,
            Ok(false) => {}
            Err(_) => return false,
        }
    }
    false
}

fn ensure_not_write_root(path: &Path, policy: &MutationPolicySnapshot) -> Result<(), String> {
    let resolver = mutation_path_resolver()?;
    for root in policy.write_paths.iter().chain(policy.allowed_paths.iter()) {
        if resolver
            .is_same_location(path, Path::new(root))
            .map_err(|error| format!("Path policy resolution failed: {}", error))?
        {
            return Err("refusing to delete a configured write root".to_string());
        }
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

// Special bits cannot be restored safely when replacement may change the owning user or group.
fn validate_replacement_permissions(permissions: &fs::Permissions) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if permissions.mode() & 0o7000 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "atomic replacement of files with special permission bits is unsupported",
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = permissions;
    Ok(())
}

// Existing-file replacements start private even when the process umask permits broader access.
fn create_atomic_temp(parent: &Path, restricted: bool) -> std::io::Result<(fs::File, PathBuf)> {
    let temp = parent.join(format!(".{}.tmp", Uuid::new_v4()));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if restricted { 0o600 } else { 0o666 });
    }
    #[cfg(not(unix))]
    let _ = restricted;
    Ok((options.open(&temp)?, temp))
}

// Preserve ordinary Unix mode bits without exposing an initially broader temporary file.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<fs::Metadata> {
    #[cfg(unix)]
    let original = match fs::metadata(path) {
        Ok(metadata) => Some(metadata.permissions()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    #[cfg(not(unix))]
    let original: Option<fs::Permissions> = None;
    atomic_write_with_permissions(path, bytes, original.as_ref())
}

// A deleted-file rollback supplies its saved permissions before any original bytes are written.
fn atomic_write_with_permissions(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&fs::Permissions>,
) -> std::io::Result<fs::Metadata> {
    if let Some(permissions) = permissions {
        validate_replacement_permissions(permissions)?;
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let (mut file, temp) = create_atomic_temp(parent, permissions.is_some())?;
    let result = (|| {
        if let Some(permissions) = permissions {
            file.set_permissions(permissions.clone())?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        let metadata = file.metadata()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(metadata)
    })();
    match result {
        Ok(metadata) => Ok(metadata),
        Err(error) => Err(cleanup_failed_atomic_temp(&temp, error)),
    }
}

// Keep failed cleanup visible because an uncommitted temporary file can still contain data.
fn cleanup_failed_atomic_temp(temp: &Path, error: std::io::Error) -> std::io::Error {
    match fs::remove_file(temp) {
        Ok(()) => error,
        Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => error,
        Err(cleanup) => std::io::Error::new(
            error.kind(),
            format!(
                "{}; failed to remove temporary file {}: {}; partial data may remain",
                error,
                temp.display(),
                cleanup
            ),
        ),
    }
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

// A write grant permits reporting requested text, not disclosing unrelated file content or line locations.
fn preview_requested_edit(old_text: &str, new_text: &str, replacements: usize) -> (String, bool) {
    let mut output = String::new();
    let mut used = 0;
    let header = format!(
        "{replacements} exact replacement(s); file context requires an explicit read path.\nRequested old text: "
    );
    let complete = [
        header.as_str(),
        old_text,
        "\nRequested new text: ",
        new_text,
    ]
    .into_iter()
    .all(|part| append_bounded(&mut output, &mut used, DEFAULT_MAX_OUTPUT_CHARS, part));
    finish_diff_preview(output, !complete, DEFAULT_MAX_OUTPUT_CHARS)
}

// Show the actual changed lines within exact replacement windows; display must not change admission's line accounting.
fn preview_diff(
    before: &str,
    after: &str,
    old_text: &str,
    new_text: &str,
    replacements: usize,
    max_chars: usize,
) -> (String, bool) {
    const MAX_DIFF_LINES: usize = 1_000;
    const MAX_DIFF_CELLS: usize = 250_000;
    let mut windows: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut old_cursor = 0;
    let mut new_cursor = 0;
    for (old_start, _) in before.match_indices(old_text).take(replacements) {
        let new_start = new_cursor + (old_start - old_cursor);
        let old_end = old_start + old_text.len();
        let new_end = new_start + new_text.len();
        let old_boundary = old_end > 0 && before.as_bytes()[old_end - 1] == b'\n';
        let new_boundary = new_end > 0 && after.as_bytes()[new_end - 1] == b'\n';
        // Joining or splitting lines changes the following line even if the replacement ends at a newline.
        let old_window_end = if old_boundary && !new_boundary {
            next_line_end(before, old_end)
        } else {
            line_end(before, old_end)
        };
        let new_window_end = if new_boundary && !old_boundary {
            next_line_end(after, new_end)
        } else {
            line_end(after, new_end)
        };
        let window = (
            line_start(before, old_start),
            old_window_end,
            line_start(after, new_start),
            new_window_end,
        );
        if let Some(last) = windows.last_mut()
            && window.0 <= last.1
            && window.2 <= last.3
        {
            last.1 = last.1.max(window.1);
            last.3 = last.3.max(window.3);
        } else {
            windows.push(window);
        }
        old_cursor = old_end;
        new_cursor = new_end;
    }

    let mut output = String::new();
    let mut used = 0;
    let mut truncated = false;
    for (old_start, old_end, new_start, new_end) in windows {
        let old_lines: Vec<_> = before[old_start..old_end]
            .split_inclusive('\n')
            .take(MAX_DIFF_LINES + 1)
            .collect();
        let new_lines: Vec<_> = after[new_start..new_end]
            .split_inclusive('\n')
            .take(MAX_DIFF_LINES + 1)
            .collect();
        let columns = new_lines.len() + 1;
        let cells = (old_lines.len() + 1).saturating_mul(columns);
        if old_lines.len() > MAX_DIFF_LINES
            || new_lines.len() > MAX_DIFF_LINES
            || cells > MAX_DIFF_CELLS
        {
            append_bounded(
                &mut output,
                &mut used,
                max_chars,
                "Detailed diff omitted because the changed region exceeds preview limits. Re-read the file or use git_diff.\n",
            );
            truncated = true;
            break;
        }
        let mut common = vec![0u32; cells];
        for old_index in (0..old_lines.len()).rev() {
            for new_index in (0..new_lines.len()).rev() {
                let index = old_index * columns + new_index;
                common[index] = if old_lines[old_index] == new_lines[new_index] {
                    common[(old_index + 1) * columns + new_index + 1] + 1
                } else {
                    common[(old_index + 1) * columns + new_index]
                        .max(common[old_index * columns + new_index + 1])
                };
            }
        }
        let old_number = before[..old_start]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count()
            + 1;
        let new_number = after[..new_start]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count()
            + 1;
        let (mut old_index, mut new_index) = (0, 0);
        while old_index < old_lines.len() || new_index < new_lines.len() {
            if old_index < old_lines.len()
                && new_index < new_lines.len()
                && old_lines[old_index] == new_lines[new_index]
            {
                old_index += 1;
                new_index += 1;
                continue;
            }
            let remove = new_index == new_lines.len()
                || (old_index < old_lines.len()
                    && common[(old_index + 1) * columns + new_index]
                        >= common[old_index * columns + new_index + 1]);
            let complete = if remove {
                let complete = append_diff_line(
                    &mut output,
                    &mut used,
                    max_chars,
                    '-',
                    old_number + old_index,
                    old_lines[old_index],
                );
                old_index += 1;
                complete
            } else {
                let complete = append_diff_line(
                    &mut output,
                    &mut used,
                    max_chars,
                    '+',
                    new_number + new_index,
                    new_lines[new_index],
                );
                new_index += 1;
                complete
            };
            if !complete {
                truncated = true;
                break;
            }
        }
        if truncated {
            break;
        }
    }
    finish_diff_preview(output, truncated, max_chars)
}

// A preview limit is explicit evidence of incomplete display, not proof that a write was reverted.
fn finish_diff_preview(mut output: String, truncated: bool, max_chars: usize) -> (String, bool) {
    if truncated {
        const MARKER: &str = "\n...[diff truncated]...\n";
        let marker_len = MARKER.chars().count();
        if marker_len <= max_chars {
            output = output.chars().take(max_chars - marker_len).collect();
            output.push_str(MARKER);
        }
    }
    (output, truncated)
}

fn line_start(text: &str, offset: usize) -> usize {
    text[..offset].rfind('\n').map_or(0, |index| index + 1)
}

fn line_end(text: &str, offset: usize) -> usize {
    if offset > 0 && text.as_bytes()[offset - 1] == b'\n' {
        offset
    } else {
        next_line_end(text, offset)
    }
}

fn next_line_end(text: &str, offset: usize) -> usize {
    text[offset..]
        .find('\n')
        .map_or(text.len(), |index| offset + index + 1)
}

fn append_bounded(output: &mut String, used: &mut usize, limit: usize, text: &str) -> bool {
    for character in text.chars() {
        if *used == limit {
            return false;
        }
        output.push(character);
        *used += 1;
    }
    true
}

fn append_diff_line(
    output: &mut String,
    used: &mut usize,
    limit: usize,
    sign: char,
    line_number: usize,
    line: &str,
) -> bool {
    let (content, ending) = if let Some(content) = line.strip_suffix("\r\n") {
        (content, " [CRLF]\n")
    } else if let Some(content) = line.strip_suffix('\n') {
        (content, "\n")
    } else {
        (line, " [no newline]\n")
    };
    let prefix = format!("{sign}{line_number}:\"");
    if !append_bounded(output, used, limit, &prefix) {
        return false;
    }
    for character in content.chars() {
        let escaped = match character {
            '"' => Some("\\\"".to_string()),
            '\\' => Some("\\\\".to_string()),
            '\t' => Some("\\t".to_string()),
            '\r' => Some("\\r".to_string()),
            value if value.is_control() => Some(format!("\\u{:04x}", value as u32)),
            _ => None,
        };
        let mut buffer = [0u8; 4];
        let text = escaped
            .as_deref()
            .unwrap_or_else(|| character.encode_utf8(&mut buffer));
        if !append_bounded(output, used, limit, text) {
            return false;
        }
    }
    append_bounded(output, used, limit, "\"") && append_bounded(output, used, limit, ending)
}

// Candidate context comes only from the already authorized target and is advisory, never a fuzzy write target.
fn near_matches(content: &str, needle: &str) -> Vec<String> {
    const MAX_SEARCH_LINES: usize = 5_000;
    const MAX_MATCH_LINE_CHARS: usize = 256;
    const MAX_CONTEXT_CHARS: usize = 80;
    let mut anchors = Vec::new();
    for line in needle.lines().take(8) {
        let chars: Vec<_> = line.trim().chars().take(128).collect();
        if chars.len() < 3 {
            continue;
        }
        let width = (chars.len() / 2).clamp(3, 16).min(chars.len());
        anchors.push(chars[..width].iter().collect::<String>());
        anchors.push(chars[chars.len() - width..].iter().collect::<String>());
    }
    let snippet = |line: &str, offset: usize| {
        let prefix_chars = line[..offset].chars().count();
        let skip = prefix_chars.saturating_sub(30);
        let mut characters = line.chars().skip(skip);
        let mut text: String = characters.by_ref().take(MAX_CONTEXT_CHARS).collect();
        if skip > 0 {
            text.insert_str(0, "...");
        }
        if characters.next().is_some() {
            text.push_str("...");
        }
        text
    };
    let mut matches = Vec::new();
    for (index, line) in content.lines().take(MAX_SEARCH_LINES).enumerate() {
        let bounded_line: String = line.chars().take(MAX_MATCH_LINE_CHARS).collect();
        if let Some(offset) = anchors
            .iter()
            .filter_map(|anchor| bounded_line.find(anchor))
            .min()
        {
            matches.push(format!("line {}: {}", index + 1, snippet(line, offset)));
            if matches.len() == 3 {
                break;
            }
        }
    }
    matches
}

fn parse_unified_diff(input: &str) -> Result<Vec<PatchFile>, String> {
    let mut files = Vec::new();
    let mut current: Option<PatchFile> = None;
    let mut current_hunk: Option<PatchHunk> = None;
    for line in input.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            if let Some(hunk) = current_hunk.take()
                && let Some(file) = current.as_mut()
            {
                file.hunks.push(hunk);
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
            if let Some(hunk) = current_hunk.take()
                && let Some(file) = current.as_mut()
            {
                file.hunks.push(hunk);
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
    if let Some(hunk) = current_hunk
        && let Some(file) = current.as_mut()
    {
        file.hunks.push(hunk);
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

fn path_entry_exists(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

fn mutation_path_resolver() -> Result<PathPolicyResolver, String> {
    PathPolicyResolver::new().map_err(|error| format!("Path policy resolution failed: {}", error))
}

fn exceeds(limit: Option<usize>, value: usize) -> bool {
    limit.is_some_and(|limit| value > limit)
}

// Bound the complete serialized edit result so the shared executor can retain its reason and advisory context when the fixed fields fit.
fn file_edit_result(mut output: MutationOutput, success: bool, limit: Option<usize>) -> ToolResult {
    let original_hints = output.near_matches.clone();
    let json = match serialize_bounded_file_edit(&mut output, success, limit) {
        Ok(json) => json,
        Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
    };
    if let Some(limit) = limit {
        let projected = match file_edit_model_output_len(&json, success) {
            Ok(projected) => projected,
            Err(error) => return ToolResult::error(format!("Serialization error: {}", error)),
        };
        if projected > limit {
            if !success {
                match compact_file_edit_failure(&output, &original_hints, limit) {
                    Ok(Some(compact)) => {
                        return ToolResult {
                            success: false,
                            output: compact,
                            metadata: None,
                        };
                    }
                    Ok(None) => {
                        for reason in [
                            "file edit failed; re-read the file (output limit)",
                            "edit failed; re-read",
                            "edit failed",
                        ] {
                            if file_edit_model_output_len(reason, false)
                                .is_ok_and(|length| length <= limit)
                            {
                                return ToolResult::error(reason);
                            }
                        }
                        let marker = "{\"truncated\":true}";
                        if marker.chars().count() <= limit {
                            return ToolResult::error(marker);
                        }
                        return ToolResult::error(
                            "file edit failed; output limit prevents details",
                        );
                    }
                    Err(error) => {
                        return ToolResult::error(format!("Serialization error: {}", error));
                    }
                }
            }
            let minimal = serde_json::json!({
                "dry_run": output.dry_run,
                "mutation_performed": output.mutation_performed,
                "truncated": true
            })
            .to_string();
            if minimal.chars().count() <= limit {
                return ToolResult::ok(minimal);
            }
            let effect_only = serde_json::json!({
                "mutation_performed": output.mutation_performed,
                "truncated": true
            })
            .to_string();
            if effect_only.chars().count() <= limit {
                return ToolResult::ok(effect_only);
            }
        }
    }
    ToolResult {
        success,
        output: json,
        metadata: None,
    }
}

// A partial failure envelope preserves the reason and authorized hints without pretending omitted effect fields are complete.
fn compact_file_edit_failure(
    output: &MutationOutput,
    hints: &[String],
    limit: usize,
) -> Result<Option<String>, serde_json::Error> {
    let mut compact = serde_json::json!({
        "mutation_performed": false,
        "truncated": true,
        "diff_summary": output.diff_summary,
        "near_matches": []
    });
    let mut best = serde_json::to_string(&compact)?;
    if file_edit_model_output_len(&best, false)? > limit {
        return Ok(None);
    }
    let max = hints
        .iter()
        .map(|hint| hint.chars().count())
        .max()
        .unwrap_or(0);
    let mut low = 0;
    let mut high = max.saturating_add(1);
    while low < high {
        let middle = low + (high - low) / 2;
        compact["near_matches"] = Value::Array(
            hints
                .iter()
                .map(|hint| hint.chars().take(middle).collect::<String>())
                .filter(|hint| !hint.is_empty())
                .map(Value::String)
                .collect(),
        );
        let candidate = serde_json::to_string(&compact)?;
        if file_edit_model_output_len(&candidate, false)? <= limit {
            best = candidate;
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(Some(best))
}

// Reserve the existing runtime error wrapper as well as its escaped reason within the model-facing limit.
fn file_edit_model_output_len(output: &str, success: bool) -> Result<usize, serde_json::Error> {
    if success {
        return Ok(output.chars().count());
    }
    let wrapper = serde_json::json!({
        "success": false,
        "error": {
            "kind": "tool_error",
            "reason": output,
            "retryable": false,
            "requires_user_action": false
        }
    });
    Ok("Error: Tool error: ".len() + serde_json::to_string(&wrapper)?.chars().count())
}

// Preserve failure reasons and stable effect fields; only display text and nearby hints may be shortened.
fn serialize_bounded_file_edit(
    output: &mut MutationOutput,
    success: bool,
    limit: Option<usize>,
) -> Result<String, serde_json::Error> {
    let full = serde_json::to_string(output)?;
    let Some(limit) = limit else {
        return Ok(full);
    };
    if file_edit_model_output_len(&full, success)? <= limit {
        return Ok(full);
    }
    let producer_truncated = output.truncated;
    if output.near_matches.is_empty() && (!success || output.diff_summary.is_empty()) {
        return Ok(full);
    }
    output.truncated = true;
    let diff = std::mem::take(&mut output.diff_summary);
    let hints = std::mem::take(&mut output.near_matches);
    if !success {
        output.diff_summary = diff.clone();
    }
    let mut best = serde_json::to_string(output)?;
    if file_edit_model_output_len(&best, success)? > limit {
        return Ok(best);
    }
    let max = hints
        .iter()
        .map(|hint| hint.chars().count())
        .chain(success.then(|| diff.chars().count()))
        .max()
        .unwrap_or(0);
    let mut low = 0;
    let mut high = max.saturating_add(1);
    while low < high {
        let middle = low + (high - low) / 2;
        if success {
            output.diff_summary = truncate_head_tail(&diff, middle).0;
        }
        output.near_matches = hints
            .iter()
            .map(|hint| hint.chars().take(middle).collect::<String>())
            .filter(|hint| !hint.is_empty())
            .collect();
        let candidate = serde_json::to_string(output)?;
        if file_edit_model_output_len(&candidate, success)? <= limit {
            best = candidate;
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if !producer_truncated {
        let original: Value = serde_json::from_str(&full)?;
        let candidate: Value = serde_json::from_str(&best)?;
        if original["diff_summary"] == candidate["diff_summary"]
            && original["near_matches"] == candidate["near_matches"]
        {
            let mut reduced: MutationOutput = serde_json::from_str(&best)?;
            if success && !reduced.diff_summary.is_empty() {
                let length = reduced.diff_summary.chars().count();
                reduced.diff_summary = truncate_head_tail(&reduced.diff_summary, length - 1).0;
            } else if let Some(hint) = reduced.near_matches.last_mut() {
                let length = hint.chars().count();
                *hint = hint.chars().take(length.saturating_sub(1)).collect();
                reduced.near_matches.retain(|hint| !hint.is_empty());
            } else {
                return Ok(full);
            }
            return serde_json::to_string(&reduced);
        }
    }
    Ok(best)
}

fn json_result<T: Serialize>(output: &T) -> ToolResult {
    match serde_json::to_string(output) {
        Ok(json) => ToolResult::ok(json),
        Err(error) => ToolResult::error(format!("Serialization error: {}", error)),
    }
}

fn json_error_result<T: Serialize>(output: &T) -> ToolResult {
    match serde_json::to_string(output) {
        Ok(json) => ToolResult {
            success: false,
            output: json,
            metadata: None,
        },
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
    #[serde(default)]
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
    #[serde(default)]
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
    #[serde(default)]
    dry_run: bool,
}

#[derive(Debug, Serialize)]
struct PathMutationOutput {
    source_path: Option<String>,
    destination_path: Option<String>,
    path: Option<String>,
    dry_run: bool,
    mutation_performed: bool,
    copied: bool,
    moved: bool,
    deleted: bool,
    recursive: bool,
    overwritten: bool,
    bytes_affected: usize,
    items_affected: usize,
    approval_required: bool,
    diff_summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cleanup_warning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retained_backup_path: Option<String>,
}

#[async_trait]
impl Tool for CopyPathTool {
    fn declared_write_footprint(
        &self,
        args: &Value,
        _ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        let input: CopyPathInput = serde_json::from_value(args.clone())?;
        footprint_transfer(
            self.id(),
            Path::new(&input.source_path),
            Path::new(&input.destination_path),
            input.dry_run,
            false,
            input.create_parent_dirs,
            max_targets,
        )
        .map(Some)
    }
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
        path_mutation_classification(&self.safety_metadata(), args)
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
        if !path_entry_exists(&source) {
            return ToolResult::error(format!("source path does not exist: {}", input.source_path));
        }
        if let Err(reason) = validate_copy_destination(&source, &destination) {
            return ToolResult::error(reason);
        }
        let (bytes_affected, items_affected, recursive) = match inspect_copy_source(&source) {
            Ok(details) => details,
            Err(error) => return ToolResult::error(format!("Copy source error: {}", error)),
        };
        let destination_exists = path_entry_exists(&destination);
        if destination_exists && !input.overwrite {
            return ToolResult::error("overwrite must be true to replace an existing destination");
        }
        if destination_exists && !policy.overwrite_existing && !input.dry_run {
            return ToolResult::error("overwrite_existing policy is false for this destination");
        }
        if let Some(parent) = destination.parent()
            && !parent.exists()
        {
            if !(input.create_parent_dirs && policy.create_parent_dirs) {
                return ToolResult::error(
                    "destination parent directory does not exist or create_parent_dirs is not allowed",
                );
            }
            if !input.dry_run
                && let Err(error) = fs::create_dir_all(parent)
            {
                return ToolResult::error(format!("Create parent directory error: {}", error));
            }
        }
        let mut cleanup_warning = None;
        let mut retained_backup_path = None;
        if !input.dry_run {
            if destination_exists {
                match replace_copy_path(&source, &destination) {
                    ReplacementOutcome::Committed {
                        cleanup_warning: warning,
                        retained_backup_path: backup,
                    } => {
                        cleanup_warning = warning;
                        retained_backup_path = backup;
                    }
                    ReplacementOutcome::Unchanged(error) => {
                        return ToolResult::error(format!("Copy error: {}", error));
                    }
                    ReplacementOutcome::RecoveryIncomplete {
                        error,
                        retained_backup_path,
                    } => {
                        return json_error_result(&PathMutationOutput {
                            source_path: Some(input.source_path.clone()),
                            destination_path: Some(input.destination_path.clone()),
                            path: None,
                            dry_run: false,
                            mutation_performed: true,
                            copied: false,
                            moved: false,
                            deleted: false,
                            recursive,
                            overwritten: false,
                            bytes_affected,
                            items_affected,
                            approval_required: policy.approval_required(),
                            diff_summary:
                                "copy failed after moving the previous destination to a backup"
                                    .to_string(),
                            error: Some(error),
                            cleanup_warning: None,
                            retained_backup_path: Some(retained_backup_path),
                        });
                    }
                }
            } else if let Err(error) = copy_path(&source, &destination) {
                return ToolResult::error(format!("Copy error: {}", error));
            }
        }
        json_result(&PathMutationOutput {
            source_path: Some(input.source_path.clone()),
            destination_path: Some(input.destination_path.clone()),
            path: None,
            dry_run: input.dry_run,
            mutation_performed: !input.dry_run,
            copied: !input.dry_run,
            moved: false,
            deleted: false,
            recursive,
            overwritten: !input.dry_run && destination_exists,
            bytes_affected,
            items_affected,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary: format!(
                "{} {} -> {} ({} bytes)",
                if input.dry_run {
                    "plan to copy"
                } else {
                    "copied"
                },
                input.source_path,
                input.destination_path,
                bytes_affected
            ),
            error: None,
            cleanup_warning,
            retained_backup_path,
        })
    }
}

#[async_trait]
impl Tool for MovePathTool {
    fn declared_write_footprint(
        &self,
        args: &Value,
        _ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        let input: MovePathInput = serde_json::from_value(args.clone())?;
        footprint_transfer(
            self.id(),
            Path::new(&input.source_path),
            Path::new(&input.destination_path),
            input.dry_run,
            true,
            input.create_parent_dirs,
            max_targets,
        )
        .map(Some)
    }
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
        path_mutation_classification(&self.safety_metadata(), args)
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
        if !path_entry_exists(&source) {
            return ToolResult::error(format!("source path does not exist: {}", input.source_path));
        }
        let resolved_source = match source.canonicalize() {
            Ok(path) => path,
            Err(error) => return ToolResult::error(format!("Source path error: {}", error)),
        };
        let resolved_destination = match mutation_path_resolver().and_then(|resolver| {
            resolver
                .resolve_path(&destination)
                .map_err(|error| format!("Path policy resolution failed: {}", error))
        }) {
            Ok(path) => path,
            Err(error) => return ToolResult::error(error),
        };
        if resolved_source == resolved_destination {
            return ToolResult::error("source and destination resolve to the same path");
        }
        let destination_exists = path_entry_exists(&destination);
        if destination_exists && !input.overwrite {
            return ToolResult::error("overwrite must be true to replace an existing destination");
        }
        if destination_exists && !policy.overwrite_existing && !input.dry_run {
            return ToolResult::error("overwrite_existing policy is false for this destination");
        }
        if let Some(parent) = destination.parent()
            && !parent.exists()
        {
            if !(input.create_parent_dirs && policy.create_parent_dirs) {
                return ToolResult::error(
                    "destination parent directory does not exist or create_parent_dirs is not allowed",
                );
            }
            if !input.dry_run
                && let Err(error) = fs::create_dir_all(parent)
            {
                return ToolResult::error(format!("Create parent directory error: {}", error));
            }
        }
        let recursive = source.is_dir();
        let bytes_affected = path_size(&source).unwrap_or(0);
        let items_affected = path_item_count(&source).unwrap_or(1);
        let mut cleanup_warning = None;
        let mut retained_backup_path = None;
        if !input.dry_run {
            if destination_exists {
                match replace_moved_path(&source, &destination) {
                    ReplacementOutcome::Committed {
                        cleanup_warning: warning,
                        retained_backup_path: backup,
                    } => {
                        cleanup_warning = warning;
                        retained_backup_path = backup;
                    }
                    ReplacementOutcome::Unchanged(error) => {
                        return ToolResult::error(format!("Move error: {}", error));
                    }
                    ReplacementOutcome::RecoveryIncomplete {
                        error,
                        retained_backup_path,
                    } => {
                        return json_error_result(&PathMutationOutput {
                            source_path: Some(input.source_path.clone()),
                            destination_path: Some(input.destination_path.clone()),
                            path: None,
                            dry_run: false,
                            mutation_performed: true,
                            copied: false,
                            moved: false,
                            deleted: false,
                            recursive,
                            overwritten: false,
                            bytes_affected,
                            items_affected,
                            approval_required: policy.approval_required(),
                            diff_summary:
                                "move failed after moving the previous destination to a backup"
                                    .to_string(),
                            error: Some(error),
                            cleanup_warning: None,
                            retained_backup_path: Some(retained_backup_path),
                        });
                    }
                }
            } else if let Err(error) = fs::rename(&source, &destination) {
                return ToolResult::error(format!("Move error: {}", error));
            }
        }
        json_result(&PathMutationOutput {
            source_path: Some(input.source_path.clone()),
            destination_path: Some(input.destination_path.clone()),
            path: None,
            dry_run: input.dry_run,
            mutation_performed: !input.dry_run,
            copied: false,
            moved: !input.dry_run,
            deleted: false,
            recursive,
            overwritten: !input.dry_run && destination_exists,
            bytes_affected,
            items_affected,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary: format!(
                "{} {} -> {} ({} bytes)",
                if input.dry_run {
                    "plan to move"
                } else {
                    "moved"
                },
                input.source_path,
                input.destination_path,
                bytes_affected
            ),
            error: None,
            cleanup_warning,
            retained_backup_path,
        })
    }
}

#[async_trait]
impl Tool for DeletePathTool {
    fn declared_write_footprint(
        &self,
        args: &Value,
        _ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        let input: DeletePathInput = serde_json::from_value(args.clone())?;
        footprint_single(
            self.id(),
            Path::new(&input.path),
            input.dry_run,
            true,
            false,
            max_targets,
        )
        .map(Some)
    }
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
        path_mutation_classification(&self.safety_metadata(), args)
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
        if let Err(reason) = ensure_not_write_root(&path, &policy) {
            return ToolResult::error(reason);
        }
        if !path_entry_exists(&path) {
            return ToolResult::error(format!("path does not exist: {}", input.path));
        }
        if path.is_dir() && !input.recursive {
            return ToolResult::error("recursive must be true to delete a directory");
        }
        let recursive = path.is_dir();
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
            mutation_performed: !input.dry_run,
            copied: false,
            moved: false,
            deleted: !input.dry_run,
            recursive,
            overwritten: false,
            bytes_affected,
            items_affected,
            approval_required: !input.dry_run && policy.approval_required(),
            diff_summary: format!(
                "{} {} ({} bytes, {} items)",
                if input.dry_run {
                    "plan to delete"
                } else {
                    "deleted"
                },
                input.path,
                bytes_affected,
                items_affected
            ),
            error: None,
            cleanup_warning: None,
            retained_backup_path: None,
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
        supports_cancellation: false,
        default_requires_approval: true,
        should_defer_schema: false,
        max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
        max_result_size_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
    }
}

fn path_mutation_classification(
    metadata: &ToolSafetyMetadata,
    args: &Value,
) -> ToolCallClassification {
    let dry_run = args
        .get("dry_run")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut classification = ToolCallClassification::from_metadata(metadata);
    classification.safely_retryable = dry_run;
    if dry_run {
        classification.read_only = true;
        classification.concurrency_safe = true;
        classification.destructive = false;
        classification.side_effect_level = ToolSideEffectLevel::None;
        classification.requires_approval = false;
    }
    classification
}

fn ensure_read_allowed(path: &Path, policy: &MutationPolicySnapshot) -> Result<(), String> {
    let resolver = mutation_path_resolver()?;
    resolver
        .resolve_path(path)
        .map_err(|error| format!("Path policy resolution failed: {}", error))?;
    for blocked in &policy.blocked_paths {
        if resolver
            .matches_restriction(path, Path::new(blocked))
            .map_err(|error| format!("Path policy resolution failed: {}", error))?
        {
            return Err("source path is blocked by policy".to_string());
        }
    }
    Ok(())
}

fn validate_copy_destination(source: &Path, destination: &Path) -> Result<(), String> {
    let source = source
        .canonicalize()
        .map_err(|error| format!("Source canonicalization failed: {}", error))?;
    let destination = mutation_path_resolver()?
        .resolve_path(destination)
        .map_err(|error| format!("Path policy resolution failed: {}", error))?;
    if destination == source || source.is_dir() && destination.starts_with(&source) {
        return Err("destination must not equal or be nested under the source".to_string());
    }
    Ok(())
}

fn inspect_copy_source(path: &Path) -> std::io::Result<(usize, usize, bool)> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("symbolic links are not supported: {}", path.display()),
        ));
    }
    if !metadata.is_dir() {
        return Ok((metadata.len() as usize, 1, false));
    }

    let mut bytes = 0usize;
    let mut items = 1usize;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let (entry_bytes, entry_items, _) = inspect_copy_source(&entry.path())?;
        bytes = bytes.saturating_add(entry_bytes);
        items = items.saturating_add(entry_items);
    }
    Ok((bytes, items, true))
}

fn copy_path(source: &Path, destination: &Path) -> std::io::Result<()> {
    if source.is_dir() {
        copy_directory(source, destination)
    } else {
        if let Some(parent) = destination.parent()
            && !parent.exists()
        {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source, destination)?;
        Ok(())
    }
}

enum ReplacementOutcome {
    Committed {
        cleanup_warning: Option<String>,
        retained_backup_path: Option<String>,
    },
    Unchanged(String),
    RecoveryIncomplete {
        error: String,
        retained_backup_path: String,
    },
}

fn replace_copy_path(source: &Path, destination: &Path) -> ReplacementOutcome {
    let staged = unique_sibling_path(destination, "copy");
    if let Err(error) = copy_path(source, &staged) {
        if path_entry_exists(&staged) {
            let _ = remove_path(&staged);
        }
        return ReplacementOutcome::Unchanged(error.to_string());
    }
    let outcome = replace_existing_path(&staged, destination);
    if matches!(
        outcome,
        ReplacementOutcome::Unchanged(_) | ReplacementOutcome::RecoveryIncomplete { .. }
    ) && path_entry_exists(&staged)
    {
        let _ = remove_path(&staged);
    }
    outcome
}

fn replace_moved_path(source: &Path, destination: &Path) -> ReplacementOutcome {
    replace_existing_path(source, destination)
}

fn replace_existing_path(prepared: &Path, destination: &Path) -> ReplacementOutcome {
    replace_existing_path_with(prepared, destination, remove_path)
}

fn replace_existing_path_with<F>(
    prepared: &Path,
    destination: &Path,
    remove_backup: F,
) -> ReplacementOutcome
where
    F: FnOnce(&Path) -> std::io::Result<()>,
{
    let backup = unique_sibling_path(destination, "backup");
    if let Err(error) = fs::rename(destination, &backup) {
        return ReplacementOutcome::Unchanged(error.to_string());
    }
    if let Err(error) = fs::rename(prepared, destination) {
        return match fs::rename(&backup, destination) {
            Ok(()) => ReplacementOutcome::Unchanged(error.to_string()),
            Err(restore_error) => ReplacementOutcome::RecoveryIncomplete {
                error: format!(
                    "replacement failed: {}; destination restore failed: {}",
                    error, restore_error
                ),
                retained_backup_path: backup.to_string_lossy().into_owned(),
            },
        };
    }

    //
    // The prepared-to-destination rename is the commit point. Cleanup must never hide a committed replacement.
    //
    match remove_backup(&backup) {
        Ok(()) => ReplacementOutcome::Committed {
            cleanup_warning: None,
            retained_backup_path: None,
        },
        Err(error) => ReplacementOutcome::Committed {
            cleanup_warning: Some(format!(
                "replacement committed but backup cleanup failed: {}",
                error
            )),
            retained_backup_path: Some(backup.to_string_lossy().into_owned()),
        },
    }
}

fn unique_sibling_path(path: &Path, label: &str) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "path".into());
    parent.join(format!(".{}.{}.{}", name, label, Uuid::new_v4()))
}

fn remove_path(path: &Path) -> std::io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

fn copy_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let from = entry.path();
        let metadata = fs::symlink_metadata(&from)?;
        if metadata.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("symbolic links are not supported: {}", from.display()),
            ));
        }
        let to = destination.join(entry.file_name());
        if metadata.is_dir() {
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

/// Entry mutations resolve the parent but never follow the leaf; two links to one referent remain distinct targets.
pub(super) fn add_footprint_entry(
    path: &Path,
    targets: &mut std::collections::BTreeSet<String>,
    limit: usize,
) -> ai_agents_core::Result<()> {
    let resolved = footprint_entry_path(path)?;
    add_footprint_identity(&resolved, targets, limit)
}

// Parent resolution is shared with policy, while the leaf remains an operation-owned entry.
fn footprint_entry_path(path: &Path) -> ai_agents_core::Result<PathBuf> {
    validate_safe_target(path).map_err(ai_agents_core::AgentError::Tool)?;
    let name = path.file_name().ok_or_else(|| {
        ai_agents_core::AgentError::Tool("footprint cannot mutate a filesystem root".into())
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let resolver = mutation_path_resolver().map_err(ai_agents_core::AgentError::Tool)?;
    let resolved = resolver
        .resolve_path(parent)
        .map_err(|error| ai_agents_core::AgentError::Tool(error.to_string()))?
        .join(name);
    Ok(resolved)
}

// Projected replacement descendants must not resolve through the old tree's symlinks.
fn add_footprint_identity(
    resolved: &Path,
    targets: &mut std::collections::BTreeSet<String>,
    limit: usize,
) -> ai_agents_core::Result<()> {
    targets.insert(serde_json::to_string(resolved)?);
    if targets.len() > limit {
        return Err(ai_agents_core::AgentError::Tool(
            "declared footprint exceeds target bound".into(),
        ));
    }
    Ok(())
}

/// Missing parent creations are logical writes even when the leaf operation later fails.
pub(super) fn add_footprint_parents(
    path: &Path,
    targets: &mut std::collections::BTreeSet<String>,
    limit: usize,
) -> ai_agents_core::Result<()> {
    let mut parent = path.parent();
    while let Some(path) = parent {
        if path.as_os_str().is_empty() {
            break;
        }
        match fs::symlink_metadata(path) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                add_footprint_entry(path, targets, limit)?
            }
            Err(error) => return Err(ai_agents_core::AgentError::Tool(error.to_string())),
        }
        parent = path.parent();
    }
    Ok(())
}

/// Bounded traversal does not follow symbolic links and includes directory entries and overwritten descendants.
fn add_footprint_tree(
    path: &Path,
    mapped: &Path,
    targets: &mut std::collections::BTreeSet<String>,
    limit: usize,
) -> ai_agents_core::Result<()> {
    let mut stack = vec![(path.to_path_buf(), footprint_entry_path(mapped)?)];
    while let Some((path, mapped)) = stack.pop() {
        add_footprint_identity(&mapped, targets, limit)?;
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(ai_agents_core::AgentError::Tool(error.to_string())),
        };
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            for child in fs::read_dir(path)
                .map_err(|error| ai_agents_core::AgentError::Tool(error.to_string()))?
            {
                let child =
                    child.map_err(|error| ai_agents_core::AgentError::Tool(error.to_string()))?;
                // Bound queued enumeration as well as the final union so one directory cannot hide unbounded allocation.
                if stack.len() >= limit {
                    return Err(ai_agents_core::AgentError::Tool(
                        "recursive footprint exceeds target bound".into(),
                    ));
                }
                stack.push((child.path(), mapped.join(child.file_name())));
            }
        }
    }
    Ok(())
}

pub(super) fn footprint_single(
    id: &str,
    path: &Path,
    preview: bool,
    recursive: bool,
    parents: bool,
    limit: usize,
) -> ai_agents_core::Result<ai_agents_core::autonomy::ToolWriteFootprint> {
    let binding_identity = format!("builtin.{id}.v1");
    if preview {
        return Ok(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            binding_identity,
        ));
    }
    let mut targets = std::collections::BTreeSet::new();
    if recursive {
        add_footprint_tree(path, path, &mut targets, limit)?;
    } else {
        add_footprint_entry(path, &mut targets, limit)?;
    }
    if parents {
        add_footprint_parents(path, &mut targets, limit)?;
    }
    Ok(ai_agents_core::autonomy::ToolWriteFootprint {
        binding_identity,
        targets: targets.into_iter().collect(),
    })
}

fn footprint_transfer(
    id: &str,
    source: &Path,
    destination: &Path,
    preview: bool,
    moving: bool,
    parents: bool,
    limit: usize,
) -> ai_agents_core::Result<ai_agents_core::autonomy::ToolWriteFootprint> {
    let binding_identity = format!("builtin.{id}.v1");
    if preview {
        return Ok(ai_agents_core::autonomy::ToolWriteFootprint::empty(
            binding_identity,
        ));
    }
    let mut targets = std::collections::BTreeSet::new();
    add_footprint_tree(destination, destination, &mut targets, limit)?;
    add_footprint_tree(source, destination, &mut targets, limit)?;
    if moving {
        add_footprint_tree(source, source, &mut targets, limit)?;
    }
    if parents {
        add_footprint_parents(destination, &mut targets, limit)?;
    }
    Ok(ai_agents_core::autonomy::ToolWriteFootprint {
        binding_identity,
        targets: targets.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn mutation_context(tool_id: &str, root: &Path) -> ToolExecutionContext {
        let mut context = ToolExecutionContext::test(tool_id);
        context.policy_snapshot = serde_json::json!({
            "read_paths": [root.to_string_lossy()],
            "write_paths": [root.to_string_lossy()],
            "overwrite_existing": true,
            "create_parent_dirs": true,
            "allow_without_confirmation": true
        });
        context
    }

    fn result_json(result: &ToolResult) -> Value {
        serde_json::from_str(&result.output).unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn atomic_temp_creation_is_private_under_permissive_umask() {
        use std::os::unix::fs::PermissionsExt;

        const CHILD: &str = "AI_AGENTS_ATOMIC_TEMP_UMASK_TEST";
        if std::env::var_os(CHILD).is_some() {
            let dir = tempdir().unwrap();
            fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
            let (private, private_path) = create_atomic_temp(dir.path(), true).unwrap();
            assert_eq!(private.metadata().unwrap().permissions().mode() & 0o077, 0);
            assert_eq!(fs::metadata(private_path).unwrap().len(), 0);
            let (new_file, _) = create_atomic_temp(dir.path(), false).unwrap();
            let ordinary = fs::File::create(dir.path().join("ordinary.txt")).unwrap();
            assert_eq!(
                new_file.metadata().unwrap().permissions().mode() & 0o777,
                ordinary.metadata().unwrap().permissions().mode() & 0o777
            );
            return;
        }
        let output = std::process::Command::new("/bin/sh")
            .args([
                "-c",
                "umask 000; exec \"$1\" --exact builtin::fs_mutation::tests::atomic_temp_creation_is_private_under_permissive_umask --nocapture",
                "sh",
            ])
            .arg(std::env::current_exe().unwrap())
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_edit_preserves_existing_executable_mode() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("script.sh");
        fs::write(&path, "old\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "old",
                    "new_text": "new",
                    "dry_run": false
                }),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_edit_refuses_existing_special_mode_without_mutation() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("special.txt");
        fs::write(&path, "old\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o1700)).unwrap();
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "old",
                    "new_text": "new",
                    "dry_run": false
                }),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(!result.success);
        assert!(result.output.contains("special permission bits"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "old\n");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o7777,
            0o1700
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_write_preserves_existing_private_mode() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("private.txt");
        fs::write(&path, "old\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let result = FileWriteTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "content": "new\n",
                    "overwrite": true,
                    "dry_run": false
                }),
                mutation_context("file_write", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

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
        assert!(
            output["diff_summary"]
                .as_str()
                .unwrap()
                .contains("old_text was not found")
        );
        assert!(output["diff_summary"].as_str().unwrap().contains("re-read"));
    }

    #[tokio::test]
    async fn output_and_edit_regression_missing_text_has_bounded_context_without_mutation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let before = "before\nStatus: draft\nafter\n";
        fs::write(&path, before).unwrap();
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "Status: ready",
                    "new_text": "Status: reviewed",
                    "dry_run": false
                }),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(!result.success);
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
        let output = result_json(&result);
        let hints = output["near_matches"].as_array().unwrap();
        assert!(!hints.is_empty());
        let hints = hints
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(hints.contains("Status: draft"));
        assert!(hints.contains('2'));
        assert!(hints.chars().count() <= 1_000);
    }

    #[tokio::test]
    async fn output_and_edit_regression_diff_reports_insert_delete_and_end_of_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let tool = FileEditTool::new();
        for (before, old, new, expected, unexpected) in [
            (
                "one\nkeep\nlast\n",
                "one\n",
                "one\ninserted\n",
                "+2:\"inserted\"",
                "-2:\"keep\"",
            ),
            (
                "one\nremove\nkeep\n",
                "remove\n",
                "",
                "-2:\"remove\"",
                "+2:\"keep\"",
            ),
            (
                "keep\n",
                "keep\n",
                "keep\nlast\n",
                "+2:\"last\"",
                "-1:\"keep\"",
            ),
        ] {
            fs::write(&path, before).unwrap();
            let result = tool
                .execute(
                    serde_json::json!({
                        "path": path.to_string_lossy(),
                        "old_text": old,
                        "new_text": new,
                        "dry_run": true
                    }),
                    mutation_context("file_edit", dir.path()),
                )
                .await;
            assert!(result.success, "{}", result.output);
            let output = result_json(&result);
            let diff = output["diff_summary"].as_str().unwrap();
            assert!(diff.contains(expected), "{diff:?}");
            assert!(!diff.contains(unexpected), "{diff:?}");
            assert_eq!(fs::read_to_string(&path).unwrap(), before);
        }
    }

    #[tokio::test]
    async fn output_and_edit_regression_edit_results_fit_the_effective_output_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bounded.txt");
        let before = (0..6)
            .map(|i| format!("Status: draft {i} {}\n", "\\\"".repeat(80)))
            .collect::<String>();
        fs::write(&path, &before).unwrap();
        let mut context = mutation_context("file_edit", dir.path());
        context.limits.max_output_chars = Some(512);
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "Status: ready",
                    "new_text": "Status: reviewed",
                    "dry_run": true
                }),
                context,
            )
            .await;
        assert!(!result.success);
        assert!(result.output.chars().count() <= 512);
        let model_error = serde_json::json!({
            "success": false,
            "error": {
                "kind": "tool_error",
                "reason": result.output,
                "retryable": false,
                "requires_user_action": false
            }
        })
        .to_string();
        assert!("Error: Tool error: ".len() + model_error.chars().count() <= 512);
        let output = result_json(&result);
        assert_eq!(output["truncated"], true);
        assert!(output["diff_summary"].as_str().unwrap().contains("re-read"));
        assert!(
            output["near_matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|hint| {
                    hint.as_str()
                        .is_some_and(|hint| hint.contains("Status: draft"))
                })
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[tokio::test]
    async fn output_and_edit_regression_one_character_over_limit_remains_valid_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        fs::write(&path, "a\n").unwrap();
        let tool = FileEditTool::new();
        let args = serde_json::json!({"path":path.to_string_lossy(),"old_text":"a","new_text":"b","dry_run":true});
        let original = tool
            .execute(args.clone(), mutation_context("file_edit", dir.path()))
            .await;
        assert!(original.success);
        let cap = original.output.chars().count() - 1;
        let mut context = mutation_context("file_edit", dir.path());
        context.limits.max_output_chars = Some(cap);
        let bounded = tool.execute(args, context).await;
        assert!(bounded.success);
        assert!(bounded.output.chars().count() <= cap);
        let output = result_json(&bounded);
        assert_eq!(output["mutation_performed"], false);
        assert_eq!(output["truncated"], true);
        assert_ne!(
            output["diff_summary"],
            result_json(&original)["diff_summary"]
        );
    }

    #[tokio::test]
    async fn output_and_edit_regression_small_limit_reports_actual_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        fs::write(&path, "before\n").unwrap();
        let mut context = mutation_context("file_edit", dir.path());
        context.limits.max_output_chars = Some(48);
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({"path":path.to_string_lossy(),"old_text":"before","new_text":"after","dry_run":false}),
                context,
            )
            .await;
        assert!(result.success);
        assert!(result.output.chars().count() <= 48);
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["truncated"], true);
        assert_eq!(fs::read_to_string(&path).unwrap(), "after\n");
    }

    #[tokio::test]
    async fn output_and_edit_regression_small_error_limit_bounds_model_payload() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        fs::write(&path, "a\n").unwrap();
        let mut context = mutation_context("file_edit", dir.path());
        context.policy_snapshot["read_paths"] = serde_json::json!([]);
        context.limits.max_output_chars = Some(180);
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({"path":path.to_string_lossy(),"old_text":"absent","new_text":"x","dry_run":true}),
                context,
            )
            .await;
        assert!(!result.success);
        let model_error = serde_json::json!({
            "success": false,
            "error": {"kind":"tool_error","reason":result.output,"retryable":false,"requires_user_action":false}
        })
        .to_string();
        assert!("Error: Tool error: ".len() + model_error.chars().count() <= 180);
        assert_eq!(fs::read_to_string(&path).unwrap(), "a\n");
    }

    #[tokio::test]
    async fn output_and_edit_regression_diff_preserves_crlf_and_repeated_edits() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("unicode.txt");
        let before = "\u{feff}첫줄\r\n항목=old\r\n중간\r\n항목=old";
        fs::write(&path, before).unwrap();
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "항목=old",
                    "new_text": "항목=new",
                    "replace_all": true,
                    "dry_run": false
                }),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            before.replace("항목=old", "항목=new")
        );
        let output = result_json(&result);
        let diff = output["diff_summary"].as_str().unwrap();
        assert!(diff.contains("-2:\"항목=old\" [CRLF]"));
        assert!(diff.contains("+2:\"항목=new\" [CRLF]"));
        assert!(diff.contains("-4:\"항목=old\" [no newline]"));
        assert!(diff.contains("+4:\"항목=new\" [no newline]"));
        assert!(!diff.contains("중간"));
        assert_eq!(output["replacements"], 2);
    }

    #[tokio::test]
    async fn output_and_edit_regression_write_only_does_not_disclose_file_context() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("private.txt");
        let before = "Status: private-fixture-value\n";
        fs::write(&path, before).unwrap();
        let mut context = ToolExecutionContext::test("file_edit");
        context.policy_snapshot = serde_json::json!({
            "write_paths": [dir.path().to_string_lossy()],
            "allow_without_confirmation": true
        });
        let tool = FileEditTool::new();
        let missing = tool
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "old_text": "Status: ready", "new_text": "Status: done", "dry_run": true}),
                context.clone(),
            )
            .await;
        assert!(!missing.success);
        assert!(!missing.output.contains("private-fixture-value"));
        assert!(
            result_json(&missing)["near_matches"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        let preview = tool
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "old_text": "Status", "new_text": "State", "dry_run": true}),
                context,
            )
            .await;
        assert!(preview.success);
        assert!(!preview.output.contains("private-fixture-value"));
        assert!(preview.output.contains("Requested old text"));
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[tokio::test]
    async fn output_and_edit_regression_diff_handles_joined_split_and_unchanged_lines() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("lines.txt");
        let tool = FileEditTool::new();
        for (before, old, new, expected, unexpected) in [
            (
                "a\nb",
                "\n",
                "",
                vec!["-1:\"a\"", "-2:\"b\"", "+1:\"ab\""],
                vec![],
            ),
            (
                "ab",
                "a",
                "a\n",
                vec!["-1:\"ab\"", "+1:\"a\"", "+2:\"b\""],
                vec![],
            ),
            (
                "a\nkeep\nb\n",
                "a\nkeep\nb\n",
                "x\nkeep\ny\n",
                vec!["-1:\"a\"", "-3:\"b\"", "+1:\"x\"", "+3:\"y\""],
                vec!["-2:\"keep\"", "+2:\"keep\""],
            ),
        ] {
            fs::write(&path, before).unwrap();
            let result = tool
                .execute(
                    serde_json::json!({"path": path.to_string_lossy(), "old_text": old, "new_text": new, "dry_run": true}),
                    mutation_context("file_edit", dir.path()),
                )
                .await;
            assert!(result.success, "{}", result.output);
            let output = result_json(&result);
            let diff = output["diff_summary"].as_str().unwrap();
            for expected in expected {
                assert!(diff.contains(expected), "missing {expected:?}: {diff:?}");
            }
            for unexpected in unexpected {
                assert!(
                    !diff.contains(unexpected),
                    "unexpected {unexpected:?}: {diff:?}"
                );
            }
        }
    }

    #[test]
    fn output_and_edit_regression_diff_preserves_all_unchanged_lines() {
        fn lines(text: &str) -> Vec<&str> {
            text.split_inclusive('\n').collect()
        }
        for before in ["a\nb", "a\nb\n", "ab\nc", "a\n\nb", "a\r\nb", "ab", ""] {
            let mut ranges = Vec::new();
            for start in 0..before.len() {
                for end in start + 1..=before.len() {
                    let old = &before[start..end];
                    if before.match_indices(old).count() == 1 {
                        ranges.push(old);
                    }
                }
            }
            if before.is_empty() {
                ranges.push("");
            }
            for old in ranges {
                for new in ["", "x", "\n", "x\n", "\nx"] {
                    if old == new {
                        continue;
                    }
                    let after = before.replacen(old, new, 1);
                    let (diff, truncated) = preview_diff(before, &after, old, new, 1, 20_000);
                    assert!(!truncated);
                    assert!(
                        !diff.is_empty(),
                        "before={before:?}, old={old:?}, new={new:?}"
                    );
                    let original = lines(before);
                    let edited = lines(&after);
                    let mut removed = HashSet::new();
                    let mut added = HashSet::new();
                    for row in diff.lines() {
                        let sign = row.chars().next().unwrap();
                        let (number, encoded) = row[1..].split_once(':').unwrap();
                        let number: usize = number.parse().unwrap();
                        let (encoded, ending) =
                            if let Some(encoded) = encoded.strip_suffix(" [CRLF]") {
                                (encoded, "\r\n")
                            } else if let Some(encoded) = encoded.strip_suffix(" [no newline]") {
                                (encoded, "")
                            } else {
                                (encoded, "\n")
                            };
                        let decoded: String = serde_json::from_str(encoded).unwrap();
                        let rendered = format!("{decoded}{ending}");
                        match sign {
                            '-' => {
                                assert!(removed.insert(number));
                                assert_eq!(original[number - 1], rendered);
                            }
                            '+' => {
                                assert!(added.insert(number));
                                assert_eq!(edited[number - 1], rendered);
                            }
                            _ => panic!("invalid diff line: {row}"),
                        }
                    }
                    let kept_old: Vec<_> = original
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| !removed.contains(&(index + 1)))
                        .map(|(_, line)| *line)
                        .collect();
                    let kept_new: Vec<_> = edited
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| !added.contains(&(index + 1)))
                        .map(|(_, line)| *line)
                        .collect();
                    assert_eq!(
                        kept_old, kept_new,
                        "before={before:?}, old={old:?}, new={new:?}, diff={diff:?}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn output_and_edit_regression_diff_distinguishes_literal_line_ending_markers() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("markers.txt");
        fs::write(&path, "a [CRLF]\n").unwrap();
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "old_text": " [CRLF]\n", "new_text": "\r\n", "dry_run": true}),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        let output = result_json(&result);
        let diff = output["diff_summary"].as_str().unwrap();
        assert!(diff.contains("-1:\"a [CRLF]\""), "{diff}");
        assert!(diff.contains("+1:\"a\" [CRLF]"), "{diff}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn output_and_edit_regression_unresolvable_read_root_fails_closed() {
        use std::os::unix::fs::symlink;
        let dir = tempdir().unwrap();
        let path = dir.path().join("private.txt");
        fs::write(&path, "Status: private-fixture-value\n").unwrap();
        let dangling = dir.path().join("dangling");
        symlink(dir.path().join("missing"), &dangling).unwrap();
        let mut context = mutation_context("file_edit", dir.path());
        context.policy_snapshot["read_paths"] =
            serde_json::json!([dangling.to_string_lossy(), dir.path().to_string_lossy()]);
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "old_text": "Status: ready", "new_text": "Status: done", "dry_run": true}),
                context,
            )
            .await;
        assert!(!result.success);
        assert!(!result.output.contains("private-fixture-value"));
        assert!(
            result_json(&result)["near_matches"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn output_and_edit_regression_fixed_fields_do_not_falsely_claim_preview_truncation() {
        let mut output = MutationOutput {
            path: Some("a".repeat(200)),
            dry_run: true,
            mutation_performed: false,
            changed_files: 0,
            changed_lines: 0,
            replacements: 0,
            bytes_written: 0,
            created: false,
            overwritten: false,
            truncated: false,
            approval_required: false,
            diff_summary: "old_text was not found; re-read the file and use exact text".into(),
            changed_paths: Vec::new(),
            version: None,
            near_matches: Vec::new(),
        };
        let full = serde_json::to_string(&output).unwrap();
        let cap = full.chars().count() - 1;
        let result = serialize_bounded_file_edit(&mut output, false, Some(cap)).unwrap();
        assert_eq!(result, full);
        assert!(!output.truncated);
    }

    #[tokio::test]
    async fn output_and_edit_regression_empty_text_keeps_replacement_limit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large.txt");
        let before = "한".repeat(50_000);
        fs::write(&path, &before).unwrap();
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({"path":path.to_string_lossy(),"old_text":"","new_text":"x","replace_all":true,"max_replacements":3,"dry_run":false}),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(!result.success);
        assert!(result.output.contains("max_replacements"));
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[tokio::test]
    async fn output_and_edit_regression_large_diff_is_explicitly_incomplete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large.txt");
        let before = (0..501).map(|n| format!("old-{n}\n")).collect::<String>();
        let after = (0..501).map(|n| format!("new-{n}\n")).collect::<String>();
        fs::write(&path, &before).unwrap();
        let result = FileEditTool::new()
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "old_text": before, "new_text": after, "dry_run": true}),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        let output = result_json(&result);
        assert_eq!(output["truncated"], true);
        assert!(output["diff_summary"].as_str().unwrap().contains("omitted"));
    }

    #[tokio::test]
    async fn output_and_edit_regression_stale_read_cannot_mutate_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stale.txt");
        let before = "status=old\n";
        fs::write(&path, before).unwrap();
        let versions = FileVersionStore::default();
        versions.record(file_version_evidence(&path, before.as_bytes()).unwrap());
        let changed = "external change\nstatus=old\n";
        fs::write(&path, changed).unwrap();
        let mut context = mutation_context("file_edit", dir.path());
        context.policy_snapshot["require_read_before_write"] = Value::Bool(true);
        let result = FileEditTool::with_version_store(versions)
            .execute(
                serde_json::json!({"path": path.to_string_lossy(), "old_text": "status=old", "new_text": "status=new", "dry_run": false}),
                context,
            )
            .await;
        assert!(!result.success);
        assert!(result.output.contains("must not change before write"));
        assert_eq!(fs::read_to_string(&path).unwrap(), changed);
    }

    #[tokio::test]
    async fn file_write_denies_actual_without_write_policy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let tool = FileWriteTool::new();
        let result = tool
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "content": "hello",
                    "dry_run": false
                }),
                ToolExecutionContext::test("file_write"),
            )
            .await;
        assert!(!result.success);
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

    #[tokio::test]
    async fn patch_delete_dry_run_reports_planned_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delete.txt");
        fs::write(&path, "old\n").unwrap();
        let result = PatchTool::new()
            .execute(
                serde_json::json!({
                    "base_path": dir.path().to_string_lossy(),
                    "allow_delete": true,
                    "dry_run": true,
                    "patch": "--- a/delete.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-old\n"
                }),
                ToolExecutionContext::test("patch"),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert!(path.exists());
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], false);
        assert_eq!(output["created"], false);
        assert_eq!(output["overwritten"], false);
    }

    #[tokio::test]
    async fn new_file_patch_rejects_existing_target() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("existing.txt");
        fs::write(&path, "original\n").unwrap();
        let result = PatchTool::new()
            .execute(
                serde_json::json!({
                    "base_path": dir.path().to_string_lossy(),
                    "allow_new_files": true,
                    "patch": "--- /dev/null\n+++ b/existing.txt\n@@ -0,0 +1,1 @@\n+created\n"
                }),
                mutation_context("patch", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("already exists"));
        assert_eq!(fs::read_to_string(path).unwrap(), "original\n");
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
        assert_eq!(output["mutation_performed"], false);
        assert_eq!(output["deleted"], false);
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
        assert_eq!(output["mutation_performed"], false);
        assert_eq!(output["copied"], false);
        assert_eq!(output["dry_run"], true);
    }

    #[tokio::test]
    async fn copy_path_rejects_destination_nested_under_source() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("file.txt"), "hello").unwrap();
        let destination = source.join("nested");

        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": true
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("nested under the source"));
        assert!(!destination.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn copy_path_rejects_symlinks_inside_source_tree() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let source = dir.path().join("source");
        let outside = dir.path().join("outside");
        let destination = dir.path().join("destination");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(&outside, source.join("alias")).unwrap();

        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": true
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("symbolic links are not supported"));
        assert!(!destination.exists());
    }

    #[test]
    fn mutation_schemas_default_to_execution() {
        for schema in [
            FileWriteTool::new().input_schema(),
            FileEditTool::new().input_schema(),
            PatchTool::new().input_schema(),
            CopyPathTool::new().input_schema(),
            MovePathTool::new().input_schema(),
            DeletePathTool::new().input_schema(),
        ] {
            assert_eq!(schema["properties"]["dry_run"]["default"], false);
        }
    }

    #[test]
    fn omitted_dry_run_classifies_as_mutation() {
        for classification in [
            FileWriteTool::new().classify_call(&serde_json::json!({})),
            FileEditTool::new().classify_call(&serde_json::json!({})),
            PatchTool::new().classify_call(&serde_json::json!({})),
            CopyPathTool::new().classify_call(&serde_json::json!({})),
            MovePathTool::new().classify_call(&serde_json::json!({})),
            DeletePathTool::new().classify_call(&serde_json::json!({})),
        ] {
            assert!(!classification.read_only);
            assert!(!classification.concurrency_safe);
            assert!(!classification.safely_retryable);
            assert!(classification.requires_approval);
        }
    }

    #[test]
    fn actual_delete_classification_remains_destructive_delete() {
        let classification =
            DeletePathTool::new().classify_call(&serde_json::json!({"dry_run": false}));
        assert!(matches!(
            classification.operation,
            ToolOperationKind::Delete
        ));
        assert!(matches!(
            classification.side_effect_level,
            ToolSideEffectLevel::Destructive
        ));
        assert!(classification.destructive);
        assert!(!classification.read_only);
    }

    #[tokio::test]
    async fn file_write_omitted_dry_run_applies_and_records_version() {
        let dir = tempdir().unwrap();
        let parent = dir.path().join("missing");
        let path = parent.join("new.txt");
        let versions = FileVersionStore::default();
        let tool = FileWriteTool::with_version_store(versions.clone());
        let result = tool
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "content": "hello",
                    "create_parent_dirs": true
                }),
                mutation_context("file_write", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");
        assert!(versions.get(&path).is_some());
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["created"], true);
        assert_eq!(output["overwritten"], false);
        assert_eq!(output["bytes_written"], 5);
    }

    #[tokio::test]
    async fn file_edit_omitted_dry_run_applies_and_records_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        fs::write(&path, "before\n").unwrap();
        let versions = FileVersionStore::default();
        let tool = FileEditTool::with_version_store(versions.clone());
        let result = tool
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "old_text": "before",
                    "new_text": "after"
                }),
                mutation_context("file_edit", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert_eq!(fs::read_to_string(&path).unwrap(), "after\n");
        assert!(versions.get(&path).is_some());
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["overwritten"], true);
    }

    #[tokio::test]
    async fn omitted_dry_run_copies_moves_and_deletes() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let copy_parent = dir.path().join("copy-parent");
        let copy_destination = copy_parent.join("copy.txt");
        let move_parent = dir.path().join("move-parent");
        let move_destination = move_parent.join("move.txt");
        fs::write(&source, "content").unwrap();

        let copy_result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": copy_destination.to_string_lossy(),
                    "create_parent_dirs": true
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;
        assert!(copy_result.success, "{}", copy_result.output);
        assert_eq!(fs::read_to_string(&copy_destination).unwrap(), "content");
        let copy_output = result_json(&copy_result);
        assert_eq!(copy_output["mutation_performed"], true);
        assert_eq!(copy_output["copied"], true);

        let move_result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": move_destination.to_string_lossy(),
                    "create_parent_dirs": true
                }),
                mutation_context("move_path", dir.path()),
            )
            .await;
        assert!(move_result.success, "{}", move_result.output);
        assert!(!source.exists());
        assert_eq!(fs::read_to_string(&move_destination).unwrap(), "content");
        let move_output = result_json(&move_result);
        assert_eq!(move_output["mutation_performed"], true);
        assert_eq!(move_output["moved"], true);

        let delete_result = DeletePathTool::new()
            .execute(
                serde_json::json!({"path": move_destination.to_string_lossy()}),
                mutation_context("delete_path", dir.path()),
            )
            .await;
        assert!(delete_result.success, "{}", delete_result.output);
        assert!(!move_destination.exists());
        let delete_output = result_json(&delete_result);
        assert_eq!(delete_output["mutation_performed"], true);
        assert_eq!(delete_output["deleted"], true);
    }

    #[tokio::test]
    async fn move_path_rejects_same_source_and_destination() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("same.txt");
        fs::write(&path, "content").unwrap();
        let result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": path.to_string_lossy(),
                    "destination_path": path.to_string_lossy(),
                    "overwrite": true
                }),
                mutation_context("move_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("same path"));
        assert_eq!(fs::read_to_string(path).unwrap(), "content");
    }

    #[tokio::test]
    async fn delete_path_explicit_dry_run_false_reports_applied_mutation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delete.txt");
        fs::write(&path, "content").unwrap();
        let result = DeletePathTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("delete_path", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert!(!path.exists());
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["deleted"], true);
    }

    #[tokio::test]
    async fn patch_omitted_dry_run_applies_and_records_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("patch.txt");
        fs::write(&path, "old\n").unwrap();
        let versions = FileVersionStore::default();
        let result = PatchTool::with_version_store(versions.clone())
            .execute(
                serde_json::json!({
                    "base_path": dir.path().to_string_lossy(),
                    "patch": "--- a/patch.txt\n+++ b/patch.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
                }),
                mutation_context("patch", dir.path()),
            )
            .await;
        assert!(result.success, "{}", result.output);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
        assert!(versions.get(&path).is_some());
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["created"], false);
        assert_eq!(output["overwritten"], true);
        assert_eq!(output["bytes_written"], 4);
    }

    #[test]
    fn patch_apply_failure_rolls_back_prior_writes() {
        let dir = tempdir().unwrap();
        let first = dir.path().join("first.txt");
        let invalid_parent = dir.path().join("not-a-directory");
        fs::write(&first, "old\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&first, fs::Permissions::from_mode(0o600)).unwrap();
        }
        fs::write(&invalid_parent, "blocker\n").unwrap();
        let outputs = vec![
            PatchApply::Write {
                path: first.clone(),
                content: "new\n".to_string(),
                expected: ExpectedPathState::Present(read_file_snapshot(&first).unwrap()),
            },
            PatchApply::Write {
                path: invalid_parent.join("child.txt"),
                content: "child\n".to_string(),
                expected: ExpectedPathState::Absent,
            },
        ];
        let error = apply_patch_transaction(&outputs).unwrap_err();
        assert!(error.contains("patch rollback restored file contents and permissions"));
        assert!(error.contains("other metadata may have changed"));
        assert_eq!(fs::read_to_string(&first).unwrap(), "old\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&first).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        assert!(!invalid_parent.join("child.txt").exists());
    }

    #[cfg(unix)]
    #[test]
    fn patch_delete_rollback_restores_private_bytes_and_reports_metadata_limits() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let dir = tempdir().unwrap();
        let path = dir.path().join("private.txt");
        let link = dir.path().join("link.txt");
        let invalid_parent = dir.path().join("blocker");
        fs::write(&path, "private data\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::hard_link(&path, &link).unwrap();
        fs::write(&invalid_parent, "blocker\n").unwrap();
        let outputs = vec![
            PatchApply::Delete {
                path: path.clone(),
                expected: read_file_snapshot(&path).unwrap(),
            },
            PatchApply::Write {
                path: invalid_parent.join("child"),
                content: "new\n".into(),
                expected: ExpectedPathState::Absent,
            },
        ];
        let error = apply_patch_transaction(&outputs).unwrap_err();
        assert!(error.contains("patch rollback restored file contents and permissions"));
        assert!(error.contains("other metadata may have changed"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "private data\n");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(fs::read_to_string(&link).unwrap(), "private data\n");
        assert_ne!(
            fs::metadata(&path).unwrap().ino(),
            fs::metadata(&link).unwrap().ino()
        );
    }

    #[cfg(unix)]
    #[test]
    fn patch_created_file_rollback_preserves_external_permission_changes() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("created.txt");
        fs::write(&path, "created\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let written_metadata = fs::metadata(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        let error = rollback_patch(
            &[PatchRollback::RemoveCreated {
                path: path.clone(),
                written: b"created\n".to_vec(),
                written_metadata,
            }],
            &[],
        )
        .unwrap_err();
        assert!(error.contains("permissions changed"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "created\n");
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[cfg(unix)]
    #[test]
    fn patch_created_file_rollback_preserves_same_content_replacement() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("created.txt");
        let replacement = dir.path().join("replacement.txt");
        fs::write(&path, "created\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let written_metadata = fs::metadata(&path).unwrap();
        fs::write(&replacement, "created\n").unwrap();
        fs::set_permissions(&replacement, fs::Permissions::from_mode(0o600)).unwrap();
        fs::rename(&replacement, &path).unwrap();
        let error = rollback_patch(
            &[PatchRollback::RemoveCreated {
                path: path.clone(),
                written: b"created\n".to_vec(),
                written_metadata,
            }],
            &[],
        )
        .unwrap_err();
        assert!(error.contains("created file was replaced"));
        assert_eq!(fs::read_to_string(path).unwrap(), "created\n");
    }

    #[cfg(unix)]
    #[test]
    fn patch_delete_rollback_preserves_a_dangling_symlink() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("target.txt");
        fs::write(&path, "original\n").unwrap();
        let original = read_file_snapshot(&path).unwrap();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink("missing-target", &path).unwrap();
        let error = rollback_patch(
            &[PatchRollback::RestoreDelete {
                path: path.clone(),
                original,
            }],
            &[],
        )
        .unwrap_err();
        assert!(error.contains("rollback conflict"));
        assert!(
            fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn atomic_temp_cleanup_failure_is_reported() {
        let dir = tempdir().unwrap();
        let original = std::io::Error::other("write failed");
        let error = cleanup_failed_atomic_temp(dir.path(), original);
        assert!(error.to_string().contains("write failed"));
        assert!(error.to_string().contains("partial data may remain"));
        assert!(dir.path().is_dir());
    }

    #[test]
    fn patch_delete_rollback_does_not_overwrite_an_external_replacement() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("target.txt");
        fs::write(&path, "original\n").unwrap();
        let original = read_file_snapshot(&path).unwrap();
        fs::remove_file(&path).unwrap();
        fs::write(&path, "external\n").unwrap();
        let error = rollback_patch(
            &[PatchRollback::RestoreDelete {
                path: path.clone(),
                original,
            }],
            &[],
        )
        .unwrap_err();
        assert!(error.contains("rollback conflict"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "external\n");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn patch_special_mode_delete_is_rejected_before_earlier_writes() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let dir = tempdir().unwrap();
        let first = dir.path().join("first.txt");
        let special = dir.path().join("special.txt");
        fs::write(&first, "old\n").unwrap();
        fs::write(&special, "protected\n").unwrap();
        fs::set_permissions(&special, fs::Permissions::from_mode(0o1700)).unwrap();
        let inode = fs::metadata(&first).unwrap().ino();
        let patch = concat!(
            "--- a/first.txt\n+++ b/first.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n",
            "--- a/special.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-protected\n"
        );
        let result = PatchTool::new()
            .execute(
                serde_json::json!({
                    "base_path": dir.path().to_string_lossy(),
                    "patch": patch,
                    "allow_delete": true,
                    "dry_run": false
                }),
                mutation_context("patch", dir.path()),
            )
            .await;
        assert!(!result.success);
        assert!(result.output.contains("special permission bits"));
        assert_eq!(fs::read_to_string(&first).unwrap(), "old\n");
        assert_eq!(fs::metadata(&first).unwrap().ino(), inode);
        assert_eq!(fs::read_to_string(&special).unwrap(), "protected\n");
        assert_eq!(
            fs::metadata(&special).unwrap().permissions().mode() & 0o7777,
            0o1700
        );
    }

    #[test]
    fn patch_rejects_pre_apply_content_and_absence_conflicts() {
        let dir = tempdir().unwrap();
        let existing = dir.path().join("existing.txt");
        let created = dir.path().join("created.txt");
        fs::write(&existing, "original\n").unwrap();
        let existing_output = PatchApply::Write {
            path: existing.clone(),
            content: "patched\n".to_string(),
            expected: ExpectedPathState::Present(read_file_snapshot(&existing).unwrap()),
        };
        fs::write(&existing, "external\n").unwrap();
        let error = apply_patch_transaction(&[existing_output]).unwrap_err();
        assert!(error.contains("pre-apply conflict"));
        assert_eq!(fs::read_to_string(&existing).unwrap(), "external\n");

        let created_output = PatchApply::Write {
            path: created.clone(),
            content: "patched\n".to_string(),
            expected: ExpectedPathState::Absent,
        };
        fs::write(&created, "external\n").unwrap();
        let error = apply_patch_transaction(&[created_output]).unwrap_err();
        assert!(error.contains("pre-apply conflict"));
        assert_eq!(fs::read_to_string(&created).unwrap(), "external\n");
    }

    #[test]
    fn patch_rollback_conflict_preserves_external_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("target.txt");
        fs::write(&path, "original\n").unwrap();
        let original = read_file_snapshot(&path).unwrap();
        fs::write(&path, "external\n").unwrap();
        let rollback = PatchRollback::RestoreWrite {
            path: path.clone(),
            original: original.clone(),
            written: b"transaction\n".to_vec(),
            written_permissions: original.permissions,
        };
        let error = rollback_patch(&[rollback], &[]).unwrap_err();
        assert!(error.contains("rollback conflict"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "external\n");
    }

    #[cfg(unix)]
    #[test]
    fn patch_rollback_conflict_preserves_external_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("target.txt");
        fs::write(&path, "original\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let original = read_file_snapshot(&path).unwrap();
        fs::write(&path, "transaction\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        let rollback = PatchRollback::RestoreWrite {
            path: path.clone(),
            original: original.clone(),
            written: b"transaction\n".to_vec(),
            written_permissions: original.permissions,
        };

        let error = rollback_patch(&[rollback], &[]).unwrap_err();

        assert!(error.contains("permissions changed"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "transaction\n");
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[tokio::test]
    async fn patch_preflights_every_file_before_mutating_any_target() {
        let dir = tempdir().unwrap();
        let first = dir.path().join("first.txt");
        let second = dir.path().join("second.txt");
        fs::write(&first, "first old\n").unwrap();
        fs::write(&second, "second old\n").unwrap();
        let patch = concat!(
            "--- a/first.txt\n+++ b/first.txt\n@@ -1,1 +1,1 @@\n-first old\n+first new\n",
            "--- a/second.txt\n+++ b/second.txt\n@@ -1,1 +1,1 @@\n-not present\n+second new\n"
        );
        let result = PatchTool::new()
            .execute(
                serde_json::json!({
                    "base_path": dir.path().to_string_lossy(),
                    "patch": patch,
                    "dry_run": false
                }),
                mutation_context("patch", dir.path()),
            )
            .await;
        assert!(!result.success);
        assert_eq!(fs::read_to_string(&first).unwrap(), "first old\n");
        assert_eq!(fs::read_to_string(&second).unwrap(), "second old\n");
    }

    #[tokio::test]
    async fn copy_path_rejects_missing_source_without_creating_destination() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("missing.txt");
        let destination = dir.path().join("destination.txt");
        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("source path does not exist"));
        assert!(!destination.exists());
    }

    #[tokio::test]
    async fn move_path_rejects_missing_source_without_creating_destination() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("missing.txt");
        let destination = dir.path().join("destination.txt");
        let result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("move_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("source path does not exist"));
        assert!(!destination.exists());
    }

    #[tokio::test]
    async fn delete_path_rejects_missing_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("missing.txt");
        let result = DeletePathTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("delete_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("path does not exist"));
    }

    #[tokio::test]
    async fn copy_path_collision_requires_overwrite() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("destination.txt");
        fs::write(&source, "source").unwrap();
        fs::write(&destination, "destination").unwrap();
        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("overwrite must be true"));
        assert_eq!(fs::read_to_string(&source).unwrap(), "source");
        assert_eq!(fs::read_to_string(&destination).unwrap(), "destination");
    }

    #[tokio::test]
    async fn move_path_collision_requires_overwrite() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("destination.txt");
        fs::write(&source, "source").unwrap();
        fs::write(&destination, "destination").unwrap();
        let result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("move_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("overwrite must be true"));
        assert_eq!(fs::read_to_string(&source).unwrap(), "source");
        assert_eq!(fs::read_to_string(&destination).unwrap(), "destination");
    }

    #[tokio::test]
    async fn copy_path_overwrite_replaces_existing_file_and_reports_effect() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("destination.txt");
        fs::write(&source, "source").unwrap();
        fs::write(&destination, "old").unwrap();
        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "overwrite": true,
                    "dry_run": false
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;

        assert!(result.success, "{}", result.output);
        assert_eq!(fs::read_to_string(&source).unwrap(), "source");
        assert_eq!(fs::read_to_string(&destination).unwrap(), "source");
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["copied"], true);
        assert_eq!(output["overwritten"], true);
        assert_eq!(output["bytes_affected"], 6);
    }

    #[tokio::test]
    async fn copy_path_overwrite_replaces_existing_directory_tree() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source");
        let destination = dir.path().join("destination");
        fs::create_dir_all(source.join("nested")).unwrap();
        fs::create_dir_all(&destination).unwrap();
        fs::write(source.join("nested/new.txt"), "new").unwrap();
        fs::write(destination.join("stale.txt"), "stale").unwrap();
        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "overwrite": true,
                    "dry_run": false
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;

        assert!(result.success, "{}", result.output);
        assert_eq!(
            fs::read_to_string(destination.join("nested/new.txt")).unwrap(),
            "new"
        );
        assert!(!destination.join("stale.txt").exists());
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["copied"], true);
        assert_eq!(output["overwritten"], true);
        assert_eq!(output["recursive"], true);
        assert_eq!(output["bytes_affected"], 3);
    }

    #[tokio::test]
    async fn move_path_overwrite_replaces_existing_file_and_reports_effect() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("destination.txt");
        fs::write(&source, "source").unwrap();
        fs::write(&destination, "old").unwrap();
        let result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "overwrite": true,
                    "dry_run": false
                }),
                mutation_context("move_path", dir.path()),
            )
            .await;

        assert!(result.success, "{}", result.output);
        assert!(!source.exists());
        assert_eq!(fs::read_to_string(&destination).unwrap(), "source");
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["moved"], true);
        assert_eq!(output["overwritten"], true);
        assert_eq!(output["bytes_affected"], 6);
    }

    #[tokio::test]
    async fn move_path_overwrite_replaces_existing_directory_tree() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source");
        let destination = dir.path().join("destination");
        fs::create_dir_all(source.join("nested")).unwrap();
        fs::create_dir_all(&destination).unwrap();
        fs::write(source.join("nested/new.txt"), "new").unwrap();
        fs::write(destination.join("stale.txt"), "stale").unwrap();
        let result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "overwrite": true,
                    "dry_run": false
                }),
                mutation_context("move_path", dir.path()),
            )
            .await;

        assert!(result.success, "{}", result.output);
        assert!(!source.exists());
        assert_eq!(
            fs::read_to_string(destination.join("nested/new.txt")).unwrap(),
            "new"
        );
        assert!(!destination.join("stale.txt").exists());
        let output = result_json(&result);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["moved"], true);
        assert_eq!(output["overwritten"], true);
        assert_eq!(output["recursive"], true);
        assert_eq!(output["bytes_affected"], 3);
    }

    #[tokio::test]
    async fn copy_path_parent_creation_requires_request_opt_in() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("missing/destination.txt");
        fs::write(&source, "source").unwrap();
        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "create_parent_dirs": false,
                    "dry_run": false
                }),
                mutation_context("copy_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("parent directory does not exist"));
        assert!(!destination.exists());
        assert!(!dir.path().join("missing").exists());
    }

    #[tokio::test]
    async fn move_path_parent_creation_requires_request_opt_in() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let destination = dir.path().join("missing/destination.txt");
        fs::write(&source, "source").unwrap();
        let result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "create_parent_dirs": false,
                    "dry_run": false
                }),
                mutation_context("move_path", dir.path()),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("parent directory does not exist"));
        assert!(source.exists());
        assert!(!destination.exists());
        assert!(!dir.path().join("missing").exists());
    }

    #[tokio::test]
    async fn copy_and_move_parent_creation_requires_policy_opt_in() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        fs::create_dir_all(&root).unwrap();
        let copy_source = root.join("copy-source.txt");
        let move_source = root.join("move-source.txt");
        let copy_destination = root.join("copy-parent/destination.txt");
        let move_destination = root.join("move-parent/destination.txt");
        fs::write(&copy_source, "copy").unwrap();
        fs::write(&move_source, "move").unwrap();
        let mut copy_context = mutation_context("copy_path", &root);
        copy_context.policy_snapshot["create_parent_dirs"] = Value::Bool(false);
        let mut move_context = mutation_context("move_path", &root);
        move_context.policy_snapshot["create_parent_dirs"] = Value::Bool(false);

        let copy_result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": copy_source.to_string_lossy(),
                    "destination_path": copy_destination.to_string_lossy(),
                    "create_parent_dirs": true,
                    "dry_run": false
                }),
                copy_context,
            )
            .await;
        assert!(!copy_result.success);
        assert!(!copy_destination.exists());
        assert!(!root.join("copy-parent").exists());

        let move_result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": move_source.to_string_lossy(),
                    "destination_path": move_destination.to_string_lossy(),
                    "create_parent_dirs": true,
                    "dry_run": false
                }),
                move_context,
            )
            .await;
        assert!(!move_result.success);
        assert!(move_source.exists());
        assert!(!move_destination.exists());
        assert!(!root.join("move-parent").exists());
    }

    #[tokio::test]
    async fn copy_and_move_overwrite_require_policy_opt_in() {
        let dir = tempdir().unwrap();
        let copy_source = dir.path().join("copy-source.txt");
        let copy_destination = dir.path().join("copy-destination.txt");
        let move_source = dir.path().join("move-source.txt");
        let move_destination = dir.path().join("move-destination.txt");
        fs::write(&copy_source, "new copy").unwrap();
        fs::write(&copy_destination, "old copy").unwrap();
        fs::write(&move_source, "new move").unwrap();
        fs::write(&move_destination, "old move").unwrap();
        let mut copy_context = mutation_context("copy_path", dir.path());
        copy_context.policy_snapshot["overwrite_existing"] = Value::Bool(false);
        let mut move_context = mutation_context("move_path", dir.path());
        move_context.policy_snapshot["overwrite_existing"] = Value::Bool(false);

        let copy_result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": copy_source.to_string_lossy(),
                    "destination_path": copy_destination.to_string_lossy(),
                    "overwrite": true,
                    "dry_run": false
                }),
                copy_context,
            )
            .await;
        assert!(!copy_result.success);
        assert!(
            copy_result
                .output
                .contains("overwrite_existing policy is false")
        );
        assert_eq!(fs::read_to_string(&copy_destination).unwrap(), "old copy");

        let move_result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": move_source.to_string_lossy(),
                    "destination_path": move_destination.to_string_lossy(),
                    "overwrite": true,
                    "dry_run": false
                }),
                move_context,
            )
            .await;
        assert!(!move_result.success);
        assert!(
            move_result
                .output
                .contains("overwrite_existing policy is false")
        );
        assert!(move_source.exists());
        assert_eq!(fs::read_to_string(&move_destination).unwrap(), "old move");
    }

    #[tokio::test]
    async fn delete_path_recursively_removes_directory_and_reports_effect() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tree");
        fs::create_dir_all(path.join("nested")).unwrap();
        fs::write(path.join("one.txt"), "one").unwrap();
        fs::write(path.join("nested/two.txt"), "two").unwrap();
        let result = DeletePathTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "recursive": true,
                    "dry_run": false
                }),
                mutation_context("delete_path", dir.path()),
            )
            .await;

        assert!(result.success, "{}", result.output);
        assert!(!path.exists());
        let output = result_json(&result);
        assert_eq!(output["dry_run"], false);
        assert_eq!(output["mutation_performed"], true);
        assert_eq!(output["deleted"], true);
        assert_eq!(output["recursive"], true);
        assert_eq!(output["bytes_affected"], 6);
        assert_eq!(output["approval_required"], false);
    }

    #[tokio::test]
    async fn file_write_allows_creation_beneath_missing_absolute_root() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("missing-root");
        let path = root.join("nested/file.txt");
        let result = FileWriteTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "content": "created",
                    "create_parent_dirs": true,
                    "dry_run": false
                }),
                mutation_context("file_write", &root),
            )
            .await;

        assert!(result.success, "{}", result.output);
        assert_eq!(fs::read_to_string(path).unwrap(), "created");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_write_rejects_dangling_allowed_root() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let missing_target = dir.path().join("missing-target");
        let root = dir.path().join("dangling-root");
        symlink(&missing_target, &root).unwrap();
        let path = root.join("file.txt");
        let result = FileWriteTool::new()
            .execute(
                serde_json::json!({
                    "path": path.to_string_lossy(),
                    "content": "blocked",
                    "create_parent_dirs": true,
                    "dry_run": false
                }),
                mutation_context("file_write", &root),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("Path policy resolution failed"));
        assert!(!missing_target.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn copy_path_rejects_blocked_source_through_symlink_alias() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        let blocked = dir.path().join("blocked");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("source.txt"), "source").unwrap();
        symlink(&blocked, root.join("alias")).unwrap();
        let destination = root.join("destination.txt");
        let mut context = mutation_context("copy_path", &root);
        context.policy_snapshot["blocked_paths"] = serde_json::json!([blocked.to_string_lossy()]);

        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": root.join("alias/source.txt").to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "dry_run": false
                }),
                context,
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("source path is blocked"));
        assert!(!destination.exists());
    }

    #[tokio::test]
    async fn mutation_tools_reject_paths_outside_write_root() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        let copy_source = root.join("copy-source.txt");
        let move_source = root.join("move-source.txt");
        let delete_target = outside.join("delete-target.txt");
        fs::write(&copy_source, "copy").unwrap();
        fs::write(&move_source, "move").unwrap();
        fs::write(&delete_target, "delete").unwrap();

        let copy_result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": copy_source.to_string_lossy(),
                    "destination_path": outside.join("copied.txt").to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("copy_path", &root),
            )
            .await;
        assert!(!copy_result.success);
        assert!(copy_result.output.contains("allowed write root"));
        assert!(!outside.join("copied.txt").exists());

        let move_result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": move_source.to_string_lossy(),
                    "destination_path": outside.join("moved.txt").to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("move_path", &root),
            )
            .await;
        assert!(!move_result.success);
        assert!(move_result.output.contains("allowed write root"));
        assert!(move_source.exists());
        assert!(!outside.join("moved.txt").exists());

        let delete_result = DeletePathTool::new()
            .execute(
                serde_json::json!({
                    "path": delete_target.to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("delete_path", &root),
            )
            .await;
        assert!(!delete_result.success);
        assert!(delete_result.output.contains("allowed write root"));
        assert_eq!(fs::read_to_string(delete_target).unwrap(), "delete");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mutation_tools_reject_symlink_write_escape() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("escape")).unwrap();
        let copy_source = root.join("copy-source.txt");
        let move_source = root.join("move-source.txt");
        let delete_target = outside.join("delete-target.txt");
        fs::write(&copy_source, "copy").unwrap();
        fs::write(&move_source, "move").unwrap();
        fs::write(&delete_target, "delete").unwrap();

        let copy_result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": copy_source.to_string_lossy(),
                    "destination_path": root.join("escape/copied.txt").to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("copy_path", &root),
            )
            .await;
        assert!(!copy_result.success);
        assert!(!outside.join("copied.txt").exists());

        let move_result = MovePathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": move_source.to_string_lossy(),
                    "destination_path": root.join("escape/moved.txt").to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("move_path", &root),
            )
            .await;
        assert!(!move_result.success);
        assert!(move_source.exists());
        assert!(!outside.join("moved.txt").exists());

        let delete_result = DeletePathTool::new()
            .execute(
                serde_json::json!({
                    "path": root.join("escape/delete-target.txt").to_string_lossy(),
                    "dry_run": false
                }),
                mutation_context("delete_path", &root),
            )
            .await;
        assert!(!delete_result.success);
        assert_eq!(fs::read_to_string(delete_target).unwrap(), "delete");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn copy_path_rejects_broken_destination_symlink_escape() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        let source = root.join("source.txt");
        let escaped = outside.join("escaped.txt");
        let destination = root.join("destination.txt");
        fs::write(&source, "source").unwrap();
        symlink(&escaped, &destination).unwrap();

        let result = CopyPathTool::new()
            .execute(
                serde_json::json!({
                    "source_path": source.to_string_lossy(),
                    "destination_path": destination.to_string_lossy(),
                    "overwrite": true,
                    "dry_run": false
                }),
                mutation_context("copy_path", &root),
            )
            .await;

        assert!(!result.success);
        assert!(!escaped.exists());
        assert!(
            fs::symlink_metadata(destination)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[tokio::test]
    async fn delete_path_rejects_configured_write_root() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("root");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("keep.txt"), "keep").unwrap();
        let result = DeletePathTool::new()
            .execute(
                serde_json::json!({
                    "path": root.to_string_lossy(),
                    "recursive": true,
                    "dry_run": false
                }),
                mutation_context("delete_path", &root),
            )
            .await;

        assert!(!result.success);
        assert!(result.output.contains("write root"));
        assert_eq!(fs::read_to_string(root.join("keep.txt")).unwrap(), "keep");
    }

    #[test]
    fn replacement_leaves_destination_unchanged_when_backup_rename_fails() {
        let dir = tempdir().unwrap();
        let prepared = dir.path().join("prepared.txt");
        let destination = dir.path().join("missing.txt");
        fs::write(&prepared, "new").unwrap();

        let outcome = replace_existing_path(&prepared, &destination);

        assert!(matches!(outcome, ReplacementOutcome::Unchanged(_)));
        assert_eq!(fs::read_to_string(prepared).unwrap(), "new");
        assert!(!destination.exists());
    }

    #[test]
    fn replacement_restores_destination_when_commit_rename_fails() {
        let dir = tempdir().unwrap();
        let prepared = dir.path().join("missing-prepared.txt");
        let destination = dir.path().join("destination.txt");
        fs::write(&destination, "old").unwrap();

        let outcome = replace_existing_path(&prepared, &destination);

        assert!(matches!(outcome, ReplacementOutcome::Unchanged(_)));
        assert_eq!(fs::read_to_string(destination).unwrap(), "old");
    }

    #[test]
    fn replacement_reports_committed_mutation_when_cleanup_fails() {
        let dir = tempdir().unwrap();
        let prepared = dir.path().join("prepared.txt");
        let destination = dir.path().join("destination.txt");
        fs::write(&prepared, "new").unwrap();
        fs::write(&destination, "old").unwrap();

        let outcome = replace_existing_path_with(&prepared, &destination, |_| {
            Err(std::io::Error::other("injected cleanup failure"))
        });

        let ReplacementOutcome::Committed {
            cleanup_warning,
            retained_backup_path,
        } = outcome
        else {
            panic!("replacement must remain committed");
        };
        assert_eq!(fs::read_to_string(destination).unwrap(), "new");
        assert!(cleanup_warning.unwrap().contains("cleanup failed"));
        let backup = PathBuf::from(retained_backup_path.unwrap());
        assert_eq!(fs::read_to_string(backup).unwrap(), "old");
    }
}
