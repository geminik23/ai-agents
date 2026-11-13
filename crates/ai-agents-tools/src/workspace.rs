//! Host-owned protection for file tools that work inside one workspace root.
//!
//! The guard wraps the built-in `file_read`, `file_write` and `file_edit` tools on the shared registry path.
//! It confines their targets to the root or to explicitly selected entries, tracks files that it created, and maintains a cooperative revision.
//! It is not an OS sandbox, a file watcher or a lock: the host must prevent untrusted external writers and path replacement for the session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::security::path::normalize_lexically;
use crate::{
    FileEditTool, FileReadTool, FileWriteTool, Tool, ToolCallClassification, ToolError,
    ToolExecutionContext, ToolPolicyBindings, ToolRegistry, ToolResult, ToolSafetyMetadata,
};

// Mutation tools that bypass the guard must not share a registry with it.
const UNGUARDED_MUTATION_TOOLS: [&str; 5] =
    ["file", "patch", "copy_path", "move_path", "delete_path"];
const GUARDED_TOOLS: [&str; 3] = ["file_read", "file_write", "file_edit"];

/// Which paths under the workspace root the guarded file tools may address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceScope {
    /// Any path under the root; the configured tool policy still decides which reads and writes are allowed or need approval.
    Root,
    /// Only these absolute, lexically normalized paths under the root.
    Selected(Vec<PathBuf>),
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[cfg(not(unix))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileIdentity;

// Identity is the device and inode of a regular file, so a replaced path is distinguishable from the created one.
#[cfg(unix)]
fn file_identity(path: &Path) -> std::io::Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(std::io::Error::other(
            "workspace file is not a regular file",
        ));
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

// Without a portable identity, created-file ownership cannot be established.
#[cfg(not(unix))]
fn file_identity(_: &Path) -> std::io::Result<FileIdentity> {
    Err(std::io::Error::other(
        "workspace new-file ownership is supported only on Unix",
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
    Edit,
}

struct State {
    revision: u64,
    in_flight: bool,
    poisoned: bool,
    // A declared new path maps to None until this guard observes its own successful create.
    created: BTreeMap<PathBuf, Option<FileIdentity>>,
    reads: BTreeMap<PathBuf, FileIdentity>,
}

/// Confines built-in file tools to one workspace and maintains a cooperative mutation revision.
///
/// Every guarded mutation makes the revision unavailable while it runs.
/// A performed or uncertain effect advances it to a never-reused value, while a failure that the tool reports without uncertainty leaves it unchanged.
/// An uncertain effect, a success without a structured receipt, a dropped mutation or a replaced created file poisons the guard until the host reconciles outside this session.
/// The revision only reflects mutations made through the guarded tools; it is valid as freshness evidence only when the host controls all other writers.
pub struct WorkspaceGuard {
    root: PathBuf,
    scope: WorkspaceScope,
    operations: tokio::sync::Mutex<()>,
    installed: AtomicBool,
    state: Mutex<State>,
}

impl WorkspaceGuard {
    /// Validates the canonical root, the scope and any declared new files before any tool can run.
    ///
    /// `root` must be an absolute canonical directory, so a symlinked spelling cannot alias another tree.
    /// Each declared new file must be absent, inside the scope and below existing non-symlinked directories.
    pub fn new(
        root: impl Into<PathBuf>,
        scope: WorkspaceScope,
        new_files: Vec<PathBuf>,
    ) -> Result<Arc<Self>, ToolError> {
        let root = root.into();
        let invalid = |message: &str| ToolError::InvalidArguments(message.to_string());
        if !root.is_absolute() {
            return Err(invalid("workspace root must be absolute"));
        }
        let canonical =
            std::fs::canonicalize(&root).map_err(|_| invalid("workspace root is unavailable"))?;
        if canonical != root || !canonical.is_dir() {
            return Err(invalid("workspace root must be a canonical directory"));
        }
        let within_root = |path: &Path| {
            path.is_absolute()
                && normalize_lexically(path) == path
                && path != root
                && path.starts_with(&root)
        };
        if let WorkspaceScope::Selected(entries) = &scope
            && !entries.iter().all(|entry| within_root(entry))
        {
            return Err(invalid(
                "selected entries must be normalized absolute paths under the root",
            ));
        }
        if !new_files.is_empty() && !cfg!(unix) {
            return Err(invalid(
                "workspace new-file ownership is supported only on Unix",
            ));
        }
        for path in &new_files {
            if !within_root(path) {
                return Err(invalid(
                    "new files must be normalized absolute paths under the root",
                ));
            }
            if let WorkspaceScope::Selected(entries) = &scope
                && !entries.contains(path)
            {
                return Err(invalid("new files must also be selected entries"));
            }
            if !matches!(std::fs::symlink_metadata(path), Err(error) if error.kind() == std::io::ErrorKind::NotFound)
            {
                return Err(invalid(
                    "new files must not exist when the guard is created",
                ));
            }
            let parent = path.parent().unwrap_or(&root);
            let mut current = root.clone();
            for part in parent
                .strip_prefix(&root)
                .unwrap_or(Path::new(""))
                .components()
            {
                current.push(part.as_os_str());
                match std::fs::symlink_metadata(&current) {
                    Ok(metadata) if metadata.is_dir() => {}
                    _ => {
                        return Err(invalid(
                            "new files require existing non-symlinked parent directories",
                        ));
                    }
                }
            }
        }
        Ok(Arc::new(Self {
            root,
            scope,
            operations: tokio::sync::Mutex::new(()),
            installed: AtomicBool::new(false),
            state: Mutex::new(State {
                revision: 0,
                in_flight: false,
                poisoned: false,
                created: new_files.into_iter().map(|path| (path, None)).collect(),
                reads: BTreeMap::new(),
            }),
        }))
    }

    /// Registers guarded `file_read`, `file_write` and `file_edit` sharing the registry's file version store.
    ///
    /// A guard binds to exactly one registry so unacknowledged effects cannot transfer to a separately built runtime.
    /// Installation fails when the registry already has these tools or an unguarded file mutation tool.
    /// Tools registered after installation are not covered, so the host must not add other mutation tools for the same root.
    pub fn install(self: &Arc<Self>, registry: &mut ToolRegistry) -> Result<(), ToolError> {
        if let Some(id) = GUARDED_TOOLS
            .iter()
            .chain(UNGUARDED_MUTATION_TOOLS.iter())
            .find(|id| registry.get(id).is_some())
        {
            return Err(ToolError::AlreadyRegistered(format!(
                "{id} cannot share a registry with a workspace guard"
            )));
        }
        if self.installed.swap(true, Ordering::AcqRel) {
            return Err(ToolError::Duplicate(
                "workspace guard is already installed in a registry".into(),
            ));
        }
        let versions = registry.file_version_store();
        for (tool, access) in [
            (
                Arc::new(FileReadTool::with_version_store(versions.clone())) as Arc<dyn Tool>,
                Access::Read,
            ),
            (
                Arc::new(FileWriteTool::with_version_store(versions.clone())) as Arc<dyn Tool>,
                Access::Write,
            ),
            (
                Arc::new(FileEditTool::with_version_store(versions)) as Arc<dyn Tool>,
                Access::Edit,
            ),
        ] {
            registry.register(Arc::new(GuardedTool {
                inner: tool,
                guard: self.clone(),
                access,
            }))?;
        }
        Ok(())
    }

    /// Returns the current revision, or `None` while a mutation is running or after the guard was poisoned.
    pub fn revision(&self) -> Option<String> {
        let state = self.state.lock().ok()?;
        (!state.poisoned && !state.in_flight).then(|| state.revision.to_string())
    }

    /// Reports whether an unverifiable effect blocks further guarded work until the host reconciles the workspace.
    pub fn requires_reconciliation(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.poisoned)
            .unwrap_or(true)
    }

    // Resolve lexical aliases against the process cwd like the built-in tools, then reject symlinks and paths outside the scope.
    fn target_path(&self, args: &Value) -> Result<PathBuf, String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or("missing workspace file path")?;
        let path = Path::new(path);
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|_| "workspace cwd unavailable")?
                .join(path)
        };
        let normalized = normalize_lexically(&absolute);
        let missing_suffix_allowed = match &self.scope {
            WorkspaceScope::Root => true,
            WorkspaceScope::Selected(entries) => {
                if !entries.contains(&normalized) {
                    return Err("file path is not an explicitly selected entry".into());
                }
                false
            }
        };
        let relative = normalized
            .strip_prefix(&self.root)
            .map_err(|_| "file path is outside the workspace root")?;
        let parts: Vec<_> = relative.components().collect();
        if parts.is_empty() {
            return Err("the workspace root is not a file target".into());
        }
        let mut current = self.root.clone();
        for (index, part) in parts.iter().enumerate() {
            current.push(part.as_os_str());
            let last = index + 1 == parts.len();
            match std::fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err("symlinked workspace paths are unsupported".into());
                }
                Ok(metadata) if !last && !metadata.is_dir() => {
                    return Err("workspace parent is not a directory".into());
                }
                Ok(_) => {}
                // Root scope may name a new nested path; the tool policy still decides whether parents can be created.
                Err(error)
                    if error.kind() == std::io::ErrorKind::NotFound
                        && (last || missing_suffix_allowed) =>
                {
                    break;
                }
                Err(_) => return Err("workspace path is unavailable".into()),
            }
        }
        Ok(normalized)
    }

    // A declared or guard-created file can be read or edited only while it is still the identity this guard created.
    fn authorize(&self, path: &Path, operation: Access) -> Result<(), String> {
        let (identity, read) = {
            let state = self
                .state
                .lock()
                .map_err(|_| "workspace guard unavailable")?;
            if state.poisoned {
                return Err("workspace effects require reconciliation".into());
            }
            (
                state.created.get(path).copied(),
                state.reads.get(path).copied(),
            )
        };
        let Some(identity) = identity else {
            if operation == Access::Write && !cfg!(unix) && !path.exists() {
                return Err("workspace new-file ownership is supported only on Unix".into());
            }
            return Ok(());
        };
        match (operation, identity) {
            (Access::Write, None) => {}
            (Access::Write, Some(_)) => {
                return Err("new file was already created; use a reviewed edit".into());
            }
            (Access::Read | Access::Edit, None) => {
                return Err("new file has not been created by this host".into());
            }
            (Access::Read | Access::Edit, Some(expected)) => {
                if operation == Access::Edit && read != Some(expected) {
                    return Err("new file requires a successful file_read before editing".into());
                }
                if file_identity(path).ok() != Some(expected) {
                    if let Ok(mut state) = self.state.lock() {
                        state.poisoned = true;
                    }
                    return Err("created file identity changed; reconciliation is required".into());
                }
            }
        }
        Ok(())
    }

    // Only an actual successful read of the current created identity authorizes its next edit.
    fn acknowledge_read(&self, path: &Path) -> Result<(), String> {
        let expected = {
            let state = self
                .state
                .lock()
                .map_err(|_| "workspace guard unavailable")?;
            state.created.get(path).copied().flatten()
        };
        if let Some(expected) = expected {
            let observed = file_identity(path).ok();
            let mut state = self
                .state
                .lock()
                .map_err(|_| "workspace guard unavailable")?;
            if observed != Some(expected)
                || state.created.get(path).copied().flatten() != Some(expected)
            {
                state.poisoned = true;
                return Err(
                    "created file changed during the read; reconciliation is required".into(),
                );
            }
            state.reads.insert(path.to_path_buf(), expected);
        }
        Ok(())
    }

    // The revision advances before the body runs so no observer can pair the old token with a partially applied effect.
    fn begin(self: &Arc<Self>) -> Result<MutationLease, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "workspace guard unavailable")?;
        if state.poisoned || state.in_flight || state.revision >= u64::MAX - 1 {
            return Err("workspace mutation or revision is unavailable".into());
        }
        let previous = state.revision;
        state.revision += 1;
        state.in_flight = true;
        Ok(MutationLease {
            guard: self.clone(),
            previous,
            finished: false,
        })
    }
}

struct MutationLease {
    guard: Arc<WorkspaceGuard>,
    // Restored only when the call provably changed nothing.
    previous: u64,
    finished: bool,
}

/// How one guarded call settled under the shared tool-effect contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Settlement {
    // A plain failure or a receipt with `mutation_performed: false` changed nothing.
    NoEffect,
    Performed,
    Unknown,
}

// Built-in file tools report possible residue through `task_effect_uncertain`, which is also what the shared executor retains.
// A plain failure therefore proves no change, while a success must still carry a structured receipt to be trusted.
fn settle(result: &ToolResult) -> Settlement {
    if result.has_uncertain_effect() {
        return Settlement::Unknown;
    }
    let performed = serde_json::from_str::<Value>(&result.output)
        .ok()
        .and_then(|value| value.get("mutation_performed").and_then(Value::as_bool));
    match (result.success, performed) {
        // A failure that still claims a mutation contradicts the contract and is never treated as settled.
        (false, Some(true)) => Settlement::Unknown,
        (false, _) | (true, Some(false)) => Settlement::NoEffect,
        (true, Some(true)) => Settlement::Performed,
        (true, None) => Settlement::Unknown,
    }
}

impl MutationLease {
    // A no-effect call restores the pre-call revision because the in-flight value was never observable and the workspace is unchanged.
    // Any performed or unknown effect keeps the advanced, never-reused revision, and an unknown or unobservable creation poisons the guard.
    fn finish(mut self, path: &Path, result: &ToolResult) {
        let settlement = settle(result);
        let reported_create = result.success
            && serde_json::from_str::<Value>(&result.output)
                .ok()
                .and_then(|value| value.get("created").and_then(Value::as_bool))
                == Some(true);
        let tracked = self
            .guard
            .state
            .lock()
            .map(|state| state.created.contains_key(path))
            .unwrap_or(false);
        let identity = (settlement == Settlement::Performed && (tracked || reported_create))
            .then(|| file_identity(path));
        if let Ok(mut state) = self.guard.state.lock() {
            match (settlement, identity) {
                (Settlement::NoEffect, _) => state.revision = self.previous,
                (Settlement::Performed, Some(Ok(identity))) => {
                    state.created.insert(path.to_path_buf(), Some(identity));
                    state.reads.remove(path);
                    state.revision = state.revision.saturating_add(1);
                }
                (Settlement::Performed, None) => {
                    state.revision = state.revision.saturating_add(1);
                }
                (Settlement::Performed, Some(Err(_))) | (Settlement::Unknown, _) => {
                    state.poisoned = true;
                    state.revision = state.revision.saturating_add(1);
                }
            }
            state.in_flight = false;
            self.finished = true;
        }
    }
}

impl Drop for MutationLease {
    // Cancellation, timeout or a panic never proves that the write did not happen or was rolled back.
    fn drop(&mut self) {
        if !self.finished
            && let Ok(mut state) = self.guard.state.lock()
        {
            state.poisoned = true;
            state.in_flight = false;
        }
    }
}

// Delegates identity, schema, policy bindings and classification so the shared executor applies the unchanged policy before the guard runs.
struct GuardedTool {
    inner: Arc<dyn Tool>,
    guard: Arc<WorkspaceGuard>,
    access: Access,
}

#[async_trait]
impl Tool for GuardedTool {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }
    fn safety_metadata(&self) -> ToolSafetyMetadata {
        self.inner.safety_metadata()
    }
    fn policy_bindings(&self) -> ToolPolicyBindings {
        self.inner.policy_bindings()
    }
    fn classify_call(&self, args: &Value) -> ToolCallClassification {
        self.inner.classify_call(args)
    }
    fn declared_write_footprint(
        &self,
        args: &Value,
        ctx: &ToolExecutionContext,
        max_targets: usize,
    ) -> ai_agents_core::Result<Option<ai_agents_core::autonomy::ToolWriteFootprint>> {
        self.inner.declared_write_footprint(args, ctx, max_targets)
    }

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        // Never wait inside an already admitted tool body for another workspace operation.
        let Ok(_operation) = self.guard.operations.try_lock() else {
            return ToolResult::error("another workspace file operation is in progress");
        };
        let path = match self.guard.target_path(&args) {
            Ok(path) => path,
            Err(error) => return ToolResult::error(error),
        };
        if let Err(error) = self.guard.authorize(&path, self.access) {
            return ToolResult::error(error);
        }
        if self.access == Access::Read {
            let result = self.inner.execute(args, ctx).await;
            if result.success
                && let Err(error) = self.guard.acknowledge_read(&path)
            {
                return ToolResult::error(error);
            }
            return result;
        }
        // The built-in edit rejects identical strings before touching the file, so no revision change is needed.
        if self.access == Access::Edit
            && let (Some(old), Some(new)) = (
                args.get("old_text").and_then(Value::as_str),
                args.get("new_text").and_then(Value::as_str),
            )
            && old == new
        {
            return self.inner.execute(args, ctx).await;
        }
        let lease = match self.guard.begin() {
            Ok(lease) => lease,
            Err(error) => return ToolResult::error(error),
        };
        let result = self.inner.execute(args, ctx).await;
        lease.finish(&path, &result);
        result
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::PatchTool;
    use serde_json::json;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let root = std::fs::canonicalize(dir.path()).unwrap();
            Self { _dir: dir, root }
        }
    }

    fn installed(guard: &Arc<WorkspaceGuard>) -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        guard.install(&mut registry).unwrap();
        registry
    }

    async fn call(registry: &ToolRegistry, tool: &str, args: Value) -> ToolResult {
        registry
            .get(tool)
            .unwrap()
            .execute(args, ToolExecutionContext::test(tool))
            .await
    }

    // Built-in file tools read the policy snapshot from the context; tests grant writes under the root directly.
    async fn write(registry: &ToolRegistry, root: &Path, args: Value) -> ToolResult {
        let mut ctx = ToolExecutionContext::test("file_write");
        ctx.policy_snapshot = json!({"write_paths":[root],"overwrite_existing":true,"create_parent_dirs":true,"allow_without_confirmation":true});
        registry.get("file_write").unwrap().execute(args, ctx).await
    }

    #[test]
    fn construction_rejects_aliases_outside_entries_and_existing_new_files() {
        let fixture = Fixture::new();
        let root = &fixture.root;
        std::fs::write(root.join("existing.rs"), "").unwrap();
        std::os::unix::fs::symlink(root, root.join("alias")).unwrap();
        assert!(WorkspaceGuard::new("relative", WorkspaceScope::Root, vec![]).is_err());
        assert!(WorkspaceGuard::new(root.join("alias"), WorkspaceScope::Root, vec![]).is_err());
        assert!(
            WorkspaceGuard::new(
                root.clone(),
                WorkspaceScope::Selected(vec![root.parent().unwrap().join("outside.rs")]),
                vec![]
            )
            .is_err()
        );
        assert!(
            WorkspaceGuard::new(
                root.clone(),
                WorkspaceScope::Root,
                vec![root.join("existing.rs")]
            )
            .is_err()
        );
        assert!(
            WorkspaceGuard::new(
                root.clone(),
                WorkspaceScope::Root,
                vec![root.join("missing").join("new.rs")]
            )
            .is_err()
        );
        let unselected_new = WorkspaceGuard::new(
            root.clone(),
            WorkspaceScope::Selected(vec![root.join("a.rs")]),
            vec![root.join("b.rs")],
        );
        assert!(unselected_new.is_err());
    }

    #[test]
    fn install_binds_one_registry_and_refuses_unguarded_mutation_tools() {
        let fixture = Fixture::new();
        let guard =
            WorkspaceGuard::new(fixture.root.clone(), WorkspaceScope::Root, vec![]).unwrap();
        let mut unguarded = ToolRegistry::new();
        unguarded.register(Arc::new(PatchTool::new())).unwrap();
        assert!(guard.install(&mut unguarded).is_err());
        let _registry = installed(&guard);
        assert!(guard.install(&mut ToolRegistry::new()).is_err());
    }

    #[tokio::test]
    async fn root_scope_rejects_outside_and_symlinked_targets_before_execution() {
        let fixture = Fixture::new();
        let root = &fixture.root;
        let outside = tempfile::tempdir().unwrap();
        let outside_file = std::fs::canonicalize(outside.path())
            .unwrap()
            .join("user.txt");
        std::fs::write(&outside_file, "user").unwrap();
        std::os::unix::fs::symlink(outside.path(), root.join("link")).unwrap();
        let guard = WorkspaceGuard::new(root.clone(), WorkspaceScope::Root, vec![]).unwrap();
        let registry = installed(&guard);
        for path in [outside_file.clone(), root.join("link").join("user.txt")] {
            let result = write(
                &registry,
                outside.path(),
                json!({"path":path,"content":"agent","overwrite":true}),
            )
            .await;
            assert!(!result.success);
            assert!(
                result.output.contains("outside the workspace root")
                    || result.output.contains("symlinked workspace paths"),
                "{}",
                result.output
            );
            let read = call(&registry, "file_read", json!({"path":path})).await;
            assert!(!read.success);
            assert!(!read.output.contains("user"));
        }
        assert_eq!(std::fs::read_to_string(&outside_file).unwrap(), "user");
        assert_eq!(guard.revision().as_deref(), Some("0"));
    }

    #[tokio::test]
    async fn root_scope_tracks_created_files_and_poisons_on_replacement() {
        let fixture = Fixture::new();
        let root = &fixture.root;
        let guard = WorkspaceGuard::new(root.clone(), WorkspaceScope::Root, vec![]).unwrap();
        let registry = installed(&guard);
        let path = root.join("src").join("new.rs");
        let created = write(
            &registry,
            root,
            json!({"path":path,"content":"one","create_parent_dirs":true}),
        )
        .await;
        assert!(created.success, "{}", created.output);
        assert_eq!(guard.revision().as_deref(), Some("2"));
        let rewrite = write(
            &registry,
            root,
            json!({"path":path,"content":"two","overwrite":true}),
        )
        .await;
        assert!(
            rewrite.output.contains("already created"),
            "{}",
            rewrite.output
        );
        let edit = call(
            &registry,
            "file_edit",
            json!({"path":path,"old_text":"one","new_text":"two"}),
        )
        .await;
        assert!(
            edit.output.contains("requires a successful file_read"),
            "{}",
            edit.output
        );
        assert_eq!(guard.revision().as_deref(), Some("2"));
        let read = call(&registry, "file_read", json!({"path":path})).await;
        assert!(read.success, "{}", read.output);
        let replacement = root.join("replacement.rs");
        std::fs::write(&replacement, "replacement").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        let read = call(&registry, "file_read", json!({"path":path})).await;
        assert!(read.output.contains("identity changed"), "{}", read.output);
        assert!(guard.requires_reconciliation());
        assert_eq!(guard.revision(), None);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replacement");
    }

    #[tokio::test]
    async fn declared_new_file_created_externally_is_not_host_owned() {
        let fixture = Fixture::new();
        let path = fixture.root.join("new.rs");
        let guard = WorkspaceGuard::new(
            fixture.root.clone(),
            WorkspaceScope::Selected(vec![path.clone()]),
            vec![path.clone()],
        )
        .unwrap();
        let registry = installed(&guard);
        std::fs::write(&path, "private user content").unwrap();
        let read = call(&registry, "file_read", json!({"path":path})).await;
        assert!(
            read.output.contains("not been created by this host"),
            "{}",
            read.output
        );
        assert!(!read.output.contains("private user content"));
        let edit = call(
            &registry,
            "file_edit",
            json!({"path":path,"old_text":"private","new_text":"changed"}),
        )
        .await;
        assert!(!edit.success);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "private user content"
        );
    }

    #[tokio::test]
    async fn selected_scope_rejects_unselected_existing_files() {
        let fixture = Fixture::new();
        let selected = fixture.root.join("selected.rs");
        let other = fixture.root.join("other.rs");
        std::fs::write(&selected, "a").unwrap();
        std::fs::write(&other, "user").unwrap();
        let guard = WorkspaceGuard::new(
            fixture.root.clone(),
            WorkspaceScope::Selected(vec![selected]),
            vec![],
        )
        .unwrap();
        let registry = installed(&guard);
        let result = call(&registry, "file_read", json!({"path":other})).await;
        assert!(
            result.output.contains("not an explicitly selected entry"),
            "{}",
            result.output
        );
    }

    #[test]
    fn revision_is_unavailable_in_flight_and_only_advances_for_effects() {
        let fixture = Fixture::new();
        let path = fixture.root.join("x");
        let guard =
            WorkspaceGuard::new(fixture.root.clone(), WorkspaceScope::Root, vec![]).unwrap();
        let lease = guard.begin().unwrap();
        assert_eq!(guard.revision(), None);
        assert!(guard.begin().is_err());
        lease.finish(&path, &ToolResult::error("overwrite must be true"));
        assert_eq!(guard.revision().as_deref(), Some("0"));
        guard
            .begin()
            .unwrap()
            .finish(&path, &ToolResult::ok(r#"{"mutation_performed":false}"#));
        assert_eq!(guard.revision().as_deref(), Some("0"));
        guard
            .begin()
            .unwrap()
            .finish(&path, &ToolResult::ok(r#"{"mutation_performed":true}"#));
        assert_eq!(guard.revision().as_deref(), Some("2"));
        guard
            .begin()
            .unwrap()
            .finish(&path, &ToolResult::error(r#"{"mutation_performed":false}"#));
        assert_eq!(guard.revision().as_deref(), Some("2"));
        assert!(!guard.requires_reconciliation());
    }

    #[test]
    fn uncertain_unreceipted_and_dropped_mutations_poison_without_claiming_cleanup() {
        let fixture = Fixture::new();
        let path = fixture.root.join("x");
        for result in [
            ToolResult::error_with_uncertain_effect("write failed; partial data may remain"),
            ToolResult::ok("written"),
            ToolResult::error(r#"{"mutation_performed":true}"#),
        ] {
            let guard =
                WorkspaceGuard::new(fixture.root.clone(), WorkspaceScope::Root, vec![]).unwrap();
            guard.begin().unwrap().finish(&path, &result);
            assert!(guard.requires_reconciliation());
            assert_eq!(guard.revision(), None);
            assert!(guard.begin().is_err());
        }
        let guard =
            WorkspaceGuard::new(fixture.root.clone(), WorkspaceScope::Root, vec![]).unwrap();
        drop(guard.begin().unwrap());
        assert!(guard.requires_reconciliation());
        assert!(guard.authorize(&path, Access::Write).is_err());
    }

    #[tokio::test]
    async fn ordinary_tool_rejections_keep_the_revision_and_allow_recovery() {
        let fixture = Fixture::new();
        let root = &fixture.root;
        let existing = root.join("lib.rs");
        std::fs::write(&existing, "old").unwrap();
        let guard = WorkspaceGuard::new(root.clone(), WorkspaceScope::Root, vec![]).unwrap();
        let registry = installed(&guard);

        let missing_overwrite =
            write(&registry, root, json!({"path":existing,"content":"new"})).await;
        assert!(
            missing_overwrite.output.contains("overwrite must be true"),
            "{}",
            missing_overwrite.output
        );
        let mut limited = ToolExecutionContext::test("file_write");
        limited.policy_snapshot = json!({"write_paths":[root],"create_parent_dirs":true,"allow_without_confirmation":true});
        limited.limits.max_changed_lines = Some(1);
        let new_file = root.join("src").join("new.rs");
        let over_limit = registry
            .get("file_write")
            .unwrap()
            .execute(
                json!({"path":new_file,"content":"a\nb\nc\n","create_parent_dirs":true}),
                limited,
            )
            .await;
        assert!(
            over_limit.output.contains("changed-line cap"),
            "{}",
            over_limit.output
        );
        assert!(!root.join("src").exists());
        assert_eq!(guard.revision().as_deref(), Some("0"));
        assert!(!guard.requires_reconciliation());

        let created = write(
            &registry,
            root,
            json!({"path":new_file,"content":"a","create_parent_dirs":true}),
        )
        .await;
        assert!(created.success, "{}", created.output);
        assert_eq!(guard.revision().as_deref(), Some("2"));
    }

    #[tokio::test]
    async fn concurrent_operation_is_rejected_without_waiting() {
        let fixture = Fixture::new();
        let path = fixture.root.join("a.rs");
        std::fs::write(&path, "a").unwrap();
        let guard =
            WorkspaceGuard::new(fixture.root.clone(), WorkspaceScope::Root, vec![]).unwrap();
        let registry = installed(&guard);
        let _held = guard.operations.try_lock().unwrap();
        let result = call(&registry, "file_read", json!({"path":path})).await;
        assert!(!result.success);
        assert!(result.output.contains("in progress"));
    }
}
