// Keep selected-file creation authority and a cooperative host revision on the shared tool path.
// This is not an OS sandbox: the caller must prevent untrusted external path replacement and writes.
use ai_agents::autonomy::HostRevisionSource;
use ai_agents::tools::{
    CommandTool, FileEditTool, FileListTool, FileReadTool, FileWriteTool, GlobTool, GrepTool, Tool,
    ToolCallClassification, ToolExecutionContext, ToolPolicyBindings, ToolRegistry, ToolResult,
    ToolSafetyMetadata,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[cfg(unix)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[cfg(not(unix))]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileIdentity;

#[cfg(unix)]
fn file_identity(path: &Path) -> std::io::Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(std::io::Error::other("selected file is not a regular file"));
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(not(unix))]
fn file_identity(_: &Path) -> std::io::Result<FileIdentity> {
    Err(std::io::Error::other(
        "new-file ownership is supported only on Unix",
    ))
}

struct State {
    revision: u64,
    in_flight: bool,
    poisoned: bool,
    created: BTreeMap<PathBuf, Option<FileIdentity>>,
    reads: BTreeMap<PathBuf, FileIdentity>,
}

/// The revision is valid only when the embedding host enforces its external-input ownership contract.
pub struct WorkspaceGuard {
    root: PathBuf,
    selected: Vec<PathBuf>,
    operations: tokio::sync::Mutex<()>,
    registry_taken: AtomicBool,
    state: Mutex<State>,
}

impl WorkspaceGuard {
    pub fn new(root: PathBuf, selected: Vec<PathBuf>, new_files: Vec<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            root,
            selected,
            operations: tokio::sync::Mutex::new(()),
            registry_taken: AtomicBool::new(false),
            state: Mutex::new(State {
                revision: 0,
                in_flight: false,
                poisoned: false,
                created: new_files.into_iter().map(|path| (path, None)).collect(),
                reads: BTreeMap::new(),
            }),
        })
    }

    // Resolve lexical aliases against the process cwd exactly once, without accepting an alternate symlink spelling.
    fn selected_path(&self, args: &Value) -> Result<PathBuf, String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or("missing selected file path")?;
        let path = Path::new(path);
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|_| "workspace cwd unavailable")?
                .join(path)
        };
        let mut normalized = PathBuf::new();
        for component in absolute.components() {
            match component {
                Component::CurDir => {}
                Component::ParentDir => {
                    normalized.pop();
                }
                other => normalized.push(other.as_os_str()),
            }
        }
        if !self.selected.contains(&normalized) {
            return Err("file path is not an explicitly selected entry".into());
        }
        let relative = normalized
            .strip_prefix(&self.root)
            .map_err(|_| "selected path escaped workspace")?;
        let mut current = self.root.clone();
        let parts: Vec<_> = relative.components().collect();
        for (index, part) in parts.iter().enumerate() {
            current.push(part.as_os_str());
            match std::fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err("symlinked selected paths are unsupported".into());
                }
                Ok(metadata) if index + 1 < parts.len() && !metadata.is_dir() => {
                    return Err("selected parent is not a directory".into());
                }
                Ok(_) => {}
                Err(error)
                    if index + 1 == parts.len() && error.kind() == std::io::ErrorKind::NotFound => {
                }
                Err(_) => return Err("selected path is unavailable".into()),
            }
        }
        Ok(normalized)
    }

    // A new entry can be read or edited only after this host observed its own successful create.
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
        if let Some(identity) = identity {
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
                        return Err(
                            "new file requires a successful file_read before editing".into()
                        );
                    }
                    if file_identity(path).ok() != Some(expected) {
                        if let Ok(mut state) = self.state.lock() {
                            state.poisoned = true;
                        }
                        return Err(
                            "created file identity changed; reconciliation is required".into()
                        );
                    }
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

    fn begin(self: &Arc<Self>) -> Result<MutationLease, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "workspace guard unavailable")?;
        if state.poisoned || state.in_flight || state.revision >= u64::MAX - 1 {
            return Err("workspace mutation or revision is unavailable".into());
        }
        state.revision += 1;
        state.in_flight = true;
        Ok(MutationLease {
            guard: self.clone(),
            finished: false,
        })
    }

    // A live guard cannot transfer unacknowledged effects to a separately built runtime.
    pub fn registry(self: &Arc<Self>) -> Result<ToolRegistry, String> {
        if self.registry_taken.swap(true, Ordering::AcqRel) {
            return Err("workspace guard is already bound to a runtime builder".into());
        }
        let mut registry = ToolRegistry::new();
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
            registry
                .register(Arc::new(GuardedTool {
                    inner: tool,
                    guard: self.clone(),
                    access,
                }))
                .expect("unique workspace tool");
        }
        registry
            .register(Arc::new(FileListTool::new()))
            .expect("unique file_list");
        registry
            .register(Arc::new(GlobTool::new()))
            .expect("unique glob");
        registry
            .register(Arc::new(GrepTool::new()))
            .expect("unique grep");
        registry
            .register(Arc::new(CommandTool::new(registry.command_runner_slot())))
            .expect("unique command");
        Ok(registry)
    }
}

impl HostRevisionSource for WorkspaceGuard {
    fn revision(&self) -> ai_agents::Result<String> {
        let state = self
            .state
            .lock()
            .map_err(|_| ai_agents::AgentError::Config("workspace revision unavailable".into()))?;
        if state.poisoned || state.in_flight {
            return Err(ai_agents::AgentError::Config(
                "workspace revision unavailable".into(),
            ));
        }
        Ok(state.revision.to_string())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
    Edit,
}

struct MutationLease {
    guard: Arc<WorkspaceGuard>,
    finished: bool,
}

impl MutationLease {
    // A failed or unverifiable effect poisons this source; a timeout/drop is never treated as a rollback.
    fn finish(mut self, path: &Path, result: &ToolResult) {
        // The wrapped built-in's structured no-match receipt proves that it did not enter the write path.
        let performed = (!result.has_uncertain_effect())
            .then(|| serde_json::from_str::<Value>(&result.output).ok())
            .flatten()
            .and_then(|value| value.get("mutation_performed").and_then(Value::as_bool));
        let performed = if !result.success && performed == Some(true) {
            None
        } else {
            performed
        };
        let is_new = self
            .guard
            .state
            .lock()
            .map(|state| state.created.contains_key(path))
            .unwrap_or(false);
        let identity = if performed == Some(true) && is_new {
            Some(file_identity(path))
        } else {
            None
        };
        if let Ok(mut state) = self.guard.state.lock() {
            match (performed, identity) {
                (Some(true), Some(Ok(identity))) => {
                    state.created.insert(path.to_path_buf(), Some(identity));
                    state.reads.remove(path);
                }
                (Some(true), Some(Err(_))) | (None, _) => state.poisoned = true,
                _ => {}
            }
            state.revision = state.revision.saturating_add(1);
            state.in_flight = false;
            self.finished = true;
        }
    }
}

impl Drop for MutationLease {
    fn drop(&mut self) {
        if !self.finished
            && let Ok(mut state) = self.guard.state.lock()
        {
            state.poisoned = true;
            state.in_flight = false;
        }
    }
}

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

    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        // Never wait inside an already admitted tool body for another selected-file operation.
        let Ok(_operation) = self.guard.operations.try_lock() else {
            return ToolResult::error("another selected-file operation is in progress");
        };
        let path = match self.guard.selected_path(&args) {
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
        // The built-in rejects identical strings before reading or writing the selected file.
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (PathBuf, PathBuf, Arc<WorkspaceGuard>) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "workspace-guard-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("new.rs");
        let guard = WorkspaceGuard::new(root.clone(), vec![path.clone()], vec![path.clone()]);
        (root, path, guard)
    }

    #[test]
    fn external_creation_does_not_become_host_owned() {
        let (root, path, guard) = fixture();
        assert!(guard.authorize(&path, Access::Read).is_err());
        std::fs::write(&path, "user content").unwrap();
        assert!(guard.authorize(&path, Access::Read).is_err());
        assert!(guard.authorize(&path, Access::Edit).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "user content");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn tool_wrapper_rejects_external_new_file_before_read_or_edit() {
        let (root, path, guard) = fixture();
        let registry = guard.registry().unwrap();
        assert!(guard.registry().is_err());
        std::fs::write(&path, "private user content").unwrap();
        let read = registry
            .get("file_read")
            .unwrap()
            .execute(
                serde_json::json!({"path": path}),
                ToolExecutionContext::test("file_read"),
            )
            .await;
        assert!(!read.success);
        assert!(!read.output.contains("private user content"));
        let edit = registry
            .get("file_edit")
            .unwrap()
            .execute(
                serde_json::json!({"path": path, "old_text":"private", "new_text":"changed"}),
                ToolExecutionContext::test("file_edit"),
            )
            .await;
        assert!(!edit.success);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "private user content"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn created_identity_is_required_for_later_edits_and_replacement_poisoned() {
        let (root, path, guard) = fixture();
        let lease = guard.begin().unwrap();
        assert!(guard.revision().is_err());
        std::fs::write(&path, "host content").unwrap();
        lease.finish(&path, &ToolResult::ok(r#"{"mutation_performed":true}"#));
        assert_eq!(guard.revision().unwrap(), "2");
        assert!(guard.authorize(&path, Access::Read).is_ok());
        assert!(guard.authorize(&path, Access::Edit).is_err());
        guard.acknowledge_read(&path).unwrap();
        assert!(guard.authorize(&path, Access::Edit).is_ok());
        let replacement = root.join("replacement.rs");
        std::fs::write(&replacement, "replacement").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert!(guard.authorize(&path, Access::Edit).is_err());
        assert!(guard.revision().is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replacement");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn structured_no_effect_failure_does_not_poison_the_revision() {
        let (root, path, guard) = fixture();
        let lease = guard.begin().unwrap();
        lease.finish(&path, &ToolResult::error(r#"{"mutation_performed":false}"#));
        assert_eq!(guard.revision().unwrap(), "2");
        assert!(guard.begin().is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dropped_mutation_poisoning_does_not_claim_cleanup() {
        let (root, _path, guard) = fixture();
        let lease = guard.begin().unwrap();
        assert!(guard.revision().is_err());
        drop(lease);
        assert!(guard.revision().is_err());
        assert!(guard.begin().is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
