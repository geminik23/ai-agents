use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use ai_agents_core::{AgentError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;

use crate::ContextProvider;

const MAX_TARGETS: usize = 128;
const MAX_CANDIDATES: usize = 256;
const MAX_FILES: usize = 32;
const MAX_FILE_BYTES: u64 = 64 * 1024;
const MAX_TOTAL_BYTES: usize = 256 * 1024;

/// A host-owned instruction source limited to selected workspace paths and their ancestor directories.
/// The host must prevent concurrent untrusted path replacement; this source is not a filesystem sandbox.
pub struct ScopedInstructionFiles {
    root: PathBuf,
    directories: Vec<PathBuf>,
    names: Vec<String>,
}

impl ScopedInstructionFiles {
    /// Binds an existing root, relative target paths and literal filenames without walking unrelated directories.
    pub fn new(root: impl AsRef<Path>, targets: &[PathBuf], names: &[String]) -> Result<Self> {
        let root = std::fs::canonicalize(root.as_ref())
            .map_err(|_| AgentError::Config("instruction root is unavailable".into()))?;
        if !root.is_dir() || targets.is_empty() || targets.len() > MAX_TARGETS || names.is_empty() {
            return Err(AgentError::Config("invalid instruction scope".into()));
        }
        let mut unique_names = BTreeSet::new();
        for name in names {
            let path = Path::new(name);
            if !matches!(
                (path.components().next(), path.components().nth(1)),
                (Some(Component::Normal(_)), None)
            ) || !unique_names.insert(name.clone())
            {
                return Err(AgentError::Config(
                    "invalid or duplicate instruction filename".into(),
                ));
            }
        }
        let mut directories = BTreeSet::new();
        directories.insert(PathBuf::new());
        for target in targets {
            if target.as_os_str().is_empty()
                || target.is_absolute()
                || target
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(AgentError::Config(
                    "instruction targets must be relative normal paths".into(),
                ));
            }
            let parent = target.parent().unwrap_or_else(|| Path::new(""));
            let mut current = PathBuf::new();
            for component in parent.components() {
                current.push(component.as_os_str());
                directories.insert(current.clone());
                if directories.len().saturating_mul(unique_names.len()) > MAX_CANDIDATES {
                    return Err(AgentError::Config(
                        "instruction discovery exceeds the candidate limit".into(),
                    ));
                }
            }
        }
        if directories.len().saturating_mul(unique_names.len()) > MAX_CANDIDATES {
            return Err(AgentError::Config(
                "instruction discovery exceeds the candidate limit".into(),
            ));
        }
        let mut directories: Vec<_> = directories.into_iter().collect();
        directories.sort_by(|a, b| {
            a.components()
                .count()
                .cmp(&b.components().count())
                .then_with(|| a.cmp(b))
        });
        Ok(Self {
            root,
            directories,
            names: unique_names.into_iter().collect(),
        })
    }

    // Revalidates every path component before each request; an inaccessible ancestor is not an absent instruction.
    async fn checked_directory(&self, relative: &Path) -> Result<PathBuf> {
        let mut path = self.root.clone();
        let root = tokio::fs::symlink_metadata(&path)
            .await
            .map_err(|_| AgentError::Config("instruction root is unavailable".into()))?;
        if !root.file_type().is_dir() || root.file_type().is_symlink() {
            return Err(AgentError::Config(
                "instruction root is not an available directory".into(),
            ));
        }
        for part in relative.components() {
            path.push(part.as_os_str());
            let metadata = tokio::fs::symlink_metadata(&path).await.map_err(|_| {
                AgentError::Config("instruction scope directory is unavailable".into())
            })?;
            if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
                return Err(AgentError::Config(
                    "instruction scope contains an unsupported directory".into(),
                ));
            }
        }
        Ok(path)
    }

    // Rebuilds the entire projection before publication; read errors never publish a partial replacement.
    async fn load(&self) -> Result<String> {
        let mut files = Vec::new();
        let mut total = 0usize;
        for directory in &self.directories {
            for name in &self.names {
                let parent = self.checked_directory(directory).await?;
                let path = parent.join(name);
                let metadata = match tokio::fs::symlink_metadata(&path).await {
                    Ok(metadata) => metadata,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(_) => {
                        return Err(AgentError::Config("instruction file is unavailable".into()));
                    }
                };
                if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                    return Err(AgentError::Config(
                        "instruction file must be a regular non-symlink file".into(),
                    ));
                }
                if files.len() >= MAX_FILES || metadata.len() > MAX_FILE_BYTES {
                    return Err(AgentError::Config("instruction file limit exceeded".into()));
                }
                let file = tokio::fs::File::open(&path)
                    .await
                    .map_err(|_| AgentError::Config("instruction file is unavailable".into()))?;
                let opened = file
                    .metadata()
                    .await
                    .map_err(|_| AgentError::Config("instruction file is unavailable".into()))?;
                if !opened.is_file() {
                    return Err(AgentError::Config("instruction file is not regular".into()));
                }
                let mut bytes = Vec::new();
                file.take(MAX_FILE_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .await
                    .map_err(|_| AgentError::Config("instruction file could not be read".into()))?;
                total = total.saturating_add(bytes.len());
                if bytes.len() as u64 > MAX_FILE_BYTES || total > MAX_TOTAL_BYTES {
                    return Err(AgentError::Config("instruction byte limit exceeded".into()));
                }
                let content = String::from_utf8(bytes)
                    .map_err(|_| AgentError::Config("instruction file is not UTF-8".into()))?;
                let scope = directory
                    .to_str()
                    .ok_or_else(|| AgentError::Config("instruction scope is not UTF-8".into()))?;
                files.push(json!({"scope": scope, "file": name, "content": content}));
            }
        }
        if files.is_empty() {
            return Ok(String::new());
        }
        let entries = serde_json::to_string(&files)?;
        Ok(format!(
            "Each entry applies only to its scope and descendants; an empty scope is the workspace root. Do not apply sibling-specific instructions to unrelated paths.\n{entries}"
        ))
    }
}

#[async_trait]
impl ContextProvider for ScopedInstructionFiles {
    /// Reads are bounded and do not publish output if the caller drops the future.
    fn task_instruction_safe(&self) -> bool {
        true
    }

    /// Returns one complete projection; missing candidates are distinct from read or scope errors.
    async fn get(&self, _key: &str, _current_context: &Value) -> Result<Value> {
        Ok(Value::String(self.load().await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        let root = std::env::temp_dir().join(format!("instructions-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("src/nested")).unwrap();
        std::fs::create_dir_all(root.join("other")).unwrap();
        root
    }

    #[tokio::test]
    async fn scoped_instructions_refresh_and_delete_without_sibling_disclosure() {
        let root = fixture();
        std::fs::write(root.join("AGENTS.md"), "root-v1").unwrap();
        std::fs::write(root.join("src/AGENTS.md"), "src-v1").unwrap();
        std::fs::write(root.join("other/AGENTS.md"), "not-authorized").unwrap();
        let source = ScopedInstructionFiles::new(
            &root,
            &["src/one.rs".into(), "src/nested/new.rs".into()],
            &["AGENTS.md".into()],
        )
        .unwrap();
        let initial = source.load().await.unwrap();
        let entries: Vec<Value> =
            serde_json::from_str(initial.split_once('\n').unwrap().1).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["content"], "root-v1");
        assert_eq!(entries[1]["scope"], "src");
        assert!(!initial.contains("not-authorized"));
        std::fs::write(root.join("src/AGENTS.md"), "src-v2").unwrap();
        std::fs::remove_file(root.join("AGENTS.md")).unwrap();
        let changed = source.load().await.unwrap();
        assert!(changed.contains("src-v2"));
        assert!(!changed.contains("root-v1"));
        assert!(!changed.contains("src-v1"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn invalid_instruction_is_not_a_missing_file_or_partial_success() {
        let root = fixture();
        std::fs::write(root.join("AGENTS.md"), "root").unwrap();
        std::fs::write(root.join("src/AGENTS.md"), [0xff]).unwrap();
        let source =
            ScopedInstructionFiles::new(&root, &["src/one.rs".into()], &["AGENTS.md".into()])
                .unwrap();
        assert!(source.load().await.is_err());
        std::fs::write(
            root.join("src/AGENTS.md"),
            vec![b'x'; MAX_FILE_BYTES as usize + 1],
        )
        .unwrap();
        assert!(source.load().await.is_err());
        std::fs::remove_file(root.join("src/AGENTS.md")).unwrap();
        assert!(source.load().await.unwrap().contains("root"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_in_scope_or_candidate_is_refused() {
        let root = fixture();
        std::os::unix::fs::symlink(root.join("other"), root.join("link")).unwrap();
        let linked =
            ScopedInstructionFiles::new(&root, &["link/a.rs".into()], &["AGENTS.md".into()])
                .unwrap();
        assert!(linked.load().await.is_err());
        std::os::unix::fs::symlink(root.join("other/AGENTS.md"), root.join("src/AGENTS.md"))
            .unwrap();
        let source =
            ScopedInstructionFiles::new(&root, &["src/a.rs".into()], &["AGENTS.md".into()])
                .unwrap();
        assert!(source.load().await.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn aggregate_file_and_byte_limits_fail_without_a_partial_projection() {
        let root = fixture();
        let names: Vec<_> = (0..33).map(|index| format!("rules-{index:02}")).collect();
        for name in &names {
            std::fs::write(root.join(name), "ok").unwrap();
        }
        let source = ScopedInstructionFiles::new(&root, &["one.rs".into()], &names).unwrap();
        assert!(source.load().await.is_err());
        for name in &names {
            std::fs::remove_file(root.join(name)).unwrap();
        }
        let names: Vec<_> = (0..5).map(|index| format!("large-{index}")).collect();
        for name in &names {
            std::fs::write(root.join(name), vec![b'x'; MAX_FILE_BYTES as usize]).unwrap();
        }
        let source = ScopedInstructionFiles::new(&root, &["one.rs".into()], &names).unwrap();
        assert!(source.load().await.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_targets_and_candidate_overflow_are_refused() {
        let root = fixture();
        assert!(
            ScopedInstructionFiles::new(&root, &["../outside".into()], &["AGENTS.md".into()])
                .is_err()
        );
        assert!(
            ScopedInstructionFiles::new(&root, &["src/a".into()], &["../AGENTS.md".into()])
                .is_err()
        );
        let names: Vec<_> = (0..257).map(|n| format!("rules-{n}")).collect();
        assert!(ScopedInstructionFiles::new(&root, &["a".into()], &names).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
