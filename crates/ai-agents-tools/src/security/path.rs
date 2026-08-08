use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

/// Reports why a path could not be resolved safely for policy evaluation.
#[derive(Debug, Error)]
pub(crate) enum PathResolutionError {
    #[error("failed to read the current workspace: {0}")]
    CurrentWorkspace(#[source] io::Error),
    #[error("failed to inspect path '{path}': {source}")]
    Metadata {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to canonicalize existing path '{path}': {source}")]
    Canonicalize {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "relative policy root '{root}' resolves outside workspace '{workspace}' as '{resolved}'"
    )]
    RelativeRootEscape {
        root: PathBuf,
        workspace: PathBuf,
        resolved: PathBuf,
    },
}

#[derive(Debug, Clone)]
struct ResolvedPath {
    lexical: PathBuf,
    resolved: PathBuf,
}

/// Resolves candidate paths and policy roots against one canonical host-owned workspace snapshot.
#[derive(Debug, Clone)]
pub(crate) struct PathPolicyResolver {
    workspace: PathBuf,
}

impl PathPolicyResolver {
    /// Captures the canonical current directory used as the relative policy boundary.
    /// This point-in-time check assumes the host prevents untrusted replacement after validation.
    pub(crate) fn new() -> Result<Self, PathResolutionError> {
        let workspace = std::env::current_dir().map_err(PathResolutionError::CurrentWorkspace)?;
        Self::for_workspace(&workspace)
    }

    fn for_workspace(workspace: &Path) -> Result<Self, PathResolutionError> {
        inspect_existing_entry(workspace)?;
        let workspace =
            fs::canonicalize(workspace).map_err(|source| PathResolutionError::Canonicalize {
                path: workspace.to_path_buf(),
                source,
            })?;
        Ok(Self {
            workspace: normalize_lexically(&workspace),
        })
    }

    /// Resolves a candidate while preserving ordinary missing suffix components.
    pub(crate) fn resolve_path(&self, path: &Path) -> Result<PathBuf, PathResolutionError> {
        Ok(self.resolve_candidate(path)?.resolved)
    }

    /// Allows only resolved containment so lexical prefixes cannot authorize symlink escapes.
    pub(crate) fn is_allowed(
        &self,
        candidate: &Path,
        root: &Path,
    ) -> Result<bool, PathResolutionError> {
        let candidate = self.resolve_candidate(candidate)?;
        let root = self.resolve_policy_root(root)?;
        Ok(candidate.resolved.starts_with(&root.resolved))
    }

    /// Restrictions match either spelling so symlinks cannot weaken deny, unavailable, or approval policy.
    pub(crate) fn matches_restriction(
        &self,
        candidate: &Path,
        root: &Path,
    ) -> Result<bool, PathResolutionError> {
        let candidate = self.resolve_candidate(candidate)?;
        let root = self.resolve_policy_root(root)?;
        Ok(candidate.lexical.starts_with(&root.lexical)
            || candidate.resolved.starts_with(&root.resolved))
    }

    /// Compares both spellings when destructive tools refuse configured roots themselves.
    pub(crate) fn is_same_location(
        &self,
        candidate: &Path,
        root: &Path,
    ) -> Result<bool, PathResolutionError> {
        let candidate = self.resolve_candidate(candidate)?;
        let root = self.resolve_policy_root(root)?;
        Ok(candidate.lexical == root.lexical || candidate.resolved == root.resolved)
    }

    fn resolve_candidate(&self, path: &Path) -> Result<ResolvedPath, PathResolutionError> {
        self.resolve(path)
    }

    fn resolve_policy_root(&self, root: &Path) -> Result<ResolvedPath, PathResolutionError> {
        let resolved_root = self.resolve(root)?;
        if !root.is_absolute() && !resolved_root.resolved.starts_with(&self.workspace) {
            return Err(PathResolutionError::RelativeRootEscape {
                root: root.to_path_buf(),
                workspace: self.workspace.clone(),
                resolved: resolved_root.resolved,
            });
        }
        Ok(resolved_root)
    }

    fn resolve(&self, path: &Path) -> Result<ResolvedPath, PathResolutionError> {
        let lexical = if path.is_absolute() {
            normalize_lexically(path)
        } else {
            normalize_lexically(&self.workspace.join(path))
        };
        let resolved = resolve_with_missing_suffix(&lexical)?;
        Ok(ResolvedPath { lexical, resolved })
    }
}

fn inspect_existing_entry(path: &Path) -> Result<(), PathResolutionError> {
    fs::symlink_metadata(path)
        .map(|_| ())
        .map_err(|source| PathResolutionError::Metadata {
            path: path.to_path_buf(),
            source,
        })
}

// Entry-aware metadata stops at dangling symlinks so canonicalization fails closed instead of reclassifying them as missing suffixes.
fn resolve_with_missing_suffix(path: &Path) -> Result<PathBuf, PathResolutionError> {
    let mut ancestor = path;
    let mut missing = Vec::<OsString>::new();
    loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => {
                let mut resolved = fs::canonicalize(ancestor).map_err(|source| {
                    PathResolutionError::Canonicalize {
                        path: ancestor.to_path_buf(),
                        source,
                    }
                })?;
                for component in missing.iter().rev() {
                    resolved.push(component);
                }
                return Ok(normalize_lexically(&resolved));
            }
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                let Some(name) = ancestor.file_name() else {
                    return Err(PathResolutionError::Metadata {
                        path: ancestor.to_path_buf(),
                        source,
                    });
                };
                missing.push(name.to_os_string());
                let Some(parent) = ancestor.parent() else {
                    return Err(PathResolutionError::Metadata {
                        path: ancestor.to_path_buf(),
                        source,
                    });
                };
                ancestor = parent;
            }
            Err(source) => {
                return Err(PathResolutionError::Metadata {
                    path: ancestor.to_path_buf(),
                    source,
                });
            }
        }
    }
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[cfg(unix)]
    fn symlink_dir(target: &Path, link: &Path) -> bool {
        std::os::unix::fs::symlink(target, link).unwrap();
        true
    }

    #[cfg(windows)]
    fn symlink_dir(target: &Path, link: &Path) -> bool {
        match std::os::windows::fs::symlink_dir(target, link) {
            Ok(()) => true,
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => false,
            Err(error) => panic!("failed to create directory symlink: {error}"),
        }
    }

    #[cfg(unix)]
    fn symlink_file(target: &Path, link: &Path) -> bool {
        std::os::unix::fs::symlink(target, link).unwrap();
        true
    }

    #[cfg(windows)]
    fn symlink_file(target: &Path, link: &Path) -> bool {
        match std::os::windows::fs::symlink_file(target, link) {
            Ok(()) => true,
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => false,
            Err(error) => panic!("failed to create file symlink: {error}"),
        }
    }

    #[test]
    fn allows_existing_and_missing_targets_under_resolved_roots() {
        let directory = tempdir().unwrap();
        let workspace = directory.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(workspace.join("existing-root")).unwrap();
        fs::write(workspace.join("existing-root/file.txt"), "data").unwrap();
        let resolver = PathPolicyResolver::for_workspace(&workspace).unwrap();

        assert!(
            resolver
                .is_allowed(
                    Path::new("existing-root/file.txt"),
                    Path::new("existing-root")
                )
                .unwrap()
        );
        assert!(
            resolver
                .is_allowed(
                    Path::new("existing-root/missing/file.txt"),
                    Path::new("existing-root")
                )
                .unwrap()
        );
        assert!(
            resolver
                .is_allowed(
                    Path::new("ordinary/missing-root/file.txt"),
                    Path::new("ordinary/missing-root")
                )
                .unwrap()
        );
    }

    #[test]
    fn absolute_root_authorizes_resolved_location_and_missing_descendants() {
        let directory = tempdir().unwrap();
        let workspace = directory.path().join("workspace");
        let outside = directory.path().join("outside");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(&outside).unwrap();
        let resolver = PathPolicyResolver::for_workspace(&workspace).unwrap();

        assert!(
            resolver
                .is_allowed(&outside.join("missing/file.txt"), &outside)
                .unwrap()
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn absolute_symlink_root_authorizes_its_resolved_location() {
        let directory = tempdir().unwrap();
        let workspace = directory.path().join("workspace");
        let outside = directory.path().join("outside");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(&outside).unwrap();
        let absolute_root = workspace.join("absolute-root");
        if !symlink_dir(&outside, &absolute_root) {
            return;
        }
        let resolver = PathPolicyResolver::for_workspace(&workspace).unwrap();

        assert!(
            resolver
                .is_allowed(&absolute_root.join("missing/file.txt"), &absolute_root)
                .unwrap()
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn relative_root_beneath_escaping_symlink_is_rejected() {
        let directory = tempdir().unwrap();
        let workspace = directory.path().join("workspace");
        let outside = directory.path().join("outside");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(&outside).unwrap();
        if !symlink_dir(&outside, &workspace.join("escape")) {
            return;
        }
        let resolver = PathPolicyResolver::for_workspace(&workspace).unwrap();

        let error = resolver
            .is_allowed(Path::new("escape/new/file.txt"), Path::new("escape/new"))
            .unwrap_err();

        assert!(matches!(
            error,
            PathResolutionError::RelativeRootEscape { .. }
        ));
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn dangling_candidate_and_policy_root_fail_closed() {
        let directory = tempdir().unwrap();
        let workspace = directory.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        if !symlink_file(
            &workspace.join("missing-target"),
            &workspace.join("dangling"),
        ) {
            return;
        }
        let resolver = PathPolicyResolver::for_workspace(&workspace).unwrap();

        assert!(
            resolver
                .is_allowed(Path::new("dangling"), Path::new("."))
                .is_err()
        );
        assert!(
            resolver
                .is_allowed(Path::new("ordinary.txt"), Path::new("dangling"))
                .is_err()
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn resolved_allow_rejects_escape_but_restriction_matches_alias() {
        let directory = tempdir().unwrap();
        let workspace = directory.path().join("workspace");
        let outside = directory.path().join("outside");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(workspace.join("allowed")).unwrap();
        fs::create_dir(&outside).unwrap();
        if !symlink_dir(&outside, &workspace.join("allowed/escape")) {
            return;
        }
        let resolver = PathPolicyResolver::for_workspace(&workspace).unwrap();
        let candidate = Path::new("allowed/escape/secret.txt");

        assert!(
            !resolver
                .is_allowed(candidate, Path::new("allowed"))
                .unwrap()
        );
        assert!(resolver.matches_restriction(candidate, &outside).unwrap());
        assert!(
            resolver
                .matches_restriction(candidate, Path::new("allowed/escape"))
                .is_err()
        );
    }
}
