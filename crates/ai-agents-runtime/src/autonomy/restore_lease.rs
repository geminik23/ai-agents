//! An OS lease excludes simultaneous configured reconstruction workers without claiming external-effect quiescence.

use ai_agents_core::{AgentError, Result};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use sha2::{Digest, Sha256};
use std::path::Path;

/// Keeps a host's actual preparation exclusion alive through the acknowledged transaction.
/// Holding a lease alone does not authorize recovery or prove that an external effect stopped.
pub trait TaskRestoreWorkLease: Send + Sync {}

// The descriptor remains open for the whole work transaction; lock files are never unlinked while another process might reference the inode.
pub(crate) struct TaskFileLease {
    _file: std::fs::File,
}
impl TaskRestoreWorkLease for TaskFileLease {}

impl TaskFileLease {
    // Local SQLite and a live OS file lock provide the configured host's process-exclusion evidence.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(crate) fn acquire(database: &Path, namespace: &str, run_id: &str) -> Result<Self> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
        let database = database.canonicalize()?;
        let mut name = database.as_os_str().to_os_string();
        name.push(".task-locks");
        let directory = std::path::PathBuf::from(name);
        match std::fs::DirBuilder::new().mode(0o700).create(&directory) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        if !std::fs::symlink_metadata(&directory)?.file_type().is_dir() {
            return Err(AgentError::Config(
                "task lease directory must be a real directory".into(),
            ));
        }
        let key = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(namespace, run_id))?)
        );
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(directory.join(key))?;
        if !file.metadata()?.is_file() {
            return Err(AgentError::Config(
                "task lease must be a regular file".into(),
            ));
        }
        // SAFETY: file owns a live descriptor; flock neither dereferences user memory nor changes the task checkpoint.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock {
                return Err(super::TaskRunStorageError::Owned.into());
            }
            return Err(error.into());
        }
        Ok(Self { _file: file })
    }

    // Unsupported platforms cannot silently replace a process lease with an in-memory mutex.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub(crate) fn acquire(_database: &Path, _namespace: &str, _run_id: &str) -> Result<Self> {
        Err(AgentError::Config(
            "configured restoration process leases are unavailable on this platform".into(),
        ))
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;

    // Separate opens contend on one inode, and dropping the holder permits a new transaction without deleting the lock file.
    #[test]
    fn lease_excludes_competing_workers_and_survives_reopen() {
        let directory = std::env::temp_dir().join(format!("task-lease-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let database = directory.join("tasks.sqlite");
        std::fs::write(&database, []).unwrap();
        let first = TaskFileLease::acquire(&database, "actor-a", "run").unwrap();
        assert!(matches!(
            TaskFileLease::acquire(&database, "actor-a", "run"),
            Err(AgentError::TaskRunStorage(
                super::super::TaskRunStorageError::Owned
            ))
        ));
        let independent = TaskFileLease::acquire(&database, "actor-a", "other-run").unwrap();
        drop(first);
        let next = TaskFileLease::acquire(&database, "actor-a", "run").unwrap();
        drop((next, independent));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
