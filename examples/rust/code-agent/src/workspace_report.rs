// Retain bounded local task observations without sending workspace contents to the model.
// Git status and tool records are evidence, not an attribution or filesystem-locking service.
use ai_agents::{TaskRunResult, TaskRunStatus};
use ai_agents_cli::CliTaskObserver;
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

const MAX_GIT_BYTES: u64 = 1024 * 1024;
const MAX_FILE_BYTES: u64 = 1024 * 1024;
const MAX_FILES: usize = 128;
const IO_TIMEOUT: Duration = Duration::from_secs(5);

type GitStatus = BTreeMap<Vec<u8>, [u8; 2]>;

#[derive(PartialEq, Eq)]
enum FileObservation {
    Missing,
    Bytes(Vec<u8>),
    Unavailable,
}

struct Snapshot {
    git: Option<GitStatus>,
    files: Vec<FileObservation>,
}

// Limit status output and execution time; no Git write, external diff or fsmonitor is needed.
async fn git_status(root: &Path) -> Option<GitStatus> {
    let mut command = Command::new("git");
    command
        .args([
            "--no-pager",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
            "--no-renames",
            "--",
            ".",
        ])
        .current_dir(root)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_OBJECT_DIRECTORY")
        .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .stdout(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().ok()?;
    let stdout = child.stdout.take()?;
    let observation = async {
        let mut bytes = Vec::new();
        stdout
            .take(MAX_GIT_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .ok()?;
        if bytes.len() as u64 > MAX_GIT_BYTES {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return None;
        }
        if !child.wait().await.ok()?.success() {
            return None;
        }
        parse_git_status(&bytes)
    };
    tokio::time::timeout(IO_TIMEOUT, observation)
        .await
        .ok()
        .flatten()
}

// The fixed no-renames, NUL-delimited format preserves arbitrary path bytes without parsing quoted names.
fn parse_git_status(bytes: &[u8]) -> Option<GitStatus> {
    let mut result = BTreeMap::new();
    if !bytes.is_empty() && !bytes.ends_with(&[0]) {
        return None;
    }
    for entry in bytes
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
    {
        if entry.len() < 4
            || entry[2] != b' '
            || result
                .insert(entry[3..].to_vec(), [entry[0], entry[1]])
                .is_some()
        {
            return None;
        }
    }
    Some(result)
}

// Each selected file is either fully captured, missing, or unavailable; truncated data is never treated as equal.
async fn observe_file(path: &Path) -> FileObservation {
    let operation = async {
        let metadata = match tokio::fs::symlink_metadata(path).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return FileObservation::Missing;
            }
            Err(_) => return FileObservation::Unavailable,
        };
        if !metadata.file_type().is_file() || metadata.len() > MAX_FILE_BYTES {
            return FileObservation::Unavailable;
        }
        let Ok(file) = tokio::fs::File::open(path).await else {
            return FileObservation::Unavailable;
        };
        let mut bytes = Vec::new();
        if file
            .take(MAX_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .is_err()
            || bytes.len() as u64 > MAX_FILE_BYTES
        {
            return FileObservation::Unavailable;
        }
        FileObservation::Bytes(bytes)
    };
    tokio::time::timeout(IO_TIMEOUT, operation)
        .await
        .unwrap_or(FileObservation::Unavailable)
}

async fn capture(root: &Path, paths: &[PathBuf]) -> Snapshot {
    let observation = async {
        let git = git_status(root).await;
        let mut files = Vec::with_capacity(paths.len().min(MAX_FILES));
        for path in paths.iter().take(MAX_FILES) {
            files.push(observe_file(path).await);
        }
        Snapshot { git, files }
    };
    tokio::time::timeout(Duration::from_secs(15), observation)
        .await
        .unwrap_or_else(|_| Snapshot {
            git: None,
            files: Vec::new(),
        })
}

fn status_counts(status: &GitStatus) -> (usize, usize, usize) {
    let mut counts = (0, 0, 0);
    for [index, worktree] in status.values().copied() {
        if index == b'?' && worktree == b'?' {
            counts.2 += 1;
        } else {
            counts.0 += usize::from(index != b' ');
            counts.1 += usize::from(worktree != b' ');
        }
    }
    counts
}

/// The example owns one baseline per foreground task; exact bytes remain local and are discarded after reporting.
pub struct WorkspaceReport {
    root: PathBuf,
    paths: Vec<PathBuf>,
    baseline: Mutex<Option<Snapshot>>,
}

impl WorkspaceReport {
    pub fn new(root: PathBuf, paths: Vec<PathBuf>) -> Self {
        Self {
            root,
            paths,
            baseline: Mutex::new(None),
        }
    }
}

#[async_trait]
impl CliTaskObserver for WorkspaceReport {
    async fn before_task(&self) -> Result<(), String> {
        let baseline = capture(&self.root, &self.paths).await;
        *self
            .baseline
            .lock()
            .map_err(|_| "workspace observation unavailable")? = Some(baseline);
        Ok(())
    }

    async fn after_task(&self, result: Option<&TaskRunResult>) -> Option<String> {
        let baseline = self.baseline.lock().ok()?.take()?;
        let Some(result) = result else {
            return Some(
                "Workspace report: task result unavailable; effects and cleanup are unconfirmed.\n"
                    .into(),
            );
        };
        let mut report = format!(
            "Workspace report for task {} revision {} ({:?}):\n",
            result.run.key.run_id, result.run.revision, result.run.status
        );
        if let Some(git) = &baseline.git {
            let (staged, unstaged, untracked) = status_counts(git);
            report.push_str(&format!("  Before task: {staged} staged, {unstaged} unstaged, {untracked} untracked path entries (submodules excluded).\n"));
            for (path, [index, worktree]) in git.iter().take(32) {
                let name: String = String::from_utf8_lossy(path).chars().take(128).collect();
                report.push_str(&format!(
                    "    Existing {:?}: index {:?}, worktree {:?}.\n",
                    name, *index as char, *worktree as char
                ));
            }
            if git.len() > 32 || git.keys().any(|path| path.len() > 128) {
                report.push_str(
                    "    Existing path display is bounded and may omit or shorten names.\n",
                );
            }
        } else {
            report.push_str("  Before task: Git status unavailable; it is not assumed clean.\n");
        }
        if matches!(
            result.run.status,
            TaskRunStatus::Paused | TaskRunStatus::Running | TaskRunStatus::RecoveryRequired
        ) {
            report.push_str("  Final workspace observation unavailable: task settlement or cleanup is unconfirmed.\n");
            return Some(report);
        }
        let after = capture(&self.root, &self.paths).await;
        match (&baseline.git, &after.git) {
            (Some(before), Some(after)) => {
                let changed = before.keys().chain(after.keys()).filter(|path| before.get(*path) != after.get(*path)).collect::<std::collections::BTreeSet<_>>().len();
                report.push_str(&format!("  Git status entries changed: {changed}; equal status codes do not prove equal contents or index.\n"));
            }
            _ => report.push_str("  Git status comparison unavailable; no clean or index-preservation claim is made.\n"),
        }
        for (index, path) in self.paths.iter().take(MAX_FILES).enumerate() {
            let recorded_mutation = result.run.evidence.tool_calls.iter().any(|evidence| {
                let record = &evidence.record;
                record.executed
                    && record.success
                    && matches!(record.canonical_id.as_str(), "file_edit" | "file_write")
                    && record
                        .executed_arguments
                        .get("path")
                        .and_then(|value| value.as_str())
                        == path.to_str()
                    && serde_json::from_str::<serde_json::Value>(&record.output)
                        .ok()
                        .and_then(|value| {
                            value
                                .get("mutation_performed")
                                .and_then(|value| value.as_bool())
                        })
                        == Some(true)
            });
            let relation = match (baseline.files.get(index), after.files.get(index)) {
                (Some(FileObservation::Bytes(a)), Some(FileObservation::Bytes(b))) if a == b => {
                    "unchanged bytes"
                }
                (Some(FileObservation::Missing), Some(FileObservation::Missing)) => "still absent",
                (Some(FileObservation::Bytes(_)), Some(FileObservation::Bytes(_)))
                | (Some(FileObservation::Missing), Some(FileObservation::Bytes(_)))
                | (Some(FileObservation::Bytes(_)), Some(FileObservation::Missing)) => {
                    "changed bytes or existence"
                }
                _ => "comparison unavailable",
            };
            let relative = path.strip_prefix(&self.root).unwrap_or(path);
            let name: String = relative.to_string_lossy().chars().take(160).collect();
            report.push_str(&format!("  Selected {:?}: {relation}; matching recorded task mutation: {recorded_mutation}.\n", name));
        }
        if self.paths.len() > MAX_FILES {
            report.push_str("  Additional selected files were not observed because the local report limit was reached.\n");
        }
        let mut mutations = 0;
        let mut checks = 0;
        let mut passed_checks = 0;
        let mut failed_checks = 0;
        let mut unknown_checks = 0;
        let mut uncertain = 0;
        // Root mode has no selected list, so the report names each path that carries an actual mutation receipt.
        let mut targets = std::collections::BTreeSet::new();
        for evidence in &result.run.evidence.tool_calls {
            let record = &evidence.record;
            if matches!(
                record.canonical_id.as_str(),
                "file_edit" | "file_write" | "patch"
            ) && record.executed
            {
                if record.success
                    && serde_json::from_str::<serde_json::Value>(&record.output)
                        .ok()
                        .and_then(|value| {
                            value
                                .get("mutation_performed")
                                .and_then(|value| value.as_bool())
                        })
                        == Some(true)
                {
                    mutations += 1;
                    if let Some(path) = record
                        .executed_arguments
                        .get("path")
                        .and_then(|value| value.as_str())
                    {
                        targets.insert(PathBuf::from(path));
                    }
                } else if record.timed_out || record.cancelled || !record.success {
                    uncertain += 1;
                }
            }
            if record.canonical_id == "command" && record.executed {
                checks += 1;
                let exit = (!record.timed_out && !record.cancelled)
                    .then(|| serde_json::from_str::<serde_json::Value>(&record.output).ok())
                    .flatten()
                    .and_then(|value| value.get("exit_code").and_then(|value| value.as_i64()));
                match exit {
                    Some(0) => passed_checks += 1,
                    Some(_) => failed_checks += 1,
                    None => unknown_checks += 1,
                }
            }
        }
        for target in targets.iter().take(32) {
            let relative = target.strip_prefix(&self.root).unwrap_or(target);
            let name: String = relative.to_string_lossy().chars().take(160).collect();
            report.push_str(&format!("  Recorded mutation target {:?}.\n", name));
        }
        if targets.len() > 32 {
            report.push_str(
                "  Additional recorded mutation targets were omitted from this bounded report.\n",
            );
        }
        report.push_str(&format!("  Recorded actual mutation receipts: {mutations}; executed commands: {checks}; failed/cancelled mutation attempts: {uncertain}.\n"));
        report.push_str(&format!("  Command exits: {passed_checks} zero, {failed_checks} nonzero, {unknown_checks} unavailable or interrupted; these are not independent proof of completion.\n"));
        report.push_str("  Status and selected-byte observations do not attribute every change to the agent or certify external freshness, index preservation, checker isolation or rollback.\n");
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_parser_preserves_unusual_names_and_rejects_partial_output() {
        let status = parse_git_status(b"MM source.rs\0?? line\nname\0").unwrap();
        assert_eq!(status_counts(&status), (1, 1, 1));
        assert!(status.contains_key(b"line\nname".as_slice()));
        assert!(parse_git_status(b" M partial").is_none());
        assert!(parse_git_status(b"bad\0").is_none());
    }
}
