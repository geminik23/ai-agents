// Exercise the real example binary, shared executor and process-backed fixture acceptance.
// The finite provider is deterministic and does not measure model problem-solving quality.
#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn offline_example_reads_edits_and_validates_the_disposable_fixture() {
    let executable = env!("CARGO_BIN_EXE_code-agent-setup");
    let cargo = std::fs::canonicalize(env!("CARGO")).unwrap();
    let rustc = cargo.parent().unwrap().join("rustc");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from("/tmp").join(format!(
        "code-agent-example-test-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&parent).unwrap();
    let workspace = parent.join("workspace");
    let prepared = Command::new(executable)
        .arg("prepare")
        .arg(&workspace)
        .output()
        .unwrap();
    assert!(
        prepared.status.success(),
        "{}",
        String::from_utf8_lossy(&prepared.stderr)
    );
    let result = Command::new(executable)
        .arg("run-offline")
        .arg(&workspace)
        .arg(&cargo)
        .arg(&rustc)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    assert!(output.contains("Initial sample-project acceptance failed as expected."));
    assert!(
        output.contains("Actual file_read and file_edit executed through the shared executor.")
    );
    assert!(output.contains("The runtime command and a fresh independent acceptance both passed."));
    let source = std::fs::read_to_string(workspace.join("src/lib.rs")).unwrap();
    assert!(source.contains("value.max(lower).min(upper)"));
    assert_eq!(
        std::fs::read(workspace.join("tests/acceptance.rs")).unwrap(),
        include_bytes!("../sample-project/tests/acceptance.rs")
    );
    let rerun = Command::new(executable)
        .arg("run-offline")
        .arg(&workspace)
        .arg(&cargo)
        .arg(&rustc)
        .output()
        .unwrap();
    assert!(!rerun.status.success());
    assert!(String::from_utf8_lossy(&rerun.stderr).contains("already passing"));
    std::fs::remove_dir_all(parent).unwrap();
}

#[test]
fn workspace_check_is_provider_free_and_live_mode_requires_explicit_consent() {
    use std::os::unix::fs::PermissionsExt;
    let executable = env!("CARGO_BIN_EXE_code-agent-setup");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from("/tmp").join(format!(
        "code-agent-workspace-cli-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&parent).unwrap();
    let workspace = parent.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let source = workspace.join("source.rs");
    std::fs::write(&source, "pub fn value() -> i32 { 1 }\n").unwrap();
    std::fs::write(workspace.join("second.rs"), "pub fn value() -> i32 { 2 }\n").unwrap();
    let checker = parent.join("checker.sh");
    let marker = parent.join("checker-ran");
    std::fs::write(
        &checker,
        format!("#!/bin/sh\nprintf invoked > '{}'\n", marker.display()),
    )
    .unwrap();
    std::fs::set_permissions(&checker, std::fs::Permissions::from_mode(0o700)).unwrap();
    let checked = Command::new(executable)
        .env_clear()
        .arg("check-workspace")
        .arg(&workspace)
        .arg("source.rs")
        .arg(&checker)
        .arg("openai")
        .arg("offline-test")
        .arg("--file")
        .arg("second.rs")
        .arg("--new-file")
        .arg("created.rs")
        .arg("--assume-exclusive-workspace")
        .arg("--allow-discovery")
        .arg("--max-output-tokens")
        .arg("4096")
        .arg("--checker-timeout-ms")
        .arg("45000")
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    assert!(
        String::from_utf8_lossy(&checked.stdout)
            .contains("without constructing a provider or running the checker")
    );
    assert!(String::from_utf8_lossy(&checked.stdout).contains("Selected source files: 3"));
    assert!(!workspace.join("created.rs").exists());
    assert!(String::from_utf8_lossy(&checked.stdout).contains("Workspace discovery enabled: true"));
    assert!(
        String::from_utf8_lossy(&checked.stdout).contains("Cooperative revision assumed: true")
    );
    assert!(String::from_utf8_lossy(&checked.stdout).contains("Maximum model output tokens: 4096"));
    assert!(String::from_utf8_lossy(&checked.stdout).contains("Checker timeout: 45000 ms"));
    assert!(!marker.exists());
    let read_only = Command::new(executable)
        .env_clear()
        .arg("check-read-only")
        .arg(&workspace)
        .arg("source.rs")
        .arg("openai")
        .arg("offline-test")
        .arg("--file")
        .arg("second.rs")
        .arg("--allow-discovery")
        .output()
        .unwrap();
    assert!(
        read_only.status.success(),
        "{}",
        String::from_utf8_lossy(&read_only.stderr)
    );
    assert!(
        String::from_utf8_lossy(&read_only.stdout)
            .contains("without constructing a provider or installing a checker")
    );
    assert!(!marker.exists());
    let missing_consent = Command::new(executable)
        .env_clear()
        .args(["run-read-only"])
        .arg(&workspace)
        .args(["source.rs", "openai", "offline-test"])
        .output()
        .unwrap();
    assert!(!missing_consent.status.success());
    assert!(!marker.exists());
    let duplicate = Command::new(executable)
        .env_clear()
        .arg("check-workspace")
        .arg(&workspace)
        .arg("source.rs")
        .arg(&checker)
        .arg("openai")
        .arg("offline-test")
        .arg("--file")
        .arg("source.rs")
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert!(
        String::from_utf8_lossy(&duplicate.stderr).contains("must not be selected more than once")
    );
    assert!(!marker.exists());
    let existing_new_file = Command::new(executable)
        .env_clear()
        .arg("check-workspace")
        .arg(&workspace)
        .args(["source.rs"])
        .arg(&checker)
        .args(["openai", "offline-test", "--new-file", "second.rs"])
        .output()
        .unwrap();
    assert!(!existing_new_file.status.success());
    assert!(String::from_utf8_lossy(&existing_new_file.stderr).contains("already exists"));
    let read_only_create = Command::new(executable)
        .env_clear()
        .arg("check-read-only")
        .arg(&workspace)
        .args([
            "source.rs",
            "openai",
            "offline-test",
            "--new-file",
            "created.rs",
        ])
        .output()
        .unwrap();
    assert!(!read_only_create.status.success());
    assert!(!workspace.join("created.rs").exists());
    assert!(!marker.exists());
    for flag in ["--max-output-tokens", "--checker-timeout-ms"] {
        let invalid = Command::new(executable)
            .env_clear()
            .arg("check-workspace")
            .arg(&workspace)
            .arg("source.rs")
            .arg(&checker)
            .arg("openai")
            .arg("offline-test")
            .arg(flag)
            .arg("0")
            .output()
            .unwrap();
        assert!(!invalid.status.success());
        assert!(String::from_utf8_lossy(&invalid.stderr).contains("must be positive"));
        assert!(!marker.exists());
    }
    std::fs::write(
        workspace.join("-instructions.md"),
        "Read the selected file.\n",
    )
    .unwrap();
    let named_instruction = Command::new(executable)
        .env_clear()
        .arg("check-workspace")
        .arg(&workspace)
        .arg("source.rs")
        .arg(&checker)
        .arg("openai")
        .arg("offline-test")
        .arg("--")
        .arg("-instructions.md")
        .output()
        .unwrap();
    assert!(
        named_instruction.status.success(),
        "{}",
        String::from_utf8_lossy(&named_instruction.stderr)
    );
    assert!(!marker.exists());
    for flags in [vec![], vec!["--trust-checker"]] {
        let mut command = Command::new(executable);
        command
            .env_clear()
            .arg("run-workspace")
            .args(flags)
            .arg(&workspace)
            .arg("source.rs")
            .arg(&checker)
            .arg("openai")
            .arg("offline-test");
        let rejected = command.output().unwrap();
        assert!(!rejected.status.success());
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("usage:"));
        assert!(!marker.exists());
        assert_eq!(
            std::fs::read_to_string(&source).unwrap(),
            "pub fn value() -> i32 { 1 }\n"
        );
    }
    std::fs::remove_dir_all(parent).unwrap();
}
