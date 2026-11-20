use super::*;
use std::path::PathBuf;

struct ProcessFixture(tempfile::TempDir);
impl ProcessFixture {
    fn new() -> Self {
        Self(tempfile::tempdir().unwrap())
    }
    fn path(&self) -> &Path {
        self.0.path()
    }
}
impl Drop for ProcessFixture {
    fn drop(&mut self) {
        let _ = std::fs::write(self.path().join("release"), b"release");
        if self.path().join("holder-ready").exists() {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !self.path().join("holder-done").exists() && std::time::Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

// The holder is explicitly released by its isolated parent fixture even when a hang guard fails.
#[test]
#[ignore = "subprocess helper"]
fn process_pipe_helper() {
    let Ok(root) = std::env::var("AI_AGENTS_PIPE_FIXTURE") else {
        return;
    };
    let root = PathBuf::from(root);
    if std::env::var("AI_AGENTS_PIPE_MODE").as_deref() == Ok("parent") {
        let _holder = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "types::process_lifetime_tests::process_pipe_helper",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env_clear()
            .env("AI_AGENTS_PIPE_FIXTURE", &root)
            .env("AI_AGENTS_PIPE_MODE", "holder")
            .stdin(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !root.join("holder-ready").exists() {
            assert!(std::time::Instant::now() < deadline, "holder did not start");
            std::thread::sleep(Duration::from_millis(5));
        }
        std::fs::write(root.join("parent-exited"), b"exited").unwrap();
        std::process::exit(0);
    }
    std::fs::write(root.join("holder-ready"), b"ready").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(8);
    while !root.join("release").exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    std::fs::write(root.join("holder-done"), b"done").unwrap();
}

// Child exit cannot restart or remove the original command deadline while an inherited pipe remains open.
#[tokio::test]
async fn command_deadline_includes_readers_after_direct_child_exit() {
    let fixture = ProcessFixture::new();
    let request = CommandRequest {
        argv: vec![
            std::env::current_exe()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            "types::process_lifetime_tests::process_pipe_helper".into(),
            "--exact".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ],
        env: HashMap::from([
            (
                "AI_AGENTS_PIPE_FIXTURE".into(),
                fixture.path().to_string_lossy().into_owned(),
            ),
            ("AI_AGENTS_PIPE_MODE".into(), "parent".into()),
        ]),
        timeout_ms: Some(2_000),
        ..Default::default()
    };
    let response = tokio::time::timeout(
        Duration::from_secs(4),
        ProcessCommandRunner.run_command(
            request,
            ai_agents_core::ToolExecutionContext::test("command"),
        ),
    )
    .await
    .expect("reader wait escaped the original command timeout");
    assert!(fixture.path().join("parent-exited").exists());
    assert!(fixture.path().join("holder-ready").exists());
    assert!(response.timed_out);
    assert!(response.truncated);
    assert_eq!(response.execution_state, CommandExecutionState::Uncertain);
}

struct PipeProbe {
    drops: Arc<std::sync::atomic::AtomicUsize>,
    polls: Arc<std::sync::atomic::AtomicUsize>,
    mode: u8,
}
impl tokio::io::AsyncRead for PipeProbe {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
        _: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        self.polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match self.mode {
            1 => std::task::Poll::Ready(Err(std::io::Error::other("read fixture"))),
            2 => std::task::Poll::Ready(Ok(())),
            _ => std::task::Poll::Pending,
        }
    }
}
impl Drop for PipeProbe {
    fn drop(&mut self) {
        self.drops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}
fn pipe(
    drops: &Arc<std::sync::atomic::AtomicUsize>,
    polls: &Arc<std::sync::atomic::AtomicUsize>,
    mode: u8,
) -> PipeProbe {
    PipeProbe {
        drops: drops.clone(),
        polls: polls.clone(),
        mode,
    }
}

// Both reader handles belong to the in-flight collector, including after wait errors and while its caller is cancelled.
#[tokio::test]
async fn collector_drop_and_wait_failure_release_both_readers() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for fail_wait in [false, true] {
        let drops = Arc::new(AtomicUsize::new(0));
        let polls = Arc::new(AtomicUsize::new(0));
        let wait = async move {
            if fail_wait {
                Err(std::io::Error::other("wait fixture"))
            } else {
                std::future::pending().await
            }
        };
        let mut collector = Box::pin(collect_process_output(
            wait,
            Some(pipe(&drops, &polls, 0)),
            Some(pipe(&drops, &polls, 0)),
            32,
        ));
        if fail_wait {
            assert!(collector.await.is_err());
        } else {
            tokio::select! { _=&mut collector => panic!("unexpected collection"), _=tokio::task::yield_now()=>{} }
            assert!(polls.load(Ordering::SeqCst) > 0);
            drop(collector);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}

// A failed read is incomplete output, not permission to cancel otherwise admitted work or classify its effects as uncertain.
#[cfg(unix)]
#[tokio::test]
async fn read_failure_keeps_wait_and_exit_settlement_separate() {
    use std::os::unix::process::ExitStatusExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let drops = Arc::new(AtomicUsize::new(0));
    let polls = Arc::new(AtomicUsize::new(0));
    let (release, wait) = tokio::sync::oneshot::channel();
    let wait = async {
        wait.await.unwrap();
        Ok(std::process::ExitStatus::from_raw(0))
    };
    let mut collector = Box::pin(collect_process_output(
        wait,
        Some(pipe(&drops, &polls, 1)),
        Some(pipe(&drops, &polls, 2)),
        32,
    ));
    tokio::select! { _=&mut collector => panic!("reader failure cancelled work"), _=tokio::task::yield_now()=>{} }
    assert!(polls.load(Ordering::SeqCst) >= 2);
    release.send(()).unwrap();
    let (status, stdout, stderr) = collector.await.unwrap();
    let result = command_output_response(status, stdout, stderr, CommandRequest::default(), 32);
    assert!(result.success && result.truncated);
    assert_eq!(result.execution_state, CommandExecutionState::Finished);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn pre_spawn_failures_and_legacy_response_shapes_preserve_state() {
    for argv in [vec![], vec!["/no-such-fixture-executable/command".into()]] {
        let response = ProcessCommandRunner
            .run_command(
                CommandRequest {
                    argv,
                    ..Default::default()
                },
                ai_agents_core::ToolExecutionContext::test("command"),
            )
            .await;
        assert_eq!(response.execution_state, CommandExecutionState::NotStarted);
    }
    let legacy = serde_json::json!({"success":true,"termination":"exited"});
    let response: CommandResponse = serde_json::from_value(legacy).unwrap();
    assert_eq!(response.execution_state, CommandExecutionState::Unspecified);
    let encoded = serde_json::to_value(&response).unwrap();
    assert!(encoded.get("execution_state").is_none());
    assert_eq!(
        serde_json::from_value::<CommandResponse>(encoded)
            .unwrap()
            .execution_state,
        CommandExecutionState::Unspecified
    );
    assert!(
        serde_json::from_value::<CommandResponse>(
            serde_json::json!({"success":true,"termination":"exited","execution_state":"made_up"})
        )
        .is_err()
    );
}
