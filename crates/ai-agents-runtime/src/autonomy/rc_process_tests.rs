//! Independent processes contend at the actual SQLite claim boundary and recover only explicitly reviewed interrupted construction.
use super::*;

struct ProcessClaimStore {
    inner: Arc<ScopedTaskRunStore>,
    directory: std::path::PathBuf,
    own: &'static str,
    peer: &'static str,
}
#[async_trait]
impl TaskRunStore for ProcessClaimStore {
    // The process barrier changes only arrival order, never the production storage protocol or exact receipt.
    async fn create(&self, s: &TaskRunSnapshot) -> Result<()> {
        self.inner.create(s).await
    }
    async fn load(&self, id: &str) -> Result<Option<TaskRunSnapshot>> {
        self.inner.load(id).await
    }
    async fn list(&self) -> Result<Vec<TaskRunSummary>> {
        self.inner.list().await
    }
    async fn delete(&self, id: &str, rev: u64) -> Result<()> {
        self.inner.delete(id, rev).await
    }
    async fn mutate(&self, id: &str, m: &TaskRunMutation) -> Result<TaskRunSnapshot> {
        self.inner.mutate(id, m).await
    }
    fn supports_reconstruction(&self) -> bool {
        true
    }
    async fn checkpoint_reconstruction(
        &self,
        id: &str,
        checkpoint: &TaskReconstructionCheckpoint,
    ) -> Result<TaskRunSnapshot> {
        self.inner.checkpoint_reconstruction(id, checkpoint).await
    }
    // Both processes finish source validation before either may claim; no runtime gate or Rust Arc is shared across them.
    async fn claim_paused(
        &self,
        id: &str,
        revision: u64,
        token: String,
    ) -> Result<Option<TaskRunSnapshot>> {
        std::fs::write(self.directory.join(self.own), b"validated").unwrap();
        tokio::time::timeout(Duration::from_secs(30), async {
            while !self.directory.join(self.peer).exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        self.inner.claim_paused(id, revision, token).await
    }
}

// A subprocess has its own runtime, host, provider and SQLite connection; only isolated files cross the boundary.
async fn worker(directory: &std::path::Path, mode: &str) -> std::process::Output {
    let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "autonomy::restoration_tests::rc_tests::followup::process::rc_process_worker",
            "--nocapture",
        ])
        .env("AI_AGENTS_RC_DIRECTORY", directory)
        .env("AI_AGENTS_RC_MODE", mode)
        .kill_on_drop(true);
    tokio::time::timeout(Duration::from_secs(120), command.output())
        .await
        .unwrap()
        .unwrap()
}

// Process death is not an effect-stop proof; the parent additionally knows this controlled constructor performs no domain work.
#[tokio::test]
async fn rc_independent_process_claim_race_and_interrupted_constructor_recovery() {
    for crash in [false, true] {
        let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../temp/restoration-tests")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&directory).unwrap();
        let mut host = host();
        Arc::get_mut(&mut host).unwrap().effect_file = Some(directory.join("effects.log"));
        let storage = Arc::new(
            ai_agents_storage::SqliteStorage::new(directory.join("tasks.sqlite").to_str().unwrap())
                .await
                .unwrap(),
        );
        let store = Arc::new(
            ScopedTaskRunStore::new(
                storage.clone(),
                "RestoreRoot".into(),
                None,
                "restore.v1".into(),
            )
            .unwrap(),
        );
        let runner = AutonomyRunner::try_new(
            host.runtime(host.root.clone()),
            persistent_config(false),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap()
        .with_restore_host(host.clone());
        let paused = runner
            .run("support case across processes", None)
            .await
            .unwrap();
        std::fs::write(
            directory.join("request.json"),
            serde_json::to_vec(&json!({"id":paused.run.key.run_id,"revision":paused.run.revision}))
                .unwrap(),
        )
        .unwrap();
        drop(runner);
        drop(store);
        storage.close().await;
        if crash {
            let output = worker(&directory, "crash").await;
            assert_eq!(
                output.status.code(),
                Some(73),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let storage = Arc::new(
                ai_agents_storage::SqliteStorage::new(
                    directory.join("tasks.sqlite").to_str().unwrap(),
                )
                .await
                .unwrap(),
            );
            let store = Arc::new(
                ScopedTaskRunStore::new(
                    storage.clone(),
                    "RestoreRoot".into(),
                    None,
                    "restore.v1".into(),
                )
                .unwrap(),
            );
            let source = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
            assert_eq!(source.status, TaskRunStatus::Running);
            let proof = RecoveryProof::new(&source);
            let recovered = AutonomyRunner::reconcile_recovery(
                store.clone(),
                &source.key.run_id,
                decision(&source, TaskRecoveryDisposition::ResumePrepared),
                proof,
            )
            .await
            .unwrap();
            std::fs::write(
                directory.join("request.json"),
                serde_json::to_vec(
                    &json!({"id":recovered.key.run_id,"revision":recovered.revision}),
                )
                .unwrap(),
            )
            .unwrap();
            drop(store);
            storage.close().await;
            let output = worker(&directory, "consumer").await;
            assert!(
                output.status.success(),
                "{} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        } else {
            let (left, right) =
                tokio::join!(worker(&directory, "race-a"), worker(&directory, "race-b"));
            assert!(
                left.status.success() && right.status.success(),
                "left: {} {} right: {} {}",
                String::from_utf8_lossy(&left.stdout),
                String::from_utf8_lossy(&left.stderr),
                String::from_utf8_lossy(&right.stdout),
                String::from_utf8_lossy(&right.stderr)
            );
            let output = format!(
                "{}{}",
                String::from_utf8_lossy(&left.stdout),
                String::from_utf8_lossy(&right.stdout)
            );
            assert_eq!(output.matches("rc-winner").count(), 1);
            assert_eq!(output.matches("rc-loser").count(), 1);
        }
        assert_eq!(
            std::fs::read_to_string(directory.join("effects.log"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        crate::remove_sqlite_test_directory(&directory)
            .await
            .unwrap();
    }
}

// Normal suite discovery performs no work; only the parent's explicit child environment activates this helper.
#[tokio::test]
async fn rc_process_worker() {
    let Ok(mode) = std::env::var("AI_AGENTS_RC_MODE") else {
        return;
    };
    let directory = std::path::PathBuf::from(std::env::var_os("AI_AGENTS_RC_DIRECTORY").unwrap());
    let mut host = host();
    Arc::get_mut(&mut host).unwrap().effect_file = Some(directory.join("effects.log"));
    let storage = Arc::new(
        ai_agents_storage::SqliteStorage::new(directory.join("tasks.sqlite").to_str().unwrap())
            .await
            .unwrap(),
    );
    let inner = Arc::new(
        ScopedTaskRunStore::new(
            storage.clone(),
            "RestoreRoot".into(),
            None,
            "restore.v1".into(),
        )
        .unwrap(),
    );
    let data: Value =
        serde_json::from_slice(&std::fs::read(directory.join("request.json")).unwrap()).unwrap();
    let request = TaskRestoreRequest {
        run_id: data["id"].as_str().unwrap().into(),
        expected_revision: data["revision"].as_u64().unwrap(),
        config_identity: "restore.v1".into(),
        config: persistent_config(false),
        ceilings: Default::default(),
    };
    let store: Arc<dyn TaskRunStore> = if mode.starts_with("race-") {
        Arc::new(ProcessClaimStore {
            inner: inner.clone(),
            directory: directory.clone(),
            own: if mode == "race-a" {
                "a.ready"
            } else {
                "b.ready"
            },
            peer: if mode == "race-a" {
                "b.ready"
            } else {
                "a.ready"
            },
        })
    } else {
        inner.clone()
    };
    if mode == "crash" {
        host.stage_hold.store(true, Ordering::SeqCst);
        let task = tokio::spawn({
            let host = host.clone();
            async move { AutonomyRunner::restore_paused(request, store, host).await }
        });
        host.stage_reached.acquire().await.unwrap().forget();
        assert!(!task.is_finished());
        std::process::exit(73);
    }
    match AutonomyRunner::restore_paused(request, store, host.clone()).await {
        Ok((runner, paused)) => {
            assert_eq!(host.calls.load(Ordering::SeqCst), 0);
            let paused = approve(&runner, &paused.run).await;
            let completed = answer(&runner, &paused.run).await.unwrap();
            assert_eq!(completed.run.status, TaskRunStatus::Completed);
            assert_eq!(completed.run.counters.llm_attempts, 2);
            assert_eq!(host.effects.load(Ordering::SeqCst), 0);
            println!("rc-winner");
            drop(runner);
        }
        Err(error) => {
            assert!(mode.starts_with("race-"), "{error}");
            assert_eq!(host.stages.load(Ordering::SeqCst), 0);
            assert_eq!(host.calls.load(Ordering::SeqCst), 0);
            println!("rc-loser");
        }
    }
    drop(inner);
    storage.close().await;
}
