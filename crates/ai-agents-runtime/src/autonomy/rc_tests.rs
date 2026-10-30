//! Recovery acceptance uses actual stores, prepared runtimes and independent process witnesses; host proofs remain explicit fixtures.
#[path = "rc_followup_tests.rs"]
mod followup;
use super::*;
use ai_agents_core::{AgentError, Result};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

struct RecoveryProof {
    source: Value,
    allow: AtomicBool,
    now: parking_lot::Mutex<Option<chrono::DateTime<chrono::Utc>>>,
}
impl RecoveryProof {
    // The fixture certifies a reviewed exact snapshot only after its controlled work has actually stopped.
    fn new(source: &TaskRunSnapshot) -> Arc<Self> {
        Arc::new(Self {
            source: serde_json::to_value(source).unwrap(),
            allow: AtomicBool::new(true),
            now: Default::default(),
        })
    }
}
impl TaskRecoveryHost for RecoveryProof {
    // IDs alone do not authorize recovery; the proof binds the complete original snapshot and can be revoked before commit.
    fn validate(&self, snapshot: &TaskRunSnapshot, _: &TaskRecoveryDecision) -> Result<()> {
        if !self.allow.load(Ordering::SeqCst) || serde_json::to_value(snapshot)? != self.source {
            return Err(AgentError::Config(
                "fixture recovery proof revoked or mismatched".into(),
            ));
        }
        Ok(())
    }
    // Controlled samples exercise the same timestamp domain rather than an independent Instant-derived epoch.
    fn now(&self) -> Result<chrono::DateTime<chrono::Utc>> {
        Ok(self.now.lock().unwrap_or_else(chrono::Utc::now))
    }
    // Recovery itself is active work even when its source was safely paused.
    fn recovery_budget(&self) -> Duration {
        Duration::from_secs(1)
    }
}

// Every decision is authenticated Rust input bound to the observed source, never deserialized authority.
fn decision(
    source: &TaskRunSnapshot,
    disposition: TaskRecoveryDisposition,
) -> TaskRecoveryDecision {
    TaskRecoveryDecision {
        expected_revision: source.revision,
        reason: "reviewed isolated fixture effects and stop".into(),
        stopped_at: None,
        interrupted_started_at: None,
        disposition,
        effects: Default::default(),
        children: Default::default(),
    }
}

// Persistence is selected through the existing declarative profile rather than a second runner configuration.
fn persistent_config(step: bool) -> AutonomyConfig {
    let mut config = config();
    config.defaults.persistence = Some(AutonomyPersistenceConfig {
        enabled: Some(true),
        save_after_each_step: Some(step),
        resume_on_start: Some(true),
    });
    config
}

// All reconstructed invocations receive current config and ceilings; no serialized provider authority is installed.
fn request(run: &TaskRun, config: AutonomyConfig) -> TaskRestoreRequest {
    TaskRestoreRequest {
        run_id: run.key.run_id.clone(),
        expected_revision: run.revision,
        config_identity: "restore.v1".into(),
        config,
        ceilings: Default::default(),
    }
}

// Current profile settings automatically prepare a genuine restart point, preserve paid/write consumption, and keep original decisions separate.
#[tokio::test]
async fn rc_persistence_startup_and_paid_pending_execution_preserve_original_ledger() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let mut host = host();
        Arc::get_mut(&mut host).unwrap().priced = true;
        let root = host.runtime(host.root.clone());
        let store =
            crate::autonomy::continuation_test_support::store(root.info().id, "restore.v1", sqlite)
                .await;
        let mut config = persistent_config(false);
        config.defaults.max_cost_usd = Some(UsdAmount::parse("0.000017").unwrap());
        config.defaults.max_declared_write_paths = Some(1);
        let ceilings = AutonomyHostCeilings {
            max_cost_usd: config.defaults.max_cost_usd.clone(),
            max_declared_write_paths: Some(1),
            ..Default::default()
        };
        let runner = AutonomyRunner::try_new(
            root,
            config.clone(),
            ceilings.clone(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap()
        .with_restore_host(host.clone());
        let paused = runner
            .run("verify durable support evidence", None)
            .await
            .unwrap();
        assert_eq!(paused.run.status, TaskRunStatus::Paused);
        let before = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
        assert!(
            before.payload["adapters"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a["id"] == "runtime.restore")
        );
        let charges = before.payload["counters"]["charged_micro_usd"].clone();
        let targets = before.payload["declared_write_targets"].clone();
        let old_request = paused.run.pending.as_ref().unwrap().id.clone();
        let calls = host.calls.load(Ordering::SeqCst);
        drop(runner);
        let (restored, prepared) = AutonomyRunner::restore_on_start(
            TaskStartupRequest {
                config,
                ceilings,
                config_identity: "restore.v1".into(),
                profile: None,
            },
            store.clone(),
            host.clone(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(host.calls.load(Ordering::SeqCst), calls);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
        assert_ne!(prepared.run.pending.as_ref().unwrap().id, old_request);
        assert!(
            answer(&restored, &prepared.run).await.is_err(),
            "restoration must be reviewed separately"
        );
        let saved = store.load(&prepared.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(saved.payload["counters"]["charged_micro_usd"], charges);
        assert_eq!(saved.payload["declared_write_targets"], targets);
        let prepared = approve(&restored, &prepared.run).await;
        let completed = answer(&restored, &prepared.run).await.unwrap();
        assert_eq!(completed.run.status, TaskRunStatus::Completed);
        let saved = store
            .load(&completed.run.key.run_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(saved.payload["counters"]["charged_micro_usd"], 14);
        assert_eq!(saved.payload["declared_write_targets"], targets);
        assert_eq!(host.effects.load(Ordering::SeqCst), 1);
    }
}

// False suppresses only the optional settled-step snapshot; mandatory admission, effects, runtime and terminal data remain exact.
#[tokio::test]
async fn rc_persistence_step_policy_and_preflight_preserve_ordinary_defaults() {
    for step in [false, true] {
        let mut host = host();
        Arc::get_mut(&mut host).unwrap().root = AgentSpec::from_yaml_strict(
            "name: RestoreRoot\nsystem_prompt: restore-skill-root\ntools: []\n",
        )
        .unwrap();
        let root = host.runtime(host.root.clone());
        let store = Arc::new(
            ScopedTaskRunStore::in_memory(root.info().id, None, "restore.v1".into()).unwrap(),
        );
        let runner = AutonomyRunner::try_new(
            root.clone(),
            persistent_config(step),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap();
        assert!(runner.run("tool-free analysis", None).await.is_err());
        assert!(store.list().await.unwrap().is_empty());
        let runner = runner.with_restore_host(host.clone());
        let result = runner.run("tool-free analysis", None).await.unwrap();
        assert_eq!(result.run.status, TaskRunStatus::Completed);
        let saved = store.load(&result.run.key.run_id).await.unwrap().unwrap();
        assert_eq!(
            saved.payload["adapters"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a["id"] == "runtime.persistence_step"),
            step
        );
        assert_eq!(saved.payload["reservations"].as_array().unwrap().len(), 1);
        assert!(
            saved.payload["reservations"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r["state"] == "completed")
        );
        assert!(root.set_context("ordinary", json!(true)).is_ok());
    }
}

// Ambiguity never falls back to a latest timestamp, and no candidate means no provider construction or new task.
#[tokio::test]
async fn rc_startup_zero_multiple_and_disabled_candidates_never_guess() {
    let host = host();
    let store = Arc::new(
        ScopedTaskRunStore::in_memory("RestoreRoot".into(), None, "restore.v1".into()).unwrap(),
    );
    let startup = || TaskStartupRequest {
        config: persistent_config(false),
        ceilings: Default::default(),
        config_identity: "restore.v1".into(),
        profile: None,
    };
    assert!(
        AutonomyRunner::restore_on_start(startup(), store.clone(), host.clone())
            .await
            .unwrap()
            .is_none()
    );
    let mut runs = vec![];
    for _ in 0..2 {
        let runner = AutonomyRunner::try_new(
            host.runtime(host.root.clone()),
            persistent_config(false),
            Default::default(),
            store.clone(),
            "restore.v1".into(),
        )
        .unwrap()
        .with_restore_host(host.clone());
        runs.push(runner.run("support", None).await.unwrap().run);
        drop(runner);
    }
    let calls = host.calls.load(Ordering::SeqCst);
    assert!(
        AutonomyRunner::restore_on_start(startup(), store.clone(), host.clone())
            .await
            .is_err()
    );
    assert_eq!(host.stages.load(Ordering::SeqCst), 0);
    assert_eq!(host.calls.load(Ordering::SeqCst), calls);
    for run in runs {
        assert_eq!(
            store.load(&run.key.run_id).await.unwrap().unwrap().revision,
            run.revision
        );
    }
}

// An interrupted claim or constructor can recover only its unchanged prepared cursor after explicit stop review, without replaying its effects.
#[tokio::test]
async fn rc_reconciled_claim_and_staging_resume_pending_without_replay() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for staged in [false, true] {
            let (host, store, prepared) = prepared_fixture(sqlite).await;
            let id = prepared.run.key.run_id.clone();
            if staged {
                host.stage_hold.store(true, Ordering::SeqCst);
                let task = tokio::spawn({
                    let store = store.clone();
                    let host = host.clone();
                    let request = request(&prepared.run, config());
                    async move { AutonomyRunner::restore_paused(request, store, host).await }
                });
                host.stage_reached.acquire().await.unwrap().forget();
                task.abort();
                assert!(matches!(task.await, Err(error) if error.is_cancelled()));
                host.stage_hold.store(false, Ordering::SeqCst);
            } else {
                store
                    .mutate(
                        &id,
                        &TaskRunMutation::Claim {
                            expected_revision: prepared.run.revision,
                            owner_token: "abandoned-constructor".into(),
                        },
                    )
                    .await
                    .unwrap();
            }
            let source = store.load(&id).await.unwrap().unwrap();
            assert_eq!(source.status, TaskRunStatus::Running);
            let proof = RecoveryProof::new(&source);
            let decision = decision(&source, TaskRecoveryDisposition::ResumePrepared);
            let effects = host.effects.load(Ordering::SeqCst);
            let calls = host.calls.load(Ordering::SeqCst);
            let recovered = AutonomyRunner::reconcile_recovery(store.clone(), &id, decision, proof)
                .await
                .unwrap();
            assert_eq!(recovered.status, TaskRunStatus::Paused);
            let (runner, paused) = AutonomyRunner::restore_paused(
                TaskRestoreRequest {
                    expected_revision: recovered.revision,
                    ..request(&prepared.run, config())
                },
                store.clone(),
                host.clone(),
            )
            .await
            .unwrap();
            assert_eq!(host.calls.load(Ordering::SeqCst), calls);
            let paused = approve(&runner, &paused.run).await;
            let result = answer(&runner, &paused.run).await.unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(host.effects.load(Ordering::SeqCst), effects);
        }
    }
}

// A controlled persisted open interval distinguishes sampled usage from cancel-request metadata and retains exact paid/write commitments.
async fn recovery_clock_fixture(
    sqlite: bool,
) -> (
    Arc<ScopedTaskRunStore>,
    TaskRunSnapshot,
    chrono::DateTime<chrono::Utc>,
) {
    let host = host();
    let root = host.runtime(host.root.clone());
    let store =
        crate::autonomy::continuation_test_support::store(root.info().id, "restore.v1", sqlite)
            .await;
    let now = chrono::Utc::now();
    let profile = resolve_profile(
        &config(),
        AutonomyScope::Task,
        None,
        None,
        None,
        Some("effects"),
        &Default::default(),
    )
    .unwrap();
    let mut payload = TaskCheckpointPayload::new(
        "effects".into(),
        "restore.v1".into(),
        &profile,
        TaskRuntimeCheckpoint::between_turns(root.save_state_full().await.unwrap()).unwrap(),
    );
    payload.clocks.active_millis = 3000;
    payload.clocks.active_interval_started_at = Some(now - chrono::Duration::seconds(10));
    payload.clocks.expires_at = Some(now + chrono::Duration::seconds(60));
    payload.adapters.push(TaskAdapterCheckpoint {
        id: "runtime.recovery_clock".into(),
        adapter: "runtime.recovery_clock".into(),
        contract_version: 1,
        config: json!({"contract":"sampled_active_utc_v1"}),
        state: json!({"sampled_at":now-chrono::Duration::seconds(7),"active_millis":3000}),
    });
    payload.counters.llm_attempts = 1;
    payload.counters.charged_micro_usd = 3;
    payload.declared_write_targets.push("ticket:fixture".into());
    payload.reservations.push(TaskReservation {
        id: "known".into(),
        state: TaskEffectState::Completed,
        reserved_micro_usd: 3,
        charged_micro_usd: 3,
        write_targets: vec![],
        result: Some(json!({"actual":"known"})),
    });
    payload.reservations.push(TaskReservation {
        id: "unknown".into(),
        state: TaskEffectState::Dispatched,
        reserved_micro_usd: 17,
        charged_micro_usd: 0,
        write_targets: vec!["ticket:fixture".into()],
        result: None,
    });
    let source = payload
        .bind(TaskRunSnapshot {
            schema_version: TASK_RUN_SCHEMA_VERSION,
            key: TaskRunKey {
                agent_id: "RestoreRoot".into(),
                run_id: uuid::Uuid::new_v4().to_string(),
            },
            actor_id: None,
            revision: 0,
            status: TaskRunStatus::Running,
            owner_token: Some("reviewed-abandoned".into()),
            cancel_requested: false,
            created_at: now - chrono::Duration::seconds(10),
            updated_at: now - chrono::Duration::seconds(3),
            payload: Value::Null,
        })
        .unwrap();
    store.create(&source).await.unwrap();
    (store, source, now)
}

// Explicit stop evidence bounds the interrupted interval; metadata-only cancellation cannot erase it or turn reconciliation into completion.
#[tokio::test]
async fn rc_interrupted_clock_and_effect_reconciliation_are_monotonic_and_audit_only() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        let (store, source, now) = recovery_clock_fixture(sqlite).await;
        let source = store
            .mutate(
                &source.key.run_id,
                &TaskRunMutation::RequestCancel {
                    expected_revision: source.revision,
                },
            )
            .await
            .unwrap();
        let proof = RecoveryProof::new(&source);
        *proof.now.lock() = Some(now + chrono::Duration::seconds(1));
        let mut decision = decision(&source, TaskRecoveryDisposition::Stop);
        decision.stopped_at = Some(now - chrono::Duration::seconds(2));
        decision.effects.insert(
            "unknown".into(),
            json!({"ticket_revision":"verified-applied"}),
        );
        let saved = AutonomyRunner::reconcile_recovery(
            store.clone(),
            &source.key.run_id,
            decision.clone(),
            proof.clone(),
        )
        .await
        .unwrap();
        assert_eq!(saved.status, TaskRunStatus::Cancelled);
        let saved = store.load(&source.key.run_id).await.unwrap().unwrap();
        assert_eq!(saved.payload["clocks"]["interrupted_interval_millis"], 5000);
        assert_eq!(saved.payload["clocks"]["active_millis"], 4000);
        assert_eq!(
            saved.payload["clocks"]["expires_at"],
            source.payload["clocks"]["expires_at"]
        );
        assert_eq!(saved.payload["counters"]["charged_micro_usd"], 20);
        assert_eq!(saved.payload["counters"]["llm_attempts"], 1);
        assert_eq!(
            saved.payload["declared_write_targets"],
            source.payload["declared_write_targets"]
        );
        assert_eq!(
            saved.payload["reservations"][0],
            source.payload["reservations"][0]
        );
        assert_eq!(
            saved.payload["reservations"][1]["result"]["task_result_eligible"],
            false
        );
        assert!(
            AutonomyRunner::reconcile_recovery(store.clone(), &source.key.run_id, decision, proof)
                .await
                .is_err()
        );
        assert_eq!(
            store
                .load(&source.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .revision,
            saved.revision
        );
    }
}

// Missing outcomes, stale proof, invalid clocks and invented operation IDs never produce a safe pause or release existing commitments.
#[tokio::test]
async fn rc_recovery_rejects_incomplete_mismatched_and_invalid_clock_decisions_before_write() {
    for fault in [
        "missing",
        "unknown",
        "revoked",
        "rollback",
        "stop_before_sample",
        "resume_uncertain",
    ] {
        let (store, source, now) = recovery_clock_fixture(false).await;
        let proof = RecoveryProof::new(&source);
        *proof.now.lock() = Some(now + chrono::Duration::seconds(1));
        let mut decision = decision(&source, TaskRecoveryDisposition::Stop);
        decision.effects.insert("unknown".into(), json!("verified"));
        match fault {
            "missing" => {
                decision.effects.clear();
            }
            "unknown" => {
                decision
                    .effects
                    .insert("another-operation".into(), json!(true));
            }
            "revoked" => proof.allow.store(false, Ordering::SeqCst),
            "rollback" => {
                *proof.now.lock() = Some(source.updated_at - chrono::Duration::milliseconds(1))
            }
            "stop_before_sample" => decision.stopped_at = Some(now - chrono::Duration::seconds(8)),
            "resume_uncertain" => {
                decision.disposition = TaskRecoveryDisposition::ResumePrepared;
                decision.effects.clear();
            }
            _ => unreachable!(),
        }
        assert!(
            AutonomyRunner::reconcile_recovery(store.clone(), &source.key.run_id, decision, proof)
                .await
                .is_err(),
            "{fault}"
        );
        assert_eq!(
            serde_json::to_value(store.load(&source.key.run_id).await.unwrap().unwrap()).unwrap(),
            serde_json::to_value(source).unwrap()
        );
    }
}

struct RecoveryFaultStore {
    inner: Arc<ScopedTaskRunStore>,
    mode: &'static str,
    reached: tokio::sync::Semaphore,
    proof: Arc<RecoveryProof>,
}
#[async_trait]
impl TaskRunStore for RecoveryFaultStore {
    // All ordinary writes remain the production conditional journal.
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
    // Only the reviewed recovery publication is faulted; the opaque checkpoint and real backend guard are preserved.
    fn supports_recovery(&self) -> bool {
        true
    }
    async fn checkpoint_recovery(
        &self,
        id: &str,
        c: &TaskRecoveryCheckpoint,
    ) -> Result<TaskRunSnapshot> {
        if self.mode == "reject" {
            return Err(AgentError::Persistence(
                "injected recovery rejection".into(),
            ));
        }
        if self.mode == "revoke" {
            self.proof.allow.store(false, Ordering::SeqCst);
        }
        if self.mode == "cancel" {
            let source = self
                .inner
                .load(id)
                .await?
                .ok_or(TaskRunStorageError::NotFound)?;
            self.inner
                .mutate(
                    id,
                    &TaskRunMutation::RequestCancel {
                        expected_revision: source.revision,
                    },
                )
                .await?;
        }
        let saved = self.inner.checkpoint_recovery(id, c).await?;
        self.reached.add_permits(1);
        if self.mode == "drop" {
            std::future::pending::<()>().await;
        }
        if self.mode == "lost" {
            return Err(AgentError::Persistence("injected lost recovery ACK".into()));
        }
        Ok(saved)
    }
}

// Failed/lost/dropped publication and a proof revoked at the backend boundary never authorize replay or fabricate an acknowledgement.
#[tokio::test]
async fn rc_recovery_publication_faults_preserve_authoritative_phase_and_single_settlement() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mode in ["reject", "lost", "drop", "revoke"] {
            let (inner, source, now) = recovery_clock_fixture(sqlite).await;
            let proof = RecoveryProof::new(&source);
            *proof.now.lock() = Some(now + chrono::Duration::seconds(1));
            let store = Arc::new(RecoveryFaultStore {
                inner: inner.clone(),
                mode,
                reached: tokio::sync::Semaphore::new(0),
                proof: proof.clone(),
            });
            let mut decision = decision(&source, TaskRecoveryDisposition::Stop);
            decision
                .effects
                .insert("unknown".into(), json!({"verified":"applied"}));
            let mut work = Box::pin(AutonomyRunner::reconcile_recovery(
                store.clone(),
                &source.key.run_id,
                decision,
                proof,
            ));
            if mode == "drop" {
                tokio::select! { result=&mut work => panic!("publication was not held: {result:?}"), permit=store.reached.acquire()=>permit.unwrap().forget() }
                drop(work);
            } else {
                assert!(work.await.is_err(), "{mode}");
            }
            let saved = inner.load(&source.key.run_id).await.unwrap().unwrap();
            if mode == "lost" || mode == "drop" {
                assert_eq!(saved.status, TaskRunStatus::Incomplete);
                assert_eq!(saved.payload["counters"]["charged_micro_usd"], 20);
            } else {
                assert_eq!(saved.status, TaskRunStatus::RecoveryRequired);
                assert_eq!(
                    saved.payload["reservations"],
                    source.payload["reservations"]
                );
            }
        }
    }
}

// A real admitted provider future must stop before reviewed cleanup; the framework does not infer that fact from abort alone.
#[tokio::test]
async fn rc_actual_abandoned_provider_reconciles_without_replay_and_releases_only_after_ack() {
    let provider = crate::autonomy::runner_tests::RecordingProvider::gated();
    let (agent, runner, store) = crate::autonomy::runner_tests::fixture(provider.clone(), 2);
    let runner = Arc::new(runner);
    let task = tokio::spawn({
        let runner = runner.clone();
        async move { runner.run("tool-free analysis", None).await }
    });
    provider
        .entered
        .as_ref()
        .unwrap()
        .acquire()
        .await
        .unwrap()
        .forget();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let source = store
        .load(&store.list().await.unwrap()[0].key.run_id)
        .await
        .unwrap()
        .unwrap();
    let proof = RecoveryProof::new(&source);
    let mut decision = decision(&source, TaskRecoveryDisposition::Stop);
    for r in source.payload["reservations"].as_array().unwrap() {
        if r["state"] != "completed" {
            decision.effects.insert(
                r["id"].as_str().unwrap().into(),
                json!({"provider":"stopped-no-returned-evidence"}),
            );
        }
    }
    let reconciled =
        AutonomyRunner::reconcile_recovery(store.clone(), &source.key.run_id, decision, proof)
            .await
            .unwrap();
    assert_eq!(reconciled.status, TaskRunStatus::Incomplete);
    assert!(agent.set_context("still_protected", json!(true)).is_err());
    runner
        .acknowledge_recovery_release(&source.key.run_id, reconciled.revision)
        .await
        .unwrap();
    assert!(agent.set_context("released", json!(true)).is_ok());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}
