use super::*;

// A backward UTC step at resume cannot put a new active marker outside its acknowledged envelope or refill monotonic usage.
#[tokio::test]
async fn host_evidence_resume_clock_anchor_survives_wall_clock_rollback() {
    for sqlite in super::super::continuation_test_support::backends() {
        let (agent, _, _) = super::super::runner_tests::fixture(
            super::super::runner_tests::RecordingProvider::new(&[]),
            1,
        );
        let store =
            super::super::continuation_test_support::store(agent.info().id, "config-v1", *sqlite)
                .await;
        let mut initial = super::super::tests::checkpoint("clock-rollback");
        initial.created_at += chrono::Duration::seconds(60);
        initial.updated_at = initial.created_at;
        initial.payload["clocks"]["active_millis"] = json!(37);
        initial.payload["clocks"]["interrupted_interval_millis"] = json!(11);
        store.create(&initial).await.unwrap();
        let claimed = store
            .mutate(
                &initial.key.run_id,
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
            )
            .await
            .unwrap();
        let owner = agent
            .reserve_autonomy_run(initial.key.run_id.clone())
            .await
            .unwrap();
        let payload: TaskCheckpointPayload =
            serde_json::from_value(claimed.payload.clone()).unwrap();
        let execution = RunExecution::new(
            store,
            &claimed,
            LifecycleState::new(&payload.settings, false).unwrap(),
            owner.clone(),
            std::time::Instant::now(),
        )
        .unwrap();
        let saved = execution
            .update(|payload| {
                payload.clocks.active_interval_started_at = Some(chrono::Utc::now());
                Ok(())
            })
            .await
            .unwrap();
        let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
        assert_eq!(
            payload.clocks.active_interval_started_at,
            Some(claimed.updated_at)
        );
        assert_eq!(payload.clocks.interrupted_interval_millis, 11);
        assert!(payload.clocks.active_millis >= 37);
        agent.release_autonomy_run(&owner).await.unwrap();
    }
}

// Exact checkpoint identity must survive JSON transport for arbitrary finite callback metrics and timing metadata.
#[test]
fn host_evidence_json_identity_survives_finite_metrics() {
    let mut bits = 1u64;
    for _ in 0..100_000 {
        bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
        let metric =
            f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1020 + ((bits >> 52) % 8)) << 52);
        let expected = json!({"metric":metric});
        let text = serde_json::to_string(&expected).unwrap();
        assert_eq!(
            expected,
            serde_json::from_str::<Value>(&text).unwrap(),
            "transport changed finite metric: {text}"
        );
    }
}

// Failure origin, not the public error enum variant, determines whether custody can be released.
#[tokio::test]
async fn host_evidence_io_failures_racing_timeout_keep_original_error_and_owner() {
    for phase in ["round", "terminal", "pause"] {
        let (mut f, callbacks) = configured(
            Revision::new(),
            if phase == "pause" {
                "repeat"
            } else {
                "external"
            },
            ExternalUnavailablePolicy::Stop,
        );
        f.config
            .defaults
            .validation
            .as_mut()
            .unwrap()
            .checks
            .as_mut()
            .unwrap()[0]
            .timeout_seconds = Some(1);
        let fault = install_fault(&mut f, phase, "io", None);
        let first = f.runner.run("objective", None).await.unwrap();
        fault.armed.store(true, Ordering::SeqCst);
        let mut input = signal(&first.run);
        if phase == "pause"
            && let TaskResumeInput::ExternalSignal(signal) = &mut input
        {
            signal.payload = json!(false);
        }
        let error = f
            .runner
            .resume(&first.run.key.run_id, first.run.revision, input)
            .await
            .unwrap_err();
        assert!(matches!(error, AgentError::IoError(_)), "{error}");
        assert_eq!(
            callbacks.load(Ordering::SeqCst),
            if phase == "round" { 1 } else { 2 }
        );
        assert!(f.agent.chat("blocked").await.is_err());
        assert_eq!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskRunStatus::Running
        );
    }
}

// A target change while finalization is entering storage cannot create durable Completed state.
#[tokio::test]
async fn host_evidence_final_admission_rechecks_target_inside_store_boundary() {
    for sqlite in super::super::continuation_test_support::backends() {
        let source = Revision::new();
        let (mut f, callbacks) = fixture(
            *sqlite,
            source.clone(),
            "external",
            ExternalUnavailablePolicy::Stop,
        )
        .await;
        let fault = install_fault(&mut f, "terminal", "before", Some(source));
        let first = f.runner.run("objective", None).await.unwrap();
        fault.armed.store(true, Ordering::SeqCst);
        assert!(
            f.runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    signal(&first.run)
                )
                .await
                .is_err()
        );
        assert!(!fault.armed.load(Ordering::SeqCst));
        assert_ne!(
            f.store
                .load(&first.run.key.run_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskRunStatus::Completed
        );
        assert_eq!(callbacks.load(Ordering::SeqCst), 2);
        assert!(f.agent.chat("blocked").await.is_err());
    }
}

#[derive(Debug)]
struct BackendGuard {
    calls: AtomicUsize,
    allowed: usize,
}
impl TaskFinalAdmission for BackendGuard {
    fn validate(&self) -> Result<()> {
        if self.calls.fetch_add(1, Ordering::SeqCst) >= self.allowed {
            return Err(AgentError::Config(
                "target changed at backend admission".into(),
            ));
        }
        Ok(())
    }
}

// Memory samples within its lock; SQLite additionally samples before commit and rolls the transaction back on denial.
#[tokio::test]
async fn host_evidence_backend_guards_reject_late_proof_without_consuming_revision() {
    use ai_agents_core::AgentStorage;
    for sqlite in super::super::continuation_test_support::backends() {
        let storage: Arc<dyn AgentStorage> = if *sqlite {
            #[cfg(feature = "sqlite")]
            {
                Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap())
            }
            #[cfg(not(feature = "sqlite"))]
            {
                unreachable!()
            }
        } else {
            Arc::new(ai_agents_storage::InMemoryTaskStorage::default())
        };
        let initial = super::super::tests::checkpoint("evidence-final");
        storage.create_task_run(&initial).await.unwrap();
        let running = storage
            .mutate_task_run(
                &initial.key,
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
            )
            .await
            .unwrap();
        let guard = Arc::new(BackendGuard {
            calls: AtomicUsize::new(0),
            allowed: if *sqlite { 2 } else { 1 },
        });
        let mutation = TaskRunMutation::FinalCheckpoint {
            expected_revision: running.revision,
            owner_token: "owner".into(),
            status: TaskRunStatus::Completed,
            payload: running.payload.clone(),
            release: true,
            expires_at: None,
            deadline: Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
            evidence: Some(guard.clone()),
        };
        mutation
            .validate_final_deadline(chrono::Utc::now())
            .unwrap();
        assert!(
            storage
                .mutate_task_run(&initial.key, &mutation)
                .await
                .is_err()
        );
        assert_eq!(guard.calls.load(Ordering::SeqCst), guard.allowed + 1);
        let retained = storage.load_task_run(&initial.key).await.unwrap().unwrap();
        assert_eq!(retained.status, TaskRunStatus::Running);
        assert_eq!(retained.revision, running.revision);
    }
}

struct ManualExecutor {
    generation: Option<Arc<AtomicUsize>>,
}
#[async_trait]
impl ValidationObservationExecutor for ManualExecutor {
    fn capture_scope(&self, scope: &EvaluationScope) -> Result<EvaluationScope> {
        let mut scope = scope.clone();
        if let Some(generation) = &self.generation {
            scope.mutation_generation = generation.load(Ordering::SeqCst) as u64;
        }
        Ok(scope)
    }
    async fn execute(
        &self,
        _: &ValidationObservationRequest,
        _: &EvidenceIdentity,
    ) -> Result<ObservationResult> {
        panic!("pure completion must not request effects")
    }
}
struct MemoryJournal;
#[async_trait]
impl ValidationJournal for MemoryJournal {
    async fn checkpoint(&self, _: &ValidationDriverState) -> Result<()> {
        Ok(())
    }
}

// The public driver enforces the actual frozen source even when a custom executor only clones caller-supplied scope.
#[tokio::test]
async fn host_evidence_direct_driver_rejects_forged_targets_and_stale_cached_results() {
    for scenario in ["source", "binding", "cached"] {
        let source = Revision::new();
        let (f, callbacks) =
            configured(source.clone(), "complete", ExternalUnavailablePolicy::Stop);
        let profile = resolve_profile(
            &f.config,
            AutonomyScope::Task,
            None,
            None,
            None,
            Some("objective"),
            &Default::default(),
        )
        .unwrap();
        let bound = f
            .agent
            .autonomy_extensions()
            .bind_for_agent(&f.agent, &profile)
            .unwrap();
        let mut scope = super::super::evaluation_tests::scope();
        bound.prepare_scope(&mut scope);
        bound.capture_targets(&mut scope).unwrap();
        let check = &bound.checks[0];
        let mut identity = super::super::evaluation_tests::identity(&scope);
        identity.config_identity = check.config_identity.clone();
        identity.target = Some("ticket".into());
        identity.target_revision = Some("r1".into());
        if scenario == "binding" {
            identity.target = Some("forged".into());
            scope.target_revisions.insert("forged".into(), "r1".into());
        }
        let mut driver = ValidationDriverState::new(check, identity).unwrap();
        let generation = Arc::new(AtomicUsize::new(0));
        let executor = ManualExecutor {
            generation: (scenario == "cached").then_some(generation.clone()),
        };
        let cancellation =
            ai_agents_core::ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
        let evidence = super::super::evaluation_tests::current(&scope);
        let context = ValidationDriveContext {
            scope: &scope,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &executor,
            journal: &MemoryJournal,
            limits: Default::default(),
            cancellation: &cancellation,
            deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(10)),
        };
        if scenario == "source" {
            source.change();
        }
        if scenario == "cached" {
            assert!(matches!(
                driver.drive(check, &context).await.unwrap(),
                ValidationDriverOutcome::Complete(_)
            ));
            generation.store(1, Ordering::SeqCst);
        }
        assert!(driver.drive(check, &context).await.is_err());
        assert!(!driver.publication_acknowledged());
        assert_eq!(
            callbacks.load(Ordering::SeqCst),
            usize::from(scenario == "cached")
        );
    }
}

// An unrelated source cannot satisfy a required check; optional unavailable checks remain Unknown without callbacks.
#[tokio::test]
async fn host_evidence_required_and_optional_revision_capabilities_are_check_specific() {
    for required in [false, true] {
        let callbacks = Arc::new(AtomicUsize::new(0));
        let observed = callbacks.clone();
        let mut f =
            super::super::controller_tests::fixture_configured("external", move |registry| {
                registry
                    .replace_validator(Arc::new(Validator {
                        descriptor: AdapterDescriptor {
                            id: "host.observation".into(),
                            contract_version: 1,
                            requires_host_revision: true,
                            needs_judge: true,
                            ..Default::default()
                        },
                        calls: observed,
                        mode: "complete",
                    }))
                    .unwrap();
                registry
                    .register_validation_target("unrelated", "other", Revision::new())
                    .unwrap();
            });
        f.config
            .defaults
            .validation
            .as_mut()
            .unwrap()
            .checks
            .as_mut()
            .unwrap()[0]
            .required = Some(required);
        f.config.defaults.completion = Some(CompletionGate::ResponseContains("done".into()));
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            f.config.clone(),
            Default::default(),
            f.store.clone(),
            "controller.v1".into(),
        )
        .unwrap();
        let result = f.runner.run("objective", None).await;
        if required {
            assert!(result.is_err());
            assert!(f.store.list().await.unwrap().is_empty());
        } else {
            assert_eq!(result.unwrap().run.status, TaskRunStatus::Completed);
        }
        assert_eq!(callbacks.load(Ordering::SeqCst), 0);
    }
}

// Persisted cancellation during a successful journal ACK must be visible before any next callback or result collection.
#[tokio::test]
async fn host_evidence_cancel_during_round_publication_and_rewait_ack() {
    for sqlite in super::super::continuation_test_support::backends() {
        for phase in ["round", "publish", "rewait"] {
            let (mut f, callbacks) = fixture(
                *sqlite,
                Revision::new(),
                if phase == "rewait" {
                    "repeat"
                } else {
                    "external"
                },
                ExternalUnavailablePolicy::Stop,
            )
            .await;
            let fault = install_fault(&mut f, phase, "gate", None);
            let first = f.runner.run("objective", None).await.unwrap();
            fault.armed.store(true, Ordering::SeqCst);
            let mut input = signal(&first.run);
            if phase == "rewait"
                && let TaskResumeInput::ExternalSignal(signal) = &mut input
            {
                signal.payload = json!(false);
            }
            let mut future = Box::pin(f.runner.resume(
                &first.run.key.run_id,
                first.run.revision,
                input,
            ));
            tokio::select! { result = &mut future => panic!("returned before gate: {:?}", result.as_ref().err()), permit = fault.entered.acquire() => permit.unwrap().forget(), _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("gate not reached") }
            let saved = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
            f.store
                .mutate(
                    &saved.key.run_id,
                    &TaskRunMutation::RequestCancel {
                        expected_revision: saved.revision,
                    },
                )
                .await
                .unwrap();
            fault.release.add_permits(1);
            let done = future.await.unwrap();
            assert_eq!(done.run.status, TaskRunStatus::Cancelled);
            assert_eq!(
                callbacks.load(Ordering::SeqCst),
                if phase == "round" { 1 } else { 2 }
            );
            assert_eq!(f.agent.chat("ordinary").await.unwrap().content, "ordinary");
        }
    }
}

// Every control competes on the same acknowledged revision; a limit winner keeps the original signal request alive.
#[tokio::test]
async fn host_evidence_signal_control_races_have_one_winner() {
    for sqlite in super::super::continuation_test_support::backends() {
        for action in ["limits", "objective", "stop", "cancel", "unavailable"] {
            for response_first in [false, true] {
                let (f, callbacks) = fixture(
                    *sqlite,
                    Revision::new(),
                    "external",
                    ExternalUnavailablePolicy::Stop,
                )
                .await;
                let first = f.runner.run("objective", None).await.unwrap();
                let id = &first.run.key.run_id;
                let control = async {
                    match action {
                        "limits" => {
                            let mut limits = first.run.limits.clone();
                            limits.max_turns += 1;
                            f.runner
                                .revise_limits_paused(id, first.run.revision, limits, "capacity")
                                .await
                        }
                        "objective" => {
                            f.runner
                                .modify_objective_paused(
                                    id,
                                    first.run.revision,
                                    "new",
                                    "change work",
                                )
                                .await
                        }
                        "stop" => {
                            f.runner
                                .stop_paused(id, first.run.revision, "stop work")
                                .await
                        }
                        "cancel" => f.runner.cancel_paused(id, first.run.revision).await,
                        _ => {
                            f.runner
                                .external_unavailable_paused(
                                    id,
                                    first.run.revision,
                                    &first.run.pending.as_ref().unwrap().id,
                                    "channel lost",
                                )
                                .await
                        }
                    }
                };
                let response = f.runner.resume(id, first.run.revision, signal(&first.run));
                let (control, response) = if response_first {
                    let (response, control) = tokio::join!(response, control);
                    (control, response)
                } else {
                    tokio::join!(control, response)
                };
                assert_ne!(
                    control.is_ok(),
                    response.is_ok(),
                    "{action}/{response_first}"
                );
                if let Ok(result) = control {
                    if action == "limits" {
                        assert_eq!(
                            result.run.pending.as_ref().unwrap().id,
                            first.run.pending.as_ref().unwrap().id
                        );
                        let done = f
                            .runner
                            .resume(id, result.run.revision, signal(&result.run))
                            .await
                            .unwrap();
                        assert_eq!(done.run.status, TaskRunStatus::Completed);
                    } else if action == "objective" {
                        assert!(
                            f.runner
                                .resume(id, result.run.revision, signal(&first.run))
                                .await
                                .is_err()
                        );
                        f.runner
                            .stop_paused(id, result.run.revision, "close revised run")
                            .await
                            .unwrap();
                    }
                }
                assert_eq!(f.calls.load(Ordering::SeqCst), 0);
                assert!(callbacks.load(Ordering::SeqCst) <= 2);
                assert_eq!(f.agent.chat("ordinary").await.unwrap().content, "ordinary");
            }
        }
    }
}

// A dropped acknowledgement never becomes permission to retry a callback, reissue a wait or reuse a completed receipt.
#[tokio::test]
async fn host_evidence_drop_at_consumption_rewait_publication_and_terminal_keeps_custody() {
    for sqlite in super::super::continuation_test_support::backends() {
        for phase in ["accept", "round", "publish", "rewait", "pause", "terminal"] {
            let mode = if matches!(phase, "rewait" | "pause") {
                "repeat"
            } else {
                "external"
            };
            let (mut f, callbacks) = fixture(
                *sqlite,
                Revision::new(),
                mode,
                ExternalUnavailablePolicy::Stop,
            )
            .await;
            let fault = install_fault(&mut f, phase, "gate", None);
            let first = f.runner.run("objective", None).await.unwrap();
            fault.armed.store(true, Ordering::SeqCst);
            let mut input = signal(&first.run);
            if mode == "repeat"
                && let TaskResumeInput::ExternalSignal(signal) = &mut input
            {
                signal.payload = json!(false);
            }
            let mut future = Box::pin(f.runner.resume(
                &first.run.key.run_id,
                first.run.revision,
                input,
            ));
            tokio::select! {
                result = &mut future => panic!("returned before {phase} acknowledgement: {:?}", result.as_ref().err()),
                permit = fault.entered.acquire() => permit.unwrap().forget(),
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("{phase} boundary not reached"),
            }
            drop(future);
            assert_eq!(
                callbacks.load(Ordering::SeqCst),
                if matches!(phase, "accept" | "round") {
                    1
                } else {
                    2
                }
            );
            assert!(f.agent.chat("blocked").await.is_err());
            let saved = f.store.load(&first.run.key.run_id).await.unwrap().unwrap();
            assert!(
                f.runner
                    .resume(&saved.key.run_id, saved.revision, signal(&first.run))
                    .await
                    .is_err()
            );
        }
    }
}

// Failed re-wait issuance and final pause receipts are not usable pauses, even if the new request reached storage.
#[tokio::test]
async fn host_evidence_rewait_receipt_faults_never_expose_new_authority() {
    for sqlite in super::super::continuation_test_support::backends() {
        for phase in ["rewait", "pause"] {
            for mode in ["reject", "no_write", "lost", "stale", "payload", "owner"] {
                let (mut f, callbacks) = fixture(
                    *sqlite,
                    Revision::new(),
                    "repeat",
                    ExternalUnavailablePolicy::Stop,
                )
                .await;
                let fault = install_fault(&mut f, phase, mode, None);
                let first = f.runner.run("objective", None).await.unwrap();
                fault.armed.store(true, Ordering::SeqCst);
                let mut input = signal(&first.run);
                if let TaskResumeInput::ExternalSignal(signal) = &mut input {
                    signal.payload = json!(false);
                }
                let error = f
                    .runner
                    .resume(&first.run.key.run_id, first.run.revision, input)
                    .await
                    .unwrap_err();
                assert!(
                    !fault.armed.load(Ordering::SeqCst),
                    "{sqlite}/{phase}/{mode} did not reach selected write: {error}"
                );
                assert_eq!(callbacks.load(Ordering::SeqCst), 2);
                assert!(f.agent.chat("blocked").await.is_err());
            }
        }
    }
}

// Configured unavailable cleanup must acknowledge its real terminal write before freeing the runtime.
#[tokio::test]
async fn host_evidence_unavailable_cleanup_receipts_and_drop_keep_protection() {
    for sqlite in super::super::continuation_test_support::backends() {
        for mode in [
            "reject", "no_write", "lost", "stale", "payload", "owner", "gate",
        ] {
            let (mut f, callbacks) = fixture(
                *sqlite,
                Revision::new(),
                "external",
                ExternalUnavailablePolicy::Fail,
            )
            .await;
            let fault = install_fault(&mut f, "terminal", mode, None);
            let first = f.runner.run("objective", None).await.unwrap();
            fault.armed.store(true, Ordering::SeqCst);
            let mut future = Box::pin(f.runner.external_unavailable_paused(
                &first.run.key.run_id,
                first.run.revision,
                &first.run.pending.as_ref().unwrap().id,
                "delivery lost",
            ));
            if mode == "gate" {
                tokio::select! { result = &mut future => panic!("returned before ack: {:?}", result.as_ref().err()), permit = fault.entered.acquire() => permit.unwrap().forget(), _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => panic!("cleanup boundary not reached") }
                drop(future);
            } else {
                assert!(future.await.is_err(), "{mode}");
            }
            assert!(!fault.armed.load(Ordering::SeqCst));
            assert_eq!(callbacks.load(Ordering::SeqCst), 1);
            assert!(f.agent.chat("blocked").await.is_err());
        }
    }
}

// Stage review freezes execution credit, but cannot turn a stale target result into current proof.
#[tokio::test]
async fn host_evidence_stage_review_rechecks_original_target_before_and_after_claim() {
    for sqlite in super::super::continuation_test_support::backends() {
        for after_claim in [false, true] {
            let source = Revision::new();
            let (mut f, callbacks) = fixture(
                *sqlite,
                source.clone(),
                "complete",
                ExternalUnavailablePolicy::Stop,
            )
            .await;
            f.config.defaults.lifecycle =
                Some(serde_json::from_value(json!([{"id":"work"},{"id":"verify"}])).unwrap());
            f.config.defaults.supervision = Some(SupervisionConfig {
                stage_changes: Some(ReviewPolicy::RequireApproval),
                plan_changes: None,
            });
            let fault = install_fault(&mut f, "claim", "change", Some(source.clone()));
            let first = f.runner.run("objective", None).await.unwrap();
            assert!(matches!(
                first.run.pending.as_ref().unwrap().kind,
                TaskPendingKind::Approval
            ));
            if after_claim {
                fault.armed.store(true, Ordering::SeqCst);
            } else {
                source.change();
            }
            let outcome = f
                .runner
                .resume(
                    &first.run.key.run_id,
                    first.run.revision,
                    TaskResumeInput::Approval {
                        request_id: first.run.pending.as_ref().unwrap().id.clone(),
                        result: ai_agents_hitl::ApprovalResult::Approved,
                    },
                )
                .await;
            if after_claim {
                let result = outcome.unwrap();
                assert_eq!(result.run.status, TaskRunStatus::Incomplete);
            } else {
                assert!(outcome.is_err());
                assert_eq!(
                    f.store
                        .load(&first.run.key.run_id)
                        .await
                        .unwrap()
                        .unwrap()
                        .revision,
                    first.run.revision
                );
                f.runner
                    .cancel_paused(&first.run.key.run_id, first.run.revision)
                    .await
                    .unwrap();
            }
            assert_eq!(callbacks.load(Ordering::SeqCst), 1);
        }
    }
}

// Unrelated registry targets are not sampled, and source failure after parking does not prevent objective retirement.
#[tokio::test]
async fn host_evidence_objective_revision_and_unrelated_targets_preserve_bindings() {
    for sqlite in super::super::continuation_test_support::backends() {
        let relevant = Revision::new();
        let unrelated = Revision::new();
        unrelated.unavailable.store(true, Ordering::SeqCst);
        let a = relevant.clone();
        let b = unrelated.clone();
        let mut f =
            super::super::controller_tests::fixture_configured("external", move |registry| {
                registry
                    .register_validation_target("check", "ticket", a)
                    .unwrap();
                registry
                    .register_validation_target("unused", "other", b)
                    .unwrap();
            });
        f.store = super::super::continuation_test_support::store(
            f.agent.info().id,
            "controller.v1",
            *sqlite,
        )
        .await;
        f.config.defaults.max_llm_calls = Some(2);
        f.runner = AutonomyRunner::try_new(
            f.agent.clone(),
            f.config.clone(),
            Default::default(),
            f.store.clone(),
            "controller.v1".into(),
        )
        .unwrap();
        let first = f.runner.run("old", None).await.unwrap();
        assert_eq!(unrelated.reads.load(Ordering::SeqCst), 0);
        relevant.unavailable.store(true, Ordering::SeqCst);
        let revised = f
            .runner
            .modify_objective_paused(
                &first.run.key.run_id,
                first.run.revision,
                "new",
                "retire stale proof",
            )
            .await
            .unwrap();
        assert!(
            f.runner
                .resume(
                    &revised.run.key.run_id,
                    revised.run.revision,
                    signal(&first.run)
                )
                .await
                .is_err()
        );
        relevant.unavailable.store(false, Ordering::SeqCst);
        relevant.change();
        let next = f
            .runner
            .resume(
                &revised.run.key.run_id,
                revised.run.revision,
                TaskResumeInput::Continue,
            )
            .await
            .unwrap();
        let done = f
            .runner
            .resume(
                &next.run.key.run_id,
                next.run.revision,
                TaskResumeInput::ExternalSignal(ExternalSignal {
                    request_id: next.run.pending.as_ref().unwrap().id.clone(),
                    event_id: "new-event".into(),
                    target_revision: Some("r2".into()),
                    payload: json!(true),
                }),
            )
            .await
            .unwrap();
        assert_eq!(done.run.status, TaskRunStatus::Completed);
        assert_eq!(done.run.counters.turns, 2);
        assert_eq!(unrelated.reads.load(Ordering::SeqCst), 0);
    }
}
