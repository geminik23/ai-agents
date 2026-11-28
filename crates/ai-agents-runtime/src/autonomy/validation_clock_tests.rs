use super::super::evaluation_tests::{bound_script, current, identity};
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

// Simulates a backwards wall-clock projection without changing the process clock or waiting for a real UTC adjustment.
#[tokio::test]
async fn retained_clock_restore_keeps_earlier_monotonic_deadline() {
    let (mut bound, mut scope, calls) = bound_script("external");
    bound.checks[0].check.timeout_seconds = Some(10);
    bound.checks[0].config_identity =
        canonical_identity(&serde_json::to_value(&bound.checks[0].check).unwrap()).unwrap();
    bound.prepare_scope(&mut scope);
    let check = &bound.checks[0];
    let mut identity = identity(&scope);
    identity.config_identity = check.config_identity.clone();
    let mut state = ValidationDriverState::new(check, identity).unwrap();
    let original = std::time::Instant::now() - std::time::Duration::from_millis(1);
    state.check_projection = Some(original);
    state.retained = Some(Arc::new(RetainedValidation {
        identity: state.identity.clone(),
        binding_identity: state.binding_identity.clone(),
        deadline: state.check_deadline,
        projection: Some(original),
    }));
    let json = serde_json::to_value(&state).unwrap();
    let audit =
        ValidationDriverState::restore(json.clone(), check, &scope, &bound.extensions).unwrap();
    assert!(!audit.expired());
    let mut retained = ValidationDriverState::restore_retained(
        json.clone(),
        &state,
        check,
        &scope,
        &bound.extensions,
    )
    .unwrap();
    assert!(retained.expired());
    assert_eq!(retained.check_projection, Some(original));
    assert!(
        ValidationDriverState::restore_retained(json, &audit, check, &scope, &bound.extensions)
            .is_err()
    );
    let cancellation =
        ai_agents_core::ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let evidence = current(&scope);
    let journal = super::super::evaluation_tests::MemoryJournal::default();
    struct NoEffects;
    #[async_trait::async_trait]
    impl ValidationObservationExecutor for NoEffects {
        async fn execute(
            &self,
            _: &ValidationObservationRequest,
            _: &EvidenceIdentity,
        ) -> ai_agents_core::Result<ObservationResult> {
            panic!("expired retained validation must not execute")
        }
    }
    let executor = NoEffects;
    assert!(
        retained
            .drive(
                check,
                &ValidationDriveContext {
                    scope: &scope,
                    evidence: &evidence,
                    extensions: &bound.extensions,
                    executor: &executor,
                    journal: &journal,
                    limits: Default::default(),
                    cancellation: &cancellation,
                    deadline: None
                }
            )
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

struct DeadlineFreeExecutor {
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl ValidationObservationExecutor for DeadlineFreeExecutor {
    // A truly unbounded drive must not manufacture an invocation timeout before dispatching its observation.
    async fn execute(
        &self,
        request: &ValidationObservationRequest,
        _: &EvidenceIdentity,
    ) -> ai_agents_core::Result<ObservationResult> {
        assert!(ai_agents_llm::current_invocation_bounds().is_none());
        self.calls.fetch_add(1, Ordering::SeqCst);
        let ValidationObservationRequest::Tool {
            tool, arguments, ..
        } = request
        else {
            panic!("unexpected observation")
        };
        Ok(ObservationResult::Tool {
            record: Box::new(super::super::evaluation_tests::record(
                tool,
                arguments.clone(),
                serde_json::json!({"value": 1}),
            )),
        })
    }
}

// No aggregate or check deadline allows progress without dropping journal publication or scoped evidence.
#[tokio::test]
async fn deadline_free_validation_completes_without_fabricated_invocation_bounds() {
    let (bound, scope, calls) = bound_script("complete");
    let check = &bound.checks[0];
    let mut identity = identity(&scope);
    identity.config_identity = check.config_identity.clone();
    let mut state = ValidationDriverState::new(check, identity).unwrap();
    let cancellation =
        ai_agents_core::ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let evidence = current(&scope);
    let journal = super::super::evaluation_tests::MemoryJournal::default();
    let executor = DeadlineFreeExecutor {
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let context = ValidationDriveContext {
        scope: &scope,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancellation,
        deadline: None,
    };
    assert!(state.check_deadline.is_none());
    assert!(state.check_projection.is_none());
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        state.drive(check, &context),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        matches!(outcome, ValidationDriverOutcome::Complete(result) if result.outcome == GateOutcome::Pass)
    );
    assert!(state.publication_acknowledged());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

struct CancellingJournal {
    flag: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl ValidationJournal for CancellingJournal {
    // Cancellation arriving with the pre-dispatch ACK must stop a deadline-free drive before its first effect.
    async fn checkpoint(&self, state: &ValidationDriverState) -> ai_agents_core::Result<()> {
        if state.in_flight.is_some() {
            self.flag.store(true, Ordering::SeqCst);
        }
        Ok(())
    }
}

// Removing the time restriction never removes cancellation checks before or after journal admission.
#[tokio::test]
async fn deadline_free_validation_rechecks_cancellation_after_journal() {
    for cancelled_before in [false, true] {
        let (bound, scope, _) = bound_script("complete");
        let check = &bound.checks[0];
        let mut identity = identity(&scope);
        identity.config_identity = check.config_identity.clone();
        let mut state = ValidationDriverState::new(check, identity).unwrap();
        let flag = Arc::new(AtomicBool::new(cancelled_before));
        let cancellation = ai_agents_core::ToolCancellationToken::new(flag.clone(), None);
        let evidence = current(&scope);
        let journal = CancellingJournal { flag };
        let executor = DeadlineFreeExecutor {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let context = ValidationDriveContext {
            scope: &scope,
            evidence: &evidence,
            extensions: &bound.extensions,
            executor: &executor,
            journal: &journal,
            limits: Default::default(),
            cancellation: &cancellation,
            deadline: None,
        };
        assert!(state.drive(check, &context).await.is_err());
        assert!(!state.publication_acknowledged());
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    }
}

// A parent monotonic restriction cannot be discarded by a context with no aggregate UTC deadline.
#[tokio::test]
async fn deadline_free_validation_keeps_inherited_monotonic_expiry() {
    let (bound, scope, calls) = bound_script("complete");
    let check = &bound.checks[0];
    let mut identity = identity(&scope);
    identity.config_identity = check.config_identity.clone();
    let mut state = ValidationDriverState::new(check, identity).unwrap();
    let cancellation =
        ai_agents_core::ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let evidence = current(&scope);
    let journal = super::super::evaluation_tests::MemoryJournal::default();
    let executor = DeadlineFreeExecutor {
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let context = ValidationDriveContext {
        scope: &scope,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancellation,
        deadline: None,
    };
    let bounds = ai_agents_llm::InvocationBounds::new(
        std::time::SystemTime::now() + std::time::Duration::from_secs(30),
        std::time::Instant::now() - std::time::Duration::from_millis(1),
    );
    assert!(
        ai_agents_llm::scope_invocation_bounds(Some(bounds), state.drive(check, &context))
            .await
            .is_err()
    );
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
