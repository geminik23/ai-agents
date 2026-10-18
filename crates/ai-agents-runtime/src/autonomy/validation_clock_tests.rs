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
                    deadline: chrono::Utc::now() + chrono::Duration::seconds(10)
                }
            )
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
