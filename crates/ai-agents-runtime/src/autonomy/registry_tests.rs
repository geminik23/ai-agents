use super::evaluation_tests::{current, identity, scope};
use super::*;
use ai_agents_core::{AgentError, FinishReason, LLMResponse, Result, ToolCancellationToken};
use ai_agents_llm::mock::MockLLMProvider;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64},
    },
};

struct Declared {
    descriptor: AdapterDescriptor,
}
impl AutonomyValidator for Declared {
    /// Dependencies remain explicit metadata, not recursively evaluated callbacks.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    /// Domain config remains object-shaped and cannot overwrite framework binding fields.
    fn validate_config(&self, c: &Value) -> Result<()> {
        if c.is_object() {
            Ok(())
        } else {
            Err(AgentError::Config("shape".into()))
        }
    }
    /// Pure fixtures expose registry binding behavior without hidden provider or tool calls.
    fn evaluate(&self, _: &ValidationInput<'_>) -> Result<ValidationDecision> {
        Ok(ValidationDecision::Complete {
            outcome: GateOutcome::Pass,
            reason: "fixture".into(),
            metrics: json!({}),
            evidence_refs: vec![],
        })
    }
}
impl ProgressAdapter for Declared {
    /// Progress capability needs are checked at the same binding boundary as validators.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    /// Only registered implementations interpret their extension config.
    fn validate_config(&self, _: &Value) -> Result<()> {
        Ok(())
    }
    /// Empty observations carry no fabricated advancement signals.
    fn observe(&self, _: &ProgressInput<'_>) -> Result<ProgressObservation> {
        Ok(ProgressObservation {
            metrics: json!({}),
            signals: BTreeMap::new(),
            comparisons: BTreeMap::new(),
            evidence_refs: vec![],
        })
    }
}

// Duplicate IDs, forward/cyclic inputs, unknown versions and missing progress prerequisites fail before callbacks.
#[test]
fn registry_freezes_versions_and_rejects_dependency_and_capability_gaps() {
    let mut registry = AutonomyExtensions::builtins();
    let adapter = Arc::new(Declared {
        descriptor: AdapterDescriptor {
            id: "host.check".into(),
            contract_version: 1,
            checks: vec!["later".into()],
            ..Default::default()
        },
    });
    registry.register_validator(adapter.clone()).unwrap();
    assert!(registry.register_validator(adapter).is_err());
    let profile:AutonomyProfile=serde_yaml::from_str("validation: {checks: [{id: first, adapter: host.check}, {id: later, adapter: host.check}]}" ).unwrap();
    assert!(
        registry
            .freeze()
            .bind(&profile, &Default::default())
            .is_err()
    );
    let mut registry = AutonomyExtensions::builtins();
    registry
        .register_progress(Arc::new(Declared {
            descriptor: AdapterDescriptor {
                id: "host.progress".into(),
                contract_version: 1,
                tools: vec!["unavailable".into()],
                ..Default::default()
            },
        }))
        .unwrap();
    let profile: AutonomyProfile =
        serde_yaml::from_str("progress: {adapter: host.progress}").unwrap();
    assert!(
        registry
            .freeze()
            .bind(&profile, &Default::default())
            .is_err()
    );
    let profile: AutonomyProfile =
        serde_yaml::from_str("progress: {adapter: builtin.evidence, contract_version: 2}").unwrap();
    assert!(
        AutonomyExtensions::builtins()
            .freeze()
            .bind(&profile, &Default::default())
            .is_err()
    );
    let profile:AutonomyProfile=serde_yaml::from_str("progress: {adapter: builtin.evidence, config: {signals: [{id: cycle, source: progress, path: metrics.score, comparison: increase}]}}").unwrap();
    assert!(
        AutonomyExtensions::builtins()
            .freeze()
            .bind(&profile, &Default::default())
            .is_err()
    );
}

// Verified completion of a new canonical item advances, while repeated or prior-objective items do not.
#[test]
fn canonical_todo_progress_advances_and_progress_conflicts_are_rejected() {
    let profile: AutonomyProfile =
        serde_yaml::from_str("progress: {adapter: builtin.todo}").unwrap();
    let bound = AutonomyExtensions::builtins()
        .freeze()
        .bind(&profile, &Default::default())
        .unwrap();
    let mut s = scope();
    let store = ai_agents_tools::TodoStore::default();
    let adapter = RunTodoAdapter::begin(store.clone(), &s.key).unwrap();
    store.set(vec![ai_agents_tools::TodoItem {
        id: "a".into(),
        content: "work".into(),
        active_form: None,
        status: ai_agents_tools::TodoStatus::Pending,
    }]);
    let mut e = current(&s);
    e.current.as_mut().unwrap().value.todos = Some(adapter.checkpoint().unwrap());
    let mut state = ProgressState::new(s.clone());
    state
        .observe(&s, &bound.observe_progress(&s, &e).unwrap())
        .unwrap();
    store.update(
        "a",
        None,
        None,
        Some(ai_agents_tools::TodoStatus::Completed),
    );
    s.cycle += 1;
    e.current.as_mut().unwrap().identity.cycle = s.cycle;
    e.current.as_mut().unwrap().value.todos = Some(adapter.checkpoint().unwrap());
    assert_eq!(
        state
            .observe(&s, &bound.observe_progress(&s, &e).unwrap())
            .unwrap(),
        ProgressDelta::Advanced
    );
    assert_eq!(state.cycles_without_progress, 0);
    s.cycle += 1;
    e.current.as_mut().unwrap().identity.cycle = s.cycle;
    assert_eq!(
        state
            .observe(&s, &bound.observe_progress(&s, &e).unwrap())
            .unwrap(),
        ProgressDelta::Unchanged
    );
    let profile:AutonomyProfile=serde_yaml::from_str("validation: {checks: [{id: check, adapter: builtin.evidence, config: {assertion: {path: count, gte: 1}}}]}").unwrap();
    let bound = AutonomyExtensions::builtins()
        .freeze()
        .bind(&profile, &Default::default())
        .unwrap();
    bound.prepare_scope(&mut s);
    let mut i = identity(&s);
    i.config_identity = bound.checks[0].config_identity.clone();
    let passed = ValidationResult {
        check_id: "check".into(),
        identity: i,
        outcome: GateOutcome::Pass,
        reason: "verified".into(),
        metrics: json!({}),
        evidence_refs: vec![],
    };
    let mut failed = passed.clone();
    failed.outcome = GateOutcome::Fail;
    e.required_checks = vec!["check".into()];
    e.validations = vec![passed.clone(), failed.clone()];
    assert!(bound.observe_progress(&s, &e).is_err());
    e.validations = vec![failed, passed.clone()];
    assert!(bound.observe_progress(&s, &e).is_err());
    e.validations = vec![passed.clone(), passed];
    assert!(bound.observe_progress(&s, &e).is_ok());
}

// Alternate todo names must reference this runtime's actual canonical authority rather than a JSON tool lookalike.
#[test]
fn canonical_todo_alias_binding_checks_actual_store_identity() {
    let store = ai_agents_tools::TodoStore::default();
    let registry = ai_agents_tools::ToolRegistry::new();
    let canonical = registry.todo_store();
    let mut extensions = AutonomyExtensions::builtins();
    extensions
        .register_todo_binding("tasks".into(), store)
        .unwrap();
    let agent = crate::AgentBuilder::new()
        .system_prompt("test")
        .llm(Arc::new(MockLLMProvider::new("default")))
        .tools(registry)
        .autonomy_extensions(extensions)
        .build()
        .unwrap();
    let mut profile = resolve_profile(
        &AutonomyConfig::default(),
        AutonomyScope::Task,
        None,
        None,
        None,
        None,
        &AutonomyHostCeilings::default(),
    )
    .unwrap();
    profile.settings.progress = Some(ProgressConfig {
        todo_tool: Some("tasks".into()),
        ..Default::default()
    });
    assert!(
        agent
            .autonomy_extensions()
            .bind_for_agent(&agent, &profile)
            .is_err()
    );
    assert!(canonical.shares_store(&canonical.clone()));
}

// A selected bounded intervention uses the existing planner once and cached resume does not generate another plan.
#[tokio::test]
async fn bounded_replanning_uses_existing_planner_without_reset_or_replay() {
    let mut provider = MockLLMProvider::new("planner");
    provider.add_response(LLMResponse::new("{\"steps\":[{\"id\":\"one\",\"description\":\"Revise plan\",\"action_type\":\"think\",\"action_target\":\"reason\",\"dependencies\":[]}]}",FinishReason::Stop));
    let provider = Arc::new(provider);
    let agent = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("test")
            .llm(provider.clone())
            .build()
            .unwrap(),
    );
    let profile:AutonomyProfile=serde_yaml::from_str("progress: {stagnation: {max_cycles_without_progress: 1, action: replan, max_replans: 1, on_exhausted: stop}}").unwrap();
    let bound = agent
        .autonomy_extensions()
        .bind(
            &profile,
            &ValidationCapabilities {
                planner: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut s = scope();
    let mut progress = ProgressState::new(s.clone());
    progress.cycles_without_progress = 1;
    assert_eq!(
        progress
            .stagnation(
                profile.progress.as_ref().unwrap().stagnation.as_ref(),
                false,
                true
            )
            .unwrap(),
        Some(StagnationAction::Replan)
    );
    let check = bound.replan_check(&progress, "objective").unwrap();
    s.validation_bindings
        .insert(check.check.id.clone(), check.config_identity.clone());
    s.validation_attempts
        .insert(check.check.id.clone(), check.check.id.clone());
    let mut i = identity(&s);
    i.config_identity = check.config_identity.clone();
    i.attempt = Some(check.check.id.clone());
    let mut driver = ValidationDriverState::new(&check, i).unwrap();
    let executor = RuntimeObservationExecutor {
        agent,
        profile: None,
        scope_mode: "task".into(),
        run_revision: Arc::new(AtomicU64::new(0)),
        validator: check.check.adapter.clone(),
        contract_version: 1,
    };
    let journal = super::evaluation_tests::MemoryJournal::default();
    let evidence = current(&s);
    let cancel = ToolCancellationToken::new(Arc::new(AtomicBool::new(false)), None);
    let ctx = ValidationDriveContext {
        scope: &s,
        evidence: &evidence,
        extensions: &bound.extensions,
        executor: &executor,
        journal: &journal,
        limits: Default::default(),
        cancellation: &cancel,
        deadline: chrono::Utc::now() + chrono::Duration::seconds(2),
    };
    driver.drive(&check, &ctx).await.unwrap();
    driver.drive(&check, &ctx).await.unwrap();
    assert_eq!(provider.call_count(), 1);
    assert_eq!(progress.replans, 1);
    assert_eq!(progress.cycles_without_progress, 1);
}
