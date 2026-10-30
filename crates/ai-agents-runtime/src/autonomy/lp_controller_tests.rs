use super::continuation_test_support as support;
use super::runner_tests::RecordingProvider;
use super::*;
use crate::AgentBuilder;
use ai_agents_core::autonomy::{ProviderCostBound, ProviderCostSettlement, ProviderRequest};
use ai_agents_core::{ChatMessage, LLMConfig, LLMError, LLMFeature, LLMProvider, LLMResponse};
use ai_agents_hitl::ApprovalResult;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{Arc, atomic::Ordering};

struct Check {
    descriptor: AdapterDescriptor,
    calls: Arc<parking_lot::Mutex<Vec<(u64, String)>>>,
}
impl AutonomyValidator for Check {
    // Domain outcomes are observations, not tool effects or model-written completion claims.
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    // Opaque domain fixture values affect observations only and install no execution authority.
    fn validate_config(&self, _: &Value) -> ai_agents_core::Result<()> {
        Ok(())
    }
    // Cycle-labelled records distinguish scheduled execution from repeated callback or controller work.
    fn evaluate(&self, input: &ValidationInput<'_>) -> ai_agents_core::Result<ValidationDecision> {
        self.calls.lock().push((
            input.scope.cycle,
            input.config["name"].as_str().unwrap_or("check").into(),
        ));
        Ok(ValidationDecision::Complete {
            outcome: match input.config["outcome"].as_str() {
                Some("fail") => GateOutcome::Fail,
                Some("unknown") => GateOutcome::Unknown,
                _ => GateOutcome::Pass,
            },
            reason: "domain observation".into(),
            metrics: json!({"count": input.config["values"].as_array().and_then(|values| values.get(input.scope.cycle.saturating_sub(1) as usize)).cloned().unwrap_or(json!(input.scope.cycle)), "ready": true}),
            evidence_refs: vec![],
        })
    }
}

type DomainFixture = (
    Arc<crate::RuntimeAgent>,
    Arc<RecordingProvider>,
    AutonomyConfig,
    Arc<parking_lot::Mutex<Vec<(u64, String)>>>,
);

// Every case uses the production registry and runner, but no network, command or filesystem tool.
fn configured(replies: &[&str], yaml: &str) -> DomainFixture {
    let provider = RecordingProvider::new(replies);
    let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut extensions = AutonomyExtensions::builtins();
    extensions
        .register_validator(Arc::new(Check {
            descriptor: AdapterDescriptor {
                id: "host.domain".into(),
                contract_version: 1,
                ..Default::default()
            },
            calls: calls.clone(),
        }))
        .unwrap();
    let agent = Arc::new(
        AgentBuilder::new()
            .system_prompt("domain task")
            .llm(provider.clone())
            .autonomy_extensions(extensions)
            .build()
            .unwrap(),
    );

    // Parse the complete selected fixture without involving an enabled agent YAML declaration.
    let mut parsed: AutonomyConfig = serde_yaml::from_str(yaml).unwrap();
    parsed.defaults.enabled = Some(true);
    if parsed.defaults.max_turns.is_none() {
        parsed.defaults.max_turns = Some(12);
    }
    if parsed.defaults.max_llm_calls.is_none() {
        parsed.defaults.max_llm_calls = Some(12);
    }
    if parsed.defaults.completion.is_none() {
        parsed.defaults.completion = Some(CompletionGate::ResponseContains("done".into()));
    }
    (agent, provider, parsed, calls)
}

// Check schedule is observable independently of whether the model happens to say done early.
#[tokio::test]
async fn lp_schedules_stage_end_and_completion_only_at_their_candidate_boundaries() {
    for sqlite in support::backends() {
        let (agent, provider, config, calls) = configured(
            &["draft", "ready", "done"],
            r#"
validation:
  checks:
    - {id: cycle, adapter: host.domain, schedule: each_cycle, config: {name: cycle}}
    - {id: final, adapter: host.domain, schedule: completion, config: {name: final}}
lifecycle:
  - id: work
    completion: {response_contains: ready}
    validation:
      checks: [{id: stage, adapter: host.domain, schedule: stage_end, config: {name: stage}}]
  - id: review
completion: {response_contains: done}
"#,
        );
        let store = support::store(agent.info().id, "lp.v1", *sqlite).await;
        let runner =
            AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
                .unwrap();
        let done = runner.run("report", None).await.unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(
            *calls.lock(),
            vec![
                (1, "cycle".into()),
                (2, "cycle".into()),
                (2, "stage".into()),
                (3, "cycle".into()),
                (3, "final".into())
            ]
        );
        assert_eq!(done.run.counters.turns, 3);
        assert_eq!(done.run.counters.continuations, 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    }
}

// An unavailable validator gate must allow the very observation needed to prove it, without an extra model call.
#[tokio::test]
async fn lp_stage_validator_gate_can_start_its_own_deferred_validation() {
    let (agent, provider, config, calls) = configured(
        &["done"],
        r#"
max_llm_calls: 1
lifecycle:
  - id: verify
    completion: {validator_passed: stage}
    validation:
      checks: [{id: stage, adapter: host.domain, schedule: stage_end}]
"#,
    );
    let store = support::store(agent.info().id, "lp.v1", false).await;
    let done = AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
        .unwrap()
        .run("report", None)
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(calls.lock().len(), 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Current-cycle progress wins over both stagnation and inability to admit another provider call.
#[tokio::test]
async fn lp_current_progress_completes_at_the_last_slot_without_a_followup_turn() {
    let (agent, provider, config, _) = configured(
        &["analysis"],
        r#"
max_llm_calls: 1
validation:
  checks: [{id: domain, adapter: host.domain}]
progress:
  adapter: builtin.evidence
  config:
    signals: [{id: count, validator: domain, path: metrics.count, comparison: increase}]
  stagnation: {max_cycles_without_progress: 1, action: fail}
completion: {progress_path: {path: metrics.signals.count, gte: 1}}
"#,
    );
    let store = support::store(agent.info().id, "lp.v1", false).await;
    let done = AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
        .unwrap()
        .run("analysis", None)
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Completed);
    assert_eq!(done.run.progress.observation_sequence, 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// An optional failed domain observation does not erase independently verified escalation.
#[tokio::test]
async fn lp_verified_escalation_is_a_valid_alternative_to_failed_health() {
    let (agent, provider, config, calls) = configured(
        &["escalation recorded"],
        r#"
max_llm_calls: 1
lifecycle:
  - id: verify
    validation:
      checks:
        - {id: healthy, adapter: host.domain, required: false, schedule: stage_end, config: {name: health, outcome: fail}}
        - {id: escalated, adapter: host.domain, required: false, schedule: stage_end, config: {name: escalation}}
    completion: {any: [{validator_passed: healthy}, {validator_passed: escalated}]}
completion: {any: [{validator_passed: healthy}, {validator_passed: escalated}]}
"#,
    );
    let store = support::store(agent.info().id, "lp.v1", false).await;
    let done = AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
        .unwrap()
        .run("incident", None)
        .await
        .unwrap();
    assert_eq!(
        done.run.status,
        TaskRunStatus::Completed,
        "{:?}",
        done.run.stop_reason
    );
    assert_eq!(calls.lock().len(), 2);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

// Repair is bounded lifecycle work, not a premature finish, and its instruction must not leak into verification.
#[tokio::test]
async fn lp_fix_work_is_classified_once_and_does_not_spend_forced_continuations() {
    for sqlite in support::backends() {
        let (agent, provider, config, _) = configured(
            &["work", "bad", "work", "verified done"],
            r#"
max_llm_calls: 4
max_turns: 4
premature_finish: {action: fail}
lifecycle:
  - id: work
  - id: verify
    completion: {response_contains: verified}
    retry_on_failure: true
    on_failure_stage: work
    max_fix_cycles: 1
"#,
        );
        let store = support::store(agent.info().id, "lp.v1", *sqlite).await;
        let done =
            AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
                .unwrap()
                .run("repair", None)
                .await
                .unwrap();
        assert_eq!(
            done.run.status,
            TaskRunStatus::Completed,
            "{:?}",
            done.run.stop_reason
        );
        assert_eq!(done.run.counters.continuations, 0);
        assert_eq!(done.run.counters.turns, 4);
        let messages = provider.captured_messages();
        assert!(
            messages[2]
                .iter()
                .any(|message| message.content.contains("Required validation failed"))
        );
        assert!(
            !messages[3]
                .iter()
                .any(|message| message.content.contains("Required validation failed"))
        );
    }
}

struct Priced(Arc<RecordingProvider>);
#[async_trait]
impl LLMProvider for Priced {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<LLMResponse, LLMError> {
        self.0.complete(messages, config).await
    }
    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: Option<&LLMConfig>,
    ) -> std::result::Result<
        Box<
            dyn futures::Stream<Item = std::result::Result<ai_agents_core::LLMChunk, LLMError>>
                + Unpin
                + Send,
        >,
        LLMError,
    > {
        self.0.complete_stream(messages, config).await
    }
    fn provider_name(&self) -> &str {
        "lp-priced-fixture"
    }
    fn supports(&self, _: LLMFeature) -> bool {
        false
    }
    fn priced_capability_identity(&self) -> Option<String> {
        Some("lp.price.v1".into())
    }
    fn request_cost_bound(
        &self,
        request: &ProviderRequest<'_>,
    ) -> std::result::Result<Option<ProviderCostBound>, LLMError> {
        Ok(Some(ProviderCostBound {
            provider_identity: "lp.price.v1".into(),
            pricing_identity: "fixed.v1".into(),
            request_id: request.request_id.into(),
            max_micro_usd: 10,
        }))
    }
    fn settle_request_cost(
        &self,
        bound: &ProviderCostBound,
        _: &LLMResponse,
    ) -> std::result::Result<Option<ProviderCostSettlement>, LLMError> {
        Ok(Some(ProviderCostSettlement {
            bound: bound.clone(),
            charged_micro_usd: 10,
        }))
    }
}

// Actual priced refusal reaches the dynamic planner driver, unlike a top-level count preflight.
#[tokio::test]
async fn lp_replan_capacity_resume_preserves_the_selected_decision_and_original_work() {
    for sqlite in support::backends() {
        for review in [false, true] {
            let plan = r#"{"steps":[{"id":"one","description":"Verify sources","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
            let provider = RecordingProvider::new(&["working", plan, "done"]);
            let agent = Arc::new(
                AgentBuilder::new()
                    .system_prompt("research")
                    .llm(Arc::new(Priced(provider.clone())))
                    .build()
                    .unwrap(),
            );
            let store = support::store(agent.info().id, "lp.v1", *sqlite).await;
            let mut config = super::runner_tests::config(5);
            config.defaults.max_cost_usd = Some(UsdAmount::parse("0.000010").unwrap());
            config.defaults.on_limit = Some(serde_yaml::from_str("action: pause_run").unwrap());
            config.defaults.progress = Some(serde_yaml::from_str("stagnation: {max_cycles_without_progress: 1, action: replan, max_replans: 1, on_exhausted: stop}").unwrap());
            if review {
                config.defaults.supervision = Some(SupervisionConfig {
                    stage_changes: None,
                    plan_changes: Some(ReviewPolicy::RequireApproval),
                });
            }
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                config,
                AutonomyHostCeilings {
                    max_cost_usd: Some(UsdAmount::parse("0.000030").unwrap()),
                    ..Default::default()
                },
                store.clone(),
                "lp.v1".into(),
            )
            .unwrap();
            let paused = runner.run("research", None).await.unwrap();
            assert_eq!(
                paused.run.status,
                TaskRunStatus::Paused,
                "{:?}",
                paused.run.stop_reason
            );
            assert!(matches!(
                paused.run.pending.as_ref().unwrap().kind,
                TaskPendingKind::Limit
            ));
            let saved = store.load(&paused.run.key.run_id).await.unwrap().unwrap();
            let payload: TaskCheckpointPayload = serde_json::from_value(saved.payload).unwrap();
            let state = controller_checkpoint(&payload).unwrap().unwrap();
            assert_eq!(state.phase, super::controller::ControllerPhase::Replan);
            assert_eq!(state.progress.replans, 1);
            assert_eq!(state.progress.sequence, 1);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let mut limits = payload.limits;
            limits.max_micro_usd = Some(30);
            let revised = runner
                .revise_limits_paused(
                    &paused.run.key.run_id,
                    paused.run.revision,
                    limits,
                    "allow planner and next work",
                )
                .await
                .unwrap();
            let mut done = runner
                .resume(
                    &revised.run.key.run_id,
                    revised.run.revision,
                    TaskResumeInput::Continue,
                )
                .await
                .unwrap();
            if review {
                assert_eq!(done.run.status, TaskRunStatus::Paused);
                assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
                done = runner
                    .resume(
                        &done.run.key.run_id,
                        done.run.revision,
                        TaskResumeInput::Approval {
                            request_id: done.run.pending.as_ref().unwrap().id.clone(),
                            result: ApprovalResult::Approved,
                        },
                    )
                    .await
                    .unwrap();
            }
            assert_eq!(
                done.run.status,
                TaskRunStatus::Completed,
                "{:?}",
                done.run.stop_reason
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
            assert_eq!(done.run.counters.turns, 2);
            assert_eq!(done.run.counters.llm_attempts, 3);
            assert_eq!(done.run.progress.replans, 1);
            assert_eq!(done.run.progress.observation_sequence, 2);
            assert_eq!(done.run.counters.charged_micro_usd, 30);
            let final_payload: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&done.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            assert_eq!(
                final_payload.controller_state["remaining_micro_usd"],
                json!(0)
            );
            let records: Vec<_> = done
                .run
                .evidence
                .records
                .iter()
                .filter(|record| record.origin == "controller")
                .collect();
            assert_eq!(records.len(), 2);
            assert!(
                provider.captured_messages()[2]
                    .iter()
                    .any(|message| message.content.contains("Verify sources"))
            );
            assert!(agent.set_context("released", json!(true)).is_ok());
        }
    }
}

// Policy-only execution cannot waive a required approval; per-interaction review must still execute after an explicit response.
#[tokio::test]
async fn lp_interaction_profiles_preserve_policy_denial_and_require_real_review() {
    for action in [
        InteractionAction::Reject,
        InteractionAction::AutoIfPolicy,
        InteractionAction::AskEachTime,
    ] {
        let f = super::controller_tests::fixture("tool");
        let mut config = f.config;
        config.defaults.hitl.as_mut().unwrap().on_approval_required = Some(action);
        let runner = AutonomyRunner::try_new(
            f.agent.clone(),
            config,
            Default::default(),
            f.store,
            "controller.v1".into(),
        )
        .unwrap();
        let mut result = runner.run("domain task", None).await.unwrap();
        if action == InteractionAction::AskEachTime {
            assert_eq!(result.run.status, TaskRunStatus::Paused);
            result = runner
                .resume(
                    &result.run.key.run_id,
                    result.run.revision,
                    TaskResumeInput::Approval {
                        request_id: result.run.pending.as_ref().unwrap().id.clone(),
                        result: ApprovalResult::Approved,
                    },
                )
                .await
                .unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(f.calls.load(Ordering::SeqCst), 1);
        } else {
            assert_ne!(result.run.status, TaskRunStatus::Completed);
            assert_eq!(f.calls.load(Ordering::SeqCst), 0);
        }
    }
}

// Returning to a prior high-water mark is not new progress even when another unused response could finish.
#[tokio::test]
async fn lp_progress_oscillation_does_not_replenish_stagnation_or_add_an_extra_turn() {
    let (agent, provider, config, _) = configured(
        &["working", "working", "working", "working", "done"],
        r#"
validation:
  checks: [{id: sources, adapter: host.domain, config: {values: [1, 2, 1, 2]}}]
progress:
  adapter: builtin.evidence
  config:
    signals: [{id: unique, validator: sources, path: metrics.count, comparison: increase}]
  stagnation: {max_cycles_without_progress: 2, action: stop}
"#,
    );
    let store = support::store(agent.info().id, "lp.v1", false).await;
    let done = AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
        .unwrap()
        .run("research", None)
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Incomplete);
    assert_eq!(done.run.stop_reason.as_deref(), Some("no_progress"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
    assert_eq!(done.run.progress.observation_sequence, 4);
    assert_eq!(done.run.progress.cycles_without_progress, 2);
}

// Exhausted repair history must expose unmet validation to policy instead of repeatedly reopening work.
#[tokio::test]
async fn lp_fix_exhaustion_cannot_skip_required_validation_or_leak_repair_source() {
    let (agent, provider, config, _) = configured(
        &["work", "bad", "work", "bad", "done"],
        r#"
premature_finish: {action: fail}
lifecycle:
  - id: work
  - id: verify
    completion: {response_contains: verified}
    retry_on_failure: true
    on_failure_stage: work
    max_fix_cycles: 1
"#,
    );
    let store = support::store(agent.info().id, "lp.v1", false).await;
    let done = AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
        .unwrap()
        .run("repair", None)
        .await
        .unwrap();
    assert_eq!(done.run.status, TaskRunStatus::Failed);
    assert_eq!(done.run.stop_reason.as_deref(), Some("premature_finish"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
}

// Named overlays retain inherited limits and produce a policy outcome without a summary-provider call.
#[tokio::test]
async fn lp_selected_profiles_merge_and_limit_outcomes_are_deterministic() {
    for action in ["summarize_and_stop", "fail"] {
        let (agent, provider, mut config, _) = configured(&["working"], "max_llm_calls: 1\n");
        config.profiles.insert(
            "report".into(),
            serde_yaml::from_str(&format!("on_limit: {{action: {action}}}\nmax_turns: 1\n"))
                .unwrap(),
        );
        let store = support::store(agent.info().id, "lp.v1", false).await;
        let result =
            AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
                .unwrap()
                .run("report", Some("report"))
                .await
                .unwrap();
        assert_eq!(
            result.run.status,
            if action == "fail" {
                TaskRunStatus::Failed
            } else {
                TaskRunStatus::LimitReached
            }
        );
        assert_eq!(result.run.profile.as_deref(), Some("report"));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.run.counters.turns, 1);
    }
}

// Each scheduled prefix is retained independently, including an earlier phase's completed checks.
#[tokio::test]
async fn lp_schedule_reentry_preserves_completed_checks_and_one_observation() {
    for sqlite in support::backends() {
        for schedule in ["each_cycle", "stage_end", "completion"] {
            let f = super::controller_tests::fixture("tool");
            let store = support::store(f.agent.info().id, "controller.v1", *sqlite).await;
            let mut config = f.config;
            config.defaults.max_tool_calls = Some(3);
            let checks = format!(
                "checks: [{{id: first, adapter: host.observation, schedule: {schedule}}}, {{id: second, adapter: host.observation, schedule: {schedule}}}]"
            );
            config.defaults.completion = Some(CompletionGate::ValidatorPassed("second".into()));
            let requests = if schedule == "each_cycle" {
                config.defaults.validation = Some(serde_yaml::from_str(&checks).unwrap());
                2
            } else {
                config.defaults.validation = Some(
                    serde_yaml::from_str("checks: [{id: prefix, adapter: host.observation}]")
                        .unwrap(),
                );
                if schedule == "stage_end" {
                    let validation: Value = serde_yaml::from_str(&checks).unwrap();
                    config.defaults.lifecycle = Some(
                        serde_json::from_value(
                            json!([{ "id":"verify", "validation": validation }]),
                        )
                        .unwrap(),
                    );
                } else {
                    let mut validation: ValidationConfig = serde_yaml::from_str(&checks).unwrap();
                    validation.checks.as_mut().unwrap().insert(
                        0,
                        serde_yaml::from_str("{id: prefix, adapter: host.observation}").unwrap(),
                    );
                    config.defaults.validation = Some(validation);
                }
                3
            };
            let runner = AutonomyRunner::try_new(
                f.agent,
                config,
                Default::default(),
                store.clone(),
                "controller.v1".into(),
            )
            .unwrap();
            let mut result = runner.run("report", None).await.unwrap();
            for completed in 0..requests {
                assert_eq!(result.run.status, TaskRunStatus::Paused, "{schedule}");
                assert_eq!(f.calls.load(Ordering::SeqCst), completed);
                assert_eq!(result.run.counters.turns, 1);
                assert_eq!(result.run.progress.observation_sequence, 0);
                result = runner
                    .resume(
                        &result.run.key.run_id,
                        result.run.revision,
                        TaskResumeInput::Approval {
                            request_id: result.run.pending.as_ref().unwrap().id.clone(),
                            result: ApprovalResult::Approved,
                        },
                    )
                    .await
                    .unwrap();
            }
            assert_eq!(
                result.run.status,
                TaskRunStatus::Completed,
                "{schedule} {:?}",
                result.run.stop_reason
            );
            assert_eq!(f.calls.load(Ordering::SeqCst), requests);
            assert_eq!(result.run.progress.observation_sequence, 1);
            assert_eq!(result.run.counters.llm_attempts, 1);
            let payload: TaskCheckpointPayload = serde_json::from_value(
                store
                    .load(&result.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .payload,
            )
            .unwrap();
            assert_eq!(payload.controller_state["remaining_llm_calls"], json!(0));
            assert_eq!(payload.controller_state["progress"]["sequence"], json!(1));
        }
    }
}

// A currently failed cumulative leaf must not suppress the deferred check that can satisfy it.
#[tokio::test]
async fn lp_completion_observation_can_satisfy_a_currently_failing_tool_count_gate() {
    let f = super::controller_tests::fixture("tool");
    let mut config = f.config;
    config.defaults.max_tool_calls = Some(2);
    config.defaults.validation = Some(serde_yaml::from_str("checks: [{id: first, adapter: host.observation}, {id: second, adapter: host.observation, schedule: completion}]").unwrap());
    config.defaults.completion = Some(
        serde_yaml::from_str(
            "tool_called: {id: probe, count_gte: 2, executed: true, success: true}",
        )
        .unwrap(),
    );
    let runner = AutonomyRunner::try_new(
        f.agent,
        config,
        Default::default(),
        f.store,
        "controller.v1".into(),
    )
    .unwrap();
    let mut result = runner.run("verify", None).await.unwrap();
    for _ in 0..2 {
        assert_eq!(result.run.status, TaskRunStatus::Paused);
        result = runner
            .resume(
                &result.run.key.run_id,
                result.run.revision,
                TaskResumeInput::Approval {
                    request_id: result.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Approved,
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.turns, 1);
    assert_eq!(f.calls.load(Ordering::SeqCst), 2);
}

// Required supervision and a restrictive interaction policy resolve to incomplete, never implicit approval.
#[tokio::test]
async fn lp_restrictive_profiles_cannot_auto_approve_required_stage_or_plan_reviews() {
    for action in ["reject", "auto_if_policy"] {
        for replan in [false, true] {
            let plan = r#"{"steps":[{"id":"one","description":"Verify","action_type":"think","action_target":"reason","dependencies":[]}]}"#;
            let yaml = if replan {
                format!(
                    "hitl: {{on_approval_required: {action}}}\nsupervision: {{plan_changes: require_approval}}\nprogress: {{stagnation: {{max_cycles_without_progress: 1, action: replan, max_replans: 1, on_exhausted: stop}}}}"
                )
            } else {
                format!(
                    "hitl: {{on_approval_required: {action}}}\nsupervision: {{stage_changes: require_approval}}\nlifecycle: [{{id: review}}]"
                )
            };
            let (agent, provider, config, _) = configured(&["working", plan], &yaml);
            let store = support::store(agent.info().id, "lp.v1", false).await;
            let result =
                AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into())
                    .unwrap()
                    .run("report", None)
                    .await
                    .unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Incomplete);
            assert_eq!(
                result.run.stop_reason.as_deref(),
                Some("interaction_rejected")
            );
            assert!(result.run.pending.is_none());
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                if replan { 2 } else { 1 }
            );
        }
    }
}

// Stage review resumes after completed work and does not consume a second turn or forced continuation.
#[tokio::test]
async fn lp_ask_each_time_reviews_actual_stage_boundaries_without_replaying_work() {
    let (agent, provider, config, _) = configured(
        &["done", "done"],
        "lifecycle: [{id: explore}, {id: review}]\nhitl: {on_approval_required: ask_each_time}\npremature_finish: {action: fail}\n",
    );
    let store = support::store(agent.info().id, "lp.v1", false).await;
    let runner =
        AutonomyRunner::try_new(agent, config, Default::default(), store, "lp.v1".into()).unwrap();
    let mut result = runner.run("report", None).await.unwrap();
    for calls in 1..=2 {
        assert_eq!(result.run.status, TaskRunStatus::Paused);
        assert_eq!(provider.calls.load(Ordering::SeqCst), calls);
        result = runner
            .resume(
                &result.run.key.run_id,
                result.run.revision,
                TaskResumeInput::Approval {
                    request_id: result.run.pending.as_ref().unwrap().id.clone(),
                    result: ApprovalResult::Approved,
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.turns, 2);
    assert_eq!(result.run.counters.continuations, 0);
}
