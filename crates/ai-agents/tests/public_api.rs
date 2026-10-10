use std::{collections::HashMap, sync::Arc};

use ai_agents::agent::{AgentStreamEvent, RuntimeAgent, RuntimeControlHandle};
use ai_agents::persistence::{AgentStorage, NoopStorage, StorageCapability};
use ai_agents::spec::{
    AgentSpec, AutoSpawnEntry, LLMConfigOrSelector, ManagementToolsConfig,
    OrchestrationToolsConfig, SpawnerConfig, TemplateSource,
};
use ai_agents::tools::{
    CommandRunner, CopyPathTool, DeletePathTool, DiagnosticsProvider, MovePathTool,
    QuestionHandler, ToolError, ToolSchemaPromptMode, WebSearchProvider, WebSearchRequest,
    WebSearchResponse, WebSearchResultItem, WebSearchSafeSearch, WebSearchTool,
};
use ai_agents::{NativeCallBinding, NativeProviderState, NativeProviderTarget};

#[test]
fn development_standalone_and_provenance_migration_surface_compiles() {
    use ai_agents::autonomy::{AutonomyConfig, AutonomyHostCeilings, AutonomyRunner, TaskRunStore};
    fn build(
        agent: Arc<RuntimeAgent>,
        config: AutonomyConfig,
        store: Arc<dyn TaskRunStore>,
    ) -> ai_agents::error::Result<()> {
        let runner = AutonomyRunner::try_new(
            agent.clone(),
            config,
            AutonomyHostCeilings::default(),
            store,
            "prepared-host-binding".into(),
        )?;
        let _run = runner.run("objective", None);
        let _cancel = runner.request_cancel("retained-run");
        let _resume = runner.resume(
            "retained-run",
            1,
            ai_agents::autonomy::TaskResumeInput::Approval {
                request_id: "acknowledged-request".into(),
                result: ai_agents::hitl::ApprovalResult::Approved,
            },
        );
        let _answer = runner.resume(
            "retained-run",
            1,
            ai_agents::autonomy::TaskResumeInput::UserAnswer {
                request_id: "acknowledged-question".into(),
                answer: r#"{"answered":true,"selected":["yes"]}"#.parse().unwrap(),
            },
        );
        let _timeout = runner.resume(
            "retained-run",
            1,
            ai_agents::autonomy::TaskResumeInput::QuestionTimeout {
                request_id: "expired-question".into(),
            },
        );
        let _recovery = runner.acknowledge_recovery_release("retained-run", 1);
        let _close = runner.cancel_paused("retained-run", 1);
        agent.clear_actor_id()?;
        let _removed = agent.remove_context("obsolete")?;
        Ok(())
    }
    let _surface = build;
    let message = ai_agents::ChatMessage {
        provenance: None,
        role: ai_agents::Role::User,
        content: "legacy literal".into(),
        name: None,
        timestamp: None,
    };
    assert!(message.provenance.is_none());
    let source = ai_agents::MessageProvenance {
        run_id: "run".into(),
    };
    assert_eq!(source.run_id, "run");
    let bound = ai_agents::autonomy::ProviderCostBound {
        provider_identity: "host-provider/config-v1".into(),
        pricing_identity: "declared-schedule-v1".into(),
        request_id: "exact-request".into(),
        max_micro_usd: 500_000,
    };
    let settlement = ai_agents::autonomy::ProviderCostSettlement {
        bound: bound.clone(),
        charged_micro_usd: 200_000,
    };
    ai_agents::autonomy::validate_cost_settlement(&bound, &settlement).unwrap();
    let footprint = ai_agents::autonomy::ToolWriteFootprint::empty("host-read-only/v1");
    assert!(footprint.targets.is_empty());
    fn quote(provider: &dyn ai_agents::LLMProvider, messages: &[ai_agents::ChatMessage]) {
        let request = ai_agents::autonomy::ProviderRequest {
            request_id: "request",
            messages,
            config: None,
            tools: None,
            streaming: false,
        };
        let _bound = provider.request_cost_bound(&request);
    }
    let _surface = quote;
}

fn configure_host_integrations(
    agent: &RuntimeAgent,
    question_handler: Arc<dyn QuestionHandler>,
    diagnostics_provider: Arc<dyn DiagnosticsProvider>,
    command_runner: Arc<dyn CommandRunner>,
    web_search_provider: Arc<dyn WebSearchProvider>,
) {
    agent.set_question_handler(Some(question_handler));
    agent.set_diagnostics_provider(diagnostics_provider);
    agent.set_command_runner(command_runner);
    agent.set_web_search_provider(web_search_provider);
}

fn supports_snapshots(storage: &dyn AgentStorage) -> bool {
    storage.supports(StorageCapability::Snapshot)
}

#[test]
fn task_tool_control_transfer_and_effect_state_surface_compiles() {
    fn invoke(tool: &dyn ai_agents::Tool, context: ai_agents::tools::ToolExecutionContext) {
        let _supported = tool.supports_task_messages();
        drop(tool.execute_task(Default::default(), context));
    }
    fn effect(state: ai_agents::autonomy::TaskEffectState) -> &'static str {
        use ai_agents::autonomy::TaskEffectState;
        match state {
            TaskEffectState::Reserved => "reserved",
            TaskEffectState::Dispatched => "dispatched",
            TaskEffectState::Suspended => "suspended",
            TaskEffectState::Completed => "completed",
            TaskEffectState::Uncertain => "uncertain",
        }
    }
    let _surface = invoke;
    assert_eq!(
        effect(ai_agents::autonomy::TaskEffectState::Suspended),
        "suspended"
    );
}

#[test]
fn autonomy_config_migration_exposes_typed_fields_without_enabling_chat() {
    use ai_agents::{
        AutonomyConfig, AutonomyOverride, AutonomyProfile, SkillDefinition, StateDefinition,
    };

    let spec = AgentSpec {
        autonomy: AutonomyConfig {
            defaults: AutonomyProfile {
                enabled: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
        ..AgentSpec::default()
    };
    let state = StateDefinition {
        autonomy: Some(AutonomyOverride::default()),
        ..Default::default()
    };
    let skill = SkillDefinition {
        autonomy: Some(AutonomyOverride::default()),
        id: "review".into(),
        description: "Review the result".into(),
        trigger: "When requested".into(),
        steps: vec![],
        reasoning: None,
        reflection: None,
        disambiguation: None,
    };
    spec.validate().unwrap();
    assert!(!spec.autonomy.defaults.enabled.unwrap());
    assert!(state.autonomy.is_some() && skill.autonomy.is_some());
}

#[test]
fn autonomy_storage_api_is_additive_and_separates_results_from_envelopes() {
    use ai_agents::autonomy::*;
    use ai_agents::persistence::{AgentSnapshot, InMemoryTaskStorage};
    use ai_agents::{TaskRun, TaskRunStore};

    let snapshot = AgentSnapshot::new("host-agent".into());
    let now = snapshot.timestamp;
    let profile = resolve_profile(
        &AutonomyConfig::default(),
        AutonomyScope::Task,
        None,
        None,
        None,
        None,
        &AutonomyHostCeilings::default(),
    )
    .unwrap();
    let payload = TaskCheckpointPayload::new(
        "objective".into(),
        "prepared-config-v1".into(),
        &profile,
        TaskRuntimeCheckpoint::between_turns(snapshot).unwrap(),
    );
    let envelope = payload
        .bind(TaskRunSnapshot {
            schema_version: TASK_RUN_SCHEMA_VERSION,
            key: TaskRunKey {
                agent_id: "host-agent".into(),
                run_id: "run".into(),
            },
            actor_id: None,
            revision: 0,
            status: TaskRunStatus::Paused,
            owner_token: None,
            cancel_requested: false,
            created_at: now,
            updated_at: now,
            payload: Default::default(),
        })
        .unwrap();
    let store = ScopedTaskRunStore::new(
        Arc::new(InMemoryTaskStorage::default()),
        "host-agent".into(),
        None,
        "prepared-config-v1".into(),
    )
    .unwrap();
    drop(store.create(&envelope));
    drop(store.load("run"));
    assert_eq!(
        TaskRun::from_checkpoint(&envelope, "prepared-config-v1")
            .unwrap()
            .objective,
        "objective"
    );
    assert!(!NoopStorage.supports(StorageCapability::TaskRuns));
    drop(NoopStorage.load_task_run(&envelope.key));
}

#[test]
fn autonomy_evaluation_migration_exposes_bounded_host_primitives() {
    use ai_agents::autonomy::{
        AutonomyExtensions, ValidationCapabilities, ValidationCheck, ValidationConfig,
        ValidationSchedule,
    };
    use ai_agents::{AutonomyProfile, CompletionGateEvaluator, GateOutcome};
    let check = ValidationCheck {
        id: "quality".into(),
        adapter: "builtin.evidence".into(),
        contract_version: Some(1),
        required: Some(true),
        schedule: Some(ValidationSchedule::Completion),
        timeout_seconds: None,
        max_evaluation_rounds: Some(2),
        max_observations_per_round: Some(1),
        config: Some(
            "{\"assertion\":{\"path\":\"ready\",\"exists\":true}}"
                .parse()
                .unwrap(),
        ),
    };
    let profile = AutonomyProfile {
        validation: Some(ValidationConfig {
            checks: Some(vec![check]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let bound = AutonomyExtensions::builtins()
        .freeze()
        .bind(&profile, &ValidationCapabilities::default())
        .unwrap();
    assert_eq!(bound.checks.len(), 1);
    assert_eq!(GateOutcome::Unknown.negate(), GateOutcome::Unknown);
    assert!(CompletionGateEvaluator::default().redact);
}

#[test]
fn facade_exposes_reviewed_v1_type_closure() {
    let mut templates = HashMap::new();
    templates.insert(
        "worker".to_string(),
        TemplateSource::Inline("name: Worker\nsystem_prompt: worker\n".to_string()),
    );
    let spawner = SpawnerConfig {
        templates,
        auto_spawn: vec![AutoSpawnEntry {
            id: "worker".to_string(),
            agent: "worker.yaml".to_string(),
        }],
        management_tools: ManagementToolsConfig::default(),
        orchestration_tools: OrchestrationToolsConfig::default(),
        ..SpawnerConfig::default()
    };
    let spec = AgentSpec {
        llm: LLMConfigOrSelector::default(),
        spawner: Some(spawner),
        ..AgentSpec::default()
    };

    let request = WebSearchRequest {
        query: "rust agents".to_string(),
        safe_search: Some(WebSearchSafeSearch::Moderate),
        ..WebSearchRequest::default()
    };
    let response = WebSearchResponse {
        available: true,
        results: vec![WebSearchResultItem {
            title: "Result".to_string(),
            url: "https://example.com".to_string(),
            ..WebSearchResultItem::default()
        }],
        ..WebSearchResponse::default()
    };

    let _ = spec;
    let _ = request;
    let _ = response;
    let storage = NoopStorage;
    assert!(!supports_snapshots(&storage));
    let _ = ToolSchemaPromptMode::Compact;
    let _ = ToolError::NotFound("missing".to_string());
    let _ = CopyPathTool::new();
    let _ = MovePathTool::new();
    let _ = DeletePathTool::new();
    let _ = WebSearchTool::new();
    let _: Option<RuntimeControlHandle> = None;
    let _: Option<AgentStreamEvent> = None;
    let _: Option<NativeCallBinding> = None;
    let _: Option<NativeProviderState> = None;
    let _: Option<NativeProviderTarget> = None;
    type HostIntegrationConfigurator = fn(
        &RuntimeAgent,
        Arc<dyn QuestionHandler>,
        Arc<dyn DiagnosticsProvider>,
        Arc<dyn CommandRunner>,
        Arc<dyn WebSearchProvider>,
    );
    let _: HostIntegrationConfigurator = configure_host_integrations;
}
