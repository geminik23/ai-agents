use super::*;
use crate::{
    AgentBuilder,
    spec::{AgentSpec, LLMConfigOrSelector, LLMSelector},
};
use ai_agents_core::Tool;
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_llm::{LLMRole, RouterRolesConfig};

// Independent queues prove owning call selection without relying on model wording or tracing caches.
fn fixture(
    extra: &str,
) -> (
    AgentSpec,
    LLMRegistry,
    HashMap<LLMRole, MockLLMProvider>,
    MockLLMProvider,
) {
    let mut spec =
        AgentSpec::from_yaml_strict(&format!("name: routed\nsystem_prompt: Help.\n{extra}"))
            .unwrap();
    let mut tree = serde_json::json!({});
    let mut registry = LLMRegistry::new();
    let mut main = MockLLMProvider::new("main");
    main.set_response("main answer");
    registry.register("main", Arc::new(main.clone()));
    registry.register("default", Arc::new(main.clone()));
    let mut providers = HashMap::new();
    for role in LLMRole::ALL {
        let field = role.as_path().split('.').nth(1).unwrap();
        tree[role.group()][field] = Value::String(role.as_path().into());
        let mut provider = MockLLMProvider::new(role.as_path());
        provider.set_response("role result");
        registry.register(role.as_path(), Arc::new(provider.clone()));
        providers.insert(*role, provider);
    }
    let roles: RouterRolesConfig = serde_json::from_value(tree).unwrap();
    registry.set_default("main");
    registry.set_router_roles(roles.clone());
    spec.llm = LLMConfigOrSelector::Selector(LLMSelector::new("main").with_router_roles(roles));
    (spec, registry, providers, main)
}

// Assert both the expected count and absence of accidental calls to every other role.
fn counts(providers: &HashMap<LLMRole, MockLLMProvider>, expected: &[(LLMRole, usize)]) {
    for (role, provider) in providers {
        let count = expected
            .iter()
            .find(|(candidate, _)| candidate == role)
            .map(|(_, count)| *count)
            .unwrap_or(0);
        assert_eq!(provider.call_count(), count, "{}", role.as_path());
    }
}

#[tokio::test]
async fn hierarchy_state_skill_and_semantic_condition_use_distinct_owners() {
    let (spec, registry, mut providers, main) = fixture(
        "states:\n  initial: a\n  states:\n    a:\n      extract:\n        - key: item\n          description: Extract the item.\n      transitions:\n        - to: b\n          when: Move when requested.\n    b: {}\nskills:\n  - id: helper\n    description: Help\n    trigger: Select when requested.\n    steps:\n      - prompt: Help.\n",
    );
    providers
        .get_mut(&LLMRole::StateTransition)
        .unwrap()
        .set_response("1");
    providers
        .get_mut(&LLMRole::SkillsSelection)
        .unwrap()
        .set_response("0");
    providers
        .get_mut(&LLMRole::ToolsCondition)
        .unwrap()
        .set_response(r#"{"result":true,"confidence":1.0,"reason":"allowed"}"#);
    let agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .build()
        .unwrap();
    let extracted = agent.run_context_extractors_staged("item").await.unwrap();
    assert_eq!(extracted["item"], "role result");
    assert!(
        agent
            .select_skill_candidate("help")
            .await
            .unwrap()
            .is_none()
    );
    let getter = RegistryLLMGetter {
        registry: agent.llm_registry.clone(),
    };
    let condition = ai_agents_state::ToolCondition::Semantic {
        when: "allowed".into(),
        llm: None,
        threshold: 0.7,
    };
    assert!(
        ConditionEvaluator::new(getter)
            .evaluate(&condition, &Default::default())
            .await
            .unwrap()
    );
    assert!(agent.evaluate_transitions("move", "answer").await.unwrap());
    counts(
        &providers,
        &[
            (LLMRole::StateExtract, 1),
            (LLMRole::StateTransition, 1),
            (LLMRole::SkillsSelection, 1),
            (LLMRole::ToolsCondition, 1),
        ],
    );
    assert_eq!(main.call_count(), 0);
}

#[tokio::test]
async fn hierarchy_parallel_transition_retains_reservation_and_main_is_not_routed() {
    let (spec, registry, mut providers, main) = fixture(
        "runtime:\n  optimization:\n    enabled: true\n    speculative_state_transitions: true\n    max_speculative_llm_calls_per_turn: 2\nstates:\n  initial: a\n  states:\n    a:\n      transitions:\n        - to: b\n          when: Move.\n          timing: parallel\n    b: {}\n",
    );
    providers
        .get_mut(&LLMRole::StateTransition)
        .unwrap()
        .set_response("1");
    let agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .build()
        .unwrap();
    assert!(matches!(
        agent
            .select_parallel_transition_candidate("move")
            .await
            .unwrap(),
        ParallelTransitionSelection::Candidate(_)
    ));
    counts(&providers, &[(LLMRole::StateTransition, 1)]);
    assert_eq!(main.call_count(), 0);
    let (spec, registry, providers, main) = fixture("");
    let agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .build()
        .unwrap();
    assert_eq!(agent.chat("hello").await.unwrap().content, "main answer");
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 1);
}

#[tokio::test]
async fn hierarchy_reasoning_reflection_overflow_and_localized_response_are_separate() {
    let (spec, registry, mut providers, main) = fixture("reasoning:\n  mode: auto\n");
    providers
        .get_mut(&LLMRole::ReasoningSelection)
        .unwrap()
        .set_response("cot");
    providers
        .get_mut(&LLMRole::ReasoningPlanning)
        .unwrap()
        .set_response(r#"{"steps":[]}"#);
    providers
        .get_mut(&LLMRole::ReasoningReflectionDecision)
        .unwrap()
        .set_response("YES");
    providers
        .get_mut(&LLMRole::ReasoningReflectionEvaluation)
        .unwrap()
        .set_response("1. PASS\nCONFIDENCE: 1.0\nOVERALL: PASS");
    let agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .build()
        .unwrap();
    assert_eq!(
        agent
            .determine_reasoning_mode_strict("analyze")
            .await
            .unwrap(),
        ReasoningMode::CoT
    );
    agent.generate_plan("plan").await.unwrap();
    let reflection = ai_agents_reasoning::ReflectionConfig::auto();
    assert!(
        agent
            .should_reflect_with_config("question", "answer", &reflection)
            .await
            .unwrap()
    );
    assert!(
        agent
            .evaluate_response_with_config("question", "answer", &reflection)
            .await
            .unwrap()
            .passed
    );
    assert_eq!(
        agent
            .generate_localized_apology("Apologize", "unclear")
            .await
            .unwrap(),
        "role result"
    );
    let mut messages = vec![
        ChatMessage::system("system"),
        ChatMessage::user("earlier"),
        ChatMessage::assistant("recent"),
    ];
    agent
        .summarize_context(&mut messages, None, 100, None, 1, None)
        .await
        .unwrap();
    counts(
        &providers,
        &[
            (LLMRole::ReasoningSelection, 1),
            (LLMRole::ReasoningPlanning, 1),
            (LLMRole::ReasoningReflectionDecision, 1),
            (LLMRole::ReasoningReflectionEvaluation, 1),
            (LLMRole::DisambiguationResponse, 1),
            (LLMRole::ContextSummarize, 1),
        ],
    );
    assert_eq!(main.call_count(), 0);
}

#[tokio::test]
async fn hierarchy_process_stages_and_compacting_merge_call_their_own_providers() {
    let (spec, registry, mut providers, main) = fixture(
        "memory:\n  type: compacting\n  compress_threshold: 2\n  max_recent_messages: 0\n  summarize_batch_size: 2\nprocess:\n  input:\n    - type: detect\n      config:\n        detect: [language]\n    - type: extract\n      config:\n        schema:\n          item:\n            type: string\n    - type: sanitize\n      config:\n        remove: [sensitive]\n    - type: transform\n      config:\n        prompt: Rewrite.\n    - type: validate\n      config:\n        criteria: [Valid.]\n",
    );
    providers
        .get_mut(&LLMRole::ProcessDetect)
        .unwrap()
        .set_response(r#"{"language":"ko"}"#);
    providers
        .get_mut(&LLMRole::ProcessExtract)
        .unwrap()
        .set_response(r#"{"item":"one"}"#);
    providers
        .get_mut(&LLMRole::ProcessValidate)
        .unwrap()
        .set_response(r#"{"score":1.0,"passes":true}"#);
    let agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    agent
        .process_processor
        .as_ref()
        .unwrap()
        .process_input("test")
        .await
        .unwrap();
    for text in ["one", "two", "three", "four"] {
        agent
            .memory
            .add_message(ChatMessage::user(text))
            .await
            .unwrap();
    }
    agent.memory.compress(None).await.unwrap();
    agent.memory.compress(None).await.unwrap();
    counts(
        &providers,
        &[
            (LLMRole::ProcessDetect, 1),
            (LLMRole::ProcessExtract, 1),
            (LLMRole::ProcessSanitize, 1),
            (LLMRole::ProcessTransform, 1),
            (LLMRole::ProcessValidate, 1),
            (LLMRole::MemorySummarize, 2),
            (LLMRole::MemoryMerge, 1),
        ],
    );
    assert_eq!(main.call_count(), 0);
}

// Child agents remain ordinary turns, not auxiliary role invocations.
async fn participants() -> Arc<crate::spawner::AgentRegistry> {
    let registry = Arc::new(crate::spawner::AgentRegistry::new());
    for id in ["a", "b"] {
        let spec = AgentSpec {
            name: id.into(),
            ..Default::default()
        };
        let mut provider = MockLLMProvider::new(id);
        provider.set_response(format!("{id} answer"));
        let agent = AgentBuilder::from_spec(spec.clone())
            .llm(Arc::new(provider))
            .build()
            .unwrap();
        registry
            .register(crate::spawner::SpawnedAgent::from_runtime(
                id.into(),
                agent,
                spec,
            ))
            .await
            .unwrap();
    }
    registry
}

#[tokio::test]
async fn hierarchy_orchestration_tools_and_vote_tiebreak_are_independent() {
    use crate::orchestration::tools::{
        ConcurrentAskTool, GroupDiscussionTool, HandoffConversationTool, RouteToAgentTool,
    };
    let (_, registry, mut providers, main) = fixture("");
    providers
        .get_mut(&LLMRole::OrchestrationRouting)
        .unwrap()
        .set_response("a");
    providers
        .get_mut(&LLMRole::OrchestrationHandoff)
        .unwrap()
        .set_response(r#"{"action":"stay"}"#);
    providers
        .get_mut(&LLMRole::OrchestrationConsensus)
        .unwrap()
        .set_response("YES");
    providers
        .get_mut(&LLMRole::OrchestrationVote)
        .unwrap()
        .set_responses(
            vec!["one".into(), "two".into(), "one".into(), "two".into()],
            false,
        );
    providers
        .get_mut(&LLMRole::OrchestrationTiebreak)
        .unwrap()
        .set_response("one");
    let registry = Arc::new(registry);
    let children = participants().await;
    let result = RouteToAgentTool::new(children.clone(), registry.clone())
        .execute(
            serde_json::json!({"input":"route","candidates":["a","b"]}),
            ai_agents_core::ToolExecutionContext::test("routing-test"),
        )
        .await;
    assert!(result.success, "{}", result.output);
    let result = HandoffConversationTool::new(children.clone(), registry.clone())
        .execute(
            serde_json::json!({"input":"handoff","initial_agent":"a","available_agents":["a","b"]}),
            ai_agents_core::ToolExecutionContext::test("routing-test"),
        )
        .await;
    assert!(result.success, "{}", result.output);
    let result = ConcurrentAskTool::new(children.clone(), registry.clone())
        .execute(
            serde_json::json!({"question":"ask","agents":["a","b"],"aggregation":"llm_synthesis"}),
            ai_agents_core::ToolExecutionContext::test("routing-test"),
        )
        .await;
    assert!(result.success, "{}", result.output);
    let result = GroupDiscussionTool::new(children.clone(),registry.clone()).execute(serde_json::json!({"topic":"agree","participants":["a","b"],"style":"consensus","max_rounds":1}),ai_agents_core::ToolExecutionContext::test("routing-test")).await;
    assert!(result.success, "{}", result.output);
    let config: ai_agents_state::AggregationConfig =
        serde_yaml::from_str("strategy: voting\nvote:\n  tiebreaker: router_decides\n").unwrap();
    let results = [
        crate::orchestration::AgentResult {
            agent_index: 0,
            agent_id: "a".into(),
            response: Some(AgentResponse::new("a")),
            duration_ms: 0,
            success: true,
            error: None,
        },
        crate::orchestration::AgentResult {
            agent_index: 1,
            agent_id: "b".into(),
            response: Some(AgentResponse::new("b")),
            duration_ms: 0,
            success: true,
            error: None,
        },
    ];
    let providers_bundle =
        crate::orchestration::AggregationProviders::resolve(&registry, None).unwrap();
    for parallelism in [None, Some(2)] {
        crate::orchestration::aggregation::aggregate_with_llms(
            &results,
            &config,
            providers_bundle.as_refs(),
            &HashMap::new(),
            parallelism,
        )
        .await
        .unwrap();
    }
    let group: ai_agents_state::GroupChatStateConfig = serde_yaml::from_str(
        "participants:\n  - id: a\nmax_rounds: 1\nmanager:\n  method: llm_directed\n",
    )
    .unwrap();
    let speaker =
        crate::orchestration::role_provider(&registry, LLMRole::OrchestrationSpeaker, None)
            .unwrap();
    crate::orchestration::group_chat_with_llms(
        &children,
        "speak",
        &group,
        crate::orchestration::GroupLLMs {
            speaker: speaker.as_deref(),
            consensus: None,
        },
        None,
    )
    .await
    .unwrap();
    counts(
        &providers,
        &[
            (LLMRole::OrchestrationRouting, 1),
            (LLMRole::OrchestrationHandoff, 1),
            (LLMRole::OrchestrationSynthesis, 1),
            (LLMRole::OrchestrationConsensus, 1),
            (LLMRole::OrchestrationSpeaker, 1),
            (LLMRole::OrchestrationVote, 4),
            (LLMRole::OrchestrationTiebreak, 2),
        ],
    );
    assert_eq!(main.call_count(), 0);
}

#[tokio::test]
async fn hierarchy_spawner_generation_repair_and_delegate_summary_are_distinct() {
    let (spec, registry, mut providers, main) = fixture("");
    providers
        .get_mut(&LLMRole::SpawnerGeneration)
        .unwrap()
        .set_response("invalid");
    providers
        .get_mut(&LLMRole::SpawnerRepair)
        .unwrap()
        .set_response("name: kid\nsystem_prompt: Help.\n");
    let children = participants().await;
    let spawner = Arc::new(crate::spawner::AgentSpawner::new().with_shared_llms(registry.clone()));
    let tool = crate::spawner::GenerateAgentTool::new(
        spawner,
        children.clone(),
        Arc::new(registry.clone()),
    );
    let result = tool
        .execute(
            serde_json::json!({"name":"kid","description":"help"}),
            ai_agents_core::ToolExecutionContext::test("routing-test"),
        )
        .await;
    assert!(result.success, "{}", result.output);
    let mut agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .build()
        .unwrap();
    agent.spawner_registry = Some(children);
    agent
        .memory
        .add_message(ChatMessage::user("earlier"))
        .await
        .unwrap();
    let state: ai_agents_state::StateDefinition =
        serde_yaml::from_str("delegate: a\ndelegate_context: summary\n").unwrap();
    agent
        .handle_delegated_state("now", "a", &state)
        .await
        .unwrap();
    counts(
        &providers,
        &[
            (LLMRole::SpawnerGeneration, 1),
            (LLMRole::SpawnerRepair, 1),
            (LLMRole::OrchestrationSummary, 1),
        ],
    );
    assert_eq!(main.call_count(), 0);
}

#[tokio::test]
async fn hierarchy_hitl_message_selects_only_text_generation_not_approval() {
    let (_, registry, providers, main) = fixture("");
    let config: ai_agents_hitl::MessageLanguageConfig =
        serde_yaml::from_str("strategy: llm_generate\nllm_generate: {}\n").unwrap();
    let message = ai_agents_hitl::MessageResolver::new(&config)
        .with_llm_registry(&registry)
        .resolve(
            &ai_agents_hitl::ApprovalMessage::simple("Allow?"),
            None,
            &HashMap::new(),
            &ai_agents_hitl::RejectAllHandler::new(),
        )
        .await
        .unwrap();
    assert_eq!(message, "role result");
    counts(&providers, &[(LLMRole::HitlMessage, 1)]);
    assert_eq!(main.call_count(), 0);
}

#[test]
fn hierarchy_presence_and_errors_are_preflighted_without_calls() {
    for fragment in [
        "llm:\n  router:\n    state:\n      transiton: fast\n",
        "llm:\n  router:\n    default: ' '\n",
        "disambiguation:\n  detection:\n    llm: null\n",
        "states:\n  initial: a\n  states:\n    a:\n      extract:\n        - key: k\n          llm: null\n",
    ] {
        assert!(
            AgentSpec::from_yaml_strict(&format!("name: bad\nsystem_prompt: Help.\n{fragment}"))
                .is_err(),
            "{fragment}"
        );
    }
    let (mut spec, registry, providers, main) = fixture("");
    if let LLMConfigOrSelector::Selector(selector) = &mut spec.llm {
        selector.router = Some(ai_agents_llm::RouterSelector::Hierarchical(Box::new(
            RouterRolesConfig {
                default: Some("missing".into()),
                ..Default::default()
            },
        )));
    }
    assert!(
        AgentBuilder::from_spec(spec)
            .llm_registry(registry)
            .build()
            .is_err()
    );
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 0);
}

#[tokio::test]
async fn hierarchy_freeze_rejects_same_alias_replacement_and_external_skill_is_snapshotted_once() {
    let (spec, registry, _, main) = fixture("spawner:\n  shared_llms: true\n");
    let builder = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .auto_configure_spawner()
        .await
        .unwrap();
    assert!(builder.llm_alias("main", Arc::new(main)).build().is_err());
    let directory =
        std::env::temp_dir().join(format!("ai-agents-routing-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("helper.yaml");
    std::fs::write(&path,"id: helper\ndescription: Help\ntrigger: Help\nreasoning:\n  mode: auto\n  judge_llm: reasoning.selection\nsteps:\n  - prompt: Original.\n").unwrap();
    let (spec, registry, _, _) = fixture("skills:\n  - file: helper.yaml\n");
    let original = spec.clone();
    let child = AgentBuilder::from_spec_with_base_dir(spec, directory.clone())
        .llm_registry(registry.clone())
        .build()
        .unwrap();
    std::fs::write(&path, "invalid").unwrap();
    let saved = child.prepared_persistence_spec(&original).unwrap();
    assert!(matches!(
        original.skills[0],
        ai_agents_skills::SkillRef::File { .. }
    ));
    assert!(matches!(
        saved.skills[0],
        ai_agents_skills::SkillRef::Inline(_)
    ));
    std::fs::remove_dir_all(directory).unwrap();
    let restored = AgentBuilder::from_spec(saved.clone())
        .llm_registry(registry)
        .build()
        .unwrap();
    assert_eq!(
        saved.routing_projection().unwrap(),
        restored
            .prepared_persistence_spec(&saved)
            .unwrap()
            .routing_projection()
            .unwrap()
    );
    let mut prompt_only = saved.clone();
    if let ai_agents_skills::SkillRef::Inline(skill) = &mut prompt_only.skills[0]
        && let ai_agents_skills::SkillStep::Prompt { prompt, .. } = &mut skill.steps[0]
    {
        *prompt = "Other.".into();
    }
    assert_eq!(
        saved.routing_projection().unwrap(),
        prompt_only.routing_projection().unwrap()
    );
    if let ai_agents_skills::SkillRef::Inline(skill) = &mut prompt_only.skills[0] {
        skill.reasoning.as_mut().unwrap().judge_llm = Some("main".into());
    }
    assert_ne!(
        saved.routing_projection().unwrap(),
        prompt_only.routing_projection().unwrap()
    );
}

#[tokio::test]
async fn legacy_builder_chaining_and_hierarchy_preflight_regressions() {
    let mut first = MockLLMProvider::new("first");
    first.set_response("first");
    let mut second = MockLLMProvider::new("second");
    second.set_response("second");
    let agent = AgentBuilder::new()
        .system_prompt("Help.")
        .llm(Arc::new(first.clone()))
        .auto_configure_spawner()
        .await
        .unwrap()
        .llm(Arc::new(second.clone()))
        .build()
        .unwrap();
    assert_eq!(agent.chat("hello").await.unwrap().content, "second");
    assert_eq!(first.call_count(), 0);
    let (spec, registry, providers, main) = fixture("disambiguation:\n  enabled: true\n");
    assert!(
        AgentBuilder::from_spec(spec)
            .llm_registry(registry)
            .build()
            .is_ok()
    );
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 0);
    let (spec, registry, providers, main) = fixture(
        "reasoning:\n  mode: auto\n  planning:\n    planner_llm: missing\nspawner:\n  shared_llms: true\n",
    );
    assert!(
        AgentBuilder::from_spec(spec)
            .llm_registry(registry)
            .auto_configure_spawner()
            .await
            .is_err()
    );
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 0);
    let (mut saved, _, _, _) = fixture(
        "states:\n  initial: a\n  states:\n    a:\n      concurrent:\n        agents: [a]\n        aggregation:\n          strategy: llm_synthesis\n          synthesizer_llm: main\n",
    );
    let original = saved.routing_projection().unwrap();
    saved
        .states
        .as_mut()
        .unwrap()
        .states
        .get_mut("a")
        .unwrap()
        .concurrent
        .as_mut()
        .unwrap()
        .aggregation
        .synthesizer_llm = Some("orchestration.synthesis".into());
    assert_ne!(original, saved.routing_projection().unwrap());
}

#[tokio::test]
async fn hierarchy_direct_consumers_do_not_recover_invalid_configuration() {
    let (_, registry, providers, main) = fixture("");
    let config: ai_agents_process::ProcessConfig = serde_yaml::from_str("input:\n  - type: sanitize\n    config:\n      llm: missing\n      remove: [sensitive]\nsettings:\n  on_stage_error:\n    default: continue\n").unwrap();
    let processor = ai_agents_process::ProcessProcessor::new(config)
        .with_llm_registry(Arc::new(registry.clone()));
    assert!(matches!(
        processor.process_input("sensitive").await,
        Err(AgentError::Config(_))
    ));
    let config: ai_agents_hitl::MessageLanguageConfig =
        serde_yaml::from_str("strategy: llm_generate\nllm_generate:\n  llm: missing\n").unwrap();
    let result = ai_agents_hitl::MessageResolver::new(&config)
        .with_llm_registry(&registry)
        .resolve(
            &ai_agents_hitl::ApprovalMessage::simple("Allow?"),
            None,
            &HashMap::new(),
            &ai_agents_hitl::RejectAllHandler::new(),
        )
        .await;
    assert!(matches!(result, Err(AgentError::Config(_))));
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 0);
}

#[tokio::test]
async fn hierarchy_manual_builder_and_dynamic_condition_validate_local_aliases() {
    let (_, registry, _, main) = fixture("");
    let config: ai_agents_reasoning::ReasoningConfig =
        serde_yaml::from_str("mode: plan_and_execute\nplanning:\n  planner_llm: missing\n")
            .unwrap();
    assert!(
        AgentBuilder::new()
            .system_prompt("Help.")
            .llm_registry(registry)
            .reasoning(config)
            .build()
            .is_err()
    );
    assert_eq!(main.call_count(), 0);
    let (spec, registry, providers, main) = fixture("tools: [echo]\n");
    let agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .auto_configure_features()
        .unwrap()
        .build()
        .unwrap();
    let config: ai_agents_state::StateConfig = serde_yaml::from_str("initial: a\nstates:\n  a:\n    tools:\n      - id: echo\n        condition:\n          semantic:\n            when: Allow.\n            llm: missing\n").unwrap();
    let evaluator = Arc::new(ai_agents_state::LLMTransitionEvaluator::new(Arc::new(
        providers[&LLMRole::StateTransition].clone(),
    )));
    let agent = agent.with_state_machine(
        Arc::new(ai_agents_state::StateMachine::new(config).unwrap()),
        evaluator,
    );
    assert!(matches!(
        agent.get_available_tool_ids().await,
        Err(AgentError::Config(_))
    ));
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 0);
}

#[tokio::test]
async fn hierarchy_freeze_covers_observation_and_snapshot_uses_actual_process_and_hitl() {
    let (spec, registry, _, _) = fixture("spawner:\n  shared_llms: true\n");
    let builder = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .auto_configure_spawner()
        .await
        .unwrap();
    let manager = ai_agents_observability::ObservabilityManager::new(Default::default());
    assert!(builder.observability(manager).build().is_err());
    let (spec, registry, providers, main) = fixture("");
    let process: ai_agents_process::ProcessConfig = serde_yaml::from_str(
        "input:\n  - type: transform\n    config:\n      llm: main\n      prompt: Rewrite.\n",
    )
    .unwrap();
    let hitl: ai_agents_hitl::HITLConfig = serde_yaml::from_str(
        "message_language:\n  strategy: llm_generate\n  llm_generate:\n    llm: main\n",
    )
    .unwrap();
    let agent = AgentBuilder::from_spec(spec.clone())
        .llm_registry(registry)
        .process_processor(ai_agents_process::ProcessProcessor::new(process.clone()))
        .hitl_engine(ai_agents_hitl::HITLEngine::new(hitl.clone()))
        .build()
        .unwrap();
    let captured = agent.prepared_persistence_spec(&spec).unwrap();
    let mut expected = spec.clone();
    expected.process = process;
    expected.hitl = Some(hitl);
    assert_eq!(
        captured.routing_projection().unwrap(),
        expected.routing_projection().unwrap()
    );
    assert_ne!(
        captured.routing_projection().unwrap(),
        spec.routing_projection().unwrap()
    );
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 0);
}

#[test]
fn hierarchy_recovery_preflight_and_prompt_independent_snapshot_use_actual_config() {
    let (spec, registry, providers, main) = fixture("");
    let config: ai_agents_recovery::ErrorRecoveryConfig = serde_yaml::from_str(
        "llm:\n  on_context_overflow:\n    action: summarize\n    summarizer_llm: missing\n",
    )
    .unwrap();
    assert!(
        AgentBuilder::from_spec(spec.clone())
            .llm_registry(registry.clone())
            .recovery_manager(ai_agents_recovery::RecoveryManager::try_new(config).unwrap())
            .build()
            .is_err()
    );
    let config: ai_agents_recovery::ErrorRecoveryConfig = serde_yaml::from_str("llm:\n  on_context_overflow:\n    action: summarize\n    summarizer_llm: main\n    custom_prompt: Original.\n").unwrap();
    let agent = AgentBuilder::from_spec(spec.clone())
        .llm_registry(registry)
        .recovery_manager(ai_agents_recovery::RecoveryManager::try_new(config).unwrap())
        .build()
        .unwrap();
    let captured = agent.prepared_persistence_spec(&spec).unwrap();
    assert_ne!(
        captured.routing_projection().unwrap(),
        spec.routing_projection().unwrap()
    );
    let mut prompt_only = captured.clone();
    if let ai_agents_recovery::ContextOverflowAction::Summarize { custom_prompt, .. } =
        &mut prompt_only.error_recovery.llm.on_context_overflow
    {
        *custom_prompt = Some("Different.".into());
    }
    assert_eq!(
        captured.routing_projection().unwrap(),
        prompt_only.routing_projection().unwrap()
    );
    counts(&providers, &[]);
    assert_eq!(main.call_count(), 0);
}

#[test]
fn hierarchy_framework_web_slots_are_agent_local_but_legacy_mapping_is_unchanged() {
    let source = ai_agents_tools::create_builtin_registry();
    let first_tools = source.map_tools(|tool| tool);
    let second_tools = source.map_tools(|tool| tool);
    assert!(Arc::ptr_eq(
        &source.web_fetch_extractor_slot(),
        &first_tools.web_fetch_extractor_slot()
    ));
    let (spec, registry, _, _) = fixture("");
    let first = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .tools(first_tools)
        .build()
        .unwrap();
    let (spec, registry, _, _) = fixture("");
    let second = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .tools(second_tools)
        .build()
        .unwrap();
    let first_slot = first.tools.web_fetch_extractor_slot();
    let second_slot = second.tools.web_fetch_extractor_slot();
    assert!(!Arc::ptr_eq(&first_slot, &second_slot));
    assert!(source.web_fetch_extractor_slot().read().is_none());
    assert!(Arc::ptr_eq(
        first_slot.read().as_ref().unwrap(),
        &first.llm_registry.get("web.extract").unwrap()
    ));
    assert!(Arc::ptr_eq(
        second_slot.read().as_ref().unwrap(),
        &second.llm_registry.get("web.extract").unwrap()
    ));
}

#[test]
fn shared_child_modes_preserve_only_legacy_scalar_inheritance() {
    use ai_agents_llm::RouterSelector;
    for parent_hierarchy in [false, true] {
        let mut parent = LLMRegistry::new();
        for alias in ["main", "parent", "child"] {
            parent.register(alias, Arc::new(MockLLMProvider::new(alias)));
        }
        parent.set_default("main");
        if parent_hierarchy {
            parent.set_router_roles(RouterRolesConfig {
                default: Some("parent".into()),
                ..Default::default()
            });
        } else {
            parent.set_router("parent");
        }
        for child_mode in [0, 1, 2] {
            let mut selector = LLMSelector::new("main");
            selector.router = match child_mode {
                0 => None,
                1 => Some(RouterSelector::Alias("child".into())),
                _ => Some(RouterSelector::Hierarchical(Box::default())),
            };
            let spec = AgentSpec {
                llm: LLMConfigOrSelector::Selector(selector),
                ..Default::default()
            };
            let child = AgentBuilder::from_spec(spec)
                .authoritative_llm_registry(parent.clone(), false)
                .build()
                .unwrap();
            let expected = if child_mode == 1 {
                "child"
            } else if child_mode == 0 && !parent_hierarchy {
                "parent"
            } else {
                "main"
            };
            assert!(Arc::ptr_eq(
                &child.llm_registry.router().unwrap(),
                &parent.get(expected).unwrap()
            ));
            assert_eq!(child.llm_registry.router_roles().is_some(), child_mode == 2);
        }
        assert!(Arc::ptr_eq(
            &parent.router().unwrap(),
            &parent.get("parent").unwrap()
        ));
    }
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn hierarchy_fact_and_relationship_factories_capture_role_providers() {
    let (spec, registry, mut providers, main) = fixture(
        "memory:\n  facts:\n    enabled: true\n  relationships:\n    enabled: true\n    auto_update:\n      enabled: true\n",
    );
    providers
        .get_mut(&LLMRole::MemoryFacts)
        .unwrap()
        .set_response("[]");
    providers
        .get_mut(&LLMRole::MemoryRelationships)
        .unwrap()
        .set_response(r#"{"changes":[]}"#);
    let storage = Arc::new(ai_agents_storage::SqliteStorage::in_memory().await.unwrap());
    let agent = AgentBuilder::from_spec(spec)
        .llm_registry(registry)
        .storage(storage)
        .build()
        .unwrap();
    agent.init_storage().await.unwrap();
    let extractor = agent.fact_extractor.read().clone().unwrap();
    extractor
        .extract(&[ChatMessage::user("remember")], &[], Some("actor"), &[])
        .await
        .unwrap();
    agent
        .relationship_manager
        .as_ref()
        .unwrap()
        .auto_update("actor", &[ChatMessage::user("remember")])
        .await
        .unwrap();
    counts(
        &providers,
        &[(LLMRole::MemoryFacts, 1), (LLMRole::MemoryRelationships, 1)],
    );
    assert_eq!(main.call_count(), 0);
}
