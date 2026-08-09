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
    type HostIntegrationConfigurator = fn(
        &RuntimeAgent,
        Arc<dyn QuestionHandler>,
        Arc<dyn DiagnosticsProvider>,
        Arc<dyn CommandRunner>,
        Arc<dyn WebSearchProvider>,
    );
    let _: HostIntegrationConfigurator = configure_host_integrations;
}
