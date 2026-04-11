use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use ai_agents_core::{Tool, ToolResult};
use ai_agents_llm::LLMRegistry;
use ai_agents_tools::generate_schema;

use super::types::RoutingMethod;
use crate::spawner::AgentRegistry;

//
// RouteToAgentTool
//

pub struct RouteToAgentTool {
    registry: Arc<AgentRegistry>,
    llm: Arc<LLMRegistry>,
    counter: AtomicUsize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct RouteToAgentInput {
    /// The message to route to the best-matched agent.
    input: String,
    /// Agent IDs to consider for routing.
    candidates: Vec<String>,
    /// Routing method: "llm" or "round_robin". Defaults to "llm".
    method: Option<String>,
}

impl RouteToAgentTool {
    pub fn new(registry: Arc<AgentRegistry>, llm: Arc<LLMRegistry>) -> Self {
        Self {
            registry,
            llm,
            counter: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Tool for RouteToAgentTool {
    fn id(&self) -> &str {
        "route_to_agent"
    }

    fn name(&self) -> &str {
        "Route to Agent"
    }

    fn description(&self) -> &str {
        "Send input to the best-matched agent from a set of candidates."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<RouteToAgentInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input = match args.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field: input"),
        };
        let candidates: Vec<String> = match args.get("candidates").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None => return ToolResult::error("missing required field: candidates"),
        };
        let method = match args.get("method").and_then(|v| v.as_str()) {
            Some("round_robin") => RoutingMethod::RoundRobin,
            _ => RoutingMethod::Llm,
        };

        let llm = match self.llm.get("router") {
            Ok(p) => p,
            Err(_) => return ToolResult::error("no router LLM configured"),
        };

        match super::route(
            &self.registry,
            llm.as_ref(),
            input,
            &candidates,
            method,
            Some(&self.counter),
        )
        .await
        {
            Ok(result) => ToolResult::ok(
                json!({
                    "selected_agent": result.selected_agent,
                    "response": result.response.content,
                    "reason": result.reason,
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("routing failed: {}", e)),
        }
    }
}

//
// PipelineProcessTool
//

pub struct PipelineProcessTool {
    registry: Arc<AgentRegistry>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct PipelineProcessInput {
    /// Initial input for the first pipeline stage.
    input: String,
    /// Ordered agent IDs forming the pipeline stages.
    stages: Vec<String>,
    /// Optional per-stage input templates with {{ previous_output }} and {{ original_input }} variables.
    stage_inputs: Option<Vec<Option<String>>>,
}

impl PipelineProcessTool {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for PipelineProcessTool {
    fn id(&self) -> &str {
        "pipeline_process"
    }

    fn name(&self) -> &str {
        "Pipeline Process"
    }

    fn description(&self) -> &str {
        "Chain agents sequentially. Each agent processes the previous output."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<PipelineProcessInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input = match args.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field: input"),
        };
        let stage_ids: Vec<String> = match args.get("stages").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None => return ToolResult::error("missing required field: stages"),
        };

        let stage_inputs: Vec<Option<String>> = args
            .get("stage_inputs")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let pipeline_stages: Vec<super::types::PipelineStage> = stage_ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| {
                let mut stage = super::types::PipelineStage::id(id);
                if let Some(tmpl) = stage_inputs.get(i).and_then(|o| o.clone()) {
                    stage = stage.with_input(tmpl);
                }
                stage
            })
            .collect();

        match super::pipeline(&self.registry, input, &pipeline_stages, None, None, None).await {
            Ok(result) => ToolResult::ok(
                json!({
                    "response": result.response.content,
                    "stages": result.stage_outputs.iter().map(|s| {
                        json!({
                            "agent_id": s.agent_id,
                            "output": s.output,
                            "duration_ms": s.duration_ms,
                            "skipped": s.skipped,
                        })
                    }).collect::<Vec<_>>(),
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("pipeline failed: {}", e)),
        }
    }
}

//
// ConcurrentAskTool
//

pub struct ConcurrentAskTool {
    registry: Arc<AgentRegistry>,
    llm: Arc<LLMRegistry>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct ConcurrentAskInput {
    /// The question to ask all agents in parallel.
    question: String,
    /// Agent IDs to query concurrently.
    agents: Vec<String>,
    /// Aggregation strategy: "all", "llm_synthesis", "first_wins", or "voting". Defaults to "all".
    aggregation: Option<String>,
}

impl ConcurrentAskTool {
    pub fn new(registry: Arc<AgentRegistry>, llm: Arc<LLMRegistry>) -> Self {
        Self { registry, llm }
    }
}

#[async_trait]
impl Tool for ConcurrentAskTool {
    fn id(&self) -> &str {
        "concurrent_ask"
    }

    fn name(&self) -> &str {
        "Concurrent Ask"
    }

    fn description(&self) -> &str {
        "Ask multiple agents the same question in parallel and aggregate results."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<ConcurrentAskInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let question = match args.get("question").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field: question"),
        };
        let agent_ids: Vec<String> = match args.get("agents").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None => return ToolResult::error("missing required field: agents"),
        };
        let strategy_str = args
            .get("aggregation")
            .and_then(|v| v.as_str())
            .unwrap_or("all");

        let strategy = match strategy_str {
            "llm_synthesis" => ai_agents_state::AggregationStrategy::LlmSynthesis,
            "first_wins" => ai_agents_state::AggregationStrategy::FirstWins,
            "voting" => ai_agents_state::AggregationStrategy::Voting,
            _ => ai_agents_state::AggregationStrategy::All,
        };

        let agents: Vec<ai_agents_state::ConcurrentAgentRef> = agent_ids
            .iter()
            .map(|id| ai_agents_state::ConcurrentAgentRef::Id(id.clone()))
            .collect();

        let aggregation = ai_agents_state::AggregationConfig {
            strategy,
            synthesizer_llm: Some("router".to_string()),
            synthesizer_prompt: None,
            vote: None,
        };

        let llm = self.llm.get("router").ok();

        match super::concurrent(
            &self.registry,
            question,
            &agents,
            &aggregation,
            llm.as_deref(),
            None,
            None,
            ai_agents_state::PartialFailureAction::ProceedWithAvailable,
        )
        .await
        {
            Ok(result) => ToolResult::ok(
                json!({
                    "response": result.response.content,
                    "agent_results": result.agent_results.iter().map(|ar| {
                        json!({
                            "agent_id": ar.agent_id,
                            "response": ar.response.as_ref().map(|r| r.content.as_str()),
                            "success": ar.success,
                        })
                    }).collect::<Vec<_>>(),
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("concurrent ask failed: {}", e)),
        }
    }
}

//
// GroupDiscussionTool
//

pub struct GroupDiscussionTool {
    registry: Arc<AgentRegistry>,
    llm: Arc<LLMRegistry>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct GroupDiscussionInput {
    /// The discussion topic.
    topic: String,
    /// Agent IDs participating in the discussion.
    participants: Vec<String>,
    /// Discussion style: "brainstorm", "debate", or "consensus". Defaults to "brainstorm".
    style: Option<String>,
    /// Maximum conversation rounds. Defaults to 3.
    max_rounds: Option<u32>,
}

impl GroupDiscussionTool {
    pub fn new(registry: Arc<AgentRegistry>, llm: Arc<LLMRegistry>) -> Self {
        Self { registry, llm }
    }
}

#[async_trait]
impl Tool for GroupDiscussionTool {
    fn id(&self) -> &str {
        "group_discussion"
    }

    fn name(&self) -> &str {
        "Group Discussion"
    }

    fn description(&self) -> &str {
        "Run a multi-agent conversation on a topic with configurable style."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<GroupDiscussionInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let topic = match args.get("topic").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field: topic"),
        };
        let participant_ids: Vec<String> = match args.get("participants").and_then(|v| v.as_array())
        {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None => return ToolResult::error("missing required field: participants"),
        };
        let style = match args.get("style").and_then(|v| v.as_str()) {
            Some("debate") => ai_agents_state::ChatStyle::Debate,
            Some("consensus") => ai_agents_state::ChatStyle::Consensus,
            _ => ai_agents_state::ChatStyle::Brainstorm,
        };
        let max_rounds = args
            .get("max_rounds")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(3);

        let participants: Vec<ai_agents_state::ChatParticipant> = participant_ids
            .iter()
            .map(|id| ai_agents_state::ChatParticipant {
                id: id.clone(),
                role: None,
            })
            .collect();

        let config = ai_agents_state::GroupChatStateConfig {
            participants,
            style,
            max_rounds,
            manager: None,
            termination: ai_agents_state::TerminationConfig {
                method: ai_agents_state::TerminationMethod::MaxRounds,
                max_stall_rounds: 2,
            },
            debate: None,
            maker_checker: None,
            timeout_ms: None,
            input: None,
            context_mode: None,
        };

        let llm = self.llm.get("router").ok();

        match super::group_chat(&self.registry, topic, &config, llm.as_deref(), None).await {
            Ok(result) => ToolResult::ok(
                json!({
                    "conclusion": result.response.content,
                    "rounds": result.rounds_completed,
                    "termination": result.termination_reason,
                    "transcript": result.transcript.iter().map(|t| {
                        json!({
                            "speaker": t.speaker,
                            "round": t.round,
                            "content": t.content,
                        })
                    }).collect::<Vec<_>>(),
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("group discussion failed: {}", e)),
        }
    }
}

//
// HandoffConversationTool
//

pub struct HandoffConversationTool {
    registry: Arc<AgentRegistry>,
    llm: Arc<LLMRegistry>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct HandoffConversationInput {
    /// The initial message to start the conversation.
    input: String,
    /// Starting agent ID.
    initial_agent: String,
    /// Agent IDs that can receive handoffs.
    available_agents: Vec<String>,
    /// Maximum number of handoffs allowed. Defaults to 5.
    max_handoffs: Option<u32>,
}

impl HandoffConversationTool {
    pub fn new(registry: Arc<AgentRegistry>, llm: Arc<LLMRegistry>) -> Self {
        Self { registry, llm }
    }
}

#[async_trait]
impl Tool for HandoffConversationTool {
    fn id(&self) -> &str {
        "handoff_conversation"
    }

    fn name(&self) -> &str {
        "Handoff Conversation"
    }

    fn description(&self) -> &str {
        "Start a conversation with one agent and allow dynamic handoffs to other agents."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<HandoffConversationInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input = match args.get("input").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field: input"),
        };
        let initial_agent = match args.get("initial_agent").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("missing required field: initial_agent"),
        };
        let available_agents: Vec<String> =
            match args.get("available_agents").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
                None => return ToolResult::error("missing required field: available_agents"),
            };
        let max_handoffs = args
            .get("max_handoffs")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(5);

        let llm = match self.llm.get("router") {
            Ok(p) => p,
            Err(_) => return ToolResult::error("no router LLM configured"),
        };

        match super::handoff(
            &self.registry,
            input,
            initial_agent,
            &available_agents,
            max_handoffs,
            llm.as_ref(),
            None,
        )
        .await
        {
            Ok(result) => ToolResult::ok(
                json!({
                    "response": result.response.content,
                    "final_agent": result.final_agent,
                    "handoffs": result.handoff_chain.iter().map(|h| {
                        json!({
                            "from": h.from_agent,
                            "to": h.to_agent,
                            "reason": h.reason,
                        })
                    }).collect::<Vec<_>>(),
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("handoff failed: {}", e)),
        }
    }
}

/// Build the list of orchestration tools enabled by the config.
pub fn configure_orchestration_tools(
    config: &crate::spec::OrchestrationToolsConfig,
    registry: Arc<AgentRegistry>,
    llm: Arc<LLMRegistry>,
) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

    if config.includes("route_to_agent") {
        tools.push(Arc::new(RouteToAgentTool::new(
            Arc::clone(&registry),
            Arc::clone(&llm),
        )));
    }
    if config.includes("pipeline_process") {
        tools.push(Arc::new(PipelineProcessTool::new(Arc::clone(&registry))));
    }
    if config.includes("concurrent_ask") {
        tools.push(Arc::new(ConcurrentAskTool::new(
            Arc::clone(&registry),
            Arc::clone(&llm),
        )));
    }
    if config.includes("group_discussion") {
        tools.push(Arc::new(GroupDiscussionTool::new(
            Arc::clone(&registry),
            Arc::clone(&llm),
        )));
    }
    if config.includes("handoff_conversation") {
        tools.push(Arc::new(HandoffConversationTool::new(
            Arc::clone(&registry),
            Arc::clone(&llm),
        )));
    }

    tools
}
