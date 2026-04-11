use serde::{Deserialize, Serialize};

use ai_agents_core::AgentResponse;

/// Result from routing to the best-matched agent.
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub response: AgentResponse,
    pub selected_agent: String,
    pub reason: String,
    pub confidence: Option<f64>,
}

/// Result from a sequential pipeline of agents.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub response: AgentResponse,
    pub stage_outputs: Vec<StageOutput>,
}

/// Output from a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageOutput {
    pub agent_id: String,
    pub output: String,
    pub duration_ms: u64,
    pub skipped: bool,
}

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub agent_id: String,
    /// Optional Jinja2 input template for this stage.
    /// Available variables:
    ///   `{{ previous_output }}`    - output from the immediately previous stage.
    ///   `{{ original_input }}`     - the user's original input.
    ///   `{{ user_input }}`         - alias for original_input (consistent with concurrent).
    ///   `{{ stages.<agent_id> }}` - output from any earlier stage by agent ID.
    /// When None, stage 1 receives `original_input`, stages 2+ receive `previous_output`.
    pub input: Option<String>,
}

impl PipelineStage {
    pub fn id(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            input: None,
        }
    }

    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }
}

impl From<String> for PipelineStage {
    fn from(agent_id: String) -> Self {
        Self::id(agent_id)
    }
}

impl From<&str> for PipelineStage {
    fn from(agent_id: &str) -> Self {
        Self::id(agent_id)
    }
}

/// Result from concurrent agent execution.
#[derive(Debug, Clone)]
pub struct ConcurrentResult {
    pub response: AgentResponse,
    pub agent_results: Vec<AgentResult>,
    pub aggregation_strategy: String,
}

/// Result from a single agent in a concurrent execution.
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub agent_id: String,
    pub response: Option<AgentResponse>,
    pub duration_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

/// Result from a group chat session.
#[derive(Debug, Clone)]
pub struct GroupChatResult {
    pub response: AgentResponse,
    pub transcript: Vec<ChatTurn>,
    pub rounds_completed: u32,
    pub termination_reason: String,
}

/// A single turn in a group chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    pub speaker: String,
    pub round: u32,
    pub content: String,
}

/// Result from a handoff chain.
#[derive(Debug, Clone)]
pub struct HandoffResult {
    pub response: AgentResponse,
    pub handoff_chain: Vec<HandoffEvent>,
    pub final_agent: String,
}

/// A single handoff event between agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffEvent {
    pub from_agent: String,
    pub to_agent: String,
    pub reason: String,
}

/// Routing method for the route function.
#[derive(Debug, Clone, Default)]
pub enum RoutingMethod {
    #[default]
    Llm,
    RoundRobin,
}
