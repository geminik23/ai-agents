use ai_agents_core::{AgentError, Result};
use ai_agents_disambiguation::StateDisambiguationOverride;
use ai_agents_process::ProcessConfig;
use ai_agents_reasoning::{ReasoningConfig, ReflectionConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PromptMode {
    #[default]
    Append,
    Replace,
    Prepend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateConfig {
    pub initial: String,
    #[serde(default)]
    pub states: HashMap<String, StateDefinition>,
    #[serde(default)]
    pub global_transitions: Vec<Transition>,
    #[serde(default)]
    pub fallback: Option<String>,
    #[serde(default)]
    pub max_no_transition: Option<u32>,

    /// Whether to re-generate a response after state transitions (default: true).
    #[serde(default = "default_true")]
    pub regenerate_on_transition: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateDefinition {
    #[serde(default)]
    pub prompt: Option<String>,

    #[serde(default)]
    pub prompt_mode: PromptMode,

    #[serde(default)]
    pub llm: Option<String>,

    #[serde(default)]
    pub skills: Vec<String>,

    /// Tool availability for this state.
    /// - `None` (omitted in YAML): inherit from parent or agent-level tools
    /// - `Some([])` (`tools: []` in YAML): explicitly no tools available
    /// - `Some([...])`: only these tools available
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolRef>>,

    #[serde(default)]
    pub transitions: Vec<Transition>,

    #[serde(default)]
    pub max_turns: Option<u32>,

    #[serde(default)]
    pub timeout_to: Option<String>,

    #[serde(default)]
    pub initial: Option<String>,

    #[serde(default)]
    pub states: Option<HashMap<String, StateDefinition>>,

    #[serde(default = "default_inherit_parent")]
    pub inherit_parent: bool,

    #[serde(default)]
    pub on_enter: Vec<StateAction>,

    /// Actions on re-entering a previously visited state. Falls back to on_enter if empty.
    #[serde(default)]
    pub on_reenter: Vec<StateAction>,

    #[serde(default)]
    pub on_exit: Vec<StateAction>,

    /// Per-state override: skip re-generation on entering this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regenerate_on_enter: Option<bool>,

    /// Context extractors: pull structured data from user input into context.
    #[serde(default)]
    pub extract: Vec<ContextExtractor>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection: Option<ReflectionConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disambiguation: Option<StateDisambiguationOverride>,

    /// Per-state process pipeline override (replaces agent-level pipeline for this state).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process: Option<ProcessConfig>,

    /// Delegate state messages to a registry agent by ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegate: Option<String>,

    /// Context mode for delegated states.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegate_context: Option<DelegateContextMode>,

    /// Run multiple registry agents concurrently in this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrent: Option<ConcurrentStateConfig>,

    /// Run a multi-agent group chat in this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_chat: Option<GroupChatStateConfig>,

    /// Run a sequential agent pipeline in this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<PipelineStateConfig>,

    /// Run an LLM-directed handoff chain in this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff: Option<HandoffStateConfig>,
}

fn default_inherit_parent() -> bool {
    true
}

fn default_true() -> bool {
    true
}

fn default_extractor_llm() -> String {
    "router".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolRef {
    Simple(String),
    Conditional {
        id: String,
        condition: ToolCondition,
    },
}

impl ToolRef {
    pub fn id(&self) -> &str {
        match self {
            ToolRef::Simple(id) => id,
            ToolRef::Conditional { id, .. } => id,
        }
    }

    pub fn condition(&self) -> Option<&ToolCondition> {
        match self {
            ToolRef::Simple(_) => None,
            ToolRef::Conditional { condition, .. } => Some(condition),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCondition {
    Context(HashMap<String, ContextMatcher>),
    State(StateMatcher),
    AfterTool(String),
    ToolResult {
        tool: String,
        result: HashMap<String, Value>,
    },
    Semantic {
        when: String,
        #[serde(default = "default_semantic_llm")]
        llm: String,
        #[serde(default = "default_threshold")]
        threshold: f32,
    },
    Time(TimeMatcher),
    All(Vec<ToolCondition>),
    Any(Vec<ToolCondition>),
    Not(Box<ToolCondition>),
}

fn default_semantic_llm() -> String {
    "router".to_string()
}

fn default_threshold() -> f32 {
    0.7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContextMatcher {
    // Order matters for serde untagged: structured variants must come before
    // Exact(Value) because Value matches any valid JSON — including objects
    // like `{ "exists": true }` or `{ "eq": "admin" }` that should be parsed
    // as Exists or Compare instead.
    Exists { exists: bool },
    Compare(CompareOp),
    Exact(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Eq(Value),
    Neq(Value),
    Gt(f64),
    Gte(f64),
    Lt(f64),
    Lte(f64),
    In(Vec<Value>),
    Contains(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateMatcher {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub turn_count: Option<CompareOp>,
    #[serde(default)]
    pub previous: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeMatcher {
    #[serde(default)]
    pub hours: Option<CompareOp>,
    #[serde(default)]
    pub day_of_week: Option<Vec<String>>,
    #[serde(default)]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub to: String,
    #[serde(default)]
    pub when: String,
    #[serde(default)]
    pub guard: Option<TransitionGuard>,
    /// Intent label for deterministic routing after disambiguation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    #[serde(default = "default_auto")]
    pub auto: bool,
    #[serde(default)]
    pub priority: u8,

    /// Minimum turns before this transition can fire again after last use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown_turns: Option<u32>,
}

fn default_auto() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TransitionGuard {
    Expression(String),
    Conditions(GuardConditions),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardConditions {
    All(Vec<String>),
    Any(Vec<String>),
    Context(HashMap<String, ContextMatcher>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StateAction {
    Tool {
        tool: String,
        #[serde(default)]
        args: Option<Value>,
    },
    Skill {
        skill: String,
    },
    Prompt {
        prompt: String,
        #[serde(default)]
        llm: Option<String>,
        #[serde(default)]
        store_as: Option<String>,
    },
    SetContext {
        set_context: HashMap<String, Value>,
    },
}

/// Extract structured data from conversation into context via LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextExtractor {
    /// Context key to store the extracted value.
    pub key: String,

    /// Short description of what to extract (LLM-based).
    #[serde(default)]
    pub description: Option<String>,

    /// Custom LLM extraction prompt (takes precedence over `description`).
    #[serde(default)]
    pub llm_extract: Option<String>,

    /// LLM alias for extraction (default: "router").
    #[serde(default = "default_extractor_llm")]
    pub llm: String,

    /// If true, extraction failure is logged as a warning.
    #[serde(default)]
    pub required: bool,
}

//
// Multi-agent orchestration config types for state delegation, concurrent execution, and group chat.
//

/// Context mode for delegated states.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelegateContextMode {
    /// Delegated agent receives only the user's current message.
    #[default]
    InputOnly,
    /// Parent summarizes recent conversation via router LLM.
    Summary,
    /// Parent passes full recent message history.
    Full,
}

/// Config for running multiple registry agents concurrently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentStateConfig {
    /// Agent IDs in the registry (simple list or weighted entries).
    pub agents: Vec<ConcurrentAgentRef>,
    /// Jinja2 template for input sent to each agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// How to aggregate results from all agents.
    pub aggregation: AggregationConfig,
    /// Minimum agents that must succeed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_required: Option<usize>,
    /// What to do when some agents fail.
    #[serde(default)]
    pub on_partial_failure: PartialFailureAction,
    /// Per-agent timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Parent conversation context forwarded to each agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_mode: Option<DelegateContextMode>,
}

/// Either a plain agent ID string or a weighted entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConcurrentAgentRef {
    Id(String),
    Weighted { id: String, weight: f64 },
}

impl ConcurrentAgentRef {
    pub fn id(&self) -> &str {
        match self {
            Self::Id(id) => id,
            Self::Weighted { id, .. } => id,
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            Self::Id(_) => 1.0,
            Self::Weighted { weight, .. } => *weight,
        }
    }
}

/// How to aggregate results from concurrent agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Aggregation strategy.
    pub strategy: AggregationStrategy,
    /// LLM alias for synthesis or vote extraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesizer_llm: Option<String>,
    /// Custom prompt for LLM synthesis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesizer_prompt: Option<String>,
    /// Voting sub-config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vote: Option<VoteConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationStrategy {
    Voting,
    LlmSynthesis,
    FirstWins,
    All,
}

/// Voting config for concurrent agent aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteConfig {
    #[serde(default)]
    pub method: VoteMethod,
    #[serde(default)]
    pub tiebreaker: TiebreakerStrategy,
    /// Custom prompt for extracting a vote from each agent's response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vote_prompt: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteMethod {
    #[default]
    Majority,
    Weighted,
    Unanimous,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TiebreakerStrategy {
    #[default]
    First,
    Random,
    RouterDecides,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialFailureAction {
    #[default]
    ProceedWithAvailable,
    Abort,
}

/// Group chat state config for multi-agent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupChatStateConfig {
    /// Participant agent IDs with optional roles.
    pub participants: Vec<ChatParticipant>,
    /// Conversation style.
    #[serde(default)]
    pub style: ChatStyle,
    /// Maximum conversation rounds.
    #[serde(default = "default_max_rounds")]
    pub max_rounds: u32,
    /// Chat manager config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manager: Option<ChatManagerConfig>,
    /// When and how to terminate.
    #[serde(default)]
    pub termination: TerminationConfig,
    /// Debate-specific config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debate: Option<DebateStyleConfig>,
    /// Maker-checker-specific config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maker_checker: Option<MakerCheckerConfig>,
    /// Total timeout for the group chat in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Jinja2 template for the topic sent to participants.
    /// {{ user_input }} is the user's message. {{ context.<key> }} accesses context values. When omitted, the raw user message is used as the topic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// Parent conversation context included in the topic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_mode: Option<DelegateContextMode>,
}

/// A participant in a group chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatParticipant {
    /// Agent ID in the registry.
    pub id: String,
    /// Role description visible to all participants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatStyle {
    #[default]
    Brainstorm,
    Debate,
    MakerChecker,
    Consensus,
}

/// Chat manager config for controlling turn order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatManagerConfig {
    /// Registry agent ID for chat management.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// Built-in turn policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<TurnMethod>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnMethod {
    RoundRobin,
    Random,
    LlmDirected,
}

/// Termination config for group chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminationConfig {
    #[serde(default)]
    pub method: TerminationMethod,
    #[serde(default = "default_stall_rounds")]
    pub max_stall_rounds: u32,
}

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            method: TerminationMethod::default(),
            max_stall_rounds: default_stall_rounds(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationMethod {
    #[default]
    ManagerDecides,
    MaxRounds,
    ConsensusReached,
}

/// Debate-specific config for group chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateStyleConfig {
    #[serde(default = "default_debate_rounds")]
    pub rounds: u32,
    /// Agent ID that synthesizes the final answer.
    pub synthesizer: String,
}

/// Maker-checker-specific config for group chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MakerCheckerConfig {
    #[serde(default = "default_maker_checker_iterations")]
    pub max_iterations: u32,
    /// LLM-evaluated acceptance criteria.
    pub acceptance_criteria: String,
    #[serde(default)]
    pub on_max_iterations: MaxIterationsAction,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxIterationsAction {
    #[default]
    AcceptLast,
    Escalate,
    Fail,
}

fn default_max_rounds() -> u32 {
    5
}
fn default_stall_rounds() -> u32 {
    2
}
fn default_debate_rounds() -> u32 {
    3
}
fn default_maker_checker_iterations() -> u32 {
    3
}

/// Config for a pipeline state type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStateConfig {
    pub stages: Vec<PipelineStageEntry>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Parent conversation context forwarded to the first stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_mode: Option<DelegateContextMode>,
}

/// A single stage in a pipeline state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PipelineStageEntry {
    /// Simple agent ID string.
    Id(String),
    /// Agent with optional input template.
    Config {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<String>,
    },
}

impl PipelineStageEntry {
    pub fn id(&self) -> &str {
        match self {
            Self::Id(id) => id,
            Self::Config { id, .. } => id,
        }
    }

    pub fn input(&self) -> Option<&str> {
        match self {
            Self::Id(_) => None,
            Self::Config { input, .. } => input.as_deref(),
        }
    }
}

/// Config for a handoff state type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffStateConfig {
    pub initial_agent: String,
    pub available_agents: Vec<String>,

    #[serde(default = "default_max_handoffs")]
    pub max_handoffs: u32,

    /// Jinja2 template for the input sent to the initial agent.
    /// {{ user_input }} is the user's message. {{ context.<key> }} accesses context values. When omitted, the raw user message is forwarded directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// Parent conversation context forwarded to the initial agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_mode: Option<DelegateContextMode>,
}

fn default_max_handoffs() -> u32 {
    5
}

impl StateConfig {
    pub fn validate(&self) -> Result<()> {
        if self.initial.is_empty() {
            return Err(AgentError::InvalidSpec(
                "State machine initial state cannot be empty".into(),
            ));
        }
        if !self.states.contains_key(&self.initial) {
            return Err(AgentError::InvalidSpec(format!(
                "Initial state '{}' not found in states",
                self.initial
            )));
        }
        self.validate_states(&self.states, &[])?;

        // Warn about unreachable states (non-fatal)
        for warning in self.check_reachability() {
            tracing::warn!("{}", warning);
        }

        Ok(())
    }

    fn validate_states(
        &self,
        states: &HashMap<String, StateDefinition>,
        parent_path: &[String],
    ) -> Result<()> {
        for (name, def) in states {
            let current_path: Vec<String> = parent_path
                .iter()
                .cloned()
                .chain(std::iter::once(name.clone()))
                .collect();

            for transition in &def.transitions {
                if !self.is_valid_transition_target(&transition.to, &current_path, states) {
                    return Err(AgentError::InvalidSpec(format!(
                        "State '{}' has transition to unknown state '{}'",
                        current_path.join("."),
                        transition.to
                    )));
                }
            }

            if let Some(ref timeout_state) = def.timeout_to {
                if !self.is_valid_transition_target(timeout_state, &current_path, states) {
                    return Err(AgentError::InvalidSpec(format!(
                        "State '{}' has timeout_to unknown state '{}'",
                        current_path.join("."),
                        timeout_state
                    )));
                }
            }

            if let Some(ref sub_states) = def.states {
                if let Some(ref initial) = def.initial {
                    if !sub_states.contains_key(initial) {
                        return Err(AgentError::InvalidSpec(format!(
                            "State '{}' has initial sub-state '{}' that doesn't exist",
                            current_path.join("."),
                            initial
                        )));
                    }
                }
                self.validate_states(sub_states, &current_path)?;
            }
        }
        Ok(())
    }

    fn is_valid_transition_target(
        &self,
        target: &str,
        current_path: &[String],
        states: &HashMap<String, StateDefinition>,
    ) -> bool {
        if target.starts_with('^') {
            let target_name = &target[1..];
            return self.states.contains_key(target_name);
        }

        if states.contains_key(target) {
            return true;
        }

        if current_path.len() > 1 {
            let parent_path = &current_path[..current_path.len() - 1];
            if let Some(parent_states) = self.get_states_at_path(parent_path) {
                if parent_states.contains_key(target) {
                    return true;
                }
            }
        }

        self.states.contains_key(target)
    }

    fn get_states_at_path(&self, path: &[String]) -> Option<&HashMap<String, StateDefinition>> {
        let mut current = &self.states;
        for segment in path {
            if let Some(def) = current.get(segment) {
                if let Some(ref sub_states) = def.states {
                    current = sub_states;
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        Some(current)
    }

    pub fn get_state(&self, path: &str) -> Option<&StateDefinition> {
        let parts: Vec<&str> = path.split('.').collect();
        self.get_state_by_path(&parts)
    }

    fn get_state_by_path(&self, path: &[&str]) -> Option<&StateDefinition> {
        if path.is_empty() {
            return None;
        }

        let mut current = self.states.get(path[0])?;
        for segment in &path[1..] {
            if let Some(ref sub_states) = current.states {
                current = sub_states.get(*segment)?;
            } else {
                return None;
            }
        }
        Some(current)
    }

    /// Resolve a transition target to a full dotted state path.
    /// Order: `^prefix` (parent-level) → top-level → sibling → child → fallback literal.
    pub fn resolve_full_path(&self, current_path: &str, target: &str) -> String {
        if target.starts_with('^') {
            return target[1..].to_string();
        }

        if self.states.contains_key(target) {
            return target.to_string();
        }

        if !current_path.is_empty() {
            let parts: Vec<&str> = current_path.split('.').collect();
            if parts.len() > 1 {
                let parent_path = parts[..parts.len() - 1].join(".");
                let potential = format!("{}.{}", parent_path, target);
                if self.get_state(&potential).is_some() {
                    return potential;
                }
            }

            let potential = format!("{}.{}", current_path, target);
            if self.get_state(&potential).is_some() {
                return potential;
            }
        }

        target.to_string()
    }

    /// Check for unreachable states. Returns warning messages.
    pub fn check_reachability(&self) -> Vec<String> {
        let mut reachable: HashSet<String> = HashSet::new();
        reachable.insert(self.initial.clone());

        if let Some(ref fb) = self.fallback {
            reachable.insert(fb.clone());
        }
        for gt in &self.global_transitions {
            reachable.insert(self.normalize_target(&gt.to));
        }

        let mut queue: Vec<String> = reachable.iter().cloned().collect();
        while let Some(state_path) = queue.pop() {
            if let Some(def) = self.get_state(&state_path) {
                for t in &def.transitions {
                    let target = self.resolve_full_path(&state_path, &t.to);
                    if reachable.insert(target.clone()) {
                        queue.push(target);
                    }
                }
                if let Some(ref timeout) = def.timeout_to {
                    let target = self.resolve_full_path(&state_path, timeout);
                    if reachable.insert(target.clone()) {
                        queue.push(target);
                    }
                }
                if let (Some(initial), Some(_sub)) = (&def.initial, &def.states) {
                    let sub_path = format!("{}.{}", state_path, initial);
                    if reachable.insert(sub_path.clone()) {
                        queue.push(sub_path);
                    }
                }
            }
        }

        let all_states = self.collect_all_state_paths(&self.states, &[]);
        let mut warnings = Vec::new();
        for state_path in &all_states {
            if !reachable.contains(state_path) {
                warnings.push(format!(
                    "State '{}' appears unreachable — no transitions lead to it",
                    state_path
                ));
            }
        }
        warnings
    }

    fn normalize_target(&self, target: &str) -> String {
        if target.starts_with('^') {
            target[1..].to_string()
        } else {
            target.to_string()
        }
    }

    fn collect_all_state_paths(
        &self,
        states: &HashMap<String, StateDefinition>,
        parent: &[String],
    ) -> Vec<String> {
        let mut paths = Vec::new();
        for (name, def) in states {
            let mut current: Vec<String> = parent.to_vec();
            current.push(name.clone());
            paths.push(current.join("."));
            if let Some(ref sub) = def.states {
                paths.extend(self.collect_all_state_paths(sub, &current));
            }
        }
        paths
    }
}

impl StateDefinition {
    pub fn has_sub_states(&self) -> bool {
        self.states.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
    }

    pub fn get_effective_tools<'a>(
        &'a self,
        parent: Option<&'a StateDefinition>,
    ) -> Option<Vec<&'a ToolRef>> {
        match &self.tools {
            // Explicitly set (including empty): use as-is, no inheritance
            Some(tools) => Some(tools.iter().collect()),
            // Not set: inherit from parent if available
            None => {
                if !self.inherit_parent {
                    return None;
                }
                parent
                    .and_then(|p| p.tools.as_ref())
                    .map(|t| t.iter().collect())
            }
        }
    }

    pub fn get_effective_skills<'a>(
        &'a self,
        parent: Option<&'a StateDefinition>,
    ) -> Vec<&'a String> {
        if !self.inherit_parent || parent.is_none() {
            return self.skills.iter().collect();
        }

        let parent = parent.unwrap();
        let mut skills: Vec<&'a String> = parent.skills.iter().collect();
        skills.extend(self.skills.iter());
        skills
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_config_deserialize() {
        let yaml = r#"
initial: greeting
states:
  greeting:
    prompt: "Welcome!"
    transitions:
      - to: support
        when: "user needs help"
        auto: true
  support:
    prompt: "How can I help?"
    llm: fast
    tools:
      - search
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.initial, "greeting");
        assert_eq!(config.states.len(), 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_prompt_mode_default() {
        let def = StateDefinition::default();
        assert_eq!(def.prompt_mode, PromptMode::Append);
    }

    #[test]
    fn test_invalid_initial_state() {
        let config = StateConfig {
            initial: "nonexistent".into(),
            states: HashMap::new(),
            global_transitions: vec![],
            fallback: None,
            max_no_transition: None,
            regenerate_on_transition: true,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_transition_target() {
        let mut states = HashMap::new();
        states.insert(
            "start".into(),
            StateDefinition {
                transitions: vec![Transition {
                    to: "nonexistent".into(),
                    when: "always".into(),
                    guard: None,
                    intent: None,
                    auto: true,
                    priority: 0,
                    cooldown_turns: None,
                }],
                ..Default::default()
            },
        );
        let config = StateConfig {
            initial: "start".into(),
            states,
            global_transitions: vec![],
            fallback: None,
            max_no_transition: None,
            regenerate_on_transition: true,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hierarchical_states() {
        let yaml = r#"
initial: problem_solving
states:
  problem_solving:
    initial: gathering_info
    prompt: "Solving customer problem"
    states:
      gathering_info:
        prompt: "Ask questions"
        transitions:
          - to: proposing_solution
            when: "understood"
      proposing_solution:
        prompt: "Offer solution"
        transitions:
          - to: ^closing
            when: "resolved"
  closing:
    prompt: "Thank you"
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.validate().is_ok());
        assert!(
            config
                .states
                .get("problem_solving")
                .unwrap()
                .has_sub_states()
        );
    }

    #[test]
    fn test_tool_ref_simple() {
        let yaml = r#"
tools:
  - calculator
  - search
"#;
        #[derive(Deserialize)]
        struct Test {
            tools: Vec<ToolRef>,
        }
        let t: Test = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(t.tools.len(), 2);
        assert_eq!(t.tools[0].id(), "calculator");
    }

    #[test]
    fn test_tool_ref_conditional() {
        let yaml = r#"
tools:
  - calculator
  - id: admin_tool
    condition:
      context:
        user.role: "admin"
"#;
        #[derive(Deserialize)]
        struct Test {
            tools: Vec<ToolRef>,
        }
        let t: Test = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(t.tools.len(), 2);
        assert_eq!(t.tools[1].id(), "admin_tool");
        assert!(t.tools[1].condition().is_some());
    }

    #[test]
    fn test_transition_with_guard() {
        let yaml = r#"
to: next_state
when: "user wants to proceed"
guard: "{{ context.has_data }}"
auto: true
priority: 10
"#;
        let t: Transition = serde_yaml::from_str(yaml).unwrap();
        assert!(t.guard.is_some());
        assert_eq!(t.priority, 10);
    }

    #[test]
    fn test_state_action() {
        let yaml = r#"
- tool: log_event
  args:
    event: "entered"
- skill: greeting_skill
- set_context:
    entered: true
"#;
        let actions: Vec<StateAction> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(actions.len(), 3);
        match &actions[0] {
            StateAction::Tool { tool, .. } => assert_eq!(tool, "log_event"),
            _ => panic!("Expected Tool action"),
        }
        match &actions[1] {
            StateAction::Skill { skill } => assert_eq!(skill, "greeting_skill"),
            _ => panic!("Expected Skill action"),
        }
        match &actions[2] {
            StateAction::SetContext { set_context } => {
                assert!(set_context.contains_key("entered"));
            }
            _ => panic!("Expected SetContext action"),
        }
    }

    #[test]
    fn test_complex_tool_condition() {
        let yaml = r#"
id: refund_tool
condition:
  all:
    - context:
        user.verified: true
    - semantic:
        when: "user wants refund"
        threshold: 0.85
"#;
        let tool: ToolRef = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(tool.id(), "refund_tool");
        match tool.condition().unwrap() {
            ToolCondition::All(conditions) => assert_eq!(conditions.len(), 2),
            _ => panic!("Expected All condition"),
        }
    }

    #[test]
    fn test_state_get_path() {
        let yaml = r#"
initial: problem_solving
states:
  problem_solving:
    initial: gathering_info
    states:
      gathering_info:
        prompt: "Ask"
      proposing:
        prompt: "Propose"
  closing:
    prompt: "Done"
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.get_state("problem_solving").is_some());
        assert!(config.get_state("problem_solving.gathering_info").is_some());
        assert!(config.get_state("closing").is_some());
        assert!(config.get_state("nonexistent").is_none());
    }

    #[test]
    fn test_resolve_full_path() {
        let yaml = r#"
initial: problem_solving
states:
  problem_solving:
    initial: gathering_info
    states:
      gathering_info:
        prompt: "Ask"
      proposing:
        prompt: "Propose"
  closing:
    prompt: "Done"
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(
            config.resolve_full_path("problem_solving.gathering_info", "proposing"),
            "problem_solving.proposing"
        );
        assert_eq!(
            config.resolve_full_path("problem_solving.gathering_info", "^closing"),
            "closing"
        );
        assert_eq!(
            config.resolve_full_path("problem_solving", "closing"),
            "closing"
        );
    }

    #[test]
    fn test_inherit_parent() {
        let parent = StateDefinition {
            tools: Some(vec![ToolRef::Simple("parent_tool".into())]),
            skills: vec!["parent_skill".into()],
            ..Default::default()
        };

        let child = StateDefinition {
            tools: Some(vec![ToolRef::Simple("child_tool".into())]),
            skills: vec!["child_skill".into()],
            inherit_parent: true,
            ..Default::default()
        };

        let effective_tools = child.get_effective_tools(Some(&parent)).unwrap();
        assert_eq!(effective_tools.len(), 1); // explicit tools override, no merge

        let effective_skills = child.get_effective_skills(Some(&parent));
        assert_eq!(effective_skills.len(), 2);
    }

    #[test]
    fn test_no_inherit_parent() {
        let parent = StateDefinition {
            tools: Some(vec![ToolRef::Simple("parent_tool".into())]),
            ..Default::default()
        };

        let child = StateDefinition {
            tools: Some(vec![ToolRef::Simple("child_tool".into())]),
            inherit_parent: false,
            ..Default::default()
        };

        let effective_tools = child.get_effective_tools(Some(&parent)).unwrap();
        assert_eq!(effective_tools.len(), 1);
        assert_eq!(effective_tools[0].id(), "child_tool");
    }

    #[test]
    fn test_tools_none_inherits() {
        let parent = StateDefinition {
            tools: Some(vec![ToolRef::Simple("parent_tool".into())]),
            ..Default::default()
        };

        let child = StateDefinition {
            tools: None, // not specified → inherit
            inherit_parent: true,
            ..Default::default()
        };

        let effective_tools = child.get_effective_tools(Some(&parent)).unwrap();
        assert_eq!(effective_tools.len(), 1);
        assert_eq!(effective_tools[0].id(), "parent_tool");
    }

    #[test]
    fn test_tools_empty_means_no_tools() {
        let parent = StateDefinition {
            tools: Some(vec![ToolRef::Simple("parent_tool".into())]),
            ..Default::default()
        };

        let child = StateDefinition {
            tools: Some(vec![]), // explicitly empty → no tools
            inherit_parent: true,
            ..Default::default()
        };

        let effective_tools = child.get_effective_tools(Some(&parent)).unwrap();
        assert!(effective_tools.is_empty());
    }

    #[test]
    fn test_state_with_disambiguation_override() {
        let yaml = r#"
initial: greeting
states:
  greeting:
    prompt: "Hello"
    transitions:
      - to: payment
        when: "User wants to pay"
  payment:
    prompt: "Processing payment"
    disambiguation:
      threshold: 0.95
      require_confirmation: true
      required_clarity:
        - recipient
        - amount
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        let payment = config.get_state("payment").unwrap();
        let disambig = payment.disambiguation.as_ref().unwrap();
        assert_eq!(disambig.threshold, Some(0.95));
        assert!(disambig.require_confirmation);
        assert_eq!(disambig.required_clarity.len(), 2);
        assert!(disambig.required_clarity.contains(&"recipient".to_string()));

        let greeting = config.get_state("greeting").unwrap();
        assert!(greeting.disambiguation.is_none());
    }

    #[test]
    fn test_context_extractor_vec_deserialize() {
        let yaml = r#"
initial: a
states:
  a:
    extract:
      - key: user_email
        description: "The user's email address"
      - key: order_id
        llm_extract: "Extract the order ID"
        required: true
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        let state = config.get_state("a").unwrap();
        assert_eq!(state.extract.len(), 2);
        assert_eq!(state.extract[0].key, "user_email");
        assert_eq!(
            state.extract[0].description.as_deref(),
            Some("The user's email address")
        );
        assert!(!state.extract[0].required);
        assert_eq!(state.extract[0].llm, "router");
        assert_eq!(state.extract[1].key, "order_id");
        assert!(state.extract[1].required);
        assert!(state.extract[1].llm_extract.is_some());
    }

    #[test]
    fn test_context_extractor_default_empty() {
        let yaml = r#"
initial: a
states:
  a:
    prompt: "Hello"
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        let state = config.get_state("a").unwrap();
        assert!(state.extract.is_empty());
    }

    #[test]
    fn test_state_process_override_deserialize() {
        let yaml = r#"
initial: a
states:
  a:
    process:
      input:
        - type: normalize
          config:
            trim: true
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        let state = config.get_state("a").unwrap();
        assert!(state.process.is_some());
        assert_eq!(state.process.as_ref().unwrap().input.len(), 1);
    }

    #[test]
    fn test_state_process_default_none() {
        let yaml = r#"
initial: a
states:
  a:
    prompt: "Hello"
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        let state = config.get_state("a").unwrap();
        assert!(state.process.is_none());
    }
}
