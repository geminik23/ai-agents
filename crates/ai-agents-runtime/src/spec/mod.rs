//! Agent specification types

mod llm;
mod memory;
mod provider;
mod storage;
mod tool;

pub use llm::{LLMConfig, LLMSelector};
pub use memory::MemoryConfig;
pub use provider::{
    BuiltinProviderConfig, ProviderPolicyConfig, ProviderSecurityConfig, ProvidersConfig,
    ToolAliasesConfig, ToolPolicyConfig, YamlProviderConfig, YamlToolConfig,
};
pub use storage::{FileStorageConfig, RedisStorageConfig, SqliteStorageConfig, StorageConfig};
pub use tool::ToolConfig;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use ai_agents_context::ContextSource;
use ai_agents_core::{AgentError, Result};
use ai_agents_hitl::HITLConfig;
use ai_agents_process::ProcessConfig;
use ai_agents_reasoning::{ReasoningConfig, ReflectionConfig};
use ai_agents_recovery::ErrorRecoveryConfig;
use ai_agents_skills::SkillRef;
use ai_agents_state::StateConfig;
use ai_agents_tools::ToolSecurityConfig;

use super::{ParallelToolsConfig, StreamingConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,

    #[serde(default = "default_version")]
    pub version: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    pub system_prompt: String,

    #[serde(default)]
    pub llm: LLMConfigOrSelector,

    #[serde(default)]
    pub llms: HashMap<String, LLMConfig>,

    #[serde(default)]
    pub skills: Vec<SkillRef>,

    #[serde(default)]
    pub memory: MemoryConfig,

    #[serde(default)]
    pub storage: StorageConfig,

    #[serde(default)]
    pub tools: Vec<ToolConfig>,

    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: u32,

    #[serde(default)]
    pub error_recovery: ErrorRecoveryConfig,

    #[serde(default)]
    pub tool_security: ToolSecurityConfig,

    #[serde(default)]
    pub process: ProcessConfig,

    #[serde(default)]
    pub context: HashMap<String, ContextSource>,

    #[serde(default)]
    pub states: Option<StateConfig>,

    #[serde(default)]
    pub parallel_tools: ParallelToolsConfig,

    #[serde(default)]
    pub streaming: StreamingConfig,

    #[serde(default)]
    pub hitl: Option<HITLConfig>,

    #[serde(default)]
    pub reasoning: ReasoningConfig,

    #[serde(default)]
    pub reflection: ReflectionConfig,

    #[serde(default)]
    pub providers: ProvidersConfig,

    #[serde(default)]
    pub provider_security: ProviderSecurityConfig,

    #[serde(default)]
    pub tool_aliases: ToolAliasesConfig,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LLMConfigOrSelector {
    Config(LLMConfig),
    Selector(LLMSelector),
}

impl Default for LLMConfigOrSelector {
    fn default() -> Self {
        LLMConfigOrSelector::Config(LLMConfig::default())
    }
}

impl LLMConfigOrSelector {
    pub fn as_config(&self) -> Option<&LLMConfig> {
        match self {
            LLMConfigOrSelector::Config(c) => Some(c),
            LLMConfigOrSelector::Selector(_) => None,
        }
    }

    pub fn as_selector(&self) -> Option<&LLMSelector> {
        match self {
            LLMConfigOrSelector::Config(_) => None,
            LLMConfigOrSelector::Selector(s) => Some(s),
        }
    }

    pub fn get_default_alias(&self) -> String {
        match self {
            LLMConfigOrSelector::Config(_) => "default".to_string(),
            LLMConfigOrSelector::Selector(s) => s.default.clone(),
        }
    }

    pub fn get_router_alias(&self) -> Option<String> {
        match self {
            LLMConfigOrSelector::Config(_) => None,
            LLMConfigOrSelector::Selector(s) => s.router.clone(),
        }
    }
}

fn default_version() -> String {
    "1.0.0".to_string()
}

fn default_max_iterations() -> u32 {
    10
}

fn default_max_context_tokens() -> u32 {
    4096
}

impl Default for AgentSpec {
    fn default() -> Self {
        Self {
            name: "Agent".to_string(),
            version: default_version(),
            description: None,
            system_prompt: "You are a helpful assistant.".to_string(),
            llm: LLMConfigOrSelector::default(),
            llms: HashMap::new(),
            skills: vec![],
            memory: MemoryConfig::default(),
            storage: StorageConfig::default(),
            tools: vec![],
            max_iterations: default_max_iterations(),
            max_context_tokens: default_max_context_tokens(),
            error_recovery: ErrorRecoveryConfig::default(),
            tool_security: ToolSecurityConfig::default(),
            process: ProcessConfig::default(),
            context: HashMap::new(),
            states: None,
            parallel_tools: ParallelToolsConfig::default(),
            streaming: StreamingConfig::default(),
            hitl: None,
            reasoning: ReasoningConfig::default(),
            reflection: ReflectionConfig::default(),
            providers: ProvidersConfig::default(),
            provider_security: ProviderSecurityConfig::default(),
            tool_aliases: ToolAliasesConfig::default(),
            metadata: None,
        }
    }
}

impl AgentSpec {
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(AgentError::InvalidSpec(
                "Agent name cannot be empty".to_string(),
            ));
        }

        if self.system_prompt.is_empty() {
            return Err(AgentError::InvalidSpec(
                "System prompt cannot be empty".to_string(),
            ));
        }

        if self.max_iterations == 0 {
            return Err(AgentError::InvalidSpec(
                "Max iterations must be greater than 0".to_string(),
            ));
        }

        if let Some(ref states) = self.states {
            states.validate()?;
        }

        Ok(())
    }

    pub fn has_multi_llm(&self) -> bool {
        !self.llms.is_empty()
    }

    pub fn has_skills(&self) -> bool {
        !self.skills.is_empty()
    }

    pub fn has_process(&self) -> bool {
        !self.process.input.is_empty() || !self.process.output.is_empty()
    }

    pub fn has_tool_security(&self) -> bool {
        self.tool_security.enabled
    }

    pub fn has_states(&self) -> bool {
        self.states.is_some()
    }

    pub fn has_context(&self) -> bool {
        !self.context.is_empty()
    }

    pub fn has_parallel_tools(&self) -> bool {
        self.parallel_tools.enabled
    }

    pub fn has_streaming(&self) -> bool {
        self.streaming.enabled
    }

    pub fn has_hitl(&self) -> bool {
        self.hitl.is_some()
    }

    pub fn has_storage(&self) -> bool {
        !self.storage.is_none()
    }

    pub fn has_providers(&self) -> bool {
        self.providers.yaml.is_some()
    }

    pub fn has_tool_aliases(&self) -> bool {
        !self.tool_aliases.tools.is_empty()
    }

    pub fn has_reasoning(&self) -> bool {
        self.reasoning.is_enabled()
    }

    pub fn has_reflection(&self) -> bool {
        self.reflection.requires_evaluation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_spec_minimal() {
        let yaml = r#"
name: TestAgent
system_prompt: "You are a helpful assistant."
llm:
  provider: openai
  model: gpt-4
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.name, "TestAgent");
        assert_eq!(spec.version, "1.0.0");
        assert_eq!(spec.max_iterations, 10);
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_agent_spec_with_states() {
        let yaml = r#"
name: StatefulAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
states:
  initial: greeting
  states:
    greeting:
      prompt: "Welcome!"
      transitions:
        - to: support
          when: "user needs help"
    support:
      prompt: "How can I help?"
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_states());
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_agent_spec_with_context() {
        let yaml = r#"
name: ContextAgent
system_prompt: "Hello, {{ context.user.name }}!"
llm:
  provider: openai
  model: gpt-4
context:
  user:
    type: runtime
    required: true
  time:
    type: builtin
    source: datetime
    refresh: per_turn
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_context());
        assert_eq!(spec.context.len(), 2);
    }

    #[test]
    fn test_agent_spec_with_tool_security() {
        let yaml = r#"
name: SecureAgent
version: 2.0.0
system_prompt: "You are an advanced AI."
llm:
  provider: openai
  model: gpt-4
max_context_tokens: 8192
error_recovery:
  default:
    max_retries: 5
tool_security:
  enabled: true
  default_timeout_ms: 10000
  tools:
    http:
      rate_limit: 10
      blocked_domains:
        - evil.com
process:
  input:
    - type: normalize
      config:
        trim: true
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.name, "SecureAgent");
        assert_eq!(spec.max_context_tokens, 8192);
        assert_eq!(spec.error_recovery.default.max_retries, 5);
        assert!(spec.tool_security.enabled);
        assert!(spec.has_tool_security());
        assert!(!spec.process.input.is_empty());
        assert!(spec.has_process());
    }

    #[test]
    fn test_agent_spec_with_multi_llm() {
        let yaml = r#"
name: MultiLLMAgent
system_prompt: "You are helpful."
llms:
  default:
    provider: openai
    model: gpt-4.1-nano
  router:
    provider: openai
    model: gpt-4.1-nano
llm:
  default: default
  router: router
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_multi_llm());
        assert_eq!(spec.llms.len(), 2);
        assert!(spec.llms.contains_key("default"));
        assert!(spec.llms.contains_key("router"));
    }

    #[test]
    fn test_agent_spec_with_skills() {
        let yaml = r#"
name: SkillAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
skills:
  - weather_clothes
  - file: ./custom.yaml
  - id: inline_skill
    description: "An inline skill"
    trigger: "When user asks"
    steps:
      - prompt: "Hello"
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_skills());
        assert_eq!(spec.skills.len(), 3);
    }

    #[test]
    fn test_agent_spec_validation_empty_name() {
        let mut spec = AgentSpec::default();
        spec.name = "".to_string();
        assert!(spec.validate().is_err());

        spec.name = "Valid".to_string();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_agent_spec_validation_empty_prompt() {
        let mut spec = AgentSpec::default();
        spec.system_prompt = "".to_string();
        assert!(spec.validate().is_err());

        spec.system_prompt = "Valid prompt".to_string();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_agent_spec_validation_zero_iterations() {
        let mut spec = AgentSpec::default();
        spec.max_iterations = 0;
        assert!(spec.validate().is_err());

        spec.max_iterations = 5;
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_agent_spec_with_parallel_tools() {
        let yaml = r#"
name: ParallelAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
parallel_tools:
  enabled: true
  max_parallel: 10
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_parallel_tools());
        assert_eq!(spec.parallel_tools.max_parallel, 10);
    }

    #[test]
    fn test_agent_spec_with_streaming() {
        let yaml = r#"
name: StreamingAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
streaming:
  enabled: true
  buffer_size: 64
  include_tool_events: true
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_streaming());
        assert_eq!(spec.streaming.buffer_size, 64);
    }

    #[test]
    fn test_agent_spec_defaults() {
        let spec = AgentSpec::default();
        assert!(spec.parallel_tools.enabled);
        assert_eq!(spec.parallel_tools.max_parallel, 5);
        assert!(spec.streaming.enabled);
        assert!(!spec.has_hitl());
    }

    #[test]
    fn test_agent_spec_with_hitl() {
        let yaml = r#"
name: HITLAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
hitl:
  default_timeout_seconds: 600
  on_timeout: reject
  tools:
    send_payment:
      require_approval: true
      approval_context:
        - amount
        - recipient
      approval_message: "Approve payment?"
  conditions:
    - name: high_value
      when: "amount > 1000"
      require_approval: true
  states:
    escalation:
      on_enter: require_approval
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_hitl());
        let hitl = spec.hitl.as_ref().unwrap();
        assert_eq!(hitl.default_timeout_seconds, 600);
        assert_eq!(hitl.tools.len(), 1);
        assert_eq!(hitl.conditions.len(), 1);
        assert_eq!(hitl.states.len(), 1);
    }

    #[test]
    fn test_agent_spec_with_storage_file() {
        let yaml = r#"
name: PersistentAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
storage:
  type: file
  path: "./data/sessions"
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_storage());
        assert!(spec.storage.is_file());
        assert_eq!(spec.storage.get_path(), Some("./data/sessions"));
    }

    #[test]
    fn test_agent_spec_with_storage_sqlite() {
        let yaml = r#"
name: PersistentAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
storage:
  type: sqlite
  path: "./data/sessions.db"
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_storage());
        assert!(spec.storage.is_sqlite());
    }

    #[test]
    fn test_agent_spec_with_storage_redis() {
        let yaml = r#"
name: PersistentAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
storage:
  type: redis
  url: "redis://localhost:6379"
  prefix: "myagent:"
  ttl_seconds: 86400
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_storage());
        assert!(spec.storage.is_redis());
        assert_eq!(spec.storage.get_url(), Some("redis://localhost:6379"));
        assert_eq!(spec.storage.get_prefix(), "myagent:");
        assert_eq!(spec.storage.get_ttl(), Some(86400));
    }

    #[test]
    fn test_agent_spec_no_storage_by_default() {
        let spec = AgentSpec::default();
        assert!(!spec.has_storage());
        assert!(spec.storage.is_none());
    }

    #[test]
    fn test_agent_spec_with_providers() {
        let yaml = r#"
name: ProviderAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
providers:
  builtin:
    enabled: true
  yaml:
    enabled: true
    tools:
      - id: custom_search
        name: Custom Search
        description: Search custom API
        implementation:
          type: http
          url: https://api.example.com/search
          method: GET
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_providers());
        assert!(spec.providers.builtin.enabled);
        assert!(spec.providers.yaml.is_some());
    }

    #[test]
    fn test_agent_spec_with_tool_aliases() {
        let yaml = r#"
name: AliasAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
tool_aliases:
  calculator:
    names:
      ko: 계산기
      ja: 計算機
    descriptions:
      ko: 수학 계산을 합니다
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_tool_aliases());
        let calc_aliases = spec.tool_aliases.tools.get("calculator").unwrap();
        assert_eq!(calc_aliases.get_name("ko"), Some("계산기"));
    }

    #[test]
    fn test_agent_spec_with_reasoning() {
        let yaml = r#"
    name: ReasoningAgent
    system_prompt: "You are helpful."
    llm:
      provider: openai
      model: gpt-4
    reasoning:
      mode: cot
      judge_llm: router
      output: tagged
      max_iterations: 8
    "#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_reasoning());
        assert_eq!(spec.reasoning.max_iterations, 8);
    }

    #[test]
    fn test_agent_spec_with_reflection() {
        let yaml = r#"
    name: ReflectionAgent
    system_prompt: "You are helpful."
    llm:
      provider: openai
      model: gpt-4
    reflection:
      enabled: auto
      evaluator_llm: router
      max_retries: 3
      pass_threshold: 0.8
      criteria:
        - "Response addresses the question"
        - "Response is accurate"
    "#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_reflection());
        assert_eq!(spec.reflection.max_retries, 3);
        assert_eq!(spec.reflection.criteria.len(), 2);
    }

    #[test]
    fn test_agent_spec_with_plan_and_execute() {
        let yaml = r#"
    name: PlanningAgent
    system_prompt: "You are helpful."
    llm:
      provider: openai
      model: gpt-4
    reasoning:
      mode: plan_and_execute
      planning:
        planner_llm: router
        max_steps: 15
        available:
          tools: all
          skills:
            - analyze
            - summarize
        reflection:
          enabled: true
          on_step_failure: replan
          max_replans: 3
    "#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_reasoning());
        let planning = spec.reasoning.planning.as_ref().unwrap();
        assert_eq!(planning.max_steps, 15);
        assert!(planning.reflection.enabled);
    }

    #[test]
    fn test_agent_spec_reasoning_defaults() {
        let spec = AgentSpec::default();
        assert!(!spec.has_reasoning());
        assert!(!spec.has_reflection());
    }

    #[test]
    fn test_agent_spec_state_level_reasoning_override() {
        let yaml = r#"
    name: StateReasoningAgent
    system_prompt: "You are helpful."
    llm:
      provider: openai
      model: gpt-4
    reasoning:
      mode: auto
    states:
      initial: greeting
      states:
        greeting:
          prompt: "Welcome!"
          reasoning:
            mode: none
        complex_analysis:
          prompt: "Analyze this"
          reasoning:
            mode: cot
            output: tagged
          reflection:
            enabled: true
            criteria:
              - "Analysis is thorough"
    "#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert!(spec.has_reasoning());
        assert!(spec.has_states());

        let states = spec.states.as_ref().unwrap();
        let greeting = states.states.get("greeting").unwrap();
        assert!(greeting.reasoning.is_some());
        let greeting_reasoning = greeting.reasoning.as_ref().unwrap();
        assert_eq!(
            greeting_reasoning.mode,
            ai_agents_reasoning::ReasoningMode::None
        );

        let analysis = states.states.get("complex_analysis").unwrap();
        assert!(analysis.reasoning.is_some());
        assert!(analysis.reflection.is_some());
        let analysis_reasoning = analysis.reasoning.as_ref().unwrap();
        assert_eq!(
            analysis_reasoning.mode,
            ai_agents_reasoning::ReasoningMode::CoT
        );
    }

    #[test]
    fn test_agent_spec_skill_level_reasoning_override() {
        use ai_agents_skills::SkillDefinition;

        let skill_yaml = r#"
id: complex_analysis
description: "Analyze data"
trigger: "When user asks for analysis"
reasoning:
  mode: cot
reflection:
  enabled: true
  criteria:
    - "Analysis covers all aspects"
steps:
  - prompt: "Analyze the input"
"#;
        let skill_def: SkillDefinition = serde_yaml::from_str(skill_yaml).unwrap();
        assert!(skill_def.reasoning.is_some());
        assert!(skill_def.reflection.is_some());
        let reasoning = skill_def.reasoning.as_ref().unwrap();
        assert_eq!(reasoning.mode, ai_agents_reasoning::ReasoningMode::CoT);
        let reflection = skill_def.reflection.as_ref().unwrap();
        assert!(reflection.is_enabled());

        let simple_yaml = r#"
id: simple_lookup
description: "Look up simple facts"
trigger: "When user asks for facts"
reasoning:
  mode: none
reflection:
  enabled: false
steps:
  - prompt: "Look up the fact"
"#;
        let simple_def: SkillDefinition = serde_yaml::from_str(simple_yaml).unwrap();
        assert!(simple_def.reasoning.is_some());
        let simple_reasoning = simple_def.reasoning.as_ref().unwrap();
        assert_eq!(
            simple_reasoning.mode,
            ai_agents_reasoning::ReasoningMode::None
        );
    }
}
