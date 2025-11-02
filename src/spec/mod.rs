//! Agent specification types

mod llm;
mod memory;
mod tool;

pub use llm::{LLMConfig, LLMSelector};
pub use memory::MemoryConfig;
pub use tool::ToolConfig;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::agent::{ParallelToolsConfig, StreamingConfig};
use crate::context::ContextSource;
use crate::process::ProcessConfig;
use crate::recovery::ErrorRecoveryConfig;
use crate::skill::SkillRef;
use crate::state::StateConfig;
use crate::tool_security::ToolSecurityConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,

    #[serde(default = "default_version")]
    pub version: String,

    #[serde(skip_serializing_if = "Option::is_none")]
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
            metadata: None,
        }
    }
}

impl AgentSpec {
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.name.is_empty() {
            return Err(crate::error::AgentError::InvalidSpec(
                "Agent name cannot be empty".to_string(),
            ));
        }

        if self.system_prompt.is_empty() {
            return Err(crate::error::AgentError::InvalidSpec(
                "System prompt cannot be empty".to_string(),
            ));
        }

        if self.max_iterations == 0 {
            return Err(crate::error::AgentError::InvalidSpec(
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
    }
}
