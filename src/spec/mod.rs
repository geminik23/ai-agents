//! Agent specification types

mod llm;
mod memory;
mod tool;

pub use llm::{LLMConfig, LLMSelector};
pub use memory::MemoryConfig;
pub use tool::ToolConfig;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::skill::SkillRef;

/// Complete specification for an AI agent
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
            metadata: None,
        }
    }
}

impl AgentSpec {
    pub fn validate(&self) -> crate::error::Result<()> {
        // TODO: Improve validation later
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

        Ok(())
    }

    pub fn has_multi_llm(&self) -> bool {
        !self.llms.is_empty()
    }

    pub fn has_skills(&self) -> bool {
        !self.skills.is_empty()
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
    fn test_agent_spec_full() {
        let yaml = r#"
name: FullAgent
version: 2.0.0
description: "A full-featured agent"
system_prompt: "You are an advanced AI."
llm:
  provider: openai
  model: gpt-4
  temperature: 0.8
memory:
  type: in-memory
  max_messages: 50
tools:
  - name: calculator
  - name: echo
max_iterations: 20
"#;
        let spec: AgentSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.name, "FullAgent");
        assert_eq!(spec.version, "2.0.0");
        assert_eq!(spec.description, Some("A full-featured agent".to_string()));
        assert_eq!(spec.max_iterations, 20);
        assert_eq!(spec.tools.len(), 2);
        assert!(spec.validate().is_ok());
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
        let mut spec = AgentSpec {
            name: "".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            system_prompt: "test".to_string(),
            llm: LLMConfigOrSelector::default(),
            llms: HashMap::new(),
            skills: vec![],
            memory: MemoryConfig::default(),
            tools: vec![],
            max_iterations: 10,
            metadata: None,
        };

        assert!(spec.validate().is_err());
        spec.name = "Valid".to_string();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_agent_spec_validation_empty_prompt() {
        let mut spec = AgentSpec {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            system_prompt: "".to_string(),
            llm: LLMConfigOrSelector::default(),
            llms: HashMap::new(),
            skills: vec![],
            memory: MemoryConfig::default(),
            tools: vec![],
            max_iterations: 10,
            metadata: None,
        };

        assert!(spec.validate().is_err());
        spec.system_prompt = "Valid prompt".to_string();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_agent_spec_validation_zero_iterations() {
        let mut spec = AgentSpec {
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            system_prompt: "test".to_string(),
            llm: LLMConfigOrSelector::default(),
            llms: HashMap::new(),
            skills: vec![],
            memory: MemoryConfig::default(),
            tools: vec![],
            max_iterations: 0,
            metadata: None,
        };

        assert!(spec.validate().is_err());
        spec.max_iterations = 5;
        assert!(spec.validate().is_ok());
    }
}
