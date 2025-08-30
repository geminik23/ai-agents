use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

    #[serde(default)]
    pub tools: Vec<String>,

    #[serde(default)]
    pub transitions: Vec<Transition>,

    #[serde(default)]
    pub max_turns: Option<u32>,

    #[serde(default)]
    pub timeout_to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub to: String,
    pub when: String,
    #[serde(default)]
    pub auto: bool,
    #[serde(default)]
    pub priority: u8,
}

impl StateConfig {
    pub fn validate(&self) -> crate::Result<()> {
        if self.initial.is_empty() {
            return Err(crate::AgentError::InvalidSpec(
                "State machine initial state cannot be empty".into(),
            ));
        }
        if !self.states.contains_key(&self.initial) {
            return Err(crate::AgentError::InvalidSpec(format!(
                "Initial state '{}' not found in states",
                self.initial
            )));
        }
        for (name, def) in &self.states {
            for transition in &def.transitions {
                if !self.states.contains_key(&transition.to) {
                    return Err(crate::AgentError::InvalidSpec(format!(
                        "State '{}' has transition to unknown state '{}'",
                        name, transition.to
                    )));
                }
            }
            if let Some(ref timeout_state) = def.timeout_to {
                if !self.states.contains_key(timeout_state) {
                    return Err(crate::AgentError::InvalidSpec(format!(
                        "State '{}' has timeout_to unknown state '{}'",
                        name, timeout_state
                    )));
                }
            }
        }
        Ok(())
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
                    auto: true,
                    priority: 0,
                }],
                ..Default::default()
            },
        );
        let config = StateConfig {
            initial: "start".into(),
            states,
        };
        assert!(config.validate().is_err());
    }
}
