use ai_agents_core::{AgentError, Result};
use ai_agents_reasoning::{ReasoningConfig, ReflectionConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
    #[serde(default)]
    pub global_transitions: Vec<Transition>,
    #[serde(default)]
    pub fallback: Option<String>,
    #[serde(default)]
    pub max_no_transition: Option<u32>,
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
    pub tools: Vec<ToolRef>,

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

    #[serde(default)]
    pub on_exit: Vec<StateAction>,

    #[serde(default)]
    pub extract: Option<ContextExtractor>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection: Option<ReflectionConfig>,
}

fn default_inherit_parent() -> bool {
    true
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
    Exact(Value),
    Compare(CompareOp),
    Exists { exists: bool },
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
    #[serde(default)]
    pub auto: bool,
    #[serde(default)]
    pub priority: u8,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextExtractor {
    pub key: String,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub llm_extract: Option<String>,
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
}

impl StateDefinition {
    pub fn has_sub_states(&self) -> bool {
        self.states.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
    }

    pub fn get_effective_tools<'a>(
        &'a self,
        parent: Option<&'a StateDefinition>,
    ) -> Vec<&'a ToolRef> {
        if !self.inherit_parent || parent.is_none() {
            return self.tools.iter().collect();
        }

        let parent = parent.unwrap();
        let mut tools: Vec<&'a ToolRef> = parent.tools.iter().collect();
        tools.extend(self.tools.iter());
        tools
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
                    auto: true,
                    priority: 0,
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
            tools: vec![ToolRef::Simple("parent_tool".into())],
            skills: vec!["parent_skill".into()],
            ..Default::default()
        };

        let child = StateDefinition {
            tools: vec![ToolRef::Simple("child_tool".into())],
            skills: vec!["child_skill".into()],
            inherit_parent: true,
            ..Default::default()
        };

        let effective_tools = child.get_effective_tools(Some(&parent));
        assert_eq!(effective_tools.len(), 2);

        let effective_skills = child.get_effective_skills(Some(&parent));
        assert_eq!(effective_skills.len(), 2);
    }

    #[test]
    fn test_no_inherit_parent() {
        let parent = StateDefinition {
            tools: vec![ToolRef::Simple("parent_tool".into())],
            ..Default::default()
        };

        let child = StateDefinition {
            tools: vec![ToolRef::Simple("child_tool".into())],
            inherit_parent: false,
            ..Default::default()
        };

        let effective_tools = child.get_effective_tools(Some(&parent));
        assert_eq!(effective_tools.len(), 1);
        assert_eq!(effective_tools[0].id(), "child_tool");
    }
}
