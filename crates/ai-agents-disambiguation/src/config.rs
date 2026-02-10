//! Configuration types for intent disambiguation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main disambiguation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisambiguationConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    #[serde(default)]
    pub detection: DetectionConfig,

    #[serde(default)]
    pub clarification: ClarificationConfig,

    #[serde(default)]
    pub context: ContextConfig,

    #[serde(default)]
    pub skip_when: Vec<SkipCondition>,

    #[serde(default)]
    pub cache: CacheConfig,
}

fn default_enabled() -> bool {
    false
}

impl Default for DisambiguationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            detection: DetectionConfig::default(),
            clarification: ClarificationConfig::default(),
            context: ContextConfig::default(),
            skip_when: Vec::new(),
            cache: CacheConfig::default(),
        }
    }
}

impl DisambiguationConfig {
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Detection configuration - how to identify ambiguous inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    #[serde(default = "default_llm")]
    pub llm: String,

    #[serde(default = "default_threshold")]
    pub threshold: f32,

    #[serde(default = "default_aspects")]
    pub aspects: Vec<AmbiguityAspect>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

fn default_llm() -> String {
    "router".to_string()
}

fn default_threshold() -> f32 {
    0.7
}

fn default_aspects() -> Vec<AmbiguityAspect> {
    vec![
        AmbiguityAspect::MissingTarget,
        AmbiguityAspect::MissingAction,
        AmbiguityAspect::MissingParameters,
        AmbiguityAspect::VagueReferences,
    ]
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            llm: default_llm(),
            threshold: default_threshold(),
            aspects: default_aspects(),
            prompt: None,
        }
    }
}

/// Aspects of ambiguity to detect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AmbiguityAspect {
    MissingTarget,
    MissingAction,
    MissingParameters,
    MultipleIntents,
    VagueReferences,
    ImplicitContext,
}

impl AmbiguityAspect {
    pub fn description(&self) -> &'static str {
        match self {
            Self::MissingTarget => "WHO or WHAT is the action for",
            Self::MissingAction => "WHAT action to perform",
            Self::MissingParameters => "Required information missing",
            Self::MultipleIntents => "Could mean different things",
            Self::VagueReferences => "Vague references like 'it', 'that', '그거', 'あれ'",
            Self::ImplicitContext => "Assumes shared knowledge we don't have",
        }
    }
}

/// Clarification configuration - how to ask for clarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarificationConfig {
    #[serde(default = "default_style")]
    pub style: ClarificationStyle,

    #[serde(default = "default_max_options")]
    pub max_options: usize,

    #[serde(default = "default_true")]
    pub include_other_option: bool,

    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,

    #[serde(default = "default_on_max_attempts")]
    pub on_max_attempts: MaxAttemptsAction,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
}

fn default_style() -> ClarificationStyle {
    ClarificationStyle::Auto
}

fn default_max_options() -> usize {
    4
}

fn default_true() -> bool {
    true
}

fn default_max_attempts() -> u32 {
    2
}

fn default_on_max_attempts() -> MaxAttemptsAction {
    MaxAttemptsAction::ProceedWithBestGuess
}

impl Default for ClarificationConfig {
    fn default() -> Self {
        Self {
            style: default_style(),
            max_options: default_max_options(),
            include_other_option: default_true(),
            max_attempts: default_max_attempts(),
            on_max_attempts: default_on_max_attempts(),
            llm: None,
        }
    }
}

/// Style of clarification questions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ClarificationStyle {
    #[default]
    Auto,
    Options,
    Open,
    YesNo,
    Hybrid,
}

/// Action to take when max clarification attempts reached
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MaxAttemptsAction {
    #[default]
    ProceedWithBestGuess,
    ApologizeAndStop,
    Escalate,
}

/// Context configuration - what information helps disambiguation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    #[serde(default = "default_recent_messages")]
    pub recent_messages: usize,

    #[serde(default = "default_true")]
    pub include_state: bool,

    #[serde(default = "default_true")]
    pub include_available_tools: bool,

    #[serde(default = "default_true")]
    pub include_available_skills: bool,

    #[serde(default = "default_true")]
    pub include_user_context: bool,
}

fn default_recent_messages() -> usize {
    5
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            recent_messages: default_recent_messages(),
            include_state: true,
            include_available_tools: true,
            include_available_skills: true,
            include_user_context: true,
        }
    }
}

/// Conditions when to skip disambiguation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SkipCondition {
    Social {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        examples_hint: Option<String>,
    },
    CompleteToolCall,
    AnsweringAgentQuestion,
    ShortInput {
        #[serde(default = "default_max_chars")]
        max_chars: usize,
    },
    InState {
        states: Vec<String>,
    },
    Custom {
        condition: String,
    },
}

fn default_max_chars() -> usize {
    10
}

/// Cache configuration for disambiguation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    #[serde(default = "default_ttl_seconds")]
    pub ttl_seconds: u64,
}

fn default_similarity_threshold() -> f32 {
    0.9
}

fn default_ttl_seconds() -> u64 {
    3600
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: default_similarity_threshold(),
            ttl_seconds: default_ttl_seconds(),
        }
    }
}

/// State-level disambiguation override
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateDisambiguationOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,

    #[serde(default)]
    pub require_confirmation: bool,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_clarity: Vec<String>,
}

impl StateDisambiguationOverride {
    pub fn is_empty(&self) -> bool {
        self.enabled.is_none()
            && self.threshold.is_none()
            && !self.require_confirmation
            && self.required_clarity.is_empty()
    }
}

/// Skill-level disambiguation override
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillDisambiguationOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_clarity: Vec<String>,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub clarification_templates: HashMap<String, String>,
}

impl SkillDisambiguationOverride {
    pub fn is_empty(&self) -> bool {
        self.enabled.is_none()
            && self.threshold.is_none()
            && self.required_clarity.is_empty()
            && self.clarification_templates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DisambiguationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.detection.threshold, 0.7);
        assert_eq!(config.clarification.max_attempts, 2);
    }

    #[test]
    fn test_parse_minimal() {
        let yaml = r#"
enabled: true
"#;
        let config: DisambiguationConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.detection.llm, "router");
    }

    #[test]
    fn test_parse_full_config() {
        let yaml = r#"
enabled: true
detection:
  llm: fast
  threshold: 0.8
  aspects:
    - missing_target
    - vague_references
clarification:
  style: options
  max_options: 3
  max_attempts: 3
  on_max_attempts: escalate
context:
  recent_messages: 10
  include_state: true
skip_when:
  - type: social
  - type: short_input
    max_chars: 5
cache:
  enabled: true
  ttl_seconds: 7200
"#;
        let config: DisambiguationConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.enabled);
        assert_eq!(config.detection.llm, "fast");
        assert_eq!(config.detection.threshold, 0.8);
        assert_eq!(config.detection.aspects.len(), 2);
        assert_eq!(config.clarification.style, ClarificationStyle::Options);
        assert_eq!(config.clarification.max_attempts, 3);
        assert_eq!(
            config.clarification.on_max_attempts,
            MaxAttemptsAction::Escalate
        );
        assert_eq!(config.context.recent_messages, 10);
        assert_eq!(config.skip_when.len(), 2);
        assert!(config.cache.enabled);
    }

    #[test]
    fn test_state_override() {
        let yaml = r#"
threshold: 0.95
require_confirmation: true
required_clarity:
  - recipient
  - amount
"#;
        let override_config: StateDisambiguationOverride = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(override_config.threshold, Some(0.95));
        assert!(override_config.require_confirmation);
        assert_eq!(override_config.required_clarity.len(), 2);
    }

    #[test]
    fn test_skill_override() {
        let yaml = r#"
enabled: true
threshold: 0.9
required_clarity:
  - from_account
  - to_account
clarification_templates:
  missing_recipient: "Who would you like to transfer to?"
  missing_amount: "How much would you like to transfer?"
"#;
        let override_config: SkillDisambiguationOverride = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(override_config.enabled, Some(true));
        assert_eq!(override_config.required_clarity.len(), 2);
        assert_eq!(override_config.clarification_templates.len(), 2);
    }

    #[test]
    fn test_ambiguity_aspects() {
        assert_eq!(
            AmbiguityAspect::MissingTarget.description(),
            "WHO or WHAT is the action for"
        );
        assert_eq!(
            AmbiguityAspect::VagueReferences.description(),
            "Vague references like 'it', 'that', '그거', 'あれ'"
        );
    }
}
