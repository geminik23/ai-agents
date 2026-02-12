//! Intent disambiguation support for AI Agents framework
//!
//! This crate provides LLM-based mechanisms to detect ambiguous user inputs
//! and ask clarifying questions before proceeding with potentially incorrect actions.
//!
//! # Key Features
//!
//! - **Language-agnostic**: Uses LLM for all detection (no regex), supporting all languages
//! - **Configurable thresholds**: Different disambiguation thresholds per state or skill
//! - **Multiple clarification styles**: Options, open-ended, yes/no, or auto-determined
//! - **Skip conditions**: Skip disambiguation for social messages, short inputs, etc.
//!
//! # Example
//!
//! ```yaml
//! disambiguation:
//!   enabled: true
//!   detection:
//!     llm: router
//!     threshold: 0.7
//!     aspects:
//!       - missing_target
//!       - vague_references
//!   clarification:
//!     style: auto
//!     max_attempts: 2
//! ```

mod clarifier;
mod config;
mod detector;
mod manager;
mod types;
mod util;

pub use clarifier::{ClarificationGenerator, ClarificationParseResult};
pub use config::{
    AmbiguityAspect, CacheConfig, ClarificationConfig, ClarificationStyle, ContextConfig,
    DetectionConfig, DisambiguationConfig, MaxAttemptsAction, SkillDisambiguationOverride,
    SkipCondition, StateDisambiguationOverride,
};
pub use detector::AmbiguityDetector;
pub use manager::DisambiguationManager;
pub use types::{
    AmbiguityDetectionResult, AmbiguityType, ClarificationOption, ClarificationQuestion,
    DisambiguationContext, DisambiguationResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = DisambiguationConfig::default();
        assert!(!config.is_enabled());
        assert_eq!(config.detection.threshold, 0.7);
        assert_eq!(config.clarification.max_attempts, 2);
        assert_eq!(config.clarification.style, ClarificationStyle::Auto);
    }

    #[test]
    fn test_minimal_yaml_config() {
        let yaml = r#"
enabled: true
"#;
        let config: DisambiguationConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_enabled());
        assert_eq!(config.detection.llm, "router");
        assert_eq!(config.detection.threshold, 0.7);
    }

    #[test]
    fn test_full_yaml_config() {
        let yaml = r#"
enabled: true
detection:
  llm: fast
  threshold: 0.8
  aspects:
    - missing_target
    - missing_action
    - vague_references
clarification:
  style: options
  max_options: 4
  include_other_option: true
  max_attempts: 3
  on_max_attempts: escalate
context:
  recent_messages: 10
  include_state: true
  include_available_tools: true
skip_when:
  - type: social
  - type: short_input
    max_chars: 10
  - type: in_state
    states:
      - greeting
      - farewell
cache:
  enabled: true
  similarity_threshold: 0.9
  ttl_seconds: 3600
"#;
        let config: DisambiguationConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_enabled());
        assert_eq!(config.detection.llm, "fast");
        assert_eq!(config.detection.threshold, 0.8);
        assert_eq!(config.detection.aspects.len(), 3);
        assert_eq!(config.clarification.style, ClarificationStyle::Options);
        assert_eq!(config.clarification.max_options, 4);
        assert_eq!(config.clarification.max_attempts, 3);
        assert_eq!(
            config.clarification.on_max_attempts,
            MaxAttemptsAction::Escalate
        );
        assert_eq!(config.context.recent_messages, 10);
        assert_eq!(config.skip_when.len(), 3);
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
  - currency
"#;
        let override_config: StateDisambiguationOverride = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(override_config.threshold, Some(0.95));
        assert!(override_config.require_confirmation);
        assert_eq!(override_config.required_clarity.len(), 3);
        assert!(!override_config.is_empty());
    }

    #[test]
    fn test_skill_override() {
        let yaml = r#"
enabled: true
threshold: 0.9
required_clarity:
  - from_account
  - to_account
  - amount
clarification_templates:
  missing_target: "누구에게 보내시겠습니까?"
  missing_parameters: "얼마를 보내시겠습니까?"
"#;
        let override_config: SkillDisambiguationOverride = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(override_config.enabled, Some(true));
        assert_eq!(override_config.threshold, Some(0.9));
        assert_eq!(override_config.required_clarity.len(), 3);
        assert_eq!(override_config.clarification_templates.len(), 2);
        assert!(!override_config.is_empty());
    }

    #[test]
    fn test_ambiguity_detection_result() {
        let clear = AmbiguityDetectionResult::clear();
        assert!(!clear.is_ambiguous);
        assert_eq!(clear.confidence, 1.0);

        let ambiguous = AmbiguityDetectionResult::ambiguous(
            0.3,
            AmbiguityType::VagueReference,
            "The word '그거' is a vague reference".to_string(),
            vec!["그거".to_string()],
        )
        .with_language("ko");

        assert!(ambiguous.is_ambiguous);
        assert_eq!(ambiguous.confidence, 0.3);
        assert_eq!(ambiguous.detected_language, Some("ko".to_string()));
    }

    #[test]
    fn test_clarification_question() {
        let open = ClarificationQuestion::open("What would you like to do?");
        assert_eq!(open.style, ClarificationStyle::Open);
        assert!(!open.has_options());

        let options = vec![
            ClarificationOption::new("1", "Option A"),
            ClarificationOption::new("2", "Option B").with_description("More details"),
        ];
        let with_options = ClarificationQuestion::with_options("Choose one:", options);
        assert_eq!(with_options.style, ClarificationStyle::Options);
        assert!(with_options.has_options());

        let yes_no = ClarificationQuestion::yes_no("Are you sure?");
        assert_eq!(yes_no.style, ClarificationStyle::YesNo);
        assert!(yes_no.has_options());
        assert_eq!(yes_no.options.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_disambiguation_result_variants() {
        let clear = DisambiguationResult::Clear;
        assert!(clear.is_clear());
        assert!(clear.is_resolved());

        let needs = DisambiguationResult::NeedsClarification {
            question: ClarificationQuestion::open("What?"),
            detection: AmbiguityDetectionResult::clear(),
        };
        assert!(needs.needs_clarification());
        assert!(!needs.is_resolved());

        let clarified = DisambiguationResult::Clarified {
            original_input: "Send it".to_string(),
            enriched_input: "Send the report to John".to_string(),
            resolved: std::collections::HashMap::new(),
        };
        assert!(clarified.is_resolved());

        let best_guess = DisambiguationResult::ProceedWithBestGuess {
            enriched_input: "Send it (best guess)".to_string(),
        };
        assert!(best_guess.is_resolved());
    }

    #[test]
    fn test_disambiguation_context() {
        let mut ctx = DisambiguationContext::new()
            .with_state("checkout")
            .with_tools(vec!["search".to_string(), "pay".to_string()])
            .with_skills(vec!["greet".to_string()])
            .with_recent_messages(vec!["Hello".to_string(), "I want to buy".to_string()]);

        assert_eq!(ctx.current_state, Some("checkout".to_string()));
        assert_eq!(ctx.available_tools.len(), 2);
        assert_eq!(ctx.available_skills.len(), 1);
        assert_eq!(ctx.recent_messages.len(), 2);

        ctx.increment_attempts();
        ctx.add_previous_question("What would you like?".to_string());

        assert_eq!(ctx.clarification_attempts, 1);
        assert_eq!(ctx.previous_questions.len(), 1);
    }

    #[test]
    fn test_ambiguity_aspects() {
        let aspects = vec![
            AmbiguityAspect::MissingTarget,
            AmbiguityAspect::MissingAction,
            AmbiguityAspect::VagueReferences,
        ];

        for aspect in &aspects {
            assert!(!aspect.description().is_empty());
        }

        assert!(
            AmbiguityAspect::VagueReferences
                .description()
                .contains("그거")
        );
        assert!(
            AmbiguityAspect::VagueReferences
                .description()
                .contains("あれ")
        );
    }
}
