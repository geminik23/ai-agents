//! Result types for intent disambiguation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::{AmbiguityAspect, ClarificationStyle};

/// Type of ambiguity detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AmbiguityType {
    MissingTarget,
    MissingAction,
    MissingParameters,
    MultipleIntents,
    VagueReference,
    ImplicitContext,
    Unknown,
}

impl From<&AmbiguityAspect> for AmbiguityType {
    fn from(aspect: &AmbiguityAspect) -> Self {
        match aspect {
            AmbiguityAspect::MissingTarget => Self::MissingTarget,
            AmbiguityAspect::MissingAction => Self::MissingAction,
            AmbiguityAspect::MissingParameters => Self::MissingParameters,
            AmbiguityAspect::MultipleIntents => Self::MultipleIntents,
            AmbiguityAspect::VagueReferences => Self::VagueReference,
            AmbiguityAspect::ImplicitContext => Self::ImplicitContext,
        }
    }
}

/// Result of ambiguity detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityDetectionResult {
    pub is_ambiguous: bool,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ambiguity_type: Option<AmbiguityType>,
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub what_is_unclear: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
}

impl AmbiguityDetectionResult {
    pub fn clear() -> Self {
        Self {
            is_ambiguous: false,
            confidence: 1.0,
            ambiguity_type: None,
            reasoning: "Input is clear".to_string(),
            what_is_unclear: Vec::new(),
            detected_language: None,
        }
    }

    pub fn ambiguous(
        confidence: f32,
        ambiguity_type: AmbiguityType,
        reasoning: String,
        what_is_unclear: Vec<String>,
    ) -> Self {
        Self {
            is_ambiguous: true,
            confidence,
            ambiguity_type: Some(ambiguity_type),
            reasoning,
            what_is_unclear,
            detected_language: None,
        }
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.detected_language = Some(language.into());
        self
    }
}

impl Default for AmbiguityDetectionResult {
    fn default() -> Self {
        Self::clear()
    }
}

/// An option for clarification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarificationOption {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ClarificationOption {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            description: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Generated clarification question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarificationQuestion {
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<ClarificationOption>>,
    pub style: ClarificationStyle,
    #[serde(default)]
    pub clarifying: Vec<String>,
}

impl ClarificationQuestion {
    pub fn open(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            options: None,
            style: ClarificationStyle::Open,
            clarifying: Vec::new(),
        }
    }

    pub fn with_options(question: impl Into<String>, options: Vec<ClarificationOption>) -> Self {
        Self {
            question: question.into(),
            options: Some(options),
            style: ClarificationStyle::Options,
            clarifying: Vec::new(),
        }
    }

    pub fn yes_no(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            options: Some(vec![
                ClarificationOption::new("yes", "Yes"),
                ClarificationOption::new("no", "No"),
            ]),
            style: ClarificationStyle::YesNo,
            clarifying: Vec::new(),
        }
    }

    pub fn with_clarifying(mut self, clarifying: Vec<String>) -> Self {
        self.clarifying = clarifying;
        self
    }

    pub fn has_options(&self) -> bool {
        self.options.as_ref().is_some_and(|o| !o.is_empty())
    }
}

/// Disambiguation processing result
#[derive(Debug, Clone)]
pub enum DisambiguationResult {
    /// Input is clear, proceed normally
    Clear,

    /// Need to ask user for clarification
    NeedsClarification {
        question: ClarificationQuestion,
        detection: AmbiguityDetectionResult,
    },

    /// User provided clarification, input has been enriched
    Clarified {
        original_input: String,
        enriched_input: String,
        resolved: HashMap<String, serde_json::Value>,
    },

    /// Max attempts reached, proceeding with best interpretation
    ProceedWithBestGuess { enriched_input: String },

    /// Unable to disambiguate, giving up
    GiveUp { reason: String },

    /// Escalating to human (HITL)
    Escalate { reason: String },
}

impl DisambiguationResult {
    pub fn is_clear(&self) -> bool {
        matches!(self, Self::Clear)
    }

    pub fn needs_clarification(&self) -> bool {
        matches!(self, Self::NeedsClarification { .. })
    }

    pub fn is_resolved(&self) -> bool {
        matches!(
            self,
            Self::Clear | Self::Clarified { .. } | Self::ProceedWithBestGuess { .. }
        )
    }

    pub fn get_question(&self) -> Option<&ClarificationQuestion> {
        match self {
            Self::NeedsClarification { question, .. } => Some(question),
            _ => None,
        }
    }
}

/// Context for disambiguation evaluation
#[derive(Debug, Clone, Default)]
pub struct DisambiguationContext {
    pub recent_messages: Vec<String>,
    pub current_state: Option<String>,
    pub available_tools: Vec<String>,
    pub available_skills: Vec<String>,
    pub user_context: HashMap<String, serde_json::Value>,
    pub clarification_attempts: u32,
    pub previous_questions: Vec<String>,
}

impl DisambiguationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_recent_messages(mut self, messages: Vec<String>) -> Self {
        self.recent_messages = messages;
        self
    }

    pub fn with_state(mut self, state: impl Into<String>) -> Self {
        self.current_state = Some(state.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.available_tools = tools;
        self
    }

    pub fn with_skills(mut self, skills: Vec<String>) -> Self {
        self.available_skills = skills;
        self
    }

    pub fn with_user_context(mut self, context: HashMap<String, serde_json::Value>) -> Self {
        self.user_context = context;
        self
    }

    pub fn increment_attempts(&mut self) {
        self.clarification_attempts += 1;
    }

    pub fn add_previous_question(&mut self, question: String) {
        self.previous_questions.push(question);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_result() {
        let result = AmbiguityDetectionResult::clear();
        assert!(!result.is_ambiguous);
        assert_eq!(result.confidence, 1.0);
        assert!(result.what_is_unclear.is_empty());
    }

    #[test]
    fn test_ambiguous_result() {
        let result = AmbiguityDetectionResult::ambiguous(
            0.4,
            AmbiguityType::VagueReference,
            "Vague reference detected".to_string(),
            vec!["그거".to_string()],
        )
        .with_language("ko");

        assert!(result.is_ambiguous);
        assert_eq!(result.confidence, 0.4);
        assert_eq!(result.ambiguity_type, Some(AmbiguityType::VagueReference));
        assert_eq!(result.detected_language, Some("ko".to_string()));
    }

    #[test]
    fn test_clarification_question_open() {
        let question = ClarificationQuestion::open("What would you like to do?");
        assert_eq!(question.style, ClarificationStyle::Open);
        assert!(!question.has_options());
    }

    #[test]
    fn test_clarification_question_with_options() {
        let options = vec![
            ClarificationOption::new("1", "Option A"),
            ClarificationOption::new("2", "Option B"),
        ];
        let question = ClarificationQuestion::with_options("Choose one:", options);
        assert_eq!(question.style, ClarificationStyle::Options);
        assert!(question.has_options());
        assert_eq!(question.options.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_disambiguation_result() {
        let result = DisambiguationResult::Clear;
        assert!(result.is_clear());
        assert!(result.is_resolved());

        let result = DisambiguationResult::NeedsClarification {
            question: ClarificationQuestion::open("What?"),
            detection: AmbiguityDetectionResult::clear(),
        };
        assert!(result.needs_clarification());
        assert!(!result.is_resolved());
        assert!(result.get_question().is_some());
    }

    #[test]
    fn test_disambiguation_context() {
        let mut ctx = DisambiguationContext::new()
            .with_state("checkout")
            .with_tools(vec!["search".to_string()])
            .with_recent_messages(vec!["Hello".to_string()]);

        assert_eq!(ctx.current_state, Some("checkout".to_string()));
        assert_eq!(ctx.clarification_attempts, 0);

        ctx.increment_attempts();
        assert_eq!(ctx.clarification_attempts, 1);

        ctx.add_previous_question("What do you mean?".to_string());
        assert_eq!(ctx.previous_questions.len(), 1);
    }
}
