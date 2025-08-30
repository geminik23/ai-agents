use std::sync::Arc;

use async_trait::async_trait;

use super::config::Transition;
use crate::llm::LLMProvider;
use crate::{ChatMessage, Result};

pub struct TransitionContext {
    pub user_message: String,
    pub assistant_response: String,
    pub current_state: String,
}

#[async_trait]
pub trait TransitionEvaluator: Send + Sync {
    async fn select_transition(
        &self,
        transitions: &[Transition],
        context: &TransitionContext,
    ) -> Result<Option<usize>>;
}

pub struct LLMTransitionEvaluator {
    llm: Arc<dyn LLMProvider>,
}

impl LLMTransitionEvaluator {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl TransitionEvaluator for LLMTransitionEvaluator {
    async fn select_transition(
        &self,
        transitions: &[Transition],
        context: &TransitionContext,
    ) -> Result<Option<usize>> {
        if transitions.is_empty() {
            return Ok(None);
        }

        let conditions: Vec<String> = transitions
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{}. {}", i + 1, t.when))
            .collect();

        let prompt = format!(
            r#"Based on the conversation, which condition is met?

Current state: {}
User message: {}
Assistant response: {}

Conditions:
{}
0. None of the above

Reply with ONLY the number (0-{})."#,
            context.current_state,
            context.user_message,
            context.assistant_response,
            conditions.join("\n"),
            transitions.len()
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = self.llm.complete(&messages, None).await?;

        let choice: usize = response.content.trim().parse().unwrap_or(0);

        if choice == 0 || choice > transitions.len() {
            Ok(None)
        } else {
            Ok(Some(choice - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::mock::MockLLMProvider;
    use crate::llm::{FinishReason, LLMResponse};

    #[tokio::test]
    async fn test_select_transition_none() {
        let mut mock = MockLLMProvider::new("evaluator_test");
        mock.add_response(LLMResponse::new("0", FinishReason::Stop));
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![Transition {
            to: "next".into(),
            when: "user says goodbye".into(),
            auto: true,
            priority: 0,
        }];

        let context = TransitionContext {
            user_message: "hello".into(),
            assistant_response: "hi there".into(),
            current_state: "greeting".into(),
        };

        let result = evaluator.select_transition(&transitions, &context).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_select_transition_match() {
        let mut mock = MockLLMProvider::new("evaluator_test");
        mock.add_response(LLMResponse::new("1", FinishReason::Stop));
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let transitions = vec![
            Transition {
                to: "support".into(),
                when: "user needs help".into(),
                auto: true,
                priority: 10,
            },
            Transition {
                to: "sales".into(),
                when: "user wants to buy".into(),
                auto: true,
                priority: 5,
            },
        ];

        let context = TransitionContext {
            user_message: "I need help".into(),
            assistant_response: "Sure!".into(),
            current_state: "greeting".into(),
        };

        let result = evaluator.select_transition(&transitions, &context).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(0));
    }

    #[tokio::test]
    async fn test_empty_transitions() {
        let mock = MockLLMProvider::new("evaluator_test");
        let evaluator = LLMTransitionEvaluator::new(Arc::new(mock));

        let context = TransitionContext {
            user_message: "hi".into(),
            assistant_response: "hello".into(),
            current_state: "start".into(),
        };

        let result = evaluator.select_transition(&[], &context).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
