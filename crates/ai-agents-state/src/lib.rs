//! State machine for AI Agents framework

mod config;
mod evaluator;
mod machine;

pub use ai_agents_core::{StateMachineSnapshot, StateTransitionEvent};
pub use config::{
    CompareOp, ContextExtractor, ContextMatcher, GuardConditions, PromptMode, StateAction,
    StateConfig, StateDefinition, StateMatcher, TimeMatcher, ToolCondition, ToolRef, Transition,
    TransitionGuard,
};
pub use evaluator::{
    GuardOnlyEvaluator, LLMTransitionEvaluator, TransitionContext, TransitionEvaluator,
    evaluate_conditions, evaluate_expression, evaluate_guard, get_context_value,
};
pub use machine::StateMachine;
