//! State machine for AI Agents framework

mod config;
mod evaluator;
mod machine;

pub use ai_agents_core::{StateMachineSnapshot, StateTransitionEvent};
pub use config::{
    AggregationConfig, AggregationStrategy, ChatManagerConfig, ChatParticipant, ChatStyle,
    CompareOp, ConcurrentAgentRef, ConcurrentStateConfig, ContextExtractor, ContextMatcher,
    DebateStyleConfig, DelegateContextMode, GroupChatStateConfig, GuardConditions,
    HandoffStateConfig, MakerCheckerConfig, MaxIterationsAction, PartialFailureAction,
    PipelineStageEntry, PipelineStateConfig, PromptMode, StateAction, StateConfig, StateDefinition,
    StateMatcher, TerminationConfig, TerminationMethod, TiebreakerStrategy, TimeMatcher,
    ToolCondition, ToolRef, Transition, TransitionGuard, TurnMethod, VoteConfig, VoteMethod,
};
pub use evaluator::{
    GuardOnlyEvaluator, LLMTransitionEvaluator, TransitionContext, TransitionEvaluator,
    evaluate_conditions, evaluate_expression, evaluate_guard, get_context_value,
};
pub use machine::StateMachine;
