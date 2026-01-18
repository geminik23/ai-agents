pub mod agent;
pub mod context;
pub mod error;
pub mod hitl;
pub mod hooks;
pub mod llm;
pub mod memory;
pub mod persistence;
pub mod process;
pub mod recovery;
pub mod skill;
pub mod spec;
pub mod state;
pub mod template;
pub mod tool_security;
pub mod tools;

pub use agent::{
    Agent, AgentBuilder, AgentInfo, AgentResponse, ParallelToolsConfig, RuntimeAgent, StreamChunk,
    StreamingConfig,
};
pub use error::{AgentError, Result};
pub use memory::{
    CompactingMemory, CompactingMemoryConfig, CompressResult, CompressionEvent,
    ConversationContext, EvictionReason, FactExtractedEvent, InMemoryStore, LLMSummarizer, Memory,
    MemoryBudgetEvent, MemoryBudgetState, MemoryCompressEvent, MemoryEvictEvent, MemoryTokenBudget,
    NoopSummarizer, OverflowStrategy, Summarizer, TokenAllocation, create_memory,
    create_memory_from_config, create_memory_from_config_with_llm, estimate_message_tokens,
    estimate_tokens,
};
pub use skill::{SkillDefinition, SkillExecutor, SkillLoader, SkillRef, SkillRouter, SkillStep};
pub use spec::{
    AgentSpec, BuiltinProviderConfig, FileStorageConfig, LLMConfig, LLMSelector, MemoryConfig,
    ProviderPolicyConfig, ProviderSecurityConfig, ProvidersConfig, RedisStorageConfig,
    SqliteStorageConfig, StorageConfig, ToolAliasesConfig, ToolConfig,
    ToolPolicyConfig as SpecToolPolicyConfig, YamlProviderConfig, YamlToolConfig,
};
pub use template::TemplateLoader;
#[cfg(feature = "http-tool")]
pub use tools::HttpTool;
pub use tools::{
    CalculatorTool, DateTimeTool, EchoTool, FileTool, JsonTool, MathTool, RandomTool, TemplateTool,
    TextTool, Tool, ToolRegistry, ToolResult, create_builtin_registry,
};

pub use llm::providers::{ProviderType, ProviderType as LLMProviderType, UnifiedLLMProvider};
pub use llm::{ChatMessage, LLMProvider, LLMRegistry, LLMResponse, Role};

pub use process::{ProcessConfig, ProcessData, ProcessProcessor};
pub use recovery::{
    ByRoleFilter, ErrorRecoveryConfig, FilterConfig, KeepRecentFilter, MessageFilter,
    RecoveryManager, SkipPatternFilter,
};
pub use tool_security::{
    SecurityCheckResult, ToolPolicyConfig, ToolSecurityConfig, ToolSecurityEngine,
};

pub use context::{
    BuiltinSource, ContextManager, ContextProvider, ContextSource, RefreshPolicy, TemplateRenderer,
};
#[cfg(feature = "sqlite")]
pub use persistence::SqliteStorage;
pub use persistence::{
    AgentSnapshot, AgentStorage, FileStorage, MemorySnapshot, SessionInfo, SessionMetadata,
    SessionOrderBy, SessionQuery, create_storage,
};
#[cfg(feature = "redis-storage")]
pub use persistence::{RedisSessionMeta, RedisStorage};

pub use state::{
    CompareOp, ContextExtractor, ContextMatcher, GuardConditions, GuardOnlyEvaluator,
    LLMTransitionEvaluator, PromptMode, StateAction, StateConfig, StateDefinition, StateMachine,
    StateMachineSnapshot, StateMatcher, StateTransitionEvent, TimeMatcher, ToolCondition, ToolRef,
    Transition, TransitionContext, TransitionEvaluator, TransitionGuard,
};
pub use tools::{
    ConditionEvaluator, EvaluationContext, LLMGetter, SimpleLLMGetter, ToolCallRecord,
};

pub use hooks::{AgentHooks, CompositeHooks, HookTimer, LoggingHooks, NoopHooks};

pub use hitl::{
    ApprovalCondition, ApprovalHandler, ApprovalMessage, ApprovalRequest, ApprovalResult,
    ApprovalTrigger, AutoApproveHandler, HITLCheckResult, HITLConfig, HITLEngine,
    LlmGenerateConfig, LocalizedHandler, MessageLanguageConfig, MessageLanguageStrategy,
    MessageResolver, RejectAllHandler, StateApprovalConfig, StateApprovalTrigger, TimeoutAction,
    ToolApprovalConfig, create_handler, create_localized_handler, resolve_best_language,
    resolve_tool_message,
};

// Tool Provider System (v0.5.1 - Simplified)
pub use tools::{
    ProviderHealth, ToolAliases, ToolContext, ToolDescriptor, ToolMetadata, ToolProvider,
    ToolProviderError, ToolProviderType, TrustLevel,
};
