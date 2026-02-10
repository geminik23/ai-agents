//! AI Agents Framework

pub mod agent {
    pub use ai_agents_runtime::{
        Agent, AgentBuilder, AgentInfo, AgentResponse, ParallelToolsConfig, RuntimeAgent,
        StreamChunk, StreamingConfig, ToolCall,
    };
}

pub mod context {
    pub use ai_agents_context::{
        BuiltinSource, ContextManager, ContextProvider, ContextSource, RefreshPolicy,
        TemplateRenderer,
    };
}

pub mod error {
    pub use ai_agents_core::{AgentError, Result};
}

pub mod hitl {
    pub use ai_agents_hitl::{
        ApprovalCondition, ApprovalHandler, ApprovalMessage, ApprovalRequest, ApprovalResult,
        ApprovalTrigger, AutoApproveHandler, CallbackHandler, HITLCheckResult, HITLConfig,
        HITLEngine, LlmGenerateConfig, LocalizedHandler, MessageLanguageConfig,
        MessageLanguageStrategy, MessageResolver, RejectAllHandler, StateApprovalConfig,
        StateApprovalTrigger, TimeoutAction, ToolApprovalConfig, create_handler,
        create_localized_handler, resolve_best_language, resolve_tool_message,
    };
}

pub mod hooks {
    pub use ai_agents_hooks::{AgentHooks, CompositeHooks, HookTimer, LoggingHooks, NoopHooks};
}

pub mod llm {
    pub use ai_agents_core::{
        ChatMessage, FinishReason, LLMCapability, LLMChunk, LLMConfig, LLMError, LLMFeature,
        LLMProvider, LLMResponse, Role, TaskContext, TokenUsage, ToolSelection,
    };
    pub use ai_agents_llm::LLMRegistry;

    pub mod providers {
        pub use ai_agents_llm::providers::{ProviderBuilder, ProviderType, UnifiedLLMProvider};
    }
}

pub mod memory {
    use std::sync::Arc;

    use ai_agents_core::LLMProvider;

    pub use ai_agents_core::MemorySnapshot;
    pub use ai_agents_memory::{
        CompactingMemory, CompactingMemoryConfig, CompressResult, CompressionEvent,
        ConversationContext, EvictionReason, FactExtractedEvent, InMemoryStore, LLMSummarizer,
        Memory, MemoryBudgetEvent, MemoryBudgetState, MemoryCompressEvent, MemoryEvictEvent,
        MemoryTokenBudget, NoopSummarizer, OverflowStrategy, Summarizer, TokenAllocation,
        estimate_message_tokens, estimate_tokens,
    };
    pub use ai_agents_runtime::spec::MemoryConfig;

    pub fn create_memory(memory_type: &str, max_messages: usize) -> Arc<dyn Memory> {
        match memory_type {
            "in-memory" => Arc::new(InMemoryStore::new(max_messages)),
            "compacting" => {
                let summarizer: Arc<dyn Summarizer> = Arc::new(NoopSummarizer);
                Arc::new(CompactingMemory::with_default_config(summarizer))
            }
            _ => Arc::new(InMemoryStore::new(max_messages)),
        }
    }

    pub fn create_memory_from_config(config: &MemoryConfig) -> Arc<dyn Memory> {
        if config.is_compacting() {
            let summarizer: Arc<dyn Summarizer> = Arc::new(NoopSummarizer);
            let compacting_config = config.to_compacting_config();
            Arc::new(CompactingMemory::new(summarizer, compacting_config))
        } else {
            Arc::new(InMemoryStore::new(config.max_messages))
        }
    }

    pub fn create_memory_from_config_with_llm(
        config: &MemoryConfig,
        llm: Option<Arc<dyn LLMProvider>>,
    ) -> Arc<dyn Memory> {
        if config.is_compacting() {
            let summarizer: Arc<dyn Summarizer> = match llm {
                Some(provider) => Arc::new(LLMSummarizer::new(provider)),
                None => Arc::new(NoopSummarizer),
            };
            let compacting_config = config.to_compacting_config();
            Arc::new(CompactingMemory::new(summarizer, compacting_config))
        } else {
            Arc::new(InMemoryStore::new(config.max_messages))
        }
    }

    pub fn create_compacting_memory(
        summarizer: Arc<dyn Summarizer>,
        config: CompactingMemoryConfig,
    ) -> Arc<dyn Memory> {
        Arc::new(CompactingMemory::new(summarizer, config))
    }

    pub fn create_compacting_memory_from_config(
        summarizer: Arc<dyn Summarizer>,
        config: &MemoryConfig,
    ) -> Arc<dyn Memory> {
        let compacting_config = config.to_compacting_config();
        Arc::new(CompactingMemory::new(summarizer, compacting_config))
    }
}

pub mod persistence {
    use std::sync::Arc;

    pub use ai_agents_core::{AgentSnapshot, AgentStorage, MemorySnapshot, Result};
    #[cfg(feature = "sqlite")]
    pub use ai_agents_storage::SqliteStorage;
    pub use ai_agents_storage::{
        FileStorage, SessionInfo, SessionMetadata, SessionOrderBy, SessionQuery,
    };
    #[cfg(feature = "redis-storage")]
    pub use ai_agents_storage::{RedisSessionMeta, RedisStorage};

    pub async fn create_storage(
        config: &crate::spec::StorageConfig,
    ) -> Result<Option<Arc<dyn AgentStorage>>> {
        let storage_config = match config {
            crate::spec::StorageConfig::None => ai_agents_storage::StorageConfig::None,
            crate::spec::StorageConfig::File(fc) => ai_agents_storage::StorageConfig::File {
                path: fc.path.clone(),
            },
            crate::spec::StorageConfig::Sqlite(sc) => ai_agents_storage::StorageConfig::Sqlite {
                path: sc.path.clone(),
            },
            crate::spec::StorageConfig::Redis(rc) => ai_agents_storage::StorageConfig::Redis {
                url: rc.url.clone(),
                prefix: rc.prefix.clone(),
                ttl_seconds: rc.ttl_seconds,
            },
        };

        ai_agents_storage::create_storage(&storage_config).await
    }
}

pub mod process {
    pub use ai_agents_process::{ProcessConfig, ProcessData, ProcessProcessor};
}

pub mod recovery {
    pub use ai_agents_recovery::{
        ByRoleFilter, ErrorRecoveryConfig, FilterConfig, KeepRecentFilter, MessageFilter,
        RecoveryManager, SkipPatternFilter,
    };
}

pub mod skill {
    pub use ai_agents_skills::{
        SkillContext, SkillDefinition, SkillExecutor, SkillLoader, SkillRef, SkillRouter,
        SkillStep, StepResult,
    };
}

pub mod spec {
    pub use ai_agents_runtime::spec::{
        AgentSpec, BuiltinProviderConfig, FileStorageConfig, LLMConfig, LLMSelector, MemoryConfig,
        ProviderPolicyConfig, ProviderSecurityConfig, ProvidersConfig, RedisStorageConfig,
        SqliteStorageConfig, StorageConfig, ToolAliasesConfig, ToolConfig, ToolPolicyConfig,
        YamlProviderConfig, YamlToolConfig,
    };
}

pub mod state {
    pub use ai_agents_state::{
        CompareOp, ContextExtractor, ContextMatcher, GuardConditions, GuardOnlyEvaluator,
        LLMTransitionEvaluator, PromptMode, StateAction, StateConfig, StateDefinition,
        StateMachine, StateMachineSnapshot, StateMatcher, StateTransitionEvent, TimeMatcher,
        ToolCondition, ToolRef, Transition, TransitionContext, TransitionEvaluator,
        TransitionGuard,
    };
}

pub mod template {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use ai_agents_template::TemplateLoader as InnerTemplateLoader;
    pub use ai_agents_template::{TemplateInheritance, TemplateRenderer};

    use crate::error::Result;
    use crate::spec::AgentSpec;

    pub struct TemplateLoader {
        inner: InnerTemplateLoader,
    }

    impl TemplateLoader {
        pub fn new() -> Self {
            Self {
                inner: InnerTemplateLoader::new(),
            }
        }

        pub fn add_search_path(&mut self, path: impl Into<PathBuf>) -> &mut Self {
            self.inner.add_search_path(path);
            self
        }

        pub fn set_variable(
            &mut self,
            key: impl Into<String>,
            value: impl Into<String>,
        ) -> &mut Self {
            self.inner.set_variable(key, value);
            self
        }

        pub fn set_variables(&mut self, vars: HashMap<String, String>) -> &mut Self {
            self.inner.set_variables(vars);
            self
        }

        pub fn get_variable(&self, key: &str) -> Option<&str> {
            self.inner.get_variable(key)
        }

        pub fn load_template(&self, name: &str) -> Result<String> {
            self.inner.load_template(name)
        }

        pub fn template_exists(&self, name: &str) -> bool {
            self.inner.template_exists(name)
        }

        pub fn search_paths(&self) -> &[PathBuf] {
            self.inner.search_paths()
        }

        pub fn variables(&self) -> &HashMap<String, String> {
            self.inner.variables()
        }

        pub fn load_and_parse(&self, template_name: &str) -> Result<AgentSpec> {
            let renderer = TemplateRenderer::new();
            let variables = self.variables();

            let load_and_render = |name: &str| -> Result<String> {
                let content = self.load_template(name)?;
                renderer.render(&content, variables)
            };

            let rendered_root = load_and_render(template_name)?;
            let processed = TemplateInheritance::process(&rendered_root, load_and_render)?;
            let spec: AgentSpec = serde_yaml::from_str(&processed)?;
            spec.validate()?;

            Ok(spec)
        }
    }

    impl Default for TemplateLoader {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AsRef<InnerTemplateLoader> for TemplateLoader {
        fn as_ref(&self) -> &InnerTemplateLoader {
            &self.inner
        }
    }

    impl From<InnerTemplateLoader> for TemplateLoader {
        fn from(inner: InnerTemplateLoader) -> Self {
            Self { inner }
        }
    }
}

pub mod reasoning {
    pub use ai_agents_reasoning::{
        CriterionResult, EvaluationResult, Plan, PlanAction, PlanAvailableActions,
        PlanReflectionConfig, PlanStatus, PlanStep, PlanningConfig, ReasoningConfig,
        ReasoningMetadata, ReasoningMode, ReasoningOutput, ReflectionAttempt, ReflectionConfig,
        ReflectionMetadata, ReflectionMode, StepFailureAction, StepStatus, StringOrList,
    };
}

pub mod disambiguation {
    pub use ai_agents_disambiguation::{
        AmbiguityAspect, AmbiguityDetectionResult, AmbiguityDetector, AmbiguityType, CacheConfig,
        ClarificationConfig, ClarificationGenerator, ClarificationOption, ClarificationParseResult,
        ClarificationQuestion, ClarificationStyle, ContextConfig, DetectionConfig,
        DisambiguationConfig, DisambiguationContext, DisambiguationManager, DisambiguationResult,
        MaxAttemptsAction, SkillDisambiguationOverride, SkipCondition, StateDisambiguationOverride,
    };
}

pub mod tool_security {
    pub use ai_agents_tools::{
        SecurityCheckResult, ToolPolicyConfig, ToolSecurityConfig, ToolSecurityEngine,
    };
}

pub mod tools {
    pub use ai_agents_core::{Tool, ToolInfo, ToolResult};
    #[cfg(feature = "http-tool")]
    pub use ai_agents_tools::HttpTool;
    pub use ai_agents_tools::{
        CalculatorTool, ConditionEvaluator, DateTimeTool, EchoTool, EvaluationContext, FileTool,
        JsonTool, LLMGetter, MathTool, ProviderHealth, RandomTool, SimpleLLMGetter, TemplateTool,
        TextTool, ToolAliases, ToolCallRecord, ToolContext, ToolDescriptor, ToolMetadata,
        ToolProvider, ToolProviderError, ToolProviderType, ToolRegistry, TrustLevel,
        create_builtin_registry,
    };
}

// Top-level re-exports (legacy interface)
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

// Reasoning & Reflection (v0.5.3)
pub use reasoning::{
    CriterionResult, EvaluationResult, Plan, PlanAction, PlanAvailableActions,
    PlanReflectionConfig, PlanStatus, PlanStep, PlanningConfig, ReasoningConfig, ReasoningMetadata,
    ReasoningMode, ReasoningOutput, ReflectionAttempt, ReflectionConfig, ReflectionMetadata,
    ReflectionMode, StepFailureAction, StepStatus, StringOrList,
};

// Intent Disambiguation (v0.5.4)
pub use disambiguation::{
    AmbiguityAspect, AmbiguityDetectionResult, AmbiguityDetector, AmbiguityType, CacheConfig,
    ClarificationConfig, ClarificationGenerator, ClarificationOption, ClarificationParseResult,
    ClarificationQuestion, ClarificationStyle, ContextConfig, DetectionConfig,
    DisambiguationConfig, DisambiguationContext, DisambiguationManager, DisambiguationResult,
    MaxAttemptsAction, SkillDisambiguationOverride, SkipCondition, StateDisambiguationOverride,
};
