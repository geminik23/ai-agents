use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use ai_agents_context::ContextManager;
use ai_agents_core::{AgentError, AgentStorage, LLMProvider, Result};
use ai_agents_hitl::{ApprovalHandler, HITLEngine, RejectAllHandler};
use ai_agents_hooks::AgentHooks;
use ai_agents_llm::LLMRegistry;
use ai_agents_llm::providers::{ProviderType, UnifiedLLMProvider};
use ai_agents_memory::{
    CompactingMemory, InMemoryStore, LLMSummarizer, Memory, NoopSummarizer, Summarizer,
};
use ai_agents_process::ProcessProcessor;
use ai_agents_recovery::{MessageFilter, RecoveryManager};
use ai_agents_skills::{SkillDefinition, SkillLoader};
use ai_agents_state::{LLMTransitionEvaluator, StateMachine, TransitionEvaluator};
use ai_agents_template::{TemplateInheritance, TemplateLoader, TemplateRenderer};
use ai_agents_tools::{ToolRegistry, ToolSecurityEngine};

use super::AgentInfo;
use super::runtime::RuntimeAgent;
use crate::spec::{AgentSpec, StorageConfig};

pub struct AgentBuilder {
    spec: Option<AgentSpec>,
    llm: Option<Arc<dyn LLMProvider>>,
    llm_registry: Option<LLMRegistry>,
    memory: Option<Arc<dyn Memory>>,
    tools: Option<ToolRegistry>,
    skills: Vec<SkillDefinition>,
    skill_loader: Option<SkillLoader>,
    system_prompt: Option<String>,
    tools_prompt: Option<String>,
    auto_tools_prompt: bool,
    max_iterations: Option<u32>,
    max_context_tokens: Option<u32>,
    recovery_manager: Option<RecoveryManager>,
    tool_security: Option<ToolSecurityEngine>,
    process_processor: Option<ProcessProcessor>,
    message_filters: HashMap<String, Arc<dyn MessageFilter>>,
    context_manager: Option<Arc<ContextManager>>,
    state_machine: Option<Arc<StateMachine>>,
    transition_evaluator: Option<Arc<dyn TransitionEvaluator>>,
    hooks: Option<Arc<dyn AgentHooks>>,
    hitl_engine: Option<HITLEngine>,
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
    storage_config: Option<StorageConfig>,
    storage: Option<Arc<dyn AgentStorage>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            spec: None,
            llm: None,
            llm_registry: None,
            memory: None,
            tools: None,
            skills: Vec::new(),
            skill_loader: None,
            system_prompt: None,
            tools_prompt: None,
            auto_tools_prompt: true,
            max_iterations: None,
            max_context_tokens: None,
            recovery_manager: None,
            tool_security: None,
            process_processor: None,
            message_filters: HashMap::new(),
            context_manager: None,
            state_machine: None,
            transition_evaluator: None,
            hooks: None,
            hitl_engine: None,
            approval_handler: None,
            storage_config: None,
            storage: None,
        }
    }

    pub fn from_spec(spec: AgentSpec) -> Self {
        let system_prompt = spec.system_prompt.clone();
        let max_iterations = Some(spec.max_iterations);
        let max_context_tokens = Some(spec.max_context_tokens);

        Self {
            spec: Some(spec),
            llm: None,
            llm_registry: None,
            memory: None,
            tools: None,
            skills: Vec::new(),
            skill_loader: None,
            system_prompt: Some(system_prompt),
            tools_prompt: None,
            auto_tools_prompt: true,
            max_iterations,
            max_context_tokens,
            recovery_manager: None,
            tool_security: None,
            process_processor: None,
            message_filters: HashMap::new(),
            context_manager: None,
            state_machine: None,
            transition_evaluator: None,
            hooks: None,
            hitl_engine: None,
            approval_handler: None,
            storage_config: None,
            storage: None,
        }
    }

    pub fn from_yaml(yaml_content: &str) -> Result<Self> {
        let spec: AgentSpec = serde_yaml::from_str(yaml_content)?;
        spec.validate()?;
        Ok(Self::from_spec(spec))
    }

    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(AgentError::IoError)?;
        Self::from_yaml(&content)
    }

    pub fn from_template(template_name: &str) -> Result<Self> {
        let loader = TemplateLoader::new();
        Self::from_template_with_loader(template_name, &loader)
    }

    pub fn from_template_with_loader(template_name: &str, loader: &TemplateLoader) -> Result<Self> {
        let renderer = TemplateRenderer::new();
        let variables = loader.variables();

        let load_and_render = |name: &str| -> Result<String> {
            let content = loader.load_template(name)?;
            renderer.render(&content, variables)
        };

        let rendered_root = load_and_render(template_name)?;
        let processed = TemplateInheritance::process(&rendered_root, load_and_render)?;
        let spec: AgentSpec = serde_yaml::from_str(&processed)?;
        spec.validate()?;
        Ok(Self::from_spec(spec))
    }

    pub fn auto_configure_llms(mut self) -> Result<Self> {
        let spec = self
            .spec
            .as_ref()
            .ok_or_else(|| AgentError::Config("Cannot auto-configure LLMs without spec".into()))?;

        if !spec.llms.is_empty() {
            let mut registry = LLMRegistry::new();

            for (alias, config) in &spec.llms {
                let provider_type = ProviderType::from_str(&config.provider)
                    .map_err(|e| AgentError::Config(e.to_string()))?;

                let provider = UnifiedLLMProvider::from_env(provider_type, &config.model)
                    .map_err(|e| AgentError::LLM(e.to_string()))?;

                registry.register(alias, Arc::new(provider));
            }

            let default_alias = spec.llm.get_default_alias();
            let router_alias = spec.llm.get_router_alias();

            registry.set_default(&default_alias);
            if let Some(router) = router_alias {
                registry.set_router(&router);
            }

            self.llm_registry = Some(registry);
        } else if let Some(config) = spec.llm.as_config() {
            let provider_type = ProviderType::from_str(&config.provider)
                .map_err(|e| AgentError::Config(e.to_string()))?;

            let provider = UnifiedLLMProvider::from_env(provider_type, &config.model)
                .map_err(|e| AgentError::LLM(e.to_string()))?;

            self.llm = Some(Arc::new(provider));
        }

        Ok(self)
    }

    pub fn auto_configure_features(mut self) -> Result<Self> {
        if let Some(ref spec) = self.spec {
            self.recovery_manager = Some(RecoveryManager::new(spec.error_recovery.clone()));
            self.tool_security = Some(ToolSecurityEngine::new(spec.tool_security.clone()));

            if spec.has_process() {
                let mut processor = ProcessProcessor::new(spec.process.clone());
                if let Some(ref registry) = self.llm_registry {
                    processor = processor.with_llm_registry(Arc::new(registry.clone()));
                }
                self.process_processor = Some(processor);
            }
        }
        Ok(self)
    }

    pub fn llm(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn llm_alias(mut self, alias: impl Into<String>, provider: Arc<dyn LLMProvider>) -> Self {
        if self.llm_registry.is_none() {
            self.llm_registry = Some(LLMRegistry::new());
        }
        if let Some(ref mut registry) = self.llm_registry {
            registry.register(alias, provider);
        }
        self
    }

    pub fn llm_registry(mut self, registry: LLMRegistry) -> Self {
        self.llm_registry = Some(registry);
        self
    }

    pub fn memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn skill(mut self, skill: SkillDefinition) -> Self {
        self.skills.push(skill);
        self
    }

    pub fn skills(mut self, skills: Vec<SkillDefinition>) -> Self {
        self.skills.extend(skills);
        self
    }

    pub fn skill_loader(mut self, loader: SkillLoader) -> Self {
        self.skill_loader = Some(loader);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn tools_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.tools_prompt = Some(prompt.into());
        self.auto_tools_prompt = false;
        self
    }

    pub fn auto_tools_prompt(mut self, auto: bool) -> Self {
        self.auto_tools_prompt = auto;
        self
    }

    pub fn max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = Some(max);
        self
    }

    pub fn max_context_tokens(mut self, tokens: u32) -> Self {
        self.max_context_tokens = Some(tokens);
        self
    }

    pub fn recovery_manager(mut self, manager: RecoveryManager) -> Self {
        self.recovery_manager = Some(manager);
        self
    }

    pub fn tool_security(mut self, engine: ToolSecurityEngine) -> Self {
        self.tool_security = Some(engine);
        self
    }

    pub fn process_processor(mut self, processor: ProcessProcessor) -> Self {
        self.process_processor = Some(processor);
        self
    }

    pub fn message_filter(
        mut self,
        name: impl Into<String>,
        filter: Arc<dyn MessageFilter>,
    ) -> Self {
        self.message_filters.insert(name.into(), filter);
        self
    }

    pub fn context_manager(mut self, manager: Arc<ContextManager>) -> Self {
        self.context_manager = Some(manager);
        self
    }

    pub fn state_machine(mut self, machine: Arc<StateMachine>) -> Self {
        self.state_machine = Some(machine);
        self
    }

    pub fn transition_evaluator(mut self, evaluator: Arc<dyn TransitionEvaluator>) -> Self {
        self.transition_evaluator = Some(evaluator);
        self
    }

    pub fn hooks(mut self, hooks: Arc<dyn AgentHooks>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    pub fn approval_handler(mut self, handler: Arc<dyn ApprovalHandler>) -> Self {
        self.approval_handler = Some(handler);
        self
    }

    pub fn hitl_engine(mut self, engine: HITLEngine) -> Self {
        self.hitl_engine = Some(engine);
        self
    }

    pub fn storage_config(mut self, config: StorageConfig) -> Self {
        self.storage_config = Some(config);
        self
    }

    pub fn storage(mut self, storage: Arc<dyn AgentStorage>) -> Self {
        self.storage = Some(storage);
        self
    }

    pub fn build(mut self) -> Result<RuntimeAgent> {
        let base_prompt = self
            .system_prompt
            .ok_or_else(|| AgentError::Config("System prompt is required".into()))?;

        let tools = self.tools.unwrap_or_else(ToolRegistry::new);

        // ERROR NOTE: Don't include tools prompt here
        // - it will be added AFTER template rendering in get_effective_system_prompt() to avoid Jinja2 parsing JSON braces
        let system_prompt = base_prompt;

        let max_iterations = self.max_iterations.unwrap_or(10);

        let info = if let Some(ref spec) = self.spec {
            AgentInfo::new(&spec.name, &spec.name, &spec.version)
                .with_description(spec.description.clone().unwrap_or_default())
        } else {
            AgentInfo::new("agent", "Agent", "1.0.0")
        };

        if let Some(ref spec) = self.spec {
            if !spec.skills.is_empty() {
                let mut loader = self.skill_loader.take().unwrap_or_else(SkillLoader::new);
                let loaded_skills = loader.load_refs(&spec.skills)?;
                self.skills.extend(loaded_skills);
            }
        }

        let mut llm_registry = self.llm_registry.unwrap_or_else(LLMRegistry::new);

        if let Some(llm) = self.llm {
            if !llm_registry.has("default") {
                llm_registry.register("default", llm.clone());
            }
        }

        if let Some(ref spec) = self.spec {
            let default_alias = spec.llm.get_default_alias();
            let router_alias = spec.llm.get_router_alias();

            llm_registry.set_default(&default_alias);
            if let Some(router) = router_alias {
                llm_registry.set_router(&router);
            }
        }

        if llm_registry.is_empty() {
            return Err(AgentError::Config(
                "At least one LLM provider is required".into(),
            ));
        }

        // Create memory after LLM registry is ready (needed for CompactingMemory summarizer)
        let memory = self.memory.unwrap_or_else(|| {
            if let Some(ref spec) = self.spec {
                if spec.memory.is_compacting() {
                    let summarizer_llm = spec
                        .memory
                        .summarizer_llm
                        .as_ref()
                        .and_then(|alias| llm_registry.get(alias).ok())
                        .or_else(|| llm_registry.router().ok())
                        .or_else(|| llm_registry.default().ok());

                    let summarizer: Arc<dyn Summarizer> = match summarizer_llm {
                        Some(llm) => Arc::new(LLMSummarizer::new(llm)),
                        None => Arc::new(NoopSummarizer),
                    };
                    let config = spec.memory.to_compacting_config();
                    return Arc::new(CompactingMemory::new(summarizer, config));
                }
                Arc::new(InMemoryStore::new(spec.memory.max_messages))
            } else {
                Arc::new(InMemoryStore::new(100))
            }
        });

        let tools_arc = Arc::new(tools);
        let llm_registry_arc = Arc::new(llm_registry);

        let mut agent = RuntimeAgent::new(
            info,
            llm_registry_arc.clone(),
            memory,
            tools_arc,
            self.skills,
            system_prompt,
            max_iterations,
        );

        if let Some(tokens) = self.max_context_tokens {
            agent = agent.with_max_context_tokens(tokens);
        }

        if let Some(manager) = self.recovery_manager {
            agent = agent.with_recovery_manager(manager);
        } else if let Some(ref spec) = self.spec {
            agent = agent.with_recovery_manager(RecoveryManager::new(spec.error_recovery.clone()));
        }

        if let Some(engine) = self.tool_security {
            agent = agent.with_tool_security(engine);
        } else if let Some(ref spec) = self.spec {
            agent = agent.with_tool_security(ToolSecurityEngine::new(spec.tool_security.clone()));
        }

        if let Some(processor) = self.process_processor {
            agent = agent.with_process_processor(processor);
        } else if let Some(ref spec) = self.spec {
            if spec.has_process() {
                let processor = ProcessProcessor::new(spec.process.clone())
                    .with_llm_registry(llm_registry_arc.clone());
                agent = agent.with_process_processor(processor);
            }
        }

        for (name, filter) in self.message_filters {
            agent.register_message_filter(name, filter);
        }

        // Configure state machine from spec or builder
        if let Some(state_machine) = self.state_machine {
            let evaluator = self.transition_evaluator.unwrap_or_else(|| {
                let eval_llm = llm_registry_arc
                    .get("evaluator")
                    .or_else(|_| llm_registry_arc.router())
                    .or_else(|_| llm_registry_arc.default())
                    .expect("At least one LLM required for transition evaluator");
                Arc::new(LLMTransitionEvaluator::new(eval_llm))
            });
            agent = agent.with_state_machine(state_machine, evaluator);
        } else if let Some(ref spec) = self.spec {
            if let Some(ref state_config) = spec.states {
                let state_machine = StateMachine::new(state_config.clone())?;
                let evaluator = self.transition_evaluator.unwrap_or_else(|| {
                    let eval_llm = llm_registry_arc
                        .get("evaluator")
                        .or_else(|_| llm_registry_arc.router())
                        .or_else(|_| llm_registry_arc.default())
                        .expect("At least one LLM required for transition evaluator");
                    Arc::new(LLMTransitionEvaluator::new(eval_llm))
                });
                agent = agent.with_state_machine(Arc::new(state_machine), evaluator);
            }
        }

        // Configure context manager from spec or builder
        if let Some(context_manager) = self.context_manager {
            agent = agent.with_context_manager(context_manager);
        } else if let Some(ref spec) = self.spec {
            if !spec.context.is_empty() {
                let context_manager = ContextManager::new(
                    spec.context.clone(),
                    spec.name.clone(),
                    spec.version.clone(),
                );
                agent = agent.with_context_manager(Arc::new(context_manager));
            }
        }

        // Configure parallel tools and streaming from spec
        if let Some(ref spec) = self.spec {
            agent = agent.with_parallel_tools(spec.parallel_tools.clone());
            agent = agent.with_streaming(spec.streaming.clone());

            // Configure memory token budget if specified
            if let Some(ref budget) = spec.memory.token_budget {
                agent = agent.with_memory_token_budget(budget.clone());
            }

            // Configure storage from spec if not explicitly set
            if self.storage_config.is_none() && spec.has_storage() {
                agent = agent.with_storage_config(spec.storage.clone());
            }
        }

        // Configure storage from builder
        if let Some(storage_config) = self.storage_config {
            agent = agent.with_storage_config(storage_config);
        }
        if let Some(storage) = self.storage {
            agent = agent.with_storage(storage);
        }

        // Configure hooks
        if let Some(hooks) = self.hooks {
            agent = agent.with_hooks(hooks);
        }

        // Configure HITL from spec or builder
        if let Some(hitl_engine) = self.hitl_engine {
            let handler = self
                .approval_handler
                .unwrap_or_else(|| Arc::new(RejectAllHandler::new()));
            agent = agent.with_hitl(hitl_engine, handler);
        } else if let Some(ref spec) = self.spec {
            if let Some(ref hitl_config) = spec.hitl {
                let hitl_engine = HITLEngine::new(hitl_config.clone());
                let handler = self
                    .approval_handler
                    .unwrap_or_else(|| Arc::new(RejectAllHandler::new()));
                agent = agent.with_hitl(hitl_engine, handler);
            }
        }

        Ok(agent)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_new() {
        let builder = AgentBuilder::new();
        assert!(builder.spec.is_none());
        assert!(builder.system_prompt.is_none());
    }

    #[test]
    fn test_builder_from_yaml() {
        let yaml = r#"
name: TestAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        assert_eq!(builder.spec.as_ref().unwrap().name, "TestAgent");
    }

    #[test]
    fn test_builder_from_yaml_with_tool_security() {
        let yaml = r#"
name: SecureAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
max_context_tokens: 8192
error_recovery:
  default:
    max_retries: 5
tool_security:
  enabled: true
  tools:
    http:
      rate_limit: 10
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        let spec = builder.spec.as_ref().unwrap();
        assert_eq!(spec.max_context_tokens, 8192);
        assert_eq!(spec.error_recovery.default.max_retries, 5);
        assert!(spec.tool_security.enabled);
    }

    #[test]
    fn test_builder_from_yaml_with_skills() {
        let yaml = r#"
name: SkillAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
skills:
  - id: greeting
    description: "Greet users"
    trigger: "When user says hello"
    steps:
      - prompt: "Hello!"
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        assert!(!builder.spec.as_ref().unwrap().skills.is_empty());
    }

    #[test]
    fn test_builder_from_spec() {
        let spec = AgentSpec {
            name: "test".to_string(),
            version: "1.0".to_string(),
            description: Some("Test agent".to_string()),
            system_prompt: "You are helpful".to_string(),
            ..Default::default()
        };

        let builder = AgentBuilder::from_spec(spec);
        assert!(builder.spec.is_some());
        assert_eq!(builder.system_prompt, Some("You are helpful".to_string()));
    }

    #[test]
    fn test_builder_chain() {
        let builder = AgentBuilder::new()
            .system_prompt("Test prompt")
            .max_iterations(5)
            .max_context_tokens(4096);

        assert_eq!(builder.system_prompt, Some("Test prompt".to_string()));
        assert_eq!(builder.max_iterations, Some(5));
        assert_eq!(builder.max_context_tokens, Some(4096));
    }

    #[test]
    fn test_builder_skills() {
        use ai_agents_skills::{SkillDefinition, SkillStep};

        let skill = SkillDefinition {
            id: "test".to_string(),
            description: "Test skill".to_string(),
            trigger: "When testing".to_string(),
            steps: vec![SkillStep::Prompt {
                prompt: "Hello".to_string(),
                llm: None,
            }],
        };

        let builder = AgentBuilder::new().skill(skill.clone()).skills(vec![skill]);

        assert_eq!(builder.skills.len(), 2);
    }

    #[test]
    fn test_builder_from_yaml_with_states() {
        let yaml = r#"
name: StatefulAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
states:
  initial: greeting
  states:
    greeting:
      prompt: "Welcome!"
      transitions:
        - to: support
          when: "user needs help"
    support:
      prompt: "How can I help?"
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        let spec = builder.spec.as_ref().unwrap();
        assert!(spec.has_states());
        assert!(spec.states.is_some());
        let states = spec.states.as_ref().unwrap();
        assert_eq!(states.initial, "greeting");
        assert_eq!(states.states.len(), 2);
    }

    #[test]
    fn test_builder_from_yaml_with_context() {
        let yaml = r#"
name: ContextAgent
system_prompt: "Hello, {{ context.user.name }}!"
llm:
  provider: openai
  model: gpt-4
context:
  user:
    type: runtime
    required: true
  time:
    type: builtin
    source: datetime
    refresh: per_turn
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        let spec = builder.spec.as_ref().unwrap();
        assert!(spec.has_context());
        assert_eq!(spec.context.len(), 2);
        assert!(spec.context.contains_key("user"));
        assert!(spec.context.contains_key("time"));
    }

    #[test]
    fn test_builder_from_yaml_with_full_v04_features() {
        let yaml = r#"
name: FullFeaturedAgent
version: "0.4.0"
system_prompt: |
  You are a helpful assistant.
  User: {{ context.user.name }}
  Language: {{ context.user.language }}
llm:
  provider: openai
  model: gpt-4
context:
  user:
    type: runtime
    required: true
    default:
      name: "Guest"
      language: "en"
  time:
    type: builtin
    source: datetime
    refresh: per_turn
states:
  initial: greeting
  states:
    greeting:
      prompt: "Welcome to our service!"
      prompt_mode: append
      transitions:
        - to: support
          when: "user needs help"
          auto: true
          priority: 10
    support:
      prompt: "I'm here to help you."
      max_turns: 5
      timeout_to: escalation
      transitions:
        - to: closing
          when: "issue resolved"
          auto: true
    escalation:
      prompt: "Let me connect you with a human agent."
    closing:
      prompt: "Thank you for using our service!"
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        let spec = builder.spec.as_ref().unwrap();

        // Check context
        assert!(spec.has_context());
        assert_eq!(spec.context.len(), 2);

        // Check states
        assert!(spec.has_states());
        let states = spec.states.as_ref().unwrap();
        assert_eq!(states.initial, "greeting");
        assert_eq!(states.states.len(), 4);

        // Check greeting state details
        let greeting = states.states.get("greeting").unwrap();
        assert!(greeting.prompt.is_some());
        assert_eq!(greeting.transitions.len(), 1);
        assert_eq!(greeting.transitions[0].to, "support");
        assert!(greeting.transitions[0].auto);

        // Check support state has timeout
        let support = states.states.get("support").unwrap();
        assert_eq!(support.max_turns, Some(5));
        assert_eq!(support.timeout_to, Some("escalation".to_string()));
    }

    #[test]
    fn test_builder_from_yaml_with_hitl() {
        let yaml = r#"
name: HITLAgent
system_prompt: "You are a secure assistant."
llm:
  provider: openai
  model: gpt-4
hitl:
  default_timeout_seconds: 600
  on_timeout: reject
  tools:
    send_payment:
      require_approval: true
      approval_context:
        - amount
        - recipient
      approval_message: "Approve payment?"
    delete_record:
      require_approval: true
  conditions:
    - name: high_value
      when: "amount > 1000"
      require_approval: true
      approval_message: "High value transaction"
  states:
    escalation:
      on_enter: require_approval
      approval_message: "Escalate to human?"
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        let spec = builder.spec.as_ref().unwrap();

        assert!(spec.has_hitl());
        let hitl = spec.hitl.as_ref().unwrap();
        assert_eq!(hitl.default_timeout_seconds, 600);
        assert_eq!(hitl.tools.len(), 2);
        assert!(hitl.tools.get("send_payment").unwrap().require_approval);
        assert_eq!(hitl.conditions.len(), 1);
        assert_eq!(hitl.conditions[0].name, "high_value");
        assert_eq!(hitl.states.len(), 1);
    }

    #[test]
    fn test_builder_from_yaml_with_compacting_memory() {
        let yaml = r#"
name: CompactingAgent
system_prompt: "You are a helpful assistant."
memory:
  type: compacting
  max_messages: 100
  max_recent_messages: 20
  compress_threshold: 15
  summarize_batch_size: 5
  summarizer_llm: router
llm:
  provider: openai
  model: gpt-4
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        let spec = builder.spec.as_ref().unwrap();

        assert!(spec.memory.is_compacting());
        assert_eq!(spec.memory.max_recent_messages, Some(20));
        assert_eq!(spec.memory.compress_threshold, Some(15));
        assert_eq!(spec.memory.summarize_batch_size, Some(5));
        assert_eq!(spec.memory.summarizer_llm, Some("router".to_string()));

        let compacting_config = spec.memory.to_compacting_config();
        assert_eq!(compacting_config.max_recent_messages, 20);
        assert_eq!(compacting_config.compress_threshold, 15);
        assert_eq!(compacting_config.summarize_batch_size, 5);
    }

    #[test]
    fn test_builder_from_yaml_with_token_budget() {
        let yaml = r#"
name: BudgetAgent
system_prompt: "You are a helpful assistant."
memory:
  type: compacting
  max_messages: 100
  token_budget:
    total: 8192
    allocation:
      summary: 2048
      recent_messages: 4096
      facts: 1024
    overflow_strategy: summarize_more
    warn_at_percent: 75
llm:
  provider: openai
  model: gpt-4
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        assert!(builder.spec.is_some());
        let spec = builder.spec.as_ref().unwrap();

        assert!(spec.memory.token_budget.is_some());
        let budget = spec.memory.token_budget.as_ref().unwrap();
        assert_eq!(budget.total, 8192);
        assert_eq!(budget.allocation.summary, 2048);
        assert_eq!(budget.allocation.recent_messages, 4096);
        assert_eq!(budget.allocation.facts, 1024);
        assert_eq!(budget.warn_at_percent, 75);
    }

    #[test]
    fn test_builder_from_yaml_with_overflow_strategies() {
        use ai_agents_memory::OverflowStrategy;

        let yaml = r#"
name: TruncateAgent
system_prompt: "You are helpful."
memory:
  type: compacting
  token_budget:
    total: 4096
    overflow_strategy: truncate_oldest
llm:
  provider: openai
  model: gpt-4
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let budget = builder
            .spec
            .as_ref()
            .unwrap()
            .memory
            .token_budget
            .as_ref()
            .unwrap();
        assert_eq!(budget.overflow_strategy, OverflowStrategy::TruncateOldest);

        let yaml = r#"
name: ErrorAgent
system_prompt: "You are helpful."
memory:
  type: compacting
  token_budget:
    total: 4096
    overflow_strategy: error
llm:
  provider: openai
  model: gpt-4
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let budget = builder
            .spec
            .as_ref()
            .unwrap()
            .memory
            .token_budget
            .as_ref()
            .unwrap();
        assert_eq!(budget.overflow_strategy, OverflowStrategy::Error);
    }

    #[test]
    fn test_builder_from_yaml_with_storage_file() {
        let yaml = r#"
name: PersistentAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
storage:
  type: file
  path: "./data/sessions"
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let spec = builder.spec.as_ref().unwrap();
        assert!(spec.has_storage());
        assert!(spec.storage.is_file());
        assert_eq!(spec.storage.get_path(), Some("./data/sessions"));
    }

    #[test]
    fn test_builder_from_yaml_with_storage_sqlite() {
        let yaml = r#"
name: PersistentAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
storage:
  type: sqlite
  path: "./data/sessions.db"
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let spec = builder.spec.as_ref().unwrap();
        assert!(spec.has_storage());
        assert!(spec.storage.is_sqlite());
    }

    #[test]
    fn test_builder_from_yaml_with_storage_redis() {
        let yaml = r#"
name: PersistentAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
storage:
  type: redis
  url: "redis://localhost:6379"
  prefix: "myagent:"
  ttl_seconds: 86400
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let spec = builder.spec.as_ref().unwrap();
        assert!(spec.has_storage());
        assert!(spec.storage.is_redis());
        assert_eq!(spec.storage.get_prefix(), "myagent:");
        assert_eq!(spec.storage.get_ttl(), Some(86400));
    }

    #[test]
    fn test_builder_no_storage_by_default() {
        let yaml = r#"
name: SimpleAgent
system_prompt: "You are helpful."
llm:
  provider: openai
  model: gpt-4
"#;
        let builder = AgentBuilder::from_yaml(yaml).unwrap();
        let spec = builder.spec.as_ref().unwrap();
        assert!(!spec.has_storage());
    }
}
