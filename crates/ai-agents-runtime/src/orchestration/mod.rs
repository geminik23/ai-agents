//! Multi-agent orchestration functions.
//! Coordination patterns built on top of AgentRegistry primitives.

pub mod aggregation;
pub mod concurrent;
pub mod context;
pub mod group_chat;
pub mod handoff;
pub mod pipeline;
pub mod route;
pub mod tools;
pub mod types;

pub use concurrent::{concurrent, concurrent_with_llms};
pub use group_chat::{group_chat, group_chat_with_llms};
pub use handoff::handoff;
pub use pipeline::pipeline;
pub use route::route;
pub use types::*;

/// Borrowed providers for independent aggregation responsibilities.
#[derive(Clone, Copy)]
pub struct AggregationLLMs<'a> {
    pub synthesis: Option<&'a dyn ai_agents_llm::LLMProvider>,
    pub vote: Option<&'a dyn ai_agents_llm::LLMProvider>,
    pub tiebreak: Option<&'a dyn ai_agents_llm::LLMProvider>,
}

impl<'a> AggregationLLMs<'a> {
    /// Keeps the historical one-provider aggregation contract.
    pub fn shared(provider: Option<&'a dyn ai_agents_llm::LLMProvider>) -> Self {
        Self {
            synthesis: provider,
            vote: provider,
            tiebreak: provider,
        }
    }
}

/// Borrowed providers for framework-directed group decisions, not participant turns.
#[derive(Clone, Copy)]
pub struct GroupLLMs<'a> {
    pub speaker: Option<&'a dyn ai_agents_llm::LLMProvider>,
    pub consensus: Option<&'a dyn ai_agents_llm::LLMProvider>,
}

impl<'a> GroupLLMs<'a> {
    /// Keeps the historical one-provider group contract.
    pub fn shared(provider: Option<&'a dyn ai_agents_llm::LLMProvider>) -> Self {
        Self {
            speaker: provider,
            consensus: provider,
        }
    }
}

// Literal router lookup is retained only for legacy orchestration consumers.
pub(crate) fn role_provider(
    registry: &ai_agents_llm::LLMRegistry,
    role: ai_agents_llm::LLMRole,
    local: Option<&str>,
) -> ai_agents_core::Result<Option<std::sync::Arc<dyn ai_agents_llm::LLMProvider>>> {
    Ok(
        match registry
            .resolve_role_override(role, local)
            .map_err(|e| ai_agents_core::AgentError::Config(e.to_string()))?
        {
            Some(resolved) => Some(resolved.provider),
            None => registry.get(local.unwrap_or("router")).ok(),
        },
    )
}

pub(crate) struct AggregationProviders {
    synthesis: Option<std::sync::Arc<dyn ai_agents_llm::LLMProvider>>,
    vote: Option<std::sync::Arc<dyn ai_agents_llm::LLMProvider>>,
    tiebreak: Option<std::sync::Arc<dyn ai_agents_llm::LLMProvider>>,
}

impl AggregationProviders {
    // Capture one agent's resolved handles before participant execution begins.
    pub(crate) fn resolve(
        registry: &ai_agents_llm::LLMRegistry,
        local: Option<&str>,
    ) -> ai_agents_core::Result<Self> {
        use ai_agents_llm::LLMRole;
        Ok(Self {
            synthesis: role_provider(registry, LLMRole::OrchestrationSynthesis, local)?,
            vote: role_provider(registry, LLMRole::OrchestrationVote, local)?,
            tiebreak: role_provider(registry, LLMRole::OrchestrationTiebreak, local)?,
        })
    }

    // Borrow captured handles without rebuilding clients or bypassing wrappers.
    pub(crate) fn as_refs(&self) -> AggregationLLMs<'_> {
        AggregationLLMs {
            synthesis: self.synthesis.as_deref(),
            vote: self.vote.as_deref(),
            tiebreak: self.tiebreak.as_deref(),
        }
    }
}
