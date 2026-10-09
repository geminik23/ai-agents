//! Dependency-light invocation admission contract; live wrappers and execution state belong to their owning crates.

use crate::{ChatMessage, LLMConfig, LLMError, LLMToolRequest};
use serde::{Deserialize, Serialize};

/// The exact borrowed inputs that the managed wrapper will dispatch to the bound implementation.
/// Providers must cover internal retries, native services and all billable units, not estimated tokens.
pub struct ProviderRequest<'a> {
    pub request_id: &'a str,
    pub messages: &'a [ChatMessage],
    pub config: Option<&'a LLMConfig>,
    pub tools: Option<&'a LLMToolRequest>,
    pub streaming: bool,
}

/// A trusted implementation's conservative upper bound under an immutable declared pricing schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostBound {
    pub provider_identity: String,
    pub pricing_identity: String,
    pub request_id: String,
    pub max_micro_usd: u64,
}

/// Matching settlement may release only the unused part of this exact request's upper bound.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostSettlement {
    pub bound: ProviderCostBound,
    pub charged_micro_usd: u64,
}

/// An implementation-bound conservative union of logical mutation targets, not a filesystem sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolWriteFootprint {
    pub binding_identity: String,
    pub targets: Vec<String>,
}

impl ToolWriteFootprint {
    /// Declares a proven side-effect-free implementation or preview, not model-supplied read-only metadata.
    pub fn empty(binding_identity: impl Into<String>) -> Self {
        Self {
            binding_identity: binding_identity.into(),
            targets: Vec::new(),
        }
    }
}

impl ProviderCostBound {
    /// Invalid identities cannot become durable pricing bindings.
    pub fn validate(&self, request_id: &str) -> Result<(), LLMError> {
        if self.request_id != request_id
            || self.provider_identity.trim().is_empty()
            || self.pricing_identity.trim().is_empty()
        {
            return Err(LLMError::Config("invalid priced request binding".into()));
        }
        Ok(())
    }
}

/// Only the provider implementation may derive a matching priced settlement from returned usage.
pub fn validate_cost_settlement(
    bound: &ProviderCostBound,
    settlement: &ProviderCostSettlement,
) -> Result<(), LLMError> {
    if &settlement.bound != bound || settlement.charged_micro_usd > bound.max_micro_usd {
        return Err(LLMError::Config(
            "priced settlement does not match its request bound".into(),
        ));
    }
    Ok(())
}

use async_trait::async_trait;

/// Shared provider consumers use a host-installed, invocation-scoped ledger rather than mutable global budgets.
#[async_trait]
pub trait InvocationAdmission: Send + Sync {
    /// Reserves one managed attempt before invoking a provider implementation.
    async fn admit_llm(&self) -> Result<String, LLMError>;
    /// Additive priced admission; legacy ledgers remain usable for ordinary attempt accounting.
    async fn admit_priced_llm(
        &self,
        _bound: Option<ProviderCostBound>,
    ) -> Result<String, LLMError> {
        self.admit_llm().await
    }
    /// A malformed or unbounded priced request cannot be hidden by a provider fallback.
    fn reject_priced_request(&self) {}
    /// Requests pricing only when the run enforces a declared-priced ceiling.
    fn requires_priced_llm(&self) -> bool {
        false
    }
    /// Settles matching priced usage before publication; missing usage retains the conservative bound.
    async fn settle_priced_llm(
        &self,
        id: &str,
        result: serde_json::Value,
        _settlement: Option<ProviderCostSettlement>,
    ) -> Result<(), LLMError> {
        self.settle_llm(id, result).await
    }
    /// Acknowledges the exact result before publishing it to the caller.
    async fn settle_llm(&self, id: &str, result: serde_json::Value) -> Result<(), LLMError>;
    /// Drop cannot await storage; retain dispatched reservations for explicit recovery.
    fn abandon_llm(&self, id: &str);
    /// Run-wide stops cannot be hidden by ordinary provider retry/fallback behavior.
    fn execution_stopped(&self) -> bool;
    /// A cooperative deadline bounds managed waits without promising physical termination of host work.
    fn remaining_duration(&self) -> Option<std::time::Duration> {
        None
    }
}
