//! Declared-priced accounting for first-party adapters, independent of telemetry estimates.
//!
//! The trusted host supplies an audited model/endpoint contract with server-enforced token ceilings.
//! We reserve those complete ceilings, not guessed prompt tokens or an unenforced model label.

use ai_agents_core::autonomy::{ProviderCostBound, ProviderCostSettlement, ProviderRequest};
use ai_agents_core::{LLMConfig, LLMError, LLMResponse};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Immutable host pricing and model-limit contract for one exact provider/model/endpoint.
/// Token rates are integer micro-USD per million tokens; reservations round upward once per request.
/// The host must verify the server-enforced ceilings and rates, including reasoning and tiered/cache pricing.
/// These are declared-price guarantees, not discovery of current prices or a provider invoice guarantee.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredPricing {
    pub schedule_id: String,
    pub model: String,
    pub endpoint: String,
    pub server_input_token_limit: u32,
    pub server_output_token_limit: u32,
    pub input_micro_usd_per_million: u64,
    pub output_micro_usd_per_million: u64,
    #[serde(default)]
    pub request_fee_micro_usd: u64,
    /// Maximum total upstream attempts, including the initial request, not agent-level retries.
    #[serde(default = "one")]
    pub max_internal_attempts: u32,
    /// True only when the host has audited that reported usage covers every billable token class.
    /// Otherwise even successful unary results retain the full reservation.
    #[serde(default)]
    pub complete_usage: bool,
}

// A missing trailing separator is equivalent, but repeated path separators can select another server route.
fn normalize_endpoint(endpoint: &str) -> Result<String, LLMError> {
    let mut url = reqwest::Url::parse(endpoint).map_err(|_| invalid("invalid endpoint"))?;
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url.to_string())
}

// Display aliases for the same chat protocol and exact endpoint/model cannot open a second price-pin namespace.
fn accounting_adapter(provider: &str) -> &str {
    match provider {
        "openai" | "openai-compatible" | "groq" | "cohere" | "mistral" | "openrouter" => {
            "openai-chat.v1"
        }
        other => other,
    }
}

fn one() -> u32 {
    1
}

fn invalid(message: &str) -> LLMError {
    LLMError::Config(format!("declared pricing: {message}"))
}

impl DeclaredPricing {
    /// Validates the exact non-secret endpoint and model binding before consumers capture the adapter.
    pub(crate) fn validate(&self, model: &str, endpoint: &str) -> Result<(), LLMError> {
        let url = reqwest::Url::parse(endpoint).map_err(|_| invalid("invalid endpoint"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || url.cannot_be_a_base()
            || self.schedule_id.trim().is_empty()
            || self.model != model
            || normalize_endpoint(&self.endpoint)? != normalize_endpoint(endpoint)?
            || self.server_input_token_limit == 0
            || self.server_output_token_limit == 0
            || self.max_internal_attempts == 0
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(invalid(
                "model, endpoint, schedule or server limits are invalid",
            ));
        }
        self.maximum()?;
        Ok(())
    }

    /// Charges each class at its declared maximum rate; discounts cannot lower the reserved upper bound.
    fn cost(&self, input: u32, output: u32) -> Result<u64, LLMError> {
        let numerator = u128::from(input) * u128::from(self.input_micro_usd_per_million)
            + u128::from(output) * u128::from(self.output_micro_usd_per_million);
        let rounded = numerator.div_ceil(1_000_000);
        u64::try_from(rounded)
            .ok()
            .and_then(|cost| cost.checked_add(self.request_fee_micro_usd))
            .ok_or_else(|| invalid("cost overflow"))
    }

    /// All internal attempts are covered even when upstream exposes only the final attempt's usage.
    fn maximum(&self) -> Result<u64, LLMError> {
        self.cost(
            self.server_input_token_limit,
            self.server_output_token_limit,
        )?
        .checked_mul(u64::from(self.max_internal_attempts))
        .ok_or_else(|| invalid("reservation overflow"))
    }

    /// Identifies the actual billing target independently of its schedule, so a new alias cannot evade the run's price pin.
    pub(crate) fn provider_identity(&self, provider: &str) -> String {
        let provider = accounting_adapter(provider);
        let endpoint =
            normalize_endpoint(&self.endpoint).expect("installed endpoint was validated");
        let bytes = serde_json::to_vec(&("declared-adapter.v1", provider, &self.model, endpoint))
            .expect("provider binding contains scalar fields");
        format!("first-party.{provider}.{:x}", Sha256::digest(bytes))
    }

    /// Pins numeric contract content as well as its human-readable version without exposing endpoint credentials.
    pub(crate) fn identity(&self, provider: &str) -> String {
        let provider = accounting_adapter(provider);
        let mut canonical = self.clone();
        canonical.endpoint =
            normalize_endpoint(&canonical.endpoint).expect("installed endpoint was validated");
        let bytes = serde_json::to_vec(&("declared-adapter.v1", provider, canonical))
            .expect("pricing contains only serializable scalar fields");
        format!("first-party.{provider}.{:x}", Sha256::digest(bytes))
    }

    /// Uses the actual effective configuration; unsupported billable services and arbitrary body overrides fail closed.
    pub(crate) fn quote(
        &self,
        provider: &str,
        request: &ProviderRequest<'_>,
        default: &LLMConfig,
    ) -> Result<ProviderCostBound, LLMError> {
        let config = request.config.unwrap_or(default);
        let output = config
            .max_tokens
            .ok_or_else(|| invalid("explicit output limit required"))?;
        if output == 0 || output > self.server_output_token_limit {
            return Err(invalid("output exceeds the audited server contract"));
        }
        if let Some(effort) = config.reasoning_effort.as_deref().or_else(|| {
            config
                .extra
                .get("reasoning_effort")
                .and_then(|v| v.as_str())
        }) && !matches!(
            effort.to_ascii_lowercase().as_str(),
            "low" | "medium" | "high"
        ) {
            return Err(invalid("unsupported reasoning effort"));
        }
        for (key, value) in &config.extra {
            match key.as_str() {
                "declared_pricing"
                    if serde_json::from_value::<Self>(value.clone()).ok().as_ref()
                        == Some(self) => {}
                "timeout_seconds"
                | "reasoning"
                | "reasoning_effort"
                | "reasoning_budget_tokens"
                | "resilient"
                | "resilient_attempts"
                | "resilient_base_delay_ms"
                | "resilient_max_delay_ms"
                | "resilient_jitter" => {}
                _ => return Err(invalid("unbounded provider-specific option")),
            }
        }
        // Server ceilings include hidden reasoning, native history, schemas and framing, regardless of prompt length.
        let reasoning_budget = config.reasoning_budget_tokens.map(u64::from).or_else(|| {
            config
                .extra
                .get("reasoning_budget_tokens")
                .and_then(|v| v.as_u64())
        });
        if reasoning_budget.is_some_and(|budget| budget > u64::from(self.server_output_token_limit))
        {
            return Err(invalid("reasoning exceeds the audited server contract"));
        }
        if config.extra.get("resilient").and_then(|v| v.as_bool()) == Some(true) {
            let attempts = config
                .extra
                .get("resilient_attempts")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| invalid("internal retries need an explicit total-attempt bound"))?;
            if attempts == 0 || attempts > u64::from(self.max_internal_attempts) {
                return Err(invalid("internal retries exceed the declared bound"));
            }
        }
        Ok(ProviderCostBound {
            provider_identity: self.provider_identity(provider),
            pricing_identity: self.identity(provider),
            request_id: request.request_id.into(),
            max_micro_usd: self.maximum()?,
        })
    }

    /// Missing, partial or retry-only usage cannot release any of the complete bound.
    pub(crate) fn settle(
        &self,
        provider: &str,
        bound: &ProviderCostBound,
        response: &LLMResponse,
    ) -> Result<Option<ProviderCostSettlement>, LLMError> {
        if bound.provider_identity != self.provider_identity(provider)
            || bound.pricing_identity != self.identity(provider)
            || bound.max_micro_usd != self.maximum()?
        {
            return Err(invalid("settlement binding changed"));
        }
        if !self.complete_usage || self.max_internal_attempts != 1 {
            return Ok(None);
        }
        if provider == "google"
            && response.metadata.get("_ai_agents_complete_priced_usage")
                != Some(&serde_json::Value::Bool(true))
        {
            return Ok(None);
        }
        let Some(usage) = response.usage else {
            return Ok(None);
        };
        if usage.prompt_tokens > self.server_input_token_limit
            || usage.completion_tokens > self.server_output_token_limit
            || usage.prompt_tokens.checked_add(usage.completion_tokens) != Some(usage.total_tokens)
        {
            return Err(invalid(
                "reported usage exceeds or contradicts the admitted contract",
            ));
        }
        Ok(Some(ProviderCostSettlement {
            bound: bound.clone(),
            charged_micro_usd: self.cost(usage.prompt_tokens, usage.completion_tokens)?,
        }))
    }
}
