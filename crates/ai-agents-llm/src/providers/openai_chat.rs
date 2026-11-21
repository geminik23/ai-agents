//! Documented constraints for the official OpenAI Chat Completions endpoint.
//! Model IDs stay open. Entries describe known restrictions on individual request features.
use super::ProviderType;
use crate::sdk::{ChatRequest, Error, Message};
use ai_agents_core::LLMConfig;
use serde::Deserialize;
use serde_json::Value;
use std::sync::LazyLock;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Availability {
    Always,
    Never,
    WithoutReasoning,
}

impl Availability {
    fn allows(&self, effort: &str) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::WithoutReasoning => effort == "none",
        }
    }
}

#[derive(Deserialize)]
struct Rule {
    models: Vec<String>,
    sampling: Availability,
    tools: Availability,
    default_effort: String,
    allows_none: bool,
}

static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("openai_chat_models.json"))
        .expect("invalid bundled OpenAI Chat rules")
});

fn rule(model: &str) -> Option<&'static Rule> {
    RULES.iter().find(|r| r.models.iter().any(|id| id == model))
}

// Capability reporting uses the same effort sources as the request adapter.
// Option collisions and invalid values are rejected when constructing the actual request.
pub(super) fn allows_tools(provider: ProviderType, model: &str, config: &LLMConfig) -> bool {
    if provider != ProviderType::OpenAI {
        return true;
    }
    let Some(rule) = rule(model) else { return true };
    let effort = config
        .reasoning_effort
        .as_deref()
        .or_else(|| config.extra.get("reasoning_effort").and_then(Value::as_str))
        .or_else(|| {
            ["extra_body", "provider_options"]
                .iter()
                .find_map(|key| config.extra.get(*key)?.get("reasoning_effort")?.as_str())
        })
        .unwrap_or(&rule.default_effort);
    rule.tools.allows(effort)
}

pub(crate) fn validate(request: &ChatRequest, native_tools: bool) -> Result<(), Error> {
    let Some(rule) = rule(&request.model) else {
        return Ok(());
    };
    let effort = request
        .reasoning_effort
        .as_deref()
        .or_else(|| {
            request
                .provider_options
                .get("reasoning_effort")
                .and_then(Value::as_str)
        })
        .unwrap_or(&rule.default_effort);
    if effort == "none" && !rule.allows_none {
        return Err(Error::Config(format!(
            "{} does not support reasoning_effort none",
            request.model
        )));
    }
    if !rule.sampling.allows(effort) {
        for (field, set) in [
            ("temperature", request.temperature.is_some()),
            ("top_p", request.top_p.is_some()),
            ("logprobs", false),
            ("top_logprobs", false),
        ] {
            if set || request.provider_options.contains_key(field) {
                return Err(Error::Config(format!(
                    "{} Chat Completions does not support {field} with reasoning_effort {effort}",
                    request.model
                )));
            }
        }
    }
    if !rule.tools.allows(effort)
        && (native_tools
            || !request.functions.is_empty()
            || request.tool_choice.is_some()
            || request.messages.iter().any(|message| {
                matches!(message, Message::ToolResult { .. })
                    || matches!(message, Message::Assistant { calls, .. } if !calls.is_empty())
            }))
    {
        return Err(Error::Config(format!(
            "{} requires the Responses API for native tool calls with reasoning_effort {effort}",
            request.model
        )));
    }
    Ok(())
}
