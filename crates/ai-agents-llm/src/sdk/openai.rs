//! Documented constraints for the official OpenAI protocol endpoints.
//! Model IDs stay open. Entries describe known restrictions on individual request features.
use crate::sdk::{ChatRequest, Error, Message};
use serde::Deserialize;
use serde_json::Value;
use std::sync::LazyLock;

pub(crate) const API_BASE: &str = "https://api.openai.com/v1/";

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

static CHAT_RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("openai_chat_models.json"))
        .expect("invalid bundled OpenAI Chat rules")
});

#[derive(Clone, Copy)]
pub(crate) enum Protocol {
    Chat,
    Responses,
}

impl Protocol {
    fn rule(self, model: &str) -> Option<&'static Rule> {
        let rules = match self {
            Self::Chat => &CHAT_RULES,
            Self::Responses => &RESPONSES_RULES,
        };
        rules
            .iter()
            .find(|rule| rule.models.iter().any(|id| id == model))
    }

    pub(crate) fn allows_tools(self, model: &str, effort: Option<&str>) -> bool {
        self.rule(model)
            .is_none_or(|rule| rule.tools.allows(effort.unwrap_or(&rule.default_effort)))
    }
}

pub(crate) fn validate(request: &ChatRequest, native_tools: bool) -> Result<(), Error> {
    let Some(rule) = Protocol::Chat.rule(&request.model) else {
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

static RESPONSES_RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("openai_responses_models.json"))
        .expect("invalid Responses rules")
});

pub(super) fn validate_responses(r: &super::responses::ResponsesRequest) -> Result<(), Error> {
    if let Some(rule) = Protocol::Responses.rule(&r.model) {
        let effort = r
            .reasoning_effort
            .as_deref()
            .unwrap_or(&rule.default_effort);
        if !rule.tools.allows(effort) && (!r.functions.is_empty() || r.tool_choice.is_some()) {
            return Err(Error::Config(
                "unsupported Responses native tools for the selected model and effort".into(),
            ));
        }
        if (effort == "none" && !rule.allows_none)
            || (!rule.sampling.allows(effort) && (r.temperature.is_some() || r.top_p.is_some()))
        {
            return Err(Error::Config(
                "unsupported sampling or effort for the selected Responses model".into(),
            ));
        }
    }
    Ok(())
}
