use crate::sdk::openai::Protocol;
pub(super) use crate::sdk::openai::validate;
use ai_agents_core::LLMConfig;
use serde_json::Value;

// Capability queries use configured defaults. Actual dispatch validates the merged per-call settings.
pub(super) fn allows_tools(protocol: Protocol, model: &str, config: &LLMConfig) -> bool {
    let effort = config
        .reasoning_effort
        .as_deref()
        .or_else(|| match protocol {
            Protocol::Chat => config
                .extra
                .get("reasoning_effort")
                .and_then(Value::as_str)
                .or_else(|| {
                    ["extra_body", "provider_options"]
                        .iter()
                        .find_map(|k| config.extra.get(*k)?.get("reasoning_effort")?.as_str())
                }),
            Protocol::Responses => config
                .extra
                .get("provider_options")?
                .get("reasoning")?
                .get("effort")?
                .as_str(),
        });
    protocol.allows_tools(model, effort)
}
