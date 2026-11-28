mod chat_adapter;
mod google;
pub mod local;
mod messages_adapter;
mod ollama_adapter;
mod ollama_local;
pub(crate) mod openai;
mod openrouter;
mod pricing;
mod responses_adapter;
pub mod unified;
pub use pricing::DeclaredPricing;

pub use local::{LocalModelAdapter, LocalModelProvider, ModelInfo};
pub use unified::{ProviderBuilder, ProviderType, UnifiedLLMProvider};

// Prompt-tool history identifies the tool in user-visible text when the wire protocol has no native result.
fn prompt_tool_result(name: Option<&str>, content: &str) -> String {
    format!("[{} result]: {content}", name.unwrap_or("tool"))
}

pub(super) fn sdk_error(e: crate::sdk::Error) -> ai_agents_core::LLMError {
    use crate::sdk;
    use ai_agents_core::LLMError;
    match e {
        sdk::Error::Config(msg) => LLMError::Config(msg),
        sdk::Error::Protocol(msg) => LLMError::Serialization(msg.into()),
        sdk::Error::Refused => LLMError::ContentFiltered("provider refused completion".into()),
        sdk::Error::Transport => LLMError::Network("provider transport failed".into()),
        sdk::Error::Http(429) => LLMError::RateLimit { retry_after: None },
        sdk::Error::Http(code) => LLMError::API {
            message: format!("provider endpoint returned HTTP {code}"),
            status: Some(code),
        },
    }
}
