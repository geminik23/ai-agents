use super::{Completion, Error, ToolCall, Usage};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;

/// The model reported alongside a Chat response or usage frame. A request model is never receipt evidence.
#[derive(Deserialize)]
pub(super) struct Reported<T> {
    pub model: Option<String>,
    #[serde(flatten)]
    pub envelope: Envelope<T>,
}

#[derive(Deserialize)]
pub(super) struct Envelope<T> {
    pub error: Option<Value>,
    pub choices: Vec<T>,
    pub usage: Option<UsageReceipt>,
}

#[derive(Deserialize)]
pub(super) struct UsageReceipt {
    pub cost: Option<Value>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    prompt_tokens_details: Option<std::collections::BTreeMap<String, Option<u32>>>,
    completion_tokens_details: Option<std::collections::BTreeMap<String, Option<u32>>>,
}

impl UsageReceipt {
    pub fn complete(self) -> Option<Usage> {
        let (prompt_tokens, completion_tokens, total_tokens) = (
            self.prompt_tokens?,
            self.completion_tokens?,
            self.total_tokens?,
        );
        (prompt_tokens.checked_add(completion_tokens) == Some(total_tokens)).then_some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            prompt_tokens_details: self.prompt_tokens_details,
            completion_tokens_details: self.completion_tokens_details,
        })
    }
}

#[derive(Default, Deserialize)]
pub(super) struct FunctionData {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Default, Deserialize)]
pub(super) struct CallData {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: FunctionData,
}

pub(super) fn finish_calls(
    calls: impl IntoIterator<Item = CallData>,
    reason: &str,
) -> Result<Vec<ToolCall>, Error> {
    let mut ids = HashSet::new();
    let mut result = Vec::new();
    for call in calls {
        if !matches!(reason, "stop" | "tool_calls")
            || call.id.is_empty()
            || call.function.name.is_empty()
            || call.kind.as_deref().is_some_and(|v| v != "function")
            || !ids.insert(call.id.clone())
        {
            return Err(Error::Protocol(
                "unexpected, ambiguous or incomplete function call",
            ));
        }
        let arguments: Value = serde_json::from_str(&call.function.arguments)
            .map_err(|_| Error::Protocol("invalid function arguments"))?;
        if !arguments.is_object() {
            return Err(Error::Protocol("function arguments must be an object"));
        }
        result.push(ToolCall {
            id: call.id,
            name: call.function.name,
            arguments,
        });
    }
    if result.is_empty() && reason == "tool_calls" {
        return Err(Error::Protocol("missing function calls"));
    }
    Ok(result)
}

#[derive(Deserialize)]
pub(super) struct Choice {
    message: Assistant,
    finish_reason: String,
}

#[derive(Deserialize)]
struct Assistant {
    function_call: Option<Value>,
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<CallData>>,
    refusal: Option<String>,
}

pub(super) fn complete(response: Envelope<Choice>) -> Result<Completion, Error> {
    if response.error.is_some() {
        return Err(Error::Protocol("provider returned an error"));
    }
    let [choice]: [Choice; 1] = response
        .choices
        .try_into()
        .map_err(|_| Error::Protocol("expected one choice"))?;
    if choice.message.role != "assistant"
        || choice.message.function_call.is_some()
        || matches!(choice.finish_reason.as_str(), "" | "error")
    {
        return Err(Error::Protocol("invalid assistant choice"));
    }
    if choice.message.refusal.is_some() {
        return Err(Error::Refused);
    }
    let calls = finish_calls(
        choice.message.tool_calls.unwrap_or_default(),
        &choice.finish_reason,
    )?;
    Ok(Completion {
        text: choice.message.content.unwrap_or_default(),
        finish_reason: choice.finish_reason,
        tool_calls: calls,
        usage: response.usage.and_then(UsageReceipt::complete),
    })
}
