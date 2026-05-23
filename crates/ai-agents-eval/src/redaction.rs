use serde::{Deserialize, Serialize};
use serde_json::Value;

/// String value that records whether redaction was applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedString {
    /// Stored value, redacted when requested by settings.
    pub value: String,
    /// Whether the value was redacted for output.
    #[serde(default)]
    pub redacted: bool,
}

impl RedactedString {
    pub fn plain(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            redacted: false,
        }
    }

    pub fn redacted(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            redacted: true,
        }
    }
}

pub fn redact_text(text: &str, enabled: bool, max_chars: usize) -> RedactedString {
    if !enabled {
        return RedactedString::plain(text.to_string());
    }
    if max_chars == 0 {
        return RedactedString::redacted("[redacted]");
    }
    let mut chars = text.chars();
    let preview: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        RedactedString::redacted(format!("{}...", preview))
    } else {
        RedactedString::redacted(preview)
    }
}

pub fn redact_value(value: Value, enabled: bool, max_chars: usize) -> Value {
    if !enabled {
        return value;
    }
    match value {
        Value::String(text) => Value::String(redact_text(&text, true, max_chars).value),
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| redact_value(value, true, max_chars))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, redact_value(value, true, max_chars)))
                .collect(),
        ),
        other => other,
    }
}
