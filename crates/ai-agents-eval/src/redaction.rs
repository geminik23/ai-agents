use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedString {
    pub value: String,
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
