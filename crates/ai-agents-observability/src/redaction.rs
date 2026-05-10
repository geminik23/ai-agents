use crate::config::PrivacyConfig;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Redactor {
    config: PrivacyConfig,
    keys: HashSet<String>,
}

impl Redactor {
    pub fn new(config: PrivacyConfig) -> Self {
        let keys = config
            .redact_keys
            .iter()
            .map(|key| key.to_lowercase())
            .collect();
        Self { config, keys }
    }

    pub fn redact_value(&self, value: &Value) -> Value {
        let mut value = value.clone();
        self.redact_recursive(&mut value);
        for path in &self.config.redact_paths {
            redact_path(&mut value, path);
        }
        value
    }

    pub fn redact_text(&self, text: &str) -> Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "length".to_string(),
            Value::from(text.chars().count() as u64),
        );
        if self.config.hash_inputs {
            map.insert("hash".to_string(), Value::from(stable_hash(text)));
        }
        if self.config.max_text_chars > 0 {
            map.insert(
                "text".to_string(),
                Value::from(truncate_chars(text, self.config.max_text_chars)),
            );
        }
        Value::Object(map)
    }

    fn redact_recursive(&self, value: &mut Value) {
        match value {
            Value::Object(map) => {
                let keys: Vec<String> = map.keys().cloned().collect();
                for key in keys {
                    if self.keys.contains(&key.to_lowercase()) {
                        map.insert(key, redacted_marker());
                    } else if let Some(child) = map.get_mut(&key) {
                        self.redact_recursive(child);
                    }
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.redact_recursive(item);
                }
            }
            Value::String(text) => {
                if self.config.max_text_chars == 0 {
                    *value = Value::Object(
                        [(
                            "length".to_string(),
                            Value::from(text.chars().count() as u64),
                        )]
                        .into_iter()
                        .collect(),
                    );
                } else {
                    *text = truncate_chars(text, self.config.max_text_chars);
                }
            }
            _ => {}
        }
    }
}

fn redacted_marker() -> Value {
    serde_json::json!({"redacted": true})
}

fn redact_path(value: &mut Value, path: &str) {
    let parts: Vec<&str> = path.split('.').filter(|part| !part.is_empty()).collect();
    if parts.is_empty() {
        return;
    }
    redact_path_parts(value, &parts);
}

fn redact_path_parts(value: &mut Value, parts: &[&str]) {
    if parts.is_empty() {
        *value = redacted_marker();
        return;
    }
    match value {
        Value::Object(map) => {
            if let Some(child) = map.get_mut(parts[0]) {
                redact_path_parts(child, &parts[1..]);
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_path_parts(item, parts);
            }
        }
        _ => {}
    }
}

pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        text.to_string()
    }
}

pub fn stable_hash(text: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_nested_keys() {
        let redactor = Redactor::new(PrivacyConfig::default());
        let value = serde_json::json!({"headers": {"authorization": "Bearer secret"}});
        let redacted = redactor.redact_value(&value);
        assert_eq!(
            redacted["headers"]["authorization"],
            serde_json::json!({"redacted": true})
        );
    }

    #[test]
    fn truncates_on_char_boundaries() {
        let text = "안녕하세요🙂world";
        let truncated = truncate_chars(text, 3);
        assert_eq!(truncated, "안녕하...");
    }

    #[test]
    fn hash_is_stable() {
        assert_eq!(stable_hash("abc"), stable_hash("abc"));
        assert_ne!(stable_hash("abc"), stable_hash("abd"));
    }
}
