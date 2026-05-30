use crate::config::PrivacyConfig;
use crate::event::{EventType, ObservationError, ObservationEvent};
use serde_json::Value;
use std::collections::HashSet;

/// Privacy helper that redacts keys, paths, errors, tags, and raw text.
#[derive(Debug, Clone)]
pub struct Redactor {
    config: PrivacyConfig,
    keys: HashSet<String>,
}

impl Redactor {
    /// Creates a redactor with lowercase key lookup for configured redact keys.
    pub fn new(config: PrivacyConfig) -> Self {
        let keys = config
            .redact_keys
            .iter()
            .map(|key| key.to_lowercase())
            .collect();
        Self { config, keys }
    }

    /// Redacts a JSON value by configured keys, dotted paths, and text rules.
    pub fn redact_value(&self, value: &Value) -> Value {
        let mut value = value.clone();
        self.redact_recursive(&mut value);
        for path in &self.config.redact_paths {
            redact_path(&mut value, path);
        }
        value
    }

    /// Redacts every event surface that can carry user or domain text.
    pub fn redact_event(&self, mut event: ObservationEvent) -> ObservationEvent {
        if let Some(payload) = event.payload.take() {
            event.payload = Some(self.redact_value(&payload));
        }
        if let Some(error) = event.error.take() {
            event.error = Some(self.redact_error(error));
        }
        for value in event.tags.values_mut() {
            *value = self.redact_text_to_string(value);
        }
        for (key, value) in event.dimensions.iter_mut() {
            if !is_safe_dimension(key) {
                *value = self.redact_text_to_string(value);
            }
        }
        if let EventType::HitlApproval { trigger } = &mut event.event_type {
            *trigger = self.redact_text_to_string(trigger);
        }
        event
    }

    /// Redacts the error message while preserving the error kind.
    pub fn redact_error(&self, mut error: ObservationError) -> ObservationError {
        error.message = self.redact_text_to_string(&error.message);
        error
    }

    /// Converts text into a safe string summary for tags and errors.
    pub fn redact_text_to_string(&self, text: &str) -> String {
        if self.config.max_text_chars > 0 {
            truncate_chars(text, self.config.max_text_chars)
        } else if self.config.hash_inputs {
            format!("length:{} hash:{}", text.chars().count(), stable_hash(text))
        } else {
            format!("length:{}", text.chars().count())
        }
    }

    /// Converts text into a structured safe summary with length and optional hash.
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
                    let key_lower = key.to_lowercase();
                    if self.keys.contains(&key_lower) {
                        map.insert(key, redacted_marker());
                    } else if key_lower == "hash"
                        || key_lower == "length"
                        || key_lower == "redacted"
                    {
                        continue;
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
                    let mut map = serde_json::Map::new();
                    map.insert(
                        "length".to_string(),
                        Value::from(text.chars().count() as u64),
                    );
                    if self.config.hash_inputs {
                        map.insert("hash".to_string(), Value::from(stable_hash(text)));
                    }
                    *value = Value::Object(map);
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

fn is_safe_dimension(key: &str) -> bool {
    matches!(
        key,
        "agent"
            | "actor"
            | "session"
            | "purpose"
            | "status"
            | "provider"
            | "model"
            | "alias"
            | "language"
            | "state"
            | "tool"
            | "skill"
            | "orchestration_pattern"
            | "branch_status"
            | "runtime.branch_status"
            | "winner"
            | "runtime.winner"
            | "optimization"
            | "runtime.optimization"
            | "commit_behavior"
            | "runtime.commit_behavior"
            | "background"
            | "runtime.background"
            | "maintenance"
            | "runtime.maintenance"
            | "maintenance_stage"
            | "runtime.maintenance_stage"
            | "await_before_next_turn"
            | "runtime.await_before_next_turn"
            | "maintenance_mode"
            | "runtime.maintenance_mode"
    )
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

/// Truncates by Unicode scalar values rather than byte offsets.
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

/// Produces a stable non-cryptographic hash for correlation.
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
        let text = "안녕하세요 world";
        let truncated = truncate_chars(text, 3);
        assert_eq!(truncated, "안녕하...");
    }

    #[test]
    fn hash_is_stable() {
        assert_eq!(stable_hash("abc"), stable_hash("abc"));
        assert_ne!(stable_hash("abc"), stable_hash("abd"));
    }

    #[test]
    fn redacts_event_tags_errors_and_payloads() {
        use crate::event::{EventStatus, ObservationPurpose};
        use chrono::Utc;
        use std::collections::HashMap;

        let redactor = Redactor::new(PrivacyConfig::default());
        let mut tags = HashMap::new();
        tags.insert("reason".to_string(), "사용자 secret token 값".to_string());
        let event = ObservationEvent {
            trace_id: "trace".to_string(),
            span_id: "span".to_string(),
            parent_span_id: None,
            turn_id: "turn".to_string(),
            agent_id: "agent".to_string(),
            actor_id: None,
            session_id: None,
            event_type: EventType::HitlApproval {
                trigger: "tool with private args".to_string(),
            },
            purpose: ObservationPurpose::HitlLocalization,
            status: EventStatus::Error,
            timestamp: Utc::now(),
            duration_ms: 1,
            tokens: None,
            cost: None,
            error: Some(ObservationError::new("tool", "비밀 응답 secret")),
            dimensions: HashMap::new(),
            tags,
            payload: Some(
                serde_json::json!({"authorization": "Bearer secret", "text": "こんにちはsecret"}),
            ),
        };

        let redacted = redactor.redact_event(event);
        assert!(!redacted.error.unwrap().message.contains("비밀"));
        assert!(!redacted.tags["reason"].contains("사용자"));
        assert_eq!(
            redacted.payload.unwrap()["authorization"],
            serde_json::json!({"redacted": true})
        );
    }
}
