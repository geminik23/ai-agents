//! Memory hook event types

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCompressEvent {
    pub messages_compressed: usize,
    pub summary_tokens_before: u32,
    pub summary_tokens_after: u32,
    pub compression_ratio: f64,
}

impl MemoryCompressEvent {
    pub fn new(
        messages_compressed: usize,
        summary_tokens_before: u32,
        summary_tokens_after: u32,
    ) -> Self {
        let ratio = if summary_tokens_before > 0 {
            summary_tokens_after as f64 / summary_tokens_before as f64
        } else {
            0.0
        };
        Self {
            messages_compressed,
            summary_tokens_before,
            summary_tokens_after,
            compression_ratio: ratio,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvictEvent {
    pub reason: EvictionReason,
    pub messages_evicted: usize,
    pub importance_scores: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EvictionReason {
    TokenBudgetExceeded,
    MessageCountExceeded,
    StateTransition,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudgetEvent {
    pub component: String,
    pub used_tokens: u32,
    pub budget_tokens: u32,
    pub usage_percent: f64,
}

impl MemoryBudgetEvent {
    pub fn new(component: impl Into<String>, used_tokens: u32, budget_tokens: u32) -> Self {
        let usage_percent = if budget_tokens > 0 {
            (used_tokens as f64 / budget_tokens as f64) * 100.0
        } else {
            0.0
        };
        Self {
            component: component.into(),
            used_tokens,
            budget_tokens,
            usage_percent,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactExtractedEvent {
    pub category: String,
    pub content: String,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_compress_event() {
        let event = MemoryCompressEvent::new(10, 1000, 200);
        assert_eq!(event.messages_compressed, 10);
        assert_eq!(event.summary_tokens_before, 1000);
        assert_eq!(event.summary_tokens_after, 200);
        assert!((event.compression_ratio - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_memory_compress_event_zero_before() {
        let event = MemoryCompressEvent::new(5, 0, 100);
        assert_eq!(event.compression_ratio, 0.0);
    }

    #[test]
    fn test_memory_budget_event() {
        let event = MemoryBudgetEvent::new("summary", 800, 1000);
        assert_eq!(event.component, "summary");
        assert_eq!(event.used_tokens, 800);
        assert_eq!(event.budget_tokens, 1000);
        assert!((event.usage_percent - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_budget_event_zero_budget() {
        let event = MemoryBudgetEvent::new("test", 100, 0);
        assert_eq!(event.usage_percent, 0.0);
    }

    #[test]
    fn test_eviction_reason_serialize() {
        let reason = EvictionReason::TokenBudgetExceeded;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"token_budget_exceeded\"");

        let reason = EvictionReason::StateTransition;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"state_transition\"");
    }

    #[test]
    fn test_fact_extracted_event() {
        let event = FactExtractedEvent {
            category: "preference".to_string(),
            content: "User prefers dark mode".to_string(),
            confidence: 0.95,
        };
        assert_eq!(event.category, "preference");
        assert!(event.confidence > 0.9);
    }
}
