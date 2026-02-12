//! Token budget management for memory

use serde::{Deserialize, Serialize};

// !!

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTokenBudget {
    #[serde(default = "default_total_budget")]
    pub total: u32,

    #[serde(default)]
    pub allocation: TokenAllocation,

    #[serde(default)]
    pub overflow_strategy: OverflowStrategy,

    #[serde(default = "default_warn_percent")]
    pub warn_at_percent: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAllocation {
    #[serde(default = "default_summary_tokens")]
    pub summary: u32,

    #[serde(default = "default_recent_tokens")]
    pub recent_messages: u32,

    #[serde(default = "default_facts_tokens")]
    pub facts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OverflowStrategy {
    #[default]
    TruncateOldest,
    SummarizeMore,
    Error,
}

fn default_total_budget() -> u32 {
    4096
}

fn default_summary_tokens() -> u32 {
    1024
}

fn default_recent_tokens() -> u32 {
    2048
}

fn default_facts_tokens() -> u32 {
    512
}

fn default_warn_percent() -> u8 {
    80
}

impl Default for TokenAllocation {
    fn default() -> Self {
        Self {
            summary: default_summary_tokens(),
            recent_messages: default_recent_tokens(),
            facts: default_facts_tokens(),
        }
    }
}

impl Default for MemoryTokenBudget {
    fn default() -> Self {
        Self {
            total: default_total_budget(),
            allocation: TokenAllocation::default(),
            overflow_strategy: OverflowStrategy::default(),
            warn_at_percent: default_warn_percent(),
        }
    }
}

impl MemoryTokenBudget {
    pub fn new(total: u32) -> Self {
        Self {
            total,
            ..Default::default()
        }
    }

    pub fn with_allocation(mut self, allocation: TokenAllocation) -> Self {
        self.allocation = allocation;
        self
    }

    pub fn with_overflow_strategy(mut self, strategy: OverflowStrategy) -> Self {
        self.overflow_strategy = strategy;
        self
    }

    pub fn with_warn_at_percent(mut self, percent: u8) -> Self {
        self.warn_at_percent = percent.min(100);
        self
    }

    pub fn warn_threshold(&self) -> u32 {
        (self.total as f64 * (self.warn_at_percent as f64 / 100.0)) as u32
    }

    pub fn is_over_warn_threshold(&self, used: u32) -> bool {
        used >= self.warn_threshold()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryBudgetState {
    pub total_tokens_used: u32,
    pub summary_tokens: u32,
    pub recent_tokens: u32,
    pub facts_tokens: u32,
    pub last_warning_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl MemoryBudgetState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn usage_percent(&self, budget: &MemoryTokenBudget) -> f64 {
        if budget.total == 0 {
            return 0.0;
        }
        (self.total_tokens_used as f64 / budget.total as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_token_budget_default() {
        let budget = MemoryTokenBudget::default();
        assert_eq!(budget.total, 4096);
        assert_eq!(budget.allocation.summary, 1024);
        assert_eq!(budget.allocation.recent_messages, 2048);
        assert_eq!(budget.allocation.facts, 512);
        assert_eq!(budget.warn_at_percent, 80);
    }

    #[test]
    fn test_warn_threshold() {
        let budget = MemoryTokenBudget::new(1000).with_warn_at_percent(75);
        assert_eq!(budget.warn_threshold(), 750);
        assert!(!budget.is_over_warn_threshold(700));
        assert!(budget.is_over_warn_threshold(750));
        assert!(budget.is_over_warn_threshold(800));
    }

    #[test]
    fn test_budget_state_usage() {
        let budget = MemoryTokenBudget::new(1000);
        let mut state = MemoryBudgetState::new();
        state.total_tokens_used = 500;
        assert!((state.usage_percent(&budget) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_overflow_strategy_deserialize() {
        let yaml = r#"truncate_oldest"#;
        let strategy: OverflowStrategy = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(strategy, OverflowStrategy::TruncateOldest);

        let yaml = r#"summarize_more"#;
        let strategy: OverflowStrategy = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(strategy, OverflowStrategy::SummarizeMore);
    }
}
