use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ObservabilityError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub latency: LatencyConfig,
    #[serde(default)]
    pub tokens: TokenConfig,
    #[serde(default)]
    pub cost: CostConfig,
    #[serde(default)]
    pub language: LanguageConfig,
    #[serde(default)]
    pub aggregation: AggregationConfig,
    #[serde(default)]
    pub privacy: PrivacyConfig,
    #[serde(default)]
    pub export: ExportConfig,
    #[serde(default)]
    pub buffer: BufferConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            latency: LatencyConfig::default(),
            tokens: TokenConfig::default(),
            cost: CostConfig::default(),
            language: LanguageConfig::default(),
            aggregation: AggregationConfig::default(),
            privacy: PrivacyConfig::default(),
            export: ExportConfig::default(),
            buffer: BufferConfig::default(),
        }
    }
}

impl ObservabilityConfig {
    pub fn validate(&self) -> Result<(), ObservabilityError> {
        if self.aggregation.window_size == 0 {
            return Err(ObservabilityError::Config(
                "observability.aggregation.window_size must be greater than zero".to_string(),
            ));
        }
        for percentile in &self.aggregation.percentiles {
            if !(0.0..=1.0).contains(percentile) {
                return Err(ObservabilityError::Config(format!(
                    "observability.aggregation.percentiles value {} is outside 0.0..=1.0",
                    percentile
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    #[serde(default = "default_true")]
    pub track_llm: bool,
    #[serde(default = "default_true")]
    pub track_tools: bool,
    #[serde(default = "default_true")]
    pub track_skills: bool,
    #[serde(default = "default_true")]
    pub track_orchestration: bool,
    #[serde(default = "default_true")]
    pub track_hitl: bool,
    #[serde(default)]
    pub detailed_breakdown: bool,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            track_llm: true,
            track_tools: true,
            track_skills: true,
            track_orchestration: true,
            track_hitl: true,
            detailed_breakdown: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    #[serde(default = "default_true")]
    pub count_input: bool,
    #[serde(default = "default_true")]
    pub count_output: bool,
    #[serde(default = "default_true")]
    pub estimate_when_missing: bool,
    #[serde(default)]
    pub breakdown_by_component: bool,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            count_input: true,
            count_output: true,
            estimate_when_missing: true,
            breakdown_by_component: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub pricing: HashMap<String, ModelPricing>,
    #[serde(default)]
    pub pricing_file: Option<String>,
    #[serde(default)]
    pub unknown_price_policy: UnknownPricePolicy,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pricing: HashMap::new(),
            pricing_file: None,
            unknown_price_policy: UnknownPricePolicy::Omit,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_per_1k: f64,
    pub output_per_1k: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnknownPricePolicy {
    #[default]
    Omit,
    Zero,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    #[serde(default = "default_language_paths")]
    pub paths: Vec<String>,
    #[serde(default = "default_unknown")]
    pub fallback: String,
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            paths: default_language_paths(),
            fallback: default_unknown(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    #[serde(default = "default_dimensions")]
    pub dimensions: Vec<AggregationDimension>,
    #[serde(default = "default_percentiles")]
    pub percentiles: Vec<f64>,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            dimensions: default_dimensions(),
            percentiles: default_percentiles(),
            window_size: default_window_size(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AggregationDimension {
    Agent,
    Actor,
    Model,
    Provider,
    Alias,
    Purpose,
    Language,
    State,
    Tool,
    Skill,
    OrchestrationPattern,
    Status,
    Custom(String),
}

impl AggregationDimension {
    pub fn key(&self) -> String {
        match self {
            Self::Agent => "agent".to_string(),
            Self::Actor => "actor".to_string(),
            Self::Model => "model".to_string(),
            Self::Provider => "provider".to_string(),
            Self::Alias => "alias".to_string(),
            Self::Purpose => "purpose".to_string(),
            Self::Language => "language".to_string(),
            Self::State => "state".to_string(),
            Self::Tool => "tool".to_string(),
            Self::Skill => "skill".to_string(),
            Self::OrchestrationPattern => "orchestration_pattern".to_string(),
            Self::Status => "status".to_string(),
            Self::Custom(name) => format!("custom:{}", name),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    #[serde(default)]
    pub include_prompts: bool,
    #[serde(default)]
    pub include_responses: bool,
    #[serde(default)]
    pub include_tool_args: bool,
    #[serde(default)]
    pub include_tool_outputs: bool,
    #[serde(default)]
    pub max_text_chars: usize,
    #[serde(default = "default_true")]
    pub hash_inputs: bool,
    #[serde(default = "default_redact_keys")]
    pub redact_keys: Vec<String>,
    #[serde(default = "default_redact_paths")]
    pub redact_paths: Vec<String>,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            include_prompts: false,
            include_responses: false,
            include_tool_args: false,
            include_tool_outputs: false,
            max_text_chars: 0,
            hash_inputs: true,
            redact_keys: default_redact_keys(),
            redact_paths: default_redact_paths(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    #[serde(default = "default_export_formats")]
    pub formats: Vec<ExportFormat>,
    #[serde(default = "default_export_path")]
    pub path: String,
    #[serde(default)]
    pub write_raw_events: bool,
    #[serde(default = "default_true")]
    pub write_report: bool,
    #[serde(default)]
    pub raw_events_format: RawEventsFormat,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            formats: default_export_formats(),
            path: default_export_path(),
            write_raw_events: false,
            write_report: true,
            raw_events_format: RawEventsFormat::Jsonl,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    #[default]
    Json,
    Csv,
    Jsonl,
    Prometheus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RawEventsFormat {
    #[default]
    Jsonl,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    #[serde(default = "default_event_buffer")]
    pub event_buffer: usize,
    #[serde(default = "default_raw_event_limit")]
    pub raw_event_limit: usize,
    #[serde(default = "default_true")]
    pub drop_on_full: bool,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            event_buffer: default_event_buffer(),
            raw_event_limit: default_raw_event_limit(),
            drop_on_full: true,
        }
    }
}

pub fn default_true() -> bool {
    true
}

fn default_unknown() -> String {
    "unknown".to_string()
}

fn default_language_paths() -> Vec<String> {
    vec![
        "detected_language".to_string(),
        "input.language".to_string(),
        "user.language".to_string(),
        "context.user.language".to_string(),
    ]
}

fn default_dimensions() -> Vec<AggregationDimension> {
    vec![AggregationDimension::Model, AggregationDimension::Purpose]
}

fn default_percentiles() -> Vec<f64> {
    vec![0.5, 0.9, 0.95, 0.99]
}

fn default_window_size() -> usize {
    1000
}

fn default_redact_keys() -> Vec<String> {
    vec![
        "api_key".to_string(),
        "authorization".to_string(),
        "token".to_string(),
        "password".to_string(),
        "secret".to_string(),
    ]
}

fn default_redact_paths() -> Vec<String> {
    vec![
        "actor_facts".to_string(),
        "relationship_memory".to_string(),
        "persona.secrets".to_string(),
    ]
}

fn default_export_formats() -> Vec<ExportFormat> {
    vec![ExportFormat::Json]
}

fn default_export_path() -> String {
    "./observability_data/".to_string()
}

fn default_event_buffer() -> usize {
    4096
}

fn default_raw_event_limit() -> usize {
    10_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_privacy_safe() {
        let config = ObservabilityConfig::default();
        assert!(!config.enabled);
        assert!(!config.privacy.include_prompts);
        assert!(!config.privacy.include_responses);
        assert!(!config.privacy.include_tool_args);
        assert_eq!(config.privacy.max_text_chars, 0);
    }

    #[test]
    fn deserializes_minimal_enabled_config() {
        let yaml = r#"
observability:
  enabled: true
  aggregation:
    dimensions: [agent, model, purpose, language]
"#;
        #[derive(Deserialize)]
        struct Wrapper {
            observability: ObservabilityConfig,
        }
        let parsed: Wrapper = serde_yaml::from_str(yaml).unwrap();
        assert!(parsed.observability.enabled);
        assert_eq!(parsed.observability.aggregation.dimensions.len(), 4);
    }

    #[test]
    fn validation_rejects_bad_percentile() {
        let mut config = ObservabilityConfig::default();
        config.aggregation.percentiles = vec![1.2];
        assert!(config.validate().is_err());
    }
}
