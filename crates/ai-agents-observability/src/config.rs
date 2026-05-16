use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ObservabilityError;

/// Top-level YAML and Rust configuration for metrics, privacy, aggregation, cost, and export behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Enables all observability collection when true.
    #[serde(default)]
    pub enabled: bool,
    /// Controls which latency events are recorded.
    #[serde(default)]
    pub latency: LatencyConfig,
    /// Controls token counting and estimation behavior.
    #[serde(default)]
    pub tokens: TokenConfig,
    /// Controls cost estimation and pricing lookup behavior.
    #[serde(default)]
    pub cost: CostConfig,
    /// Controls how the language dimension is resolved from runtime context.
    #[serde(default)]
    pub language: LanguageConfig,
    /// Controls the configured aggregate metrics table.
    #[serde(default)]
    pub aggregation: AggregationConfig,
    /// Controls raw text retention, hashing, truncation, and redaction.
    #[serde(default)]
    pub privacy: PrivacyConfig,
    /// Controls file export formats and paths.
    #[serde(default)]
    pub export: ExportConfig,
    /// Controls event queue and raw event buffer limits.
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
    /// Validates bounds that would otherwise make aggregation or buffering unusable.
    pub fn validate(&self) -> Result<(), ObservabilityError> {
        if self.aggregation.window_size == 0 {
            return Err(ObservabilityError::Config(
                "observability.aggregation.window_size must be greater than zero".to_string(),
            ));
        }
        if self.buffer.event_buffer == 0 {
            return Err(ObservabilityError::Config(
                "observability.buffer.event_buffer must be greater than zero".to_string(),
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

    /// Loads cost.pricing_file and merges it with inline pricing.
    pub fn with_pricing_file_loaded(
        mut self,
        base_dir: Option<&Path>,
    ) -> Result<Self, ObservabilityError> {
        let Some(path) = self.cost.pricing_file.clone() else {
            return Ok(self);
        };
        let resolved = resolve_pricing_path(&path, base_dir);
        let content = std::fs::read_to_string(&resolved).map_err(ObservabilityError::Io)?;
        let mut file_pricing = parse_pricing_file(&resolved, &content)?;
        let inline_pricing = std::mem::take(&mut self.cost.pricing);
        for (key, value) in inline_pricing {
            file_pricing.insert(key.to_lowercase(), value);
        }
        self.cost.pricing = file_pricing;
        Ok(self)
    }
}

/// Latency switches for categories that can produce duration events.
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

/// Token counting settings for provider usage and fallback estimation.
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

/// Cost estimation settings and model pricing sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    /// Enables cost estimation when token usage is available.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Inline pricing keyed by model or provider/model.
    #[serde(default)]
    pub pricing: HashMap<String, ModelPricing>,
    /// Optional JSON or YAML pricing file loaded by the runtime builder or Rust helper.
    #[serde(default)]
    pub pricing_file: Option<String>,
    /// Controls how unknown model prices are represented.
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

/// Per-thousand-token price for one model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Price for one thousand input tokens in USD.
    pub input_per_1k: f64,
    /// Price for one thousand output tokens in USD.
    pub output_per_1k: f64,
}

/// Behavior when token usage exists but no configured price matches the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnknownPricePolicy {
    #[default]
    Omit,
    Zero,
    Error,
}

/// Language dimension lookup rules for reports and aggregations.
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

/// Grouping and rolling-window settings for aggregate metrics.
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

/// Supported fields that can be used as aggregation dimensions.
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
    /// Returns the stable dimension key used in reports and CSV output.
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

/// Privacy controls for raw payload retention, hashes, truncation, and redaction.
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

/// File export settings for reports, aggregates, raw events, and Prometheus metrics.
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

/// Export formats supported by ObservabilityManager::export.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    #[default]
    Json,
    Csv,
    Jsonl,
    Prometheus,
}

/// Raw event file shape used when raw event export is enabled.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RawEventsFormat {
    #[default]
    Jsonl,
    Json,
}

/// Bounded queue and raw event retention limits.
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

fn resolve_pricing_path(path: &str, base_dir: Option<&Path>) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else if let Some(base_dir) = base_dir {
        base_dir.join(path)
    } else {
        path
    }
}

fn parse_pricing_file(
    path: &Path,
    content: &str,
) -> Result<HashMap<String, ModelPricing>, ObservabilityError> {
    let parsed: HashMap<String, ModelPricing> = match path.extension().and_then(|ext| ext.to_str())
    {
        Some("json") => serde_json::from_str(content).map_err(ObservabilityError::Serialization)?,
        Some("yaml") | Some("yml") | None => serde_yaml::from_str(content).map_err(|error| {
            ObservabilityError::Config(format!(
                "failed to parse observability.cost.pricing_file '{}': {}",
                path.display(),
                error
            ))
        })?,
        Some(other) => {
            return Err(ObservabilityError::Config(format!(
                "unsupported observability.cost.pricing_file extension '{}': {}",
                other,
                path.display()
            )));
        }
    };
    Ok(parsed
        .into_iter()
        .map(|(key, value)| (key.to_lowercase(), value))
        .collect())
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

    #[test]
    fn pricing_file_loads_and_inline_overrides() {
        let dir = std::env::temp_dir().join(format!(
            "ai_agents_observability_pricing_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("pricing.yaml"),
            "openai/test:\n  input_per_1k: 0.1\n  output_per_1k: 0.2\nopenai/other:\n  input_per_1k: 1.0\n  output_per_1k: 2.0\n",
        )
        .unwrap();

        let mut config = ObservabilityConfig::default();
        config.cost.pricing_file = Some("pricing.yaml".to_string());
        config.cost.pricing.insert(
            "openai/test".to_string(),
            ModelPricing {
                input_per_1k: 0.3,
                output_per_1k: 0.4,
            },
        );

        let loaded = config.with_pricing_file_loaded(Some(&dir)).unwrap();
        let overridden = loaded.cost.pricing.get("openai/test").unwrap();
        assert_eq!(overridden.input_per_1k, 0.3);
        assert_eq!(overridden.output_per_1k, 0.4);
        assert!(loaded.cost.pricing.contains_key("openai/other"));

        let _ = std::fs::remove_dir_all(dir);
    }
}
