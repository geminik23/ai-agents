pub mod aggregator;
pub mod config;
pub mod context;
pub mod cost;
pub mod event;
pub mod export;
pub mod hooks;
pub mod manager;
pub mod redaction;
pub mod report;
pub mod span;
pub mod wrappers;

pub use aggregator::{AggregatedMetrics, CostStats, LatencyStats, MetricsAggregator, TokenStats};
pub use config::{
    AggregationConfig, AggregationDimension, BufferConfig, CostConfig, ExportConfig, ExportFormat,
    LanguageConfig, LatencyConfig, ModelPricing, ObservabilityConfig, PrivacyConfig,
    RawEventsFormat, TokenConfig, UnknownPricePolicy,
};
pub use context::{
    SpanContext, current_observation_context, with_observation_context, with_observation_purpose,
};
pub use cost::CostEstimator;
pub use event::{
    CostEstimate, CostSource, EventStatus, EventType, ObservationError, ObservationEvent,
    ObservationPurpose, ObservationTokenUsage, TokenUsageSource,
};
pub use export::ExportResult;
pub use hooks::ObservabilityHooks;
pub use manager::{ObservabilityManager, new_session_id, resolve_language_from_context};
pub use redaction::{Redactor, stable_hash, truncate_chars};
pub use report::{CostBreakdown, ObservabilityReport, ReportSummary, TokenBreakdown};
pub use span::SpanGuard;
pub use wrappers::{ObservedLLMProvider, ObservedTool};

pub type Result<T> = std::result::Result<T, ObservabilityError>;

#[derive(Debug, thiserror::Error)]
pub enum ObservabilityError {
    #[error("observability config error: {0}")]
    Config(String),
    #[error("observability IO error: {0}")]
    Io(std::io::Error),
    #[error("observability serialization error: {0}")]
    Serialization(serde_json::Error),
}

impl From<serde_json::Error> for ObservabilityError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

impl From<ObservabilityError> for ai_agents_core::AgentError {
    fn from(error: ObservabilityError) -> Self {
        ai_agents_core::AgentError::Other(error.to_string())
    }
}
