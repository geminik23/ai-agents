use crate::config::{CostConfig, ModelPricing, UnknownPricePolicy};
use crate::event::{CostEstimate, CostSource, ObservationTokenUsage};

/// Converts token usage into USD estimates using configured model pricing.
#[derive(Debug, Clone)]
pub struct CostEstimator {
    config: CostConfig,
}

impl CostEstimator {
    /// Creates an estimator and normalizes pricing keys for lookup.
    pub fn new(mut config: CostConfig) -> Self {
        config.pricing = config
            .pricing
            .into_iter()
            .map(|(key, value)| (key.to_lowercase(), value))
            .collect();
        Self { config }
    }

    /// Estimates cost for one LLM event or returns None when pricing is omitted.
    pub fn estimate(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
        usage: Option<&ObservationTokenUsage>,
    ) -> Option<CostEstimate> {
        if !self.config.enabled {
            return None;
        }
        let usage = usage?;
        let pricing = self.find_pricing(provider, model);
        match pricing {
            Some(pricing) => {
                let input_usd = usage.input_tokens as f64 / 1000.0 * pricing.input_per_1k;
                let output_usd = usage.output_tokens as f64 / 1000.0 * pricing.output_per_1k;
                Some(CostEstimate {
                    input_usd,
                    output_usd,
                    total_usd: input_usd + output_usd,
                    source: CostSource::Configured,
                })
            }
            None => match self.config.unknown_price_policy {
                UnknownPricePolicy::Omit => None,
                UnknownPricePolicy::Zero => Some(CostEstimate {
                    input_usd: 0.0,
                    output_usd: 0.0,
                    total_usd: 0.0,
                    source: CostSource::Unknown,
                }),
                UnknownPricePolicy::Error => None,
            },
        }
    }

    fn find_pricing(&self, provider: Option<&str>, model: Option<&str>) -> Option<ModelPricing> {
        let model = model?;
        if let Some(provider) = provider {
            let key = format!("{}/{}", provider, model).to_lowercase();
            if let Some(pricing) = self.config.pricing.get(&key) {
                return Some(*pricing);
            }
        }
        self.config.pricing.get(&model.to_lowercase()).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::TokenUsageSource;
    use std::collections::HashMap;

    #[test]
    fn unknown_price_is_omitted_by_default() {
        let estimator = CostEstimator::new(CostConfig::default());
        let usage = ObservationTokenUsage::new(1000, 1000, TokenUsageSource::Provider);
        assert!(
            estimator
                .estimate(Some("openai"), Some("unknown"), Some(&usage))
                .is_none()
        );
    }

    #[test]
    fn configured_price_is_used() {
        let mut pricing = HashMap::new();
        pricing.insert(
            "openai/test".to_string(),
            ModelPricing {
                input_per_1k: 0.1,
                output_per_1k: 0.2,
            },
        );
        let estimator = CostEstimator::new(CostConfig {
            pricing,
            ..CostConfig::default()
        });
        let usage = ObservationTokenUsage::new(1000, 2000, TokenUsageSource::Provider);
        let cost = estimator
            .estimate(Some("openai"), Some("test"), Some(&usage))
            .unwrap();
        assert!((cost.total_usd - 0.5).abs() < f64::EPSILON);
    }
}
