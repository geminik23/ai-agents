//! Raw, presence-preserving autonomy configuration shared without runtime dependencies.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

use super::CompletionGate;

/// A positive decimal USD amount stored without floating-point rounding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsdAmount(String);

impl UsdAmount {
    /// Parses at most six decimal places into checked fixed-point microdollars.
    pub fn parse(value: &str) -> Result<Self, String> {
        let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
        if whole.is_empty()
            || !whole.bytes().all(|byte| byte.is_ascii_digit())
            || fraction.len() > 6
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
            || value.ends_with('.')
        {
            return Err("cost must be a positive decimal string with at most six places".into());
        }
        let whole: u64 = whole.parse().map_err(|_| "cost is out of range")?;
        let fraction_len = fraction.len();
        let fraction: u64 = if fraction.is_empty() {
            0
        } else {
            fraction.parse().map_err(|_| "cost is out of range")?
        };
        let micro = whole
            .checked_mul(1_000_000)
            .and_then(|value| value.checked_add(fraction * 10_u64.pow((6 - fraction_len) as u32)))
            .ok_or("cost is out of range")?;
        if micro == 0 {
            return Err("cost must be positive".into());
        }
        Ok(Self(value.to_string()))
    }

    /// Returns the declared amount in exact microdollars.
    pub fn micro_usd(&self) -> u64 {
        let (whole, fraction) = self.0.split_once('.').unwrap_or((&self.0, ""));
        let whole: u64 = whole.parse().expect("validated whole amount");
        let fraction_len = fraction.len();
        let fraction: u64 = if fraction.is_empty() {
            0
        } else {
            fraction.parse().expect("validated fraction")
        };
        whole * 1_000_000 + fraction * 10_u64.pow((6 - fraction_len) as u32)
    }
}

impl Serialize for UsdAmount {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for UsdAmount {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Value::String(value) = Value::deserialize(deserializer)? else {
            return Err(serde::de::Error::custom(
                "cost must be a quoted decimal string",
            ));
        };
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

/// Scope ownership is independent of execution strategy and human review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutonomyMode {
    Disabled,
    UntilComplete,
    StateUntilExit,
    SkillUntilComplete,
}

/// An opt-in controller execution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStrategy {
    ModelLed,
    Lifecycle,
    Skill,
}

/// Human review cannot replace tool policy or other host authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewPolicy {
    Inherit,
    RequireApproval,
}

/// Optional strategy override; omission preserves the selection rules.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<ExecutionStrategy>,
}

/// Review options remain independent of task scope and tool permissions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupervisionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_changes: Option<ReviewPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_changes: Option<ReviewPolicy>,
}

/// Named validation schedule, not a new tool execution source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSchedule {
    EachCycle,
    StageEnd,
    Completion,
}

/// A host-registered or built-in validator binding and its opaque, validated extension data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationCheck {
    pub id: String,
    pub adapter: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<ValidationSchedule>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
}

/// Optional built-in diagnostics configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fail_on_error: Option<bool>,
}

/// Optional built-in judge configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass_threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Vec<String>>,
}

/// Raw validation lists retain an explicit empty replacement.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commands: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checks: Option<Vec<ValidationCheck>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<DiagnosticsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeConfig>,
}

/// Stagnation intervention; it never changes objective or authority by itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StagnationAction {
    Continue,
    Replan,
    AskUser,
    Stop,
    Fail,
}

/// Bounded response to non-advancing progress observations.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StagnationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cycles_without_progress: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<StagnationAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_replans: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_exhausted: Option<StagnationAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_unavailable: Option<StagnationAction>,
}

/// Optional canonical-todo requirements and one named progress observation binding.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub todo_tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_todos: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_active_todo: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforce_for_tools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_create_todo_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_open_todos: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stagnation: Option<StagnationConfig>,
}

/// A linear controller stage with optional validation and a bounded fix target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutonomyStage {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_before_tools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_tools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<CompletionGate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commands: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_on_failure: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_failure_stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fix_cycles: Option<u32>,
}

/// Action for a premature model final response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrematureFinishAction {
    Continue,
    AskUser,
    SummarizeAndStop,
    Fail,
}

/// Bounded response when gates reject a proposed final answer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrematureFinishConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<PrematureFinishAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_continuations: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

/// Action when another managed invocation is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitAction {
    SummarizeAndStop,
    Fail,
    AskUser,
    PauseRun,
}

/// Hard-limit outcome configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<LimitAction>,
}

/// Allowed interaction response; it cannot bypass an existing approval requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionAction {
    PauseRun,
    Reject,
    AutoIfPolicy,
    AskEachTime,
}

/// Autonomy-specific interaction behavior without tool authority.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutonomyHitlConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_approval_required: Option<InteractionAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_user_question: Option<InteractionAction>,
}

/// Persistence options do not grant durable task-store capability.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutonomyPersistenceConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_after_each_step: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_on_start: Option<bool>,
}

/// Raw profile settings preserve omission, false, and empty lists until after merging.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutonomyProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<AutonomyMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supervision: Option<SupervisionConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_active_time_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_wall_time_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_llm_calls: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_command_calls: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cost_usd: Option<UsdAmount>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_declared_write_paths: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<ProgressConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle: Option<Vec<AutonomyStage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<CompletionGate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub premature_finish: Option<PrematureFinishConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_limit: Option<LimitConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hitl: Option<AutonomyHitlConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistence: Option<AutonomyPersistenceConfig>,
}

/// Agent-wide defaults and named profiles share one flat external YAML mapping.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AutonomyConfig {
    pub defaults: AutonomyProfile,
    pub default_profile: Option<String>,
    pub profiles: HashMap<String, AutonomyProfile>,
}

/// State and skill overrides are flattened externally without accepting unknown framework keys.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AutonomyOverride {
    pub profile: Option<String>,
    pub settings: AutonomyProfile,
}

// Parses a flattened raw profile only after removing the reserved keys from the mapping.
fn parse_flat_profile<D: serde::de::Error>(value: Value) -> Result<AutonomyProfile, D> {
    serde_yaml::from_value(value).map_err(D::custom)
}

impl<'de> Deserialize<'de> for AutonomyConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Value::Mapping(mut fields) = Value::deserialize(deserializer)? else {
            return Err(serde::de::Error::custom("autonomy must be a mapping"));
        };
        let default_profile = fields
            .remove(Value::String("default_profile".into()))
            .map(serde_yaml::from_value)
            .transpose()
            .map_err(serde::de::Error::custom)?;
        let profiles = fields
            .remove(Value::String("profiles".into()))
            .map(serde_yaml::from_value)
            .transpose()
            .map_err(serde::de::Error::custom)?
            .unwrap_or_default();
        Ok(Self {
            defaults: parse_flat_profile(Value::Mapping(fields))?,
            default_profile,
            profiles,
        })
    }
}

impl Serialize for AutonomyConfig {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Value::Mapping(mut fields) =
            serde_yaml::to_value(&self.defaults).map_err(serde::ser::Error::custom)?
        else {
            return Err(serde::ser::Error::custom(
                "autonomy profile must be a mapping",
            ));
        };
        if let Some(profile) = &self.default_profile {
            fields.insert(
                Value::String("default_profile".into()),
                Value::String(profile.clone()),
            );
        }
        if !self.profiles.is_empty() {
            fields.insert(
                Value::String("profiles".into()),
                serde_yaml::to_value(&self.profiles).map_err(serde::ser::Error::custom)?,
            );
        }
        Value::Mapping(fields).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AutonomyOverride {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Value::Mapping(mut fields) = Value::deserialize(deserializer)? else {
            return Err(serde::de::Error::custom(
                "autonomy override must be a mapping",
            ));
        };
        let profile = fields
            .remove(Value::String("profile".into()))
            .map(serde_yaml::from_value)
            .transpose()
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            profile,
            settings: parse_flat_profile(Value::Mapping(fields))?,
        })
    }
}

impl Serialize for AutonomyOverride {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Value::Mapping(mut fields) =
            serde_yaml::to_value(&self.settings).map_err(serde::ser::Error::custom)?
        else {
            return Err(serde::ser::Error::custom(
                "autonomy profile must be a mapping",
            ));
        };
        if let Some(profile) = &self.profile {
            fields.insert(
                Value::String("profile".into()),
                Value::String(profile.clone()),
            );
        }
        Value::Mapping(fields).serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_profiles_and_overrides_round_trip_without_dropping_presence() {
        let yaml = "enabled: true\ndefault_profile: research\nprofiles:\n  research:\n    enabled: false\n    lifecycle: []\n    validation: {commands: [], checks: []}\n    progress:\n      require_todos: false\n      adapter: host.sources\n      config: {domain_specific: [a, b]}\n    completion: {todos_done: true}\n";
        let config: AutonomyConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.profiles["research"].enabled, Some(false));
        assert_eq!(config.profiles["research"].lifecycle, Some(vec![]));
        let encoded = serde_yaml::to_value(&config).unwrap();
        assert_eq!(encoded, serde_yaml::from_str::<Value>(yaml).unwrap());
        assert_eq!(
            serde_yaml::from_value::<AutonomyConfig>(encoded).unwrap(),
            config
        );

        let override_: AutonomyOverride =
            serde_yaml::from_str("profile: research\nenabled: false\ncompletion: {state: done}\n")
                .unwrap();
        assert_eq!(override_.settings.enabled, Some(false));
        assert_eq!(
            serde_yaml::from_value::<AutonomyOverride>(serde_yaml::to_value(&override_).unwrap())
                .unwrap(),
            override_
        );
        assert!(
            serde_yaml::from_str::<AutonomyOverride>("profile: research\nenabled: nope").is_err()
        );
    }

    #[test]
    fn strict_framework_keys_keep_only_named_adapter_config_open() {
        for yaml in [
            "enabeld: true",
            "profiles: {research: {max_tuns: 10}}",
            "progress: {require_todoss: true}",
            "validation: {checks: [{id: check, adapter: host.check, bad_key: true}]}",
        ] {
            assert!(
                serde_yaml::from_str::<AutonomyConfig>(yaml).is_err(),
                "{yaml}"
            );
        }
        let config: AutonomyConfig = serde_yaml::from_str(
            "progress: {adapter: host.check, config: {arbitrary_domain_key: true}}",
        )
        .unwrap();
        assert_eq!(
            config.defaults.progress.unwrap().config.unwrap()["arbitrary_domain_key"],
            true
        );
    }

    #[test]
    fn priced_amounts_are_positive_strings_with_checked_precision() {
        let amount: UsdAmount = serde_yaml::from_str("\"2.000001\"").unwrap();
        assert_eq!(amount.micro_usd(), 2_000_001);
        for invalid in ["0", "-1", "2.0000001", "1.", "18446744073710", "2.00"] {
            let value = if invalid == "2.00" {
                invalid.to_string()
            } else {
                format!("\"{invalid}\"")
            };
            assert!(
                serde_yaml::from_str::<UsdAmount>(&value).is_err(),
                "{value}"
            );
        }
    }
}
