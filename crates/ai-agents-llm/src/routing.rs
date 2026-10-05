//! Typed selection of existing aliases for auxiliary model responsibilities.

use std::sync::Arc;

use ai_agents_core::{LLMError, LLMProvider};
use serde::{Deserialize, Deserializer, Serialize};

macro_rules! routing_schema {
    ($( $group:ident : $config:ident { $( $field:ident => $role:ident ),+ } ),+ $(,)?) => {
        $(
            #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
            #[serde(deny_unknown_fields)]
            pub struct $config {
                #[serde(default, skip_serializing_if = "Option::is_none")]
                pub default: Option<String>,
                $(#[serde(default, skip_serializing_if = "Option::is_none")]
                pub $field: Option<String>,)+
            }
        )+

        /// Fixed auxiliary roles; configuration does not enable their owning features.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum LLMRole { $($( $role, )+)+ }

        /// Agent-local hierarchy of literal registry aliases.
        #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct RouterRolesConfig {
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub default: Option<String>,
            $(#[serde(default, skip_serializing_if = "Option::is_none")]
            pub $group: Option<$config>,)+
        }

        impl LLMRole {
            pub const ALL: &'static [Self] = &[$($(Self::$role,)+)+];

            pub fn as_path(self) -> &'static str {
                match self { $($(Self::$role => concat!(stringify!($group), ".", stringify!($field)),)+)+ }
            }

            pub fn group(self) -> &'static str {
                match self { $($(Self::$role => stringify!($group),)+)+ }
            }
        }

        impl RouterRolesConfig {
            /// Selects an alias without retrying missing registrations or provider failures.
            pub fn select<'a>(&'a self, role: LLMRole, local: Option<&'a str>, main: &'a str) -> (&'a str, LLMSelectionSource) {
                if let Some(alias) = local { return (alias, LLMSelectionSource::Local); }
                let (leaf, group) = match role {
                    $($(LLMRole::$role => self.$group.as_ref().map(|config| (config.$field.as_deref(), config.default.as_deref())).unwrap_or((None, None)),)+)+
                };
                if let Some(alias) = leaf { return (alias, LLMSelectionSource::Role); }
                if let Some(alias) = group { return (alias, LLMSelectionSource::Group); }
                if let Some(alias) = self.default.as_deref() { return (alias, LLMSelectionSource::Router); }
                (main, LLMSelectionSource::Default)
            }

            /// Lists only explicit tree aliases, including inactive roles, for preflight.
            pub fn configured_aliases(&self) -> Vec<(&'static str, &str)> {
                let mut aliases = Vec::new();
                if let Some(alias) = self.default.as_deref() { aliases.push(("llm.router.default", alias)); }
                $(if let Some(config) = &self.$group {
                    if let Some(alias) = config.default.as_deref() { aliases.push((concat!("llm.router.", stringify!($group), ".default"), alias)); }
                    $(if let Some(alias) = config.$field.as_deref() { aliases.push((concat!("llm.router.", stringify!($group), ".", stringify!($field)), alias)); })+
                })+
                aliases
            }

            /// Rejects empty aliases without changing their exact registry keys.
            pub fn validate(&self) -> Result<(), LLMError> {
                for (path, alias) in self.configured_aliases() {
                    if alias.trim().is_empty() { return Err(LLMError::Config(format!("Invalid {path}: alias must not be empty"))); }
                }
                Ok(())
            }
        }
    }
}

routing_schema! {
    state: StateRouterConfig { transition => StateTransition, extract => StateExtract },
    skills: SkillsRouterConfig { selection => SkillsSelection },
    tools: ToolsRouterConfig { condition => ToolsCondition },
    process: ProcessRouterConfig { detect => ProcessDetect, extract => ProcessExtract, sanitize => ProcessSanitize, transform => ProcessTransform, validate => ProcessValidate },
    disambiguation: DisambiguationRouterConfig { detection => DisambiguationDetection, skip => DisambiguationSkip, clarification => DisambiguationClarification, parse => DisambiguationParse, confirmation => DisambiguationConfirmation, confirmation_parse => DisambiguationConfirmationParse, response => DisambiguationResponse },
    reasoning: ReasoningRouterConfig { selection => ReasoningSelection, planning => ReasoningPlanning, reflection_decision => ReasoningReflectionDecision, reflection_evaluation => ReasoningReflectionEvaluation },
    memory: MemoryRouterConfig { summarize => MemorySummarize, merge => MemoryMerge, facts => MemoryFacts, relationships => MemoryRelationships },
    context: ContextRouterConfig { summarize => ContextSummarize },
    orchestration: OrchestrationRouterConfig { routing => OrchestrationRouting, handoff => OrchestrationHandoff, speaker => OrchestrationSpeaker, consensus => OrchestrationConsensus, synthesis => OrchestrationSynthesis, vote => OrchestrationVote, tiebreak => OrchestrationTiebreak, summary => OrchestrationSummary },
    hitl: HitlRouterConfig { message => HitlMessage },
    spawner: SpawnerRouterConfig { generation => SpawnerGeneration, repair => SpawnerRepair },
    web: WebRouterConfig { extract => WebExtract },
    evaluation: EvaluationRouterConfig { response => EvaluationResponse, facts => EvaluationFacts },
}

/// Scalar selection preserves legacy behavior; even an empty mapping opts into hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum RouterSelector {
    Alias(String),
    Hierarchical(Box<RouterRolesConfig>),
}

impl<'de> Deserialize<'de> for RouterSelector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(alias) = value.as_str() {
            return Ok(Self::Alias(alias.to_string()));
        }
        validate_tree_shape(&value).map_err(serde::de::Error::custom)?;
        let config: RouterRolesConfig =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        config.validate().map_err(serde::de::Error::custom)?;
        Ok(Self::Hierarchical(Box::new(config)))
    }
}

// Check paths before typed deserialization so mapping errors retain their YAML location.
fn validate_tree_shape(value: &serde_json::Value) -> Result<(), String> {
    let root = value
        .as_object()
        .ok_or("Invalid llm.router: expected alias or mapping")?;
    for (group, value) in root {
        if group == "default" {
            validate_alias_value("llm.router.default", value)?;
            continue;
        }
        if !LLMRole::ALL.iter().any(|role| role.group() == group) {
            return Err(format!("Invalid llm.router.{group}: unknown group"));
        }
        if value.is_null() {
            continue;
        }
        let fields = value
            .as_object()
            .ok_or_else(|| format!("Invalid llm.router.{group}: expected mapping"))?;
        for (field, value) in fields {
            let path = format!("{group}.{field}");
            if field != "default" && !LLMRole::ALL.iter().any(|role| role.as_path() == path) {
                return Err(format!("Invalid llm.router.{path}: unknown field"));
            }
            validate_alias_value(&format!("llm.router.{path}"), value)?;
        }
    }
    Ok(())
}

fn validate_alias_value(path: &str, value: &serde_json::Value) -> Result<(), String> {
    if value.is_null() || value.as_str().is_some_and(|alias| !alias.trim().is_empty()) {
        Ok(())
    } else {
        Err(format!("Invalid {path}: expected nonempty alias or null"))
    }
}

/// Identifies the configuration boundary that selected an alias, not execution recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LLMSelectionSource {
    Local,
    Role,
    Group,
    Router,
    Default,
}

/// A resolved role retains its exact alias and existing provider handle.
#[derive(Clone)]
pub struct ResolvedRoleLLM {
    pub role: LLMRole,
    pub alias: String,
    pub source: LLMSelectionSource,
    pub provider: Arc<dyn LLMProvider>,
}

/// Deserializes a present string while letting serde defaults preserve omitted fields.
pub fn deserialize_present_alias<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::String(alias) => Ok(Some(alias)),
        _ => Err(serde::de::Error::custom(
            "expected a present alias string; null is not supported",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_roles_inherit_and_preserve_sources() {
        let mut tree = RouterRolesConfig::default();
        assert_eq!(LLMRole::ALL.len(), 39);
        for role in LLMRole::ALL {
            assert_eq!(
                tree.select(*role, None, "main"),
                ("main", LLMSelectionSource::Default)
            );
            assert_eq!(
                tree.select(*role, Some("router"), "main"),
                ("router", LLMSelectionSource::Local)
            );
        }
        tree.default = Some("fast".into());
        for role in LLMRole::ALL {
            assert_eq!(
                tree.select(*role, None, "main"),
                ("fast", LLMSelectionSource::Router)
            );
        }
        tree.state = Some(StateRouterConfig {
            default: Some("group".into()),
            transition: Some("leaf".into()),
            extract: None,
        });
        assert_eq!(
            tree.select(LLMRole::StateTransition, None, "main"),
            ("leaf", LLMSelectionSource::Role)
        );
        assert_eq!(
            tree.select(LLMRole::StateExtract, None, "main"),
            ("group", LLMSelectionSource::Group)
        );
    }

    #[test]
    fn strict_tree_and_empty_mapping_round_trip() {
        let empty: RouterSelector = serde_json::from_str("{}").unwrap();
        assert!(matches!(empty, RouterSelector::Hierarchical(_)));
        assert_eq!(serde_json::to_string(&empty).unwrap(), "{}");
        for input in [
            r#"{"state":{"transiton":"fast"}}"#,
            r#"{"state":"fast"}"#,
            r#"{"state":{"transition":" "}}"#,
            "false",
        ] {
            assert!(
                serde_json::from_str::<RouterSelector>(input).is_err(),
                "{input}"
            );
        }
        let scalar: RouterSelector = serde_json::from_str(r#""router""#).unwrap();
        assert_eq!(scalar, RouterSelector::Alias("router".into()));
    }
}
