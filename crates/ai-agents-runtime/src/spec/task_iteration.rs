//! Source-aware task iteration policy alongside the stable, materialized agent schema.

use super::AgentSpec;
use ai_agents_core::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Controls which already-configured iteration limits apply to autonomous model loops.
/// Ordinary chat always uses its existing numeric limits, regardless of this policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskIterationPolicy {
    root_configured: bool,
    reasoning_configured: bool,
    state_reasoning: BTreeMap<String, bool>,
}

impl TaskIterationPolicy {
    /// Conservatively treats every numeric value from a typed or legacy specification as configured.
    pub fn configured() -> Self {
        Self {
            root_configured: true,
            reasoning_configured: true,
            state_reasoning: BTreeMap::new(),
        }
    }

    /// Removes implicit task-loop caps without changing ordinary chat or other task budgets.
    pub fn unbounded() -> Self {
        Self {
            root_configured: false,
            reasoning_configured: false,
            state_reasoning: BTreeMap::new(),
        }
    }

    /// Selects whether the current root numeric value limits task model loops.
    pub fn with_root_limit(mut self, configured: bool) -> Self {
        self.root_configured = configured;
        self
    }

    /// Selects whether the current agent-level reasoning numeric value limits task model loops.
    pub fn with_reasoning_limit(mut self, configured: bool) -> Self {
        self.reasoning_configured = configured;
        self
    }

    /// Selects whether one full state path's replacement reasoning configuration limits task model loops.
    pub fn with_state_reasoning_limit(mut self, path: impl Into<String>, configured: bool) -> Self {
        self.state_reasoning.insert(path.into(), configured);
        self
    }

    // Raw presence is captured before serde materializes defaults; missing state entries remain conservative for typed replacements.
    fn from_source(value: &serde_yaml::Value) -> Result<Self> {
        let fields = value
            .as_mapping()
            .ok_or_else(|| AgentError::InvalidSpec("agent must be a mapping".into()))?;
        let key = |name: &str| serde_yaml::Value::String(name.to_string());
        let present = |value: Option<&serde_yaml::Value>| {
            value
                .and_then(serde_yaml::Value::as_mapping)
                .is_some_and(|map| map.contains_key(key("max_iterations")))
        };
        let mut policy = Self::unbounded();
        policy.root_configured = fields.contains_key(key("max_iterations"));
        policy.reasoning_configured = present(fields.get(key("reasoning")));
        fn states(
            value: &serde_yaml::Value,
            prefix: &str,
            out: &mut BTreeMap<String, bool>,
        ) -> Result<()> {
            let Some(entries) = value.as_mapping() else {
                return Ok(());
            };
            for (name, definition) in entries {
                let name = name
                    .as_str()
                    .ok_or_else(|| AgentError::InvalidSpec("state name must be a string".into()))?;
                let path = if prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{prefix}.{name}")
                };
                let Some(fields) = definition.as_mapping() else {
                    continue;
                };
                if let Some(reasoning) = fields.get(serde_yaml::Value::String("reasoning".into()))
                    && let Some(reasoning) = reasoning.as_mapping()
                {
                    out.insert(
                        path.clone(),
                        reasoning.contains_key(serde_yaml::Value::String("max_iterations".into())),
                    );
                }
                if let Some(children) = fields.get(serde_yaml::Value::String("states".into())) {
                    states(children, &path, out)?;
                }
            }
            Ok(())
        }
        if let Some(root_states) = fields
            .get(key("states"))
            .and_then(serde_yaml::Value::as_mapping)
            && let Some(entries) = root_states.get(key("states"))
        {
            states(entries, "", &mut policy.state_reasoning)?;
        }
        Ok(policy)
    }

    // Typed state replacements have no source witness and must retain their numeric restriction.
    pub(crate) fn limit_for_state(&self, state: Option<&str>, has_state_reasoning: bool) -> bool {
        if has_state_reasoning {
            state
                .and_then(|path| self.state_reasoning.get(path))
                .copied()
                .unwrap_or(true)
        } else {
            self.reasoning_configured
        }
    }

    pub(crate) fn root_configured(&self) -> bool {
        self.root_configured
    }

    pub(crate) fn forget_state_sources(&mut self) {
        self.state_reasoning.clear();
    }
}

/// A parsed stable specification and the source-presence information needed by autonomous loops.
#[derive(Debug, Clone)]
pub struct PreparedAgentSpec {
    spec: AgentSpec,
    policy: TaskIterationPolicy,
}

impl PreparedAgentSpec {
    /// Strictly parses source without guessing whether a materialized default was explicit.
    pub fn from_yaml_strict(source: &str) -> Result<Self> {
        let spec = AgentSpec::from_yaml_strict(source)?;
        let source: serde_yaml::Value = serde_yaml::from_str(source)?;
        Ok(Self {
            spec,
            policy: TaskIterationPolicy::from_source(&source)?,
        })
    }

    /// Explicitly pairs a typed specification with a host-selected policy.
    pub fn new(spec: AgentSpec, policy: TaskIterationPolicy) -> Self {
        Self { spec, policy }
    }

    /// Returns the typed schema without discarding the prepared source information.
    pub fn spec(&self) -> &AgentSpec {
        &self.spec
    }

    /// Returns the task-loop policy for explicit construction and durable identity checks.
    pub fn task_iteration_policy(&self) -> &TaskIterationPolicy {
        &self.policy
    }

    /// Separates the typed schema from its source policy for a caller that will preserve both.
    pub fn into_parts(self) -> (AgentSpec, TaskIterationPolicy) {
        (self.spec, self.policy)
    }

    // The persistence header keeps the body valid legacy YAML while preserving policy as data, not as execution authority.
    pub(crate) fn to_persistence_yaml(&self) -> Result<String> {
        let spec = serde_yaml::to_string(&self.spec)?;
        if self.policy == TaskIterationPolicy::configured() {
            return Ok(spec);
        }
        let policy = serde_json::to_string(&self.policy)?;
        Ok(format!(
            "# ai-agents-task-iteration-policy-v1: {policy}\n{spec}"
        ))
    }

    // A legacy persisted declaration has lost source presence; never infer unbounded semantics from its materialized YAML.
    pub(crate) fn from_persistence_yaml(source: &str) -> Result<Self> {
        const PREFIX: &str = "# ai-agents-task-iteration-policy-v1: ";
        let spec = AgentSpec::from_yaml_strict(source)?;
        let policy = match source
            .lines()
            .next()
            .and_then(|line| line.strip_prefix(PREFIX))
        {
            Some(encoded) => serde_json::from_str(encoded)
                .map_err(|_| AgentError::Config("invalid saved task iteration policy".into()))?,
            None => TaskIterationPolicy::configured(),
        };
        Ok(Self { spec, policy })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_policy_round_trips_and_legacy_declarations_remain_conservative() {
        let prepared = PreparedAgentSpec::from_yaml_strict(
            "name: Agent\nsystem_prompt: test\nreasoning: {mode: cot}\n",
        )
        .unwrap();
        let encoded = prepared.to_persistence_yaml().unwrap();
        assert!(encoded.starts_with("# ai-agents-task-iteration-policy-v1: "));
        let restored = PreparedAgentSpec::from_persistence_yaml(&encoded).unwrap();
        assert_eq!(
            restored.task_iteration_policy(),
            prepared.task_iteration_policy()
        );
        assert_eq!(
            serde_json::to_value(restored.spec()).unwrap(),
            serde_json::to_value(prepared.spec()).unwrap()
        );
        let legacy = serde_yaml::to_string(prepared.spec()).unwrap();
        let legacy = PreparedAgentSpec::from_persistence_yaml(&legacy).unwrap();
        assert_eq!(
            legacy.task_iteration_policy(),
            &TaskIterationPolicy::configured()
        );
        let invalid = format!(
            "# ai-agents-task-iteration-policy-v1: broken\n{}",
            serde_yaml::to_string(prepared.spec()).unwrap()
        );
        assert!(PreparedAgentSpec::from_persistence_yaml(&invalid).is_err());
    }

    #[test]
    fn prepared_source_distinguishes_explicit_defaults_and_nested_replacements() {
        let source = "name: Agent\nsystem_prompt: test\nreasoning: {mode: cot}\nstates:\n  initial: parent\n  states:\n    parent:\n      reasoning: {mode: cot, max_iterations: 5}\n      states:\n        child: {reasoning: {mode: cot}}\n";
        let prepared = PreparedAgentSpec::from_yaml_strict(source).unwrap();
        assert_eq!(prepared.spec.max_iterations, 10);
        assert!(!prepared.policy.root_configured());
        assert!(!prepared.policy.limit_for_state(None, false));
        assert!(prepared.policy.limit_for_state(Some("parent"), true));
        assert!(!prepared.policy.limit_for_state(Some("parent.child"), true));
        assert!(prepared.policy.limit_for_state(Some("unknown"), true));
        let explicit = PreparedAgentSpec::from_yaml_strict("name: Agent\nsystem_prompt: test\nmax_iterations: 10\nreasoning: {mode: cot, max_iterations: 5}\n").unwrap();
        assert!(explicit.policy.root_configured());
        assert!(explicit.policy.limit_for_state(None, false));
    }
}
