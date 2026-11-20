//! Dependency-light completion gate schema; evaluation belongs to the runtime.

use serde::de::DeserializeOwned;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

const MAX_GATE_DEPTH: usize = 32;
const MAX_GATE_NODES: usize = 256;

// A present null is an equality operand, not the absence of an equality predicate.
fn deserialize_present_equality<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error> {
    serde_json::Value::deserialize(deserializer).map(Some)
}

/// A typed comparison against current, eligible evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathAssertion {
    pub path: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present_equality"
    )]
    pub eq: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gte: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lte: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exists: Option<bool>,
}

/// A tool invocation predicate, not a grant or proof that a denied request executed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCalledGate {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count_gte: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
}

/// A matching authorized validation command's exit result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandExitGate {
    pub command: String,
    pub code: i32,
}

/// Diagnostics must also be available and fresh when this gate is evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticsClearGate {
    pub severity: String,
}

/// Semantic quality check configuration; the runtime owns judge execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeGate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass_threshold: Option<f64>,
    pub criteria: Vec<String>,
}

/// One strict YAML gate operator per node, with bounded boolean composition.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionGate {
    TodosDone(bool),
    TodoCountGte(usize),
    State(String),
    StateIn(Vec<String>),
    ContextPath(PathAssertion),
    ToolCalled(ToolCalledGate),
    CommandExit(CommandExitGate),
    DiagnosticsClear(DiagnosticsClearGate),
    ValidationPassed(bool),
    ValidatorPassed(String),
    ProgressPath(PathAssertion),
    ResponseNotEmpty(bool),
    ResponseContains(String),
    ResponseContainsAny(Vec<String>),
    ArtifactExists(String),
    Observability(PathAssertion),
    Judge(JudgeGate),
    All(Vec<CompletionGate>),
    Any(Vec<CompletionGate>),
    Not(Box<CompletionGate>),
}

impl CompletionGate {
    /// Checks semantic operands and the bound for programmatically constructed trees.
    pub fn validate(&self) -> Result<(), String> {
        fn visit(gate: &CompletionGate, depth: usize, remaining: &mut usize) -> Result<(), String> {
            if depth > MAX_GATE_DEPTH || *remaining == 0 {
                return Err("completion gate exceeds depth or node limit".into());
            }
            *remaining -= 1;
            match gate {
                CompletionGate::All(gates) | CompletionGate::Any(gates) => {
                    if gates.is_empty() {
                        return Err("all/any completion gate cannot be empty".into());
                    }
                    for gate in gates {
                        visit(gate, depth + 1, remaining)?;
                    }
                }
                CompletionGate::Not(gate) => visit(gate, depth + 1, remaining)?,
                CompletionGate::ContextPath(path)
                | CompletionGate::ProgressPath(path)
                | CompletionGate::Observability(path) => {
                    if path.path.trim().is_empty()
                        || ![path.gte, path.gt, path.lte, path.lt]
                            .iter()
                            .flatten()
                            .all(|value| value.is_finite())
                    {
                        return Err("completion path or numeric comparison is invalid".into());
                    }
                }
                CompletionGate::ToolCalled(gate) if gate.id.trim().is_empty() => {
                    return Err("tool_called.id cannot be empty".into());
                }
                CompletionGate::CommandExit(gate) if gate.command.trim().is_empty() => {
                    return Err("command_exit.command cannot be empty".into());
                }
                CompletionGate::Judge(gate)
                    if gate.criteria.is_empty()
                        || gate
                            .criteria
                            .iter()
                            .any(|criterion| criterion.trim().is_empty())
                        || gate.pass_threshold.is_some_and(|threshold| {
                            !threshold.is_finite() || !(0.0..=1.0).contains(&threshold)
                        }) =>
                {
                    return Err("judge criteria or threshold is invalid".into());
                }
                CompletionGate::StateIn(values) | CompletionGate::ResponseContainsAny(values)
                    if values.is_empty() || values.iter().any(|value| value.trim().is_empty()) =>
                {
                    return Err("completion list cannot be empty".into());
                }
                CompletionGate::State(value)
                | CompletionGate::ValidatorPassed(value)
                | CompletionGate::ResponseContains(value)
                | CompletionGate::ArtifactExists(value)
                    if value.trim().is_empty() =>
                {
                    return Err("completion operand cannot be empty".into());
                }
                _ => {}
            }
            Ok(())
        }
        let mut remaining = MAX_GATE_NODES;
        visit(self, 0, &mut remaining)
    }
}

impl<'de> Deserialize<'de> for CompletionGate {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        fn leaf<T: DeserializeOwned>(value: Value) -> Result<T, String> {
            serde_yaml::from_value(value).map_err(|error| error.to_string())
        }
        fn parse(
            value: Value,
            depth: usize,
            remaining: &mut usize,
        ) -> Result<CompletionGate, String> {
            if depth > MAX_GATE_DEPTH || *remaining == 0 {
                return Err("completion gate exceeds depth or node limit".into());
            }
            *remaining -= 1;
            let Value::Mapping(map) = value else {
                return Err("completion gate must be a mapping with one operator".into());
            };
            if map.len() != 1 {
                return Err("completion gate must contain exactly one operator".into());
            }
            let (key, operand) = map.into_iter().next().unwrap();
            let key = key
                .as_str()
                .ok_or("completion gate operator must be a string")?;
            let gate = match key {
                "todos_done" => CompletionGate::TodosDone(leaf(operand)?),
                "todo_count_gte" => CompletionGate::TodoCountGte(leaf(operand)?),
                "state" => CompletionGate::State(leaf(operand)?),
                "state_in" => CompletionGate::StateIn(leaf(operand)?),
                "context_path" => CompletionGate::ContextPath(leaf(operand)?),
                "tool_called" => CompletionGate::ToolCalled(leaf(operand)?),
                "command_exit" => CompletionGate::CommandExit(leaf(operand)?),
                "diagnostics_clear" => CompletionGate::DiagnosticsClear(leaf(operand)?),
                "validation_passed" => CompletionGate::ValidationPassed(leaf(operand)?),
                "validator_passed" => CompletionGate::ValidatorPassed(leaf(operand)?),
                "progress_path" => CompletionGate::ProgressPath(leaf(operand)?),
                "response_not_empty" => CompletionGate::ResponseNotEmpty(leaf(operand)?),
                "response_contains" => CompletionGate::ResponseContains(leaf(operand)?),
                "response_contains_any" => CompletionGate::ResponseContainsAny(leaf(operand)?),
                "artifact_exists" => CompletionGate::ArtifactExists(leaf(operand)?),
                "observability" => CompletionGate::Observability(leaf(operand)?),
                "judge" => CompletionGate::Judge(leaf(operand)?),
                "not" => CompletionGate::Not(Box::new(parse(operand, depth + 1, remaining)?)),
                "all" | "any" => {
                    let Value::Sequence(values) = operand else {
                        return Err(format!("{key} requires a nonempty list"));
                    };
                    if values.is_empty() {
                        return Err(format!("{key} requires a nonempty list"));
                    }
                    let values = values
                        .into_iter()
                        .map(|value| parse(value, depth + 1, remaining))
                        .collect::<Result<Vec<_>, _>>()?;
                    if key == "all" {
                        CompletionGate::All(values)
                    } else {
                        CompletionGate::Any(values)
                    }
                }
                _ => return Err(format!("unknown completion gate operator '{key}'")),
            };
            Ok(gate)
        }
        let mut remaining = MAX_GATE_NODES;
        let gate = parse(Value::deserialize(deserializer)?, 0, &mut remaining)
            .map_err(serde::de::Error::custom)?;
        gate.validate().map_err(serde::de::Error::custom)?;
        Ok(gate)
    }
}

impl Serialize for CompletionGate {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(1))?;
        match self {
            Self::TodosDone(value) => map.serialize_entry("todos_done", value)?,
            Self::TodoCountGte(value) => map.serialize_entry("todo_count_gte", value)?,
            Self::State(value) => map.serialize_entry("state", value)?,
            Self::StateIn(value) => map.serialize_entry("state_in", value)?,
            Self::ContextPath(value) => map.serialize_entry("context_path", value)?,
            Self::ToolCalled(value) => map.serialize_entry("tool_called", value)?,
            Self::CommandExit(value) => map.serialize_entry("command_exit", value)?,
            Self::DiagnosticsClear(value) => map.serialize_entry("diagnostics_clear", value)?,
            Self::ValidationPassed(value) => map.serialize_entry("validation_passed", value)?,
            Self::ValidatorPassed(value) => map.serialize_entry("validator_passed", value)?,
            Self::ProgressPath(value) => map.serialize_entry("progress_path", value)?,
            Self::ResponseNotEmpty(value) => map.serialize_entry("response_not_empty", value)?,
            Self::ResponseContains(value) => map.serialize_entry("response_contains", value)?,
            Self::ResponseContainsAny(value) => {
                map.serialize_entry("response_contains_any", value)?
            }
            Self::ArtifactExists(value) => map.serialize_entry("artifact_exists", value)?,
            Self::Observability(value) => map.serialize_entry("observability", value)?,
            Self::Judge(value) => map.serialize_entry("judge", value)?,
            Self::All(value) => map.serialize_entry("all", value)?,
            Self::Any(value) => map.serialize_entry("any", value)?,
            Self::Not(value) => map.serialize_entry("not", value)?,
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_round_trips_the_exact_operator_mapping() {
        let yaml = "all:\n  - todos_done: true\n  - tool_called:\n      id: file_read\n      count_gte: 1\n      executed: true\n      success: true\n  - not:\n      state: blocked\n";
        let gate: CompletionGate = serde_yaml::from_str(yaml).unwrap();
        let encoded = serde_yaml::to_value(&gate).unwrap();
        let expected: Value = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(encoded, expected);
        assert_eq!(
            serde_yaml::from_value::<CompletionGate>(encoded).unwrap(),
            gate
        );
    }

    #[test]
    fn explicit_null_equality_survives_yaml_and_json_round_trips() {
        let yaml = "context_path: {path: ticket.result, eq: null}";
        let gate: CompletionGate = serde_yaml::from_str(yaml).unwrap();
        assert!(
            matches!(&gate, CompletionGate::ContextPath(path) if path.eq == Some(serde_json::Value::Null))
        );
        let yaml_value = serde_yaml::to_value(&gate).unwrap();
        assert_eq!(yaml_value, serde_yaml::from_str::<Value>(yaml).unwrap());
        assert_eq!(
            serde_yaml::from_value::<CompletionGate>(yaml_value).unwrap(),
            gate
        );
        let json = serde_json::to_value(&gate).unwrap();
        assert_eq!(json["context_path"]["eq"], serde_json::Value::Null);
        assert_eq!(
            serde_json::from_value::<CompletionGate>(json).unwrap(),
            gate
        );
    }

    #[test]
    fn gate_rejects_ambiguous_unknown_and_unbounded_trees() {
        for yaml in [
            "state: ready\ntodos_done: true",
            "state: ready\nunknown: true",
            "all: []",
            "any: []",
            "not: []",
            "state: ''",
            "tool_called: {id: file_read, typo: true}",
            "judge: {criteria: []}",
            "context_path: {path: progress, gte: .nan}",
        ] {
            assert!(
                serde_yaml::from_str::<CompletionGate>(yaml).is_err(),
                "{yaml}"
            );
        }
        let mut tree = Value::Mapping(serde_yaml::Mapping::from_iter([(
            Value::String("state".into()),
            Value::String("ready".into()),
        )]));
        for _ in 0..34 {
            tree = Value::Mapping(serde_yaml::Mapping::from_iter([(
                Value::String("not".into()),
                tree,
            )]));
        }
        assert!(serde_yaml::from_value::<CompletionGate>(tree).is_err());
    }
}
