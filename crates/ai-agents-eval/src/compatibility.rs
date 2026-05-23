use serde::Deserialize;
use serde_json::Value;

use crate::assertion::Assertion;
use crate::suite::{EvalSettings, EvalSuite, Scenario, Turn};
use crate::{EvalError, Result};

/// Input row shape accepted by the JSONL compatibility adapter.
#[derive(Debug, Deserialize)]
struct JsonlRow {
    /// Stable identifier for this item.
    id: String,
    /// Optional language label for filtering, metrics, and judge context.
    #[serde(default)]
    language: Option<String>,
    /// User input sent to the runtime.
    input: String,
    /// Expected assertion object for a generated turn.
    #[serde(default)]
    expected: Option<Assertion>,
    /// Tags used by filters and grouped metrics.
    #[serde(default)]
    tags: Vec<String>,
    /// Actor ID used for this scenario, turn, or assertion.
    #[serde(default)]
    actor: Option<String>,
    /// Runtime or fixture context value.
    #[serde(default)]
    context: Value,
}

pub fn suite_from_jsonl(name: String, content: &str) -> Result<EvalSuite> {
    let mut scenarios = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: JsonlRow = serde_json::from_str(line).map_err(|error| {
            EvalError::Config(format!("invalid JSONL row {}: {}", line_idx + 1, error))
        })?;
        scenarios.push(Scenario {
            id: row.id,
            name: None,
            tags: row.tags,
            language: row.language,
            actor: row.actor,
            context: row.context,
            env: Default::default(),
            skip: Default::default(),
            turns: vec![Turn {
                input: row.input,
                actor: None,
                context: Value::Null,
                stream: None,
                timeout_ms: None,
                assertions: row.expected,
            }],
            steps: Vec::new(),
        });
    }
    Ok(EvalSuite {
        name,
        agent: None,
        settings: EvalSettings::default(),
        observability: None,
        fixtures: Default::default(),
        scenarios,
    })
}
