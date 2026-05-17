use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::evidence::{DisambiguationStatus, ToolExecutionRecord, TurnEvidence};
use crate::judge::{JudgeAssertion, LLMJudge};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Assertion {
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub state_in: Option<Vec<String>>,
    #[serde(default)]
    pub state_not: Option<String>,
    #[serde(default)]
    pub state_history_contains: Option<String>,
    #[serde(default)]
    pub response_contains: Option<StringList>,
    #[serde(default)]
    pub response_contains_any: Option<StringList>,
    #[serde(default)]
    pub response_not_contains: Option<StringList>,
    #[serde(default)]
    pub response_not_empty: Option<bool>,
    #[serde(default)]
    pub response_semantic: Option<JudgeAssertion>,
    #[serde(default)]
    pub disambiguation: Option<DisambiguationExpectation>,
    #[serde(default)]
    pub no_disambiguation: Option<bool>,
    #[serde(default)]
    pub tool_called: Option<ToolCalledAssertion>,
    #[serde(default)]
    pub tool_not_called: Option<String>,
    #[serde(default)]
    pub skill_triggered: Option<String>,
    #[serde(default)]
    pub metadata_contains: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub metadata_path: Option<PathAssertion>,
    #[serde(default)]
    pub context_path: Option<PathAssertion>,
    #[serde(default)]
    pub facts_include: Option<FactsAssertion>,
    #[serde(default)]
    pub relationship: Option<RelationshipAssertion>,
    #[serde(default)]
    pub persona_secret_revealed: Option<SecretAssertion>,
    #[serde(default)]
    pub orchestration: Option<OrchestrationAssertion>,
    #[serde(default)]
    pub observability: Option<ObservabilityAssertion>,
    #[serde(default)]
    pub judge: Option<JudgeAssertion>,
    #[serde(default)]
    pub any: Option<Vec<Assertion>>,
    #[serde(default)]
    pub all: Option<Vec<Assertion>>,
    #[serde(default)]
    pub not: Option<Box<Assertion>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringList {
    One(String),
    Many(Vec<String>),
}

impl StringList {
    fn items(&self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value.clone()],
            Self::Many(values) => values.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolCalledAssertion {
    Id(String),
    Object(ToolCalledObject),
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ToolCalledObject {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub count: Option<usize>,
    #[serde(default)]
    pub count_gte: Option<usize>,
    #[serde(default)]
    pub success: Option<bool>,
    #[serde(default)]
    pub source_in: Option<Vec<String>>,
    #[serde(default)]
    pub args: Option<PathAssertion>,
    #[serde(default)]
    pub args_original: Option<PathAssertion>,
    #[serde(default)]
    pub args_executed: Option<PathAssertion>,
    #[serde(default)]
    pub result_path: Option<PathAssertion>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DisambiguationExpectation {
    Triggered,
    Skipped,
    Clarified,
    Abandoned,
    GiveUp,
    Escalated,
    BestGuess,
    Clear,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PathAssertion {
    pub path: String,
    #[serde(default)]
    pub eq: Option<Value>,
    #[serde(default)]
    pub neq: Option<Value>,
    #[serde(default, rename = "in")]
    pub in_values: Option<Vec<Value>>,
    #[serde(default)]
    pub contains: Option<Value>,
    #[serde(default)]
    pub exists: Option<bool>,
    #[serde(default)]
    pub gte: Option<f64>,
    #[serde(default)]
    pub lte: Option<f64>,
    #[serde(default)]
    pub gt: Option<f64>,
    #[serde(default)]
    pub lt: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct FactsAssertion {
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub semantic: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RelationshipAssertion {
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub exists: Option<bool>,
    #[serde(default)]
    pub perspective: Option<String>,
    #[serde(default)]
    pub dimension: Option<String>,
    #[serde(default)]
    pub gte: Option<f64>,
    #[serde(default)]
    pub lte: Option<f64>,
    #[serde(default)]
    pub gt: Option<f64>,
    #[serde(default)]
    pub lt: Option<f64>,
    #[serde(default)]
    pub eq: Option<f64>,
    #[serde(default)]
    pub interaction_count_gte: Option<u64>,
    #[serde(default)]
    pub notable_event_count_gte: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SecretAssertion {
    Bool(bool),
    Id(String),
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct OrchestrationAssertion {
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default, rename = "type")]
    pub type_name: Option<String>,
    #[serde(default)]
    pub final_agent_in: Option<Vec<String>>,
    #[serde(default)]
    pub agents_include: Option<Vec<String>>,
    #[serde(default)]
    pub stages: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ObservabilityAssertion {
    #[serde(default)]
    pub total_llm_calls_lte: Option<u64>,
    #[serde(default)]
    pub total_cost_usd_lte: Option<f64>,
    #[serde(default)]
    pub purpose_counts: HashMap<String, PathAssertion>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssertionResultDetail {
    pub assertion: String,
    pub passed: bool,
    pub actual: Value,
    pub expected: Value,
    pub message: Option<String>,
}

pub enum AssertionOutcome {
    Passed(Vec<AssertionResultDetail>),
    Failed(Vec<AssertionResultDetail>),
    Error(String),
}

impl AssertionResultDetail {
    fn pass(name: impl Into<String>, actual: Value, expected: Value) -> Self {
        Self {
            assertion: name.into(),
            passed: true,
            actual,
            expected,
            message: None,
        }
    }

    fn fail(
        name: impl Into<String>,
        actual: Value,
        expected: Value,
        message: impl Into<String>,
    ) -> Self {
        Self {
            assertion: name.into(),
            passed: false,
            actual,
            expected,
            message: Some(message.into()),
        }
    }
}

pub async fn evaluate_assertion(
    assertion: &Assertion,
    evidence: &TurnEvidence,
    response: &str,
    judge: Option<&LLMJudge>,
) -> AssertionOutcome {
    let mut details = Vec::new();

    if let Some(children) = &assertion.all {
        for child in children {
            match evaluate_assertion_boxed(child, evidence, response, judge).await {
                AssertionOutcome::Passed(mut d) => details.append(&mut d),
                AssertionOutcome::Failed(mut d) => {
                    details.append(&mut d);
                    return AssertionOutcome::Failed(details);
                }
                AssertionOutcome::Error(e) => return AssertionOutcome::Error(e),
            }
        }
        details.push(AssertionResultDetail::pass("all", json!(true), json!(true)));
    }

    if let Some(children) = &assertion.any {
        let mut failures = Vec::new();
        for child in children {
            match evaluate_assertion_boxed(child, evidence, response, judge).await {
                AssertionOutcome::Passed(mut d) => {
                    details.append(&mut d);
                    details.push(AssertionResultDetail::pass("any", json!(true), json!(true)));
                    return AssertionOutcome::Passed(details);
                }
                AssertionOutcome::Failed(mut d) => failures.append(&mut d),
                AssertionOutcome::Error(e) => failures.push(AssertionResultDetail::fail(
                    "any_branch_error",
                    json!(e),
                    json!("pass"),
                    "branch error",
                )),
            }
        }
        details.extend(failures);
        details.push(AssertionResultDetail::fail(
            "any",
            json!(false),
            json!(true),
            "no branch passed",
        ));
    }

    if let Some(child) = &assertion.not {
        match evaluate_assertion_boxed(child, evidence, response, judge).await {
            AssertionOutcome::Passed(_) => details.push(AssertionResultDetail::fail(
                "not",
                json!(true),
                json!(false),
                "child assertion passed",
            )),
            AssertionOutcome::Failed(_) => details.push(AssertionResultDetail::pass(
                "not",
                json!(false),
                json!(false),
            )),
            AssertionOutcome::Error(e) => return AssertionOutcome::Error(e),
        }
    }

    evaluate_simple(assertion, evidence, response, judge, &mut details).await;

    if details.iter().any(|d| !d.passed) {
        AssertionOutcome::Failed(details)
    } else {
        AssertionOutcome::Passed(details)
    }
}

fn evaluate_assertion_boxed<'a>(
    assertion: &'a Assertion,
    evidence: &'a TurnEvidence,
    response: &'a str,
    judge: Option<&'a LLMJudge>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = AssertionOutcome> + Send + 'a>> {
    Box::pin(evaluate_assertion(assertion, evidence, response, judge))
}

async fn evaluate_simple(
    assertion: &Assertion,
    evidence: &TurnEvidence,
    response: &str,
    judge: Option<&LLMJudge>,
    details: &mut Vec<AssertionResultDetail>,
) {
    if let Some(expected) = &assertion.state {
        check_eq("state", evidence.state.clone(), expected.clone(), details);
    }
    if let Some(expected) = &assertion.state_in {
        push_bool(
            "state_in",
            evidence
                .state
                .as_ref()
                .is_some_and(|s| expected.contains(s)),
            json!(evidence.state),
            json!(expected),
            details,
        );
    }
    if let Some(expected) = &assertion.state_not {
        push_bool(
            "state_not",
            evidence.state.as_ref().is_none_or(|s| s != expected),
            json!(evidence.state),
            json!(expected),
            details,
        );
    }
    if let Some(expected) = &assertion.state_history_contains {
        let passed = evidence
            .state_history
            .iter()
            .any(|event| &event.to == expected || &event.from == expected);
        push_bool(
            "state_history_contains",
            passed,
            json!(evidence.state_history),
            json!(expected),
            details,
        );
    }
    if let Some(expected) = &assertion.response_contains {
        for item in expected.items() {
            push_bool(
                "response_contains",
                response.contains(&item),
                json!(response),
                json!(item),
                details,
            );
        }
    }
    if let Some(expected) = &assertion.response_contains_any {
        let items = expected.items();
        push_bool(
            "response_contains_any",
            items.iter().any(|item| response.contains(item)),
            json!(response),
            json!(items),
            details,
        );
    }
    if let Some(expected) = &assertion.response_not_contains {
        for item in expected.items() {
            push_bool(
                "response_not_contains",
                !response.contains(&item),
                json!(response),
                json!(item),
                details,
            );
        }
    }
    if let Some(expected) = assertion.response_not_empty {
        push_bool(
            "response_not_empty",
            (!response.trim().is_empty()) == expected,
            json!(response),
            json!(expected),
            details,
        );
    }
    if let Some(expected) = &assertion.disambiguation {
        let actual = evidence.disambiguation.as_ref().map(|d| &d.status);
        push_bool(
            "disambiguation",
            actual.is_some_and(|status| disambiguation_matches(status, expected)),
            json!(actual),
            json!(expected),
            details,
        );
    }
    if let Some(expected) = assertion.no_disambiguation {
        let triggered = evidence.disambiguation.as_ref().is_some_and(|d| {
            matches!(
                d.status,
                DisambiguationStatus::Triggered
                    | DisambiguationStatus::Clarified
                    | DisambiguationStatus::BestGuess
            )
        });
        push_bool(
            "no_disambiguation",
            (!triggered) == expected,
            json!(!triggered),
            json!(expected),
            details,
        );
    }
    if let Some(tool) = &assertion.tool_called {
        evaluate_tool_called(tool, evidence, details);
    }
    if let Some(tool_id) = &assertion.tool_not_called {
        let passed = !evidence
            .tool_executions
            .iter()
            .any(|record| &record.tool_id == tool_id || &record.requested_name == tool_id);
        push_bool(
            "tool_not_called",
            passed,
            json!(tool_id),
            json!("not called"),
            details,
        );
    }
    if let Some(skill_id) = &assertion.skill_triggered {
        let passed = evidence.skill.as_ref().is_some_and(|skill| {
            skill.selected_skill_id.as_deref() == Some(skill_id)
                || skill.executed_skill_id.as_deref() == Some(skill_id)
        });
        push_bool(
            "skill_triggered",
            passed,
            json!(evidence.skill),
            json!(skill_id),
            details,
        );
    }
    if let Some(path) = &assertion.metadata_path {
        evaluate_path(
            "metadata_path",
            evidence.response_metadata.as_ref(),
            path,
            details,
        );
    }
    if let Some(path) = &assertion.context_path {
        evaluate_path("context_path", Some(&evidence.context), path, details);
    }
    if let Some(expected) = &assertion.facts_include {
        evaluate_facts(expected, evidence, details);
    }
    if let Some(expected) = &assertion.relationship {
        evaluate_relationship(expected, evidence, details);
    }
    if let Some(expected) = &assertion.persona_secret_revealed {
        evaluate_secret(expected, evidence, details);
    }
    if let Some(expected) = &assertion.orchestration {
        evaluate_orchestration(expected, evidence, details);
    }
    if let Some(expected) = &assertion.observability {
        evaluate_observability(expected, evidence, details);
    }
    if let Some(criteria) = assertion
        .judge
        .as_ref()
        .or(assertion.response_semantic.as_ref())
    {
        if let Some(judge) = judge {
            match judge.evaluate(response, criteria).await {
                Ok(result) => push_bool(
                    "judge",
                    result.passed,
                    json!(result.overall_score),
                    json!(criteria.pass_threshold),
                    details,
                ),
                Err(error) => details.push(AssertionResultDetail::fail(
                    "judge",
                    json!(error.to_string()),
                    json!("valid judge result"),
                    "judge failed",
                )),
            }
        } else {
            details.push(AssertionResultDetail::fail(
                "judge",
                json!(null),
                json!("judge configured"),
                "no judge LLM available",
            ));
        }
    }
}

fn check_eq<T: PartialEq + Serialize>(
    name: &str,
    actual: Option<T>,
    expected: T,
    details: &mut Vec<AssertionResultDetail>,
) {
    push_bool(
        name,
        actual.as_ref().is_some_and(|a| *a == expected),
        json!(actual),
        json!(expected),
        details,
    );
}
fn push_bool(
    name: &str,
    passed: bool,
    actual: Value,
    expected: Value,
    details: &mut Vec<AssertionResultDetail>,
) {
    if passed {
        details.push(AssertionResultDetail::pass(name, actual, expected));
    } else {
        details.push(AssertionResultDetail::fail(
            name,
            actual,
            expected,
            "assertion failed",
        ));
    }
}

fn disambiguation_matches(
    actual: &DisambiguationStatus,
    expected: &DisambiguationExpectation,
) -> bool {
    matches!(
        (actual, expected),
        (
            DisambiguationStatus::Triggered,
            DisambiguationExpectation::Triggered
        ) | (
            DisambiguationStatus::Skipped,
            DisambiguationExpectation::Skipped
        ) | (
            DisambiguationStatus::Clarified,
            DisambiguationExpectation::Clarified
        ) | (
            DisambiguationStatus::Abandoned,
            DisambiguationExpectation::Abandoned
        ) | (
            DisambiguationStatus::GiveUp,
            DisambiguationExpectation::GiveUp
        ) | (
            DisambiguationStatus::Escalated,
            DisambiguationExpectation::Escalated
        ) | (
            DisambiguationStatus::BestGuess,
            DisambiguationExpectation::BestGuess
        ) | (
            DisambiguationStatus::Clear,
            DisambiguationExpectation::Clear
        )
    )
}

fn evaluate_tool_called(
    assertion: &ToolCalledAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let (id, object) = match assertion {
        ToolCalledAssertion::Id(id) => (Some(id.as_str()), None),
        ToolCalledAssertion::Object(object) => (object.id.as_deref(), Some(object)),
    };
    let mut records: Vec<&ToolExecutionRecord> = evidence.tool_executions.iter().collect();
    if let Some(id) = id {
        records.retain(|record| record.tool_id == id || record.requested_name == id);
    }
    if let Some(object) = object {
        if let Some(success) = object.success {
            records.retain(|record| record.success == success);
        }
        if let Some(sources) = &object.source_in {
            records.retain(|record| {
                sources
                    .iter()
                    .any(|source| *source == serde_plain_source(&record.source))
            });
        }
    }
    let count = records.len();
    let mut passed = count > 0;
    if let Some(object) = object {
        if let Some(expected) = object.count {
            passed &= count == expected;
        }
        if let Some(expected) = object.count_gte {
            passed &= count >= expected;
        }
        if let Some(path) = object.args.as_ref().or(object.args_executed.as_ref()) {
            passed &= records
                .iter()
                .any(|record| path_matches(&record.arguments_executed, path));
        }
        if let Some(path) = &object.args_original {
            passed &= records
                .iter()
                .any(|record| path_matches(&record.arguments_original, path));
        }
        if let Some(path) = &object.result_path {
            passed &= records.iter().any(|record| {
                record
                    .output
                    .as_ref()
                    .is_some_and(|value| path_matches(value, path))
            });
        }
    }
    push_bool(
        "tool_called",
        passed,
        json!(count),
        json!(assertion),
        details,
    );
}

fn serde_plain_source(source: &crate::evidence::ToolExecutionSource) -> String {
    serde_json::to_string(source)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}
fn evaluate_path(
    name: &str,
    root: Option<&Value>,
    assertion: &PathAssertion,
    details: &mut Vec<AssertionResultDetail>,
) {
    let actual = root.and_then(|value| get_path(value, &assertion.path));
    push_bool(
        name,
        path_actual_matches(actual, assertion),
        json!(actual),
        json!(assertion),
        details,
    );
}
fn path_matches(root: &Value, assertion: &PathAssertion) -> bool {
    path_actual_matches(get_path(root, &assertion.path), assertion)
}

fn path_actual_matches(actual: Option<&Value>, assertion: &PathAssertion) -> bool {
    if let Some(exists) = assertion.exists {
        if exists != actual.is_some() {
            return false;
        }
    }
    let Some(actual) = actual else {
        return assertion.exists == Some(false);
    };
    if let Some(expected) = &assertion.eq {
        if actual != expected {
            return false;
        }
    }
    if let Some(expected) = &assertion.neq {
        if actual == expected {
            return false;
        }
    }
    if let Some(values) = &assertion.in_values {
        if !values.contains(actual) {
            return false;
        }
    }
    if let Some(expected) = &assertion.contains {
        let contains = match (actual, expected) {
            (Value::String(a), Value::String(e)) => a.contains(e),
            (Value::Array(arr), e) => arr.contains(e),
            _ => false,
        };
        if !contains {
            return false;
        }
    }
    if let Some(expected) = assertion.gte {
        if actual.as_f64().unwrap_or(f64::NAN) < expected {
            return false;
        }
    }
    if let Some(expected) = assertion.lte {
        if actual.as_f64().unwrap_or(f64::NAN) > expected {
            return false;
        }
    }
    if let Some(expected) = assertion.gt {
        if actual.as_f64().unwrap_or(f64::NAN) <= expected {
            return false;
        }
    }
    if let Some(expected) = assertion.lt {
        if actual.as_f64().unwrap_or(f64::NAN) >= expected {
            return false;
        }
    }
    true
}

fn get_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(value);
    }
    let mut current = value;
    for part in path.split('.') {
        current = current.get(part)?;
    }
    Some(current)
}

fn evaluate_facts(
    assertion: &FactsAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let facts = evidence
        .facts
        .as_ref()
        .map(|f| &f.facts)
        .cloned()
        .unwrap_or_default();
    let passed = facts.iter().any(|fact| {
        assertion.category.as_ref().is_none_or(|category| {
            fact.get("category")
                .map(|value| value.to_string().trim_matches('"').to_string())
                .is_some_and(|actual| actual == *category || actual.ends_with(category))
        })
    });
    push_bool(
        "facts_include",
        passed,
        json!(facts),
        json!(assertion),
        details,
    );
}

fn evaluate_relationship(
    assertion: &RelationshipAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let Some(rel) = &evidence.relationship else {
        push_bool(
            "relationship",
            assertion.exists == Some(false),
            json!(null),
            json!(assertion),
            details,
        );
        return;
    };
    let current = rel.current.as_ref();
    let mut passed = assertion
        .exists
        .map(|expected| expected == current.is_some())
        .unwrap_or(true);
    let perspective = assertion.perspective.as_deref().unwrap_or("agent_to_actor");
    if !rel.available_perspectives.iter().any(|p| p == perspective) {
        details.push(AssertionResultDetail::fail(
            "relationship",
            json!(rel.available_perspectives),
            json!(perspective),
            "relationship perspective unavailable for model",
        ));
        return;
    }
    if let Some(count) = assertion.interaction_count_gte {
        let actual = current
            .and_then(|v| v.get("interaction_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        passed &= actual >= count;
    }
    if let Some(count) = assertion.notable_event_count_gte {
        let actual = current
            .and_then(|v| v.get("notable_events"))
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        passed &= actual >= count;
    }
    if let Some(dimension) = &assertion.dimension {
        let value = relationship_dimension_value(current, perspective, dimension);
        let mut dim_pass = value.is_some();
        if let Some(v) = assertion.gte {
            dim_pass &= value.unwrap_or(f64::NAN) >= v;
        }
        if let Some(v) = assertion.lte {
            dim_pass &= value.unwrap_or(f64::NAN) <= v;
        }
        if let Some(v) = assertion.gt {
            dim_pass &= value.unwrap_or(f64::NAN) > v;
        }
        if let Some(v) = assertion.lt {
            dim_pass &= value.unwrap_or(f64::NAN) < v;
        }
        if let Some(v) = assertion.eq {
            dim_pass &= (value.unwrap_or(f64::NAN) - v).abs() < f64::EPSILON;
        }
        passed &= dim_pass;
    }
    push_bool(
        "relationship",
        passed,
        json!(current),
        json!(assertion),
        details,
    );
}

fn relationship_dimension_value(
    current: Option<&Value>,
    perspective: &str,
    dimension: &str,
) -> Option<f64> {
    let current = current?;
    match perspective {
        "agent_to_actor" => current.get("dimensions")?.get(dimension)?.as_f64(),
        "perceived_actor_to_agent" => current
            .get("perceived_actor_to_agent")?
            .get(dimension)?
            .as_f64(),
        "mutual" => current.get("dimensions")?.get(dimension)?.as_f64(),
        _ => None,
    }
}

fn evaluate_secret(
    assertion: &SecretAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let persona = evidence.persona.as_ref();
    let actual = persona.is_some_and(|p| p.secret_revealed);
    let passed = match assertion {
        SecretAssertion::Bool(expected) => actual == *expected,
        SecretAssertion::Id(id) => persona.is_some_and(|p| p.revealed_secret_ids.contains(id)),
    };
    push_bool(
        "persona_secret_revealed",
        passed,
        json!(actual),
        json!(assertion),
        details,
    );
}

fn evaluate_orchestration(
    assertion: &OrchestrationAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let Some(value) = &evidence.orchestration else {
        push_bool(
            "orchestration",
            false,
            json!(null),
            json!(assertion),
            details,
        );
        return;
    };
    let mut passed = true;
    if let Some(pattern) = assertion.pattern.as_ref().or(assertion.type_name.as_ref()) {
        passed &= value
            .get("type")
            .or_else(|| value.get("pattern"))
            .and_then(Value::as_str)
            == Some(pattern.as_str());
    }
    if let Some(finals) = &assertion.final_agent_in {
        let actual = value.get("final_agent").and_then(Value::as_str);
        passed &= actual.is_some_and(|a| finals.iter().any(|f| f == a));
    }
    if let Some(stages) = assertion.stages {
        let actual = value
            .get("stages")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        passed &= actual == stages;
    }
    push_bool(
        "orchestration",
        passed,
        value.clone(),
        json!(assertion),
        details,
    );
}

fn evaluate_observability(
    assertion: &ObservabilityAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let report = evidence
        .observability
        .as_ref()
        .and_then(|o| o.report.as_ref());
    let Some(report) = report else {
        push_bool(
            "observability",
            false,
            json!(null),
            json!(assertion),
            details,
        );
        return;
    };
    let mut passed = true;
    if let Some(max) = assertion.total_llm_calls_lte {
        passed &= report.summary.total_llm_calls <= max;
    }
    if let Some(max) = assertion.total_cost_usd_lte {
        passed &= report.summary.total_cost_usd <= max;
    }
    for (purpose, path_assertion) in &assertion.purpose_counts {
        let count = report
            .by_purpose
            .iter()
            .find(|metric| metric.dimensions.get("purpose") == Some(purpose))
            .map(|metric| metric.count)
            .unwrap_or(0);
        passed &= path_matches(&json!({"count": count}), path_assertion);
    }
    push_bool(
        "observability",
        passed,
        json!(report.summary),
        json!(assertion),
        details,
    );
}
