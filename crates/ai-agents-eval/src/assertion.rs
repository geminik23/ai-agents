use ai_agents_core::Role;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::evidence::{
    ApprovalDecision, ApprovalEvidence, ApprovalTriggerEvidence, DisambiguationStatus,
    ToolExecutionRecord, TurnEvidence,
};
use crate::judge::{JudgeAssertion, JudgeInput, JudgeResolver};

/// Collection of assertion clauses evaluated against one turn.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Assertion {
    /// Current or expected state name.
    #[serde(default)]
    pub state: Option<String>,
    /// Allowed current state names.
    #[serde(default)]
    pub state_in: Option<Vec<String>>,
    /// State name that must not be current.
    #[serde(default)]
    pub state_not: Option<String>,
    /// State name expected in transition history.
    #[serde(default)]
    pub state_history_contains: Option<String>,
    /// Required substring or substrings in the response.
    #[serde(default)]
    pub response_contains: Option<StringList>,
    /// Response passes when any listed substring is present.
    #[serde(default)]
    pub response_contains_any: Option<StringList>,
    /// Substring or substrings that must be absent from the response.
    #[serde(default)]
    pub response_not_contains: Option<StringList>,
    /// Whether the response must contain non-whitespace text.
    #[serde(default)]
    pub response_not_empty: Option<bool>,
    /// Semantic response-quality judge assertion.
    #[serde(default)]
    pub response_semantic: Option<JudgeAssertion>,
    /// Expected disambiguation status or evidence.
    #[serde(default)]
    pub disambiguation: Option<DisambiguationExpectation>,
    /// Whether no active disambiguation should have occurred.
    #[serde(default)]
    pub no_disambiguation: Option<bool>,
    /// Tool call assertion in string or object form.
    #[serde(default)]
    pub tool_called: Option<ToolCalledAssertion>,
    /// LLM request assertion over captured message roles and content.
    #[serde(default, alias = "llm_messages")]
    pub llm_request: Option<LlmRequestAssertion>,
    /// Approval request assertion with optional trigger and result filters.
    #[serde(default)]
    pub approval_requested: Option<ApprovalAssertion>,
    /// Assertion that no approval request matches the optional filters.
    #[serde(default)]
    pub approval_not_requested: Option<ApprovalAssertion>,
    /// Tool ID that must not appear in tool evidence.
    #[serde(default)]
    pub tool_not_called: Option<String>,
    /// Skill ID expected in skill evidence.
    #[serde(default)]
    pub skill_triggered: Option<String>,
    /// Top-level response metadata key-value expectations.
    #[serde(default)]
    pub metadata_contains: Option<HashMap<String, Value>>,
    /// Dot-path assertion over response metadata.
    #[serde(default)]
    pub metadata_path: Option<PathAssertion>,
    /// Dot-path assertion over runtime context.
    #[serde(default)]
    pub context_path: Option<PathAssertion>,
    /// Fact assertion for actor memory evidence.
    #[serde(default)]
    pub facts_include: Option<FactsAssertion>,
    /// Relationship memory assertion or evidence.
    #[serde(default)]
    pub relationship: Option<RelationshipAssertion>,
    /// Persona secret reveal assertion.
    #[serde(default)]
    pub persona_secret_revealed: Option<SecretAssertion>,
    /// Orchestration metadata assertion or evidence.
    #[serde(default)]
    pub orchestration: Option<OrchestrationAssertion>,
    /// Observability assertion, setting, or report value.
    #[serde(default)]
    pub observability: Option<ObservabilityAssertion>,
    /// LLM judge assertion or resolver for semantic quality.
    #[serde(default)]
    pub judge: Option<JudgeAssertion>,
    /// Child assertions where at least one must pass.
    #[serde(default)]
    pub any: Option<Vec<Assertion>>,
    /// Child assertions where every child must pass.
    #[serde(default)]
    pub all: Option<Vec<Assertion>>,
    /// Child assertion that must fail.
    #[serde(default)]
    pub not: Option<Box<Assertion>>,
}

/// YAML helper accepting either one string or a list of strings.
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

/// String or object form for tool call assertions.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolCalledAssertion {
    Id(String),
    Object(ToolCalledObject),
}

/// Object form for checking tool execution evidence.
///
/// Every configured record predicate must match the same execution. `count` and
/// `count_gte` are evaluated against the number of executions that satisfy all
/// configured record predicates; `tool_called` still requires at least one match.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ToolCalledObject {
    /// Tool ID or requested name required on each matching execution.
    #[serde(default)]
    pub id: Option<String>,
    /// Exact number of executions satisfying every configured record predicate.
    #[serde(default)]
    pub count: Option<usize>,
    /// Minimum number of executions satisfying every configured record predicate.
    #[serde(default)]
    pub count_gte: Option<usize>,
    /// Whether the wrapped tool implementation was invoked.
    #[serde(default)]
    pub executed: Option<bool>,
    /// Whether the operation succeeded.
    #[serde(default)]
    pub success: Option<bool>,
    /// Allowed snake_case source labels, including `llm`, `skill`, and `plan`.
    #[serde(default)]
    pub source_in: Option<Vec<String>>,
    /// Alias for checking executed tool arguments.
    #[serde(default)]
    pub args: Option<PathAssertion>,
    /// Path assertion over original tool arguments.
    #[serde(default)]
    pub args_original: Option<PathAssertion>,
    /// Path assertion over executed tool arguments.
    #[serde(default)]
    pub args_executed: Option<PathAssertion>,
    /// Path assertion over parsed tool output.
    #[serde(default)]
    pub result_path: Option<PathAssertion>,
}

/// Filters and counts for matching complete LLM requests.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct LlmRequestAssertion {
    /// Required substrings in system messages.
    #[serde(default)]
    pub system_contains: Option<StringList>,
    /// Required substrings in user messages.
    #[serde(default)]
    pub user_contains: Option<StringList>,
    /// Required substrings in assistant messages.
    #[serde(default)]
    pub assistant_contains: Option<StringList>,
    /// Required substrings in messages of any role.
    #[serde(default)]
    pub any_contains: Option<StringList>,
    /// Exact number of matching requests required.
    #[serde(default)]
    pub count: Option<usize>,
    /// Minimum number of matching requests required.
    #[serde(default)]
    pub count_gte: Option<usize>,
    /// Maximum number of matching requests required.
    #[serde(default)]
    pub count_lte: Option<usize>,
    /// Whether all role checks must be satisfied by the same request.
    #[serde(default)]
    pub same_request: Option<bool>,
}

/// Boolean or object form for approval assertions.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ApprovalAssertion {
    Bool(bool),
    Object(ApprovalAssertionObject),
}

/// Filters and counts for matching approval evidence.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ApprovalAssertionObject {
    /// Exact number of matching approval records required.
    #[serde(default)]
    pub count: Option<usize>,
    /// Minimum number of matching approval records required.
    #[serde(default)]
    pub count_gte: Option<usize>,
    /// Maximum number of matching approval records required.
    #[serde(default)]
    pub count_lte: Option<usize>,
    /// Trigger fields that must match one approval record.
    #[serde(default)]
    pub trigger: Option<ApprovalTriggerAssertion>,
    /// Decision returned directly by the approval handler.
    #[serde(default)]
    pub raw_decision: Option<ApprovalDecision>,
    /// Decision after runtime resolution.
    #[serde(default)]
    pub effective_decision: Option<ApprovalDecision>,
    /// Exact localized approval message.
    #[serde(default)]
    pub message: Option<String>,
    /// Required substrings in the localized approval message.
    #[serde(default)]
    pub message_contains: Option<StringList>,
    /// Exact effective rejection reason.
    #[serde(default)]
    pub rejection_reason: Option<String>,
    /// Required substring in an effective rejection reason.
    #[serde(default)]
    pub rejection_reason_contains: Option<String>,
    /// Exact effective resolution error.
    #[serde(default)]
    pub error: Option<String>,
    /// Required substring in an effective resolution error.
    #[serde(default)]
    pub error_contains: Option<String>,
    /// Path assertion over original tool arguments.
    #[serde(default, alias = "args_original")]
    pub original_args: Option<PathAssertion>,
    /// Path assertion over modified tool arguments.
    #[serde(default, alias = "args_modified")]
    pub modified_args: Option<PathAssertion>,
    /// Path assertion over effective tool arguments.
    #[serde(default, alias = "args_effective")]
    pub effective_args: Option<PathAssertion>,
}

/// Filters for normalized approval trigger fields.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ApprovalTriggerAssertion {
    /// Normalized trigger type: tool, condition, or state.
    #[serde(default, rename = "type")]
    pub type_name: Option<String>,
    /// Tool or condition name.
    #[serde(default)]
    pub name: Option<String>,
    /// Matched condition expression.
    #[serde(default)]
    pub matched: Option<String>,
    /// Source state for state transition approval.
    #[serde(default)]
    pub from: Option<String>,
    /// Destination state for state transition approval.
    #[serde(default)]
    pub to: Option<String>,
}

/// Expected disambiguation state for an assertion.
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

/// Dot-path assertion used for metadata, context, tools, and metrics.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PathAssertion {
    /// Path used for file lookup, HTTP routing, or dot-path checks.
    pub path: String,
    /// Expected JSON value for exact equality.
    #[serde(default)]
    pub eq: Option<Value>,
    /// JSON value that must not equal the actual value.
    #[serde(default)]
    pub neq: Option<Value>,
    /// Allowed JSON values for membership checks.
    #[serde(default, rename = "in")]
    pub in_values: Option<Vec<Value>>,
    /// String substring or array element expected in the actual value.
    #[serde(default)]
    pub contains: Option<Value>,
    /// Whether the path must exist or be absent.
    #[serde(default)]
    pub exists: Option<bool>,
    /// Numeric lower bound using greater-than-or-equal.
    #[serde(default)]
    pub gte: Option<f64>,
    /// Numeric upper bound using less-than-or-equal.
    #[serde(default)]
    pub lte: Option<f64>,
    /// Numeric lower bound using greater-than.
    #[serde(default)]
    pub gt: Option<f64>,
    /// Numeric upper bound using less-than.
    #[serde(default)]
    pub lt: Option<f64>,
}

/// Assertion over actor facts collected after a turn.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct FactsAssertion {
    /// Actor ID used for this scenario, turn, or assertion.
    #[serde(default)]
    pub actor: Option<String>,
    /// Fact category that must be present.
    #[serde(default)]
    pub category: Option<String>,
    /// Semantic claim checked by a judge.
    #[serde(default)]
    pub semantic: Option<String>,
}

/// Assertion over actor relationship memory evidence.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RelationshipAssertion {
    /// Actor ID used for this scenario, turn, or assertion.
    #[serde(default)]
    pub actor: Option<String>,
    /// Whether the path must exist or be absent.
    #[serde(default)]
    pub exists: Option<bool>,
    /// Relationship perspective to inspect.
    #[serde(default)]
    pub perspective: Option<String>,
    /// Relationship dimension to compare.
    #[serde(default)]
    pub dimension: Option<String>,
    /// Numeric lower bound using greater-than-or-equal.
    #[serde(default)]
    pub gte: Option<f64>,
    /// Numeric upper bound using less-than-or-equal.
    #[serde(default)]
    pub lte: Option<f64>,
    /// Numeric lower bound using greater-than.
    #[serde(default)]
    pub gt: Option<f64>,
    /// Numeric upper bound using less-than.
    #[serde(default)]
    pub lt: Option<f64>,
    /// Expected JSON value for exact equality.
    #[serde(default)]
    pub eq: Option<f64>,
    /// Minimum interaction count expected.
    #[serde(default)]
    pub interaction_count_gte: Option<u64>,
    /// Minimum notable event count expected.
    #[serde(default)]
    pub notable_event_count_gte: Option<usize>,
}

/// Boolean or ID form for persona secret assertions.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SecretAssertion {
    Bool(bool),
    Id(String),
}

/// Assertion over orchestration metadata attached to a turn.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct OrchestrationAssertion {
    /// Expected orchestration pattern label.
    #[serde(default)]
    pub pattern: Option<String>,
    /// Expected orchestration type when YAML uses the type key.
    #[serde(default, rename = "type")]
    pub type_name: Option<String>,
    /// Allowed final agent IDs.
    #[serde(default)]
    pub final_agent_in: Option<Vec<String>>,
    /// Agent IDs expected somewhere in orchestration metadata.
    #[serde(default)]
    pub agents_include: Option<Vec<String>>,
    /// Exact number of pipeline or stage records expected.
    #[serde(default)]
    pub stages: Option<usize>,
}

/// Assertion over the observability report for a turn.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ObservabilityAssertion {
    /// Upper bound for total LLM calls.
    #[serde(default)]
    pub total_llm_calls_lte: Option<u64>,
    /// Upper bound for total tool calls.
    #[serde(default)]
    pub total_tool_calls_lte: Option<u64>,
    /// Upper bound for total tokens.
    #[serde(default)]
    pub total_tokens_lte: Option<u64>,
    /// Upper bound for total estimated cost in USD.
    #[serde(default)]
    pub total_cost_usd_lte: Option<f64>,
    /// Path assertions over counts grouped by purpose.
    #[serde(default)]
    pub purpose_counts: HashMap<String, PathAssertion>,
    /// Path assertions over counts grouped by status.
    #[serde(default)]
    pub status_counts: HashMap<String, PathAssertion>,
    /// Path assertions over counts matching configured dimensions.
    #[serde(default)]
    pub dimension_counts: Vec<ObservabilityDimensionAssertion>,
}

/// Assertion over observability metrics matching all listed dimensions.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ObservabilityDimensionAssertion {
    #[serde(default)]
    pub match_dimensions: HashMap<String, String>,
    #[serde(rename = "assert")]
    pub assertion: PathAssertion,
}

/// Result detail for one evaluated assertion clause.
#[derive(Debug, Clone, Serialize)]
pub struct AssertionResultDetail {
    /// Stable name of the assertion that produced this detail.
    pub assertion: String,
    /// Passed count or boolean result.
    pub passed: bool,
    /// Actual value observed during evaluation.
    pub actual: Value,
    /// Expected assertion object for a generated turn.
    pub expected: Value,
    /// Optional failure message for debugging.
    pub message: Option<String>,
}

/// Final outcome returned by assertion evaluation.
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

/// Runtime inputs needed while evaluating one assertion tree.
#[derive(Clone, Copy)]
pub struct AssertionEvalContext<'a> {
    /// Full assertion-time evidence for this turn.
    pub evidence: &'a TurnEvidence,
    /// Assistant response text or redacted output value.
    pub response: &'a str,
    /// Optional user input for judge prompt context.
    pub user_input: Option<&'a str>,
    /// Optional scenario ID for judge prompt context.
    pub scenario_id: Option<&'a str>,
    /// Optional language label for filtering, metrics, and judge context.
    pub language: Option<&'a str>,
    /// Optional resolver for semantic judge assertions.
    pub judge_resolver: Option<&'a JudgeResolver>,
}

pub async fn evaluate_assertion(
    assertion: &Assertion,
    context: AssertionEvalContext<'_>,
) -> AssertionOutcome {
    let mut details = Vec::new();

    if let Some(children) = &assertion.all {
        for child in children {
            match evaluate_assertion_boxed(child, context).await {
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
            match evaluate_assertion_boxed(child, context).await {
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
        match evaluate_assertion_boxed(child, context).await {
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

    evaluate_simple(assertion, context, &mut details).await;

    if details.iter().any(|d| !d.passed) {
        AssertionOutcome::Failed(details)
    } else {
        AssertionOutcome::Passed(details)
    }
}

fn evaluate_assertion_boxed<'a>(
    assertion: &'a Assertion,
    context: AssertionEvalContext<'a>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = AssertionOutcome> + Send + 'a>> {
    Box::pin(evaluate_assertion(assertion, context))
}

async fn evaluate_simple(
    assertion: &Assertion,
    context: AssertionEvalContext<'_>,
    details: &mut Vec<AssertionResultDetail>,
) {
    let evidence = context.evidence;
    let response = context.response;
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
    if let Some(llm_request) = &assertion.llm_request {
        evaluate_llm_request(llm_request, evidence, details);
    }
    if let Some(approval) = &assertion.approval_requested {
        evaluate_approval_requested(approval, evidence, details);
    }
    if let Some(approval) = &assertion.approval_not_requested {
        evaluate_approval_not_requested(approval, evidence, details);
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
    if let Some(expected) = &assertion.metadata_contains {
        evaluate_metadata_contains(expected, evidence, details);
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
        evaluate_facts(expected, evidence, context.judge_resolver, details).await;
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
        if let Some(resolver) = context.judge_resolver {
            match resolver.resolve(criteria.llm.as_deref()) {
                Ok(judge) => match judge
                    .evaluate_input(
                        JudgeInput {
                            response,
                            user_input: context.user_input,
                            scenario_id: context.scenario_id,
                            language: context.language,
                        },
                        criteria,
                    )
                    .await
                {
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
                },
                Err(error) => details.push(AssertionResultDetail::fail(
                    "judge",
                    json!(error.to_string()),
                    json!("available judge LLM"),
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
    let count = evidence
        .tool_executions
        .iter()
        .filter(|record| tool_execution_matches(record, id, object))
        .count();
    let mut passed = count > 0;
    if let Some(object) = object {
        if let Some(expected) = object.count {
            passed &= count == expected;
        }
        if let Some(expected) = object.count_gte {
            passed &= count >= expected;
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

fn tool_execution_matches(
    record: &ToolExecutionRecord,
    id: Option<&str>,
    object: Option<&ToolCalledObject>,
) -> bool {
    if id.is_some_and(|id| record.tool_id != id && record.requested_name != id) {
        return false;
    }
    let Some(object) = object else {
        return true;
    };

    object
        .executed
        .is_none_or(|executed| record.executed == executed)
        && object
            .success
            .is_none_or(|success| record.success == success)
        && object.source_in.as_ref().is_none_or(|sources| {
            let actual = serde_plain_source(&record.source);
            sources.iter().any(|source| source == &actual)
        })
        && object
            .args
            .as_ref()
            .is_none_or(|path| path_matches(&record.arguments_executed, path))
        && object
            .args_executed
            .as_ref()
            .is_none_or(|path| path_matches(&record.arguments_executed, path))
        && object
            .args_original
            .as_ref()
            .is_none_or(|path| path_matches(&record.arguments_original, path))
        && object.result_path.as_ref().is_none_or(|path| {
            record
                .output
                .as_ref()
                .is_some_and(|value| path_matches(value, path))
        })
}

fn evaluate_llm_request(
    assertion: &LlmRequestAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let checks = llm_message_checks(assertion);
    let same_request = assertion.same_request.unwrap_or(true);
    let matching_count = evidence
        .llm_requests
        .iter()
        .filter(|request| {
            if checks.is_empty() {
                return true;
            }
            if same_request {
                checks.iter().all(|(role, text)| {
                    request.messages.iter().any(|message| {
                        role.is_none_or(|role| message.role == role)
                            && message.content.contains(text)
                    })
                })
            } else {
                checks.iter().any(|(role, text)| {
                    request.messages.iter().any(|message| {
                        role.is_none_or(|role| message.role == role)
                            && message.content.contains(text)
                    })
                })
            }
        })
        .count();
    let content_matches = same_request
        .then_some(matching_count > 0)
        .unwrap_or_else(|| {
            checks.iter().all(|(role, text)| {
                evidence.llm_requests.iter().any(|request| {
                    request.messages.iter().any(|message| {
                        role.is_none_or(|role| message.role == role)
                            && message.content.contains(text)
                    })
                })
            })
        });
    let has_count_constraint =
        assertion.count.is_some() || assertion.count_gte.is_some() || assertion.count_lte.is_some();
    let passed = (checks.is_empty() || content_matches)
        && assertion.count.is_none_or(|count| matching_count == count)
        && assertion
            .count_gte
            .is_none_or(|count| matching_count >= count)
        && assertion
            .count_lte
            .is_none_or(|count| matching_count <= count)
        && (has_count_constraint || matching_count > 0);
    let mut roles: Vec<&str> = checks
        .iter()
        .map(|(role, _)| match role {
            Some(Role::System) => "system",
            Some(Role::User) => "user",
            Some(Role::Assistant) => "assistant",
            _ => "any",
        })
        .collect();
    roles.sort_unstable();
    roles.dedup();
    push_bool(
        "llm_request",
        passed,
        json!({
            "matched_count": matching_count,
            "total_count": evidence.llm_requests.len(),
        }),
        json!({
            "roles": roles,
            "contains_checks": checks.len(),
            "count": assertion.count,
            "count_gte": assertion.count_gte,
            "count_lte": assertion.count_lte,
            "same_request": same_request,
        }),
        details,
    );
}

fn llm_message_checks(assertion: &LlmRequestAssertion) -> Vec<(Option<Role>, String)> {
    let mut checks = Vec::new();
    for (role, contains) in [
        (Some(Role::System), assertion.system_contains.as_ref()),
        (Some(Role::User), assertion.user_contains.as_ref()),
        (Some(Role::Assistant), assertion.assistant_contains.as_ref()),
        (None, assertion.any_contains.as_ref()),
    ] {
        if let Some(contains) = contains {
            checks.extend(contains.items().into_iter().map(|text| (role, text)));
        }
    }
    checks
}

fn evaluate_approval_requested(
    assertion: &ApprovalAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let (matched, expected) = approval_match_counts(assertion, &evidence.approvals);
    let passed = match assertion {
        ApprovalAssertion::Bool(expected) => (matched > 0) == *expected,
        ApprovalAssertion::Object(object) => {
            matched > 0
                && object.count.is_none_or(|count| matched == count)
                && object.count_gte.is_none_or(|count| matched >= count)
                && object.count_lte.is_none_or(|count| matched <= count)
        }
    };
    push_bool(
        "approval_requested",
        passed,
        json!({"matched_count": matched, "total_count": evidence.approvals.len()}),
        expected,
        details,
    );
}

fn evaluate_approval_not_requested(
    assertion: &ApprovalAssertion,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let (matched, expected) = approval_match_counts(assertion, &evidence.approvals);
    let not_requested = matched == 0;
    let passed = match assertion {
        ApprovalAssertion::Bool(expected) => not_requested == *expected,
        ApprovalAssertion::Object(_) => not_requested,
    };
    push_bool(
        "approval_not_requested",
        passed,
        json!({"matched_count": matched, "total_count": evidence.approvals.len()}),
        expected,
        details,
    );
}

fn approval_match_counts(
    assertion: &ApprovalAssertion,
    approvals: &[ApprovalEvidence],
) -> (usize, Value) {
    match assertion {
        ApprovalAssertion::Bool(expected) => (approvals.len(), json!(expected)),
        ApprovalAssertion::Object(object) => {
            let count = approvals
                .iter()
                .filter(|approval| approval_record_matches(approval, object))
                .count();
            (count, approval_assertion_summary(object))
        }
    }
}

fn approval_record_matches(
    approval: &ApprovalEvidence,
    assertion: &ApprovalAssertionObject,
) -> bool {
    assertion
        .trigger
        .as_ref()
        .is_none_or(|trigger| approval_trigger_matches(&approval.trigger, trigger))
        && assertion
            .raw_decision
            .is_none_or(|decision| approval.raw_decision == decision)
        && assertion
            .effective_decision
            .is_none_or(|decision| approval.effective_decision == decision)
        && assertion
            .message
            .as_ref()
            .is_none_or(|message| approval.message == *message)
        && assertion.message_contains.as_ref().is_none_or(|items| {
            items
                .items()
                .iter()
                .all(|item| approval.message.contains(item))
        })
        && assertion
            .rejection_reason
            .as_ref()
            .is_none_or(|reason| approval.rejection_reason.as_ref() == Some(reason))
        && assertion
            .rejection_reason_contains
            .as_ref()
            .is_none_or(|text| {
                approval
                    .rejection_reason
                    .as_ref()
                    .is_some_and(|reason| reason.contains(text))
            })
        && assertion
            .error
            .as_ref()
            .is_none_or(|error| approval.error.as_ref() == Some(error))
        && assertion.error_contains.as_ref().is_none_or(|text| {
            approval
                .error
                .as_ref()
                .is_some_and(|error| error.contains(text))
        })
        && argument_path_matches(
            approval.original_args.as_ref(),
            assertion.original_args.as_ref(),
        )
        && argument_path_matches(
            approval.modified_args.as_ref(),
            assertion.modified_args.as_ref(),
        )
        && argument_path_matches(
            approval.effective_args.as_ref(),
            assertion.effective_args.as_ref(),
        )
}

fn argument_path_matches(value: Option<&Value>, assertion: Option<&PathAssertion>) -> bool {
    assertion.is_none_or(|assertion| value.is_some_and(|value| path_matches(value, assertion)))
}

fn approval_trigger_matches(
    trigger: &ApprovalTriggerEvidence,
    assertion: &ApprovalTriggerAssertion,
) -> bool {
    let (type_name, name, matched, from, to) = match trigger {
        ApprovalTriggerEvidence::Tool { name } => ("tool", Some(name.as_str()), None, None, None),
        ApprovalTriggerEvidence::Condition { name, matched } => (
            "condition",
            Some(name.as_str()),
            Some(matched.as_str()),
            None,
            None,
        ),
        ApprovalTriggerEvidence::State { from, to } => {
            ("state", None, None, from.as_deref(), Some(to.as_str()))
        }
    };
    assertion
        .type_name
        .as_deref()
        .is_none_or(|value| value == type_name)
        && assertion
            .name
            .as_deref()
            .is_none_or(|value| Some(value) == name)
        && assertion
            .matched
            .as_deref()
            .is_none_or(|value| Some(value) == matched)
        && assertion
            .from
            .as_deref()
            .is_none_or(|value| Some(value) == from)
        && assertion
            .to
            .as_deref()
            .is_none_or(|value| Some(value) == to)
}

fn approval_assertion_summary(assertion: &ApprovalAssertionObject) -> Value {
    json!({
        "count": assertion.count,
        "count_gte": assertion.count_gte,
        "count_lte": assertion.count_lte,
        "trigger": assertion.trigger,
        "raw_decision": assertion.raw_decision,
        "effective_decision": assertion.effective_decision,
        "message_check": assertion.message.is_some() || assertion.message_contains.is_some(),
        "rejection_reason_check": assertion.rejection_reason.is_some() || assertion.rejection_reason_contains.is_some(),
        "error_check": assertion.error.is_some() || assertion.error_contains.is_some(),
        "original_args_path": assertion.original_args.as_ref().map(|value| value.path.as_str()),
        "modified_args_path": assertion.modified_args.as_ref().map(|value| value.path.as_str()),
        "effective_args_path": assertion.effective_args.as_ref().map(|value| value.path.as_str()),
    })
}

fn serde_plain_source(source: &crate::evidence::ToolExecutionSource) -> String {
    serde_json::to_string(source)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}
fn evaluate_metadata_contains(
    expected: &HashMap<String, Value>,
    evidence: &TurnEvidence,
    details: &mut Vec<AssertionResultDetail>,
) {
    let metadata = evidence.response_metadata.as_ref();
    let passed = metadata.is_some_and(|metadata| {
        expected
            .iter()
            .all(|(key, expected)| metadata.get(key) == Some(expected))
    }) || (expected.is_empty() && metadata.is_none());
    push_bool(
        "metadata_contains",
        passed,
        json!(metadata),
        json!(expected),
        details,
    );
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

async fn evaluate_facts(
    assertion: &FactsAssertion,
    evidence: &TurnEvidence,
    judge_resolver: Option<&JudgeResolver>,
    details: &mut Vec<AssertionResultDetail>,
) {
    let Some(fact_evidence) = &evidence.facts else {
        push_bool(
            "facts_include",
            false,
            json!(null),
            json!(assertion),
            details,
        );
        return;
    };
    if let Some(actor) = &assertion.actor {
        if fact_evidence.actor_id.as_deref() != Some(actor.as_str()) {
            push_bool(
                "facts_include",
                false,
                json!(fact_evidence.actor_id),
                json!(actor),
                details,
            );
            return;
        }
    }
    let facts: Vec<Value> = fact_evidence
        .facts
        .iter()
        .filter(|fact| {
            assertion.category.as_ref().is_none_or(|category| {
                fact.get("category")
                    .map(|value| value.to_string().trim_matches('"').to_string())
                    .is_some_and(|actual| actual == *category || actual.ends_with(category))
            })
        })
        .cloned()
        .collect();
    let mut passed = !facts.is_empty();
    if let Some(semantic) = &assertion.semantic {
        if let Some(resolver) = judge_resolver {
            match resolver.resolve(None) {
                Ok(judge) => {
                    let criteria = JudgeAssertion {
                        llm: None,
                        pass_threshold: 0.75,
                        criteria: vec![crate::judge::JudgeCriterion::Text(format!(
                            "The fact set supports this claim: {}",
                            semantic
                        ))],
                    };
                    let fact_text = serde_json::to_string(&facts).unwrap_or_default();
                    match judge.evaluate(&fact_text, &criteria).await {
                        Ok(result) => passed &= result.passed,
                        Err(error) => {
                            details.push(AssertionResultDetail::fail(
                                "facts_include",
                                json!(error.to_string()),
                                json!(semantic),
                                "fact semantic judge failed",
                            ));
                            return;
                        }
                    }
                }
                Err(error) => {
                    details.push(AssertionResultDetail::fail(
                        "facts_include",
                        json!(error.to_string()),
                        json!(semantic),
                        "fact semantic judge failed",
                    ));
                    return;
                }
            }
        } else {
            details.push(AssertionResultDetail::fail(
                "facts_include",
                json!(null),
                json!(semantic),
                "semantic fact assertion requires a judge LLM",
            ));
            return;
        }
    }
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
    if let Some(actor) = &assertion.actor {
        if rel.actor_id.as_deref() != Some(actor.as_str()) {
            push_bool(
                "relationship",
                false,
                json!(rel.actor_id),
                json!(actor),
                details,
            );
            return;
        }
    }
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
        let actual = value
            .get("final_agent")
            .or_else(|| value.get("to_agent"))
            .or_else(|| value.get("agent"))
            .and_then(Value::as_str);
        passed &= actual.is_some_and(|a| finals.iter().any(|f| f == a));
    }
    if let Some(required) = &assertion.agents_include {
        let agents = collect_orchestration_agents(value);
        passed &= required
            .iter()
            .all(|agent| agents.iter().any(|a| a == agent));
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

fn collect_orchestration_agents(value: &Value) -> Vec<String> {
    let mut agents = Vec::new();
    collect_agent_strings(value, &mut agents);
    agents.sort();
    agents.dedup();
    agents
}

fn collect_agent_strings(value: &Value, agents: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if matches!(
                    key.as_str(),
                    "agent" | "agent_id" | "id" | "final_agent" | "to_agent" | "from_agent"
                ) {
                    if let Some(text) = value.as_str() {
                        agents.push(text.to_string());
                    }
                }
                collect_agent_strings(value, agents);
            }
        }
        Value::Array(values) => {
            for value in values {
                if let Some(text) = value.as_str() {
                    agents.push(text.to_string());
                }
                collect_agent_strings(value, agents);
            }
        }
        _ => {}
    }
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
    if let Some(max) = assertion.total_tool_calls_lte {
        passed &= report.summary.total_tool_calls <= max;
    }
    if let Some(max) = assertion.total_tokens_lte {
        passed &= report.summary.total_tokens <= max;
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
    for (status, path_assertion) in &assertion.status_counts {
        let count = report
            .configured
            .iter()
            .find(|metric| metric.dimensions.get("status") == Some(status))
            .map(|metric| metric.count)
            .unwrap_or(0);
        passed &= path_matches(&json!({"count": count}), path_assertion);
    }
    for dimension_assertion in &assertion.dimension_counts {
        let count: u64 = report
            .configured
            .iter()
            .filter(|metric| {
                dimension_assertion
                    .match_dimensions
                    .iter()
                    .all(|(key, value)| metric.dimensions.get(key) == Some(value))
            })
            .map(|metric| metric.count)
            .sum();
        passed &= path_matches(&json!({"count": count}), &dimension_assertion.assertion);
    }
    push_bool(
        "observability",
        passed,
        json!(report.summary),
        json!(assertion),
        details,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{FactsEvidence, ToolExecutionSource, TurnObservabilityEvidence};
    use ai_agents_observability::{
        AggregatedMetrics, CostBreakdown, CostStats, LatencyStats, ObservabilityReport,
        ReportSummary, TokenBreakdown, TokenStats,
    };

    fn evidence() -> TurnEvidence {
        TurnEvidence {
            response_metadata: Some(json!({"intent":"greeting","score":0.9})),
            state: Some("ready".to_string()),
            state_history: vec![ai_agents_core::StateTransitionEvent {
                from: "start".to_string(),
                to: "ready".to_string(),
                reason: "test".to_string(),
                timestamp: chrono::Utc::now(),
            }],
            context: json!({"user":{"tier":"vip"}}),
            tool_executions: vec![ToolExecutionRecord {
                call_id: "call-1".to_string(),
                tool_id: "lookup_order".to_string(),
                requested_name: "lookup_order".to_string(),
                source: ToolExecutionSource::Mock,
                state: None,
                actor_id: Some("actor-1".to_string()),
                arguments_original: json!({"id":"ORD-1"}),
                arguments_executed: json!({"id":"ORD-1"}),
                executed: true,
                success: true,
                output: Some(json!({"status":"cancellable"})),
                error: None,
                metadata: None,
                started_at: chrono::Utc::now(),
                duration_ms: 1,
                observability_span_id: None,
            }],
            approvals: vec![ApprovalEvidence {
                request_id: "approval-1".to_string(),
                trigger: ApprovalTriggerEvidence::Tool {
                    name: "transfer".to_string(),
                },
                raw_decision: ApprovalDecision::Modified,
                effective_decision: ApprovalDecision::Modified,
                original_args: Some(json!({"amount": 100, "currency": "USD"})),
                modified_args: Some(json!({"amount": 25, "currency": "USD"})),
                effective_args: Some(json!({"amount": 25, "currency": "USD"})),
                message: "Approve transfer for VIP customer?".to_string(),
                rejection_reason: None,
                error: None,
            }],
            llm_requests: vec![],
            skill: None,
            disambiguation: None,
            facts: Some(FactsEvidence {
                actor_id: Some("actor-1".to_string()),
                facts: vec![
                    json!({"category":"user_preference","content":"Prefers concise answers"}),
                ],
                before_count: None,
                after_count: Some(1),
            }),
            relationship: None,
            persona: None,
            orchestration: Some(json!({
                "type":"pipeline",
                "stages":[{"agent_id":"writer"},{"agent_id":"editor"}],
                "agents":["writer","editor"]
            })),
            observability: None,
        }
    }

    #[tokio::test]
    async fn evaluates_structured_assertions() {
        let mut metadata = HashMap::new();
        metadata.insert("intent".to_string(), json!("greeting"));
        let assertion = Assertion {
            state: Some("ready".to_string()),
            state_history_contains: Some("ready".to_string()),
            response_contains: Some(StringList::One("Hello".to_string())),
            metadata_contains: Some(metadata),
            context_path: Some(PathAssertion {
                path: "user.tier".to_string(),
                eq: Some(json!("vip")),
                ..Default::default()
            }),
            tool_called: Some(ToolCalledAssertion::Object(ToolCalledObject {
                id: Some("lookup_order".to_string()),
                success: Some(true),
                result_path: Some(PathAssertion {
                    path: "status".to_string(),
                    eq: Some(json!("cancellable")),
                    ..Default::default()
                }),
                ..Default::default()
            })),
            facts_include: Some(FactsAssertion {
                actor: Some("actor-1".to_string()),
                category: Some("user_preference".to_string()),
                semantic: None,
            }),
            orchestration: Some(OrchestrationAssertion {
                pattern: Some("pipeline".to_string()),
                agents_include: Some(vec!["writer".to_string(), "editor".to_string()]),
                stages: Some(2),
                ..Default::default()
            }),
            ..Default::default()
        };
        let evidence = evidence();
        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "Hello there",
                user_input: Some("Hello"),
                scenario_id: Some("test"),
                language: Some("en"),
                judge_resolver: None,
            },
        )
        .await;
        assert!(matches!(result, AssertionOutcome::Passed(_)));
    }

    #[test]
    fn tool_called_exposes_plan_as_a_distinct_source_label() {
        let mut evidence = evidence();
        evidence.tool_executions[0].source = ToolExecutionSource::Plan;
        let assertion = ToolCalledAssertion::Object(ToolCalledObject {
            id: Some("lookup_order".to_string()),
            count: Some(1),
            source_in: Some(vec!["plan".to_string()]),
            ..Default::default()
        });
        let mut details = Vec::new();

        evaluate_tool_called(&assertion, &evidence, &mut details);

        assert!(details[0].passed);
        assert_eq!(details[0].actual, json!(1));

        let llm_assertion = ToolCalledAssertion::Object(ToolCalledObject {
            id: Some("lookup_order".to_string()),
            source_in: Some(vec!["llm".to_string()]),
            ..Default::default()
        });
        let mut llm_details = Vec::new();
        evaluate_tool_called(&llm_assertion, &evidence, &mut llm_details);
        assert!(!llm_details[0].passed);
    }

    #[test]
    fn tool_called_path_predicates_do_not_match_across_execution_records() {
        let mut evidence = evidence();
        let first = &mut evidence.tool_executions[0];
        first.tool_id = "calculator".to_string();
        first.requested_name = "calculator".to_string();
        first.source = ToolExecutionSource::Plan;
        first.arguments_original = json!({"expression":"18 * 7"});
        first.arguments_executed = json!({"expression":"1 + 1"});
        first.output = Some(json!({"result":2}));

        let mut second = first.clone();
        second.call_id = "call-2".to_string();
        second.arguments_original = json!({"expression":"2 + 2"});
        second.arguments_executed = json!({"expression":"18 * 7"});
        second.output = Some(json!({"result":126}));
        evidence.tool_executions.push(second);

        let assertion = ToolCalledAssertion::Object(ToolCalledObject {
            id: Some("calculator".to_string()),
            executed: Some(true),
            success: Some(true),
            source_in: Some(vec!["plan".to_string()]),
            args_original: Some(PathAssertion {
                path: "expression".to_string(),
                eq: Some(json!("18 * 7")),
                ..Default::default()
            }),
            args_executed: Some(PathAssertion {
                path: "expression".to_string(),
                eq: Some(json!("18 * 7")),
                ..Default::default()
            }),
            result_path: Some(PathAssertion {
                path: "result".to_string(),
                eq: Some(json!(126)),
                ..Default::default()
            }),
            ..Default::default()
        });
        let mut details = Vec::new();

        evaluate_tool_called(&assertion, &evidence, &mut details);

        assert!(!details[0].passed);
        assert_eq!(details[0].actual, json!(0));
    }

    #[test]
    fn tool_called_checks_both_executed_argument_aliases_when_both_are_configured() {
        let mut evidence = evidence();
        let record = &mut evidence.tool_executions[0];
        record.arguments_executed = json!({"expression":"18 * 7"});

        let assertion = ToolCalledAssertion::Object(ToolCalledObject {
            id: Some("lookup_order".to_string()),
            args: Some(PathAssertion {
                path: "expression".to_string(),
                eq: Some(json!("18 * 7")),
                ..Default::default()
            }),
            args_executed: Some(PathAssertion {
                path: "expression".to_string(),
                eq: Some(json!("2 + 2")),
                ..Default::default()
            }),
            ..Default::default()
        });
        let mut details = Vec::new();

        evaluate_tool_called(&assertion, &evidence, &mut details);

        assert!(!details[0].passed);
        assert_eq!(details[0].actual, json!(0));
    }

    #[test]
    fn tool_called_count_counts_complete_record_matches() {
        let mut evidence = evidence();
        let first = &mut evidence.tool_executions[0];
        first.tool_id = "calculator".to_string();
        first.requested_name = "calculator".to_string();
        first.arguments_original = json!({"expression":"18 * 7"});
        first.arguments_executed = json!({"expression":"18 * 7"});
        first.output = Some(json!({"result":126}));

        let mut second = first.clone();
        second.call_id = "call-2".to_string();
        second.arguments_executed = json!({"expression":"2 + 2"});
        second.output = Some(json!({"result":4}));
        evidence.tool_executions.push(second);

        let assertion = ToolCalledAssertion::Object(ToolCalledObject {
            id: Some("calculator".to_string()),
            count: Some(1),
            executed: Some(true),
            success: Some(true),
            args_executed: Some(PathAssertion {
                path: "expression".to_string(),
                eq: Some(json!("18 * 7")),
                ..Default::default()
            }),
            result_path: Some(PathAssertion {
                path: "result".to_string(),
                eq: Some(json!(126)),
                ..Default::default()
            }),
            ..Default::default()
        });
        let mut details = Vec::new();

        evaluate_tool_called(&assertion, &evidence, &mut details);

        assert!(details[0].passed);
        assert_eq!(details[0].actual, json!(1));
    }

    #[tokio::test]
    async fn llm_request_matches_roles_within_one_request_without_serializing_content() {
        let mut evidence = evidence();
        evidence.llm_requests = vec![
            crate::evidence::LlmRequestEvidence {
                messages: vec![
                    crate::evidence::LlmMessageEvidence {
                        role: Role::System,
                        content: "persona marker and reasoning marker".to_string(),
                    },
                    crate::evidence::LlmMessageEvidence {
                        role: Role::User,
                        content: "current question".to_string(),
                    },
                ],
            },
            crate::evidence::LlmRequestEvidence {
                messages: vec![crate::evidence::LlmMessageEvidence {
                    role: Role::Assistant,
                    content: "historical answer".to_string(),
                }],
            },
        ];
        let assertion = Assertion {
            llm_request: Some(LlmRequestAssertion {
                system_contains: Some(StringList::Many(vec![
                    "persona marker".to_string(),
                    "reasoning marker".to_string(),
                ])),
                user_contains: Some(StringList::One("current question".to_string())),
                any_contains: Some(StringList::One("reasoning marker".to_string())),
                count: Some(1),
                count_gte: Some(1),
                count_lte: Some(1),
                same_request: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };

        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "ok",
                user_input: None,
                scenario_id: None,
                language: None,
                judge_resolver: None,
            },
        )
        .await;

        let AssertionOutcome::Passed(details) = result else {
            panic!("expected LLM request assertion to pass");
        };
        let serialized = serde_json::to_string(&details).unwrap();
        assert!(!serialized.contains("persona marker"));
        assert!(!serialized.contains("reasoning marker"));
        assert!(!serialized.contains("current question"));
        assert!(!serialized.contains("historical answer"));
    }

    #[tokio::test]
    async fn llm_request_does_not_combine_role_checks_across_requests() {
        let mut evidence = evidence();
        evidence.llm_requests = vec![
            crate::evidence::LlmRequestEvidence {
                messages: vec![crate::evidence::LlmMessageEvidence {
                    role: Role::System,
                    content: "persona marker".to_string(),
                }],
            },
            crate::evidence::LlmRequestEvidence {
                messages: vec![crate::evidence::LlmMessageEvidence {
                    role: Role::User,
                    content: "current question".to_string(),
                }],
            },
        ];
        let assertion = Assertion {
            llm_request: Some(LlmRequestAssertion {
                system_contains: Some(StringList::One("persona marker".to_string())),
                user_contains: Some(StringList::One("current question".to_string())),
                same_request: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };

        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "ok",
                user_input: None,
                scenario_id: None,
                language: None,
                judge_resolver: None,
            },
        )
        .await;

        assert!(matches!(result, AssertionOutcome::Failed(_)));
    }

    #[tokio::test]
    async fn approval_requested_matches_all_constraints_on_one_record() {
        let assertion = Assertion {
            approval_requested: Some(ApprovalAssertion::Object(ApprovalAssertionObject {
                count: Some(1),
                trigger: Some(ApprovalTriggerAssertion {
                    type_name: Some("tool".to_string()),
                    name: Some("transfer".to_string()),
                    ..Default::default()
                }),
                raw_decision: Some(ApprovalDecision::Modified),
                effective_decision: Some(ApprovalDecision::Modified),
                message_contains: Some(StringList::One("VIP".to_string())),
                original_args: Some(PathAssertion {
                    path: "amount".to_string(),
                    eq: Some(json!(100)),
                    ..Default::default()
                }),
                effective_args: Some(PathAssertion {
                    path: "amount".to_string(),
                    eq: Some(json!(25)),
                    ..Default::default()
                }),
                ..Default::default()
            })),
            ..Default::default()
        };
        let evidence = evidence();
        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "ok",
                user_input: None,
                scenario_id: None,
                language: None,
                judge_resolver: None,
            },
        )
        .await;

        assert!(matches!(result, AssertionOutcome::Passed(_)));
    }

    #[tokio::test]
    async fn approval_constraints_do_not_match_across_records_and_summaries_are_redacted() {
        let mut evidence = evidence();
        evidence.approvals.push(ApprovalEvidence {
            request_id: "approval-2".to_string(),
            trigger: ApprovalTriggerEvidence::Tool {
                name: "transfer".to_string(),
            },
            raw_decision: ApprovalDecision::Rejected,
            effective_decision: ApprovalDecision::Rejected,
            original_args: Some(json!({"amount": 9999})),
            modified_args: None,
            effective_args: None,
            message: "Secret second message".to_string(),
            rejection_reason: Some("private rejection".to_string()),
            error: None,
        });
        let assertion = Assertion {
            approval_requested: Some(ApprovalAssertion::Object(ApprovalAssertionObject {
                raw_decision: Some(ApprovalDecision::Rejected),
                effective_args: Some(PathAssertion {
                    path: "amount".to_string(),
                    eq: Some(json!(25)),
                    ..Default::default()
                }),
                message_contains: Some(StringList::One("Secret".to_string())),
                ..Default::default()
            })),
            ..Default::default()
        };
        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "ok",
                user_input: None,
                scenario_id: None,
                language: None,
                judge_resolver: None,
            },
        )
        .await;

        let AssertionOutcome::Failed(details) = result else {
            panic!("expected failed approval assertion");
        };
        let serialized = serde_json::to_string(&details).unwrap();
        assert!(!serialized.contains("Secret"));
        assert!(!serialized.contains("private rejection"));
        assert!(!serialized.contains("25"));
        assert!(!serialized.contains("9999"));
    }

    #[tokio::test]
    async fn approval_not_requested_uses_trigger_filters() {
        let assertion = Assertion {
            approval_not_requested: Some(ApprovalAssertion::Object(ApprovalAssertionObject {
                trigger: Some(ApprovalTriggerAssertion {
                    type_name: Some("state".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            })),
            ..Default::default()
        };
        let evidence = evidence();
        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "ok",
                user_input: None,
                scenario_id: None,
                language: None,
                judge_resolver: None,
            },
        )
        .await;

        assert!(matches!(result, AssertionOutcome::Passed(_)));
    }

    #[tokio::test]
    async fn facts_actor_mismatch_fails() {
        let assertion = Assertion {
            facts_include: Some(FactsAssertion {
                actor: Some("other".to_string()),
                category: Some("user_preference".to_string()),
                semantic: None,
            }),
            ..Default::default()
        };
        let evidence = evidence();
        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "ok",
                user_input: None,
                scenario_id: None,
                language: None,
                judge_resolver: None,
            },
        )
        .await;
        assert!(matches!(result, AssertionOutcome::Failed(_)));
    }

    #[tokio::test]
    async fn observability_dimension_counts_match_configured_metrics() {
        let mut evidence = evidence();
        let mut dimensions = HashMap::new();
        dimensions.insert("background".to_string(), "true".to_string());
        dimensions.insert("maintenance".to_string(), "facts".to_string());
        let metric = AggregatedMetrics {
            dimensions,
            count: 2,
            errors: 0,
            latency: LatencyStats::default(),
            tokens: TokenStats::default(),
            cost: CostStats::default(),
        };
        evidence.observability = Some(TurnObservabilityEvidence {
            trace_id: Some("trace".to_string()),
            span_ids: vec!["span".to_string()],
            report: Some(ObservabilityReport {
                summary: ReportSummary::default(),
                configured: vec![metric],
                by_model: vec![],
                by_purpose: vec![],
                by_language: vec![],
                by_state: vec![],
                by_agent: vec![],
                by_orchestration_pattern: vec![],
                cost_breakdown: CostBreakdown::default(),
                token_breakdown: TokenBreakdown::default(),
                dropped_events: 0,
            }),
        });
        let mut match_dimensions = HashMap::new();
        match_dimensions.insert("background".to_string(), "true".to_string());
        let assertion = Assertion {
            observability: Some(ObservabilityAssertion {
                dimension_counts: vec![ObservabilityDimensionAssertion {
                    match_dimensions,
                    assertion: PathAssertion {
                        path: "count".to_string(),
                        gte: Some(2.0),
                        ..Default::default()
                    },
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let result = evaluate_assertion(
            &assertion,
            AssertionEvalContext {
                evidence: &evidence,
                response: "ok",
                user_input: None,
                scenario_id: None,
                language: None,
                judge_resolver: None,
            },
        )
        .await;

        assert!(matches!(result, AssertionOutcome::Passed(_)));
    }
}
