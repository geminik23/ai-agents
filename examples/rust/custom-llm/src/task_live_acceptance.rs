// Pure consumer checks supplement suite assertions without changing runtime completion or exporting exact evidence.
use ai_agents::autonomy::{GateOutcome, TaskRunStatus};
use ai_agents_eval::{EvalResult, TaskEvidence};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Serialize)]
pub struct Check {
    pub id: &'static str,
    pub passed: bool,
}
#[derive(Debug, Serialize)]
pub struct Acceptance {
    pub checks: Vec<Check>,
    pub passed: bool,
}
fn add(checks: &mut Vec<Check>, id: &'static str, passed: bool) {
    checks.push(Check { id, passed });
}

/// A missing task is not a successful empty observation; these checks consume the actual live result.
pub fn evaluate(result: &EvalResult) -> Acceptance {
    let mut checks = vec![];
    for scenario in &result.scenarios {
        let tasks: Vec<&TaskEvidence> = scenario
            .attempts
            .iter()
            .flat_map(|a| &a.turns)
            .filter_map(|t| t.evidence.task.as_ref())
            .collect();
        add(&mut checks, "task_evidence_present", !tasks.is_empty());
        let Some(last) = tasks.last() else { continue };
        add(
            &mut checks,
            "acknowledged_completed",
            last.inspection.run.status == TaskRunStatus::Completed,
        );
        add(
            &mut checks,
            "events_complete",
            tasks.iter().all(|t| t.events_complete),
        );
        if scenario.id == "live-research" {
            let run = &last.inspection.run;
            let reads: Vec<_> = run
                .evidence
                .tool_calls
                .iter()
                .filter(|e| {
                    e.record.canonical_id == "research_catalog"
                        && e.record.executed_arguments["op"] == "read"
                })
                .collect();
            add(
                &mut checks,
                "research_reads_successful",
                reads.len() >= 2
                    && reads.iter().all(|e| {
                        e.record.executed && e.record.success && !e.record.output_truncated
                    }),
            );
            let answer = last
                .inspection
                .evidence
                .current
                .as_ref()
                .and_then(|c| serde_json::from_str::<Value>(&c.value.response).ok());
            add(&mut checks, "research_answer_json", answer.is_some());
            add(
                &mut checks,
                "research_sources_array",
                answer.as_ref().is_some_and(|v| v["sources"].is_array()),
            );
            add(
                &mut checks,
                "research_comparison_present",
                answer.as_ref().is_some_and(|v| v["comparison"].is_string()),
            );
            add(
                &mut checks,
                "research_uncertainty_present",
                answer
                    .as_ref()
                    .is_some_and(|v| v["uncertainty"].is_string()),
            );
            let receipt_ids = reads
                .iter()
                .filter_map(|e| e.record.executed_arguments["id"].as_str())
                .collect::<std::collections::BTreeSet<_>>();
            let answer_ids = answer
                .as_ref()
                .and_then(|v| v["sources"].as_array())
                .map(|rows| {
                    rows.iter()
                        .filter_map(|r| r["id"].as_str())
                        .collect::<std::collections::BTreeSet<_>>()
                });
            add(
                &mut checks,
                "research_citation_set",
                answer_ids.as_ref() == Some(&receipt_ids),
            );
            add(
                &mut checks,
                "research_gate_passed",
                last.inspection
                    .evidence
                    .validations
                    .iter()
                    .any(|v| v.check_id == "corpus" && v.outcome == GateOutcome::Pass),
            );
        }
        if scenario.id == "live-support" {
            let before = tasks.first().unwrap();
            let before_calls = &before.inspection.run.evidence.tool_calls;
            add(
                &mut checks,
                "support_reference_before_question",
                before_calls.iter().any(|e| {
                    e.record.canonical_id == "domain"
                        && e.record.executed
                        && e.record.success
                        && e.record.executed_arguments["op"] == "source"
                        && e.record.executed_arguments["value"] == "ticket://request-context"
                }),
            );
            add(
                &mut checks,
                "support_domain_operations_succeeded",
                before_calls
                    .iter()
                    .filter(|e| e.record.canonical_id == "domain")
                    .all(|e| e.record.executed && e.record.success),
            );

            let (a, b) = (&before.inspection.run, &last.inspection.run);
            add(
                &mut checks,
                "support_exact_retained_run",
                tasks.len() == 2
                    && a.status == TaskRunStatus::Paused
                    && a.key == b.key
                    && b.revision > a.revision,
            );
            add(
                &mut checks,
                "support_original_limits",
                serde_json::to_value(&a.limits).ok() == serde_json::to_value(&b.limits).ok(),
            );
            add(
                &mut checks,
                "support_cumulative_usage",
                b.counters.llm_attempts > a.counters.llm_attempts,
            );
            let before_counters = serde_json::to_value(&a.counters).unwrap();
            let after_counters = serde_json::to_value(&b.counters).unwrap();
            add(
                &mut checks,
                "support_all_accounting_preserved",
                before_counters.as_object().is_some_and(|fields| {
                    fields.iter().all(|(k, v)| {
                        v.as_u64()
                            .is_some_and(|n| after_counters[k].as_u64().is_some_and(|m| m >= n))
                    })
                }),
            );
            let old = serde_json::to_value(&a.evidence.tool_calls).unwrap_or(Value::Null);
            let new = serde_json::to_value(&b.evidence.tool_calls).unwrap_or(Value::Null);
            add(
                &mut checks,
                "support_completed_prefix",
                old.as_array()
                    .zip(new.as_array())
                    .is_some_and(|(a, b)| !a.is_empty() && b.starts_with(a)),
            );
            add(
                &mut checks,
                "support_actual_question_without_default",
                b.evidence.tool_calls.iter().any(|e| {
                    e.record.canonical_id == "ask_user"
                        && e.record.executed
                        && e.record.success
                        && e.record
                            .executed_arguments
                            .get("default")
                            .is_none_or(Value::is_null)
                }),
            );
            let records: Vec<_> = b
                .evidence
                .tool_calls
                .iter()
                .filter(|e| {
                    e.record.canonical_id == "domain" && e.record.executed && e.record.success
                })
                .collect();
            let source = records
                .iter()
                .filter(|e| e.record.executed_arguments["op"] == "source")
                .count();
            let resolve = records
                .iter()
                .filter(|e| e.record.executed_arguments["op"] == "resolve")
                .count();
            add(
                &mut checks,
                "support_single_reference_and_resolution",
                source == 1 && resolve == 1 && records.len() == 2,
            );
            let paused_unresolved = a
                .evidence
                .tool_calls
                .iter()
                .filter(|e| e.record.canonical_id == "domain" && e.record.executed)
                .all(|e| {
                    serde_json::from_str::<Value>(&e.record.output).is_ok_and(|v| {
                        v["ticket"] == "open" && v["sources"] == json!(["ticket://request-context"])
                    })
                });
            let resolved = records.last().is_some_and(|e| {
                serde_json::from_str::<Value>(&e.record.output).is_ok_and(|v| {
                    v["ticket"] == "resolved" && v["sources"] == json!(["ticket://request-context"])
                })
            });
            add(
                &mut checks,
                "support_business_state",
                paused_unresolved && resolved,
            );
        }
        if scenario.id == "live-operations" {
            let calls = &last.inspection.run.evidence.tool_calls;
            add(
                &mut checks,
                "operations_incident_invoked",
                calls.iter().any(|e| {
                    e.record.canonical_id == "domain"
                        && e.record.executed
                        && e.record.success
                        && e.record.executed_arguments["op"] == "incident"
                }),
            );

            let records: Vec<_> = last
                .inspection
                .run
                .evidence
                .records
                .iter()
                .filter(|r| r.origin == "controller")
                .collect();
            let mut stages = Vec::new();
            for record in &records {
                if stages.last() != Some(&record.stage.as_deref()) {
                    stages.push(record.stage.as_deref());
                }
            }
            add(
                &mut checks,
                "operations_required_stage_order",
                stages == vec![Some("remediate"), Some("verify"), Some("review")],
            );
            add(
                &mut checks,
                "operations_no_early_terminal",
                records
                    .iter()
                    .filter(|r| r.stage.as_deref() != Some("review"))
                    .all(|r| r.data["decision"]["action"] != "terminal"),
            );
            add(
                &mut checks,
                "operations_truthful_escalation",
                last.inspection
                    .evidence
                    .validations
                    .iter()
                    .any(|v| v.check_id == "health" && v.outcome == GateOutcome::Fail)
                    && last
                        .inspection
                        .evidence
                        .validations
                        .iter()
                        .any(|v| v.check_id == "records" && v.outcome == GateOutcome::Pass),
            );
        }
    }
    let passed = !checks.is_empty()
        && result.total > 0
        && result.failed == 0
        && result.skipped == 0
        && checks.iter().all(|c| c.passed);
    Acceptance { checks, passed }
}
