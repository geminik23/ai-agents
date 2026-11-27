// One trial uses the same selected-file host, fixed checker and controller as the application.
// The campaign prepares the disposable inputs; only this evaluation installs a priced profile.
use super::*;
use ai_agents::ApprovalResult;
use ai_agents::autonomy::*;
use ai_agents::tools::{ToolApprovalStatus, ToolCallSource};
use serde_json::Value;
use std::io::{self, Write};

const MODEL: &str = "gpt-5.4-mini-2026-03-17";
const ENDPOINT: &str = "https://api.openai.com/v1/";
const SOURCE: &str = "src/lib.rs";
const OBJECTIVE: &str = "Repair merge_ranges for nonempty half-open integer ranges. Return the sorted union, merging touching ranges. A shuffled input [(8,12),(2,4),(1,10)] unexpectedly splits a connected range. Preserve the public function signature. Validate the repair with the workspace acceptance check.";

fn ceilings() -> AutonomyHostCeilings {
    AutonomyHostCeilings {
        max_cost_usd: Some(UsdAmount::parse("2").expect("valid trial budget")),
        max_llm_calls: Some(16),
        ..Default::default()
    }
}

pub(super) fn configure(root: &Path, checker: &Path) -> Result<WorkspaceSetup> {
    let mut setup = prepare(root, Path::new(SOURCE), checker, "openai", MODEL, None)?;
    setup.assume_exclusive_workspace()?;
    setup.set_max_output_tokens(8192)?;
    let LLMConfigOrSelector::Config(config) = &mut setup.spec.llm else {
        return Err("expected a single priced provider".into());
    };
    config.base_url = Some(ENDPOINT.into());
    config.extra.insert(
        "declared_pricing".into(),
        json!({
            "schedule_id":"openai-gpt-5.4-mini-standard", "model":MODEL, "endpoint":ENDPOINT,
            "server_input_token_limit":400000, "server_output_token_limit":128000,
            "input_micro_usd_per_million":750000, "output_micro_usd_per_million":4500000,
            "request_fee_micro_usd":0, "max_internal_attempts":1, "complete_usage":true
        }),
    );
    setup.spec.autonomy.defaults.max_cost_usd = ceilings().max_cost_usd;
    setup.spec.observability.enabled = true;
    setup.spec.observability.export.write_report = false;
    setup.spec.validate()?;
    resolve_profile(
        &setup.spec.autonomy,
        AutonomyScope::Task,
        None,
        None,
        None,
        Some(OBJECTIVE),
        &ceilings(),
    )?;
    Ok(setup)
}

pub async fn trial(root: &Path, checker: &Path, live: bool) -> Result<()> {
    let setup = configure(root, checker)?;
    let builder = setup
        .builder()?
        .auto_configure_features()?
        .autonomy_extensions(setup.extensions()?);
    if !live {
        println!("Repair eval configuration ready: {MODEL}, 16 model attempts, USD 2 per trial.");
        return Ok(());
    }
    let agent = Arc::new(builder.auto_configure_llms()?.build()?);
    agent.set_command_runner(setup.checker_runner().ok_or("missing trusted checker")?);
    let telemetry = agent
        .observability()
        .ok_or("missing trial usage observer")?;
    let identity = "code-agent-repair-eval.v1";
    let store = Arc::new(ScopedTaskRunStore::in_memory(
        agent.info().id,
        None,
        identity.into(),
    )?);
    let runner = AutonomyRunner::try_new(
        agent,
        setup.spec.autonomy.clone(),
        ceilings(),
        store.clone(),
        identity.into(),
    )?;
    let source = setup.source_file().ok_or("missing selected file")?;
    let original = fs::read(source)?;
    let mut result = runner.run(OBJECTIVE, None).await?;
    while result.run.status == TaskRunStatus::Paused {
        let interaction = runner
            .task_interaction(&result.run.key.run_id, result.run.revision)
            .await?
            .ok_or("missing review")?;
        let TaskHostPrompt::Approval { trigger, .. } = interaction.prompt else {
            eprintln!(
                "Task needs guidance; ending trial at {:?}",
                result.run.pause_reason
            );
            runner
                .cancel_task(&result.run.key.run_id, result.run.revision)
                .await?;
            break;
        };
        println!(
            "Review request {} at revision {}: {}",
            interaction.request_id, result.run.revision, trigger
        );
        println!("Type approve to execute this request, or reject:");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        let decision = if answer.trim() == "approve" {
            ApprovalResult::Approved
        } else {
            ApprovalResult::Rejected { reason: None }
        };
        result = runner
            .resume(
                &result.run.key.run_id,
                result.run.revision,
                TaskResumeInput::Approval {
                    request_id: interaction.request_id,
                    result: decision,
                },
            )
            .await?;
    }
    let snapshot = store
        .load(&result.run.key.run_id)
        .await?
        .ok_or("missing task journal")?;
    let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
    let charged = payload.counters.charged_micro_usd;
    let outstanding: u64 = payload
        .reservations
        .iter()
        .filter(|r| r.state != TaskEffectState::Completed)
        .map(|r| r.reserved_micro_usd.saturating_sub(r.charged_micro_usd))
        .sum();
    let evidence = repair_evidence(&result.run, source)?;
    let costs: Vec<_> = payload
        .reservations
        .iter()
        .filter(|r| r.reserved_micro_usd > 0)
        .map(|r| {
            json!({
                "reserved_micro_usd": r.reserved_micro_usd,
                "charged_micro_usd": r.charged_micro_usd,
                "state": r.state,
            })
        })
        .collect();
    let actions: Vec<_> = result
        .run
        .evidence
        .tool_calls
        .iter()
        .map(|e| {
            let record = &e.record;
            let mut action = json!({
                "tool": record.canonical_id, "executed": record.executed, "success": record.success,
                "approval": record.approval.as_ref().map(|r| &r.status),
            });
            if record.canonical_id == "command" && record.executed {
                let output = serde_json::from_str::<Value>(&record.output).unwrap_or(Value::Null);
                action["exit_code"] = output["exit_code"].clone();
            }
            action
        })
        .collect();
    let usages = trial_usage(&telemetry, payload.counters.llm_attempts);
    let changed = fs::read(source)? != original;
    let pass = result.run.status == TaskRunStatus::Completed
        && changed
        && evidence
        && charged > 0
        && outstanding == 0;
    println!(
        "TRIAL {}",
        json!({
            "status": result.run.status, "reason": result.run.stop_reason, "passed": pass,
            "run_id": result.run.key.run_id, "model": MODEL, "usage": usages, "ledger": costs,
            "actions": actions, "changed": changed, "ordered_evidence": evidence,
            "llm_attempts": payload.counters.llm_attempts, "charged_micro_usd": charged,
            "outstanding_micro_usd": outstanding,
        })
    );
    if !pass {
        return Err("repair trial did not meet the acceptance criteria".into());
    }
    Ok(())
}

// Count a read, real approved mutation and later approved checker on this run's selected file.
pub(super) fn repair_evidence(run: &TaskRun, source: &Path) -> Result<bool> {
    let mut read = false;
    let mut edited = false;
    let mut checked = false;
    for event in &run.evidence.tool_calls {
        let record = &event.record;
        if event.run_id != run.key.run_id || !record.executed {
            continue;
        }
        let approved = record
            .approval
            .as_ref()
            .is_some_and(|r| matches!(r.status, ToolApprovalStatus::Approved));
        match record.canonical_id.as_str() {
            "file_read"
                if record.success
                    && record.executed_arguments["path"].as_str() == source.to_str() =>
            {
                read = true
            }
            "file_edit" => {
                if !read
                    || !approved
                    || record.executed_arguments["path"].as_str() != source.to_str()
                {
                    return Ok(false);
                }
                if record.success
                    && serde_json::from_str::<Value>(&record.output)?["mutation_performed"] == true
                {
                    edited = true;
                    checked = false;
                }
            }
            "command" => {
                if !approved || !matches!(record.source, ToolCallSource::Task) {
                    return Ok(false);
                }
                let out: Value = serde_json::from_str(&record.output)?;
                checked = edited
                    && record.success
                    && !record.timed_out
                    && !record.cancelled
                    && out["exit_code"] == 0
                    && out["termination"] == "exited";
            }
            _ => {}
        }
    }
    let valid = run
        .evidence
        .records
        .iter()
        .rev()
        .find(|r| r.origin == "controller")
        .and_then(|r| r.data["validations"].as_array())
        .is_some_and(|checks| {
            checks
                .iter()
                .any(|c| c["check_id"] == "acceptance" && c["outcome"] == "pass")
        });
    Ok(checked && valid)
}

// Each trial owns a fresh manager. Numeric observations are available with raw export disabled.
fn trial_usage(manager: &ai_agents::observability::ObservabilityManager, attempts: u64) -> Value {
    let report = manager.generate_report();
    let calls = report.summary.total_llm_calls;
    json!({
        "observed_calls": calls, "attempts_observed": calls == attempts && report.dropped_events == 0,
        "input_tokens": (calls > 0).then_some(report.token_breakdown.total_input),
        "output_tokens": (calls > 0).then_some(report.token_breakdown.total_output),
        "estimated_events": report.token_breakdown.estimated_events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents::observability::{ObservabilityConfig, ObservabilityManager};

    #[test]
    fn trial_usage_uses_numeric_observations_with_raw_export_disabled() {
        let config = ObservabilityConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(!config.export.write_raw_events);
        let manager = ObservabilityManager::new(config);
        assert_eq!(trial_usage(&manager, 1)["input_tokens"], Value::Null);
        let event = json!({
            "trace_id":"trial","span_id":"call","turn_id":"turn","agent_id":"code-agent",
            "event_type":{"llm_call":{"provider":"openai","model":MODEL,"streaming":false}},
            "purpose":"main_response","status":"success","duration_ms":1,
            "timestamp":"2026-01-01T00:00:00Z",
            "tokens":{"input_tokens":23,"output_tokens":7,"total_tokens":30,"source":"provider"}
        });
        manager.record_event(serde_json::from_value(event).unwrap());
        assert!(manager.raw_events().is_empty());
        let usage = trial_usage(&manager, 1);
        assert_eq!(usage["input_tokens"], 23);
        assert_eq!(usage["output_tokens"], 7);
        assert_eq!(usage["attempts_observed"], true);
    }
}
