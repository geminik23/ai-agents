use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use crate::Result;
use crate::suite::{EvalResult, ScenarioStatus};

pub fn write_outputs(result: &EvalResult, output_dir: &Path, write_junit: bool) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    fs::write(
        output_dir.join("summary.json"),
        serde_json::to_string_pretty(result)?,
    )?;
    fs::write(output_dir.join("summary.md"), render_summary(result))?;
    fs::write(output_dir.join("failures.md"), render_failures(result))?;
    write_jsonl(result, &output_dir.join("per_scenario.jsonl"))?;
    if write_junit {
        fs::write(output_dir.join("junit.xml"), render_junit(result))?;
    }
    Ok(())
}

fn write_jsonl(result: &EvalResult, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    for scenario in &result.scenarios {
        writeln!(file, "{}", serde_json::to_string(scenario)?)?;
    }
    Ok(())
}

fn render_summary(result: &EvalResult) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Eval Results: {}\n\n", result.suite));
    out.push_str(&format!("Agent: `{}`\n\n", result.agent));
    out.push_str(&format!(
        "Passed: {}/{} ({:.1}%) | Failed: {} | Skipped: {}\n\n",
        result.passed,
        result.total,
        if result.total == 0 {
            0.0
        } else {
            result.passed as f64 / result.total as f64 * 100.0
        },
        result.failed,
        result.skipped
    ));
    out.push_str(&format!(
        "Duration: {}ms | Avg latency: {:.1}ms/turn\n\n",
        result.duration_ms, result.metrics.avg_latency_ms
    ));
    out.push_str("## Scenarios\n\n| Scenario | Status | Attempts | Duration |\n|----------|--------|----------|----------|\n");
    for scenario in &result.scenarios {
        out.push_str(&format!(
            "| `{}` | {} | {} | {}ms |\n",
            scenario.id,
            status_label(&scenario.status),
            scenario.attempts.len(),
            scenario.duration_ms
        ));
    }
    out
}

fn render_failures(result: &EvalResult) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Eval Failures: {}\n\n", result.suite));
    for scenario in &result.scenarios {
        if !scenario.status.is_failed() && !scenario.status.is_error() {
            continue;
        }
        out.push_str(&format!("## {}\n\n", scenario.id));
        out.push_str(&format!("Status: {}\n\n", status_label(&scenario.status)));
        if let Some(attempt) = scenario.attempts.last() {
            for turn in &attempt.turns {
                let failed: Vec<_> = turn
                    .assertion_results
                    .iter()
                    .filter(|r| !r.passed)
                    .collect();
                if failed.is_empty() {
                    continue;
                }
                out.push_str(&format!("### Turn {}\n\n", turn.index + 1));
                out.push_str(&format!("Input: `{}`\n\n", turn.input.value));
                out.push_str(&format!("Response: `{}`\n\n", turn.response.value));
                for assertion in failed {
                    out.push_str(&format!(
                        "- `{}` failed: expected `{}`, actual `{}`\n",
                        assertion.assertion, assertion.expected, assertion.actual
                    ));
                }
                out.push('\n');
            }
        }
    }
    out
}

fn render_junit(result: &EvalResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "<testsuite name=\"{}\" tests=\"{}\" failures=\"{}\" skipped=\"{}\">\n",
        escape_xml(&result.suite),
        result.total,
        result.failed,
        result.skipped
    ));
    for scenario in &result.scenarios {
        out.push_str(&format!(
            "  <testcase name=\"{}\">\n",
            escape_xml(&scenario.id)
        ));
        match &scenario.status {
            ScenarioStatus::Passed => {}
            ScenarioStatus::Skipped { reason } => {
                out.push_str(&format!(
                    "    <skipped message=\"{}\" />\n",
                    escape_xml(reason.as_deref().unwrap_or("skipped"))
                ));
            }
            ScenarioStatus::Failed { reason } => {
                out.push_str(&format!(
                    "    <failure message=\"{}\" />\n",
                    escape_xml(reason)
                ));
            }
            ScenarioStatus::Error { message } => {
                out.push_str(&format!(
                    "    <error message=\"{}\" />\n",
                    escape_xml(message)
                ));
            }
        }
        out.push_str("  </testcase>\n");
    }
    out.push_str("</testsuite>\n");
    out
}

fn status_label(status: &ScenarioStatus) -> String {
    match status {
        ScenarioStatus::Passed => "passed".to_string(),
        ScenarioStatus::Failed { reason } => format!("failed: {}", reason),
        ScenarioStatus::Skipped { reason } => {
            format!("skipped: {}", reason.as_deref().unwrap_or("skipped"))
        }
        ScenarioStatus::Error { message } => format!("error: {}", message),
    }
}

fn escape_xml(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
