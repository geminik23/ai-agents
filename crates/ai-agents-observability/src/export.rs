use crate::config::{ExportFormat, RawEventsFormat};
use crate::manager::ObservabilityManager;
use std::path::{Path, PathBuf};

/// Result returned after writing configured observability export files.
#[derive(Debug, Clone, Default)]
pub struct ExportResult {
    /// Paths written by this export call.
    pub paths: Vec<PathBuf>,
}

/// Writes report, aggregate, raw event, and Prometheus files requested by config.
pub fn export_observability(manager: &ObservabilityManager) -> std::io::Result<ExportResult> {
    let mut result = ExportResult::default();
    let base = PathBuf::from(&manager.config().export.path);

    if manager.config().export.write_report && manager.wants_format(ExportFormat::Json) {
        let path = output_path(&base, "report.json", "json");
        ensure_parent(&path)?;
        let report = manager.generate_report();
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&path, json)?;
        result.paths.push(path);
    }

    if manager.wants_format(ExportFormat::Csv) {
        let path = output_path(&base, "aggregates.csv", "csv");
        ensure_parent(&path)?;
        std::fs::write(&path, render_csv(manager))?;
        result.paths.push(path);
    }

    if manager.config().export.write_raw_events {
        match manager.config().export.raw_events_format {
            RawEventsFormat::Jsonl if manager.wants_format(ExportFormat::Jsonl) => {
                let path = output_path(&base, "events.jsonl", "jsonl");
                ensure_parent(&path)?;
                std::fs::write(&path, render_jsonl(manager)?)?;
                result.paths.push(path);
            }
            RawEventsFormat::Json if manager.wants_format(ExportFormat::Json) => {
                let path = raw_json_output_path(&base, manager.config().export.write_report);
                ensure_parent(&path)?;
                std::fs::write(&path, render_json_events(manager)?)?;
                result.paths.push(path);
            }
            _ => {}
        }
    }

    if manager.wants_format(ExportFormat::Prometheus) {
        let path = output_path(&base, "metrics.prom", "prom");
        ensure_parent(&path)?;
        std::fs::write(&path, manager.render_prometheus())?;
        result.paths.push(path);
    }

    Ok(result)
}

fn output_path(base: &Path, default_file: &str, extension: &str) -> PathBuf {
    if base.extension().and_then(|ext| ext.to_str()) == Some(extension)
        || (base.extension().is_some() && !base.exists())
    {
        base.to_path_buf()
    } else {
        base.join(default_file)
    }
}

fn raw_json_output_path(base: &Path, report_enabled: bool) -> PathBuf {
    if report_enabled && base.extension().and_then(|ext| ext.to_str()) == Some("json") {
        base.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(|parent| parent.join("events.json"))
            .unwrap_or_else(|| PathBuf::from("events.json"))
    } else {
        output_path(base, "events.json", "json")
    }
}

fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn render_jsonl(manager: &ObservabilityManager) -> std::io::Result<String> {
    let mut output = String::new();
    for event in manager.raw_events() {
        output.push_str(&serde_json::to_string(&event)?);
        output.push('\n');
    }
    Ok(output)
}

fn render_json_events(manager: &ObservabilityManager) -> std::io::Result<String> {
    serde_json::to_string_pretty(&manager.raw_events()).map_err(std::io::Error::other)
}

fn render_csv(manager: &ObservabilityManager) -> String {
    let mut output = String::from(
        "dimensions,count,errors,min_ms,max_ms,avg_ms,p50_ms,p90_ms,p95_ms,p99_ms,total_tokens,total_cost_usd\n",
    );
    for metric in manager.get_metrics() {
        let dimensions = serde_json::to_string(&metric.dimensions).unwrap_or_default();
        output.push_str(&format!(
            "\"{}\",{},{},{},{},{:.3},{},{},{},{},{},{:.8}\n",
            dimensions.replace('"', "\"\""),
            metric.count,
            metric.errors,
            metric.latency.min_ms,
            metric.latency.max_ms,
            metric.latency.avg_ms,
            metric.latency.p50_ms,
            metric.latency.p90_ms,
            metric.latency.p95_ms,
            metric.latency.p99_ms,
            metric.tokens.total_tokens,
            metric.cost.total_usd
        ));
    }
    output
}
