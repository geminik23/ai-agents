use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use ai_agents_eval::{EvalRunner, EvalSuite};
use ai_agents_runtime::spawner::{AgentSpawner, resolve_templates};
use ai_agents_runtime::spec::{AgentSpec, TemplateSource};
use ai_agents_skills::{SkillDefinition, SkillLoader, SkillRef};
use ai_agents_tools::builtin::{all_builtin_tools, get_builtin_tool};
use ai_agents_tools::create_builtin_registry;
use anyhow::{Context, Result, anyhow};
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct ContractSummary {
    pub schema_version: u32,
    pub status: String,
    pub versions: VersionReport,
    pub examples: ExampleReport,
    pub eval_suites: EvalReport,
    pub markdown_fences: Vec<FenceReport>,
    pub builtin_tools: BuiltinReport,
    pub examples_readme: ReadmeCoverage,
    pub drift: Vec<DriftItem>,
    pub checker_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VersionReport {
    pub cargo_version: String,
    pub website_version: String,
    pub cargo_status: String,
    pub website_status: String,
    pub readme_status_line: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ExampleReport {
    pub counts: BTreeMap<String, usize>,
    pub files: Vec<ExampleFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExampleFile {
    pub path: String,
    pub kind: String,
    pub owners: Vec<String>,
    pub valid: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct EvalReport {
    pub counts: BTreeMap<String, usize>,
    pub files: Vec<EvalFile>,
    pub unindexed_live: Vec<String>,
    pub stale_references: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalFile {
    pub path: String,
    pub kind: String,
    pub valid: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FenceReport {
    pub path: String,
    pub index: usize,
    pub start_line: usize,
    pub kind: String,
    pub valid: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BuiltinReport {
    pub registry: Vec<String>,
    pub inventories: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ReadmeCoverage {
    pub indexed_runnable: Vec<String>,
    pub missing_runnable: Vec<String>,
    pub stale_index_entries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DriftItem {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExampleKind {
    RunnableAgent,
    OrchestrationChild,
    SkillFragment,
    SpawnerTemplate,
    PricingData,
}

impl ExampleKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::RunnableAgent => "runnable_agent",
            Self::OrchestrationChild => "orchestration_child",
            Self::SkillFragment => "skill_fragment",
            Self::SpawnerTemplate => "spawner_template",
            Self::PricingData => "pricing_data",
        }
    }

    fn is_support(self) -> bool {
        !matches!(self, Self::RunnableAgent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FenceKind {
    CompleteAgent,
    CompleteEval,
    CompleteSkill,
    ConceptualFragment,
}

impl FenceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::CompleteAgent => "complete_agent",
            Self::CompleteEval => "complete_eval",
            Self::CompleteSkill => "complete_skill",
            Self::ConceptualFragment => "conceptual_fragment",
        }
    }
}

#[derive(Debug)]
struct MarkdownFence {
    index: usize,
    start_line: usize,
    content: String,
}

/// Runs the static public contract checker without constructing providers or executing agents.
pub fn run_public_contract(root: &Path) -> Result<ContractSummary> {
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve repository root '{}'", root.display()))?;
    require_path(&root, "Cargo.toml")?;
    require_path(&root, "examples/yaml")?;
    require_path(&root, "examples/eval")?;
    require_path(&root, "website/content")?;

    let mut drift = Vec::new();
    let versions = validate_versions(&root, &mut drift)?;
    let examples = validate_examples(&root, &mut drift)?;
    let eval_suites = validate_eval_suites(&root, &mut drift)?;
    let markdown_fences = validate_markdown_fences(&root, &mut drift)?;
    let builtin_tools = validate_builtin_inventories(&root, &mut drift)?;
    let examples_readme = validate_readme_coverage(&root, &examples, &mut drift)?;
    validate_current_surface_claims(&root, &mut drift)?;

    let mut summary = ContractSummary {
        schema_version: SCHEMA_VERSION,
        status: String::new(),
        versions,
        examples,
        eval_suites,
        markdown_fences,
        builtin_tools,
        examples_readme,
        drift,
        checker_errors: Vec::new(),
    };
    sort_summary(&mut summary);
    summary.status = if summary.drift.is_empty() {
        "ok".to_string()
    } else {
        "drift".to_string()
    };
    Ok(summary)
}

/// Writes the stable, pretty-printed JSON report to the contract output directory.
pub fn write_summary(root: &Path, summary: &ContractSummary) -> Result<()> {
    let output_dir = root.join("target/public-contract");
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create '{}'", output_dir.display()))?;
    let output = output_dir.join("summary.json");
    let mut json = serde_json::to_string_pretty(summary)?;
    json.push('\n');
    fs::write(&output, json).with_context(|| format!("failed to write '{}'", output.display()))
}

/// Creates a minimal schema-compatible report for checker or repository configuration failures.
pub fn error_summary(error: String) -> ContractSummary {
    ContractSummary {
        schema_version: SCHEMA_VERSION,
        status: "error".to_string(),
        versions: VersionReport {
            cargo_version: String::new(),
            website_version: String::new(),
            cargo_status: String::new(),
            website_status: String::new(),
            readme_status_line: String::new(),
        },
        examples: ExampleReport::default(),
        eval_suites: EvalReport::default(),
        markdown_fences: Vec::new(),
        builtin_tools: BuiltinReport::default(),
        examples_readme: ReadmeCoverage::default(),
        drift: Vec::new(),
        checker_errors: vec![error],
    }
}

fn require_path(root: &Path, relative: &str) -> Result<()> {
    let path = root.join(relative);
    if path.exists() {
        Ok(())
    } else {
        Err(anyhow!("required checker input is missing: {relative}"))
    }
}

fn validate_versions(root: &Path, drift: &mut Vec<DriftItem>) -> Result<VersionReport> {
    let cargo = read_text(&root.join("Cargo.toml"))?;
    let website = read_text(&root.join("website/config.toml"))?;
    let readme = read_text(&root.join("README.md"))?;
    let cargo_version = section_value(&cargo, "workspace.package", "version")
        .ok_or_else(|| anyhow!("Cargo.toml is missing workspace.package.version"))?;
    let website_version = section_value(&website, "extra", "version")
        .ok_or_else(|| anyhow!("website/config.toml is missing extra.version"))?;
    let cargo_status = release_status(&cargo_version).to_string();
    let website_status = release_status(&website_version).to_string();
    let readme_status_line = readme
        .lines()
        .find(|line| line.trim_start().starts_with("> Status:"))
        .unwrap_or_default()
        .trim()
        .to_string();

    if cargo_version != website_version {
        push_drift(
            drift,
            "version.website",
            "website/config.toml",
            format!(
                "website version '{}' does not match workspace version '{}'",
                website_version, cargo_version
            ),
        );
    }
    if cargo_status != website_status {
        push_drift(
            drift,
            "status.website",
            "website/config.toml",
            format!(
                "website status '{}' does not match workspace status '{}'",
                website_status, cargo_status
            ),
        );
    }
    if readme_status_line.is_empty() {
        push_drift(
            drift,
            "status.readme_missing",
            "README.md",
            "README status line is missing".to_string(),
        );
    } else {
        if !readme_status_line.contains(&cargo_version) {
            push_drift(
                drift,
                "version.readme",
                "README.md",
                format!("README status does not contain workspace version '{cargo_version}'"),
            );
        }
        if cargo_status == "stable"
            && (readme_status_line.contains("release candidate")
                || readme_status_line.contains("preparation is in progress"))
        {
            push_drift(
                drift,
                "status.readme",
                "README.md",
                "stable workspace version still uses prerelease README status language".to_string(),
            );
        }
    }

    Ok(VersionReport {
        cargo_version,
        website_version,
        cargo_status,
        website_status,
        readme_status_line,
    })
}

fn section_value(content: &str, wanted_section: &str, wanted_key: &str) -> Option<String> {
    let mut section = "";
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            section = line.trim_matches(['[', ']']);
            continue;
        }
        if section != wanted_section || line.starts_with('#') {
            continue;
        }
        let (key, value) = line.split_once('=')?;
        if key.trim() == wanted_key {
            return Some(value.trim().trim_matches(['\'', '"']).to_string());
        }
    }
    None
}

fn release_status(version: &str) -> &'static str {
    if version.contains("-rc.") {
        "release_candidate"
    } else if version.contains('-') {
        "prerelease"
    } else {
        "stable"
    }
}

fn validate_examples(root: &Path, drift: &mut Vec<DriftItem>) -> Result<ExampleReport> {
    let paths = discover_files(&root.join("examples/yaml"), &["yaml", "yml"])?;
    let mut kinds = BTreeMap::new();
    let mut specs = BTreeMap::new();
    let mut errors: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ownership: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for path in &paths {
        let relative = repo_path(root, path);
        let kind = classify_example(Path::new(&relative));
        kinds.insert(relative.clone(), kind);
        match kind {
            ExampleKind::RunnableAgent | ExampleKind::OrchestrationChild => {
                match load_agent_spec(path) {
                    Ok(spec) => {
                        specs.insert(relative, spec);
                    }
                    Err(error) => record_validation_error(
                        &mut errors,
                        drift,
                        "examples.agent",
                        &relative,
                        error.to_string(),
                    ),
                }
            }
            ExampleKind::SkillFragment => {
                let mut loader = SkillLoader::new();
                if let Err(error) = loader.load_from_path(path) {
                    record_validation_error(
                        &mut errors,
                        drift,
                        "examples.skill",
                        &relative,
                        error.to_string(),
                    );
                }
            }
            ExampleKind::SpawnerTemplate | ExampleKind::PricingData => {}
        }
    }

    let agents: Vec<(String, AgentSpec)> = specs
        .iter()
        .map(|(path, spec)| (path.clone(), spec.clone()))
        .collect();
    for (relative, spec) in agents {
        let path = root.join(&relative);
        validate_agent_assets(
            root,
            &path,
            &relative,
            &spec,
            &specs,
            &mut ownership,
            &mut errors,
            drift,
        );
    }

    for (path, kind) in &kinds {
        if kind.is_support() && ownership.get(path).is_none_or(BTreeSet::is_empty) {
            record_validation_error(
                &mut errors,
                drift,
                "examples.unowned_support",
                path,
                format!(
                    "{} is not referenced by an owning runnable agent",
                    kind.as_str()
                ),
            );
        }
    }

    let mut counts = BTreeMap::new();
    let files = paths
        .iter()
        .map(|path| {
            let relative = repo_path(root, path);
            let kind = kinds[&relative];
            *counts.entry(kind.as_str().to_string()).or_insert(0) += 1;
            let file_errors = errors.remove(&relative).unwrap_or_default();
            ExampleFile {
                path: relative.clone(),
                kind: kind.as_str().to_string(),
                owners: ownership
                    .remove(&relative)
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                valid: file_errors.is_empty(),
                errors: file_errors,
            }
        })
        .collect();

    Ok(ExampleReport { counts, files })
}

fn classify_example(path: &Path) -> ExampleKind {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let components: Vec<_> = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect();
    if file_name.ends_with(".skill.yaml") || file_name.ends_with(".skill.yml") {
        ExampleKind::SkillFragment
    } else if file_name == "pricing.yaml" || file_name == "pricing.yml" {
        ExampleKind::PricingData
    } else if components
        .windows(2)
        .any(|pair| pair == ["spawner", "templates"])
        || components.contains(&"templates") && components.contains(&"spawner")
    {
        ExampleKind::SpawnerTemplate
    } else if components
        .windows(2)
        .any(|pair| pair == ["orchestration", "agents"])
        || components.contains(&"agents") && components.contains(&"orchestration")
    {
        ExampleKind::OrchestrationChild
    } else {
        ExampleKind::RunnableAgent
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_agent_assets(
    root: &Path,
    path: &Path,
    owner: &str,
    spec: &AgentSpec,
    specs: &BTreeMap<String, AgentSpec>,
    ownership: &mut BTreeMap<String, BTreeSet<String>>,
    errors: &mut BTreeMap<String, Vec<String>>,
    drift: &mut Vec<DriftItem>,
) {
    let base_dir = path.parent().unwrap_or(root);

    for skill in &spec.skills {
        if let SkillRef::File { file } = skill {
            mark_owner(root, &base_dir.join(file), owner, ownership);
        }
    }
    let mut loader = SkillLoader::new().with_base_dir(base_dir);
    if let Err(error) = loader.load_refs(&spec.skills) {
        record_validation_error(
            errors,
            drift,
            "examples.agent_skills",
            owner,
            error.to_string(),
        );
    }

    if let Some(pricing_file) = spec.observability.cost.pricing_file.as_deref() {
        mark_owner(root, &base_dir.join(pricing_file), owner, ownership);
    }
    if let Err(error) = spec.observability.validate().and_then(|_| {
        spec.observability
            .clone()
            .with_pricing_file_loaded(Some(base_dir))
            .map(|_| ())
    }) {
        record_validation_error(
            errors,
            drift,
            "examples.agent_observability",
            owner,
            error.to_string(),
        );
    }

    let Some(spawner_config) = spec.spawner.as_ref() else {
        return;
    };
    for source in spawner_config.templates.values() {
        if let TemplateSource::File {
            path: template_path,
        } = source
        {
            mark_owner(root, &base_dir.join(template_path), owner, ownership);
        }
    }
    match resolve_templates(&spawner_config.templates, Some(base_dir)) {
        Ok(templates) => {
            let mut names: Vec<_> = templates.keys().cloned().collect();
            names.sort();
            for name in names {
                let template = &templates[&name];
                if let Err(error) = validate_spawner_template(template, spawner_config) {
                    record_validation_error(
                        errors,
                        drift,
                        "examples.spawner_template",
                        owner,
                        format!("template '{name}': {error}"),
                    );
                }
            }
        }
        Err(error) => record_validation_error(
            errors,
            drift,
            "examples.spawner_template",
            owner,
            error.to_string(),
        ),
    }

    let mut spawner = AgentSpawner::new();
    if let Some(allowed) = spawner_config.allowed_tools.clone() {
        spawner = spawner.with_allowed_tools(allowed);
    }
    for child in &spawner_config.auto_spawn {
        let child_path = normalize_path(&base_dir.join(&child.agent));
        let child_relative = repo_path(root, &child_path);
        mark_owner(root, &child_path, owner, ownership);
        let Some(child_spec) = specs.get(&child_relative) else {
            record_validation_error(
                errors,
                drift,
                "examples.orchestration_child",
                owner,
                format!(
                    "auto_spawn child '{}' does not resolve to a valid classified agent at '{}'",
                    child.id, child_relative
                ),
            );
            continue;
        };
        if let Err(error) = spawner.validate_explicit_child(&child.id, child_spec) {
            record_validation_error(
                errors,
                drift,
                "examples.orchestration_child",
                &child_relative,
                format!("owner '{owner}' rejected child '{}': {error}", child.id),
            );
        }
    }
}

fn validate_spawner_template(
    template: &ai_agents_runtime::spawner::ResolvedTemplate,
    config: &ai_agents_runtime::spec::SpawnerConfig,
) -> Result<()> {
    let mut environment = Environment::new();
    environment
        .add_template("_public_contract", &template.content)
        .context("template parse failed")?;
    let compiled = environment
        .get_template("_public_contract")
        .context("template lookup failed")?;
    let mut values = Map::new();
    values.insert(
        "name".to_string(),
        Value::String("ContractAgent".to_string()),
    );
    if let Some(variables) = &template.variables {
        for key in variables.keys() {
            values.insert(key.clone(), Value::String(format!("sample_{key}")));
        }
    }
    values.insert(
        "context".to_string(),
        Value::Object(config.shared_context.clone().into_iter().collect()),
    );
    let rendered = compiled
        .render(minijinja::Value::from_serialize(Value::Object(values)))
        .context("template render failed")?;
    let spec = AgentSpec::from_yaml_strict(&rendered).context("rendered template is invalid")?;
    spec.validate()
        .context("rendered template validation failed")
}

fn load_agent_spec(path: &Path) -> Result<AgentSpec> {
    let content = read_text(path)?;
    let spec = AgentSpec::from_yaml_strict(&content)
        .with_context(|| format!("strict AgentSpec parsing failed for '{}'", path.display()))?;
    spec.validate()
        .with_context(|| format!("AgentSpec validation failed for '{}'", path.display()))?;
    Ok(spec)
}

fn mark_owner(
    root: &Path,
    path: &Path,
    owner: &str,
    ownership: &mut BTreeMap<String, BTreeSet<String>>,
) {
    ownership
        .entry(repo_path(root, &normalize_path(path)))
        .or_default()
        .insert(owner.to_string());
}

fn validate_eval_suites(root: &Path, drift: &mut Vec<DriftItem>) -> Result<EvalReport> {
    let paths = discover_files(&root.join("examples/eval"), &["yaml", "yml"])?;
    let mut counts = BTreeMap::new();
    let mut files = Vec::new();
    for path in paths {
        let relative = repo_path(root, &path);
        let kind = classify_eval(&relative);
        *counts.entry(kind.to_string()).or_insert(0) += 1;
        let mut errors = Vec::new();
        if kind == "unclassified" {
            errors.push("suite is outside mocked, live/examples, or live/quality".to_string());
        } else if let Err(error) = EvalRunner::validate_file(&path, None) {
            errors.push(error.to_string());
        }
        for error in &errors {
            push_drift(
                drift,
                "eval.config",
                &relative,
                format!("{kind} suite validation failed: {error}"),
            );
        }
        files.push(EvalFile {
            path: relative,
            kind: kind.to_string(),
            valid: errors.is_empty(),
            errors,
        });
    }
    let discovered: BTreeSet<_> = files.iter().map(|file| file.path.clone()).collect();
    let live: BTreeSet<_> = files
        .iter()
        .filter(|file| file.kind == "live" || file.kind == "quality")
        .map(|file| file.path.clone())
        .collect();
    let registry = read_text(&root.join("examples/eval/live/README.md"))?;
    let references: BTreeSet<_> = inline_code_tokens(&registry)
        .into_iter()
        .filter(|path| {
            path.starts_with("examples/eval/")
                && (path.ends_with(".yaml") || path.ends_with(".yml"))
                && !path.contains('*')
        })
        .collect();
    let unindexed_live: Vec<_> = live.difference(&references).cloned().collect();
    let stale_references: Vec<_> = references.difference(&discovered).cloned().collect();

    for path in &unindexed_live {
        push_drift(
            drift,
            "eval.live_index",
            "examples/eval/live/README.md",
            format!("live or quality suite '{path}' is missing from the coverage registry"),
        );
    }
    for path in &stale_references {
        push_drift(
            drift,
            "eval.stale_reference",
            "examples/eval/live/README.md",
            format!("referenced eval suite '{path}' does not exist"),
        );
    }

    Ok(EvalReport {
        counts,
        files,
        unindexed_live,
        stale_references,
    })
}

fn classify_eval(path: &str) -> &'static str {
    if path.starts_with("examples/eval/mocked/") {
        "mocked"
    } else if path.starts_with("examples/eval/live/examples/") {
        "live"
    } else if path.starts_with("examples/eval/live/quality/") {
        "quality"
    } else {
        "unclassified"
    }
}

fn validate_markdown_fences(root: &Path, drift: &mut Vec<DriftItem>) -> Result<Vec<FenceReport>> {
    let mut paths = vec![
        root.join("README.md"),
        root.join("CHANGELOG.md"),
        root.join("examples/README.md"),
        root.join("website/content/roadmap.md"),
    ];
    paths.extend(discover_files(&root.join("website/content/docs"), &["md"])?);
    paths.sort();
    paths.dedup();
    let mut reports = Vec::new();

    for path in paths {
        let relative = repo_path(root, &path);
        let content = read_text(&path)?;
        for fence in parse_yaml_fences(&content) {
            let kind = classify_fence_content(&fence.content);
            let error = validate_fence(kind, &fence.content)
                .err()
                .map(|error| error.to_string());
            if let Some(message) = &error {
                push_drift(
                    drift,
                    "markdown.yaml_fence",
                    &relative,
                    format!(
                        "YAML fence {} at line {} ({}) is invalid: {}",
                        fence.index,
                        fence.start_line,
                        kind.as_str(),
                        message
                    ),
                );
            }
            reports.push(FenceReport {
                path: relative.clone(),
                index: fence.index,
                start_line: fence.start_line,
                kind: kind.as_str().to_string(),
                valid: error.is_none(),
                error,
            });
        }
    }
    Ok(reports)
}

fn parse_yaml_fences(markdown: &str) -> Vec<MarkdownFence> {
    let mut fences = Vec::new();
    let mut current: Option<(usize, usize, Vec<&str>)> = None;
    let mut index = 0;
    for (line_index, line) in markdown.lines().enumerate() {
        let trimmed = line.trim_start();
        if let Some((fence_index, start_line, lines)) = current.as_mut() {
            if trimmed.starts_with("```") {
                fences.push(MarkdownFence {
                    index: *fence_index,
                    start_line: *start_line,
                    content: lines.join("\n"),
                });
                current = None;
            } else {
                lines.push(line);
            }
            continue;
        }
        let language = trimmed.strip_prefix("```").unwrap_or_default().trim();
        if language == "yaml" || language == "yml" {
            index += 1;
            current = Some((index, line_index + 2, Vec::new()));
        }
    }
    if let Some((fence_index, start_line, lines)) = current {
        fences.push(MarkdownFence {
            index: fence_index,
            start_line,
            content: lines.join("\n"),
        });
    }
    fences
}

fn classify_fence_content(content: &str) -> FenceKind {
    let keys: BTreeSet<_> = content
        .lines()
        .filter(|line| !line.starts_with([' ', '\t', '#', '-']))
        .filter_map(|line| line.split_once(':').map(|(key, _)| key.trim()))
        .filter(|key| !key.is_empty())
        .collect();
    let has = |key: &str| keys.contains(key);
    if has("name") && has("scenarios") {
        FenceKind::CompleteEval
    } else if has("name") && has("system_prompt") {
        FenceKind::CompleteAgent
    } else if (has("id") || has("skill")) && has("description") && has("trigger") && has("steps") {
        FenceKind::CompleteSkill
    } else {
        FenceKind::ConceptualFragment
    }
}

fn validate_fence(kind: FenceKind, content: &str) -> Result<()> {
    match kind {
        FenceKind::CompleteAgent => {
            let spec = AgentSpec::from_yaml_strict(content)?;
            spec.validate()?;
            Ok(())
        }
        FenceKind::CompleteEval => {
            let suite: EvalSuite = serde_yaml::from_str(content)?;
            let cli_agent = suite
                .agent
                .is_none()
                .then(|| PathBuf::from("_public_contract_agent.yaml"));
            suite.validate(cli_agent.as_ref())?;
            Ok(())
        }
        FenceKind::CompleteSkill => {
            serde_yaml::from_str::<SkillDefinition>(content)?;
            Ok(())
        }
        FenceKind::ConceptualFragment => {
            for document in serde_yaml::Deserializer::from_str(content) {
                serde_yaml::Value::deserialize(document)?;
            }
            Ok(())
        }
    }
}

fn validate_builtin_inventories(root: &Path, drift: &mut Vec<DriftItem>) -> Result<BuiltinReport> {
    let mut registry = create_builtin_registry().list_ids();
    registry.sort();
    registry.dedup();
    let registry_set: BTreeSet<_> = registry.iter().cloned().collect();
    let all_tools: BTreeSet<_> = all_builtin_tools()
        .into_iter()
        .map(|tool| tool.id().to_string())
        .collect();

    if all_tools != registry_set {
        let missing: Vec<_> = all_tools.difference(&registry_set).cloned().collect();
        let extra: Vec<_> = registry_set.difference(&all_tools).cloned().collect();
        push_drift(
            drift,
            "builtin.registry",
            "crates/ai-agents-tools",
            format!(
                "all_builtin_tools and create_builtin_registry differ; missing: [{}]; extra: [{}]",
                missing.join(", "),
                extra.join(", ")
            ),
        );
    }
    for id in &registry {
        match get_builtin_tool(id) {
            Some(tool) if tool.id() == id => {}
            Some(tool) => push_drift(
                drift,
                "builtin.lookup",
                "crates/ai-agents-tools",
                format!("lookup for '{id}' returned '{}'", tool.id()),
            ),
            None => push_drift(
                drift,
                "builtin.lookup",
                "crates/ai-agents-tools",
                format!("lookup for '{id}' returned no tool"),
            ),
        }
    }

    let readme = read_text(&root.join("README.md"))?;
    let builtin_reference = read_text(&root.join("website/content/docs/built-in-tools.md"))?;
    let yaml_reference = read_text(&root.join("website/content/docs/yaml-reference.md"))?;
    let mut inventories = BTreeMap::new();
    inventories.insert(
        "README.md".to_string(),
        inventory_from_line(&readme, "canonical built-in IDs:"),
    );
    inventories.insert(
        "website/content/docs/built-in-tools.md".to_string(),
        inventory_from_text_fence(&builtin_reference, "## Canonical inventory"),
    );
    inventories.insert(
        "website/content/docs/yaml-reference.md".to_string(),
        inventory_from_section(&yaml_reference, "### Built-in catalog"),
    );

    for (path, inventory) in &inventories {
        let actual: BTreeSet<_> = inventory.iter().cloned().collect();
        let missing: Vec<_> = registry_set.difference(&actual).cloned().collect();
        let extra: Vec<_> = actual.difference(&registry_set).cloned().collect();
        if !missing.is_empty() || !extra.is_empty() {
            push_drift(
                drift,
                "builtin.inventory",
                path,
                format!(
                    "built-in inventory differs from registry; missing: [{}]; extra: [{}]",
                    missing.join(", "),
                    extra.join(", ")
                ),
            );
        }
    }

    Ok(BuiltinReport {
        registry,
        inventories,
    })
}

fn inventory_from_text_fence(content: &str, heading: &str) -> Vec<String> {
    let mut in_section = false;
    let mut in_text_fence = false;
    let mut ids = BTreeSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == heading {
            in_section = true;
            continue;
        }
        if in_section && trimmed.starts_with("## ") {
            break;
        }
        if !in_section {
            continue;
        }
        if trimmed == "```text" {
            in_text_fence = true;
            continue;
        }
        if in_text_fence && trimmed == "```" {
            break;
        }
        if in_text_fence {
            ids.extend(
                trimmed
                    .split([',', ' ', '\t'])
                    .map(str::trim)
                    .filter(|token| is_tool_id(token))
                    .map(ToString::to_string),
            );
        }
    }
    ids.into_iter().collect()
}

fn inventory_from_line(content: &str, marker: &str) -> Vec<String> {
    let Some(line) = content.lines().find(|line| line.contains(marker)) else {
        return Vec::new();
    };
    let inventory_text = line
        .split_once(marker)
        .map(|(_, text)| text)
        .unwrap_or(line);
    inline_code_tokens(inventory_text)
        .into_iter()
        .filter(|token| is_tool_id(token))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn inventory_from_section(content: &str, heading: &str) -> Vec<String> {
    let mut in_section = false;
    let mut tokens = BTreeSet::new();
    for line in content.lines() {
        if line.trim() == heading {
            in_section = true;
            continue;
        }
        if in_section && line.starts_with("### ") {
            break;
        }
        if in_section {
            tokens.extend(
                inline_code_tokens(line)
                    .into_iter()
                    .filter(|token| is_tool_id(token)),
            );
        }
    }
    tokens.into_iter().collect()
}

fn inline_code_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut remaining = text;
    while let Some(start) = remaining.find('`') {
        remaining = &remaining[start + 1..];
        let Some(end) = remaining.find('`') else {
            break;
        };
        tokens.push(remaining[..end].to_string());
        remaining = &remaining[end + 1..];
    }
    tokens
}

fn is_tool_id(token: &str) -> bool {
    !token.is_empty()
        && token.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
        })
}

fn validate_readme_coverage(
    root: &Path,
    examples: &ExampleReport,
    drift: &mut Vec<DriftItem>,
) -> Result<ReadmeCoverage> {
    let readme = read_text(&root.join("examples/README.md"))?;
    let indexed = indexed_yaml_examples(&readme);
    let runnable: BTreeSet<_> = examples
        .files
        .iter()
        .filter(|file| file.kind == ExampleKind::RunnableAgent.as_str())
        .map(|file| file.path.trim_start_matches("examples/").to_string())
        .collect();
    let missing: Vec<_> = runnable.difference(&indexed).cloned().collect();
    let stale: Vec<_> = indexed.difference(&runnable).cloned().collect();
    for path in &missing {
        push_drift(
            drift,
            "examples.readme_index",
            "examples/README.md",
            format!("runnable YAML example '{path}' is missing from its index table"),
        );
    }
    for path in &stale {
        push_drift(
            drift,
            "examples.readme_stale",
            "examples/README.md",
            format!("indexed YAML example '{path}' is not a runnable agent"),
        );
    }
    Ok(ReadmeCoverage {
        indexed_runnable: runnable.intersection(&indexed).cloned().collect(),
        missing_runnable: missing,
        stale_index_entries: stale,
    })
}

fn indexed_yaml_examples(markdown: &str) -> BTreeSet<String> {
    let mut category: Option<String> = None;
    let mut indexed = BTreeSet::new();
    for line in markdown.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed
            .strip_prefix("### `yaml/")
            .and_then(|value| value.strip_suffix("/`"))
        {
            category = Some(value.to_string());
            continue;
        }
        let Some(category) = category.as_ref() else {
            continue;
        };
        if !trimmed.starts_with("| `") {
            continue;
        }
        let Some(token) = inline_code_tokens(trimmed).into_iter().next() else {
            continue;
        };
        if (token.ends_with(".yaml") || token.ends_with(".yml"))
            && !token.ends_with(".skill.yaml")
            && !token.ends_with(".skill.yml")
            && !token.contains('*')
        {
            indexed.insert(format!("yaml/{category}/{token}"));
        }
    }
    indexed
}

fn validate_current_surface_claims(root: &Path, drift: &mut Vec<DriftItem>) -> Result<()> {
    let mut paths = vec![
        root.join("README.md"),
        root.join("examples/README.md"),
        root.join("website/content/roadmap.md"),
    ];
    paths.extend(discover_files(&root.join("website/content/docs"), &["md"])?);

    for path in paths {
        let relative = repo_path(root, &path);
        let content = read_text(&path)?;
        if content.contains("migration-to-v1") {
            push_drift(
                drift,
                "current.migration_link",
                &relative,
                "current public surface references the intentionally omitted migration guide",
            );
        }
        for (line_index, line) in content.lines().enumerate() {
            let trimmed = line.trim_start().trim_start_matches(['-', '#']).trim();
            let removed_field = ["providers:", "provider_security:", "cache_ttl:"]
                .into_iter()
                .find(|field| trimmed.starts_with(field));
            if let Some(field) = removed_field {
                push_drift(
                    drift,
                    "current.removed_field",
                    &relative,
                    format!(
                        "line {} uses removed framework field '{}'",
                        line_index + 1,
                        field.trim_end_matches(':')
                    ),
                );
            }
            let legacy_policy = ["allowed_domains:", "blocked_domains:", "allowed_paths:"]
                .into_iter()
                .find(|field| trimmed.starts_with(field));
            if let Some(field) = legacy_policy {
                push_drift(
                    drift,
                    "current.legacy_policy_example",
                    &relative,
                    format!(
                        "line {} recommends legacy policy field '{}' in a current YAML block",
                        line_index + 1,
                        field.trim_end_matches(':')
                    ),
                );
            }
            for removed_id in ["generate_agent", "send_message"] {
                if inline_code_tokens(line)
                    .iter()
                    .any(|token| token == removed_id)
                {
                    push_drift(
                        drift,
                        "current.removed_spawner_id",
                        &relative,
                        format!(
                            "line {} references removed spawner tool ID '{removed_id}'",
                            line_index + 1
                        ),
                    );
                }
            }
            if (relative == "README.md" || relative == "website/content/roadmap.md")
                && line
                    .split(|character: char| !character.is_ascii_alphanumeric())
                    .any(is_internal_feature_id)
            {
                push_drift(
                    drift,
                    "current.internal_feature_id",
                    &relative,
                    format!("line {} exposes an internal feature ID", line_index + 1),
                );
            }
        }
    }

    let build_script = read_text(&root.join("website/build.sh"))?;
    if !build_script.contains("examples/README.md -> content/examples/_index.md")
        || !build_script.contains("tail -n +3")
    {
        push_drift(
            drift,
            "examples.website_sync",
            "website/build.sh",
            "website build no longer regenerates the examples page from examples/README.md",
        );
    }
    Ok(())
}

fn is_internal_feature_id(token: &str) -> bool {
    let Some(number) = token.strip_prefix('F') else {
        return false;
    };
    !number.is_empty() && number.chars().all(|character| character.is_ascii_digit())
}

fn discover_files(root: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    discover_files_inner(root, extensions, &mut files)?;
    files.sort();
    Ok(files)
}

fn discover_files_inner(root: &Path, extensions: &[&str], files: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs::read_dir(root)
        .with_context(|| format!("failed to read directory '{}'", root.display()))?;
    for entry in entries {
        let entry =
            entry.with_context(|| format!("failed to read entry in '{}'", root.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect '{}'", path.display()))?;
        if file_type.is_dir() {
            discover_files_inner(&path, extensions, files)?;
        } else if file_type.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extensions.contains(&extension))
        {
            files.push(path);
        }
    }
    Ok(())
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn repo_path(root: &Path, path: &Path) -> String {
    let normalized = normalize_path(path);
    let relative = normalized.strip_prefix(root).unwrap_or(&normalized);
    relative.to_string_lossy().replace('\\', "/")
}

fn read_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("failed to read '{}'", path.display()))
}

fn record_validation_error(
    errors: &mut BTreeMap<String, Vec<String>>,
    drift: &mut Vec<DriftItem>,
    code: &str,
    path: &str,
    message: String,
) {
    errors
        .entry(path.to_string())
        .or_default()
        .push(message.clone());
    push_drift(drift, code, path, message);
}

fn push_drift(
    drift: &mut Vec<DriftItem>,
    code: impl Into<String>,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    drift.push(DriftItem {
        code: code.into(),
        path: path.into(),
        message: message.into(),
    });
}

fn sort_summary(summary: &mut ContractSummary) {
    for file in &mut summary.examples.files {
        file.owners.sort();
        file.owners.dedup();
        file.errors.sort();
        file.errors.dedup();
    }
    summary
        .examples
        .files
        .sort_by(|left, right| left.path.cmp(&right.path));
    for file in &mut summary.eval_suites.files {
        file.errors.sort();
        file.errors.dedup();
    }
    summary
        .eval_suites
        .files
        .sort_by(|left, right| left.path.cmp(&right.path));
    summary.markdown_fences.sort_by(|left, right| {
        (&left.path, left.index, left.start_line).cmp(&(&right.path, right.index, right.start_line))
    });
    summary.builtin_tools.registry.sort();
    summary.builtin_tools.registry.dedup();
    for inventory in summary.builtin_tools.inventories.values_mut() {
        inventory.sort();
        inventory.dedup();
    }
    summary.examples_readme.indexed_runnable.sort();
    summary.examples_readme.indexed_runnable.dedup();
    summary.examples_readme.missing_runnable.sort();
    summary.examples_readme.missing_runnable.dedup();
    summary.examples_readme.stale_index_entries.sort();
    summary.examples_readme.stale_index_entries.dedup();
    summary.eval_suites.unindexed_live.sort();
    summary.eval_suites.unindexed_live.dedup();
    summary.eval_suites.stale_references.sort();
    summary.eval_suites.stale_references.dedup();
    summary.drift.sort_by(|left, right| {
        (&left.path, &left.code, &left.message).cmp(&(&right.path, &right.code, &right.message))
    });
    summary.drift.dedup();
    summary.checker_errors.sort();
    summary.checker_errors.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("xtask-{name}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn discovery_is_recursive_filtered_and_sorted() {
        let root = temporary_directory("discovery");
        fs::create_dir_all(root.join("b/nested")).unwrap();
        fs::create_dir_all(root.join("a")).unwrap();
        fs::write(root.join("b/nested/two.yaml"), "name: two").unwrap();
        fs::write(root.join("a/one.yml"), "name: one").unwrap();
        fs::write(root.join("a/ignored.txt"), "ignored").unwrap();

        let discovered = discover_files(&root, &["yaml", "yml"]).unwrap();
        let relative: Vec<_> = discovered
            .iter()
            .map(|path| repo_path(&root, path))
            .collect();
        assert_eq!(relative, vec!["a/one.yml", "b/nested/two.yaml"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn example_classification_covers_all_contract_kinds() {
        assert_eq!(
            classify_example(Path::new("examples/yaml/basic/chat.yaml")),
            ExampleKind::RunnableAgent
        );
        assert_eq!(
            classify_example(Path::new("examples/yaml/orchestration/agents/worker.yaml")),
            ExampleKind::OrchestrationChild
        );
        assert_eq!(
            classify_example(Path::new("examples/yaml/skills/math.skill.yaml")),
            ExampleKind::SkillFragment
        );
        assert_eq!(
            classify_example(Path::new("examples/yaml/spawner/templates/worker.yaml")),
            ExampleKind::SpawnerTemplate
        );
        assert_eq!(
            classify_example(Path::new("examples/yaml/observability/pricing.yaml")),
            ExampleKind::PricingData
        );
    }

    #[test]
    fn yaml_fence_parser_preserves_order_lines_and_classification() {
        let markdown = "before\n```yaml\nname: Agent\nsystem_prompt: hi\n```\n```yml\nskill: helper\ndescription: d\ntrigger: t\nsteps: []\n```\n";
        let fences = parse_yaml_fences(markdown);
        assert_eq!(fences.len(), 2);
        assert_eq!(fences[0].index, 1);
        assert_eq!(fences[0].start_line, 3);
        assert_eq!(fences[1].index, 2);
        assert_eq!(
            classify_fence_content(&fences[0].content),
            FenceKind::CompleteAgent
        );
        assert_eq!(
            classify_fence_content(&fences[1].content),
            FenceKind::CompleteSkill
        );

        let eval = "name: Eval\nscenarios:\n  - id: smoke\n    turns:\n      - input: hello\n";
        assert_eq!(classify_fence_content(eval), FenceKind::CompleteEval);
        validate_fence(FenceKind::CompleteEval, eval).unwrap();
    }

    #[test]
    fn report_ordering_is_stable() {
        let mut summary = error_summary("z".to_string());
        summary.status.clear();
        summary.checker_errors.push("a".to_string());
        summary.drift = vec![
            DriftItem {
                code: "b".to_string(),
                path: "z".to_string(),
                message: "last".to_string(),
            },
            DriftItem {
                code: "a".to_string(),
                path: "a".to_string(),
                message: "first".to_string(),
            },
        ];
        summary.examples.files = vec![
            ExampleFile {
                path: "z.yaml".to_string(),
                kind: "runnable_agent".to_string(),
                owners: vec!["b".to_string(), "a".to_string()],
                valid: true,
                errors: Vec::new(),
            },
            ExampleFile {
                path: "a.yaml".to_string(),
                kind: "runnable_agent".to_string(),
                owners: Vec::new(),
                valid: true,
                errors: Vec::new(),
            },
        ];

        sort_summary(&mut summary);
        assert_eq!(summary.examples.files[0].path, "a.yaml");
        assert_eq!(summary.examples.files[1].owners, vec!["a", "b"]);
        assert_eq!(summary.drift[0].path, "a");
        assert_eq!(summary.checker_errors, vec!["a", "z"]);
    }
}
