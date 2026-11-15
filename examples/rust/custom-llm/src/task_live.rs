// Shared provider-free preparation for the application and eval hosts.
// Flow: strict declarations -> registered adapter validation -> explicit live authorization -> providers.
// This checks local contracts, not credentials, endpoint access or executable runtime capability.
use crate::task_domain;
use ai_agents::autonomy::*;
use ai_agents::spec::{AgentSpec, PreparedAgentSpec};
use ai_agents::{AgentError, Result};
use std::path::{Path, PathBuf};

/// Validates every declared task profile and state against the same immutable adapter factory used at runtime.
pub fn preflight(spec: &AgentSpec) -> Result<()> {
    preflight_with_ceilings(spec, &AutonomyHostCeilings::default())
}

/// Eval-only file fixtures declare a write-target ceiling; ordinary applications install none.
pub fn preflight_eval(spec: &AgentSpec) -> Result<()> {
    preflight_with_ceilings(
        spec,
        &AutonomyHostCeilings {
            max_declared_write_paths: Some(1),
            ..Default::default()
        },
    )
}

fn preflight_with_ceilings(spec: &AgentSpec, ceilings: &AutonomyHostCeilings) -> Result<()> {
    spec.validate()?;
    let (extensions, tools) = task_domain::domain_components()?;
    let extensions = extensions.freeze();
    let mut available = tools
        .iter()
        .map(|t| t.id().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    available.extend(ai_agents::create_builtin_registry().list_ids());
    let caps = ValidationCapabilities {
        tools: available,
        judge: true,
        planner: true,
        host_revisions: true,
    };
    let check = |selected: Option<&str>, scope, scoped| -> Result<()> {
        let profile = resolve_profile(
            &spec.autonomy,
            scope,
            selected,
            scoped,
            None,
            Some("configuration check"),
            ceilings,
        )?;
        if profile.enabled {
            let bound = extensions.bind(&profile.settings, &caps)?;
            for check in &bound.checks {
                if check.adapter.descriptor().needs_judge
                    && let Some(alias) = check
                        .check
                        .config
                        .as_ref()
                        .and_then(|v| v.get("llm"))
                        .and_then(|v| v.as_str())
                    && !spec.llms.contains_key(alias)
                    && !(spec.llms.is_empty() && alias == "default")
                {
                    return Err(AgentError::Config("unknown judge alias".into()));
                }
            }
        }
        Ok(())
    };
    check(None, AutonomyScope::Task, None)?;
    for name in spec.autonomy.profiles.keys() {
        check(Some(name), AutonomyScope::Task, None)?;
    }
    if let Some(states) = &spec.states {
        let mut pending = states.states.values().collect::<Vec<_>>();
        while let Some(state) = pending.pop() {
            if let Some(settings) = &state.autonomy {
                check(None, AutonomyScope::State, Some(settings))?;
            }
            if let Some(children) = &state.states {
                pending.extend(children.values());
            }
        }
    }
    Ok(())
}

/// Reuses strict eval validation before reading the selected agent; no runner or provider is built.
pub fn preflight_suite(path: &Path) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    ai_agents_eval::EvalRunner::validate_file(path, None)?;
    let value: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
    let agent = value["agent"].as_str().ok_or("suite requires agent")?;
    let agent = path.parent().unwrap_or(Path::new(".")).join(agent);
    let spec = AgentSpec::from_yaml_strict(&std::fs::read_to_string(&agent)?)?;
    preflight_eval(&spec)?;
    Ok(agent)
}

/// Preserves YAML omission of task iteration limits instead of making deserialized defaults explicit.
pub fn load_application(path: &Path) -> Result<PreparedAgentSpec> {
    let prepared = PreparedAgentSpec::from_yaml_strict(&std::fs::read_to_string(path)?)?;
    preflight(prepared.spec())?;
    Ok(prepared)
}

/// Runs an already validated, explicitly authorized application through the existing CLI foreground driver.
/// Provider construction and domain installation stay shared with the executable; callers own consent and endpoint trust.
pub async fn run_application(spec: PreparedAgentSpec, path: &Path) -> Result<()> {
    let builder = ai_agents::AgentBuilder::from_prepared_spec_with_base_dir(
        spec,
        path.parent().unwrap_or(Path::new(".")),
    )
    .auto_configure_llms()?
    .auto_configure_features()?;
    let agent = task_domain::install_domain(builder)?
        .task_interaction_delivery(true)
        .build_task_host()
        .await?;
    ai_agents_cli::CliRepl::new(agent).hint("Research: /task run --profile research OBJECTIVE; operations/tool-free: /task run OBJECTIVE; support: ordinary input; /quit exits.").run().await
}
