//! Resolution and local validation of raw autonomy profiles without executing a task.

use std::collections::HashSet;

use ai_agents_core::autonomy::{
    AutonomyConfig, AutonomyMode, AutonomyOverride, AutonomyProfile, CompletionGate,
    DiagnosticsConfig, ExecutionStrategy, JudgeConfig, ProgressConfig, StagnationAction,
    StagnationConfig, SupervisionConfig, UsdAmount, ValidationConfig,
};
use ai_agents_core::{AgentError, Result};
use ai_agents_skills::{SkillDefinition, SkillRef};
use ai_agents_state::{StateConfig, StateDefinition};

/// The entry scope determines which task modes and default strategies are meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutonomyScope {
    Task,
    State,
    Skill,
}

/// Host ceilings are configuration bounds, not runtime invocation permits.
#[derive(Debug, Clone, Default)]
pub struct AutonomyHostCeilings {
    pub max_turns: Option<u32>,
    pub max_active_time_seconds: Option<u64>,
    pub max_wall_time_seconds: Option<u64>,
    pub max_llm_calls: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub max_command_calls: Option<u32>,
    pub max_cost_usd: Option<UsdAmount>,
    pub max_declared_write_paths: Option<u32>,
}

/// Selection output is configuration only; actual cost/write capability is rechecked at admission.
#[derive(Debug, Clone)]
pub struct EffectiveAutonomyProfile {
    pub profile: Option<String>,
    pub enabled: bool,
    pub mode: AutonomyMode,
    pub strategy: ExecutionStrategy,
    pub max_turns: u32,
    pub max_active_time_seconds: u64,
    pub max_wall_time_seconds: Option<u64>,
    pub max_llm_calls: u32,
    pub max_tool_calls: u32,
    pub max_command_calls: u32,
    pub max_cost_usd: Option<UsdAmount>,
    pub max_declared_write_paths: Option<u32>,
    pub settings: AutonomyProfile,
}

// Merges raw scalar fields only when present, preserving explicit false and empty lists.
fn merge_profile(base: &mut AutonomyProfile, overlay: &AutonomyProfile) -> Result<()> {
    macro_rules! replace {
        ($($field:ident),* $(,)?) => {
            $(if overlay.$field.is_some() { base.$field = overlay.$field.clone(); })*
        };
    }
    replace!(
        enabled,
        mode,
        max_turns,
        max_active_time_seconds,
        max_wall_time_seconds,
        max_llm_calls,
        max_tool_calls,
        max_command_calls,
        max_cost_usd,
        max_declared_write_paths,
        lifecycle,
        completion,
    );
    // A higher-precedence objective form replaces the other without inheriting a conflict.
    if let Some(objective) = &overlay.objective {
        base.objective = Some(objective.clone());
        base.objective_template = None;
    }
    if let Some(template) = &overlay.objective_template {
        base.objective_template = Some(template.clone());
        base.objective = None;
    }
    if let Some(incoming) = &overlay.execution {
        let current = base.execution.get_or_insert_with(Default::default);
        if incoming.strategy.is_some() {
            current.strategy = incoming.strategy;
        }
    }
    if let Some(incoming) = &overlay.supervision {
        let current = base
            .supervision
            .get_or_insert_with(SupervisionConfig::default);
        if incoming.stage_changes.is_some() {
            current.stage_changes = incoming.stage_changes;
        }
        if incoming.plan_changes.is_some() {
            current.plan_changes = incoming.plan_changes;
        }
    }
    if let Some(incoming) = &overlay.progress {
        let current = base.progress.get_or_insert_with(ProgressConfig::default);
        if incoming.adapter.is_some()
            || incoming.contract_version.is_some()
            || incoming.config.is_some()
        {
            if incoming
                .adapter
                .as_ref()
                .is_none_or(|id| id.trim().is_empty())
            {
                return Err(AgentError::InvalidSpec(
                    "progress binding replacement needs an adapter ID".into(),
                ));
            }
            current.adapter = incoming.adapter.clone();
            current.contract_version = incoming.contract_version;
            current.config = incoming.config.clone();
        }
        macro_rules! progress_fields {
            ($($field:ident),* $(,)?) => { $(if incoming.$field.is_some() { current.$field = incoming.$field.clone(); })* };
        }
        progress_fields!(
            todo_tool,
            require_todos,
            require_active_todo,
            enforce_for_tools,
            auto_create_todo_prompt,
            max_open_todos
        );
        if let Some(stagnation) = &incoming.stagnation {
            let target = current
                .stagnation
                .get_or_insert_with(StagnationConfig::default);
            macro_rules! stagnation_fields {
                ($($field:ident),* $(,)?) => { $(if stagnation.$field.is_some() { target.$field = stagnation.$field; })* };
            }
            stagnation_fields!(
                max_cycles_without_progress,
                action,
                max_replans,
                on_exhausted,
                on_unavailable
            );
        }
    }
    if let Some(incoming) = &overlay.validation {
        let current = base
            .validation
            .get_or_insert_with(ValidationConfig::default);
        if incoming.required.is_some() {
            current.required = incoming.required;
        }
        if incoming.commands.is_some() {
            current.commands = incoming.commands.clone();
        }
        if incoming.checks.is_some() {
            current.checks = incoming.checks.clone();
        }
        if let Some(diagnostics) = &incoming.diagnostics {
            let target = current
                .diagnostics
                .get_or_insert_with(DiagnosticsConfig::default);
            if diagnostics.enabled.is_some() {
                target.enabled = diagnostics.enabled;
            }
            if diagnostics.fail_on_error.is_some() {
                target.fail_on_error = diagnostics.fail_on_error;
            }
        }
        if let Some(judge) = &incoming.judge {
            let target = current.judge.get_or_insert_with(JudgeConfig::default);
            if judge.enabled.is_some() {
                target.enabled = judge.enabled;
            }
            if judge.llm.is_some() {
                target.llm = judge.llm.clone();
            }
            if judge.pass_threshold.is_some() {
                target.pass_threshold = judge.pass_threshold;
            }
            if judge.criteria.is_some() {
                target.criteria = judge.criteria.clone();
            }
        }
    }
    if let Some(incoming) = &overlay.premature_finish {
        let target = base.premature_finish.get_or_insert_with(Default::default);
        if incoming.action.is_some() {
            target.action = incoming.action;
        }
        if incoming.max_continuations.is_some() {
            target.max_continuations = incoming.max_continuations;
        }
        if incoming.prompt.is_some() {
            target.prompt = incoming.prompt.clone();
        }
    }
    if let Some(incoming) = &overlay.on_limit {
        let target = base.on_limit.get_or_insert_with(Default::default);
        if incoming.action.is_some() {
            target.action = incoming.action;
        }
    }
    if let Some(incoming) = &overlay.hitl {
        let target = base.hitl.get_or_insert_with(Default::default);
        if incoming.on_approval_required.is_some() {
            target.on_approval_required = incoming.on_approval_required;
        }
        if incoming.on_user_question.is_some() {
            target.on_user_question = incoming.on_user_question;
        }
    }
    if let Some(incoming) = &overlay.persistence {
        let target = base.persistence.get_or_insert_with(Default::default);
        if incoming.enabled.is_some() {
            target.enabled = incoming.enabled;
        }
        if incoming.save_after_each_step.is_some() {
            target.save_after_each_step = incoming.save_after_each_step;
        }
        if incoming.resume_on_start.is_some() {
            target.resume_on_start = incoming.resume_on_start;
        }
    }
    Ok(())
}

// Keeps the parser independent of a user objective supplied later by a task invocation.
fn validate_raw(profile: &AutonomyProfile) -> Result<()> {
    for (name, value) in [
        ("max_turns", profile.max_turns),
        ("max_llm_calls", profile.max_llm_calls),
        ("max_tool_calls", profile.max_tool_calls),
        ("max_command_calls", profile.max_command_calls),
        ("max_declared_write_paths", profile.max_declared_write_paths),
    ] {
        if value == Some(0) {
            return Err(AgentError::InvalidSpec(format!(
                "autonomy.{name} must be positive"
            )));
        }
    }
    for (name, value) in [
        ("max_active_time_seconds", profile.max_active_time_seconds),
        ("max_wall_time_seconds", profile.max_wall_time_seconds),
    ] {
        if value == Some(0) {
            return Err(AgentError::InvalidSpec(format!(
                "autonomy.{name} must be positive"
            )));
        }
    }
    if profile
        .objective
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
        || profile
            .objective_template
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
    {
        return Err(AgentError::InvalidSpec(
            "autonomy objective/template cannot be empty".into(),
        ));
    }
    if profile.objective.is_some() && profile.objective_template.is_some() {
        return Err(AgentError::InvalidSpec(
            "autonomy objective and objective_template conflict".into(),
        ));
    }
    if let Some(progress) = &profile.progress {
        if progress.contract_version == Some(0)
            || progress.max_open_todos == Some(0)
            || progress
                .adapter
                .as_ref()
                .is_some_and(|id| id.trim().is_empty())
            || progress
                .todo_tool
                .as_ref()
                .is_some_and(|id| id.trim().is_empty())
        {
            return Err(AgentError::InvalidSpec(
                "invalid progress binding or limit".into(),
            ));
        }
        if progress.adapter.is_none()
            && (progress.config.is_some() || progress.contract_version.is_some())
        {
            return Err(AgentError::InvalidSpec(
                "progress binding replacement needs an adapter ID".into(),
            ));
        }
        if let Some(stagnation) = &progress.stagnation
            && (stagnation.max_cycles_without_progress == Some(0)
                || (stagnation.action == Some(StagnationAction::Replan)
                    && stagnation.max_replans == Some(0)))
        {
            return Err(AgentError::InvalidSpec("invalid stagnation limits".into()));
        }
    }
    if let Some(premature) = &profile.premature_finish
        && premature.action == Some(ai_agents_core::autonomy::PrematureFinishAction::Continue)
        && premature.max_continuations == Some(0)
    {
        return Err(AgentError::InvalidSpec(
            "max_continuations must be positive".into(),
        ));
    }
    if let Some(validation) = &profile.validation {
        validate_validation(validation, false)?;
    }
    if let Some(stages) = &profile.lifecycle {
        let mut ids = HashSet::new();
        for (index, stage) in stages.iter().enumerate() {
            if stage.id.trim().is_empty() || !ids.insert(stage.id.as_str()) {
                return Err(AgentError::InvalidSpec(
                    "lifecycle stage IDs must be nonempty and unique".into(),
                ));
            }
            if stage.max_fix_cycles == Some(0) {
                return Err(AgentError::InvalidSpec(
                    "max_fix_cycles must be positive".into(),
                ));
            }
            if let Some(gate) = &stage.completion {
                gate.validate().map_err(AgentError::InvalidSpec)?;
            }
            if let Some(validation) = &stage.validation {
                validate_validation(validation, true)?;
            }
            if stage
                .commands
                .iter()
                .flatten()
                .any(|command| command.trim().is_empty())
            {
                return Err(AgentError::InvalidSpec(
                    "stage validation command cannot be empty".into(),
                ));
            }
            if stage.retry_on_failure == Some(true)
                && (stage.max_fix_cycles.is_none()
                    || !stage.on_failure_stage.as_ref().is_some_and(|target| {
                        stages[..index].iter().any(|earlier| earlier.id == *target)
                    }))
            {
                return Err(AgentError::InvalidSpec(
                    "validation retry needs an earlier stage and positive max_fix_cycles".into(),
                ));
            }
        }
    }
    if let Some(gate) = &profile.completion {
        gate.validate().map_err(AgentError::InvalidSpec)?;
    }
    Ok(())
}

// Stage-end checks require a stage owner; check lists keep explicit empty replacement semantics.
fn validate_validation(config: &ValidationConfig, in_stage: bool) -> Result<()> {
    for check in config.checks.iter().flatten() {
        if check.id.trim().is_empty()
            || check.adapter.trim().is_empty()
            || check.contract_version == Some(0)
            || check.timeout_seconds == Some(0)
            || check
                .max_evaluation_rounds
                .is_some_and(|n| n == 0 || n > 32)
            || check
                .max_observations_per_round
                .is_some_and(|n| n == 0 || n > 16)
            || (check.schedule == Some(ai_agents_core::autonomy::ValidationSchedule::StageEnd)
                && !in_stage)
        {
            return Err(AgentError::InvalidSpec(
                "invalid validation check binding or schedule".into(),
            ));
        }
    }
    if config
        .commands
        .iter()
        .flatten()
        .any(|command| command.trim().is_empty())
    {
        return Err(AgentError::InvalidSpec(
            "validation command cannot be empty".into(),
        ));
    }
    if let Some(judge) = &config.judge
        && judge
            .pass_threshold
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(AgentError::InvalidSpec("invalid judge threshold".into()));
    }
    Ok(())
}

// Check references after every replacement so a removed check cannot remain a success gate.
fn validate_resolved(profile: &AutonomyProfile, scope: AutonomyScope) -> Result<()> {
    validate_raw(profile)?;
    let enabled = profile.enabled.unwrap_or(false);
    if !enabled {
        return Ok(());
    }
    if let Some(stagnation) = profile
        .progress
        .as_ref()
        .and_then(|progress| progress.stagnation.as_ref())
        && (stagnation.action == Some(StagnationAction::AskUser)
            || stagnation.on_exhausted == Some(StagnationAction::AskUser))
        && !matches!(
            stagnation.on_unavailable,
            Some(StagnationAction::Stop | StagnationAction::Fail)
        )
    {
        return Err(AgentError::InvalidSpec(
            "ask_user stagnation needs an unavailable-interaction fallback".into(),
        ));
    }
    let mode = profile.mode.unwrap_or(AutonomyMode::UntilComplete);
    if mode == AutonomyMode::Disabled {
        return Err(AgentError::InvalidSpec(
            "enabled autonomy cannot use disabled mode".into(),
        ));
    }
    if matches!(
        (scope, mode),
        (
            AutonomyScope::Task,
            AutonomyMode::StateUntilExit | AutonomyMode::SkillUntilComplete
        ) | (AutonomyScope::State, AutonomyMode::SkillUntilComplete)
            | (AutonomyScope::Skill, AutonomyMode::StateUntilExit)
    ) {
        return Err(AgentError::InvalidSpec(
            "autonomy mode requires a matching state or skill scope".into(),
        ));
    }
    if profile.completion.is_none() {
        return Err(AgentError::InvalidSpec(
            "enabled autonomy requires a completion gate".into(),
        ));
    }
    let stages = profile.lifecycle.as_deref().unwrap_or_default();
    let strategy = profile
        .execution
        .as_ref()
        .and_then(|config| config.strategy)
        .unwrap_or(if scope == AutonomyScope::Skill {
            ExecutionStrategy::Skill
        } else if !stages.is_empty() {
            ExecutionStrategy::Lifecycle
        } else {
            ExecutionStrategy::ModelLed
        });
    match strategy {
        ExecutionStrategy::Skill if scope != AutonomyScope::Skill => {
            return Err(AgentError::InvalidSpec(
                "skill strategy needs a skill scope".into(),
            ));
        }
        ExecutionStrategy::Lifecycle if stages.is_empty() => {
            return Err(AgentError::InvalidSpec(
                "lifecycle strategy needs stages".into(),
            ));
        }
        ExecutionStrategy::ModelLed if !stages.is_empty() => {
            return Err(AgentError::InvalidSpec(
                "model_led cannot ignore lifecycle stages".into(),
            ));
        }
        _ => {}
    }
    let mut checks = HashSet::new();
    if let Some(validation) = &profile.validation {
        for check in validation.checks.iter().flatten() {
            if !checks.insert(check.id.as_str()) {
                return Err(AgentError::InvalidSpec(
                    "duplicate validation check ID".into(),
                ));
            }
        }
    }
    for stage in stages {
        if let Some(validation) = &stage.validation {
            for check in validation.checks.iter().flatten() {
                if !checks.insert(check.id.as_str()) {
                    return Err(AgentError::InvalidSpec(
                        "duplicate validation check ID".into(),
                    ));
                }
            }
        }
    }
    // A stage cannot prove ValidationPassed using checks owned by later stages.
    fn has_required(
        validation: &ValidationConfig,
        completion_checks: bool,
        builtins: bool,
    ) -> bool {
        validation
            .commands
            .as_ref()
            .is_some_and(|commands| !commands.is_empty())
            || validation.checks.iter().flatten().any(|check| {
                check.required.unwrap_or(true)
                    && (completion_checks
                        || check.schedule
                            != Some(ai_agents_core::autonomy::ValidationSchedule::Completion))
            })
            || (builtins
                && (validation
                    .diagnostics
                    .as_ref()
                    .is_some_and(|check| check.enabled == Some(true))
                    || validation
                        .judge
                        .as_ref()
                        .is_some_and(|check| check.enabled == Some(true))))
    }
    let has_required_validation = profile
        .validation
        .as_ref()
        .is_some_and(|validation| has_required(validation, true, true))
        || stages.iter().any(|stage| {
            stage
                .validation
                .as_ref()
                .is_some_and(|validation| has_required(validation, true, true))
                || stage
                    .commands
                    .as_ref()
                    .is_some_and(|commands| !commands.is_empty())
        });
    // A referenced check must exist in the eligible scope; empty required sets never prove validation.
    fn references(
        gate: &CompletionGate,
        checks: &HashSet<&str>,
        has_required_validation: bool,
    ) -> Result<()> {
        match gate {
            CompletionGate::ValidatorPassed(id) if !checks.contains(id.as_str()) => Err(
                AgentError::InvalidSpec(format!("unknown validator check '{id}'")),
            ),
            CompletionGate::ValidationPassed(true) if !has_required_validation => {
                Err(AgentError::InvalidSpec(
                    "validation_passed needs a required check or command".into(),
                ))
            }
            CompletionGate::All(items) | CompletionGate::Any(items) => {
                for item in items {
                    references(item, checks, has_required_validation)?;
                }
                Ok(())
            }
            CompletionGate::Not(item) => references(item, checks, has_required_validation),
            _ => Ok(()),
        }
    }
    references(
        profile.completion.as_ref().unwrap(),
        &checks,
        has_required_validation,
    )?;
    for (index, stage) in stages.iter().enumerate() {
        if let Some(gate) = &stage.completion {
            let mut eligible = HashSet::new();
            if let Some(validation) = &profile.validation {
                for check in validation.checks.iter().flatten() {
                    if check.schedule
                        != Some(ai_agents_core::autonomy::ValidationSchedule::Completion)
                    {
                        eligible.insert(check.id.as_str());
                    }
                }
            }
            for owned in &stages[..=index] {
                if let Some(validation) = &owned.validation {
                    for check in validation.checks.iter().flatten() {
                        if check.schedule
                            != Some(ai_agents_core::autonomy::ValidationSchedule::Completion)
                        {
                            eligible.insert(check.id.as_str());
                        }
                    }
                }
            }
            let stage_required = profile
                .validation
                .as_ref()
                .is_some_and(|validation| has_required(validation, false, false))
                || stages[..=index].iter().any(|owned| {
                    owned
                        .validation
                        .as_ref()
                        .is_some_and(|validation| has_required(validation, false, true))
                        || owned
                            .commands
                            .as_ref()
                            .is_some_and(|commands| !commands.is_empty())
                });
            references(gate, &eligible, stage_required)?;
        }
    }
    Ok(())
}

// Validates stored profile references and every declared state path without enabling a run.
pub(crate) fn validate_agent_config(
    config: &AutonomyConfig,
    states: Option<&StateConfig>,
    skills: &[SkillRef],
) -> Result<()> {
    validate_raw(&config.defaults)?;
    if config
        .default_profile
        .as_ref()
        .is_some_and(|name| !config.profiles.contains_key(name))
    {
        return Err(AgentError::InvalidSpec(
            "unknown autonomy.default_profile".into(),
        ));
    }
    for (name, profile) in &config.profiles {
        if name.trim().is_empty() {
            return Err(AgentError::InvalidSpec(
                "autonomy profile ID cannot be empty".into(),
            ));
        }
        validate_raw(profile)?;
        // Scoped-only profiles are checked in their matching scope; enabled defaults still need gates.
        let merged = resolve_raw(config, Some(name), None, None)?;
        let scope = match merged.mode {
            Some(AutonomyMode::StateUntilExit) => AutonomyScope::State,
            Some(AutonomyMode::SkillUntilComplete) => AutonomyScope::Skill,
            _ => AutonomyScope::Task,
        };
        validate_resolved(&merged, scope)?;
    }
    let default = resolve_raw(config, None, None, None)?;
    validate_resolved(&default, AutonomyScope::Task)?;
    // Nested state overrides inherit declared defaults but cannot silently widen task scope.
    fn state(
        config: &AutonomyConfig,
        definitions: &std::collections::HashMap<String, StateDefinition>,
    ) -> Result<()> {
        for definition in definitions.values() {
            if let Some(scoped) = &definition.autonomy {
                let raw = resolve_raw(config, None, Some(scoped), None)?;
                validate_resolved(&raw, AutonomyScope::State)?;
            }
            if let Some(children) = &definition.states {
                state(config, children)?;
            }
        }
        Ok(())
    }
    if let Some(states) = states {
        state(config, &states.states)?;
    }
    for skill in skills {
        if let SkillRef::Inline(definition) = skill {
            validate_skill(config, definition)?;
        }
    }
    Ok(())
}

// Reject only actually enabled default or scoped paths, not dormant named profiles.
pub(crate) fn has_enabled_declaration(
    config: &AutonomyConfig,
    states: Option<&StateConfig>,
    skills: &[SkillDefinition],
) -> bool {
    // Only a resolved enabled scope needs the interim build guard.
    fn enabled(config: &AutonomyConfig, scoped: Option<&AutonomyOverride>) -> bool {
        resolve_raw(config, None, scoped, None)
            .is_ok_and(|profile| profile.enabled.unwrap_or(false))
    }
    // Scan the installed machine so programmatic states cannot bypass the interim guard.
    fn state(
        config: &AutonomyConfig,
        definitions: &std::collections::HashMap<String, StateDefinition>,
    ) -> bool {
        definitions.values().any(|definition| {
            definition
                .autonomy
                .as_ref()
                .is_some_and(|scoped| enabled(config, Some(scoped)))
                || definition
                    .states
                    .as_ref()
                    .is_some_and(|children| state(config, children))
        })
    }
    enabled(config, None)
        || states.is_some_and(|states| state(config, &states.states))
        || skills.iter().any(|skill| {
            skill
                .autonomy
                .as_ref()
                .is_some_and(|scoped| enabled(config, Some(scoped)))
        })
}

// External skills are loaded later than spec parsing and must pass the same resolved checks.
pub(crate) fn validate_loaded_skills(
    config: &AutonomyConfig,
    skills: &[SkillDefinition],
) -> Result<()> {
    for skill in skills {
        validate_skill(config, skill)?;
    }
    Ok(())
}

// A selected skill override never creates tool authority or changes the outer run's owner.
fn validate_skill(config: &AutonomyConfig, skill: &SkillDefinition) -> Result<()> {
    if let Some(scoped) = &skill.autonomy {
        let raw = resolve_raw(config, None, Some(scoped), None)?;
        validate_resolved(&raw, AutonomyScope::Skill)?;
    }
    Ok(())
}

// Resolves selection before merging, including explicit per-run and scoped profile references.
fn resolve_raw(
    config: &AutonomyConfig,
    selected: Option<&str>,
    scoped: Option<&AutonomyOverride>,
    per_run: Option<&AutonomyProfile>,
) -> Result<AutonomyProfile> {
    let profile = selected
        .or_else(|| scoped.and_then(|scope| scope.profile.as_deref()))
        .or(config.default_profile.as_deref());
    let mut raw = config.defaults.clone();
    if let Some(profile) = profile {
        let selected = config.profiles.get(profile).ok_or_else(|| {
            AgentError::InvalidSpec(format!("unknown autonomy profile '{profile}'"))
        })?;
        merge_profile(&mut raw, selected)?;
    }
    if let Some(scoped) = scoped {
        merge_profile(&mut raw, &scoped.settings)?;
    }
    if let Some(per_run) = per_run {
        merge_profile(&mut raw, per_run)?;
    }
    Ok(raw)
}

/// Merges a chosen profile, scoped settings and invocation overrides before applying defaults and host ceilings.
/// This preflight does not authorize priced/provider or write-footprint execution.
pub fn resolve_profile(
    config: &AutonomyConfig,
    scope: AutonomyScope,
    selected: Option<&str>,
    scoped: Option<&AutonomyOverride>,
    per_run: Option<&AutonomyProfile>,
    objective: Option<&str>,
    host: &AutonomyHostCeilings,
) -> Result<EffectiveAutonomyProfile> {
    let profile = selected
        .or_else(|| scoped.and_then(|scope| scope.profile.as_deref()))
        .or(config.default_profile.as_deref());
    let raw = resolve_raw(config, selected, scoped, per_run)?;
    validate_resolved(&raw, scope)?;
    if [
        host.max_turns,
        host.max_llm_calls,
        host.max_tool_calls,
        host.max_command_calls,
        host.max_declared_write_paths,
    ]
    .into_iter()
    .flatten()
    .any(|limit| limit == 0)
        || [host.max_active_time_seconds, host.max_wall_time_seconds]
            .into_iter()
            .flatten()
            .any(|limit| limit == 0)
    {
        return Err(AgentError::InvalidSpec(
            "host autonomy ceilings must be positive".into(),
        ));
    }
    let enabled = raw.enabled.unwrap_or(false);
    if enabled
        && objective.is_none_or(|input| input.trim().is_empty())
        && raw.objective.is_none()
        && raw.objective_template.is_none()
    {
        return Err(AgentError::InvalidSpec(
            "task objective is required at invocation".into(),
        ));
    }
    let stages = raw.lifecycle.as_deref().unwrap_or_default();
    let strategy = raw
        .execution
        .as_ref()
        .and_then(|value| value.strategy)
        .unwrap_or(if scope == AutonomyScope::Skill {
            ExecutionStrategy::Skill
        } else if !stages.is_empty() {
            ExecutionStrategy::Lifecycle
        } else {
            ExecutionStrategy::ModelLed
        });
    macro_rules! cap {
        ($name:ident, $default:expr) => {{
            let requested = raw.$name.unwrap_or($default);
            requested.min(host.$name.unwrap_or(requested))
        }};
    }
    // A ceiling alone is not proof of a priced provider or trusted write footprint.
    // Prepared providers and final executor admission supply capability proof after numeric resolution.
    if enabled
        && let Some(cost) = &raw.max_cost_usd
        && host
            .max_cost_usd
            .as_ref()
            .is_none_or(|ceiling| cost.micro_usd() > ceiling.micro_usd())
    {
        return Err(AgentError::InvalidSpec(
            "priced cost exceeds or lacks a host ceiling".into(),
        ));
    }
    if enabled
        && let Some(paths) = raw.max_declared_write_paths
        && host
            .max_declared_write_paths
            .is_none_or(|ceiling| paths > ceiling)
    {
        return Err(AgentError::InvalidSpec(
            "declared write paths exceed or lack a host ceiling".into(),
        ));
    }
    Ok(EffectiveAutonomyProfile {
        profile: profile.map(str::to_string),
        enabled,
        mode: if enabled {
            raw.mode.unwrap_or(AutonomyMode::UntilComplete)
        } else {
            AutonomyMode::Disabled
        },
        strategy,
        max_turns: cap!(max_turns, 30),
        max_active_time_seconds: cap!(max_active_time_seconds, 900),
        max_wall_time_seconds: raw
            .max_wall_time_seconds
            .map(|requested| requested.min(host.max_wall_time_seconds.unwrap_or(requested)))
            .or(host.max_wall_time_seconds),
        max_llm_calls: cap!(max_llm_calls, 60),
        max_tool_calls: cap!(max_tool_calls, 120),
        max_command_calls: cap!(max_command_calls, 10),
        max_cost_usd: raw.max_cost_usd.clone(),
        max_declared_write_paths: raw.max_declared_write_paths,
        settings: raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::AgentSpec;

    // Reuse production parsing and local reference validation for schema fixtures.
    fn spec_with_autonomy(yaml: &str) -> AgentSpec {
        let spec = AgentSpec::from_yaml_strict(yaml).unwrap();
        spec.validate().unwrap();
        spec
    }

    // An omitted feature cannot alter existing YAML output or normal chat configuration.
    #[test]
    fn omitted_autonomy_stays_disabled_and_preserves_serialization() {
        let spec = spec_with_autonomy("name: Plain\nsystem_prompt: Just chat.\n");
        assert!(!serde_yaml::to_string(&spec).unwrap().contains("autonomy:"));
        let effective = resolve_profile(
            &spec.autonomy,
            AutonomyScope::Task,
            None,
            None,
            None,
            None,
            &AutonomyHostCeilings::default(),
        )
        .unwrap();
        assert!(!effective.enabled);
        assert_eq!(effective.mode, AutonomyMode::Disabled);
        assert_eq!(effective.max_turns, 30);
    }

    // Presence must survive all layers, while an adapter replacement must discard old config.
    #[test]
    fn merges_false_empty_lists_and_atomic_adapter_bindings() {
        let config: AutonomyConfig = serde_yaml::from_str(
            r#"
enabled: true
max_turns: 20
objective: "Default objective"
validation:
  commands: [check]
  checks:
    - {id: original, adapter: host.original}
progress:
  adapter: host.original
  config: {keep: false}
  require_todos: true
completion: {state: done}
profiles:
  custom:
    max_turns: 15
    lifecycle: [{id: review, completion: {state: done}}]
"#,
        )
        .unwrap();
        let scoped: AutonomyOverride = serde_yaml::from_str(
            r#"
profile: custom
enabled: false
objective_template: "{{ user_input }}"
lifecycle: []
validation: {commands: [], checks: []}
progress:
  adapter: host.replacement
  require_todos: false
"#,
        )
        .unwrap();
        let raw = resolve_raw(&config, None, Some(&scoped), None).unwrap();
        assert_eq!(raw.enabled, Some(false));
        assert!(raw.lifecycle.unwrap().is_empty());
        assert_eq!(
            raw.validation.unwrap().commands.unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            raw.progress.as_ref().unwrap().adapter.as_deref(),
            Some("host.replacement")
        );
        assert_eq!(raw.progress.as_ref().unwrap().require_todos, Some(false));
        assert_eq!(raw.progress.unwrap().config, None);
        assert_eq!(raw.objective, None);
        assert_eq!(raw.objective_template.as_deref(), Some("{{ user_input }}"));
        let effective = resolve_profile(
            &config,
            AutonomyScope::State,
            None,
            Some(&scoped),
            None,
            None,
            &AutonomyHostCeilings::default(),
        )
        .unwrap();
        assert!(!effective.enabled);
        assert_eq!(effective.max_turns, 15);
        let bad: AutonomyOverride =
            serde_yaml::from_str("progress: {config: {new: true}}").unwrap();
        assert!(
            resolve_profile(
                &config,
                AutonomyScope::State,
                None,
                Some(&bad),
                None,
                None,
                &AutonomyHostCeilings::default()
            )
            .is_err()
        );
    }

    // Build-time shape validation must not require a later user objective.
    #[test]
    fn enabled_profile_requires_gate_and_late_objective_then_clamps_host_limits() {
        let yaml = "name: Research\nsystem_prompt: Answer carefully.\nautonomy:\n  enabled: true\n  max_turns: 18\n  completion: {state: resolved}\n";
        let spec = spec_with_autonomy(yaml);
        assert!(
            resolve_profile(
                &spec.autonomy,
                AutonomyScope::Task,
                None,
                None,
                None,
                None,
                &AutonomyHostCeilings::default()
            )
            .is_err()
        );
        let host = AutonomyHostCeilings {
            max_turns: Some(4),
            max_active_time_seconds: Some(60),
            ..Default::default()
        };
        let resolved = resolve_profile(
            &spec.autonomy,
            AutonomyScope::Task,
            None,
            None,
            None,
            Some("Investigate"),
            &host,
        )
        .unwrap();
        assert!(resolved.enabled);
        assert_eq!(resolved.max_turns, 4);
        assert_eq!(resolved.max_active_time_seconds, 60);
        let invalid_host = AutonomyHostCeilings {
            max_tool_calls: Some(0),
            ..Default::default()
        };
        assert!(
            resolve_profile(
                &spec.autonomy,
                AutonomyScope::Task,
                None,
                None,
                None,
                Some("Investigate"),
                &invalid_host
            )
            .is_err()
        );
        let mut spec = spec;
        spec.autonomy.defaults.completion = None;
        assert!(spec.validate().is_err());
    }

    // Replacing a check list must invalidate old gates without rejecting a state-only profile.
    #[test]
    fn references_are_checked_after_list_replacement_and_scoped_modes_are_valid() {
        let spec = spec_with_autonomy(
            r#"
name: Support
system_prompt: Help with a case.
autonomy:
  profiles:
    support:
      enabled: true
      mode: state_until_exit
      validation:
        checks: [{id: case_result, adapter: host.ticket}]
      completion: {validator_passed: case_result}
states:
  initial: triage
  states:
    triage:
      autonomy: {profile: support}
"#,
        );
        assert!(
            resolve_profile(
                &spec.autonomy,
                AutonomyScope::Task,
                Some("support"),
                None,
                None,
                Some("Do it"),
                &AutonomyHostCeilings::default()
            )
            .is_err()
        );
        let state = spec.states.as_ref().unwrap().states["triage"]
            .autonomy
            .as_ref()
            .unwrap();
        assert!(
            resolve_profile(
                &spec.autonomy,
                AutonomyScope::State,
                None,
                Some(state),
                None,
                Some("Help"),
                &AutonomyHostCeilings::default()
            )
            .unwrap()
            .enabled
        );
        let bad: AutonomyOverride =
            serde_yaml::from_str("profile: support\nvalidation: {checks: []}").unwrap();
        assert!(
            resolve_profile(
                &spec.autonomy,
                AutonomyScope::State,
                None,
                Some(&bad),
                None,
                Some("Help"),
                &AutonomyHostCeilings::default()
            )
            .is_err()
        );
    }

    // Numeric host ceilings are not evidence of priced or footprint enforcement.
    #[test]
    fn hard_cap_resolution_requires_ceilings_but_does_not_mint_capabilities() {
        let mut config: AutonomyConfig =
            serde_yaml::from_str("enabled: true\ncompletion: {state: done}").unwrap();
        let host = AutonomyHostCeilings {
            max_cost_usd: Some(UsdAmount::parse("3.00").unwrap()),
            max_declared_write_paths: Some(5),
            ..Default::default()
        };
        config.defaults.max_cost_usd = Some(UsdAmount::parse("2.00").unwrap());
        let resolved = resolve_profile(
            &config,
            AutonomyScope::Task,
            None,
            None,
            None,
            Some("Task"),
            &host,
        )
        .unwrap();
        assert_eq!(resolved.max_cost_usd.unwrap().micro_usd(), 2_000_000);
        config.defaults.max_cost_usd = None;
        config.defaults.max_declared_write_paths = Some(2);
        let resolved = resolve_profile(
            &config,
            AutonomyScope::Task,
            None,
            None,
            None,
            Some("Task"),
            &host,
        )
        .unwrap();
        assert_eq!(resolved.max_declared_write_paths, Some(2));
        assert!(
            resolve_profile(
                &config,
                AutonomyScope::Task,
                None,
                None,
                None,
                Some("Task"),
                &AutonomyHostCeilings::default()
            )
            .is_err()
        );
    }

    // Inactive profiles must not block chat, while installed state overrides cannot bypass refusal.
    #[test]
    fn inactive_named_profile_does_not_change_chat_and_programmatic_state_is_guarded() {
        use crate::AgentBuilder;
        use ai_agents_llm::mock::MockLLMProvider;
        use std::sync::Arc;

        let dormant = spec_with_autonomy(
            "name: Agent\nsystem_prompt: Helpful.\nautonomy:\n  profiles:\n    later: {enabled: true, completion: {state: done}}\n",
        );
        AgentBuilder::from_spec(dormant)
            .llm(Arc::new(MockLLMProvider::new("test")))
            .build()
            .unwrap();

        let state_spec = spec_with_autonomy(
            "name: State\nsystem_prompt: Helpful.\nstates:\n  initial: work\n  states:\n    work:\n      autonomy: {enabled: true, mode: state_until_exit, completion: {state: done}}\n",
        );
        let machine =
            Arc::new(ai_agents_state::StateMachine::new(state_spec.states.unwrap()).unwrap());
        let result = AgentBuilder::new()
            .system_prompt("Helpful.")
            .llm(Arc::new(MockLLMProvider::new("test")))
            .state_machine(machine)
            .build();
        assert!(
            matches!(result, Err(AgentError::Config(message)) if message.contains("runner integration is installed"))
        );
    }

    // Merge fallback policy before checking ask-user safety and reject future-stage proof.
    #[test]
    fn layered_stagnation_fallback_and_stage_check_eligibility() {
        let valid: AutonomyConfig = serde_yaml::from_str("enabled: true\nprogress: {stagnation: {on_unavailable: stop}}\nprofiles:\n  support:\n    progress: {stagnation: {action: ask_user, max_cycles_without_progress: 2}}\n    completion: {state: done}\n").unwrap();
        let resolved = resolve_profile(
            &valid,
            AutonomyScope::Task,
            Some("support"),
            None,
            None,
            Some("Help"),
            &AutonomyHostCeilings::default(),
        )
        .unwrap();
        assert_eq!(
            resolved
                .settings
                .progress
                .unwrap()
                .stagnation
                .unwrap()
                .on_unavailable,
            Some(StagnationAction::Stop)
        );
        let missing = spec_with_autonomy(
            "name: Empty\nsystem_prompt: Helpful.\nautonomy: {enabled: false}\n",
        );
        assert!(!missing.autonomy.defaults.enabled.unwrap());
        let early_check = "name: Agent\nsystem_prompt: Helpful.\nautonomy:\n  enabled: true\n  lifecycle:\n    - id: first\n      completion: {validator_passed: later}\n    - id: second\n      validation: {checks: [{id: later, adapter: host.check, schedule: stage_end}]}\n  completion: {state: done}\n";
        let spec = AgentSpec::from_yaml_strict(early_check).unwrap();
        assert!(spec.validate().is_err());
        let late_validation = "name: Agent\nsystem_prompt: Helpful.\nautonomy:\n  enabled: true\n  lifecycle:\n    - id: first\n      completion: {validation_passed: true}\n    - id: second\n      validation: {checks: [{id: later, adapter: host.check, schedule: stage_end}]}\n  completion: {state: done}\n";
        let spec = AgentSpec::from_yaml_strict(late_validation).unwrap();
        assert!(spec.validate().is_err());
        let empty_command = "name: Agent\nsystem_prompt: Helpful.\nautonomy:\n  enabled: true\n  lifecycle: [{id: validate, commands: ['']}]\n  completion: {validation_passed: true}\n";
        let spec = AgentSpec::from_yaml_strict(empty_command).unwrap();
        assert!(spec.validate().is_err());
    }

    // Domain shapes share one strict raw profile even when host adapters are not installed.
    #[test]
    fn strict_complete_profile_fragments_round_trip_without_implying_execution() {
        let profiles = [
            "enabled: true\ncompletion: {todos_done: true}\nprogress: {require_todos: true}",
            "enabled: true\nexecution: {strategy: model_led}\nvalidation: {checks: [{id: sources_verified, adapter: host.research_sources, schedule: each_cycle, config: {min_unique_sources: 3}}]}\ncompletion: {validator_passed: sources_verified}",
            "enabled: true\nexecution: {strategy: model_led}\nvalidation: {checks: [{id: quality, adapter: builtin.judge, schedule: completion, config: {pass_threshold: 0.85}}]}\ncompletion: {validator_passed: quality}",
            "enabled: true\nmode: until_complete\nlifecycle: [{id: inspect}, {id: verify, validation: {checks: [{id: healthy, adapter: builtin.command, required: false, schedule: stage_end, config: {command: echo ok}}]}, completion: {validator_passed: healthy}}]\ncompletion: {validator_passed: healthy}",
            "enabled: true\nlifecycle: [{id: explore, completion: {tool_called: {id: file_read, count_gte: 1, executed: true, success: true}}}, {id: validate, commands: [cargo fmt --all -- --check], retry_on_failure: true, on_failure_stage: explore, max_fix_cycles: 2}]\ncompletion: {validation_passed: true}",
        ];
        for profile in profiles {
            let yaml = format!(
                "name: Example\nsystem_prompt: You are helpful.\nautonomy:\n{}\n",
                profile
                    .lines()
                    .map(|line| format!("  {line}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            let spec = spec_with_autonomy(&yaml);
            let encoded = serde_yaml::to_string(&spec).unwrap();
            let round_trip = AgentSpec::from_yaml_strict(&encoded).unwrap();
            round_trip.validate().unwrap();
            assert_eq!(round_trip.autonomy, spec.autonomy);
        }
    }

    // Enabled declarations must not masquerade as working autonomy before controller installation.
    #[test]
    fn enabled_yaml_and_rust_skill_builds_fail_instead_of_silently_chatting() {
        use crate::AgentBuilder;
        use ai_agents_llm::mock::MockLLMProvider;
        use std::sync::Arc;

        let enabled = spec_with_autonomy(
            "name: Agent\nsystem_prompt: You are helpful.\nautonomy: {enabled: true, completion: {state: done}}\n",
        );
        let error = match AgentBuilder::from_spec(enabled)
            .llm(Arc::new(MockLLMProvider::new("test")))
            .build()
        {
            Ok(_) => panic!("enabled autonomy cannot silently use chat"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("runner integration is installed"));

        let skill: SkillDefinition = serde_yaml::from_str("id: work\ndescription: Work\ntrigger: Work\nsteps: []\nautonomy: {enabled: true, mode: skill_until_complete, completion: {state: done}}\n").unwrap();
        let error = match AgentBuilder::new()
            .system_prompt("Helpful")
            .llm(Arc::new(MockLLMProvider::new("test")))
            .skill(skill)
            .build()
        {
            Ok(_) => panic!("programmatic skill autonomy cannot silently use chat"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("runner integration is installed"));

        let disabled = spec_with_autonomy(
            "name: Agent\nsystem_prompt: Helpful.\nautonomy: {enabled: false, completion: {state: done}}\n",
        );
        AgentBuilder::from_spec(disabled)
            .llm(Arc::new(MockLLMProvider::new("test")))
            .build()
            .unwrap();
    }

    // Loaded skills must follow the same strict YAML and scoped-reference rules as inline skills.
    #[test]
    fn external_skill_is_validated_after_loading_and_before_build() {
        use crate::AgentBuilder;
        use ai_agents_llm::mock::MockLLMProvider;
        use std::sync::Arc;

        let path =
            std::env::temp_dir().join(format!("autonomy-{}.skill.yaml", uuid::Uuid::new_v4()));
        let base = format!(
            "name: Agent\nsystem_prompt: Helpful.\nskills: [{{file: '{}'}}]\n",
            path.display()
        );
        std::fs::write(&path, "id: work\ndescription: Work\ntrigger: Work\nsteps: []\nautonomy: {enabled: true, profile: missing, completion: {state: done}}\n").unwrap();
        let result = AgentBuilder::from_yaml(&base)
            .unwrap()
            .llm(Arc::new(MockLLMProvider::new("test")))
            .build();
        assert!(
            matches!(result, Err(AgentError::InvalidSpec(message)) if message.contains("unknown autonomy profile"))
        );
        std::fs::write(&path, "id: work\ndescription: Work\ntrigger: Work\nsteps: []\nautonomy: {enabled: true, mode: skill_until_complete, completion: {state: done}}\n").unwrap();
        let result = AgentBuilder::from_yaml(&base)
            .unwrap()
            .llm(Arc::new(MockLLMProvider::new("test")))
            .build();
        assert!(
            matches!(result, Err(AgentError::Config(message)) if message.contains("runner integration is installed"))
        );
        std::fs::write(
            &path,
            "id: work\ndescription: Work\ntrigger: Work\nsteps: []\nautonomy: {enabeld: true}\n",
        )
        .unwrap();
        let result = AgentBuilder::from_yaml(&base)
            .unwrap()
            .llm(Arc::new(MockLLMProvider::new("test")))
            .build();
        assert!(result.is_err());
        std::fs::write(&path, "id: work\ndescription: Work\ntrigger: Work\nsteps: []\nautonomy: {progress: {adapter: host.check, config: {'<<': [unsafe]}}}\n").unwrap();
        let result = AgentBuilder::from_yaml(&base)
            .unwrap()
            .llm(Arc::new(MockLLMProvider::new("test")))
            .build();
        assert!(
            matches!(result, Err(AgentError::Skill(message)) if message.contains("Unsupported skill YAML key"))
        );
        std::fs::remove_file(path).unwrap();
    }

    // Complete domain examples prove schema fidelity, not executable task authority.
    #[test]
    fn complete_plan_agents_parse_and_round_trip_without_task_execution() {
        let agents = [
            r#"
name: ResearchAssistant
system_prompt: "You answer with grounded research."
tools: [todo]
autonomy:
  enabled: true
  mode: until_complete
  max_turns: 12
  progress:
    todo_tool: todo
    require_todos: true
  completion:
    all:
      - todos_done: true
      - response_not_empty: true
"#,
            r#"
name: ResearchRunner
system_prompt: "Investigate the objective, compare sources, and state uncertainty."
tools: [search_catalog, fetch_source]
autonomy:
  enabled: true
  default_profile: research
  profiles:
    research:
      mode: until_complete
      execution: {strategy: model_led}
      max_turns: 30
      max_active_time_seconds: 1200
      max_wall_time_seconds: 172800
      max_llm_calls: 80
      validation:
        checks:
          - id: sources_verified
            adapter: host.research_sources
            contract_version: 1
            required: true
            schedule: each_cycle
            config: {min_unique_sources: 3, require_citation_bindings: true}
          - id: answer_quality
            adapter: builtin.judge
            contract_version: 1
            required: true
            schedule: completion
            config:
              pass_threshold: 0.8
              criteria: ["The answer compares the available evidence and states uncertainty."]
      completion:
        all:
          - validator_passed: sources_verified
          - validator_passed: answer_quality
      progress:
        adapter: builtin.evidence
        config:
          signals: [{id: verified_sources, validator: sources_verified, path: metrics.unique_verified_sources, comparison: increase}]
        stagnation: {max_cycles_without_progress: 3, action: replan, max_replans: 2, on_exhausted: ask_user, on_unavailable: stop}
"#,
            r#"
name: IterativeAnalyst
system_prompt: "Develop a clear answer and improve it using the supplied feedback."
tools: []
autonomy:
  enabled: true
  execution: {strategy: model_led}
  max_turns: 8
  max_active_time_seconds: 300
  max_llm_calls: 20
  validation:
    checks:
      - id: quality
        adapter: builtin.judge
        contract_version: 1
        required: true
        schedule: completion
        config:
          pass_threshold: 0.85
          criteria: ["The answer addresses the requested criteria without contradictions."]
  completion: {validator_passed: quality}
  progress:
    adapter: builtin.evidence
    config:
      signals: [{id: quality_score, validator: quality, path: metrics.score, comparison: increase}]
    stagnation: {max_cycles_without_progress: 2, action: stop}
"#,
        ];
        for yaml in agents {
            let agent = spec_with_autonomy(yaml);
            let encoded = serde_yaml::to_string(&agent).unwrap();
            let round_trip = spec_with_autonomy(&encoded);
            assert_eq!(round_trip.autonomy, agent.autonomy);
        }
    }

    // Framework key typos and ambiguous gates must fail before any host integration is built.
    #[test]
    fn malformed_and_unknown_autonomy_fields_fail_strict_parsing() {
        for snippet in [
            "autonomy: {enabeld: true}",
            "autonomy: {enabled: true, completion: {state: ready, todos_done: true}}",
            "autonomy: {enabled: true, completion: {all: []}}",
            "autonomy: {enabled: true, mode: background}",
            "autonomy: {enabled: true, completion: {state: ready}, max_turns: 0}",
            "states: {initial: triage, states: {triage: {autonomy: {enabeld: true}}}}",
            "skills: [{id: work, description: Work, trigger: Tasks, steps: [], autonomy: {enabeld: true}}]",
        ] {
            let yaml = format!("name: Example\nsystem_prompt: Helpful.\n{snippet}\n");
            assert!(
                AgentSpec::from_yaml_strict(&yaml)
                    .and_then(|spec| spec.validate())
                    .is_err(),
                "{snippet}"
            );
        }
    }
}
