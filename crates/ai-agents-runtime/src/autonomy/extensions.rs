//! Immutable, versioned host extensions; registration never exposes a model-callable tool.

use super::{
    EvaluationEvidence, EvaluationScope, ProgressInput, ProgressObservation, ValidationDecision,
    ValidationInput,
};
use ai_agents_core::autonomy::{AutonomyProfile, ValidationCheck, ValidationSchedule};
use ai_agents_core::{AgentError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Dependencies refer to earlier named checks, never recursively evaluated current-cycle progress.
#[derive(Debug, Clone, Default)]
pub struct AdapterDescriptor {
    pub id: String,
    pub contract_version: u32,
    pub checks: Vec<String>,
    pub tools: Vec<String>,
    pub needs_judge: bool,
    pub needs_planner: bool,
    pub requires_host_revision: bool,
    pub conditions: Vec<String>,
}

/// Callbacks are bounded pure calculations; all I/O is expressed as managed observation requests.
pub trait AutonomyValidator: Send + Sync {
    /// Returns immutable identity, input dependencies and capability needs.
    fn descriptor(&self) -> &AdapterDescriptor;
    /// Validates the adapter's opaque configuration before binding or replacing it.
    fn validate_config(&self, config: &Value) -> Result<()>;
    /// Returns completion, typed observations or a typed host wait, not direct effects.
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision>;
}

pub trait ProgressAdapter: Send + Sync {
    /// Progress dependencies cannot include the current cycle's progress-dependent gates.
    fn descriptor(&self) -> &AdapterDescriptor;
    /// Checks strict signal shape and bounded history requirements.
    fn validate_config(&self, config: &Value) -> Result<()>;
    /// Observes canonical current evidence without creating another mutable todo store.
    fn observe(&self, input: &ProgressInput<'_>) -> Result<ProgressObservation>;
}

/// Host-installed exact operation binding; YAML can select it but cannot construct a permit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostValidationBinding {
    pub id: String,
    pub validator: String,
    pub version: u32,
    pub profile: Option<String>,
    pub scope: String,
    pub tool: String,
    pub arguments: Value,
    pub require_approval: bool,
}

/// A small explicit signal schema, not a claim to implement arbitrary JSON Schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExternalSignalShape {
    Boolean,
    String {
        max_chars: usize,
    },
    Number,
    Object {
        fields: BTreeMap<String, ExternalSignalShape>,
    },
}

impl ExternalSignalShape {
    /// Strict shapes reject unknown fields and bounded recursion before accepting an external signal.
    pub fn accepts(&self, value: &Value) -> bool {
        self.accepts_at(value, 0)
    }
    // Host schemas and values share a depth bound; strings have explicit finite limits.
    fn accepts_at(&self, value: &Value, depth: usize) -> bool {
        if depth > 16 {
            return false;
        }
        match self {
            Self::Boolean => value.is_boolean(),
            Self::Number => value.as_f64().is_some_and(f64::is_finite),
            Self::String { max_chars } => {
                *max_chars > 0
                    && *max_chars <= 4096
                    && value
                        .as_str()
                        .is_some_and(|s| s.chars().count() <= *max_chars)
            }
            Self::Object { fields } => {
                fields.len() <= 32
                    && value.as_object().is_some_and(|obj| {
                        obj.len() == fields.len()
                            && fields.iter().all(|(key, shape)| {
                                obj.get(key).is_some_and(|v| shape.accepts_at(v, depth + 1))
                            })
                    })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostConditionBinding {
    pub id: String,
    pub version: u32,
    pub shape: ExternalSignalShape,
}

/// Capability inventory is supplied by host preflight; it is not an invocation permit.
#[derive(Debug, Clone, Default)]
pub struct ValidationCapabilities {
    pub tools: BTreeSet<String>,
    pub judge: bool,
    pub planner: bool,
    pub host_revisions: bool,
}

#[derive(Clone)]
pub struct BoundValidationCheck {
    pub check: ValidationCheck,
    pub stage: Option<String>,
    pub adapter: Arc<dyn AutonomyValidator>,
    pub available: bool,
    pub config_identity: String,
}

#[derive(Clone)]
pub struct BoundAutonomyProfile {
    pub checks: Vec<BoundValidationCheck>,
    pub progress: Arc<dyn ProgressAdapter>,
    pub progress_config: Value,
    pub profile: AutonomyProfile,
    pub extensions: Arc<FrozenAutonomyExtensions>,
}

impl BoundAutonomyProfile {
    /// Declaration-order scheduling never executes a later stage's validation while an earlier stage is active.
    pub fn scheduled_checks(
        &self,
        scope: &EvaluationScope,
        schedule: ValidationSchedule,
    ) -> Vec<&BoundValidationCheck> {
        self.checks
            .iter()
            .filter(|check| {
                check
                    .check
                    .schedule
                    .unwrap_or(ValidationSchedule::EachCycle)
                    == schedule
                    && check
                        .stage
                        .as_ref()
                        .is_none_or(|stage| scope.stage.as_ref() == Some(stage))
            })
            .collect()
    }

    /// Required aggregate validation is separate from validation_passed:true, which rejects an empty required set.
    pub fn required_outcome(
        &self,
        scope: &EvaluationScope,
        evidence: &EvaluationEvidence,
    ) -> Result<super::GateOutcome> {
        let evidence = evidence.normalized()?;
        let required: Vec<_> = self
            .checks
            .iter()
            .filter(|check| check.check.required.unwrap_or(true))
            .collect();
        if required.is_empty() {
            return Ok(super::GateOutcome::Pass);
        }
        Ok(super::GateOutcome::all(
            &required
                .iter()
                .map(|check| {
                    super::latest_validation(&evidence.validations, &check.check.id, scope)
                        .map_or(super::GateOutcome::Unknown, |r| r.outcome)
                })
                .collect::<Vec<_>>(),
        ))
    }

    /// A stage requires only global and current-stage checks, never a future stage's unavailable proof.
    pub fn required_stage_outcome(
        &self,
        scope: &EvaluationScope,
        evidence: &EvaluationEvidence,
    ) -> Result<super::GateOutcome> {
        let evidence = evidence.normalized()?;
        let required: Vec<_> = self
            .checks
            .iter()
            .filter(|check| {
                check.check.required.unwrap_or(true)
                    && check
                        .stage
                        .as_ref()
                        .is_none_or(|stage| scope.stage.as_ref() == Some(stage))
            })
            .collect();
        if required.is_empty() {
            return Ok(super::GateOutcome::Pass);
        }
        Ok(super::GateOutcome::all(
            &required
                .iter()
                .map(|check| {
                    super::latest_validation(&evidence.validations, &check.check.id, scope)
                        .map_or(super::GateOutcome::Unknown, |result| result.outcome)
                })
                .collect::<Vec<_>>(),
        ))
    }

    /// Supplies frozen binding identities; the host explicitly selects validation attempt identities.
    pub fn prepare_scope(&self, scope: &mut EvaluationScope) {
        scope.validation_bindings = self
            .checks
            .iter()
            .map(|c| (c.check.id.clone(), c.config_identity.clone()))
            .collect();
    }
    /// Progress observes only eligible captures and prior validation, never current progress-dependent gate output.
    pub fn observe_progress(
        &self,
        scope: &EvaluationScope,
        evidence: &EvaluationEvidence,
    ) -> Result<ProgressObservation> {
        let evidence = evidence.normalized()?;
        let stripped = progress_evidence(&evidence, scope);
        let observation = self.progress.observe(&ProgressInput {
            scope,
            evidence: &stripped,
            config: &self.progress_config,
        })?;
        let mut eligible = BTreeSet::new();
        if let Some(current) = evidence
            .current
            .as_ref()
            .filter(|o| o.complete && o.identity.eligible(scope, true, true))
        {
            eligible.insert(format!("current:{}", current.identity.sequence));
        }
        for check in &self.checks {
            if let Some(result) =
                super::latest_validation(&evidence.validations, &check.check.id, scope)
            {
                eligible.insert(format!(
                    "check:{}:{}",
                    check.check.id, result.identity.sequence
                ));
            }
        }
        if (!observation.signals.is_empty() && observation.evidence_refs.is_empty())
            || observation
                .evidence_refs
                .iter()
                .any(|r| !eligible.contains(r))
        {
            return Err(AgentError::Config(
                "progress references ineligible evidence".into(),
            ));
        }
        bounded_value(&serde_json::to_value(&observation)?, 65_536)?;
        Ok(observation)
    }
}

/// Mutable registration is consumed at freeze; previous frozen handles cannot silently hot-swap.
#[derive(Default)]
pub struct AutonomyExtensions {
    todo_bindings: BTreeMap<String, ai_agents_tools::TodoStore>,
    validators: BTreeMap<(String, u32), Arc<dyn AutonomyValidator>>,
    progress: BTreeMap<(String, u32), Arc<dyn ProgressAdapter>>,
    host_validation: BTreeMap<String, HostValidationBinding>,
    conditions: BTreeMap<String, HostConditionBinding>,
}

impl AutonomyExtensions {
    /// Installs the required built-ins without granting their tools or constructing providers.
    pub fn builtins() -> Self {
        let mut registry = Self::default();
        for kind in ["command", "diagnostics", "judge", "evidence"] {
            registry
                .register_validator(Arc::new(super::builtins::BuiltinValidator::new(kind)))
                .expect("unique builtin");
        }
        registry
            .register_validator(Arc::new(super::replanning::ReplanValidator::new()))
            .expect("unique internal replanner");
        for kind in ["evidence", "todo"] {
            registry
                .register_progress(Arc::new(super::progress::BuiltinProgress::new(kind)))
                .expect("unique builtin");
        }
        registry
    }
    /// Adds an identity once; duplicate registration never partially replaces an implementation.
    pub fn register_validator(&mut self, adapter: Arc<dyn AutonomyValidator>) -> Result<()> {
        let key = descriptor_key(adapter.descriptor())?;
        if self.validators.contains_key(&key) {
            return Err(AgentError::Config("duplicate autonomy validator".into()));
        }
        self.validators.insert(key, adapter);
        Ok(())
    }
    /// Explicit replacement is atomic and affects only future frozen registries.
    pub fn replace_validator(&mut self, adapter: Arc<dyn AutonomyValidator>) -> Result<()> {
        let key = descriptor_key(adapter.descriptor())?;
        if !self.validators.contains_key(&key) {
            return Err(AgentError::Config("unknown replacement validator".into()));
        }
        self.validators.insert(key, adapter);
        Ok(())
    }
    /// Adds a literal progress identity with a fixed contract version.
    pub fn register_progress(&mut self, adapter: Arc<dyn ProgressAdapter>) -> Result<()> {
        let key = descriptor_key(adapter.descriptor())?;
        if self.progress.contains_key(&key) {
            return Err(AgentError::Config("duplicate progress adapter".into()));
        }
        self.progress.insert(key, adapter);
        Ok(())
    }
    /// Registers exact host-only authority; it never changes the ordinary tool registry or grant.
    pub fn register_host_validation(&mut self, binding: HostValidationBinding) -> Result<()> {
        if binding.id.is_empty()
            || binding.tool.is_empty()
            || binding.validator.is_empty()
            || binding.version == 0
            || !matches!(binding.scope.as_str(), "task" | "state" | "skill")
            || !binding.arguments.is_object()
            || self.host_validation.contains_key(&binding.id)
        {
            return Err(AgentError::Config(
                "invalid or duplicate host validation binding".into(),
            ));
        }
        self.host_validation.insert(binding.id.clone(), binding);
        Ok(())
    }
    /// Registers external signal delivery without adding a polling service.
    pub fn register_condition(&mut self, binding: HostConditionBinding) -> Result<()> {
        if binding.id.is_empty()
            || binding.version == 0
            || self.conditions.contains_key(&binding.id)
        {
            return Err(AgentError::Config(
                "invalid or duplicate condition binding".into(),
            ));
        }
        bounded_value(&serde_json::to_value(&binding.shape)?, 16_384)?;
        self.conditions.insert(binding.id.clone(), binding);
        Ok(())
    }
    /// Registers an alternate canonical-todo capability; a tool's JSON output is never used as store authority.
    pub fn register_todo_binding(
        &mut self,
        id: String,
        store: ai_agents_tools::TodoStore,
    ) -> Result<()> {
        if id.trim().is_empty() || self.todo_bindings.contains_key(&id) {
            return Err(AgentError::Config(
                "invalid or duplicate canonical todo binding".into(),
            ));
        }
        self.todo_bindings.insert(id, store);
        Ok(())
    }

    /// Consumes registration before any runtime consumer captures implementations.
    pub fn freeze(self) -> Arc<FrozenAutonomyExtensions> {
        Arc::new(FrozenAutonomyExtensions {
            todo_bindings: self.todo_bindings,
            validators: self.validators,
            progress: self.progress,
            host_validation: self.host_validation,
            conditions: self.conditions,
        })
    }
}

pub struct FrozenAutonomyExtensions {
    todo_bindings: BTreeMap<String, ai_agents_tools::TodoStore>,
    validators: BTreeMap<(String, u32), Arc<dyn AutonomyValidator>>,
    progress: BTreeMap<(String, u32), Arc<dyn ProgressAdapter>>,
    host_validation: BTreeMap<String, HostValidationBinding>,
    conditions: BTreeMap<String, HostConditionBinding>,
}

impl FrozenAutonomyExtensions {
    /// Binding validates merged config and declaration-order dependencies before any observations execute.
    pub fn bind(
        self: &Arc<Self>,
        profile: &AutonomyProfile,
        capabilities: &ValidationCapabilities,
    ) -> Result<BoundAutonomyProfile> {
        let mut declarations = vec![];
        if let Some(gate) = &profile.completion {
            collect_gate_judges(gate, "root", None, &mut declarations)?;
        }
        collect_checks(profile.validation.as_ref(), None, &mut declarations)?;
        for stage in profile.lifecycle.as_deref().unwrap_or_default() {
            collect_checks(
                stage.validation.as_ref(),
                Some(&stage.id),
                &mut declarations,
            )?;
            if let Some(gate) = &stage.completion {
                collect_gate_judges(
                    gate,
                    &format!("stage.{}", stage.id),
                    Some(&stage.id),
                    &mut declarations,
                )?;
            }
            for (i, command) in stage
                .commands
                .as_deref()
                .unwrap_or_default()
                .iter()
                .enumerate()
            {
                declarations.push((
                    command_check(format!("__{}.command.{i}", stage.id), command),
                    Some(stage.id.clone()),
                ));
            }
        }
        if declarations.len() > 256 {
            return Err(AgentError::Config(
                "too many bound validation checks".into(),
            ));
        }
        let mut available_ids = BTreeSet::new();
        let mut checks = vec![];
        for (check, stage) in declarations {
            if check.adapter == "internal.replan" {
                return Err(AgentError::Config(
                    "replanner requires an explicit bounded intervention".into(),
                ));
            }
            if check
                .max_evaluation_rounds
                .is_some_and(|n| n == 0 || n > 32)
                || check
                    .max_observations_per_round
                    .is_some_and(|n| n == 0 || n > 16)
                || check.timeout_seconds == Some(0)
            {
                return Err(AgentError::Config("invalid validation check limits".into()));
            }
            if !available_ids.insert(check.id.clone()) {
                return Err(AgentError::Config("duplicate validation check".into()));
            }
            let adapter = self
                .validators
                .get(&(check.adapter.clone(), check.contract_version.unwrap_or(1)))
                .ok_or_else(|| AgentError::Config("unknown validator identity/version".into()))?
                .clone();
            let config = check
                .config
                .clone()
                .unwrap_or_else(|| serde_json::json!({}));
            adapter.validate_config(&config)?;
            if adapter
                .descriptor()
                .conditions
                .iter()
                .any(|id| !self.conditions.contains_key(id))
            {
                return Err(AgentError::Config(
                    "unknown declared external condition".into(),
                ));
            }
            if check.adapter == "builtin.evidence"
                && config
                    .get("validator")
                    .and_then(Value::as_str)
                    .is_some_and(|id| id == check.id || !available_ids.contains(id))
            {
                return Err(AgentError::Config(
                    "invalid evidence validator dependency".into(),
                ));
            }
            if adapter
                .descriptor()
                .checks
                .iter()
                .any(|id| id == &check.id || !available_ids.contains(id))
            {
                return Err(AgentError::Config(
                    "forward, cyclic or unknown validator dependency".into(),
                ));
            }
            if check.schedule == Some(ValidationSchedule::StageEnd) && stage.is_none() {
                return Err(AgentError::Config(
                    "stage_end check has no stage owner".into(),
                ));
            }
            let host_binding = config.get("host_binding").and_then(Value::as_str);
            let available = adapter.descriptor().tools.iter().all(|tool| {
                capabilities.tools.contains(tool)
                    || host_binding
                        .and_then(|id| self.host_validation.get(id))
                        .is_some_and(|b| {
                            b.validator == check.adapter
                                && b.version == check.contract_version.unwrap_or(1)
                                && b.tool == *tool
                        })
            }) && (!adapter.descriptor().needs_judge || capabilities.judge)
                && (!adapter.descriptor().needs_planner || capabilities.planner)
                && (!adapter.descriptor().requires_host_revision || capabilities.host_revisions);
            if check.required.unwrap_or(true) && !available {
                return Err(AgentError::Config(
                    "required validator capability/authority unavailable".into(),
                ));
            }
            checks.push(BoundValidationCheck {
                config_identity: canonical_identity(&serde_json::to_value(&check)?)?,
                check,
                stage,
                adapter,
                available,
            });
        }
        let config = profile.progress.as_ref();
        if config
            .and_then(|p| p.todo_tool.as_deref())
            .is_some_and(|id| id != "todo" && !self.todo_bindings.contains_key(id))
        {
            return Err(AgentError::Config(
                "alternate todo tool requires a registered canonical binding".into(),
            ));
        }
        let progress = self
            .progress
            .get(&(
                config
                    .and_then(|p| p.adapter.clone())
                    .unwrap_or_else(|| "builtin.evidence".into()),
                config.and_then(|p| p.contract_version).unwrap_or(1),
            ))
            .ok_or_else(|| AgentError::Config("unknown progress identity/version".into()))?
            .clone();
        let progress_config = config
            .and_then(|p| p.config.clone())
            .unwrap_or_else(|| serde_json::json!({}));
        progress.validate_config(&progress_config)?;
        if progress
            .descriptor()
            .conditions
            .iter()
            .any(|id| !self.conditions.contains_key(id))
            || progress
                .descriptor()
                .tools
                .iter()
                .any(|tool| !capabilities.tools.contains(tool))
            || (progress.descriptor().needs_judge && !capabilities.judge)
            || (progress.descriptor().needs_planner && !capabilities.planner)
            || (progress.descriptor().requires_host_revision && !capabilities.host_revisions)
        {
            return Err(AgentError::Config(
                "progress adapter dependency/capability unavailable".into(),
            ));
        }
        if progress
            .descriptor()
            .checks
            .iter()
            .any(|id| !available_ids.contains(id))
        {
            return Err(AgentError::Config("unknown progress dependency".into()));
        }
        super::progress::validate_signal_references(&progress_config, &available_ids)?;
        Ok(BoundAutonomyProfile {
            checks,
            progress,
            progress_config,
            profile: profile.clone(),
            extensions: self.clone(),
        })
    }
    /// Selected-profile preflight checks actual registered host operations, frozen aliases and canonical todo identity without provider calls.
    pub fn bind_for_agent(
        self: &Arc<Self>,
        agent: &crate::RuntimeAgent,
        profile: &super::EffectiveAutonomyProfile,
    ) -> Result<BoundAutonomyProfile> {
        let bound = self.bind(&profile.settings, &agent.validation_capabilities())?;
        if let Some(id) = profile
            .settings
            .progress
            .as_ref()
            .and_then(|p| p.todo_tool.as_deref())
            .filter(|id| *id != "todo")
            && !self.todo_bindings[id].shares_store(&agent.todo_store())
        {
            return Err(AgentError::Config(
                "alternate todo binding is not this runtime's canonical store".into(),
            ));
        }
        for check in &bound.checks {
            if check.check.required.unwrap_or(true)
                && check
                    .adapter
                    .descriptor()
                    .tools
                    .iter()
                    .any(|tool| !agent.validation_operation_available(tool))
            {
                return Err(AgentError::Config(
                    "required host operation unavailable".into(),
                ));
            }
            if check.check.adapter == "builtin.judge" {
                super::validation_judge_provider(
                    agent.llm_registry(),
                    &serde_json::from_value(check.check.config.clone().unwrap_or(Value::Null))?,
                )?;
            }
        }
        Ok(bound)
    }

    /// Immutable binding lookup is revalidated when observation execution reconstructs its permit.
    pub fn host_binding(&self, id: &str) -> Option<&HostValidationBinding> {
        self.host_validation.get(id)
    }
    /// A restored wait must match the installed version and signal shape.
    pub fn condition(&self, id: &str) -> Option<&HostConditionBinding> {
        self.conditions.get(id)
    }
}

// Empty or ambiguous descriptors never enter a registry.
fn descriptor_key(descriptor: &AdapterDescriptor) -> Result<(String, u32)> {
    if descriptor.id.trim().is_empty() || descriptor.contract_version == 0 {
        return Err(AgentError::Config(
            "invalid autonomy adapter descriptor".into(),
        ));
    }
    Ok((descriptor.id.clone(), descriptor.contract_version))
}

/// Bounds exact managed observation/configuration data instead of silently dropping required state.
pub fn bounded_value(value: &Value, limit: usize) -> Result<()> {
    // Iterative traversal bounds nested plugin data before recursive serialization.
    let mut stack = vec![(value, 0)];
    let mut visited = 0;
    while let Some((value, depth)) = stack.pop() {
        visited += 1;
        if depth > 32 || visited > 4096 {
            return Err(AgentError::Config(
                "autonomy data exceeds depth bound".into(),
            ));
        }
        match value {
            Value::Array(values) => stack.extend(values.iter().map(|v| (v, depth + 1))),
            Value::Object(values) => stack.extend(values.values().map(|v| (v, depth + 1))),
            _ => {}
        }
        if stack.len() > 4096 {
            return Err(AgentError::Config(
                "autonomy data exceeds node bound".into(),
            ));
        }
    }
    if serde_json::to_vec(value)?.len() > limit {
        return Err(AgentError::Config(
            "autonomy data exceeds byte bound".into(),
        ));
    }
    Ok(())
}

/// Exact canonical request identity avoids an additional hash dependency and never appears in display summaries.
pub fn canonical_identity(value: &Value) -> Result<String> {
    bounded_value(value, 65_536)?;
    // Object insertion order is normalized even if another workspace crate enables preserve_order.
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .map(|(k, v)| (k.clone(), sorted(v)))
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    Ok(serde_json::to_string(&sorted(value))?)
}

// Gate judges are scheduled through the same driver but are not promoted into unconditional required checks for an any branch.
fn collect_gate_judges(
    gate: &ai_agents_core::autonomy::CompletionGate,
    path: &str,
    stage: Option<&str>,
    checks: &mut Vec<(ValidationCheck, Option<String>)>,
) -> Result<()> {
    match gate {
        ai_agents_core::autonomy::CompletionGate::Judge(config) => checks.push((
            ValidationCheck {
                id: format!("__gate.{path}"),
                adapter: "builtin.judge".into(),
                contract_version: Some(1),
                required: Some(false),
                schedule: Some(ValidationSchedule::Completion),
                timeout_seconds: None,
                max_evaluation_rounds: None,
                max_observations_per_round: None,
                config: Some(serde_json::to_value(config)?),
            },
            stage.map(str::to_owned),
        )),
        ai_agents_core::autonomy::CompletionGate::All(children)
        | ai_agents_core::autonomy::CompletionGate::Any(children) => {
            for (index, child) in children.iter().enumerate() {
                collect_gate_judges(child, &format!("{path}.{index}"), stage, checks)?;
            }
        }
        ai_agents_core::autonomy::CompletionGate::Not(child) => {
            collect_gate_judges(child, &format!("{path}.not"), stage, checks)?
        }
        _ => {}
    }
    Ok(())
}

// Legacy convenience checks are normalized once into the same named adapter registry.
fn command_check(id: String, command: &str) -> ValidationCheck {
    ValidationCheck {
        id,
        adapter: "builtin.command".into(),
        contract_version: Some(1),
        required: Some(true),
        schedule: None,
        timeout_seconds: None,
        max_evaluation_rounds: None,
        max_observations_per_round: None,
        config: Some(serde_json::json!({"command": command})),
    }
}

// Stage-owned and top-level checks retain declaration order and explicit required semantics.
fn collect_checks(
    config: Option<&ai_agents_core::autonomy::ValidationConfig>,
    stage: Option<&str>,
    checks: &mut Vec<(ValidationCheck, Option<String>)>,
) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    for check in config.checks.as_deref().unwrap_or_default() {
        let mut check = check.clone();
        check.required = Some(check.required.unwrap_or(config.required.unwrap_or(true)));
        checks.push((check, stage.map(str::to_owned)));
    }
    for (i, command) in config
        .commands
        .as_deref()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        checks.push((
            command_check(format!("__{}.command.{i}", stage.unwrap_or("run")), command),
            stage.map(str::to_owned),
        ));
    }
    for kind in ["diagnostics", "judge"] {
        let enabled = if kind == "diagnostics" {
            config
                .diagnostics
                .as_ref()
                .and_then(|d| d.enabled)
                .unwrap_or(false)
        } else {
            config
                .judge
                .as_ref()
                .and_then(|j| j.enabled)
                .unwrap_or(false)
        };
        if enabled {
            let value = if kind == "judge" {
                let judge = config.judge.as_ref().unwrap();
                serde_json::json!({"criteria":judge.criteria.clone().unwrap_or_default(),"llm":judge.llm,"pass_threshold":judge.pass_threshold.unwrap_or(0.8)})
            } else {
                serde_json::json!({})
            };
            checks.push((
                ValidationCheck {
                    id: format!("__{}.{kind}", stage.unwrap_or("run")),
                    adapter: format!("builtin.{kind}"),
                    contract_version: Some(1),
                    required: Some(config.required.unwrap_or(true)),
                    schedule: None,
                    timeout_seconds: None,
                    max_evaluation_rounds: None,
                    max_observations_per_round: None,
                    config: Some(value),
                },
                stage.map(str::to_owned),
            ));
        }
    }
    Ok(())
}

/// Strips current progress from callback input, preventing same-cycle progress/gate dependency loops.
pub(crate) fn progress_evidence(
    evidence: &EvaluationEvidence,
    _scope: &EvaluationScope,
) -> EvaluationEvidence {
    let mut evidence = evidence.clone();
    evidence.progress = None;
    evidence
}
