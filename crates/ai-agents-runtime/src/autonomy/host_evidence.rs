//! Frozen host evidence bindings are live capabilities, not authority reconstructed from checkpoint labels.

use super::*;
use ai_agents_core::{AgentError, Result};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Supplies a bounded revision from host-maintained state without performing I/O or invoking providers/tools.
/// Tokens must change for every relevant target change, including replacement; they must not be reused after an ABA change.
/// The host owns synchronization with external mutations; a sample is not a lock or a cross-store transaction.
pub trait HostRevisionSource: Send + Sync {
    /// Returns the current opaque revision, or an error when freshness cannot be established.
    fn revision(&self) -> Result<String>;
}

/// Explicit host-confirmed loss of delivery is distinct from an ordinary delayed signal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExternalUnavailablePolicy {
    #[default]
    Stop,
    Fail,
}

#[derive(Clone)]
pub(super) struct HostTarget {
    pub target: String,
    pub source: Arc<dyn HostRevisionSource>,
}

impl AutonomyExtensions {
    /// Pins a check's actual host revision source before freeze; aliases of one target must share the same implementation.
    /// Registration grants no tool authority and never samples the source during configuration validation.
    pub fn register_validation_target(
        &mut self,
        check: &str,
        target: &str,
        source: Arc<dyn HostRevisionSource>,
    ) -> Result<()> {
        if check.trim().is_empty()
            || check.len() > 128
            || target.trim().is_empty()
            || target.len() > 128
            || self.validation_targets.len() >= 256
            || self.validation_targets.contains_key(check)
            || self
                .validation_targets
                .values()
                .any(|old| old.target == target && !Arc::ptr_eq(&old.source, &source))
        {
            return Err(AgentError::Config(
                "invalid or conflicting host revision binding".into(),
            ));
        }
        self.validation_targets.insert(
            check.into(),
            HostTarget {
                target: target.into(),
                source,
            },
        );
        Ok(())
    }

    /// Selects the unavailable policy for an already installed condition without installing a transport or polling service.
    pub fn set_condition_unavailable_policy(
        &mut self,
        condition: &str,
        policy: ExternalUnavailablePolicy,
    ) -> Result<()> {
        if !self.conditions.contains_key(condition) {
            return Err(AgentError::Config("unknown external condition".into()));
        }
        self.condition_policies.insert(condition.into(), policy);
        Ok(())
    }
}

impl FrozenAutonomyExtensions {
    /// Capability inventory reflects actual installed objects; each selected check still needs its own binding.
    pub(crate) fn has_host_revisions(&self) -> bool {
        !self.validation_targets.is_empty()
    }

    /// Resolves a check only in the frozen registry retained by the runtime, never from model-supplied labels.
    pub(crate) fn validation_target(&self, check: &str) -> Option<&str> {
        self.validation_targets
            .get(check)
            .map(|binding| binding.target.as_str())
    }

    /// Samples only the selected profile's targets, once per target, and fails closed on absent or oversized tokens.
    pub(crate) fn refresh_targets(&self, scope: &mut EvaluationScope) -> Result<()> {
        let mut revisions = BTreeMap::new();
        for target in scope.target_revisions.keys() {
            let binding = self
                .validation_targets
                .values()
                .find(|binding| &binding.target == target)
                .ok_or_else(|| {
                    AgentError::Config("host revision source is no longer bound".into())
                })?;
            let revision = binding.source.revision()?;
            if revision.trim().is_empty() || revision.len() > 512 {
                return Err(AgentError::Config("invalid host target revision".into()));
            }
            revisions.insert(target.clone(), revision);
        }
        scope.target_revisions = revisions;
        Ok(())
    }

    /// Rechecks freshness without changing an attempt's original identity; cleanup deliberately does not require this proof.
    pub(crate) fn ensure_targets_current(&self, scope: &EvaluationScope) -> Result<()> {
        let mut current = scope.clone();
        self.refresh_targets(&mut current)?;
        if current.target_revisions != scope.target_revisions {
            return Err(AgentError::Config(
                "host target changed since validation".into(),
            ));
        }
        Ok(())
    }

    /// Rechecks the exact current check target without requiring unrelated completion evidence to remain fresh during cleanup.
    pub(crate) fn ensure_driver_target(&self, driver: &ValidationDriverState) -> Result<()> {
        match self.validation_targets.get(&driver.check_id) {
            Some(binding) if driver.identity.target.as_deref() == Some(binding.target.as_str()) => {
                let revision = binding.source.revision()?;
                if revision.trim().is_empty()
                    || revision.len() > 512
                    || driver.identity.target_revision.as_ref() != Some(&revision)
                {
                    return Err(AgentError::Config(
                        "host target changed since validation".into(),
                    ));
                }
                Ok(())
            }
            None if driver.identity.target.is_none() => Ok(()),
            _ => Err(AgentError::Config("host revision binding changed".into())),
        }
    }

    /// Manual retained resume is a delivery endpoint; lack of a separate service does not imply unavailability.
    pub(crate) fn condition_unavailable_policy(
        &self,
        condition: &str,
    ) -> ExternalUnavailablePolicy {
        self.condition_policies
            .get(condition)
            .copied()
            .unwrap_or_default()
    }
}

/// Retains evaluated targets through asynchronous finalization; backends sample them inside completion admission.
pub(crate) struct HostEvidenceAdmission {
    pub extensions: Arc<FrozenAutonomyExtensions>,
    pub scope: EvaluationScope,
}
impl std::fmt::Debug for HostEvidenceAdmission {
    // Debug output contains only binding metadata, never host-supplied revisions or capability objects.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostEvidenceAdmission")
            .finish_non_exhaustive()
    }
}
impl TaskFinalAdmission for HostEvidenceAdmission {
    /// Samples frozen sources at backend admission, not a stale pre-await gate evaluation.
    fn validate(&self) -> Result<()> {
        self.extensions.ensure_targets_current(&self.scope)
    }
}

impl BoundAutonomyProfile {
    /// Captures the selected checks' host targets without importing unrelated registry entries or relabelling old evidence.
    pub(crate) fn capture_targets(&self, scope: &mut EvaluationScope) -> Result<()> {
        scope.target_revisions.retain(|target, _| {
            self.checks.iter().any(|check| {
                self.extensions.validation_target(&check.check.id) == Some(target.as_str())
            })
        });
        for check in &self.checks {
            if let Some(target) = self.extensions.validation_target(&check.check.id) {
                scope.target_revisions.entry(target.into()).or_default();
            }
        }
        self.extensions.refresh_targets(scope)
    }
}
