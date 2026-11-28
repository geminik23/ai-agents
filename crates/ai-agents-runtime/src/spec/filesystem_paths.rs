//! Typed filesystem operands used when a host rebases a declarative specification.

use super::*;
use ai_agents_core::autonomy::{AutonomyProfile, CompletionGate};
use std::path::PathBuf;

// Visit nested state overrides without interpreting arbitrary state data as configuration.
fn visit_states(
    states: &mut HashMap<String, StateDefinition>,
    visit: &mut impl FnMut(&mut AutonomyProfile),
) {
    for state in states.values_mut() {
        if let Some(settings) = &mut state.autonomy {
            visit(&mut settings.settings);
        }
        if let Some(children) = &mut state.states {
            visit_states(children, visit);
        }
    }
}

// Artifact operands participate in identity comparisons, so gates must receive the same spelling as their observations.
fn visit_gate(gate: &mut CompletionGate, map: &dyn Fn(&str) -> Option<String>) {
    match gate {
        CompletionGate::ArtifactExists(path) => map_string(path, map),
        CompletionGate::All(children) | CompletionGate::Any(children) => {
            for child in children {
                visit_gate(child, map);
            }
        }
        CompletionGate::Not(child) => visit_gate(child, map),
        _ => {}
    }
}

// The caller can decline replacement, preserving the original spelling.
fn map_string(value: &mut String, map: &dyn Fn(&str) -> Option<String>) {
    if let Some(mapped) = map(value) {
        *value = mapped;
    }
}

impl AgentSpec {
    // Apply the same operation to all inline profiles; external skill files retain their own source.
    pub(crate) fn visit_profiles_mut(&mut self, mut visit: impl FnMut(&mut AutonomyProfile)) {
        visit(&mut self.autonomy.defaults);
        self.autonomy.profiles.values_mut().for_each(&mut visit);
        if let Some(state) = &mut self.states {
            visit_states(&mut state.states, &mut visit);
        }
        for skill in &mut self.skills {
            if let SkillRef::Inline(skill) = skill
                && let Some(settings) = &mut skill.autonomy
            {
                visit(&mut settings.settings);
            }
        }
    }

    /// Maps typed file references without inspecting prompt text, extension JSON or policy authority.
    pub fn map_filesystem_paths(&mut self, map: &dyn Fn(&str) -> Option<String>) {
        for policy in self.tool_security.tools.values_mut() {
            for path in policy.filesystem_paths_mut() {
                map_string(path, map);
            }
        }
        map_storage(&mut self.storage, map);
        if let Some(spawner) = &mut self.spawner {
            if let Some(storage) = &mut spawner.shared_storage {
                map_storage(storage, map);
            }
            for template in spawner.templates.values_mut() {
                if let TemplateSource::File { path } = template {
                    map_string(path, map);
                }
            }
            for child in &mut spawner.auto_spawn {
                map_string(&mut child.agent, map);
            }
        }
        for source in self.context.values_mut() {
            if let ContextSource::File { path, .. } = source {
                map_string(path, map);
            }
        }
        for skill in &mut self.skills {
            if let SkillRef::File { file } = skill
                && let Some(mapped) = file.to_str().and_then(map)
            {
                *file = PathBuf::from(mapped);
            }
        }
        map_string(&mut self.observability.export.path, map);
        if let Some(path) = &mut self.observability.cost.pricing_file {
            map_string(path, map);
        }
        self.visit_profiles_mut(|profile| {
            for gate in profile.completion.iter_mut().chain(
                profile
                    .lifecycle
                    .iter_mut()
                    .flatten()
                    .filter_map(|stage| stage.completion.as_mut()),
            ) {
                visit_gate(gate, map);
            }
        });
    }
}

// Only filesystem storage variants contain a local path.
fn map_storage(storage: &mut StorageConfig, map: &dyn Fn(&str) -> Option<String>) {
    match storage {
        StorageConfig::Sqlite(storage) => map_string(&mut storage.path, map),
        StorageConfig::File(storage) => map_string(&mut storage.path, map),
        StorageConfig::Redis(_) | StorageConfig::None => {}
    }
}
