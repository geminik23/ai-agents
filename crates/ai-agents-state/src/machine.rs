use chrono::Utc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use ai_agents_core::{AgentError, Result, StateMachineSnapshot, StateTransitionEvent};

use super::config::{StateConfig, StateDefinition, ToolRef, Transition};

pub struct StateMachine {
    config: StateConfig,
    /// Serializes state-authority snapshots with transitions, reset, and restore.
    state_guard: RwLock<()>,
    /// Monotonic generation for changes that can invalidate state-scoped authorization.
    generation: AtomicU64,
    current: RwLock<String>,
    previous: RwLock<Option<String>>,
    turn_count: RwLock<u32>,
    no_transition_count: RwLock<u32>,
    history: RwLock<Vec<StateTransitionEvent>>,
}

impl StateMachine {
    pub fn new(config: StateConfig) -> Result<Self> {
        config.validate()?;
        let initial = Self::resolve_initial_state(&config)?;
        Ok(Self {
            config,
            state_guard: RwLock::new(()),
            generation: AtomicU64::new(1),
            current: RwLock::new(initial),
            previous: RwLock::new(None),
            turn_count: RwLock::new(0),
            no_transition_count: RwLock::new(0),
            history: RwLock::new(Vec::new()),
        })
    }

    fn resolve_initial_state(config: &StateConfig) -> Result<String> {
        let mut path = config.initial.clone();
        let mut current_def = config.states.get(&config.initial);

        while let Some(def) = current_def {
            if let (Some(initial_sub), Some(sub_states)) = (&def.initial, &def.states) {
                path = format!("{}.{}", path, initial_sub);
                current_def = sub_states.get(initial_sub);
            } else {
                break;
            }
        }

        Ok(path)
    }

    pub fn current(&self) -> String {
        self.current.read().clone()
    }

    pub fn previous(&self) -> Option<String> {
        self.previous.read().clone()
    }

    pub fn current_definition(&self) -> Option<StateDefinition> {
        let current = self.current.read();
        self.config.get_state(&current).cloned()
    }

    pub fn get_definition(&self, state: &str) -> Option<&StateDefinition> {
        self.config.get_state(state)
    }

    pub fn get_parent_definition(&self) -> Option<StateDefinition> {
        let current = self.current.read();
        let parts: Vec<&str> = current.split('.').collect();
        if parts.len() <= 1 {
            return None;
        }
        let parent_path = parts[..parts.len() - 1].join(".");
        self.config.get_state(&parent_path).cloned()
    }

    /// Returns the generation and every explicit tool scope from root to current state as one authority snapshot.
    pub fn current_tool_scope_snapshot(&self) -> (u64, Vec<Vec<ToolRef>>) {
        let _guard = self.state_guard.read();
        let current = self.current.read().clone();
        let mut path = String::new();
        let mut scopes = Vec::new();

        for part in current.split('.') {
            if !path.is_empty() {
                path.push('.');
            }
            path.push_str(part);
            if let Some(tools) = self
                .config
                .get_state(&path)
                .and_then(|definition| definition.tools.clone())
            {
                scopes.push(tools);
            }
        }

        (self.generation.load(Ordering::SeqCst), scopes)
    }

    /// Returns the current state-authority generation after any active mutation completes.
    pub fn generation(&self) -> u64 {
        let _guard = self.state_guard.read();
        self.generation.load(Ordering::SeqCst)
    }

    /// Applies one valid transition and advances the authority generation after all state fields are committed.
    pub fn transition_to(&self, state: &str, reason: &str) -> Result<()> {
        let _guard = self.state_guard.write();
        let current_path = self.current.read().clone();
        let resolved_path = self.config.resolve_full_path(&current_path, state);

        if self.config.get_state(&resolved_path).is_none() {
            return Err(AgentError::InvalidSpec(format!(
                "Unknown state: {} (resolved from {})",
                resolved_path, state
            )));
        }

        let final_path = self.resolve_to_leaf_state(&resolved_path)?;

        let from = {
            let mut current = self.current.write();
            let mut previous = self.previous.write();
            let from = current.clone();
            *previous = Some(from.clone());
            *current = final_path.clone();
            from
        };

        *self.turn_count.write() = 0;
        *self.no_transition_count.write() = 0;

        let event = StateTransitionEvent {
            from,
            to: final_path,
            reason: reason.to_string(),
            timestamp: Utc::now(),
        };
        self.history.write().push(event);
        self.generation.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    fn resolve_to_leaf_state(&self, path: &str) -> Result<String> {
        let mut current_path = path.to_string();

        loop {
            let def = self.config.get_state(&current_path).ok_or_else(|| {
                AgentError::InvalidSpec(format!("State not found: {}", current_path))
            })?;

            if let (Some(initial_sub), Some(sub_states)) = (&def.initial, &def.states)
                && sub_states.contains_key(initial_sub)
            {
                current_path = format!("{}.{}", current_path, initial_sub);
                continue;
            }
            break;
        }

        Ok(current_path)
    }

    pub fn available_transitions(&self) -> Vec<Transition> {
        let mut transitions = Vec::new();

        if let Some(def) = self.current_definition() {
            transitions.extend(def.transitions.clone());
        }

        transitions.extend(self.config.global_transitions.clone());

        transitions.sort_by_key(|transition| std::cmp::Reverse(transition.priority));
        transitions
    }

    pub fn auto_transitions(&self) -> Vec<Transition> {
        self.available_transitions()
            .into_iter()
            .filter(|t| t.auto)
            .collect()
    }

    pub fn history(&self) -> Vec<StateTransitionEvent> {
        self.history.read().clone()
    }

    pub fn increment_turn(&self) {
        let _guard = self.state_guard.write();
        *self.turn_count.write() += 1;
    }

    pub fn turn_count(&self) -> u32 {
        *self.turn_count.read()
    }

    pub fn increment_no_transition(&self) {
        let _guard = self.state_guard.write();
        *self.no_transition_count.write() += 1;
    }

    pub fn no_transition_count(&self) -> u32 {
        *self.no_transition_count.read()
    }

    pub fn reset_no_transition(&self) {
        let _guard = self.state_guard.write();
        *self.no_transition_count.write() = 0;
    }

    pub fn check_fallback(&self) -> Option<String> {
        if let Some(max) = self.config.max_no_transition
            && self.no_transition_count() >= max
        {
            return self.config.fallback.clone();
        }
        None
    }

    /// Restores the initial state and advances generation only when reset changes machine state.
    pub fn reset(&self) {
        let _guard = self.state_guard.write();
        let initial =
            Self::resolve_initial_state(&self.config).unwrap_or(self.config.initial.clone());
        let changed = *self.current.read() != initial
            || self.previous.read().is_some()
            || *self.turn_count.read() != 0
            || *self.no_transition_count.read() != 0
            || !self.history.read().is_empty();
        if !changed {
            return;
        }

        *self.current.write() = initial;
        *self.previous.write() = None;
        *self.turn_count.write() = 0;
        *self.no_transition_count.write() = 0;
        self.history.write().clear();
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Captures a consistent persistence snapshot after any active state mutation completes.
    pub fn snapshot(&self) -> StateMachineSnapshot {
        let _guard = self.state_guard.read();
        StateMachineSnapshot {
            current_state: self.current.read().clone(),
            previous_state: self.previous.read().clone(),
            turn_count: *self.turn_count.read(),
            no_transition_count: *self.no_transition_count.read(),
            history: self.history.read().clone(),
        }
    }

    /// Restores a valid snapshot and advances generation only when persisted machine state differs.
    pub fn restore(&self, snapshot: StateMachineSnapshot) -> Result<()> {
        if self.config.get_state(&snapshot.current_state).is_none() {
            return Err(AgentError::InvalidSpec(format!(
                "Snapshot contains unknown state: {}",
                snapshot.current_state
            )));
        }

        let _guard = self.state_guard.write();
        let current_history = self.history.read();
        let history_matches = current_history.len() == snapshot.history.len()
            && current_history
                .iter()
                .zip(&snapshot.history)
                .all(|(left, right)| {
                    left.from == right.from
                        && left.to == right.to
                        && left.reason == right.reason
                        && left.timestamp == right.timestamp
                });
        let changed = *self.current.read() != snapshot.current_state
            || *self.previous.read() != snapshot.previous_state
            || *self.turn_count.read() != snapshot.turn_count
            || *self.no_transition_count.read() != snapshot.no_transition_count
            || !history_matches;
        drop(current_history);
        if !changed {
            return Ok(());
        }

        *self.current.write() = snapshot.current_state;
        *self.previous.write() = snapshot.previous_state;
        *self.turn_count.write() = snapshot.turn_count;
        *self.no_transition_count.write() = snapshot.no_transition_count;
        *self.history.write() = snapshot.history;
        self.generation.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    pub fn config(&self) -> &StateConfig {
        &self.config
    }

    pub fn check_timeout(&self) -> Option<String> {
        let def = self.current_definition()?;
        let max_turns = def.max_turns?;
        let timeout_to = def.timeout_to.as_ref()?;
        if self.turn_count() >= max_turns {
            let current_path = self.current.read().clone();
            Some(self.config.resolve_full_path(&current_path, timeout_to))
        } else {
            None
        }
    }

    pub fn current_depth(&self) -> usize {
        self.current.read().split('.').count()
    }

    pub fn is_in_sub_state(&self) -> bool {
        self.current_depth() > 1
    }

    pub fn parent_state(&self) -> Option<String> {
        let current = self.current.read();
        let parts: Vec<&str> = current.split('.').collect();
        if parts.len() > 1 {
            Some(parts[..parts.len() - 1].join("."))
        } else {
            None
        }
    }

    pub fn root_state(&self) -> String {
        let current = self.current.read();
        current.split('.').next().unwrap_or(&current).to_string()
    }

    /// Check if a transition target is on cooldown (was recently transitioned to).
    pub fn is_on_cooldown(&self, target: &str, cooldown_turns: u32) -> bool {
        let history = self.history.read();
        let total_transitions = history.len();
        if total_transitions == 0 || cooldown_turns == 0 {
            return false;
        }
        // Look at the last `cooldown_turns` transitions
        let start = total_transitions.saturating_sub(cooldown_turns as usize);
        history[start..].iter().any(|e| e.to == target)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::TransitionTiming;
    use super::*;
    use std::collections::HashMap;

    fn create_test_config() -> StateConfig {
        let mut states = HashMap::new();
        states.insert(
            "greeting".into(),
            StateDefinition {
                prompt: Some("Welcome!".into()),
                transitions: vec![Transition {
                    to: "support".into(),
                    when: "needs help".into(),
                    guard: None,
                    intent: None,
                    auto: true,
                    priority: 10,
                    cooldown_turns: None,
                    timing: TransitionTiming::PostResponse,
                    requires_response: false,
                    run_extractors: false,
                }],
                ..Default::default()
            },
        );
        states.insert(
            "support".into(),
            StateDefinition {
                prompt: Some("How can I help?".into()),
                max_turns: Some(3),
                timeout_to: Some("escalation".into()),
                ..Default::default()
            },
        );
        states.insert(
            "escalation".into(),
            StateDefinition {
                prompt: Some("Escalating...".into()),
                ..Default::default()
            },
        );
        StateConfig {
            initial: "greeting".into(),
            states,
            global_transitions: vec![],
            fallback: None,
            max_no_transition: None,
            regenerate_on_transition: true,
        }
    }

    fn create_hierarchical_config() -> StateConfig {
        let mut sub_states = HashMap::new();
        sub_states.insert(
            "gathering_info".into(),
            StateDefinition {
                prompt: Some("Gathering info".into()),
                transitions: vec![Transition {
                    to: "proposing".into(),
                    when: "understood".into(),
                    guard: None,
                    intent: None,
                    auto: true,
                    priority: 0,
                    cooldown_turns: None,
                    timing: TransitionTiming::PostResponse,
                    requires_response: false,
                    run_extractors: false,
                }],
                ..Default::default()
            },
        );
        sub_states.insert(
            "proposing".into(),
            StateDefinition {
                prompt: Some("Proposing solution".into()),
                transitions: vec![Transition {
                    to: "^closing".into(),
                    when: "resolved".into(),
                    guard: None,
                    intent: None,
                    auto: true,
                    priority: 0,
                    cooldown_turns: None,
                    timing: TransitionTiming::PostResponse,
                    requires_response: false,
                    run_extractors: false,
                }],
                ..Default::default()
            },
        );

        let mut states = HashMap::new();
        states.insert(
            "problem_solving".into(),
            StateDefinition {
                prompt: Some("Problem solving".into()),
                initial: Some("gathering_info".into()),
                states: Some(sub_states),
                ..Default::default()
            },
        );
        states.insert(
            "closing".into(),
            StateDefinition {
                prompt: Some("Thank you".into()),
                ..Default::default()
            },
        );

        StateConfig {
            initial: "problem_solving".into(),
            states,
            global_transitions: vec![],
            fallback: None,
            max_no_transition: None,
            regenerate_on_transition: true,
        }
    }

    #[test]
    fn test_new_state_machine() {
        let config = create_test_config();
        let sm = StateMachine::new(config).unwrap();
        assert_eq!(sm.current(), "greeting");
        assert!(sm.previous().is_none());
        assert_eq!(sm.turn_count(), 0);
    }

    #[test]
    fn test_transition() {
        let config = create_test_config();
        let sm = StateMachine::new(config).unwrap();
        sm.transition_to("support", "user asked for help").unwrap();
        assert_eq!(sm.current(), "support");
        assert_eq!(sm.previous(), Some("greeting".into()));
        assert_eq!(sm.history().len(), 1);
    }

    #[test]
    fn test_turn_counting() {
        let config = create_test_config();
        let sm = StateMachine::new(config).unwrap();
        assert_eq!(sm.turn_count(), 0);
        sm.increment_turn();
        sm.increment_turn();
        assert_eq!(sm.turn_count(), 2);
        sm.transition_to("support", "reason").unwrap();
        assert_eq!(sm.turn_count(), 0);
    }

    #[test]
    fn test_timeout_check() {
        let config = create_test_config();
        let sm = StateMachine::new(config).unwrap();
        sm.transition_to("support", "needs help").unwrap();
        assert!(sm.check_timeout().is_none());
        sm.increment_turn();
        sm.increment_turn();
        sm.increment_turn();
        assert_eq!(sm.check_timeout(), Some("escalation".into()));
    }

    #[test]
    fn test_snapshot_restore() {
        let config = create_test_config();
        let sm = StateMachine::new(config.clone()).unwrap();
        sm.transition_to("support", "reason").unwrap();
        sm.increment_turn();

        let snapshot = sm.snapshot();
        assert_eq!(snapshot.current_state, "support");
        assert_eq!(snapshot.turn_count, 1);

        let sm2 = StateMachine::new(config).unwrap();
        sm2.restore(snapshot).unwrap();
        assert_eq!(sm2.current(), "support");
        assert_eq!(sm2.turn_count(), 1);
    }

    #[test]
    fn test_reset() {
        let config = create_test_config();
        let sm = StateMachine::new(config).unwrap();
        sm.transition_to("support", "reason").unwrap();
        sm.increment_turn();
        sm.reset();
        assert_eq!(sm.current(), "greeting");
        assert!(sm.previous().is_none());
        assert_eq!(sm.turn_count(), 0);
        assert!(sm.history().is_empty());
    }

    #[test]
    fn generation_tracks_only_successful_transition_reset_and_restore_changes() {
        let sm = StateMachine::new(create_test_config()).unwrap();
        let initial_generation = sm.generation();
        let initial_snapshot = sm.snapshot();

        sm.reset();
        sm.restore(initial_snapshot.clone()).unwrap();
        assert_eq!(sm.generation(), initial_generation);

        assert!(sm.transition_to("missing", "invalid").is_err());
        assert_eq!(sm.generation(), initial_generation);

        sm.transition_to("support", "valid").unwrap();
        let transition_generation = sm.generation();
        assert_eq!(transition_generation, initial_generation + 1);
        let transitioned_snapshot = sm.snapshot();

        sm.restore(transitioned_snapshot.clone()).unwrap();
        assert_eq!(sm.generation(), transition_generation);
        sm.increment_turn();
        assert_eq!(sm.generation(), transition_generation);

        sm.reset();
        let reset_generation = sm.generation();
        assert_eq!(reset_generation, transition_generation + 1);
        sm.reset();
        assert_eq!(sm.generation(), reset_generation);

        sm.restore(transitioned_snapshot).unwrap();
        assert_eq!(sm.generation(), reset_generation + 1);
        sm.restore(initial_snapshot).unwrap();
        assert_eq!(sm.generation(), reset_generation + 2);
    }

    #[test]
    fn current_tool_scope_snapshot_keeps_complete_ancestor_chain() {
        let yaml = r#"
initial: root
states:
  root:
    tools: [root_tool]
    initial: middle
    states:
      middle:
        initial: leaf
        states:
          leaf:
            tools: []
"#;
        let config: StateConfig = serde_yaml::from_str(yaml).unwrap();
        let sm = StateMachine::new(config).unwrap();

        let (generation, scopes) = sm.current_tool_scope_snapshot();

        assert_eq!(generation, sm.generation());
        assert_eq!(scopes.len(), 2);
        assert_eq!(scopes[0][0].id(), "root_tool");
        assert!(scopes[1].is_empty());
    }

    #[test]
    fn test_hierarchical_initial_state() {
        let config = create_hierarchical_config();
        let sm = StateMachine::new(config).unwrap();
        assert_eq!(sm.current(), "problem_solving.gathering_info");
    }

    #[test]
    fn test_hierarchical_transition_sibling() {
        let config = create_hierarchical_config();
        let sm = StateMachine::new(config).unwrap();
        assert_eq!(sm.current(), "problem_solving.gathering_info");

        sm.transition_to("proposing", "understood").unwrap();
        assert_eq!(sm.current(), "problem_solving.proposing");
    }

    #[test]
    fn test_hierarchical_transition_parent() {
        let config = create_hierarchical_config();
        let sm = StateMachine::new(config).unwrap();
        sm.transition_to("proposing", "understood").unwrap();
        sm.transition_to("^closing", "resolved").unwrap();
        assert_eq!(sm.current(), "closing");
    }

    #[test]
    fn test_current_depth() {
        let config = create_hierarchical_config();
        let sm = StateMachine::new(config).unwrap();
        assert_eq!(sm.current_depth(), 2);
        assert!(sm.is_in_sub_state());

        sm.transition_to("^closing", "done").unwrap();
        assert_eq!(sm.current_depth(), 1);
        assert!(!sm.is_in_sub_state());
    }

    #[test]
    fn test_parent_state() {
        let config = create_hierarchical_config();
        let sm = StateMachine::new(config).unwrap();
        assert_eq!(sm.parent_state(), Some("problem_solving".into()));

        sm.transition_to("^closing", "done").unwrap();
        assert!(sm.parent_state().is_none());
    }

    #[test]
    fn test_root_state() {
        let config = create_hierarchical_config();
        let sm = StateMachine::new(config).unwrap();
        assert_eq!(sm.root_state(), "problem_solving");

        sm.transition_to("^closing", "done").unwrap();
        assert_eq!(sm.root_state(), "closing");
    }

    #[test]
    fn test_no_transition_count() {
        let config = create_test_config();
        let sm = StateMachine::new(config).unwrap();

        assert_eq!(sm.no_transition_count(), 0);
        sm.increment_no_transition();
        sm.increment_no_transition();
        assert_eq!(sm.no_transition_count(), 2);

        sm.reset_no_transition();
        assert_eq!(sm.no_transition_count(), 0);
    }

    #[test]
    fn test_fallback() {
        let mut config = create_test_config();
        config.fallback = Some("escalation".into());
        config.max_no_transition = Some(3);

        let sm = StateMachine::new(config).unwrap();
        assert!(sm.check_fallback().is_none());

        sm.increment_no_transition();
        sm.increment_no_transition();
        sm.increment_no_transition();
        assert_eq!(sm.check_fallback(), Some("escalation".into()));
    }

    #[test]
    fn test_global_transitions() {
        let mut config = create_test_config();
        config.global_transitions = vec![Transition {
            to: "escalation".into(),
            when: "user is angry".into(),
            guard: None,
            intent: None,
            auto: true,
            priority: 100,
            cooldown_turns: None,
            timing: TransitionTiming::PostResponse,
            requires_response: false,
            run_extractors: false,
        }];

        let sm = StateMachine::new(config).unwrap();
        let transitions = sm.available_transitions();

        assert!(
            transitions
                .iter()
                .any(|t| t.to == "escalation" && t.priority == 100)
        );
        assert_eq!(transitions[0].to, "escalation");
    }

    #[test]
    fn test_get_parent_definition() {
        let config = create_hierarchical_config();
        let sm = StateMachine::new(config).unwrap();

        let parent = sm.get_parent_definition();
        assert!(parent.is_some());
        assert_eq!(parent.unwrap().prompt, Some("Problem solving".into()));

        sm.transition_to("^closing", "done").unwrap();
        assert!(sm.get_parent_definition().is_none());
    }
}
