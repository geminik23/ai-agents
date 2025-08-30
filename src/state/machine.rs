use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::config::{StateConfig, StateDefinition, Transition};
use crate::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionEvent {
    pub from: String,
    pub to: String,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachineSnapshot {
    pub current_state: String,
    pub previous_state: Option<String>,
    pub turn_count: u32,
    pub history: Vec<StateTransitionEvent>,
}

pub struct StateMachine {
    config: StateConfig,
    current: RwLock<String>,
    previous: RwLock<Option<String>>,
    turn_count: RwLock<u32>,
    history: RwLock<Vec<StateTransitionEvent>>,
}

impl StateMachine {
    pub fn new(config: StateConfig) -> Result<Self> {
        config.validate()?;
        let initial = config.initial.clone();
        Ok(Self {
            config,
            current: RwLock::new(initial),
            previous: RwLock::new(None),
            turn_count: RwLock::new(0),
            history: RwLock::new(Vec::new()),
        })
    }

    pub fn current(&self) -> String {
        self.current.read().clone()
    }

    pub fn previous(&self) -> Option<String> {
        self.previous.read().clone()
    }

    pub fn current_definition(&self) -> Option<StateDefinition> {
        let current = self.current.read();
        self.config.states.get(&*current).cloned()
    }

    pub fn get_definition(&self, state: &str) -> Option<&StateDefinition> {
        self.config.states.get(state)
    }

    pub fn transition_to(&self, state: &str, reason: &str) -> Result<()> {
        if !self.config.states.contains_key(state) {
            return Err(AgentError::InvalidSpec(format!("Unknown state: {}", state)));
        }

        let from = {
            let mut current = self.current.write();
            let mut previous = self.previous.write();
            let from = current.clone();
            *previous = Some(from.clone());
            *current = state.to_string();
            from
        };

        *self.turn_count.write() = 0;

        let event = StateTransitionEvent {
            from,
            to: state.to_string(),
            reason: reason.to_string(),
            timestamp: Utc::now(),
        };
        self.history.write().push(event);

        Ok(())
    }

    pub fn available_transitions(&self) -> Vec<Transition> {
        self.current_definition()
            .map(|def| {
                let mut transitions = def.transitions.clone();
                transitions.sort_by(|a, b| b.priority.cmp(&a.priority));
                transitions
            })
            .unwrap_or_default()
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
        *self.turn_count.write() += 1;
    }

    pub fn turn_count(&self) -> u32 {
        *self.turn_count.read()
    }

    pub fn reset(&self) {
        *self.current.write() = self.config.initial.clone();
        *self.previous.write() = None;
        *self.turn_count.write() = 0;
        self.history.write().clear();
    }

    pub fn snapshot(&self) -> StateMachineSnapshot {
        StateMachineSnapshot {
            current_state: self.current.read().clone(),
            previous_state: self.previous.read().clone(),
            turn_count: *self.turn_count.read(),
            history: self.history.read().clone(),
        }
    }

    pub fn restore(&self, snapshot: StateMachineSnapshot) -> Result<()> {
        if !self.config.states.contains_key(&snapshot.current_state) {
            return Err(AgentError::InvalidSpec(format!(
                "Snapshot contains unknown state: {}",
                snapshot.current_state
            )));
        }
        *self.current.write() = snapshot.current_state;
        *self.previous.write() = snapshot.previous_state;
        *self.turn_count.write() = snapshot.turn_count;
        *self.history.write() = snapshot.history;
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
            Some(timeout_to.clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
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
                    auto: true,
                    priority: 10,
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
}
