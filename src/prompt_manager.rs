use std::{collections::HashMap, hash::Hash};

use sllm::message::PromptMessageGroup;

//
// PromptManager
#[derive(Debug)]
pub struct PromptManager<T: Hash + Eq> {
    prompts: HashMap<String, PromptMessageGroup>,
    patterns: HashMap<T, String>,
}

impl<T: Hash + Eq> PromptManager<T> {
    pub fn new() -> Self {
        Self {
            prompts: HashMap::new(),
            patterns: HashMap::new(),
        }
    }

    fn parse_pattern<'a>(pattern: &'a str) -> impl Iterator<Item = &'a str> {
        pattern.split_whitespace()
    }

    pub fn insert_prompt(&mut self, alias: &str, prompt: PromptMessageGroup) {
        self.prompts.insert(alias.into(), prompt);
    }

    pub fn register_pattern(&mut self, key: T, pattern: &str) {
        self.patterns.insert(key, pattern.into());
    }

    pub fn get(&self, key: T) -> Vec<PromptMessageGroup> {
        self.patterns
            .get(&key)
            .into_iter()
            .flat_map(|pattern| Self::parse_pattern(pattern))
            .filter_map(|alias| self.prompts.get(alias))
            .cloned()
            .collect()
    }
}
