use super::sync::Mutex;
pub use std::sync::Arc;
use std::{collections::HashMap, hash::Hash};

use serde::Deserialize;
use sllm::{message::PromptMessageGroup, Model};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Wrong output type")]
    WrongOutputType,
    #[error("Output is empty")]
    OutputIsEmpty,
    #[error("Input is required")]
    InputRequiredError,
    #[error("{0} not found.")]
    NotFound(String),
    #[error(transparent)]
    SLLMError(#[from] sllm::Error),
    #[error(transparent)]
    JsonParsingError(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub enum ModuleParam {
    Str(String),
    MessageBuilders(Vec<PromptMessageGroup>), // FIXME need to change the name of field.
    None,
}

impl ModuleParam {
    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            _ => false,
        }
    }

    pub fn into_message_group(self) -> Option<Vec<PromptMessageGroup>> {
        match self {
            Self::MessageBuilders(group) => Some(group),
            _ => None,
        }
    }

    pub fn into_string(self) -> Option<String> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_message_group(&self) -> Option<&Vec<PromptMessageGroup>> {
        match self {
            Self::MessageBuilders(group) => Some(group),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&String> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

impl Default for ModuleParam {
    fn default() -> Self {
        Self::None
    }
}

impl From<&str> for ModuleParam {
    fn from(val: &str) -> Self {
        ModuleParam::Str(val.into())
    }
}

impl From<Vec<PromptMessageGroup>> for ModuleParam {
    fn from(val: Vec<PromptMessageGroup>) -> Self {
        ModuleParam::MessageBuilders(val)
    }
}

impl From<String> for ModuleParam {
    fn from(val: String) -> Self {
        ModuleParam::Str(val)
    }
}

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

//
// TRAITS

#[async_trait::async_trait]
pub trait AgentTrait: std::fmt::Debug + Send + Sync {
    async fn execute(&mut self, model: &Model) -> Result<(), Error>;

    fn construct_param(&mut self) -> ModuleParam;

    fn get_result(&self) -> &ModuleParam;

    fn get_typed_result<T: for<'de> Deserialize<'de>>(&self) -> Result<T, Error> {
        match self.get_result() {
            ModuleParam::Str(result) => serde_json::from_str::<T>(result).map_err(|e| e.into()),
            ModuleParam::MessageBuilders(_) => Err(Error::WrongOutputType),
            ModuleParam::None => Err(Error::OutputIsEmpty),
        }
    }
}

#[async_trait::async_trait]
pub trait AgentModuleTrait: std::fmt::Debug + Send + Sync {
    fn get_name(&self) -> String;

    async fn execute(&mut self, model: &Model, input: ModuleParam) -> Result<ModuleParam, Error>;

    // async fn execute_typed<T: for<'de> Deserialize<'de>>(
    //     &mut self,
    //     model: &Model,
    //     input: ModuleParam,
    // ) -> Result<T, Error> {
    //     let result = self.execute(model, input).await;
    //     match result {
    //         Ok(param) => match param {
    //             ModuleParam::Str(result) => {
    //                 serde_json::from_str::<T>(&result).map_err(|e| e.into())
    //             }
    //             ModuleParam::MessageBuilders(_) => Err(Error::WrongOutputType),
    //             ModuleParam::None => Err(Error::OutputIsEmpty),
    //         },
    //         Err(err) => Err(err),
    //     }
    // }
}

#[derive(Debug)]
pub struct WrapperModule {
    name: String,
    internal: Arc<Mutex<dyn AgentModuleTrait>>,
}

impl WrapperModule {
    pub fn new(name: &str, module: Arc<Mutex<dyn AgentModuleTrait>>) -> Self {
        Self {
            name: name.into(),
            internal: module,
        }
    }
}

#[async_trait::async_trait]
impl AgentModuleTrait for WrapperModule {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    async fn execute(&mut self, model: &Model, input: ModuleParam) -> Result<ModuleParam, Error> {
        self.internal.lock().await.execute(model, input).await
    }
}

#[derive(Debug, Default)]
pub struct ModuleCascade {
    modules: Vec<Box<dyn AgentModuleTrait>>,
}

impl ModuleCascade {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_module<T>(&mut self, module: T)
    where
        T: 'static + AgentModuleTrait,
    {
        self.modules.push(Box::new(module));
    }
}

#[async_trait::async_trait]
impl AgentModuleTrait for ModuleCascade {
    fn get_name(&self) -> String {
        self.modules
            .iter()
            .map(|v| v.get_name())
            .collect::<Vec<_>>()
            .join(" - ")
    }

    async fn execute(&mut self, model: &Model, input: ModuleParam) -> Result<ModuleParam, Error> {
        let mut temp = input;
        for m in self.modules.iter_mut() {
            temp = m.execute(&model, temp).await?;
        }
        Ok(temp)
    }
}
