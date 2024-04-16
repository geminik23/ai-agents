use std::sync::Arc;

use sllm::message::{MessageBuilder, PromptMessage};

pub mod sync;
pub mod units;

mod error;
mod pipeline_net;
mod prompt_manager;
mod traits;

pub use error::Error;
pub use pipeline_net::PipelineNet;
pub use prompt_manager::PromptManager;
pub use sllm::Backend;
pub use traits::*;

pub trait ToKeywordString {
    fn to_keyword_string() -> String;
}

pub mod prelude {
    pub use super::ToKeywordString;
    pub use ai_agent_macro::*;
    pub use sllm::message::{MessageBuilder, PromptMessage, TemplatedMessage};
}

#[derive(Debug, Clone)]
pub enum ModuleParam {
    Str(String),
    MessageBuilders(Vec<PromptMessage>),
    None,
}

impl ModuleParam {
    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            _ => false,
        }
    }

    pub fn into_message_group(self) -> Option<Vec<PromptMessage>> {
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

    pub fn as_message_group(&self) -> Option<&Vec<PromptMessage>> {
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

impl From<Vec<PromptMessage>> for ModuleParam {
    fn from(val: Vec<PromptMessage>) -> Self {
        ModuleParam::MessageBuilders(val)
    }
}

impl From<String> for ModuleParam {
    fn from(val: String) -> Self {
        ModuleParam::Str(val)
    }
}

//
// Model Wrapper
#[derive(Debug, Clone)]
pub struct Model {
    model: Arc<sync::Mutex<sllm::Model>>,
}

impl Model {
    pub fn new(backend: Backend) -> Result<Self, Error> {
        let model = sllm::Model::new(backend)?;
        Ok(Self {
            model: Arc::new(sync::Mutex::new(model)),
        })
    }

    pub async fn set_temperature(&self, temperature: f64) {
        let mut model = self.model.lock().await;
        model.set_temperature(temperature);
    }

    pub async fn generate_response<T>(&self, input: T) -> Result<String, Error>
    where
        T: IntoIterator + Send,
        T::Item: MessageBuilder + Send,
    {
        let model = self.model.lock().await;
        let result = model.generate_response(input).await?;
        Ok(result)
    }
}

// pub use sllm;

#[cfg(test)]
mod tests {
    use super::Model;

    pub fn get_model() -> Model {
        dotenv::dotenv().ok();
        Model::new(sllm::Backend::ChatGPT {
            api_key: std::env::var("OPEN_API_KEY").unwrap(),
            model: "gpt-3.5-turbo".into(),
        })
        .unwrap()
    }

    use super::ToKeywordString;
    use ai_agent_macro::KeywordString;

    #[allow(dead_code)]
    #[derive(KeywordString)]
    struct SubStruct {
        prop1: i32,
        prop2: f32,
        prop3: String,
    }

    #[allow(dead_code)]
    #[derive(KeywordString)]
    struct TestStruct {
        sub: SubStruct,
        prop: Vec<SubStruct>,
    }

    #[ignore]
    #[test]
    fn test_print_keyword() {
        assert_eq!(
            TestStruct::to_keyword_string(),
            "{sub{prop1, prop2, prop3}, prop[{prop1, prop2, prop3}]}"
        );
    }
}
