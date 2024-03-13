mod chatgpt;
use crate::{traits::LLMBackend, Backend, Error};

pub fn create_llm_model(config: Backend) -> Result<Box<dyn LLMBackend>, Error> {
    match config {
        Backend::ChatGPT { api_key, model } => Ok(Box::new(chatgpt::ChatGpt::new(api_key, model))),
    }
}
