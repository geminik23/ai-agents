mod backends;
mod error;
pub mod message;
mod traits;

use backends::create_llm_model;
pub use error::Error;
use message::PromptMessageBuilder;
use traits::{LLMBackend, MessageBuilder};

//
//
//
pub enum Backend {
    ChatGPT { api_key: String, model: String },
    // TODO Future beackends
    // Llama2Cpu { path: String },
}

impl Backend {}

// manage the messages?
#[derive(Debug)]
pub struct Model {
    backend: Box<dyn LLMBackend>,
    temperature: f64,
}

impl Model {
    pub fn new(config: Backend) -> Result<Model, Error> {
        let backend = create_llm_model(config)?;
        Ok(Self {
            backend,
            temperature: 0.9,
        })
    }

    pub async fn generate_response<T>(&self, context_message_group: T) -> Result<String, Error>
    where
        T: IntoIterator + Send,
        T::Item: MessageBuilder + Send,
    {
        self.backend
            .generate_response(
                self.temperature,
                PromptMessageBuilder::new(context_message_group)
                    .build()
                    .as_str(),
            )
            .await
    }

    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_request() {
        dotenv::dotenv().ok();
        env_logger::init();

        assert!(Model::new(Backend::ChatGPT {
            api_key: env::var("OPEN_API_KEY").unwrap(),
            model: "gpt-3.5-turbo".into(),
        })
        .is_ok());
    }
}
