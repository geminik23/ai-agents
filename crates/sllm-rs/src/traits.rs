use crate::error::Error;
use async_trait::async_trait;

#[async_trait]
pub trait LLMBackend: std::fmt::Debug + Send + Sync {
    async fn generate_response(&self, temperature: f64, prompt: &str) -> Result<String, Error>;
}

pub trait MessageBuilder {
    fn build(&self) -> String;
}
