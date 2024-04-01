use crate::{traits::LLMBackend, Error};
use serde::{Deserialize, Serialize};

const CHATGPT_URL: &'static str = "https://api.openai.com/v1/chat/completions";

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Default for Role {
    fn default() -> Self {
        Self::User
    }
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletion {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug)]
pub struct ChatGpt {
    api_key: String,
    model: String,
}

impl ChatGpt {
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }
}

#[async_trait::async_trait]
impl LLMBackend for ChatGpt {
    async fn generate_response(&self, temperature: f64, prompt: &str) -> Result<String, Error> {
        let chat_completion: ChatCompletion = ChatCompletion {
            model: self.model.clone(),
            messages: vec![Message {
                role: Role::System,
                content: prompt.to_string(),
                name: None,
            }],
            temperature,
        };

        let result: OpenAIChatResponse = ureq::post(CHATGPT_URL)
            .set("authorization", &format!("Bearer {}", self.api_key))
            .set("content-type", "application/json")
            .send_json(chat_completion)?
            .into_json()?;

        // dbg!(result);
        Ok(result.choices[0].message.content.clone())
    }
}

#[cfg(test)]
mod tests {

    use super::ChatGpt;
    use crate::traits::LLMBackend;

    #[ignore]
    #[test]
    fn calling_gpt() {
        dotenv::dotenv().ok();

        smol::block_on(async {
            let gpt = ChatGpt::new(std::env::var("OPEN_API_KEY").unwrap(), "gpt-4".into());
            let result = gpt.generate_response(0.1, "Say Hello.").await;

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), "Hello.");
        });
    }
}
