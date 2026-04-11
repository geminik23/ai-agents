use ai_agents_core::{Memory, Result};
use ai_agents_llm::LLMProvider;
use ai_agents_state::DelegateContextMode;

/// Prepare the input for a delegated agent based on the context mode.
pub async fn prepare_delegate_input(
    input: &str,
    mode: &DelegateContextMode,
    memory: &dyn Memory,
    llm: Option<&dyn LLMProvider>,
) -> Result<String> {
    match mode {
        DelegateContextMode::InputOnly => Ok(input.to_string()),

        DelegateContextMode::Full => {
            let messages = memory.get_messages(Some(20)).await?;
            if messages.is_empty() {
                return Ok(input.to_string());
            }
            let history = messages
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(format!(
                "Conversation history:\n{}\n\nCurrent message: {}",
                history, input
            ))
        }

        DelegateContextMode::Summary => {
            let messages = memory.get_messages(Some(20)).await?;
            if messages.is_empty() {
                return Ok(input.to_string());
            }

            if let Some(llm) = llm {
                let history = messages
                    .iter()
                    .map(|m| format!("{:?}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                let prompt = format!(
                    "Summarize this conversation in 2-3 sentences:\n\n{}",
                    history
                );
                let summary_messages = vec![
                    ai_agents_llm::ChatMessage::system(
                        "You are a conversation summarizer. Be brief and factual.",
                    ),
                    ai_agents_llm::ChatMessage::user(&prompt),
                ];
                match llm.complete(&summary_messages, None).await {
                    Ok(response) => Ok(format!(
                        "Context summary: {}\n\nCurrent message: {}",
                        response.content.trim(),
                        input
                    )),
                    Err(_) => Ok(input.to_string()),
                }
            } else {
                // No LLM available for summarization, fall back to input only.
                Ok(input.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::MemorySnapshot;
    use async_trait::async_trait;
    use std::sync::RwLock;

    struct FakeMemory {
        messages: RwLock<Vec<ai_agents_llm::ChatMessage>>,
    }

    impl FakeMemory {
        fn new(msgs: Vec<ai_agents_llm::ChatMessage>) -> Self {
            Self {
                messages: RwLock::new(msgs),
            }
        }
    }

    #[async_trait]
    impl Memory for FakeMemory {
        async fn add_message(&self, msg: ai_agents_llm::ChatMessage) -> Result<()> {
            self.messages.write().unwrap().push(msg);
            Ok(())
        }

        async fn get_messages(
            &self,
            limit: Option<usize>,
        ) -> Result<Vec<ai_agents_llm::ChatMessage>> {
            let msgs = self.messages.read().unwrap();
            match limit {
                Some(n) => Ok(msgs.iter().rev().take(n).rev().cloned().collect()),
                None => Ok(msgs.clone()),
            }
        }

        async fn clear(&self) -> Result<()> {
            self.messages.write().unwrap().clear();
            Ok(())
        }

        fn len(&self) -> usize {
            self.messages.read().unwrap().len()
        }

        async fn restore(&self, snapshot: MemorySnapshot) -> Result<()> {
            *self.messages.write().unwrap() = snapshot.messages;
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_input_only_returns_raw_input() {
        let mem = FakeMemory::new(vec![
            ai_agents_llm::ChatMessage::user("hello"),
            ai_agents_llm::ChatMessage::assistant("hi there"),
        ]);
        let result = prepare_delegate_input("new msg", &DelegateContextMode::InputOnly, &mem, None)
            .await
            .unwrap();
        assert_eq!(result, "new msg");
    }

    #[tokio::test]
    async fn test_full_includes_history() {
        let mem = FakeMemory::new(vec![
            ai_agents_llm::ChatMessage::user("hello"),
            ai_agents_llm::ChatMessage::assistant("hi there"),
        ]);
        let result = prepare_delegate_input("new msg", &DelegateContextMode::Full, &mem, None)
            .await
            .unwrap();
        assert!(result.contains("Conversation history:"));
        assert!(result.contains("hello"));
        assert!(result.contains("hi there"));
        assert!(result.contains("Current message: new msg"));
    }

    #[tokio::test]
    async fn test_full_empty_memory_returns_raw_input() {
        let mem = FakeMemory::new(vec![]);
        let result = prepare_delegate_input("new msg", &DelegateContextMode::Full, &mem, None)
            .await
            .unwrap();
        assert_eq!(result, "new msg");
    }

    #[tokio::test]
    async fn test_summary_without_llm_falls_back() {
        let mem = FakeMemory::new(vec![ai_agents_llm::ChatMessage::user("hello")]);
        let result = prepare_delegate_input("new msg", &DelegateContextMode::Summary, &mem, None)
            .await
            .unwrap();
        assert_eq!(result, "new msg");
    }
}
