use async_trait::async_trait;
use serde_json::Value;

use ai_agents_core::Result;

#[async_trait]
pub trait ContextProvider: Send + Sync {
    async fn get(&self, key: &str, current_context: &Value) -> Result<Value>;

    /// Opts into task instruction loading only when dropping the future cannot leave untracked effects or detached disclosures.
    /// The provider must not invoke hidden models or perform side effects and must cooperate with cancellation and deadlines.
    fn task_instruction_safe(&self) -> bool {
        false
    }
}
