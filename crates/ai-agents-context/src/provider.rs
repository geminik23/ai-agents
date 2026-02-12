use async_trait::async_trait;
use serde_json::Value;

use ai_agents_core::Result;

#[async_trait]
pub trait ContextProvider: Send + Sync {
    async fn get(&self, key: &str, current_context: &Value) -> Result<Value>;
}
