use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::{Value, json};

use ai_agents_core::{AgentError, Result};

use super::builtin::get_builtin_value;
use super::provider::ContextProvider;
use super::render::TemplateRenderer;
use super::source::{ContextSource, RefreshPolicy};

pub struct ContextManager {
    schema: HashMap<String, ContextSource>,
    values: RwLock<HashMap<String, Value>>,
    providers: RwLock<HashMap<String, Arc<dyn ContextProvider>>>,
    agent_name: String,
    agent_version: String,
    renderer: TemplateRenderer,
}

impl ContextManager {
    pub fn new(
        schema: HashMap<String, ContextSource>,
        agent_name: String,
        agent_version: String,
    ) -> Self {
        Self {
            schema,
            values: RwLock::new(HashMap::new()),
            providers: RwLock::new(HashMap::new()),
            agent_name,
            agent_version,
            renderer: TemplateRenderer::new(),
        }
    }

    pub fn set(&self, key: &str, value: Value) -> Result<()> {
        self.values.write().insert(key.to_string(), value);
        Ok(())
    }

    pub fn update(&self, path: &str, value: Value) -> Result<()> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return Err(AgentError::InvalidSpec("Empty path".into()));
        }

        let mut values = self.values.write();
        let root_key = parts[0];

        if parts.len() == 1 {
            values.insert(root_key.to_string(), value);
            return Ok(());
        }

        let root = values
            .entry(root_key.to_string())
            .or_insert_with(|| json!({}));

        let mut current = root;
        for part in &parts[1..parts.len() - 1] {
            current = current
                .as_object_mut()
                .ok_or_else(|| AgentError::InvalidSpec(format!("Path {} is not an object", path)))?
                .entry(*part)
                .or_insert_with(|| json!({}));
        }

        if let Some(obj) = current.as_object_mut() {
            obj.insert(parts[parts.len() - 1].to_string(), value);
        }

        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        self.values.read().get(key).cloned()
    }

    pub fn get_path(&self, path: &str) -> Option<Value> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return None;
        }

        let values = self.values.read();
        let mut current = values.get(parts[0])?;

        for part in &parts[1..] {
            current = current.get(part)?;
        }

        Some(current.clone())
    }

    pub fn get_all(&self) -> HashMap<String, Value> {
        self.values.read().clone()
    }

    pub async fn refresh(&self, key: &str) -> Result<()> {
        let source = self
            .schema
            .get(key)
            .ok_or_else(|| AgentError::InvalidSpec(format!("Unknown context key: {}", key)))?;

        let value = self.resolve_source(key, source).await?;
        if let Some(v) = value {
            self.values.write().insert(key.to_string(), v);
        }
        Ok(())
    }

    pub async fn refresh_per_turn(&self) -> Result<()> {
        for (key, source) in &self.schema {
            if source.refresh_policy() == RefreshPolicy::PerTurn {
                if let Some(value) = self.resolve_source(key, source).await? {
                    self.values.write().insert(key.clone(), value);
                }
            }
        }
        Ok(())
    }

    pub async fn refresh_per_session(&self) -> Result<()> {
        for (key, source) in &self.schema {
            match source.refresh_policy() {
                RefreshPolicy::PerSession | RefreshPolicy::Once => {
                    if let Some(value) = self.resolve_source(key, source).await? {
                        self.values.write().insert(key.clone(), value);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub async fn initialize(&self) -> Result<()> {
        for (key, source) in &self.schema {
            if let ContextSource::Runtime { default, .. } = source {
                if let Some(default_value) = default {
                    if !self.values.read().contains_key(key) {
                        self.values
                            .write()
                            .insert(key.clone(), default_value.clone());
                    }
                }
            } else if let Some(value) = self.resolve_source(key, source).await? {
                self.values.write().insert(key.clone(), value);
            }
        }
        Ok(())
    }

    pub fn register_provider(&self, name: &str, provider: Arc<dyn ContextProvider>) {
        self.providers.write().insert(name.to_string(), provider);
    }

    pub fn validate(&self) -> Result<()> {
        for (key, source) in &self.schema {
            if source.is_required() && !self.values.read().contains_key(key) {
                return Err(AgentError::InvalidSpec(format!(
                    "Required context '{}' not provided",
                    key
                )));
            }
        }
        Ok(())
    }

    pub fn snapshot(&self) -> HashMap<String, Value> {
        self.values.read().clone()
    }

    pub fn restore(&self, snapshot: HashMap<String, Value>) {
        *self.values.write() = snapshot;
    }

    async fn resolve_source(&self, key: &str, source: &ContextSource) -> Result<Option<Value>> {
        match source {
            ContextSource::Runtime { default, .. } => Ok(default.clone()),

            ContextSource::Builtin { source: src, .. } => Ok(Some(get_builtin_value(
                src,
                &self.agent_name,
                &self.agent_version,
            ))),

            ContextSource::File { path, fallback, .. } => {
                let current_context = self.get_all();
                let resolved_path = self.renderer.render_path(path, &current_context)?;

                match tokio::fs::read_to_string(&resolved_path).await {
                    Ok(content) => Ok(Some(Value::String(content))),
                    Err(_) => {
                        if let Some(fb) = fallback {
                            let fallback_path = self.renderer.render_path(fb, &current_context)?;
                            match tokio::fs::read_to_string(&fallback_path).await {
                                Ok(content) => Ok(Some(Value::String(content))),
                                Err(_) => Ok(None),
                            }
                        } else {
                            Ok(None)
                        }
                    }
                }
            }

            #[cfg(feature = "http-context")]
            ContextSource::Http {
                url,
                method,
                headers,
                timeout_ms,
                fallback,
                ..
            } => {
                let current_context = self.get_all();
                let resolved_url = self.renderer.render(url, &current_context)?;

                let client = reqwest::Client::new();
                let mut request = match method.to_uppercase().as_str() {
                    "POST" => client.post(&resolved_url),
                    "PUT" => client.put(&resolved_url),
                    "DELETE" => client.delete(&resolved_url),
                    _ => client.get(&resolved_url),
                };

                for (k, v) in headers {
                    let resolved_value = self.renderer.render(v, &current_context)?;
                    request = request.header(k, resolved_value);
                }

                if let Some(timeout) = timeout_ms {
                    request = request.timeout(std::time::Duration::from_millis(*timeout));
                }

                match request.send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            match response.json::<Value>().await {
                                Ok(json) => Ok(Some(json)),
                                Err(_) => Ok(fallback.clone()),
                            }
                        } else {
                            Ok(fallback.clone())
                        }
                    }
                    Err(_) => Ok(fallback.clone()),
                }
            }

            #[cfg(not(feature = "http-context"))]
            ContextSource::Http { fallback, .. } => {
                // HTTP context sources require the "http-context" feature
                tracing::warn!(
                    "HTTP context source requested but 'http-context' feature is not enabled"
                );
                Ok(fallback.clone())
            }

            ContextSource::Env { name } => Ok(std::env::var(name).ok().map(Value::String)),

            ContextSource::Callback { name, .. } => {
                // Clone the provider to avoid holding the lock across await
                let provider = {
                    let providers = self.providers.read();
                    providers.get(name).cloned()
                };
                if let Some(provider) = provider {
                    let current_context = json!(self.get_all());
                    Ok(Some(provider.get(key, &current_context).await?))
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub fn schema(&self) -> &HashMap<String, ContextSource> {
        &self.schema
    }
}

#[cfg(test)]
mod tests {
    use super::super::source::BuiltinSource;
    use super::*;

    #[test]
    fn test_set_and_get() {
        let manager = ContextManager::new(HashMap::new(), "Test".into(), "1.0".into());
        manager.set("user", json!({"name": "Alice"})).unwrap();
        let user = manager.get("user").unwrap();
        assert_eq!(user.get("name").unwrap(), "Alice");
    }

    #[test]
    fn test_update_nested() {
        let manager = ContextManager::new(HashMap::new(), "Test".into(), "1.0".into());
        manager.set("user", json!({"name": "Alice"})).unwrap();
        manager.update("user.tier", json!("premium")).unwrap();
        let user = manager.get("user").unwrap();
        assert_eq!(user.get("tier").unwrap(), "premium");
        assert_eq!(user.get("name").unwrap(), "Alice");
    }

    #[test]
    fn test_get_path() {
        let manager = ContextManager::new(HashMap::new(), "Test".into(), "1.0".into());
        manager
            .set("user", json!({"preferences": {"theme": "dark"}}))
            .unwrap();
        let theme = manager.get_path("user.preferences.theme").unwrap();
        assert_eq!(theme, "dark");
    }

    #[test]
    fn test_snapshot_restore() {
        let manager = ContextManager::new(HashMap::new(), "Test".into(), "1.0".into());
        manager.set("key1", json!("value1")).unwrap();
        manager.set("key2", json!(42)).unwrap();

        let snapshot = manager.snapshot();
        assert_eq!(snapshot.len(), 2);

        let manager2 = ContextManager::new(HashMap::new(), "Test".into(), "1.0".into());
        manager2.restore(snapshot);
        assert_eq!(manager2.get("key1").unwrap(), "value1");
        assert_eq!(manager2.get("key2").unwrap(), 42);
    }

    #[test]
    fn test_validate_required() {
        let mut schema = HashMap::new();
        schema.insert(
            "user".into(),
            ContextSource::Runtime {
                required: true,
                schema: None,
                default: None,
            },
        );

        let manager = ContextManager::new(schema, "Test".into(), "1.0".into());
        assert!(manager.validate().is_err());

        manager.set("user", json!({"name": "Alice"})).unwrap();
        assert!(manager.validate().is_ok());
    }

    #[tokio::test]
    async fn test_builtin_datetime() {
        let mut schema = HashMap::new();
        schema.insert(
            "time".into(),
            ContextSource::Builtin {
                source: BuiltinSource::Datetime,
                refresh: RefreshPolicy::PerTurn,
            },
        );

        let manager = ContextManager::new(schema, "Test".into(), "1.0".into());
        manager.initialize().await.unwrap();

        let time = manager.get("time").unwrap();
        assert!(time.get("date").is_some());
        assert!(time.get("time").is_some());
    }

    #[tokio::test]
    async fn test_env_source() {
        // SAFETY: This test runs single-threaded and no other code accesses this env var
        unsafe {
            std::env::set_var("TEST_CONTEXT_VAR", "test_value");
        }

        let mut schema = HashMap::new();
        schema.insert(
            "test_env".into(),
            ContextSource::Env {
                name: "TEST_CONTEXT_VAR".into(),
            },
        );

        let manager = ContextManager::new(schema, "Test".into(), "1.0".into());
        manager.initialize().await.unwrap();

        let value = manager.get("test_env").unwrap();
        assert_eq!(value, "test_value");
    }

    #[tokio::test]
    async fn test_runtime_default() {
        let mut schema = HashMap::new();
        schema.insert(
            "settings".into(),
            ContextSource::Runtime {
                required: false,
                schema: None,
                default: Some(json!({"theme": "light"})),
            },
        );

        let manager = ContextManager::new(schema, "Test".into(), "1.0".into());
        manager.initialize().await.unwrap();

        let settings = manager.get("settings").unwrap();
        assert_eq!(settings.get("theme").unwrap(), "light");
    }
}
