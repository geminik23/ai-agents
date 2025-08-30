use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::snapshot::AgentSnapshot;
use crate::Result;

#[async_trait]
pub trait AgentStorage: Send + Sync {
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()>;
    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>>;
    async fn delete(&self, session_id: &str) -> Result<()>;
    async fn list_sessions(&self) -> Result<Vec<String>>;
}

pub struct FileStorage {
    base_path: PathBuf,
}

impl FileStorage {
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.base_path.join(format!("{}.json", session_id))
    }
}

#[async_trait]
impl AgentStorage for FileStorage {
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()> {
        tokio::fs::create_dir_all(&self.base_path).await?;
        let path = self.session_path(session_id);
        let json = serde_json::to_string_pretty(snapshot)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(None);
        }
        let json = tokio::fs::read_to_string(path).await?;
        let snapshot = serde_json::from_str(&json)?;
        Ok(Some(snapshot))
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut sessions = Vec::new();
        if !self.base_path.exists() {
            return Ok(sessions);
        }

        let mut entries = tokio::fs::read_dir(&self.base_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Some(name) = path.file_stem() {
                    sessions.push(name.to_string_lossy().to_string());
                }
            }
        }
        Ok(sessions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());

        let snapshot = AgentSnapshot::new("test-agent".into());
        storage.save("session-1", &snapshot).await.unwrap();

        let loaded = storage.load("session-1").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().agent_id, "test-agent");
    }

    #[tokio::test]
    async fn test_load_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());

        let loaded = storage.load("nonexistent").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_delete() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());

        let snapshot = AgentSnapshot::new("test-agent".into());
        storage.save("session-1", &snapshot).await.unwrap();
        assert!(storage.load("session-1").await.unwrap().is_some());

        storage.delete("session-1").await.unwrap();
        assert!(storage.load("session-1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());

        storage
            .save("session-1", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();
        storage
            .save("session-2", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();

        let sessions = storage.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.contains(&"session-1".to_string()));
        assert!(sessions.contains(&"session-2".to_string()));
    }
}
