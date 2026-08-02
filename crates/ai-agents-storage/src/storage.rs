use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(test)]
use ai_agents_core::traits::storage::StorageCapability;
use ai_agents_core::{AgentError, AgentSnapshot, AgentStorage, Result};

const HASHED_PREFIX: &str = "v2-";
const ENCODED_PREFIX: &str = "v1-";
const ENCODED_EXTENSION: &str = "snapshot";
const ENVELOPE_VERSION: u8 = 1;

#[derive(Serialize, Deserialize)]
struct SnapshotEnvelope {
    version: u8,
    session_id: String,
    snapshot: AgentSnapshot,
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
        self.base_path.join(format!(
            "{}.{}",
            hashed_session_id(session_id),
            ENCODED_EXTENSION
        ))
    }

    fn encoded_session_path(&self, session_id: &str) -> PathBuf {
        self.base_path.join(format!(
            "{}.{}",
            encode_session_id(session_id),
            ENCODED_EXTENSION
        ))
    }

    fn legacy_session_path(&self, session_id: &str) -> Option<PathBuf> {
        if !is_safe_legacy_session_id(session_id) {
            return None;
        }
        Some(self.base_path.join(format!("{session_id}.json")))
    }
}

fn hashed_session_id(session_id: &str) -> String {
    let digest = Sha256::digest(session_id.as_bytes());
    format!("{HASHED_PREFIX}{digest:x}")
}

fn encode_session_id(session_id: &str) -> String {
    let mut encoded = String::with_capacity(ENCODED_PREFIX.len() + session_id.len() * 2);
    encoded.push_str(ENCODED_PREFIX);
    for byte in session_id.as_bytes() {
        use std::fmt::Write;
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn decode_session_id(encoded: &str) -> Option<String> {
    let encoded = encoded.strip_prefix(ENCODED_PREFIX)?;
    let chunks = encoded.as_bytes().chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return None;
    }

    let bytes = chunks
        .map(|chunk| {
            let high = decode_hex_digit(chunk[0])?;
            let low = decode_hex_digit(chunk[1])?;
            Some((high << 4) | low)
        })
        .collect::<Option<Vec<_>>>()?;
    String::from_utf8(bytes).ok()
}

fn decode_hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn is_safe_legacy_session_id(session_id: &str) -> bool {
    if session_id.is_empty() || session_id.contains('/') || session_id.contains('\\') {
        return false;
    }

    let filename = format!("{session_id}.json");
    let mut components = Path::new(&filename).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}

#[async_trait]
impl AgentStorage for FileStorage {
    async fn save(&self, session_id: &str, snapshot: &AgentSnapshot) -> Result<()> {
        tokio::fs::create_dir_all(&self.base_path).await?;
        let envelope = SnapshotEnvelope {
            version: ENVELOPE_VERSION,
            session_id: session_id.to_string(),
            snapshot: snapshot.clone(),
        };
        tokio::fs::write(
            self.session_path(session_id),
            serde_json::to_vec_pretty(&envelope)?,
        )
        .await?;

        let encoded_path = self.encoded_session_path(session_id);
        if encoded_path.exists() {
            tokio::fs::remove_file(encoded_path).await?;
        }
        if let Some(legacy_path) = self.legacy_session_path(session_id)
            && legacy_path.exists()
        {
            tokio::fs::remove_file(legacy_path).await?;
        }
        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<AgentSnapshot>> {
        let path = self.session_path(session_id);
        if path.exists() {
            let envelope: SnapshotEnvelope = serde_json::from_slice(&tokio::fs::read(path).await?)?;
            if envelope.version != ENVELOPE_VERSION || envelope.session_id != session_id {
                return Err(AgentError::Persistence(
                    "File storage envelope does not match the requested session".into(),
                ));
            }
            return Ok(Some(envelope.snapshot));
        }

        let encoded_path = self.encoded_session_path(session_id);
        let fallback = if encoded_path.exists() {
            Some(encoded_path)
        } else {
            self.legacy_session_path(session_id)
                .filter(|legacy_path| legacy_path.exists())
        };
        let Some(path) = fallback else {
            return Ok(None);
        };
        let snapshot = serde_json::from_slice(&tokio::fs::read(path).await?)?;
        Ok(Some(snapshot))
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        for path in [
            self.session_path(session_id),
            self.encoded_session_path(session_id),
        ] {
            if path.exists() {
                tokio::fs::remove_file(path).await?;
            }
        }
        if let Some(legacy_path) = self.legacy_session_path(session_id)
            && legacy_path.exists()
        {
            tokio::fs::remove_file(legacy_path).await?;
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut sessions = BTreeSet::new();
        if !self.base_path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = tokio::fs::read_dir(&self.base_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };

            if extension == ENCODED_EXTENSION && stem.starts_with(HASHED_PREFIX) {
                let envelope: SnapshotEnvelope =
                    serde_json::from_slice(&tokio::fs::read(&path).await?)?;
                if envelope.version != ENVELOPE_VERSION
                    || hashed_session_id(&envelope.session_id) != stem
                {
                    return Err(AgentError::Persistence(
                        "File storage envelope does not match its filename".into(),
                    ));
                }
                sessions.insert(envelope.session_id);
            } else if extension == ENCODED_EXTENSION {
                if let Some(session_id) = decode_session_id(stem) {
                    sessions.insert(session_id);
                }
            } else if extension == "json" && is_safe_legacy_session_id(stem) {
                sessions.insert(stem.to_string());
            }
        }
        Ok(sessions.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn reports_snapshot_capability() {
        let storage = FileStorage::new("unused");

        assert!(storage.supports(StorageCapability::Snapshot));
        assert!(!storage.supports(StorageCapability::SessionMetadata));
    }

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
        assert_eq!(
            sessions,
            vec!["session-1".to_string(), "session-2".to_string()]
        );
    }

    #[tokio::test]
    async fn arbitrary_session_ids_use_flat_round_trip_paths() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());
        let session_ids = [
            "../escape",
            "nested/session",
            "nested\\session",
            ".",
            "..",
            "unicode-雪",
            "",
        ];

        for session_id in session_ids {
            let path = storage.session_path(session_id);
            assert_eq!(path.parent(), Some(temp_dir.path()));
            assert_eq!(
                path.extension().and_then(|value| value.to_str()),
                Some("snapshot")
            );
            storage
                .save(session_id, &AgentSnapshot::new(session_id.to_string()))
                .await
                .unwrap();
        }

        let sessions = storage.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), session_ids.len());
        for session_id in session_ids {
            assert!(sessions.contains(&session_id.to_string()));
            assert_eq!(
                storage.load(session_id).await.unwrap().unwrap().agent_id,
                session_id
            );
        }

        let mut entries = tokio::fs::read_dir(temp_dir.path()).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            assert!(entry.file_type().await.unwrap().is_file());
        }
    }

    #[tokio::test]
    async fn fixed_length_filename_supports_long_session_ids() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());
        let session_id = "segment/".repeat(1024);
        let snapshot = AgentSnapshot::new("agent".into());

        storage.save(&session_id, &snapshot).await.unwrap();

        let path = storage.session_path(&session_id);
        assert!(path.file_name().unwrap().len() < 100);
        assert_eq!(
            storage.list_sessions().await.unwrap(),
            vec![session_id.clone()]
        );
        assert_eq!(
            storage.load(&session_id).await.unwrap().unwrap().agent_id,
            "agent"
        );
    }

    #[tokio::test]
    async fn envelope_mismatch_fails_closed() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());
        let requested = "requested";
        let envelope = SnapshotEnvelope {
            version: ENVELOPE_VERSION,
            session_id: "different".into(),
            snapshot: AgentSnapshot::new("agent".into()),
        };
        tokio::fs::write(
            storage.session_path(requested),
            serde_json::to_vec(&envelope).unwrap(),
        )
        .await
        .unwrap();

        assert!(matches!(
            storage.load(requested).await,
            Err(AgentError::Persistence(message)) if message.contains("does not match")
        ));
    }

    #[tokio::test]
    async fn reversible_v1_snapshot_is_read_and_migrated_on_save() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path());
        let session_id = "legacy/encoded";
        let old_path = storage.encoded_session_path(session_id);
        tokio::fs::write(
            &old_path,
            serde_json::to_vec(&AgentSnapshot::new("old".into())).unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(
            storage.load(session_id).await.unwrap().unwrap().agent_id,
            "old"
        );
        storage
            .save(session_id, &AgentSnapshot::new("new".into()))
            .await
            .unwrap();
        assert!(!old_path.exists());
        assert_eq!(
            storage.load(session_id).await.unwrap().unwrap().agent_id,
            "new"
        );
    }

    #[tokio::test]
    async fn legacy_fallback_accepts_only_safe_flat_ids() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("storage");
        tokio::fs::create_dir_all(&storage_path).await.unwrap();
        let storage = FileStorage::new(&storage_path);
        let snapshot = AgentSnapshot::new("legacy-agent".into());
        tokio::fs::write(
            storage_path.join("legacy.json"),
            serde_json::to_string(&snapshot).unwrap(),
        )
        .await
        .unwrap();
        tokio::fs::write(
            temp_dir.path().join("escape.json"),
            serde_json::to_string(&snapshot).unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(
            storage.load("legacy").await.unwrap().unwrap().agent_id,
            "legacy-agent"
        );
        assert_eq!(
            storage.list_sessions().await.unwrap(),
            vec!["legacy".to_string()]
        );
        assert!(storage.load("../escape").await.unwrap().is_none());

        storage.delete("legacy").await.unwrap();
        assert!(!storage_path.join("legacy.json").exists());
        assert!(temp_dir.path().join("escape.json").exists());
    }
}
