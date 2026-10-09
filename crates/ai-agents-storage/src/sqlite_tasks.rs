//! SQLite task transitions hold a write transaction before observing ownership.

use ai_agents_core::autonomy::*;
use ai_agents_core::{AgentError, Result};
use sqlx::{Row, Sqlite, Transaction};

use super::sqlite::SqliteStorage;

fn persistence(error: sqlx::Error) -> AgentError {
    AgentError::Persistence(error.to_string())
}

pub(super) async fn migrate(pool: &sqlx::SqlitePool) -> Result<()> {
    for sql in [
        "CREATE TABLE IF NOT EXISTS task_runs (agent_id TEXT NOT NULL, run_id TEXT NOT NULL, actor_id TEXT, schema_version INTEGER NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL, owner_token TEXT, cancel_requested INTEGER NOT NULL, snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(agent_id, run_id))",
        "CREATE INDEX IF NOT EXISTS idx_task_runs_agent_status ON task_runs(agent_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_task_runs_actor ON task_runs(agent_id, actor_id)",
        "CREATE TABLE IF NOT EXISTS task_run_tombstones (agent_id TEXT NOT NULL, run_id TEXT NOT NULL, PRIMARY KEY(agent_id, run_id))",
    ] {
        sqlx::query(sql).execute(pool).await.map_err(persistence)?;
    }
    Ok(())
}

fn status_text(status: TaskRunStatus) -> Result<String> {
    serde_json::to_value(status)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| TaskRunStorageError::InvalidCheckpoint.into())
}

fn decode(row: sqlx::sqlite::SqliteRow) -> Result<TaskRunSnapshot> {
    let json: String = row.try_get("snapshot_json").map_err(persistence)?;
    if json.len() > MAX_TASK_CHECKPOINT_BYTES {
        return Err(TaskRunStorageError::CheckpointTooLarge.into());
    }
    let snapshot: TaskRunSnapshot =
        serde_json::from_str(&json).map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
    snapshot.validate()?;
    if snapshot.key.agent_id != row.try_get::<String, _>("agent_id").map_err(persistence)?
        || snapshot.key.run_id != row.try_get::<String, _>("run_id").map_err(persistence)?
        || snapshot.actor_id
            != row
                .try_get::<Option<String>, _>("actor_id")
                .map_err(persistence)?
        || snapshot.revision as i64 != row.try_get::<i64, _>("revision").map_err(persistence)?
        || snapshot.schema_version as i64
            != row
                .try_get::<i64, _>("schema_version")
                .map_err(persistence)?
        || status_text(snapshot.status)?
            != row.try_get::<String, _>("status").map_err(persistence)?
        || snapshot.owner_token
            != row
                .try_get::<Option<String>, _>("owner_token")
                .map_err(persistence)?
        || snapshot.cancel_requested
            != row
                .try_get::<bool, _>("cancel_requested")
                .map_err(persistence)?
        || snapshot.created_at.to_rfc3339()
            != row
                .try_get::<String, _>("created_at")
                .map_err(persistence)?
        || snapshot.updated_at.to_rfc3339()
            != row
                .try_get::<String, _>("updated_at")
                .map_err(persistence)?
    {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    Ok(snapshot)
}

impl SqliteStorage {
    pub(super) async fn task_create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
        snapshot.validate_create()?;
        let mut transaction = self
            .pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(persistence)?;
        let exists: i64 = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM task_runs WHERE agent_id = ? AND run_id = ?) OR EXISTS(SELECT 1 FROM task_run_tombstones WHERE agent_id = ? AND run_id = ?)")
            .bind(&snapshot.key.agent_id).bind(&snapshot.key.run_id).bind(&snapshot.key.agent_id).bind(&snapshot.key.run_id)
            .fetch_one(&mut *transaction).await.map_err(persistence)?;
        if exists != 0 {
            return Err(TaskRunStorageError::AlreadyExists.into());
        }
        sqlx::query("INSERT INTO task_runs(agent_id, run_id, actor_id, schema_version, revision, status, owner_token, cancel_requested, snapshot_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&snapshot.key.agent_id).bind(&snapshot.key.run_id).bind(&snapshot.actor_id)
            .bind(snapshot.schema_version as i64).bind(snapshot.revision as i64).bind(status_text(snapshot.status)?)
            .bind(&snapshot.owner_token).bind(snapshot.cancel_requested).bind(serde_json::to_string(snapshot)?)
            .bind(snapshot.created_at.to_rfc3339()).bind(snapshot.updated_at.to_rfc3339())
            .execute(&mut *transaction).await.map_err(persistence)?;
        transaction.commit().await.map_err(persistence)
    }

    pub(super) async fn task_load(&self, key: &TaskRunKey) -> Result<Option<TaskRunSnapshot>> {
        let row = sqlx::query("SELECT * FROM task_runs WHERE agent_id = ? AND run_id = ?")
            .bind(&key.agent_id)
            .bind(&key.run_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(persistence)?;
        row.map(decode).transpose()
    }

    pub(super) async fn task_list(&self, filter: &TaskRunFilter) -> Result<Vec<TaskRunSummary>> {
        let status = filter.status.map(status_text).transpose()?;
        let rows = sqlx::query("SELECT agent_id, run_id, actor_id, revision, status, owner_token IS NOT NULL AS owned, cancel_requested, created_at, updated_at FROM task_runs WHERE agent_id = ? AND (? IS NULL OR actor_id = ?) AND (? IS NULL OR status = ?) ORDER BY run_id LIMIT ?")
            .bind(&filter.agent_id).bind(&filter.actor_id).bind(&filter.actor_id).bind(&status).bind(&status)
            .bind(filter.limit.map_or(i64::MAX, |v| v.min(i64::MAX as usize) as i64))
            .fetch_all(&self.pool).await.map_err(persistence)?;
        rows.into_iter()
            .map(|row| {
                let status: String = row.try_get("status").map_err(persistence)?;
                let status = serde_json::from_value(serde_json::Value::String(status))
                    .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?;
                let parse_time = |field| -> Result<chrono::DateTime<chrono::Utc>> {
                    let value: String = row.try_get(field).map_err(persistence)?;
                    value
                        .parse()
                        .map_err(|_| TaskRunStorageError::InvalidCheckpoint.into())
                };
                Ok(TaskRunSummary {
                    key: TaskRunKey {
                        agent_id: row.try_get("agent_id").map_err(persistence)?,
                        run_id: row.try_get("run_id").map_err(persistence)?,
                    },
                    actor_id: row.try_get("actor_id").map_err(persistence)?,
                    revision: u64::try_from(
                        row.try_get::<i64, _>("revision").map_err(persistence)?,
                    )
                    .map_err(|_| TaskRunStorageError::InvalidCheckpoint)?,
                    status,
                    owned: row.try_get("owned").map_err(persistence)?,
                    cancel_requested: row.try_get("cancel_requested").map_err(persistence)?,
                    created_at: parse_time("created_at")?,
                    updated_at: parse_time("updated_at")?,
                })
            })
            .collect()
    }

    pub(super) async fn task_mutate(
        &self,
        key: &TaskRunKey,
        mutation: &TaskRunMutation,
    ) -> Result<TaskRunSnapshot> {
        let mut transaction = self
            .pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(persistence)?;
        let row = sqlx::query("SELECT * FROM task_runs WHERE agent_id = ? AND run_id = ?")
            .bind(&key.agent_id)
            .bind(&key.run_id)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(persistence)?
            .ok_or(TaskRunStorageError::NotFound)?;
        let previous = decode(row)?;
        let next = previous.transition(mutation, chrono::Utc::now())?;
        let affected = sqlx::query("UPDATE task_runs SET revision = ?, status = ?, owner_token = ?, cancel_requested = ?, snapshot_json = ?, updated_at = ? WHERE agent_id = ? AND run_id = ? AND revision = ?")
            .bind(next.revision as i64).bind(status_text(next.status)?).bind(&next.owner_token).bind(next.cancel_requested)
            .bind(serde_json::to_string(&next)?).bind(next.updated_at.to_rfc3339())
            .bind(&key.agent_id).bind(&key.run_id).bind(previous.revision as i64)
            .execute(&mut *transaction).await.map_err(persistence)?.rows_affected();
        if affected != 1 {
            return Err(TaskRunStorageError::Conflict.into());
        }
        mutation.validate_final_deadline(chrono::Utc::now())?;
        transaction.commit().await.map_err(persistence)?;
        Ok(next)
    }

    pub(super) async fn task_delete(&self, key: &TaskRunKey, expected_revision: u64) -> Result<()> {
        let mut transaction = self
            .pool
            .begin_with("BEGIN IMMEDIATE")
            .await
            .map_err(persistence)?;
        let row = sqlx::query("SELECT * FROM task_runs WHERE agent_id = ? AND run_id = ?")
            .bind(&key.agent_id)
            .bind(&key.run_id)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(persistence)?
            .ok_or(TaskRunStorageError::NotFound)?;
        let previous = decode(row)?;
        if previous.revision != expected_revision {
            return Err(TaskRunStorageError::Conflict.into());
        }
        if previous.owner_token.is_some() {
            return Err(TaskRunStorageError::Owned.into());
        }
        tombstone(&mut transaction, key).await?;
        transaction.commit().await.map_err(persistence)
    }
}

async fn tombstone(transaction: &mut Transaction<'_, Sqlite>, key: &TaskRunKey) -> Result<()> {
    sqlx::query("INSERT OR IGNORE INTO task_run_tombstones(agent_id, run_id) VALUES (?, ?)")
        .bind(&key.agent_id)
        .bind(&key.run_id)
        .execute(&mut **transaction)
        .await
        .map_err(persistence)?;
    sqlx::query("DELETE FROM task_runs WHERE agent_id = ? AND run_id = ?")
        .bind(&key.agent_id)
        .bind(&key.run_id)
        .execute(&mut **transaction)
        .await
        .map_err(persistence)?;
    Ok(())
}

/// Rejects active writers before removing any actor data; tombstones prevent later task resurrection.
pub(super) async fn delete_actor_tasks(
    transaction: &mut Transaction<'_, Sqlite>,
    agent_id: &str,
    actor_id: &str,
) -> Result<()> {
    let owned: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM task_runs WHERE agent_id = ? AND actor_id = ? AND owner_token IS NOT NULL")
        .bind(agent_id).bind(actor_id).fetch_one(&mut **transaction).await.map_err(persistence)?;
    if owned != 0 {
        return Err(TaskRunStorageError::Owned.into());
    }
    sqlx::query("INSERT OR IGNORE INTO task_run_tombstones(agent_id, run_id) SELECT agent_id, run_id FROM task_runs WHERE agent_id = ? AND actor_id = ?")
        .bind(agent_id).bind(actor_id).execute(&mut **transaction).await.map_err(persistence)?;
    sqlx::query("DELETE FROM task_runs WHERE agent_id = ? AND actor_id = ?")
        .bind(agent_id)
        .bind(actor_id)
        .execute(&mut **transaction)
        .await
        .map_err(persistence)?;
    Ok(())
}
