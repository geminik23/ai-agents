use super::SqliteStorage;
use sqlx::{TransactionManager, sqlite::SqliteTransactionManager};
use std::{borrow::Cow, time::Duration};

// Simulates the acknowledged custom-BEGIN cancellation window before SQLx constructs its rollback-owning Transaction.
#[tokio::test]
async fn cancelled_sqlite_initialization_rolls_back_before_connection_reuse() {
    for depth in [1, 2] {
        let storage = SqliteStorage::in_memory().await.unwrap();
        // Keep every other connection checked out so the exact released connection must be reused.
        let mut held = vec![];
        for _ in 1..storage.pool.options().get_max_connections() {
            held.push(storage.pool.acquire().await.unwrap());
        }
        let mut connection = storage.pool.acquire().await.unwrap();
        sqlx::query("CREATE TABLE cancellation_probe (value INTEGER)")
            .execute(&mut *connection)
            .await
            .unwrap();
        SqliteTransactionManager::begin(&mut connection, Some(Cow::Borrowed("BEGIN IMMEDIATE")))
            .await
            .unwrap();
        for _ in 1..depth {
            SqliteTransactionManager::begin(&mut connection, None)
                .await
                .unwrap();
        }
        sqlx::query("INSERT INTO cancellation_probe VALUES (1)")
            .execute(&mut *connection)
            .await
            .unwrap();
        assert_eq!(
            SqliteTransactionManager::get_transaction_depth(&connection),
            depth
        );
        drop(connection);
        let mut reused = tokio::time::timeout(Duration::from_secs(2), storage.pool.acquire())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SqliteTransactionManager::get_transaction_depth(&reused),
            0,
            "an acknowledged BEGIN without a Transaction guard leaked into the pool"
        );
        let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM cancellation_probe")
            .fetch_one(&mut *reused)
            .await
            .unwrap();
        assert_eq!(rows, 0, "backend cleanup must roll back uncommitted data");
        let transaction: sqlx::Transaction<'_, sqlx::Sqlite> =
            sqlx::Transaction::begin(&mut *reused, Some(Cow::Borrowed("BEGIN IMMEDIATE")))
                .await
                .unwrap();
        transaction.rollback().await.unwrap();
        drop(reused);
        drop(held);
        storage.close().await;
    }
}
