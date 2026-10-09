use super::*;
use std::sync::Arc;
use std::time::Duration;

pub(super) async fn service_test<F, Fut>(test: F)
where
    F: FnOnce(Arc<RedisStorage>) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let url = std::env::var("REDIS_URL").expect("service tests require explicit REDIS_URL");
    let storage = Arc::new(
        RedisStorage::new(&url)
            .unwrap()
            .with_prefix(format!("ai-agents-service-test:{}:", uuid::Uuid::new_v4())),
    );
    let mut task = tokio::spawn(test(storage.clone()));
    let result = tokio::time::timeout(Duration::from_secs(20), &mut task).await;
    if result.is_err() {
        task.abort();
        let _ = task.await;
    }
    // Only this test's UUID prefix is removed, including when an assertion panics.
    let cleanup = tokio::time::timeout(Duration::from_secs(5), async {
        let mut connection = storage.get_connection().await.unwrap();
        let mut cursor = 0u64;
        loop {
            let (next, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(format!("{}*", storage.prefix))
                .arg("COUNT")
                .arg(100)
                .query_async(&mut connection)
                .await
                .unwrap();
            if !keys.is_empty() {
                redis::cmd("DEL")
                    .arg(keys)
                    .query_async::<()>(&mut connection)
                    .await
                    .unwrap();
            }
            cursor = next;
            if cursor == 0 {
                break;
            }
        }
    })
    .await;
    cleanup.expect("prefix cleanup timed out");
    result
        .expect("service test timed out")
        .expect("service test failed");
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn cleanup_rejects_replaced_or_deleted_observations() {
    service_test(|storage| async move {
        storage
            .save("session", &AgentSnapshot::new("old-owner".into()))
            .await
            .unwrap();
        let (snapshot, metadata) = storage.load_raw_session("session").await.unwrap();
        let snapshot = snapshot.unwrap();
        let metadata = metadata.unwrap();
        storage
            .save("session", &AgentSnapshot::new("new-owner".into()))
            .await
            .unwrap();
        assert!(
            !storage
                .delete_if_unchanged("session", &snapshot, &metadata)
                .await
                .unwrap()
        );
        assert_eq!(
            storage.load("session").await.unwrap().unwrap().agent_id,
            "new-owner"
        );
        let (snapshot, metadata) = storage.load_raw_session("session").await.unwrap();
        storage.delete("session").await.unwrap();
        assert!(
            !storage
                .delete_if_unchanged("session", &snapshot.unwrap(), &metadata.unwrap())
                .await
                .unwrap()
        );
    })
    .await;
}

fn assert_persistence<T>(result: Result<T>) {
    assert!(matches!(result, Err(AgentError::Persistence(_))));
}

fn with_ttl(storage: &RedisStorage, ttl: u64) -> RedisStorage {
    RedisStorage {
        client: storage.client.clone(),
        prefix: storage.prefix.clone(),
        default_ttl: Some(ttl),
    }
}

async fn raw_pair(storage: &RedisStorage, id: &str) -> (String, String) {
    let (snapshot, metadata) = storage.load_raw_session(id).await.unwrap();
    (snapshot.unwrap(), metadata.unwrap())
}

async fn ttl_pair(storage: &RedisStorage, id: &str) -> (i64, i64) {
    let mut connection = storage.get_connection().await.unwrap();
    redis::pipe()
        .cmd("PTTL")
        .arg(storage.session_key(id))
        .cmd("PTTL")
        .arg(storage.meta_key(id))
        .query_async(&mut connection)
        .await
        .unwrap()
}

async fn score(storage: &RedisStorage, key: &str, id: &str) -> Option<f64> {
    let mut connection = storage.get_connection().await.unwrap();
    redis::cmd("ZSCORE")
        .arg(key)
        .arg(id)
        .query_async(&mut connection)
        .await
        .unwrap()
}

async fn set_metadata(storage: &RedisStorage, id: &str, updated_at: &str) {
    let mut metadata = storage.get_meta(id).await.unwrap().unwrap();
    metadata.updated_at = updated_at.into();
    let mut connection = storage.get_connection().await.unwrap();
    redis::cmd("SET")
        .arg(storage.meta_key(id))
        .arg(serde_json::to_string(&metadata).unwrap())
        .query_async::<()>(&mut connection)
        .await
        .unwrap();
}

#[test]
fn invalid_url_and_capability_inventory() {
    assert_persistence(RedisStorage::new("not-a-redis-url"));
    let storage = RedisStorage::new("redis://127.0.0.1:1/").unwrap();
    assert!(storage.supports(StorageCapability::Snapshot));
    for capability in [
        StorageCapability::SessionMetadata,
        StorageCapability::SessionFiltering,
        StorageCapability::ExpiryCleanup,
        StorageCapability::ActorFacts,
        StorageCapability::ActorRelationships,
        StorageCapability::ActorDataDeletion,
    ] {
        assert!(!storage.supports(capability));
    }
}

#[tokio::test]
async fn unsupported_operations_do_not_contact_redis() {
    let storage = RedisStorage::new("redis://127.0.0.1:1/").unwrap();
    assert!(matches!(
        storage.load_metadata("session").await,
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::SessionMetadata
        ))
    ));
    assert!(matches!(
        storage.cleanup_expired().await,
        Err(AgentError::UnsupportedStorageCapability(
            StorageCapability::ExpiryCleanup
        ))
    ));
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn round_trip_preserves_payload_and_metadata() {
    service_test(|storage| async move {
        assert!(storage.load("session").await.unwrap().is_none());
        storage.delete("session").await.unwrap();
        let mut snapshot = AgentSnapshot::new("agent-a".into());
        snapshot
            .memory
            .messages
            .push(ai_agents_core::ChatMessage::user("hello"));
        snapshot.context.insert(
            "nested".into(),
            serde_json::json!({"languages": ["한국어", "English"]}),
        );
        snapshot.persona = Some(serde_json::json!({"name": "tester"}));
        snapshot.relationships = Some(serde_json::json!({"actor": {"trust": 0.5}}));
        storage.save("session", &snapshot).await.unwrap();
        assert_eq!(
            serde_json::to_value(storage.load("session").await.unwrap().unwrap()).unwrap(),
            serde_json::to_value(&snapshot).unwrap()
        );
        let created_at = storage
            .get_meta("session")
            .await
            .unwrap()
            .unwrap()
            .created_at;
        snapshot.agent_id = "agent-b".into();
        storage.save("session", &snapshot).await.unwrap();
        let metadata = storage.get_meta("session").await.unwrap().unwrap();
        assert_eq!(metadata.created_at, created_at);
        assert_eq!(metadata.agent_id, "agent-b");
        assert_eq!(metadata.message_count, 1);
        assert!(
            score(&storage, &storage.agent_index_key("agent-a"), "session")
                .await
                .is_none()
        );
        assert!(
            score(&storage, &storage.agent_index_key("agent-b"), "session")
                .await
                .is_some()
        );
        storage.delete("session").await.unwrap();
        storage.delete("session").await.unwrap();
        assert_eq!(ttl_pair(&storage, "session").await, (-2, -2));
        assert!(
            score(&storage, &storage.sessions_index_key(), "session")
                .await
                .is_none()
        );
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn ttl_refresh_removal_zero_and_missing_semantics() {
    service_test(|storage| async move {
        let snapshot = AgentSnapshot::new("agent".into());
        let expiring = with_ttl(&storage, 60);
        expiring.save("session", &snapshot).await.unwrap();
        let (payload_ttl, metadata_ttl) = ttl_pair(&storage, "session").await;
        assert!((1..=60_000).contains(&payload_ttl));
        assert!((payload_ttl - metadata_ttl).abs() <= 1);
        storage.set_ttl("session", 1).await.unwrap();
        expiring.save("session", &snapshot).await.unwrap();
        assert!(ttl_pair(&storage, "session").await.0 > 1000);
        storage.save("session", &snapshot).await.unwrap();
        assert_eq!(ttl_pair(&storage, "session").await, (-1, -1));
        storage.set_ttl("session", 0).await.unwrap();
        assert_eq!(ttl_pair(&storage, "session").await, (-2, -2));
        assert!(
            score(&storage, &storage.agent_index_key("agent"), "session")
                .await
                .is_none()
        );
        storage.set_ttl("missing", u64::MAX).await.unwrap();
        assert!(!storage.exists("missing").await.unwrap());
        assert!(
            score(&storage, &storage.sessions_index_key(), "missing")
                .await
                .is_none()
        );
        expiring.save("expiring", &snapshot).await.unwrap();
        storage.set_ttl("expiring", 1).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while storage.exists("expiring").await.unwrap() {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap();
        assert!(storage.load("expiring").await.unwrap().is_none());
        assert!(storage.get_meta("expiring").await.unwrap().is_none());
        assert!(storage.list_sessions().await.unwrap().is_empty());
        assert!(
            storage
                .list_sessions_by_agent("agent")
                .await
                .unwrap()
                .is_empty()
        );
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn invalid_ttl_and_index_types_preserve_existing_data() {
    service_test(|storage| async move {
        let original = AgentSnapshot::new("agent-a".into());
        storage.save("session", &original).await.unwrap();
        let pair = raw_pair(&storage, "session").await;
        let global_score = score(&storage, &storage.sessions_index_key(), "session").await;
        for ttl in [0, u64::MAX] {
            assert_persistence(
                with_ttl(&storage, ttl)
                    .save("session", &AgentSnapshot::new("agent-b".into()))
                    .await,
            );
            assert_eq!(raw_pair(&storage, "session").await, pair);
            assert_eq!(ttl_pair(&storage, "session").await, (-1, -1));
            assert_eq!(
                score(&storage, &storage.sessions_index_key(), "session").await,
                global_score
            );
        }
        assert_persistence(storage.set_ttl("session", u64::MAX).await);
        for key in [
            storage.sessions_index_key(),
            storage.agent_index_key("agent-a"),
            storage.agent_index_key("agent-b"),
        ] {
            let mut connection = storage.get_connection().await.unwrap();
            let original_dump: Option<Vec<u8>> = redis::cmd("DUMP")
                .arg(&key)
                .query_async(&mut connection)
                .await
                .unwrap();
            redis::cmd("SET")
                .arg(&key)
                .arg("wrong-type")
                .query_async::<()>(&mut connection)
                .await
                .unwrap();
            assert_persistence(
                storage
                    .save("session", &AgentSnapshot::new("agent-b".into()))
                    .await,
            );
            assert_eq!(raw_pair(&storage, "session").await, pair);
            assert_eq!(ttl_pair(&storage, "session").await, (-1, -1));
            if key != storage.agent_index_key("agent-b") {
                assert_persistence(storage.set_ttl("session", 30).await);
                assert_persistence(storage.delete("session").await);
                assert_eq!(raw_pair(&storage, "session").await, pair);
            }
            if key == storage.sessions_index_key() {
                assert_persistence(storage.list_sessions().await);
            } else {
                let agent = if key == storage.agent_index_key("agent-a") {
                    "agent-a"
                } else {
                    "agent-b"
                };
                assert_persistence(storage.list_sessions_by_agent(agent).await);
            }
            let preserved: String = redis::cmd("GET")
                .arg(&key)
                .query_async(&mut connection)
                .await
                .unwrap();
            assert_eq!(preserved, "wrong-type");
            redis::cmd("DEL")
                .arg(&key)
                .query_async::<()>(&mut connection)
                .await
                .unwrap();
            if let Some(dump) = original_dump {
                redis::cmd("RESTORE")
                    .arg(&key)
                    .arg(0)
                    .arg(dump)
                    .query_async::<()>(&mut connection)
                    .await
                    .unwrap();
            }
        }
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn corrupt_values_fail_without_replacing_snapshots() {
    service_test(|storage| async move {
        storage
            .save("session", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();
        let pair = raw_pair(&storage, "session").await;
        let mut connection = storage.get_connection().await.unwrap();
        redis::cmd("SET")
            .arg(storage.session_key("session"))
            .arg("{invalid")
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert_persistence(storage.load("session").await);
        redis::cmd("SET")
            .arg(storage.session_key("session"))
            .arg(&pair.0)
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        for corrupt in ["{invalid", "1", "{}"] {
            redis::cmd("SET")
                .arg(storage.meta_key("session"))
                .arg(corrupt)
                .query_async::<()>(&mut connection)
                .await
                .unwrap();
            assert_persistence(storage.get_meta("session").await);
            assert_persistence(
                storage
                    .save("session", &AgentSnapshot::new("other".into()))
                    .await,
            );
            assert_persistence(storage.set_ttl("session", 30).await);
            assert_persistence(storage.delete("session").await);
            assert_eq!(
                raw_pair(&storage, "session").await,
                (pair.0.clone(), corrupt.into())
            );
        }
        redis::cmd("DEL")
            .arg(storage.meta_key("session"))
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert_persistence(storage.set_ttl("session", 30).await);
        assert_eq!(ttl_pair(&storage, "session").await, (-1, -2));
        redis::cmd("HSET")
            .arg(storage.meta_key("session"))
            .arg("field")
            .arg("value")
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert_persistence(storage.get_meta("session").await);
        assert_persistence(
            storage
                .save("session", &AgentSnapshot::new("other".into()))
                .await,
        );
        assert_persistence(storage.set_ttl("session", 30).await);
        assert_persistence(storage.delete("session").await);
        let value: String = redis::cmd("GET")
            .arg(storage.session_key("session"))
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(value, pair.0);
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn mixed_mutations_leave_one_consistent_owner() {
    service_test(|storage| async move {
        let barrier = Arc::new(tokio::sync::Barrier::new(13));
        let mut tasks = tokio::task::JoinSet::new();
        for index in 0..12 {
            let storage = storage.clone();
            let barrier = barrier.clone();
            tasks.spawn(async move {
                barrier.wait().await;
                match index % 3 {
                    0 => storage
                        .save("shared", &AgentSnapshot::new(format!("agent-{index}")))
                        .await
                        .unwrap(),
                    1 => storage.delete("shared").await.unwrap(),
                    _ => storage.set_ttl("shared", 60).await.unwrap(),
                }
            });
        }
        barrier.wait().await;
        while let Some(task) = tasks.join_next().await {
            task.unwrap();
        }
        let (snapshot, metadata) = storage.load_raw_session("shared").await.unwrap();
        let global = score(&storage, &storage.sessions_index_key(), "shared").await;
        let owner = match (snapshot, metadata) {
            (Some(snapshot), Some(metadata)) => {
                let snapshot: AgentSnapshot = serde_json::from_str(&snapshot).unwrap();
                let metadata: RedisSessionMeta = serde_json::from_str(&metadata).unwrap();
                assert_eq!(snapshot.agent_id, metadata.agent_id);
                assert!(global.is_some());
                Some(snapshot.agent_id)
            }
            (None, None) => {
                assert!(global.is_none());
                None
            }
            _ => panic!("partial snapshot/metadata mutation"),
        };
        for index in (0..12).step_by(3) {
            let agent = format!("agent-{index}");
            let entry = score(&storage, &storage.agent_index_key(&agent), "shared").await;
            assert_eq!(entry.is_some(), owner.as_ref() == Some(&agent));
            if entry.is_some() {
                assert_eq!(entry, global);
            }
        }
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn cleanup_preserves_timestamp_semantics_and_counts_deletions() {
    service_test(|storage| async move {
        let cutoff = DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        for (id, time) in [
            ("old", "2000-01-01T08:59:59+09:00"),
            ("equal", "2000-01-01T09:00:00.000+09:00"),
            ("new", "1999-12-31T19:00:01-05:00"),
            ("invalid", "not-a-date"),
        ] {
            storage
                .save(id, &AgentSnapshot::new("agent".into()))
                .await
                .unwrap();
            set_metadata(&storage, id, time).await;
        }
        assert_eq!(storage.expire_sessions(cutoff).await.unwrap(), 1);
        assert!(storage.load("old").await.unwrap().is_none());
        for id in ["equal", "new", "invalid"] {
            assert!(storage.exists(id).await.unwrap());
        }
        assert_eq!(storage.expire_sessions(cutoff).await.unwrap(), 0);
        let pair = raw_pair(&storage, "equal").await;
        assert!(
            storage
                .delete_if_unchanged("equal", &pair.0, &pair.1)
                .await
                .unwrap()
        );
        assert!(
            !storage
                .delete_if_unchanged("equal", &pair.0, &pair.1)
                .await
                .unwrap()
        );
        assert!(
            score(&storage, &storage.sessions_index_key(), "equal")
                .await
                .is_none()
        );
        assert!(
            score(&storage, &storage.agent_index_key("agent"), "equal")
                .await
                .is_none()
        );
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn cleanup_checks_payload_even_when_metadata_is_identical() {
    service_test(|storage| async move {
        storage
            .save("session", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();
        let pair = raw_pair(&storage, "session").await;
        let mut changed: AgentSnapshot = serde_json::from_str(&pair.0).unwrap();
        changed.context.insert("new-write".into(), true.into());
        let mut connection = storage.get_connection().await.unwrap();
        redis::cmd("SET")
            .arg(storage.session_key("session"))
            .arg(serde_json::to_string(&changed).unwrap())
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert!(
            !storage
                .delete_if_unchanged("session", &pair.0, &pair.1)
                .await
                .unwrap()
        );
        assert!(storage.exists("session").await.unwrap());
    })
    .await;
}

#[tokio::test]
async fn refused_and_closed_connections_return_errors() {
    let reserved = tokio::net::TcpSocket::new_v4().unwrap();
    reserved.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let storage =
        RedisStorage::new(&format!("redis://{}/", reserved.local_addr().unwrap())).unwrap();
    assert_persistence(
        tokio::time::timeout(Duration::from_secs(3), storage.load("missing"))
            .await
            .expect("refused connection must return an error"),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let storage =
        RedisStorage::new(&format!("redis://{}/", listener.local_addr().unwrap())).unwrap();
    let mut servers = tokio::task::JoinSet::new();
    servers.spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        drop(stream);
    });
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        storage.save("missing", &AgentSnapshot::new("agent".into())),
    )
    .await;
    servers.abort_all();
    while servers.join_next().await.is_some() {}
    assert_persistence(result.expect("closed connection must return an error"));
}

#[tokio::test]
async fn unresponsive_peer_is_bounded_by_caller_not_backend() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let storage =
        RedisStorage::new(&format!("redis://{}/", listener.local_addr().unwrap())).unwrap();
    let (accepted, signal) = tokio::sync::oneshot::channel();
    let mut servers = tokio::task::JoinSet::new();
    servers.spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        accepted.send(()).unwrap();
        std::future::pending::<()>().await;
    });
    let mut request = Box::pin(storage.load("missing"));
    let mut signal = Box::pin(signal);
    tokio::time::timeout(Duration::from_secs(3), async {
        tokio::select! { _ = &mut request => panic!("peer must not reply"), result = &mut signal => result.unwrap() }
    }).await.unwrap();
    let result = tokio::time::timeout(Duration::from_millis(100), &mut request).await;
    drop(request);
    servers.abort_all();
    while servers.join_next().await.is_some() {}
    assert!(
        result.is_err(),
        "the harness deadline is not a backend error"
    );
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn same_storage_recovers_on_a_later_explicit_call() {
    service_test(|storage| async move {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy = RedisStorage::new(&format!("redis://{}/", listener.local_addr().unwrap()))
            .unwrap()
            .with_prefix(&storage.prefix);
        let url = std::env::var("REDIS_URL").unwrap();
        let info = redis::Client::open(url)
            .unwrap()
            .get_connection_info()
            .addr
            .clone();
        let redis::ConnectionAddr::Tcp(host, port) = info else {
            panic!("proxy test requires a TCP Redis URL")
        };
        let mut servers = tokio::task::JoinSet::new();
        servers.spawn(async move {
            let (first, _) = listener.accept().await.unwrap();
            drop(first);
            let mut transfers = tokio::task::JoinSet::new();
            loop {
                let (mut downstream, _) = listener.accept().await.unwrap();
                let host = host.clone();
                transfers.spawn(async move {
                    let mut upstream = tokio::net::TcpStream::connect((host.as_str(), port))
                        .await
                        .unwrap();
                    let _ = tokio::io::copy_bidirectional(&mut downstream, &mut upstream).await;
                });
            }
        });
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            assert_persistence(proxy.load("session").await);
            proxy
                .save("session", &AgentSnapshot::new("recovered".into()))
                .await
                .unwrap();
            assert_eq!(
                proxy.load("session").await.unwrap().unwrap().agent_id,
                "recovered"
            );
        })
        .await;
        servers.abort_all();
        while servers.join_next().await.is_some() {}
        result.unwrap();
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn prefixes_with_identical_ids_remain_isolated() {
    service_test(|storage| async move {
        let sibling = RedisStorage::new(&std::env::var("REDIS_URL").unwrap())
            .unwrap()
            .with_prefix(format!("{}sibling:", storage.prefix));
        storage
            .save("same", &AgentSnapshot::new("parent".into()))
            .await
            .unwrap();
        sibling
            .save("same", &AgentSnapshot::new("sibling".into()))
            .await
            .unwrap();
        assert_eq!(
            storage.load("same").await.unwrap().unwrap().agent_id,
            "parent"
        );
        assert_eq!(
            sibling.load("same").await.unwrap().unwrap().agent_id,
            "sibling"
        );
        storage.delete("same").await.unwrap();
        assert!(sibling.exists("same").await.unwrap());
        assert_eq!(sibling.list_sessions().await.unwrap(), vec!["same"]);
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn cleanup_missing_or_corrupt_metadata_is_safe() {
    service_test(|storage| async move {
        storage
            .save("missing-meta", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();
        let mut connection = storage.get_connection().await.unwrap();
        redis::cmd("DEL")
            .arg(storage.meta_key("missing-meta"))
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        let cutoff = Utc::now() + chrono::Duration::days(1);
        assert_eq!(storage.expire_sessions(cutoff).await.unwrap(), 0);
        assert!(storage.exists("missing-meta").await.unwrap());
        storage
            .save("corrupt-meta", &AgentSnapshot::new("agent".into()))
            .await
            .unwrap();
        let original: String = redis::cmd("GET")
            .arg(storage.session_key("corrupt-meta"))
            .query_async(&mut connection)
            .await
            .unwrap();
        redis::cmd("SET")
            .arg(storage.meta_key("corrupt-meta"))
            .arg("{invalid")
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert_persistence(storage.expire_sessions(cutoff).await);
        let remaining: String = redis::cmd("GET")
            .arg(storage.session_key("corrupt-meta"))
            .query_async(&mut connection)
            .await
            .unwrap();
        assert_eq!(remaining, original);
        redis::cmd("DEL")
            .arg(storage.meta_key("corrupt-meta"))
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        redis::cmd("HSET")
            .arg(storage.meta_key("corrupt-meta"))
            .arg("field")
            .arg("value")
            .query_async::<()>(&mut connection)
            .await
            .unwrap();
        assert_persistence(storage.expire_sessions(cutoff).await);
        assert!(storage.exists("corrupt-meta").await.unwrap());
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn connection_drops_after_command_submission() {
    service_test(|storage| async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy = RedisStorage::new(&format!("redis://{}/", listener.local_addr().unwrap()))
            .unwrap()
            .with_prefix(&storage.prefix);
        let info = storage.client.get_connection_info().addr.clone();
        let redis::ConnectionAddr::Tcp(host, port) = info else {
            panic!("requires TCP Redis")
        };
        let (submitted, signal) = tokio::sync::oneshot::channel();
        let mut servers = tokio::task::JoinSet::new();
        servers.spawn(async move {
            let (mut client, _) = listener.accept().await.unwrap();
            let mut upstream = tokio::net::TcpStream::connect((host.as_str(), port))
                .await
                .unwrap();
            let mut command = Vec::new();
            loop {
                let mut from_client = [0u8; 4096];
                let mut from_server = [0u8; 4096];
                tokio::select! {
                    received = client.read(&mut from_client) => {
                        let received = received.unwrap();
                        if received == 0 { break; }
                        command.extend_from_slice(&from_client[..received]);
                        upstream.write_all(&from_client[..received]).await.unwrap();
                        if command.windows(7).any(|part| part == b"\r\nGET\r\n") {
                            submitted.send(()).unwrap();
                            return;
                        }
                    },
                    received = upstream.read(&mut from_server) => {
                        let received = received.unwrap();
                        if received == 0 { break; }
                        client.write_all(&from_server[..received]).await.unwrap();
                    }
                }
            }
            panic!("connection ended before the GET command");
        });
        let result = tokio::time::timeout(Duration::from_secs(5), proxy.load("missing")).await;
        servers.abort_all();
        while servers.join_next().await.is_some() {}
        assert!(
            signal.await.is_ok(),
            "fault must occur after command submission"
        );
        assert_persistence(result.expect("closed command must return an error"));
    })
    .await;
}

#[tokio::test]
#[ignore = "requires a Redis service"]
async fn concurrent_cleanup_counts_each_session_once() {
    service_test(|storage| async move {
        for index in 0..6 {
            let id = format!("session-{index}");
            storage
                .save(&id, &AgentSnapshot::new("agent".into()))
                .await
                .unwrap();
            set_metadata(&storage, &id, "2000-01-01T00:00:00Z").await;
        }
        let mut cleaners = tokio::task::JoinSet::new();
        for _ in 0..2 {
            let storage = storage.clone();
            cleaners.spawn(async move { storage.expire_sessions(Utc::now()).await.unwrap() });
        }
        let mut deleted = 0;
        while let Some(result) = cleaners.join_next().await {
            deleted += result.unwrap();
        }
        assert_eq!(deleted, 6);
        for index in 0..6 {
            let id = format!("session-{index}");
            assert_eq!(ttl_pair(&storage, &id).await, (-2, -2));
            assert!(
                score(&storage, &storage.sessions_index_key(), &id)
                    .await
                    .is_none()
            );
            assert!(
                score(&storage, &storage.agent_index_key("agent"), &id)
                    .await
                    .is_none()
            );
        }
    })
    .await;
}
