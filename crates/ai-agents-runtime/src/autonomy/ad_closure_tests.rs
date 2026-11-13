use super::*;
use crate::autonomy::{child_tests, scope_execution};

// A prior ordinary tail must settle before child admission, even when ordinary freshness explicitly permits overlap.
#[tokio::test]
async fn ad_child_handoff_drains_ordinary_tail_before_task_effects() {
    let main = Provider::new(&["ordinary", "done"], false);
    let hook = Arc::new(Tail::default());
    let mut runtime = relationship_agent(
        "background",
        main.clone(),
        Provider::new(&[RELATIONSHIP], false),
    );
    runtime
        .runtime_config
        .optimization
        .post_turn
        .relationships
        .await_before_next_turn = AwaitBeforeNextTurn::Never;
    let child = Arc::new(runtime.with_hooks(hook.clone()));
    child.set_actor_id("actor").unwrap();
    Box::pin(child.chat("ordinary objective")).await.unwrap();

    hook.entered.acquire().await.unwrap().forget();
    assert!(!child.background_maintenance.is_empty());
    let (parent, execution) = Box::pin(child_tests::execution(2)).await;

    let mut work = Box::pin(scope_execution(
        execution.clone(),
        child.chat("task objective"),
    ));
    assert!(
        futures::poll!(&mut work).is_pending(),
        "child task overlapped ordinary tail"
    );
    assert_eq!(main.calls.load(Ordering::SeqCst), 1);

    hook.release.add_permits(1);
    let response = work.await.unwrap();

    assert_eq!(response.content, "done");
    assert!(child.background_maintenance.is_empty());
    assert_eq!(main.calls.load(Ordering::SeqCst), 2);
    let payload: Box<crate::autonomy::TaskCheckpointPayload> =
        serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
    assert_eq!(
        payload.counters.llm_attempts, 1,
        "ordinary calls are not retroactively charged"
    );

    Box::pin(child_tests::finish(&parent, &execution)).await;
}

// Cancelling or dropping the incoming wait does not detach or reassign a prior ordinary tail.
#[tokio::test]
async fn ad_child_handoff_interruption_keeps_original_queue() {
    for cancel in [false, true] {
        let main = Provider::new(&["ordinary", "done"], false);
        let hook = Arc::new(Tail::default());
        let child = Arc::new(
            relationship_agent(
                "background",
                main.clone(),
                Provider::new(&[RELATIONSHIP], false),
            )
            .with_hooks(hook.clone()),
        );
        child.set_actor_id("actor").unwrap();
        Box::pin(child.chat("ordinary")).await.unwrap();
        hook.entered.acquire().await.unwrap().forget();
        let (_, execution) = Box::pin(child_tests::execution(2)).await;
        let mut work = Box::pin(scope_execution(execution.clone(), child.chat("task")));
        assert!(futures::poll!(&mut work).is_pending());
        if cancel {
            execution.request_cancel().await.unwrap();
            assert!(work.await.is_err());
        } else {
            drop(work);
        }
        assert_eq!(main.calls.load(Ordering::SeqCst), 1);
        assert!(child.autonomy_owner.read().is_none());
        assert!(!child.background_maintenance.is_empty());
        hook.release.add_permits(1);
        child.flush_background_tasks().await.unwrap();
        assert!(child.background_maintenance.is_empty());
    }
}

type ChildContexts = Arc<parking_lot::Mutex<Vec<(Option<String>, bool)>>>;

struct ChildProducer {
    fail: bool,
    provider: Arc<dyn LLMProvider>,
    child: parking_lot::Mutex<std::sync::Weak<RuntimeAgent>>,
    entered: Arc<tokio::sync::Semaphore>,
    release: Arc<tokio::sync::Semaphore>,
    jobs: parking_lot::Mutex<Vec<tokio::task::JoinHandle<Result<()>>>>,
    contexts: ChildContexts,
}

#[async_trait]
impl ai_agents_hooks::AgentHooks for ChildProducer {
    // The actual captured managed provider and caller-owned wrapper certify this fixture, not a matching alias.
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![self.provider.clone()])
    }

    // Enroll synchronously before spawning; the child result is not permission to detach its provider/tool tail.
    async fn on_response(&self, _: &AgentResponse) {
        let provider = self.provider.clone();
        let child = self.child.lock().upgrade().unwrap();
        let entered = self.entered.clone();
        let release = self.release.clone();
        let contexts = self.contexts.clone();
        let fail = self.fail;
        let job = crate::autonomy::owned_maintenance("child-callback", async move {
            contexts.lock().push((
                crate::autonomy::current_child_operation(),
                crate::autonomy::child_required(),
            ));
            entered.add_permits(1);
            release.acquire().await.unwrap().forget();
            if fail {
                return Err(AgentError::Other(
                    "acknowledged optional callback error".into(),
                ));
            }
            provider
                .complete(&[ChatMessage::user("auxiliary")], None)
                .await?;
            let record = child
                .invoke_tool(ToolExecutionRequest::new(
                    "tail",
                    "echo",
                    serde_json::json!({"message":"tail"}),
                    ToolCallSource::Task,
                ))
                .await?;
            assert!(record.executed && record.success);
            Ok(())
        });
        self.jobs.lock().push(tokio::spawn(job));
    }
}

// A real child callback uses its frozen managed handle and preserves operation attribution and optional/required policy across Tokio spawn.
#[tokio::test]
async fn ad_child_spawned_custom_tail_preserves_scope_and_blocks_completion() {
    for required in [false, true] {
        let raw = Provider::new(&["auxiliary"], false);
        let producer = Arc::new(ChildProducer {
            fail: false,
            provider: ai_agents_llm::managed_provider(raw.clone()),
            child: Default::default(),
            entered: Arc::new(tokio::sync::Semaphore::new(0)),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
            jobs: Default::default(),
            contexts: Default::default(),
        });
        let hooks = Arc::new(ai_agents_hooks::CompositeHooks::with_hooks(vec![
            producer.clone(),
        ]));
        let child = Arc::new(
            crate::AgentBuilder::from_yaml("name: Child\nsystem_prompt: test\ntools: [echo]\n")
                .unwrap()
                .llm(Provider::new(&["done"], false))
                .hooks(hooks)
                .auto_configure_features()
                .unwrap()
                .build()
                .unwrap(),
        );
        *producer.child.lock() = Arc::downgrade(&child);
        let (parent, execution) = Box::pin(child_tests::execution(2)).await;
        crate::autonomy::scope_child_requirement(
            required,
            scope_execution(execution.clone(), child.chat("child work")),
        )
        .await
        .unwrap();
        producer.entered.acquire().await.unwrap().forget();
        assert!(execution.owner.work.unsettled());
        assert!(child.chat("foreign").await.is_err());
        let mut wait = Box::pin(execution.await_owned_work());
        assert!(futures::poll!(&mut wait).is_pending());
        producer.release.add_permits(1);
        wait.await.unwrap();
        let jobs = std::mem::take(&mut *producer.jobs.lock());
        for job in jobs {
            job.await.unwrap().unwrap();
        }
        let payload: Box<crate::autonomy::TaskCheckpointPayload> =
            serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
        assert_eq!(payload.counters.llm_attempts, 2);
        assert_eq!(payload.counters.tool_attempts, 1);
        assert_eq!(raw.calls.load(Ordering::SeqCst), 1);
        let contexts = producer.contexts.lock().clone();
        assert_eq!(
            contexts,
            vec![(Some(payload.children[0].child_id.clone()), required)]
        );
        assert_eq!(
            payload.evidence.tool_calls[0].record.metadata["_task_child_operation"],
            serde_json::json!(payload.children[0].child_id)
        );
        Box::pin(child_tests::finish(&parent, &execution)).await;
    }
}

// Failure fixtures use the same production child hook and managed handles as the positive case.
fn failing_child(fail: bool) -> (Arc<RuntimeAgent>, Arc<ChildProducer>) {
    let producer = Arc::new(ChildProducer {
        fail,
        provider: ai_agents_llm::managed_provider(Provider::new(&["auxiliary"], false)),
        child: Default::default(),
        entered: Arc::new(tokio::sync::Semaphore::new(0)),
        release: Arc::new(tokio::sync::Semaphore::new(0)),
        jobs: Default::default(),
        contexts: Default::default(),
    });
    let child = Arc::new(
        crate::AgentBuilder::from_yaml("name: Child\nsystem_prompt: test\ntools: [echo]\n")
            .unwrap()
            .llm(Provider::new(&["done"], false))
            .hooks(producer.clone())
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap(),
    );
    *producer.child.lock() = Arc::downgrade(&child);
    (child, producer)
}

// Optional policy tolerates only known effect-free failure; required failure still prevents the parent's successful terminal CAS.
#[tokio::test]
async fn ad_child_background_known_failure_preserves_requiredness() {
    for required in [false, true] {
        let (child, producer) = failing_child(true);
        let (_, execution) = Box::pin(child_tests::execution(2)).await;
        crate::autonomy::scope_child_requirement(
            required,
            scope_execution(execution.clone(), child.chat("work")),
        )
        .await
        .unwrap();
        producer.entered.acquire().await.unwrap().forget();
        producer.release.add_permits(1);
        let jobs = std::mem::take(&mut *producer.jobs.lock());
        for job in jobs {
            assert!(job.await.unwrap().is_err());
        }
        assert!(!execution.owner.work.unsettled());
        assert_eq!(
            execution.stop_reason().as_deref(),
            if required {
                Some("maintenance_failed")
            } else {
                None
            }
        );
        let result = execution
            .terminal(TaskRunStatus::Completed, "candidate".into())
            .await
            .unwrap();
        assert_eq!(
            result.status,
            if required {
                TaskRunStatus::Failed
            } else {
                TaskRunStatus::Completed
            }
        );
        assert_eq!(producer.contexts.lock()[0].1, required);
    }
}

// A child result may already be durable when its tail acknowledgement is lost; both actual owners remain protected on either backend.
#[tokio::test]
async fn ad_child_background_ack_loss_and_drop_keep_parent_child_custody() {
    use crate::autonomy::continuation_test_support as support;
    for &sqlite in support::backends() {
        for mode in ["lost", "drop"] {
            let (child, producer) = failing_child(false);
            let (parent, original) = Box::pin(child_tests::execution(2)).await;
            let snapshot = original.load_owned().await.unwrap();
            let payload: Box<crate::autonomy::TaskCheckpointPayload> =
                serde_json::from_value(snapshot.payload.clone()).unwrap();
            let store = support::store(parent.info().id, "children-v1", sqlite).await;
            store.create(&snapshot).await.unwrap();
            let fault = support::ReceiptStore::checkpoint(
                store,
                mode,
                serde_json::json!({"maintenance":"child-callback", "phase":"settle"}),
            );
            fault.armed.store(true, Ordering::SeqCst);
            let execution = crate::autonomy::RunExecution::new(
                fault.clone(),
                &snapshot,
                crate::autonomy::LifecycleState::new(&payload.settings, false).unwrap(),
                original.owner.clone(),
                Instant::now(),
            )
            .unwrap();
            // Use the original immutable cycle scope, not an invented child or effect identity.
            execution.set_scope(&crate::autonomy::EvaluationScope {
                key: snapshot.key.clone(),
                objective_revision: 0,
                cycle: 1,
                stage: None,
                mutation_generation: 0,
                target_revisions: Default::default(),
                target_generations: Default::default(),
                validation_bindings: Default::default(),
                validation_attempts: Default::default(),
                validation_dependencies: Default::default(),
            });
            scope_execution(execution.clone(), child.chat("work"))
                .await
                .unwrap();
            producer.entered.acquire().await.unwrap().forget();
            producer.release.add_permits(1);
            tokio::time::timeout(std::time::Duration::from_secs(20), fault.reached.acquire())
                .await
                .unwrap()
                .unwrap()
                .forget();
            let jobs = std::mem::take(&mut *producer.jobs.lock());
            for job in jobs {
                if mode == "drop" {
                    job.abort();
                    assert!(job.await.unwrap_err().is_cancelled());
                } else {
                    assert!(job.await.unwrap().is_err());
                }
            }
            fault.assert_hit();
            assert!(execution.owner.work.unsettled());
            assert!(execution.await_owned_work().await.is_err());
            assert!(parent.chat("foreign").await.is_err());
            assert!(child.chat("foreign").await.is_err());
            let saved = fault.inner.load(&execution.run_id).await.unwrap().unwrap();
            let saved: Box<crate::autonomy::TaskCheckpointPayload> =
                serde_json::from_value(saved.payload).unwrap();
            assert!(saved.children.iter().all(|child| child.result.is_some()));
            assert_eq!(saved.counters.llm_attempts, 2);
            assert_eq!(saved.counters.tool_attempts, 1);
        }
    }
}

struct ReentrantTail {
    child: parking_lot::Mutex<std::sync::Weak<RuntimeAgent>>,
    first: AtomicBool,
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}
#[async_trait]
impl ai_agents_hooks::AgentHooks for ReentrantTail {
    // A caller-owned callback may use another ordinary turn after the earlier root gate has been released.
    fn task_providers(&self) -> Option<Vec<Arc<dyn LLMProvider>>> {
        Some(vec![])
    }
    async fn on_relationship_change(
        &self,
        _: &str,
        _: &[ai_agents_relationships::DimensionChange],
    ) {
        if self.first.swap(false, Ordering::SeqCst) {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
            let child = self.child.lock().upgrade().unwrap();
            Box::pin(child.chat("ordinary callback reentry"))
                .await
                .unwrap();
        }
    }
}

// Neither coordinating nor inherited admission may hold the root gate while an older ordinary callback needs that gate to finish.
#[tokio::test]
async fn ad_handoff_callback_reentry_does_not_deadlock() {
    for root in [false, true] {
        let main = Provider::new(&["ordinary", "callback", "done"], false);
        let hook = Arc::new(ReentrantTail {
            child: Default::default(),
            first: AtomicBool::new(true),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        });
        let mut runtime = relationship_agent(
            "background",
            main.clone(),
            Provider::new(&[RELATIONSHIP, RELATIONSHIP, RELATIONSHIP], false),
        );
        runtime
            .runtime_config
            .optimization
            .post_turn
            .relationships
            .await_before_next_turn = AwaitBeforeNextTurn::Never;
        let agent = Arc::new(runtime.with_hooks(hook.clone()));
        *hook.child.lock() = Arc::downgrade(&agent);
        agent.set_actor_id("actor").unwrap();
        Box::pin(agent.chat("ordinary")).await.unwrap();
        hook.entered.acquire().await.unwrap().forget();
        let (_, execution) = Box::pin(child_tests::execution(3)).await;
        let (runner, _) = runner(agent.clone(), 2);
        let work = async {
            if root {
                runner
                    .run("task", None)
                    .await
                    .unwrap()
                    .final_response
                    .unwrap()
                    .content
            } else {
                scope_execution(execution, agent.chat("task"))
                    .await
                    .unwrap()
                    .content
            }
        };
        let mut work = Box::pin(work);
        assert!(futures::poll!(&mut work).is_pending());
        hook.release.add_permits(1);
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(20), work)
                .await
                .unwrap(),
            "done"
        );
        assert_eq!(main.calls.load(Ordering::SeqCst), 3);
    }
}

// Actual actor post-turn failure cannot be swallowed as ordinary best-effort behavior or converted into successful task completion.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn ad_actor_overflow_runner_never_completes_after_required_work_loss() {
    for policy in [
        BackgroundOverflowPolicy::RunInline,
        BackgroundOverflowPolicy::Drop,
        BackgroundOverflowPolicy::Error,
    ] {
        let facts = Provider::new(
            &[
                r#"[{"category":"user_preference","content":"Concise replies","confidence":0.9,"actor_id":"actor"}]"#,
            ],
            true,
        );
        let yaml = "name: Overflow\nsystem_prompt: test\nruntime:\n  optimization:\n    enabled: true\n    post_turn:\n      facts:\n        mode: background\n      relationships:\n        mode: background\nmemory:\n  facts:\n    enabled: true\n    auto_extract: true\n    extractor_llm: facts\n  relationships:\n    enabled: true\n    auto_update:\n      enabled: true\n      llm: relationships\n    persistence:\n      enabled: true\ntools: []\n";
        let mut runtime = crate::AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Provider::new(&["done"], false))
            .llm_alias("facts", facts.clone())
            .llm_alias("relationships", Provider::new(&[RELATIONSHIP], false))
            .storage(Arc::new(
                ai_agents_storage::SqliteStorage::in_memory().await.unwrap(),
            ))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap();
        runtime.background_maintenance =
            Arc::new(crate::optimization::BackgroundMaintenanceQueue::new(1));
        runtime
            .runtime_config
            .optimization
            .post_turn
            .on_background_overflow = policy;
        let agent = Arc::new(runtime);
        agent.set_actor_id("actor").unwrap();
        let (runner, _) = runner(agent.clone(), 3);
        let mut work = Box::pin(runner.run("objective", None));
        if policy == BackgroundOverflowPolicy::RunInline {
            tokio::select! { result = &mut work => panic!("inline predecessor skipped: {:?}", result.err()), permit = facts.entered.acquire() => permit.unwrap().forget() }
            assert!(futures::poll!(&mut work).is_pending());
            facts.release.add_permits(1);
            assert_eq!(work.await.unwrap().run.status, TaskRunStatus::Completed);
        } else {
            let result = work.await.unwrap();
            assert_ne!(result.run.status, TaskRunStatus::Completed, "{policy:?}");
            facts.release.add_permits(1);
            let _ = agent.flush_background_tasks().await;
        }
    }
}

// One actual dispatched callback consumes the last slot; root and child contenders cannot create extra calls or refund it as uninvoked.
#[tokio::test]
async fn ad_mixed_custom_root_child_last_slot_keeps_consumed_attempt() {
    let (_, execution) = Box::pin(child_tests::execution(1)).await;
    let raw = Provider::new(&["actual auxiliary"], true);
    let managed = ai_agents_llm::managed_provider(raw.clone());
    let child_raw = Provider::new(&["must not run"], false);
    let child = Arc::new(
        crate::AgentBuilder::new()
            .system_prompt("child")
            .llm(child_raw.clone())
            .build()
            .unwrap(),
    );
    let mut first = Box::pin(scope_execution(execution.clone(), async {
        crate::autonomy::owned_maintenance("last-slot", async {
            managed
                .complete(&[ChatMessage::user("auxiliary")], None)
                .await
                .map_err(AgentError::from)
        })
        .await
    }));
    tokio::select! { result = &mut first => panic!("first did not dispatch: {:?}", result.err()), permit = raw.entered.acquire() => permit.unwrap().forget() }
    assert!(
        scope_execution(execution.clone(), child.chat("contender"))
            .await
            .is_err()
    );
    assert_eq!(child_raw.calls.load(Ordering::SeqCst), 0);
    raw.release.add_permits(1);
    assert!(first.await.is_err());
    assert_eq!(raw.calls.load(Ordering::SeqCst), 1);
    let payload: Box<crate::autonomy::TaskCheckpointPayload> =
        serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
    assert_eq!(payload.counters.llm_attempts, 1);
    assert!(
        payload
            .reservations
            .iter()
            .any(|reservation| reservation.state != TaskEffectState::Completed)
    );
}

// Managed overflow policies cannot silently lose required work, and inline overflow cannot overtake its predecessor.
#[tokio::test]
async fn ad_runtime_overflow_requiredness_and_same_key_order() {
    for policy in [
        BackgroundOverflowPolicy::RunInline,
        BackgroundOverflowPolicy::Drop,
        BackgroundOverflowPolicy::Error,
    ] {
        let (parent, execution) = Box::pin(child_tests::execution(2)).await;
        let mut runtime = crate::AgentBuilder::new()
            .system_prompt("queue")
            .llm(Provider::new(&["unused"], false))
            .build()
            .unwrap();
        runtime.background_maintenance =
            Arc::new(crate::optimization::BackgroundMaintenanceQueue::new(1));
        runtime
            .runtime_config
            .optimization
            .post_turn
            .on_background_overflow = policy;
        let events = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        scope_execution(execution.clone(), async {
            let key =
                MaintenanceSequenceKey::actor("queue", "actor", RuntimeTaskPurpose::PostTurnFacts);
            let first_events = events.clone();
            let first_entered = entered.clone();
            let first_release = release.clone();
            runtime
                .spawn_or_handle_background(
                    Some(key.clone()),
                    async move {
                        first_events.lock().push(1);
                        first_entered.add_permits(1);
                        first_release.acquire().await.unwrap().forget();
                        first_events.lock().push(2);
                        Ok(())
                    },
                    "first",
                    &Default::default(),
                )
                .await
                .unwrap();
            entered.acquire().await.unwrap().forget();
            let next_events = events.clone();
            let settings = Default::default();
            let mut next = Box::pin(runtime.spawn_or_handle_background(
                Some(key),
                async move {
                    next_events.lock().push(3);
                    Ok(())
                },
                "second",
                &settings,
            ));
            if policy == BackgroundOverflowPolicy::RunInline {
                assert!(futures::poll!(&mut next).is_pending());
                assert_eq!(*events.lock(), vec![1]);
                release.add_permits(1);
                next.await.unwrap();
                assert_eq!(*events.lock(), vec![1, 2, 3]);
            } else {
                assert!(next.await.is_err());
                assert!(execution.owner.work.unsettled());
                release.add_permits(1);
            }
            runtime.flush_background_tasks().await.unwrap();
            assert!(!execution.owner.work.unsettled());
            let payload: Box<crate::autonomy::TaskCheckpointPayload> =
                serde_json::from_value(execution.load_owned().await.unwrap().payload).unwrap();
            assert_eq!(payload.counters.llm_attempts, 0);
            assert!(
                payload
                    .reservations
                    .iter()
                    .all(|reservation| reservation.state == TaskEffectState::Completed)
            );
        })
        .await;
        Box::pin(child_tests::finish(&parent, &execution)).await;
    }
}
