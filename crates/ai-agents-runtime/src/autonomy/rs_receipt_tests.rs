use super::*;

use crate::Agent;

// Pricing receipts use the same real write-fault boundary as other continuations; an ambiguous acknowledgement never refunds or replays the request.
#[tokio::test]
async fn rs_paid_settlement_receipt_failure_retains_exact_bound_and_owner() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for mode in ["reject", "lost", "drop"] {
            let (endpoint, calls, server) =
                paid_server(vec![json!({"role":"assistant","content":"done"})]).await;
            let pricing = DeclaredPricing {
                schedule_id: "receipt.v1".into(),
                model: "fixture-model".into(),
                endpoint: endpoint.clone(),
                server_input_token_limit: 32,
                server_output_token_limit: 2048,
                input_micro_usd_per_million: 1_000_000,
                output_micro_usd_per_million: 1_000_000,
                request_fee_micro_usd: 0,
                max_internal_attempts: 1,
                complete_usage: true,
            };
            let provider = ProviderBuilder::new()
                .provider(ProviderType::OpenAI)
                .model("fixture-model")
                .api_key("isolated-receipt-fixture")
                .base_url(endpoint)
                .build()
                .unwrap()
                .with_declared_pricing(pricing)
                .unwrap();
            let agent = Arc::new(
                crate::AgentBuilder::new()
                    .system_prompt("work")
                    .llm(Arc::new(provider))
                    .build()
                    .unwrap(),
            );
            let store = crate::autonomy::continuation_test_support::store(
                agent.info().id,
                "rs.receipt.v1",
                sqlite,
            )
            .await;
            let fault = crate::autonomy::continuation_test_support::ReceiptStore::checkpoint(
                store.clone(),
                mode,
                json!({"completed":0}),
            );
            fault.armed.store(true, Ordering::SeqCst);
            let mut config = crate::autonomy::runner_tests::config(1);
            config.defaults.max_cost_usd = Some(UsdAmount::parse("0.002080").unwrap());
            let ceilings = AutonomyHostCeilings {
                max_cost_usd: config.defaults.max_cost_usd.clone(),
                ..Default::default()
            };
            let runner = AutonomyRunner::try_new(
                agent.clone(),
                config,
                ceilings,
                fault.clone(),
                "rs.receipt.v1".into(),
            )
            .unwrap();
            if mode == "drop" {
                let mut work = Box::pin(runner.run("work", None));
                tokio::select! {
                    result=&mut work=>panic!("fault must hold the persisted settlement: {result:?}"),
                    permit=fault.reached.acquire()=>permit.unwrap().forget(),
                    _=tokio::time::sleep(std::time::Duration::from_secs(15))=>panic!("settlement fixture was not reached"),
                }
                drop(work);
            } else {
                let result = runner.run("work", None).await;
                assert!(
                    !result
                        .as_ref()
                        .is_ok_and(|result| result.run.status == TaskRunStatus::Completed)
                );
            }
            fault.assert_hit();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(agent.chat("foreign").await.is_err());
            assert!(runner.run("replay", None).await.is_err());
            let runs = store.list().await.unwrap();
            assert_eq!(runs.len(), 1);
            let snapshot = store.load(&runs[0].key.run_id).await.unwrap().unwrap();
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload).unwrap();
            let paid = payload
                .reservations
                .iter()
                .find(|reservation| reservation.reserved_micro_usd != 0)
                .unwrap();
            assert_eq!(paid.reserved_micro_usd, 2080);
            assert_eq!(payload.counters.llm_attempts, 1);
            if mode == "reject" {
                assert_eq!(paid.charged_micro_usd, 0);
                assert_eq!(paid.state, TaskEffectState::Dispatched);
            } else {
                assert_eq!(paid.charged_micro_usd, 14);
                assert_eq!(payload.counters.charged_micro_usd, 14);
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            server.await.unwrap();
        }
    }
}
