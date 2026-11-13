//! Streaming domains use the same real host mutations, revision sources and validators as runtime acceptance.
use super::*;
use futures::StreamExt;

// The shared controller publishes real validation/progress across distinct domains without requiring todos or local files.
#[tokio::test]
async fn ev_real_domain_validation_and_repair_streams() {
    for &sqlite in crate::autonomy::continuation_test_support::backends() {
        for (domain, op) in [
            (
                "report",
                json!({"op":"report","value":{"title":"verified","rows":[1,2]}}),
            ),
            ("operations", json!({"op":"incident"})),
            ("world", json!({"op":"world","value":10})),
            ("code", json!({"op":"code","value":"return 2"})),
        ] {
            let state = DomainState::new();
            let call = json!({"tool":"domain","arguments":op}).to_string();
            let provider =
                RecordingProvider::new(&["unproven claim", &call, "host state repaired"]);
            let agent = domain_agent(state.clone(), provider.clone(), false);
            let yaml = format!(
                "validation:\n  checks: [{{id: records, adapter: host.records, config: {{domain: {domain}}}}}]\ncompletion: {{validator_passed: records}}\n"
            );
            let (runner, store) = runner(agent, config(&yaml), sqlite).await;
            let mut stream = runner
                .stream_operation(
                    TaskStreamOperation::Run {
                        objective: "verify real host state".into(),
                        profile: None,
                    },
                    TaskStreamConfig { capacity: 1 },
                )
                .unwrap();
            let mut events = vec![];
            while let Some(event) = stream.next().await {
                events.push(event);
            }
            let result = stream.take_result().unwrap().unwrap();
            assert_eq!(result.run.status, TaskRunStatus::Completed, "{domain}");
            assert_eq!(state.effects.load(Ordering::SeqCst), 1);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
            assert!(
                events
                    .iter()
                    .any(|e| e.kind == TaskRunEventKind::ValidationCompleted
                        && e.data["outcome"] == "fail")
            );
            assert!(
                events
                    .iter()
                    .any(|e| e.kind == TaskRunEventKind::ValidationCompleted
                        && e.data["outcome"] == "pass")
            );
            assert!(
                events.iter().any(
                    |e| e.kind == TaskRunEventKind::ToolCompleted && e.data["executed"] == true
                )
            );
            assert_eq!(events.last().unwrap().kind, TaskRunEventKind::TaskCompleted);
            assert_eq!(
                store
                    .load(&result.run.key.run_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .revision,
                result.run.revision
            );
        }
    }
}

// Duplicate sources are not new progress; semantic judge calls remain on the same admitted ledger in event mode.
#[tokio::test]
async fn ev_research_deduplication_and_judge_share_real_evidence() {
    let state = DomainState::new();
    let provider = RecordingProvider::new(&[
        r#"{"tool":"domain","arguments":{"op":"source","value":"A"}}"#,
        "source A",
        "{\"overall_score\":0.9}",
        r#"{"tool":"domain","arguments":{"op":"source","value":"A"}}"#,
        "duplicate A",
        "{\"overall_score\":0.9}",
        r#"{"tool":"domain","arguments":{"op":"source","value":"B"}}"#,
        "sources A,B",
        "{\"overall_score\":0.9}",
    ]);
    let agent = domain_agent(state.clone(), provider.clone(), false);
    let config = config(
        "validation:\n  checks: [{id: records, adapter: host.records, config: {domain: research}}, {id: quality, adapter: builtin.judge, config: {criteria: [grounded sources], pass_threshold: 0.8}}]\ncompletion: {all: [{validator_passed: records}, {validator_passed: quality}]}\nprogress:\n  adapter: builtin.evidence\n  config:\n    signals: [{id: sources, validator: records, path: metrics.unique, comparison: increase}]\n",
    );
    let (runner, _) = runner(agent, config, false).await;
    let mut stream = runner.run_stream("two distinct sources", None).unwrap();
    let mut events = vec![];
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let result = stream.take_result().unwrap().unwrap();
    assert_eq!(result.run.status, TaskRunStatus::Completed);
    assert_eq!(result.run.counters.llm_attempts, 9);
    assert_eq!(state.effects.load(Ordering::SeqCst), 3);
    let deltas: Vec<_> = events
        .iter()
        .filter(|e| e.kind == TaskRunEventKind::ProgressObserved)
        .map(|e| e.data["delta"].clone())
        .collect();
    assert_eq!(
        deltas,
        vec![json!("unchanged"), json!("unchanged"), json!("advanced")]
    );
    assert!(
        !serde_json::to_string(&events)
            .unwrap()
            .contains("two distinct sources")
    );
}
