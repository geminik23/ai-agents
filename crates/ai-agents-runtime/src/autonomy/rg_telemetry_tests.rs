//! Numeric gate evidence must remain loss-aware across ACK retries, collector changes and bounded retention.
use super::*;
use ai_agents_observability::{EventStatus, ObservabilityConfig, ObservationPurpose, SpanContext};

// No prompt, actor, session or tool result is required to construct an actual recorded collector event.
fn emit(manager: &Arc<ObservabilityManager>, id: &str, run: &str, purpose: ObservationPurpose) {
    let context = SpanContext::new_root("fixture").with_tag("task_run", run);
    let mut event = manager.build_event_from_span(
        context,
        EventType::LlmCall {
            provider: "fixture".into(),
            model: "fixture".into(),
            alias: None,
            streaming: false,
        },
        std::time::Duration::from_millis(7),
        EventStatus::Success,
        None,
        None,
        Default::default(),
        Some(json!({"prompt":"must not persist"})),
    );
    event.span_id = id.into();
    event.purpose = purpose;
    manager.record_event(event);
    manager.event_cursor();
}

// Every case starts with a valid ordinary task payload and a scope produced by the existing evaluation fixtures.
fn payload() -> TaskCheckpointPayload {
    serde_json::from_value(crate::autonomy::tests::checkpoint("run").payload).unwrap()
}

// Collectors are opt-in but absence is durable negative knowledge, not a complete zero that a later collector can repair retroactively.
#[test]
fn rg_telemetry_missing_intervals_and_collector_replacement_remain_incomplete() {
    let config = ObservabilityConfig {
        enabled: true,
        ..Default::default()
    };
    let manager = ObservabilityManager::new(config.clone());
    let absent = TaskTelemetry::default();
    absent.register(None);
    let mut payload = payload();
    absent.prepare(&mut payload, "run").unwrap();
    let fresh = TaskTelemetry::default();
    fresh.register(Some(manager.clone()));
    emit(&manager, "suffix", "run", ObservationPurpose::MainResponse);
    fresh.prepare(&mut payload, "run").unwrap();
    let identity =
        crate::autonomy::evaluation_tests::identity(&crate::autonomy::evaluation_tests::scope());
    assert!(
        !observation(&payload, identity.clone())
            .unwrap()
            .unwrap()
            .complete
    );
    let mut observed = super::tests::payload();
    observed.counters.llm_attempts = 1;
    fresh.prepare(&mut observed, "run").unwrap();
    assert!(
        observation(&observed, identity.clone())
            .unwrap()
            .unwrap()
            .complete
    );
    let reconstructed_without_collector = TaskTelemetry::default();
    reconstructed_without_collector.register(None);
    reconstructed_without_collector
        .prepare(&mut observed, "run")
        .unwrap();
    assert!(!observation(&observed, identity).unwrap().unwrap().complete);
}

// Retrying an unacknowledged cut does not double-count; purpose counters survive projection and unavailable prices/tokens are absent.
#[test]
fn rg_telemetry_ack_retry_preserves_purpose_and_metric_availability() {
    let manager = ObservabilityManager::new(ObservabilityConfig {
        enabled: true,
        ..Default::default()
    });
    let collector = TaskTelemetry::default();
    collector.register(Some(manager.clone()));
    emit(&manager, "main", "run", ObservationPurpose::MainResponse);
    emit(
        &manager,
        "other-task",
        "foreign",
        ObservationPurpose::Router,
    );
    let mut payload = payload();
    let _unacknowledged = collector.prepare(&mut payload, "run").unwrap();
    let cuts = collector.prepare(&mut payload, "run").unwrap();
    collector.acknowledge(cuts);
    emit(&manager, "router", "run", ObservationPurpose::Router);
    collector.prepare(&mut payload, "run").unwrap();
    let identity =
        crate::autonomy::evaluation_tests::identity(&crate::autonomy::evaluation_tests::scope());
    payload.counters.llm_attempts = 2;
    let observed = observation(&payload, identity).unwrap().unwrap();
    assert!(observed.complete);
    assert_eq!(observed.value["summary"]["total_llm_calls"], 2);
    assert_eq!(observed.value["summary"]["main_llm_calls"], 1);
    assert_eq!(observed.value["summary"]["router_calls"], 1);
    assert!(observed.value["summary"].get("total_cost_usd").is_none());
    assert!(observed.value["summary"].get("total_tokens").is_none());
    let scope = crate::autonomy::evaluation_tests::scope();
    let evidence = EvaluationEvidence {
        observability: Some(observed),
        ..Default::default()
    };
    for yaml in [
        "observability: {path: summary.total_cost_usd, exists: false}",
        "not: {observability: {path: summary.total_tokens, exists: true}}",
    ] {
        let gate: CompletionGate = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            CompletionGateEvaluator::default()
                .evaluate(&gate, &scope, &evidence)
                .unwrap()
                .outcome,
            GateOutcome::Unknown
        );
    }
    assert!(
        !serde_json::to_string(&payload.adapters)
            .unwrap()
            .contains("must not persist")
    );
}

// Disabled categories, queue loss and rolling eviction cannot supply passing zero/upper-bound evidence.
#[test]
fn rg_telemetry_disabled_or_evicted_observations_are_unknown() {
    for mode in ["disabled", "evicted", "dropped"] {
        let mut config = ObservabilityConfig {
            enabled: true,
            ..Default::default()
        };
        config.latency.track_llm = mode != "disabled";
        config.aggregation.window_size = 1;
        config.buffer.event_buffer = 1;
        config.buffer.drop_on_full = true;
        let manager = ObservabilityManager::new(config);
        let collector = TaskTelemetry::default();
        collector.register(Some(manager.clone()));
        if mode == "dropped" {
            let event = manager.build_event_from_span(
                SpanContext::new_root("fixture").with_tag("task_run", "run"),
                EventType::LlmCall {
                    provider: "fixture".into(),
                    model: "fixture".into(),
                    alias: None,
                    streaming: false,
                },
                std::time::Duration::ZERO,
                EventStatus::Success,
                None,
                None,
                Default::default(),
                None,
            );
            manager.record_event(event.clone());
            manager.record_event(event);
        } else {
            emit(&manager, "first", "run", ObservationPurpose::MainResponse);
            if mode == "evicted" {
                emit(&manager, "second", "run", ObservationPurpose::MainResponse);
            }
        }
        let mut payload = payload();
        payload.counters.llm_attempts = if mode == "disabled" { 1 } else { 2 };
        collector.prepare(&mut payload, "run").unwrap();
        assert_eq!(
            payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == ADAPTER)
                .unwrap()
                .state["complete"],
            false
        );
        let scope = crate::autonomy::evaluation_tests::scope();
        let observed = observation(
            &payload,
            crate::autonomy::evaluation_tests::identity(&scope),
        )
        .unwrap()
        .unwrap();
        assert!(!observed.complete);
        let evidence = EvaluationEvidence {
            observability: Some(observed),
            ..Default::default()
        };
        let gate: CompletionGate =
            serde_yaml::from_str("observability: {path: summary.total_llm_calls, lte: 10}")
                .unwrap();
        assert_eq!(
            CompletionGateEvaluator::default()
                .evaluate(&gate, &scope, &evidence)
                .unwrap()
                .outcome,
            GateOutcome::Unknown
        );
    }
}

// Disabled token measurement and unknown-price zero placeholders are not measured zero cost, even when the report contains a numeric estimate.
#[test]
fn rg_telemetry_partial_usage_and_unknown_price_zero_never_prove_cost() {
    for partial in [false, true] {
        let mut config = ObservabilityConfig {
            enabled: true,
            ..Default::default()
        };
        config.cost.unknown_price_policy = ai_agents_observability::UnknownPricePolicy::Zero;
        if partial {
            config.tokens.count_input = false;
            config.tokens.count_output = false;
            config.cost.pricing.insert(
                "fixture".into(),
                ai_agents_observability::ModelPricing {
                    input_per_1k: 1.0,
                    output_per_1k: 1.0,
                },
            );
        }
        let manager = ObservabilityManager::new(config);
        let collector = TaskTelemetry::default();
        collector.register(Some(manager.clone()));
        let context = SpanContext::new_root("fixture").with_tag("task_run", "run");
        let event = manager.build_event_from_span(
            context,
            EventType::LlmCall {
                provider: "fixture".into(),
                model: "fixture".into(),
                alias: None,
                streaming: false,
            },
            std::time::Duration::from_millis(1),
            EventStatus::Success,
            Some(ai_agents_observability::ObservationTokenUsage::new(
                10,
                10,
                ai_agents_observability::TokenUsageSource::Provider,
            )),
            None,
            Default::default(),
            None,
        );
        manager.record_event(event);
        let mut payload = payload();
        collector.prepare(&mut payload, "run").unwrap();
        let scope = crate::autonomy::evaluation_tests::scope();
        payload.counters.llm_attempts = 1;
        let observed = observation(
            &payload,
            crate::autonomy::evaluation_tests::identity(&scope),
        )
        .unwrap()
        .unwrap();
        assert!(observed.complete);
        assert!(observed.value["summary"].get("total_cost_usd").is_none());
        let evidence = EvaluationEvidence {
            observability: Some(observed),
            ..Default::default()
        };
        for yaml in [
            "observability: {path: summary.total_cost_usd, lte: 0}",
            "observability: {path: summary.total_cost_usd, exists: false}",
            "not: {observability: {path: summary.total_cost_usd, exists: true}}",
        ] {
            let gate: CompletionGate = serde_yaml::from_str(yaml).unwrap();
            assert_eq!(
                CompletionGateEvaluator::default()
                    .evaluate(&gate, &scope, &evidence)
                    .unwrap()
                    .outcome,
                GateOutcome::Unknown
            );
        }
    }
}

// Saturation freezes numeric evidence as explicitly incomplete while permitting further ACKs and cleanup publications.
#[test]
fn rg_telemetry_saturation_is_bounded_and_does_not_block_cleanup() {
    let mut config = ObservabilityConfig {
        enabled: true,
        ..Default::default()
    };
    config.aggregation.window_size = 5000;
    let manager = ObservabilityManager::new(config);
    let collector = TaskTelemetry::default();
    collector.register(Some(manager.clone()));
    for id in 0..4100 {
        emit(
            &manager,
            &format!("event-{id}"),
            "run",
            ObservationPurpose::MainResponse,
        );
    }
    let mut payload = payload();
    let cuts = collector.prepare(&mut payload, "run").unwrap();
    collector.acknowledge(cuts);
    let journal: Journal = serde_json::from_value(
        payload
            .adapters
            .iter()
            .find(|adapter| adapter.id == ADAPTER)
            .unwrap()
            .state
            .clone(),
    )
    .unwrap();
    assert_eq!(journal.events.len(), 4096);
    assert!(!journal.complete);
    emit(
        &manager,
        "cleanup-tail",
        "run",
        ObservationPurpose::MainResponse,
    );
    collector.prepare(&mut payload, "run").unwrap();
}
