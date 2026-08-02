use crate::event::{EventStatus, EventType, ObservationPurpose};
use crate::manager::ObservabilityManager;
use ai_agents_core::{AgentError, AgentResponse, KeyFact, ToolExecutionRecord};
use ai_agents_hitl::{ApprovalRequest, ApprovalResolvedOutcome, ApprovalResult, ApprovalTrigger};
use ai_agents_hooks::AgentHooks;
use ai_agents_memory::{MemoryBudgetEvent, MemoryCompressEvent, MemoryEvictEvent};
use ai_agents_relationships::{DimensionChange, Relationship, RelationshipEvent};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// AgentHooks implementation that records lifecycle events not covered by wrappers.
pub struct ObservabilityHooks {
    manager: Arc<ObservabilityManager>,
}

impl ObservabilityHooks {
    /// Creates lifecycle hooks backed by a shared manager.
    pub fn new(manager: Arc<ObservabilityManager>) -> Self {
        Self { manager }
    }

    fn record(
        &self,
        event_type: EventType,
        purpose: ObservationPurpose,
        tags: HashMap<String, String>,
    ) {
        self.manager.record_lifecycle_event(
            event_type,
            purpose,
            EventStatus::Success,
            0,
            tags,
            None,
        );
    }
}

fn approval_trigger_label(trigger: &ApprovalTrigger) -> (&'static str, Option<String>) {
    match trigger {
        ApprovalTrigger::Tool { name, .. } => ("tool", Some(name.clone())),
        ApprovalTrigger::Condition { name, .. } => ("condition", Some(name.clone())),
        ApprovalTrigger::State { to, .. } => ("state", Some(to.clone())),
    }
}

fn approval_result_label(result: &ApprovalResult) -> &'static str {
    match result {
        ApprovalResult::Approved => "approved",
        ApprovalResult::Rejected { .. } => "rejected",
        ApprovalResult::Modified { .. } => "modified",
        ApprovalResult::Timeout => "timeout",
    }
}

fn approval_outcome_label(outcome: &ApprovalResolvedOutcome) -> &'static str {
    match outcome {
        ApprovalResolvedOutcome::Approved => "approved",
        ApprovalResolvedOutcome::Rejected { .. } => "rejected",
        ApprovalResolvedOutcome::Modified { .. } => "modified",
        ApprovalResolvedOutcome::Error { .. } => "error",
    }
}

fn purpose_for_tool_record(record: &ToolExecutionRecord) -> ObservationPurpose {
    match &record.source {
        ai_agents_core::ToolCallSource::Model
        | ai_agents_core::ToolCallSource::Manual
        | ai_agents_core::ToolCallSource::EvalFixture
        | ai_agents_core::ToolCallSource::Task => ObservationPurpose::MainResponse,
        ai_agents_core::ToolCallSource::Skill { .. } => ObservationPurpose::SkillRouting,
        ai_agents_core::ToolCallSource::StateAction { .. } => ObservationPurpose::StateAction,
        ai_agents_core::ToolCallSource::Plan { .. } => ObservationPurpose::PlanStep,
        ai_agents_core::ToolCallSource::Orchestration
        | ai_agents_core::ToolCallSource::Spawner
        | ai_agents_core::ToolCallSource::Fallback { .. } => {
            ObservationPurpose::OrchestrationRouting
        }
    }
}

#[async_trait]
impl AgentHooks for ObservabilityHooks {
    async fn on_tool_execution_record(&self, record: &ToolExecutionRecord) {
        if !self.manager.config().latency.track_tools || record.executed {
            return;
        }
        let mut tags = HashMap::new();
        tags.insert("requested_name".to_string(), record.requested_name.clone());
        tags.insert(
            "policy_outcome".to_string(),
            format!("{:?}", record.policy.outcome).to_ascii_lowercase(),
        );
        tags.insert("executed".to_string(), record.executed.to_string());
        tags.insert("success".to_string(), record.success.to_string());
        let payload = Some(serde_json::json!({
            "call_id": record.call_id,
            "approval_status": record.approval.as_ref().map(|approval| format!("{:?}", approval.status).to_ascii_lowercase()),
            "policy_reason": record.policy.reason,
        }));
        self.manager.record_lifecycle_event(
            EventType::ToolCall {
                tool_id: record.canonical_id.clone(),
            },
            purpose_for_tool_record(record),
            EventStatus::Skipped,
            record.duration_ms,
            tags,
            payload,
        );
    }

    async fn on_state_transition(&self, from: Option<&str>, to: &str, reason: &str) {
        let mut tags = HashMap::new();
        tags.insert("reason".to_string(), reason.to_string());
        self.record(
            EventType::StateTransition {
                from: from.map(str::to_string),
                to: to.to_string(),
            },
            ObservationPurpose::StateTransitionEvaluation,
            tags,
        );
    }

    async fn on_error(&self, error: &AgentError) {
        self.manager.record_lifecycle_event(
            EventType::Error,
            ObservationPurpose::default(),
            EventStatus::Error,
            0,
            HashMap::new(),
            Some(serde_json::json!({"kind": "agent_error", "message": error.to_string()})),
        );
    }

    async fn on_approval_requested(&self, request: &ApprovalRequest) {
        if !self.manager.config().latency.track_hitl {
            return;
        }
        let (trigger, detail) = approval_trigger_label(&request.trigger);
        let mut tags = HashMap::new();
        tags.insert("request_id".to_string(), request.id.clone());
        if let Some(detail) = detail {
            tags.insert("trigger_id".to_string(), detail);
        }
        self.record(
            EventType::HitlApproval {
                trigger: trigger.to_string(),
            },
            ObservationPurpose::HitlLocalization,
            tags,
        );
    }

    async fn on_approval_result(&self, request_id: &str, result: &ApprovalResult) {
        if !self.manager.config().latency.track_hitl {
            return;
        }
        let mut tags = HashMap::new();
        tags.insert("request_id".to_string(), request_id.to_string());
        tags.insert(
            "result".to_string(),
            approval_result_label(result).to_string(),
        );
        self.record(
            EventType::HitlApproval {
                trigger: "result".to_string(),
            },
            ObservationPurpose::HitlLocalization,
            tags,
        );
    }

    async fn on_approval_resolved(
        &self,
        request: &ApprovalRequest,
        raw_result: &ApprovalResult,
        outcome: &ApprovalResolvedOutcome,
    ) {
        if !self.manager.config().latency.track_hitl {
            return;
        }
        let mut tags = HashMap::new();
        tags.insert("request_id".to_string(), request.id.clone());
        tags.insert(
            "raw_result".to_string(),
            approval_result_label(raw_result).to_string(),
        );
        tags.insert(
            "result".to_string(),
            approval_outcome_label(outcome).to_string(),
        );
        tags.insert(
            "request_trigger".to_string(),
            request.trigger.trigger_type().to_string(),
        );
        let status = if matches!(outcome, ApprovalResolvedOutcome::Error { .. }) {
            EventStatus::Error
        } else {
            EventStatus::Success
        };
        self.manager.record_lifecycle_event(
            EventType::HitlApproval {
                trigger: "resolved".to_string(),
            },
            ObservationPurpose::HitlLocalization,
            status,
            0,
            tags,
            Some(serde_json::json!({
                "raw_result": raw_result,
                "outcome": outcome,
                "request_trigger": &request.trigger,
            })),
        );
    }

    async fn on_memory_compress(&self, event: &MemoryCompressEvent) {
        self.manager.record_lifecycle_event(
            EventType::MemoryOperation {
                operation: "compress".to_string(),
            },
            ObservationPurpose::Summarization,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({
                "messages_compressed": event.messages_compressed,
                "compression_ratio": event.compression_ratio,
            })),
        );
    }

    async fn on_memory_evict(&self, event: &MemoryEvictEvent) {
        self.manager.record_lifecycle_event(
            EventType::MemoryOperation {
                operation: "evict".to_string(),
            },
            ObservationPurpose::Summarization,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({
                "messages_evicted": event.messages_evicted,
                "reason": format!("{:?}", event.reason),
            })),
        );
    }

    async fn on_memory_budget_warning(&self, event: &MemoryBudgetEvent) {
        let mut tags = HashMap::new();
        tags.insert("warning".to_string(), "memory_budget".to_string());
        self.manager.record_lifecycle_event(
            EventType::MemoryOperation {
                operation: "budget_warning".to_string(),
            },
            ObservationPurpose::Summarization,
            EventStatus::Success,
            0,
            tags,
            Some(serde_json::json!({
                "component": event.component,
                "used_tokens": event.used_tokens,
                "budget_tokens": event.budget_tokens,
                "usage_percent": event.usage_percent,
            })),
        );
    }

    async fn on_delegate_start(&self, agent_id: &str, state: &str) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        let mut tags = HashMap::new();
        tags.insert("agent_id".to_string(), agent_id.to_string());
        tags.insert("state".to_string(), state.to_string());
        self.record(
            EventType::Orchestration {
                pattern: "delegate".to_string(),
            },
            ObservationPurpose::OrchestrationRouting,
            tags,
        );
    }

    async fn on_delegate_complete(&self, agent_id: &str, state: &str, duration_ms: u64) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        let mut tags = HashMap::new();
        tags.insert("agent_id".to_string(), agent_id.to_string());
        tags.insert("state".to_string(), state.to_string());
        self.manager.record_lifecycle_event(
            EventType::Orchestration {
                pattern: "delegate".to_string(),
            },
            ObservationPurpose::OrchestrationRouting,
            EventStatus::Success,
            duration_ms,
            tags,
            None,
        );
    }

    async fn on_concurrent_complete(&self, agent_ids: &[String], strategy: &str, duration_ms: u64) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        self.manager.record_lifecycle_event(
            EventType::Orchestration {
                pattern: "concurrent".to_string(),
            },
            ObservationPurpose::OrchestrationAggregation,
            EventStatus::Success,
            duration_ms,
            HashMap::new(),
            Some(serde_json::json!({"agents": agent_ids, "strategy": strategy})),
        );
    }

    async fn on_group_chat_round(&self, round: u32, speaker: &str, _content: &str) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        self.manager.record_lifecycle_event(
            EventType::Orchestration {
                pattern: "group_chat".to_string(),
            },
            ObservationPurpose::OrchestrationConversation,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"round": round, "speaker": speaker})),
        );
    }

    async fn on_pipeline_stage(&self, stage: usize, agent_id: &str, duration_ms: u64) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        self.manager.record_lifecycle_event(
            EventType::Orchestration {
                pattern: "pipeline".to_string(),
            },
            ObservationPurpose::OrchestrationRouting,
            EventStatus::Success,
            duration_ms,
            HashMap::new(),
            Some(serde_json::json!({"stage": stage, "agent_id": agent_id})),
        );
    }

    async fn on_pipeline_complete(&self, stages: usize, duration_ms: u64) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        self.manager.record_lifecycle_event(
            EventType::Orchestration {
                pattern: "pipeline".to_string(),
            },
            ObservationPurpose::OrchestrationAggregation,
            EventStatus::Success,
            duration_ms,
            HashMap::new(),
            Some(serde_json::json!({"stages": stages})),
        );
    }

    async fn on_handoff_start(&self, initial_agent: &str) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        self.manager.record_lifecycle_event(
            EventType::Orchestration {
                pattern: "handoff".to_string(),
            },
            ObservationPurpose::OrchestrationRouting,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"initial_agent": initial_agent})),
        );
    }

    async fn on_handoff(&self, from: &str, to: &str, reason: &str) {
        if !self.manager.config().latency.track_orchestration {
            return;
        }
        self.manager.record_lifecycle_event(
            EventType::Orchestration {
                pattern: "handoff".to_string(),
            },
            ObservationPurpose::OrchestrationRouting,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"from": from, "to": to, "reason": reason})),
        );
    }

    async fn on_persona_evolve(
        &self,
        field: &str,
        _old_value: &Value,
        _new_value: &Value,
        reason: Option<&str>,
    ) {
        self.manager.record_lifecycle_event(
            EventType::PersonaEvent {
                event: "evolve".to_string(),
            },
            ObservationPurpose::Other("persona".to_string()),
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"field": field, "reason": reason})),
        );
    }

    async fn on_secret_revealed(&self, _content: &str) {
        self.record(
            EventType::PersonaEvent {
                event: "secret_revealed".to_string(),
            },
            ObservationPurpose::Other("persona".to_string()),
            HashMap::new(),
        );
    }

    async fn on_facts_extracted(&self, actor_id: &str, facts: &[KeyFact]) {
        self.manager.record_lifecycle_event(
            EventType::FactsEvent {
                event: "extracted".to_string(),
            },
            ObservationPurpose::FactsExtraction,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"actor_id": actor_id, "count": facts.len()})),
        );
    }

    async fn on_actor_memory_loaded(&self, actor_id: &str, fact_count: usize) {
        self.manager.record_lifecycle_event(
            EventType::FactsEvent {
                event: "actor_memory_loaded".to_string(),
            },
            ObservationPurpose::FactsExtraction,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"actor_id": actor_id, "count": fact_count})),
        );
    }

    async fn on_relationship_loaded(&self, actor_id: &str, _relationship: &Relationship) {
        self.manager.record_lifecycle_event(
            EventType::RelationshipEvent {
                event: "loaded".to_string(),
            },
            ObservationPurpose::RelationshipUpdate,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"actor_id": actor_id})),
        );
    }

    async fn on_relationship_change(&self, actor_id: &str, changes: &[DimensionChange]) {
        self.manager.record_lifecycle_event(
            EventType::RelationshipEvent {
                event: "changed".to_string(),
            },
            ObservationPurpose::RelationshipUpdate,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({"actor_id": actor_id, "changes": changes.len()})),
        );
    }

    async fn on_notable_event(&self, actor_id: &str, event: &RelationshipEvent) {
        self.manager.record_lifecycle_event(
            EventType::RelationshipEvent {
                event: "notable_event".to_string(),
            },
            ObservationPurpose::RelationshipUpdate,
            EventStatus::Success,
            0,
            HashMap::new(),
            Some(serde_json::json!({
                "actor_id": actor_id,
                "significance": event.significance,
            })),
        );
    }

    async fn on_response(&self, _response: &AgentResponse) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ExportConfig, LatencyConfig, ObservabilityConfig, PrivacyConfig};

    #[tokio::test]
    async fn records_raw_and_effective_approval_results_separately() {
        let config = ObservabilityConfig {
            enabled: true,
            export: ExportConfig {
                write_raw_events: true,
                ..Default::default()
            },
            privacy: PrivacyConfig {
                hash_inputs: false,
                max_text_chars: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        let hooks = ObservabilityHooks::new(manager.clone());

        let request = ApprovalRequest::new(
            ApprovalTrigger::tool("calculator", serde_json::json!({})),
            "Approve?",
        );
        hooks
            .on_approval_result(&request.id, &ApprovalResult::Timeout)
            .await;
        hooks
            .on_approval_resolved(
                &request,
                &ApprovalResult::Timeout,
                &ApprovalResolvedOutcome::Approved,
            )
            .await;

        let events = manager.raw_events();
        assert_eq!(events.len(), 2);
        let raw = events
            .iter()
            .find(|event| {
                matches!(
                    &event.event_type,
                    EventType::HitlApproval { trigger } if trigger == "result"
                )
            })
            .expect("raw approval event");
        assert_eq!(raw.tags.get("result").map(String::as_str), Some("timeout"));
        let resolved = events
            .iter()
            .find(|event| {
                matches!(
                    &event.event_type,
                    EventType::HitlApproval { trigger } if trigger == "resolved"
                )
            })
            .expect("resolved approval event");
        assert_eq!(
            resolved.tags.get("raw_result").map(String::as_str),
            Some("timeout")
        );
        assert_eq!(
            resolved.tags.get("result").map(String::as_str),
            Some("approved")
        );
        assert_eq!(resolved.status, EventStatus::Success);
    }

    #[tokio::test]
    async fn records_timeout_policy_errors_as_resolved_events() {
        let config = ObservabilityConfig {
            enabled: true,
            export: ExportConfig {
                write_raw_events: true,
                ..Default::default()
            },
            privacy: PrivacyConfig {
                max_text_chars: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        let hooks = ObservabilityHooks::new(manager.clone());
        let request = ApprovalRequest::new(
            ApprovalTrigger::state(Some("review".to_string()), "execute"),
            "Approve?",
        );

        hooks
            .on_approval_resolved(
                &request,
                &ApprovalResult::Timeout,
                &ApprovalResolvedOutcome::Error {
                    message: "HITL approval timeout".to_string(),
                },
            )
            .await;

        let events = manager.raw_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].status, EventStatus::Error);
        assert_eq!(
            events[0].tags.get("raw_result").map(String::as_str),
            Some("timeout")
        );
        assert_eq!(
            events[0].tags.get("result").map(String::as_str),
            Some("error")
        );
    }

    #[tokio::test]
    async fn hitl_tracking_switch_applies_to_resolved_results() {
        let config = ObservabilityConfig {
            enabled: true,
            latency: LatencyConfig {
                track_hitl: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = ObservabilityManager::new(config);
        let hooks = ObservabilityHooks::new(manager.clone());

        let request = ApprovalRequest::new(
            ApprovalTrigger::tool("calculator", serde_json::json!({})),
            "Approve?",
        );
        hooks
            .on_approval_resolved(
                &request,
                &ApprovalResult::Approved,
                &ApprovalResolvedOutcome::Approved,
            )
            .await;

        assert!(manager.raw_events().is_empty());
    }
}
