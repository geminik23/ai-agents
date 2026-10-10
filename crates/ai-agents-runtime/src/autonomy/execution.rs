//! One conditional task ledger for real invocation attempts, active clocks and terminal settlement.

use super::*;
use ai_agents_core::{AgentError, LLMError, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Instant;

/// Live state supplements, never replaces, the durable owner/revision and dispatched-effect markers.
pub(crate) struct RunExecution {
    pub(crate) store: Arc<dyn TaskRunStore>,
    pub(crate) owner: Arc<RunOwner>,
    pub(crate) participants: Arc<super::participants::Participants>,
    pub(crate) targets: Arc<super::CompositionTargets>,
    acknowledged: parking_lot::RwLock<TaskRunSnapshot>,
    limits: TaskRunLimits,
    pub(crate) expiry_projection: Option<Instant>,
    pub(crate) run_id: String,
    pub(crate) owner_token: String,
    pub(crate) revision: Arc<AtomicU64>,
    pub(crate) serial: Arc<tokio::sync::Mutex<()>>,
    // Slot validation and frame publication are atomic with respect to other message installers, not child execution.
    pub(crate) message_admission: tokio::sync::Mutex<()>,
    pub(crate) cancellation: Arc<AtomicBool>,
    started: Instant,
    initial_active_millis: u64,
    interrupted_millis: u64,
    pub(crate) parked_approvals: parking_lot::Mutex<
        std::collections::BTreeMap<String, Vec<super::suspension::BatchApproval>>,
    >,
    pub(crate) approval_deadlines: parking_lot::Mutex<std::collections::BTreeMap<String, Instant>>,
    pub(crate) parked_batches:
        parking_lot::Mutex<std::collections::BTreeMap<String, TaskBatchState>>,
    pub(crate) delegate_frames:
        parking_lot::Mutex<std::collections::BTreeMap<String, super::composition::DelegateFrame>>,
    stop: parking_lot::RwLock<Option<String>>,
    lifecycle: parking_lot::RwLock<(AutonomyProfile, LifecycleState)>,
    scope: parking_lot::RwLock<Option<EvaluationScope>>,
    mutation_generation: AtomicU64,
    todos:
        parking_lot::RwLock<Option<(ai_agents_tools::TodoStore, ai_agents_tools::TodoRunBinding)>>,
}

impl RunExecution {
    /// Called only after an acknowledged running claim; restart clocks cannot be refilled here.
    pub(crate) fn new(
        store: Arc<dyn TaskRunStore>,
        snapshot: &TaskRunSnapshot,
        lifecycle: LifecycleState,
        owner: Arc<RunOwner>,
        started: Instant,
    ) -> Result<Arc<Self>> {
        let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        Ok(Arc::new(Self {
            store,
            owner: owner.clone(),
            participants: Arc::new(super::participants::Participants::default()),
            targets: owner.targets.clone(),
            acknowledged: parking_lot::RwLock::new(snapshot.clone()),
            limits: payload.limits.clone(),
            expiry_projection: payload
                .clocks
                .expires_at
                .map(|expiry| {
                    let remaining = (expiry - chrono::Utc::now()).to_std().unwrap_or_default();
                    Instant::now()
                        .checked_add(remaining)
                        .ok_or_else(|| AgentError::Config("expiry projection overflow".into()))
                })
                .transpose()?,
            run_id: snapshot.key.run_id.clone(),
            owner_token: snapshot
                .owner_token
                .clone()
                .ok_or(TaskRunStorageError::Owned)?,
            revision: Arc::new(AtomicU64::new(snapshot.revision)),
            serial: Arc::new(tokio::sync::Mutex::new(())),
            message_admission: tokio::sync::Mutex::new(()),
            cancellation: Arc::new(AtomicBool::new(false)),
            started,
            initial_active_millis: payload.clocks.active_millis,
            interrupted_millis: payload.clocks.interrupted_interval_millis,
            parked_approvals: Default::default(),
            approval_deadlines: Default::default(),
            parked_batches: Default::default(),
            delegate_frames: Default::default(),
            stop: parking_lot::RwLock::new(None),
            lifecycle: parking_lot::RwLock::new((payload.settings, lifecycle)),
            scope: parking_lot::RwLock::new(None),
            mutation_generation: AtomicU64::new(
                payload
                    .evidence
                    .tool_calls
                    .iter()
                    .filter_map(|capture| {
                        capture
                            .record
                            .metadata
                            .get("_task_identity")
                            .and_then(|identity| identity.get("mutation_generation"))
                            .and_then(Value::as_u64)
                    })
                    .max()
                    .unwrap_or(0),
            ),
            todos: parking_lot::RwLock::new(None),
        }))
    }

    /// Explicit recovery retires the original canonical binding; it cannot replace another run's list.
    pub(crate) fn release_reconciled_todos(&self) -> Result<()> {
        let mut todos = self.todos.write();
        if let Some((store, binding)) = todos.as_ref()
            && store.list_for_run(binding).is_some()
            && !store.release_run(binding)
        {
            return Err(TaskRunStorageError::Conflict.into());
        }
        *todos = None;
        Ok(())
    }

    /// Returns the immutable expiry; pause/resume must never derive it from a later admission.
    pub(crate) fn expiry(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.acknowledged
            .read()
            .payload
            .get("clocks")
            .and_then(|clocks| clocks.get("expires_at"))
            .and_then(|expiry| serde_json::from_value(expiry.clone()).ok())
    }

    /// Captures immutable profile policy for interaction bridges without exposing mutable controller authority.
    pub(crate) fn lifecycle_profile(&self) -> AutonomyProfile {
        self.lifecycle.read().0.clone()
    }

    /// Reads the selected interaction policy without permitting controller instructions to grant approval.
    pub(crate) fn task_interaction_policy(&self) -> Option<InteractionAction> {
        self.lifecycle
            .read()
            .0
            .hitl
            .as_ref()
            .and_then(|policy| policy.on_approval_required)
    }

    /// Recomputes cumulative elapsed time; overlapping child/provider durations are not summed.
    pub(crate) fn active_millis(&self) -> Result<u64> {
        self.initial_active_millis
            .checked_add(
                u64::try_from(self.started.elapsed().as_millis())
                    .map_err(|_| AgentError::Config("active clock overflow".into()))?,
            )
            .ok_or_else(|| AgentError::Config("active clock overflow".into()))
    }

    /// Identifies the coordinating runtime within an already-admitted private execution scope.
    pub(crate) fn is_coordinator(&self, runtime_id: &str) -> bool {
        self.acknowledged.read().key.agent_id == runtime_id
    }

    /// Interrupted intervals consume admission allowance but are not folded into the persisted active counter twice.
    pub(crate) fn consumed_active_millis(&self) -> Result<u64> {
        self.active_millis()?
            .checked_add(self.interrupted_millis)
            .ok_or_else(|| AgentError::Config("recovered active clock overflow".into()))
    }

    /// Stops admission even if a legacy optional-feature error handler swallows the original error.
    pub(crate) fn stop(&self, reason: &str) {
        let mut stop = self.stop.write();
        if stop.is_none() {
            *stop = Some(reason.into());
        }
    }

    /// Cancellation remains distinct from unknown effect outcomes and terminal storage status.
    pub(crate) fn stop_reason(&self) -> Option<String> {
        if self.cancellation.load(Ordering::Acquire) {
            Some("cancel_requested".into())
        } else {
            self.stop.read().clone()
        }
    }

    /// Serializes a retained-owner cancellation request with admission and signals only after the CAS is acknowledged.
    pub(crate) async fn request_cancel(&self) -> Result<TaskRunSummary> {
        let _serial = self.serial.lock().await;
        let snapshot = self.load_owned_locked(&_serial).await?;
        if snapshot.cancel_requested {
            return Ok(snapshot.summary());
        }
        let saved = self
            .store
            .mutate(
                &self.run_id,
                &TaskRunMutation::RequestCancel {
                    expected_revision: snapshot.revision,
                },
            )
            .await?;
        self.revision.store(saved.revision, Ordering::Release);
        self.acknowledge_snapshot(&saved);
        self.cancellation.store(true, Ordering::Release);
        Ok(saved.summary())
    }

    /// Returns the last acknowledged state for retaining pause authority without reading an unowned claim again.
    pub(crate) fn acknowledged_snapshot(&self) -> TaskRunSnapshot {
        self.acknowledged.read().clone()
    }

    /// Copies a lifecycle decision before later executor admission; it can only narrow grants.
    pub(crate) fn set_lifecycle(&self, profile: &AutonomyProfile, state: &LifecycleState) {
        *self.lifecycle.write() = (profile.clone(), state.clone());
    }

    /// Child operation cursors share the coordinator cycle rather than restarting their own allowance.
    pub(crate) fn current_cycle(&self) -> u64 {
        self.scope.read().as_ref().map_or(0, |scope| scope.cycle)
    }

    /// Installs the controller's immutable capture identity before dispatching this cycle.
    pub(crate) fn set_scope(&self, scope: &EvaluationScope) {
        *self.scope.write() = Some(scope.clone());
    }

    /// Retains only the canonical run binding; checkpoint copies never authorize live progress-gated work.
    pub(crate) fn install_todos(
        &self,
        store: ai_agents_tools::TodoStore,
        binding: ai_agents_tools::TodoRunBinding,
    ) {
        *self.todos.write() = Some((store, binding));
    }

    /// Task-local todo authority is captured before polling, including across framework spawn boundaries.
    pub(crate) fn task_todos(
        &self,
    ) -> Option<(ai_agents_tools::TodoStore, ai_agents_tools::TodoRunBinding)> {
        self.todos.read().clone()
    }

    /// Progress exemptions waive only this predicate and require private host authority for validation work.
    pub(crate) fn progress_allows(
        &self,
        args: &Value,
        ctx: &ai_agents_core::ToolExecutionContext,
    ) -> bool {
        let lifecycle = self.lifecycle.read();
        let Some(progress) = &lifecycle.0.progress else {
            return true;
        };
        if ctx.canonical_id == progress.todo_tool.as_deref().unwrap_or("todo")
            || ctx.canonical_id == "ask_user"
        {
            return true;
        }
        let request = ai_agents_core::ToolExecutionRequest::new(
            &ctx.call_id,
            &ctx.requested_name,
            args.clone(),
            ctx.source.clone(),
        );
        if super::validation_extra_grant(
            &self.acknowledged.read().key.agent_id,
            &request,
            &ctx.canonical_id,
            args,
        )
        .is_some()
        {
            return true;
        }
        if progress
            .enforce_for_tools
            .as_ref()
            .is_some_and(|tools| !tools.contains(&ctx.canonical_id))
        {
            return true;
        }
        if progress.require_todos != Some(true) && progress.require_active_todo != Some(true) {
            return true;
        }
        let todos = self.todos.read();
        let Some((store, binding)) = todos.as_ref() else {
            return false;
        };
        let Some(items) = store.list_for_run(binding) else {
            return false;
        };
        (progress.require_todos != Some(true) || !items.is_empty())
            && (progress.require_active_todo != Some(true)
                || items
                    .iter()
                    .filter(|item| item.status == ai_agents_tools::TodoStatus::InProgress)
                    .count()
                    == 1)
    }

    /// Managed mutation attempts invalidate old validation even when their final effect is uncertain.
    pub(crate) fn mutation_generation(&self) -> u64 {
        self.mutation_generation.load(Ordering::Acquire)
    }

    /// A finalized shared-executor record is captured once with private run/cycle authority.
    pub(crate) async fn capture_tool_record(
        &self,
        record: &ai_agents_core::ToolExecutionRecord,
        runtime_id: &str,
    ) -> Result<()> {
        let mut scope = self
            .scope
            .read()
            .clone()
            .ok_or_else(|| AgentError::Config("task evidence scope missing".into()))?;
        if record.executed
            && record
                .metadata
                .get("classification")
                .and_then(|value| value.get("read_only"))
                .and_then(Value::as_bool)
                != Some(true)
        {
            self.mutation_generation
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                    generation.checked_add(1)
                })
                .map_err(|_| AgentError::Config("mutation generation overflow".into()))?;
        }
        scope.mutation_generation = self.mutation_generation();
        let operation = super::current_child_operation();
        self.update(|payload| {
            if payload.evidence.tool_calls.iter().any(|capture| {
                capture.record.call_id == record.call_id
                    && capture
                        .record
                        .metadata
                        .get("_task_runtime_id")
                        .and_then(Value::as_str)
                        == Some(runtime_id)
                    && capture.record.metadata.get("_task_child_operation")
                        == Some(&serde_json::to_value(&operation).unwrap_or(Value::Null))
            }) {
                return Ok(());
            }
            let sequence = payload.evidence.tool_calls.len() as u64;
            let identity = EvidenceIdentity {
                key: scope.key.clone(),
                objective_revision: scope.objective_revision,
                cycle: scope.cycle,
                sequence,
                stage: scope.stage.clone(),
                attempt: None,
                config_identity: String::new(),
                target: None,
                target_revision: None,
                mutation_generation: scope.mutation_generation,
            };
            let mut record = record.clone();
            record
                .metadata
                .insert("_task_runtime_id".into(), json!(runtime_id));
            record
                .metadata
                .insert("_task_child_operation".into(), json!(operation));
            record
                .metadata
                .insert("_task_identity".into(), serde_json::to_value(identity)?);
            payload.evidence.tool_calls.push(TaskToolEvidence {
                run_id: self.run_id.clone(),
                sequence,
                stage: scope.stage,
                attempt: None,
                origin: "shared_executor".into(),
                record,
            });
            Ok(())
        })
        .await?;
        Ok(())
    }

    /// Final invocation checks the current stage, not a predicate cached before approval.
    pub(crate) fn allows_tool(&self, tool: &str) -> bool {
        let state = self.lifecycle.read();
        state.1.allows_tool(&state.0, tool)
    }

    /// All controller, journal and invocation writes share this short CAS boundary.
    pub(crate) async fn update(
        &self,
        change: impl FnOnce(&mut TaskCheckpointPayload) -> Result<()>,
    ) -> Result<TaskRunSnapshot> {
        let _serial = self.serial.lock().await;
        let snapshot = self.load_owned_locked(&_serial).await?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        change(&mut payload)?;
        payload.clocks.active_millis = self.active_millis()?;
        self.save(snapshot, payload, TaskRunStatus::Running).await
    }

    /// Owned reads share the writer boundary so a sibling acknowledgement cannot invalidate a snapshot while storage is returning it.
    pub(crate) async fn load_owned(&self) -> Result<TaskRunSnapshot> {
        let serial = self.serial.lock().await;
        Box::pin(self.load_owned_locked(&serial)).await
    }

    /// Callers already holding the exact journal mutex avoid recursive acquisition without weakening ownership or cancellation checks.
    pub(crate) async fn load_owned_locked(
        &self,
        serial: &tokio::sync::MutexGuard<'_, ()>,
    ) -> Result<TaskRunSnapshot> {
        if !std::ptr::eq(tokio::sync::MutexGuard::mutex(serial), self.serial.as_ref()) {
            return Err(AgentError::Config(
                "owned task read requires its own journal guard".into(),
            ));
        }
        let snapshot = self
            .store
            .load(&self.run_id)
            .await
            .inspect_err(|_| self.stop("storage_failure"))?
            .ok_or_else(|| {
                self.stop("storage_failure");
                AgentError::from(TaskRunStorageError::NotFound)
            })?;
        if snapshot.cancel_requested {
            self.cancellation.store(true, Ordering::Release);
            let acknowledged = self.acknowledged.read().clone();
            if snapshot.status == TaskRunStatus::Running
                && snapshot.owner_token == acknowledged.owner_token
                && snapshot.key == acknowledged.key
                && snapshot.actor_id == acknowledged.actor_id
                && snapshot.created_at == acknowledged.created_at
                && snapshot.payload == acknowledged.payload
                && snapshot.revision > acknowledged.revision
            {
                self.revision.store(snapshot.revision, Ordering::Release);
                *self.acknowledged.write() = snapshot.clone();
            }
        }
        if snapshot.owner_token.as_deref() != Some(&self.owner_token)
            || snapshot.revision != self.revision.load(Ordering::Acquire)
            || snapshot.status != TaskRunStatus::Running
        {
            self.stop("owner_conflict");
            return Err(TaskRunStorageError::Conflict.into());
        }
        if snapshot.cancel_requested {
            self.cancellation.store(true, Ordering::Release);
        }
        Ok(snapshot)
    }

    /// Persisted marker and final checks precede dispatch; an exact-cap result is allowed to complete.
    pub(crate) async fn admit(&self, tool: Option<&str>) -> Result<String> {
        Box::pin(self.admit_resources(tool, None, None)).await
    }

    /// Footprints are computed from the actual implementation and approved arguments while existing executor locks are held.
    /// Box the conditional resource journal before nesting child-tool polling so exact checkpoint temporaries do not multiply on the host stack.
    pub(crate) async fn admit_tool(
        &self,
        tool: &dyn ai_agents_core::Tool,
        args: &Value,
        ctx: &ai_agents_core::ToolExecutionContext,
    ) -> Result<(String, Option<ToolWriteFootprint>)> {
        if !self.progress_allows(args, ctx) {
            return Err(AgentError::Tool(
                "task work requires canonical todo progress before invocation".into(),
            ));
        }
        let footprint = if let Some(limit) = self.limits.max_declared_write_paths {
            Some(
                tool.declared_write_footprint(
                    args,
                    ctx,
                    usize::try_from(limit).unwrap_or(usize::MAX),
                )?
                .ok_or_else(|| {
                    AgentError::Tool(
                        "declared write limit requires a verified footprint binding".into(),
                    )
                })?,
            )
        } else {
            None
        };
        let id = Box::pin(self.admit_resources(Some(&ctx.canonical_id), None, footprint.as_ref()))
            .await?;
        Ok((id, footprint))
    }

    /// Atomic commitment includes settled charges and every outstanding conservative upper bound or target union.
    async fn admit_resources(
        &self,
        tool: Option<&str>,
        priced: Option<ProviderCostBound>,
        footprint: Option<&ToolWriteFootprint>,
    ) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        Box::pin(self.update(|payload| {
            self.check(payload)?;
            if tool.is_none()
                && let Some(limit) = payload.limits.max_micro_usd
            {
                let bound = priced.as_ref().ok_or_else(|| {
                    self.stop("priced_capability_missing");
                    AgentError::Config(
                        "priced request has no verified provider accounting binding".into(),
                    )
                })?;
                bound.validate(&bound.request_id)?;
                let pin_id = format!("priced-provider:{}", bound.provider_identity);
                let config = json!({"pricing_identity":bound.pricing_identity});
                if let Some(pin) = payload.adapters.iter().find(|binding| binding.id == pin_id) {
                    if pin.config != config {
                        self.stop("pricing_identity_changed");
                        return Err(AgentError::Config("run pricing identity changed".into()));
                    }
                } else {
                    payload.adapters.push(TaskAdapterCheckpoint {
                        id: pin_id,
                        adapter: "runtime.priced_provider".into(),
                        contract_version: 1,
                        config,
                        state: Value::Null,
                    });
                }
                let committed = payload
                    .reservations
                    .iter()
                    .filter(|reservation| {
                        matches!(
                            reservation.state,
                            TaskEffectState::Reserved
                                | TaskEffectState::Dispatched
                                | TaskEffectState::Suspended
                                | TaskEffectState::Uncertain
                        )
                    })
                    .try_fold(payload.counters.charged_micro_usd, |total, reservation| {
                        total.checked_add(
                            reservation
                                .reserved_micro_usd
                                .saturating_sub(reservation.charged_micro_usd),
                        )
                    })
                    .and_then(|total| total.checked_add(bound.max_micro_usd));
                if committed.is_none_or(|total| total > limit) {
                    self.stop("priced_capacity");
                    return Err(AgentError::Other(
                        "autonomy priced capacity exhausted".into(),
                    ));
                }
                payload.adapters.push(TaskAdapterCheckpoint {
                    id: format!("priced-attempt:{id}"),
                    adapter: "runtime.priced_request".into(),
                    contract_version: 1,
                    config: serde_json::to_value(bound)?,
                    state: Value::Null,
                });
            }
            if let Some(footprint) = footprint {
                if footprint.binding_identity.trim().is_empty()
                    || footprint.targets.iter().any(|target| target.is_empty())
                {
                    return Err(AgentError::Config("invalid footprint binding".into()));
                }
                let mut targets: std::collections::BTreeSet<_> =
                    payload.declared_write_targets.iter().cloned().collect();
                targets.extend(footprint.targets.iter().cloned());
                if payload
                    .limits
                    .max_declared_write_paths
                    .is_some_and(|limit| targets.len() > limit as usize)
                {
                    self.stop("write_capacity");
                    return Err(AgentError::Tool(
                        "autonomy declared write capacity exhausted".into(),
                    ));
                }
                payload.declared_write_targets = targets.into_iter().collect();
                payload.adapters.push(TaskAdapterCheckpoint {
                    id: format!("footprint-attempt:{id}"),
                    adapter: "runtime.write_footprint".into(),
                    contract_version: 1,
                    config: serde_json::to_value(footprint)?,
                    state: Value::Null,
                });
            }
            if let Some(tool) = tool {
                if !self.allows_tool(tool) {
                    return Err(AgentError::Tool("autonomy lifecycle denies tool".into()));
                }
                if payload.counters.tool_attempts >= u64::from(payload.limits.max_tool_calls)
                    || (tool == "command"
                        && payload.counters.command_attempts
                            >= u64::from(payload.limits.max_command_calls))
                {
                    self.stop("tool_capacity");
                    return Err(AgentError::Other("autonomy tool capacity exhausted".into()));
                }
                payload.counters.tool_attempts += 1;
                if tool == "command" {
                    payload.counters.command_attempts += 1;
                }
            } else {
                if payload.counters.llm_attempts >= u64::from(payload.limits.max_llm_calls) {
                    self.stop("llm_capacity");
                    return Err(AgentError::Other("autonomy LLM capacity exhausted".into()));
                }
                payload.counters.llm_attempts += 1;
            }
            payload.adapters.push(TaskAdapterCheckpoint {
                id: format!("invocation:{id}"),
                adapter: "runtime.invocation".into(),
                contract_version: 1,
                config: json!({"tool":tool}),
                state: Value::Null,
            });
            payload.reservations.push(TaskReservation {
                id: id.clone(),
                state: TaskEffectState::Dispatched,
                reserved_micro_usd: priced.as_ref().map_or(0, |bound| bound.max_micro_usd),
                charged_micro_usd: 0,
                write_targets: footprint
                    .map_or_else(Vec::new, |footprint| footprint.targets.clone()),
                result: None,
            });
            Ok(())
        }))
        .await?;
        // Reload under serialization so a sibling settlement cannot turn an owned revision into a spurious conflict.
        let admission_check = {
            let _serial = self.serial.lock().await;
            let snapshot = self.load_owned_locked(&_serial).await?;
            if snapshot.cancel_requested {
                self.cancellation.store(true, Ordering::Release);
            }
            let payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload)?;
            self.check(&payload)
        };
        if let Err(error) = admission_check {
            self.settle(&id, json!({"not_invoked":true}), false).await?;
            return Err(error);
        }
        Ok(id)
    }

    /// Exact results are acknowledged before publication; uncertain effects remain non-replayable.
    pub(crate) async fn settle(&self, id: &str, result: Value, uncertain: bool) -> Result<()> {
        self.settle_with_cost(id, result, uncertain, None).await
    }

    /// Trustworthy request-matched billing releases unused price commitment; errors and missing usage retain the full bound.
    async fn settle_with_cost(
        &self,
        id: &str,
        result: Value,
        uncertain: bool,
        settlement: Option<ProviderCostSettlement>,
    ) -> Result<()> {
        if uncertain {
            self.stop("uncertain_effect");
        }
        self.update(|payload| {
            let bound = payload
                .adapters
                .iter()
                .find(|binding| binding.id == format!("priced-attempt:{id}"))
                .map(|binding| serde_json::from_value::<ProviderCostBound>(binding.config.clone()))
                .transpose()?;
            if let Some(settlement) = &settlement {
                let bound = bound.as_ref().ok_or_else(|| {
                    AgentError::Config("settlement lacks a priced admission".into())
                })?;
                validate_cost_settlement(bound, settlement)?;
            }
            let reservation = payload
                .reservations
                .iter_mut()
                .find(|r| r.id == id)
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            if !matches!(
                reservation.state,
                TaskEffectState::Dispatched | TaskEffectState::Suspended
            ) {
                return Err(TaskRunStorageError::Conflict.into());
            }
            reservation.state = if uncertain {
                TaskEffectState::Uncertain
            } else {
                TaskEffectState::Completed
            };
            if !uncertain {
                let charge = if result.get("not_invoked") == Some(&Value::Bool(true)) {
                    0
                } else {
                    settlement
                        .as_ref()
                        .map_or(reservation.reserved_micro_usd, |settlement| {
                            settlement.charged_micro_usd
                        })
                };
                reservation.charged_micro_usd = charge;
                payload.counters.charged_micro_usd = payload
                    .counters
                    .charged_micro_usd
                    .checked_add(charge)
                    .ok_or_else(|| AgentError::Config("priced settlement overflow".into()))?;
            }
            let not_invoked = !uncertain && result.get("not_invoked") == Some(&Value::Bool(true));
            let released_targets = reservation.write_targets.clone();
            reservation.result = Some(result);
            if not_invoked {
                let invocation = payload
                    .adapters
                    .iter()
                    .find(|binding| binding.id == format!("invocation:{id}"))
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                if let Some(tool) = invocation.config.get("tool").and_then(Value::as_str) {
                    payload.counters.tool_attempts = payload
                        .counters
                        .tool_attempts
                        .checked_sub(1)
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                    if tool == "command" {
                        payload.counters.command_attempts = payload
                            .counters
                            .command_attempts
                            .checked_sub(1)
                            .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                    }
                } else {
                    payload.counters.llm_attempts = payload
                        .counters
                        .llm_attempts
                        .checked_sub(1)
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                }
                payload.declared_write_targets.retain(|target| {
                    !released_targets.contains(target)
                        || payload.reservations.iter().any(|reservation| {
                            reservation.write_targets.contains(target)
                                && !(reservation.state == TaskEffectState::Completed
                                    && reservation
                                        .result
                                        .as_ref()
                                        .and_then(|result| result.get("not_invoked"))
                                        == Some(&Value::Bool(true)))
                        })
                });
            }
            Ok(())
        })
        .await?;
        Ok(())
    }

    /// Live run and inherited composition deadlines intersect; a pause or nested callback cannot refill either allowance.
    pub(crate) fn remaining_duration(&self) -> std::time::Duration {
        let active = std::time::Duration::from_millis(
            self.limits.max_active_time_seconds.saturating_mul(1000),
        )
        .saturating_sub(std::time::Duration::from_millis(
            self.consumed_active_millis().unwrap_or(u64::MAX),
        ));
        let remaining = self.expiry_projection.map_or(active, |expiry| {
            active.min(expiry.saturating_duration_since(Instant::now()))
        });
        super::composition::current_composition_deadline().map_or(remaining, |deadline| {
            remaining.min(deadline.saturating_duration_since(Instant::now()))
        })
    }

    /// Journal successors update the same acknowledged snapshot used to recognize cancellation-only revisions.
    pub(crate) fn acknowledge_snapshot(&self, snapshot: &TaskRunSnapshot) {
        *self.acknowledged.write() = snapshot.clone();
    }

    /// Run and composition windows are cooperative admission checks; synchronous host work may overrun and remains charged.
    /// An expired parent dispatch cannot admit a fresh provider/tool request while already dispatched work still settles.
    pub(crate) fn check(&self, payload: &TaskCheckpointPayload) -> Result<()> {
        // Captured task scopes must not admit new work after their foreground owner is dropped.
        if let Err(error) = self.owner.check() {
            self.stop("owner_abandoned");
            return Err(error);
        }
        if let Some(reason) = self.stop_reason() {
            return Err(AgentError::Other(format!("autonomy stopped: {reason}")));
        }
        if super::composition::current_composition_deadline()
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.stop("composition_timeout");
            return Err(AgentError::Other("composition deadline exceeded".into()));
        }
        if self.remaining_duration().is_zero()
            || self.consumed_active_millis()?
                >= payload.limits.max_active_time_seconds.saturating_mul(1000)
            || payload
                .clocks
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        {
            self.stop("time_violation");
            return Err(AgentError::Other("autonomy deadline exceeded".into()));
        }
        Ok(())
    }

    /// Final cancellation/time checks and CAS share owner admission; failed persistence never publishes success.
    /// A conservative final millisecond closes an active interval even when its last acknowledged write occurred in the same tick.
    pub(crate) async fn terminal(
        &self,
        mut status: TaskRunStatus,
        reason: String,
    ) -> Result<TaskRunSnapshot> {
        let _serial = self.serial.lock().await;
        let snapshot = self.load_owned_locked(&_serial).await?;
        let mut payload: TaskCheckpointPayload = serde_json::from_value(snapshot.payload.clone())?;
        payload.clocks.active_millis = self.active_millis()?.max(
            payload
                .clocks
                .active_millis
                .checked_add(1)
                .ok_or_else(|| AgentError::Config("terminal active clock overflow".into()))?,
        );
        let uncertain = self.participants.unsettled()
            || payload.children.iter().any(|child| child.result.is_none())
            || payload.reservations.iter().any(|r| {
                matches!(
                    r.state,
                    TaskEffectState::Dispatched
                        | TaskEffectState::Suspended
                        | TaskEffectState::Uncertain
                )
            });
        if uncertain {
            status = TaskRunStatus::RecoveryRequired;
        } else if snapshot.cancel_requested || self.cancellation.load(Ordering::Acquire) {
            status = TaskRunStatus::Cancelled;
        } else if payload
            .clocks
            .active_millis
            .checked_add(payload.clocks.interrupted_interval_millis)
            .ok_or_else(|| AgentError::Config("terminal active clock overflow".into()))?
            > payload.limits.max_active_time_seconds.saturating_mul(1000)
            || payload
                .clocks
                .expires_at
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        {
            status = TaskRunStatus::LimitReached;
        }
        if status == TaskRunStatus::LimitReached
            && payload
                .settings
                .on_limit
                .as_ref()
                .is_some_and(|policy| policy.action == Some(LimitAction::Fail))
        {
            status = TaskRunStatus::Failed;
        }
        if !uncertain && status == TaskRunStatus::Completed && self.stop_reason().is_some() {
            status = TaskRunStatus::Failed;
        }
        payload.clocks.active_interval_started_at = None;
        payload.stop_reason = Some(if uncertain {
            "uncertain_effect".into()
        } else {
            reason
        });
        self.save(snapshot, payload, status).await
    }

    /// Writes one whole validated successor without holding a database transaction over external work.
    async fn save(
        &self,
        previous: TaskRunSnapshot,
        payload: TaskCheckpointPayload,
        status: TaskRunStatus,
    ) -> Result<TaskRunSnapshot> {
        let mutation = if status == TaskRunStatus::Running {
            TaskRunMutation::Checkpoint {
                expected_revision: previous.revision,
                owner_token: self.owner_token.clone(),
                status,
                payload: serde_json::to_value(payload)?,
                release: false,
            }
        } else {
            TaskRunMutation::FinalCheckpoint {
                expected_revision: previous.revision,
                owner_token: self.owner_token.clone(),
                status,
                expires_at: payload.clocks.expires_at,
                payload: serde_json::to_value(payload)?,
                release: true,
                deadline: Instant::now()
                    .checked_add(self.remaining_duration())
                    .ok_or_else(|| AgentError::Config("final deadline overflow".into()))?,
            }
        };
        let saved = self
            .store
            .mutate(&self.run_id, &mutation)
            .await
            .inspect_err(|_| self.stop("storage_failure"))?;
        self.revision.store(saved.revision, Ordering::Release);
        self.acknowledge_snapshot(&saved);
        Ok(saved)
    }
}

#[async_trait]
impl ai_agents_core::autonomy::InvocationAdmission for RunExecution {
    /// Provider wrappers reserve every attempt, including validation and auxiliary calls.
    async fn admit_llm(&self) -> std::result::Result<String, LLMError> {
        self.admit(None)
            .await
            .map_err(|error| LLMError::Other(error.to_string()))
    }
    /// Unsupported requests stop the run before dispatch rather than escaping through recovery or fallback.
    async fn admit_priced_llm(
        &self,
        bound: Option<ProviderCostBound>,
    ) -> std::result::Result<String, LLMError> {
        self.admit_resources(None, bound, None)
            .await
            .map_err(|error| LLMError::Other(error.to_string()))
    }
    /// Price proof failures stop admission even if a legacy consumer catches the provider error.
    fn reject_priced_request(&self) {
        self.stop("priced_capability_missing");
    }
    /// Frozen wrappers request billing proof only for hard-priced runs.
    fn requires_priced_llm(&self) -> bool {
        self.limits.max_micro_usd.is_some()
    }
    /// Matching settlement shares the same conditional revision and serialization as all invocation evidence.
    async fn settle_priced_llm(
        &self,
        id: &str,
        result: Value,
        settlement: Option<ProviderCostSettlement>,
    ) -> std::result::Result<(), LLMError> {
        self.settle_with_cost(id, result, false, settlement)
            .await
            .map_err(|error| LLMError::Other(error.to_string()))
    }
    /// Store failure is an execution failure, never a model success that bypasses the journal.
    async fn settle_llm(&self, id: &str, result: Value) -> std::result::Result<(), LLMError> {
        self.settle(id, result, false)
            .await
            .map_err(|error| LLMError::Other(error.to_string()))
    }
    /// A dropped stream/provider future keeps its pre-dispatch reservation for explicit recovery.
    fn abandon_llm(&self, _id: &str) {
        self.stop("uncertain_effect");
    }
    /// Recovery must not route around a run-wide stop using another provider alias.
    fn execution_stopped(&self) -> bool {
        if self.owner.check().is_err() {
            self.stop("owner_abandoned");
        }
        if self.remaining_duration().is_zero() {
            self.stop("time_violation");
        }
        self.stop_reason().is_some()
    }
    /// Provider waits receive the same shrinking run allowance as tool attempts.
    fn remaining_duration(&self) -> Option<std::time::Duration> {
        Some(RunExecution::remaining_duration(self))
    }
}

tokio::task_local! { static RUN_EXECUTION: Arc<RunExecution>; }

/// Captures live ledger authority for explicit task-boundary propagation, not serialization.
pub(crate) fn current_execution() -> Option<Arc<RunExecution>> {
    RUN_EXECUTION.try_with(Arc::clone).ok()
}

/// Framework task boundaries must explicitly carry the captured ledger; absence preserves ordinary execution.
pub(crate) async fn scope_inherited_execution<F: std::future::Future>(
    execution: Option<Arc<RunExecution>>,
    future: F,
) -> F::Output {
    if let Some(execution) = execution {
        scope_execution(execution, future).await
    } else {
        future.await
    }
}

/// Installs the same ledger for the controller, tool executor and every frozen provider consumer.
pub(crate) async fn scope_execution<F: std::future::Future>(
    execution: Arc<RunExecution>,
    future: F,
) -> F::Output {
    let todos = execution.task_todos();
    let scoped = ai_agents_llm::scope_invocation_admission(
        execution.clone(),
        RUN_EXECUTION.scope(execution, future),
    );
    if let Some((store, binding)) = todos {
        ai_agents_tools::scope_task_todos(store, binding, scoped).await
    } else {
        scoped.await
    }
}

#[cfg(test)]
mod owned_read_tests {
    use super::*;

    struct ReadBarrierStore {
        inner: Arc<ScopedTaskRunStore>,
        block: AtomicBool,
        entered: tokio::sync::Semaphore,
        release: tokio::sync::Semaphore,
        writes: tokio::sync::Semaphore,
    }

    #[async_trait]
    impl TaskRunStore for ReadBarrierStore {
        async fn create(&self, snapshot: &TaskRunSnapshot) -> Result<()> {
            self.inner.create(snapshot).await
        }
        async fn load(&self, run_id: &str) -> Result<Option<TaskRunSnapshot>> {
            let snapshot = self.inner.load(run_id).await?;
            if self.block.swap(false, Ordering::AcqRel) {
                self.entered.add_permits(1);
                self.release.acquire().await.unwrap().forget();
            }
            Ok(snapshot)
        }
        async fn mutate(
            &self,
            run_id: &str,
            mutation: &TaskRunMutation,
        ) -> Result<TaskRunSnapshot> {
            self.writes.add_permits(1);
            self.inner.mutate(run_id, mutation).await
        }
        async fn list(&self) -> Result<Vec<TaskRunSummary>> {
            self.inner.list().await
        }
        async fn delete(&self, run_id: &str, revision: u64) -> Result<()> {
            self.inner.delete(run_id, revision).await
        }
    }

    // A real conditional store pauses delivery of a snapshot after its read, not by replacing ownership enforcement.
    async fn fixture() -> (Arc<RunExecution>, Arc<ReadBarrierStore>) {
        let snapshot = super::super::tests::checkpoint("owned-read");
        let inner = Arc::new(
            ScopedTaskRunStore::in_memory("agent".into(), None, "config-v1".into()).unwrap(),
        );
        inner.create(&snapshot).await.unwrap();
        let running = inner
            .mutate(
                "owned-read",
                &TaskRunMutation::Claim {
                    expected_revision: 0,
                    owner_token: "owner".into(),
                },
            )
            .await
            .unwrap();
        let payload: TaskCheckpointPayload =
            serde_json::from_value(running.payload.clone()).unwrap();
        let lifecycle = LifecycleState::new(&payload.settings, false).unwrap();
        let owner = RunOwner::new(
            "owned-read".into(),
            Arc::new(tokio::sync::Mutex::new(())),
            None,
        )
        .unwrap();
        let store = Arc::new(ReadBarrierStore {
            inner,
            block: AtomicBool::new(false),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
            writes: tokio::sync::Semaphore::new(0),
        });
        let execution =
            RunExecution::new(store.clone(), &running, lifecycle, owner, Instant::now()).unwrap();
        (execution, store)
    }

    // An acknowledged sibling write cannot overtake an owned read and turn the old delivered revision into a false owner conflict.
    #[tokio::test]
    async fn owned_read_serializes_snapshot_delivery_with_sibling_acknowledgement() {
        let (execution, store) = fixture().await;
        store.block.store(true, Ordering::Release);
        let reading = execution.clone();
        let reader = tokio::spawn(async move { reading.load_owned().await });
        tokio::time::timeout(std::time::Duration::from_secs(10), store.entered.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        let started = Arc::new(tokio::sync::Semaphore::new(0));
        let writing = execution.clone();
        let signalled = started.clone();
        let writer = tokio::spawn(async move {
            signalled.add_permits(1);
            writing
                .update(|payload| {
                    payload.controller_state = json!({"acknowledged":true});
                    Ok(())
                })
                .await
        });
        started.acquire().await.unwrap().forget();
        assert!(store.writes.try_acquire().is_err());
        store.release.add_permits(1);
        let snapshot = tokio::time::timeout(std::time::Duration::from_secs(10), reader)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.revision, 1);
        let written = tokio::time::timeout(std::time::Duration::from_secs(10), writer)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(written.revision, 2);
        assert_eq!(execution.load_owned().await.unwrap().revision, 2);
        assert!(execution.stop_reason().is_none());
    }

    // A guard for another run's mutex cannot opt out of this execution's owned-read serialization.
    #[tokio::test]
    async fn owned_read_rejects_foreign_serialization_guard() {
        let (execution, _) = fixture().await;
        let foreign = tokio::sync::Mutex::new(());
        let guard = foreign.lock().await;
        assert!(execution.load_owned_locked(&guard).await.is_err());
    }
}
