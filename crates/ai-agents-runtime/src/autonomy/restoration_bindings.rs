//! Typed reconstruction materializes only validated executable cursors, never strings found in arbitrary tool results.

use super::*;
use crate::autonomy::{
    controller::{ControllerContinuation, ControllerLimit, ControllerProposal},
    interaction::{TaskInteraction, TaskInteractionKind},
    location::{TaskActionsLocation, TaskLocation, TaskSkillLocation},
    message::MessageState,
    suspension::TaskBatchState,
};

pub(super) struct Rebinding<'a> {
    pub manifest: &'a TaskRestoreManifest,
    pub graph: &'a TaskRestoreGraph,
    pub revision: u64,
    pub requests: BTreeMap<String, String>,
}
impl Rebinding<'_> {
    // Every unresolved request gets a fresh identity; historical completed authorizations remain audit data behind fresh review.
    fn request(&mut self, old: &str) -> String {
        // Message-frame coordinates are operation identities, not human-response authority; native batch references must keep them.
        if self.manifest.active_frames.iter().any(|frame| frame == old) {
            return old.into();
        }
        self.requests
            .entry(old.into())
            .or_insert_with(|| uuid::Uuid::new_v4().to_string())
            .clone()
    }
    // Numeric equality in a new process is not equivalence: verify old lineage then bind the actual reconstructed generation.
    fn generations(
        &self,
        runtime: &str,
    ) -> Result<(TaskRestoreGenerations, TaskRestoreGenerations)> {
        let old = self
            .manifest
            .nodes
            .get(runtime)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .generations
            .clone();
        let new = self
            .graph
            .runtimes
            .get(runtime)
            .ok_or(TaskRunStorageError::InvalidCheckpoint)?
            .task_restore_generations();
        Ok((old, new))
    }
    // Binding translation is allowed only for a location which matched the authoritative old node at the safe checkpoint.
    fn state(&self, runtime: &str, state: &mut Option<u64>) -> Result<()> {
        let (old, new) = self.generations(runtime)?;
        if *state != old.state {
            return Err(AgentError::Config(
                "saved cursor generation differs from source image".into(),
            ));
        }
        *state = new.state;
        Ok(())
    }
    // Pending metadata is rebuilt according to its framework-owned schema; domain arguments and result strings are never rewritten.
    pub fn pending(&mut self, pending: &mut TaskPendingRequest, runtime: &str) -> Result<()> {
        pending.id = self.request(&pending.id);
        pending.issued_revision = self.revision;
        let value = &mut pending.reviewed_action;
        if value.get("call_id").is_some() && value.get("trigger").is_some() {
            let mut approval: crate::autonomy::suspension::BatchApproval =
                serde_json::from_value(value.clone())?;
            self.approval(&mut approval, runtime)?;
            *value = serde_json::to_value(approval)?;
        } else if value
            .get("request")
            .is_some_and(|request| request.get("issued_revision").is_some())
        {
            let mut child: TaskPendingRequest = serde_json::from_value(value["request"].clone())?;
            let child_runtime = value
                .get("child_operation")
                .and_then(Value::as_str)
                .and_then(|operation| self.manifest.targets.get(operation))
                .map(|target| target.runtime.clone())
                .unwrap_or_else(|| runtime.into());
            self.pending(&mut child, &child_runtime)?;
            value["request"] = serde_json::to_value(child)?;
        }
        Ok(())
    }
    // Capacity pauses retain their existing resource requirement and exact skill position, not a fresh allowance.
    fn limit(&mut self, limit: &mut ControllerLimit, runtime: &str) -> Result<()> {
        if let Some(skill) = &mut limit.skill {
            self.skill(skill, runtime)?;
        }
        Ok(())
    }
    // Only request identity and typed capacity state change; the reviewed effective arguments remain exact.
    fn approval(
        &mut self,
        approval: &mut crate::autonomy::suspension::BatchApproval,
        runtime: &str,
    ) -> Result<()> {
        approval.request_id = self.request(&approval.request_id);
        if let Some(mut limit) = approval.capacity()? {
            self.limit(&mut limit, runtime)?;
            approval.trigger["capacity"] = serde_json::to_value(limit)?;
        }
        Ok(())
    }
    // Transition position and completed actions remain immutable; fresh runtime epochs replace only the verified old binding.
    fn actions(&mut self, actions: &mut TaskActionsLocation, runtime: &str) -> Result<()> {
        let (old, new) = self.generations(runtime)?;
        if actions.transition.expected_generation != old.state.unwrap_or(0)
            || actions.transition.expected_epoch != old.epoch
        {
            return Err(AgentError::Config(
                "state action reconstruction binding differs".into(),
            ));
        }
        if actions.transition.execution_id.is_empty() {
            actions.transition.execution_id = format!(
                "restored:{}:{}:{}",
                self.manifest.revision, runtime, actions.transition.expected_generation
            );
        }
        actions.transition.expected_generation = new.state.unwrap_or(0);
        actions.transition.expected_epoch = new.epoch;
        Ok(())
    }
    // A restored script resumes its exact cursor and rendered values; loaded definition compatibility was validated separately.
    fn skill(&mut self, skill: &mut TaskSkillLocation, runtime: &str) -> Result<()> {
        self.state(runtime, &mut skill.state_generation)?;
        if let Some(actions) = &mut skill.actions {
            self.actions(actions, runtime)?;
        }
        Ok(())
    }
    // Restored manager state and its cursor use the same fresh lineage without rerunning clarification/confirmation semantics.
    fn interaction(&mut self, interaction: &mut TaskInteraction, runtime: &str) -> Result<()> {
        let (old, new) = self.generations(runtime)?;
        self.state(runtime, &mut interaction.state_generation)?;
        if interaction.epoch != old.epoch || interaction.runtime_control_version != old.control {
            return Err(AgentError::Config(
                "interaction reconstruction binding differs".into(),
            ));
        }
        interaction.epoch = new.epoch;
        interaction.runtime_control_version = new.control;
        if let Some(request) = &mut interaction.request_id {
            *request = self.request(request);
        }
        if let TaskInteractionKind::Clarification { manager, .. } = &mut interaction.kind {
            manager["origin_state_generation"] = serde_json::to_value(new.state)?;
        }
        Ok(())
    }
    // Rebinds one exact saved batch. Prior authorization receipts cannot execute until the separate fresh reconstruction review succeeds.
    pub fn batch(&mut self, batch: &mut TaskBatchState, runtime: &str) -> Result<()> {
        for approval in &mut batch.approvals {
            self.approval(approval, runtime)?;
        }
        if let Some(location) = &mut batch.location {
            match location {
                TaskLocation::Skill(skill) => self.skill(skill, runtime)?,
                TaskLocation::Actions(actions) => self.actions(actions, runtime)?,
                TaskLocation::Observation(_) => {}
            }
        }
        if let Some(interaction) = &mut batch.interaction {
            self.interaction(interaction, runtime)?;
        }
        batch.validate()
    }
    // Existing parent algorithms retain their inputs, choices, ordered outcomes and predecessor edges.
    pub fn frame(&mut self, frame: &mut DelegateFrame) -> Result<()> {
        match frame.dispatch {
            crate::autonomy::composition::CompositionDispatch::ToolBatch => {
                let mut batch: TaskBatchState = serde_json::from_value(frame.cursor.clone())?;
                self.batch(&mut batch, &frame.runtime_id)?;
                frame.cursor = serde_json::to_value(batch)?;
            }
            crate::autonomy::composition::CompositionDispatch::ToolMessage => {
                let mut state: MessageState = serde_json::from_value(frame.cursor.clone())?;
                let (old, new) = self.generations(&frame.runtime_id)?;
                self.state(&frame.runtime_id, &mut state.state_generation)?;
                if state.record.registry_version != old.tools
                    || state.record.runtime_config_version != old.control
                    || state.record.policy_version != old.policy
                {
                    return Err(AgentError::Config(
                        "message reconstruction versions differ".into(),
                    ));
                }
                state.record.registry_version = new.tools;
                state.record.runtime_config_version = new.control;
                state.record.policy_version = new.policy;
                if state
                    .record
                    .metadata
                    .contains_key("state_generation_snapshot")
                {
                    state.record.metadata.insert(
                        "state_generation_snapshot".into(),
                        serde_json::to_value(new.state)?,
                    );
                }
                if let Some(batch) = &mut state.batch {
                    self.batch(batch, &frame.runtime_id)?;
                }
                frame.cursor = serde_json::to_value(state)?;
            }
            _ => {}
        }
        Ok(())
    }
    // Coordinating and leaf request identities are renewed consistently across duplicated exact group representations.
    pub fn group(&mut self, group: &mut TaskGroupState) -> Result<()> {
        group.request_id = self.request(&group.request_id);
        self.pending(&mut group.child_request, &group.child_runtime_id)?;
        self.batch(&mut group.batch, &group.child_runtime_id)?;
        for leaf in &mut group.parked {
            self.pending(&mut leaf.request, &leaf.runtime_id)?;
            self.batch(&mut leaf.batch, &leaf.runtime_id)?;
        }
        self.frame(&mut group.frame)?;
        for frame in group.frames.values_mut() {
            self.frame(frame)?;
        }
        Ok(())
    }
    // Observation attempts, completed results and request digests remain unchanged; only unresolved transport identities are renewed.
    fn driver(&mut self, driver: &mut ValidationDriverState, runtime: &str) -> Result<()> {
        if driver.in_flight.is_some() {
            return Err(AgentError::Config(
                "uncertain observation cannot be reconstructed".into(),
            ));
        }
        if let Some(wait) = &mut driver.wait {
            wait.request_id = self.request(&wait.request_id);
        }
        if let Some(parked) = &mut driver.parked {
            self.pending(&mut parked.pending, runtime)?;
            if let Some(value) = &mut parked.group {
                let mut group: TaskGroupState = serde_json::from_value(value.clone())?;
                self.group(&mut group)?;
                *value = serde_json::to_value(&group)?;
                parked.pending.id = group.request_id.clone();
                parked.pending.reviewed_action = json!({"group_id":group.frame.id,"child_operation":group.child_operation,"request":group.child_request});
            }
        }
        Ok(())
    }
    // Host proposals keep exact meaning while their review identities and current runtime binding are freshly admitted.
    pub fn controller(
        &mut self,
        controller: &mut ControllerContinuation,
        runtime: &str,
    ) -> Result<()> {
        if let Some(driver) = &mut controller.driver {
            self.driver(driver, runtime)?;
        }
        if let Some(review) = &mut controller.review {
            let (old, new) = self.generations(runtime)?;
            self.state(runtime, &mut review.state_generation)?;
            if review.control_binding != (old.control, old.epoch) {
                return Err(AgentError::Config(
                    "controller reconstruction binding differs".into(),
                ));
            }
            review.id = self.request(&review.id);
            review.control_binding = (new.control, new.epoch);
            match &mut review.proposal {
                ControllerProposal::Question {
                    limit: Some(limit), ..
                }
                | ControllerProposal::Capacity { limit } => self.limit(limit, runtime)?,
                _ => {}
            }
        }
        Ok(())
    }
    // Native histories and completed template inputs remain byte-for-byte data; only the pending transport identity changes.
    pub fn runtime(&mut self, runtime: &mut TaskRuntimeCheckpoint) {
        if let TaskContinuation::Suspended { request_id, .. } = &mut runtime.continuation {
            *request_id = self.request(request_id);
        }
    }

    /// Materializes the existing typed drivers rather than rewriting arbitrary JSON or replaying a call stack.
    pub fn payload(&mut self, payload: &mut TaskCheckpointPayload) -> Result<()> {
        let root = self.manifest.root.clone();
        for adapter in &mut payload.adapters {
            match adapter.adapter.as_str() {
                "runtime.delegate" => {
                    let mut frame: DelegateFrame = serde_json::from_value(adapter.state.clone())?;
                    if self.manifest.active_frames.contains(&frame.id) {
                        self.frame(&mut frame)?;
                        adapter.state = serde_json::to_value(frame)?;
                    }
                }
                "runtime.tool_batch" | "runtime.child_batch" => {
                    let runtime = adapter
                        .config
                        .get("runtime_id")
                        .and_then(Value::as_str)
                        .unwrap_or(&root)
                        .to_string();
                    let active = if adapter.adapter == "runtime.child_batch" {
                        adapter
                            .config
                            .get("operation")
                            .and_then(Value::as_str)
                            .is_some_and(|operation| {
                                payload.children.iter().any(|child| {
                                    child.child_id == operation
                                        && child.pending.as_ref().is_some_and(|pending| {
                                            pending.reviewed_action.get("frame_id").is_none()
                                        })
                                })
                            })
                    } else {
                        matches!(
                            payload.runtime.continuation,
                            TaskContinuation::Suspended { .. }
                        ) && payload.pause_reason.as_deref() != Some("child_interaction")
                    };
                    if !active {
                        continue;
                    }
                    let mut batch: TaskBatchState = serde_json::from_value(adapter.state.clone())?;
                    self.batch(&mut batch, &runtime)?;
                    adapter.state = serde_json::to_value(batch)?;
                }
                "runtime.group" | "runtime.child_group" => {
                    if payload.pause_reason.as_deref() != Some("child_interaction") {
                        continue;
                    }
                    let mut group: TaskGroupState = serde_json::from_value(adapter.state.clone())?;
                    self.group(&mut group)?;
                    adapter.state = serde_json::to_value(group)?;
                }
                // Skill/action journal adapters remain audit history; executable locations are rebound through their actual batch/controller.
                "runtime.skill" | "runtime.actions" => {}
                "runtime.validation"
                    if adapter.id == "runtime.validation"
                        && adapter.state.get("retired_request").is_none() =>
                {
                    let mut controller: ControllerContinuation =
                        serde_json::from_value(adapter.state.clone())?;
                    self.controller(&mut controller, &root)?;
                    adapter.state = serde_json::to_value(controller)?;
                }
                _ => {}
            }
        }
        if let Some(pending) = &mut payload.pending {
            self.pending(pending, &root)?;
        }
        self.runtime(&mut payload.runtime);
        for child in &mut payload.children {
            if let Some(pending) = &mut child.pending {
                let runtime = child.runtime.snapshot.agent_id.clone();
                self.pending(pending, &runtime)?;
                self.runtime(&mut child.runtime);
            }
        }
        // The canonical controller/group representation determines the public pending review, never a copied display label.
        if let Some(adapter) = payload.adapters.iter().find(|adapter| {
            adapter.id == "runtime.validation" && adapter.state.get("retired_request").is_none()
        }) {
            let controller: ControllerContinuation = serde_json::from_value(adapter.state.clone())?;
            if let Some(review) = &controller.review {
                let pending = payload
                    .pending
                    .as_mut()
                    .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                pending.id = review.id.clone();
                pending.reviewed_action = serde_json::to_value(review)?;
            } else if let Some(driver) = &controller.driver {
                if let Some(wait) = &driver.wait {
                    let pending = payload
                        .pending
                        .as_mut()
                        .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
                    pending.id = wait.request_id.clone();
                    pending.reviewed_action = serde_json::to_value(wait)?;
                } else if let Some(parked) = &driver.parked {
                    payload.pending = Some(parked.pending.clone());
                }
            }
        } else if payload.pause_reason.as_deref() == Some("child_interaction") {
            let adapter = payload
                .adapters
                .iter()
                .find(|adapter| adapter.id == "runtime.group")
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            let group: TaskGroupState = serde_json::from_value(adapter.state.clone())?;
            let pending = payload
                .pending
                .as_mut()
                .ok_or(TaskRunStorageError::InvalidCheckpoint)?;
            pending.id = group.request_id.clone();
            pending.reviewed_action = json!({"group_id":group.frame.id,"child_operation":group.child_operation,"request":group.child_request});
        }
        let driver = payload
            .adapters
            .iter()
            .find(|adapter| {
                adapter.id == "runtime.validation" && adapter.state.get("retired_request").is_none()
            })
            .map(|adapter| serde_json::from_value::<ControllerContinuation>(adapter.state.clone()))
            .transpose()?
            .and_then(|controller| controller.driver);
        if let Some(driver) = driver
            && let Some(adapter) = payload.adapters.iter_mut().find(|adapter| {
                adapter.id == driver.check_id
                    && adapter.state.get("_task_validation_driver").is_some()
            })
        {
            adapter.state["_task_validation_driver"] = serde_json::to_value(driver)?;
        }
        for old in self.requests.keys() {
            if !payload.consumed_request_ids.contains(old) {
                payload.consumed_request_ids.push(old.clone());
            }
        }
        Ok(())
    }
}
