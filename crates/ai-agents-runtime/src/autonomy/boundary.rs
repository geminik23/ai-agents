//! Private run ownership and controller input classification, separate from public root-turn admission.

use ai_agents_core::{AgentError, Result};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// An internal capability is minted only while the runtime's root gate excludes competing admission.
pub(crate) struct RunOwner {
    pub(crate) run_id: String,
    pub(crate) actor_id: Option<String>,
    pub(crate) gate: crate::runtime::RootTurnGate,
    abandoned: AtomicBool,
    retained_effect_guards: parking_lot::Mutex<Vec<Box<dyn Send>>>,
}

impl RunOwner {
    /// Identifies a run without treating a serialized run ID as execution authority.
    pub(crate) fn new(
        run_id: String,
        gate: crate::runtime::RootTurnGate,
        actor_id: Option<String>,
    ) -> Result<Arc<Self>> {
        if run_id.trim().is_empty() {
            return Err(AgentError::Config("empty autonomy run identity".into()));
        }
        Ok(Arc::new(Self {
            run_id,
            actor_id,
            gate,
            abandoned: AtomicBool::new(false),
            retained_effect_guards: parking_lot::Mutex::new(Vec::new()),
        }))
    }

    /// Dropped foreground work prevents later execution until its owner acknowledges cleanup.
    pub(crate) fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
    }

    /// Recovery release is restricted to abandoned owners, never a cancellation receipt for live work.
    pub(crate) fn is_abandoned(&self) -> bool {
        self.abandoned.load(Ordering::Acquire)
    }

    /// Unknown host effects retain their framework locks until explicit recovery, not merely future drop.
    pub(crate) fn retain_effect_guard(&self, guard: impl Send + 'static) {
        self.retained_effect_guards.lock().push(Box::new(guard));
    }

    /// Pointer identity authorizes entry; abandonment never becomes an implicit ownership transfer.
    pub(crate) fn check(&self) -> Result<()> {
        if self.abandoned.load(Ordering::Acquire) {
            return Err(AgentError::Other("autonomy owner requires recovery".into()));
        }
        Ok(())
    }
}

pub(crate) type RunOwnerSlot = Arc<parking_lot::RwLock<Option<Arc<RunOwner>>>>;

/// Controller instructions are not conversational input and must never be committed as user facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum AutonomyTurnSource {
    InitialObjective,
    Continuation,
    StageInstruction,
    ValidationFix,
    ChildObjective,
    ResumeAfterApproval,
}

/// Carries a private admitted owner rather than accepting a run ID as a permit.
#[derive(Clone)]
pub(crate) struct AutonomyTurnInput {
    pub(crate) owner: Arc<RunOwner>,
    pub(crate) objective: String,
    pub(crate) controller_message: String,
    pub(crate) source: AutonomyTurnSource,
}

impl AutonomyTurnInput {
    /// Only the first coordinating input is a new user objective; later instructions are ephemeral.
    pub(crate) fn controller_only(&self) -> bool {
        self.source != AutonomyTurnSource::InitialObjective
    }

    /// A delegated objective may run its own pipeline/script once without becoming an actor conversation.
    pub(crate) fn initial_work(&self) -> bool {
        matches!(
            self.source,
            AutonomyTurnSource::InitialObjective | AutonomyTurnSource::ChildObjective
        )
    }
}

tokio::task_local! {
    static AUTONOMY_INPUT: AutonomyTurnInput;
}

/// Captures immutable input classification for redispatch without changing the runtime's global actor.
pub(crate) fn current_turn_input(gate: &crate::runtime::RootTurnGate) -> Option<AutonomyTurnInput> {
    AUTONOMY_INPUT
        .try_with(Clone::clone)
        .ok()
        .filter(|input| Arc::ptr_eq(&input.owner.gate, gate))
}

/// Installs input classification only for this admitted turn and all same-task nested polling.
pub(crate) async fn scope_turn<F: Future>(input: AutonomyTurnInput, future: F) -> F::Output {
    AUTONOMY_INPUT.scope(input, future).await
}

/// A dropped turn retains the runtime reservation and marks it unusable instead of silently detaching.
pub(crate) struct OwnedTurnCleanup {
    owner: Arc<RunOwner>,
    finished: bool,
}

impl OwnedTurnCleanup {
    /// Arms cancellation cleanup before any turn work can be polled.
    pub(crate) fn new(owner: Arc<RunOwner>) -> Self {
        Self {
            owner,
            finished: false,
        }
    }

    /// A returned result still belongs to the run controller, which must settle effects before release.
    pub(crate) fn finish(&mut self) {
        self.finished = true;
    }
}

impl Drop for OwnedTurnCleanup {
    /// Drop cannot await a checkpoint; the retained owner therefore requires explicit recovery.
    fn drop(&mut self) {
        if !self.finished {
            self.owner.abandon();
        }
    }
}
