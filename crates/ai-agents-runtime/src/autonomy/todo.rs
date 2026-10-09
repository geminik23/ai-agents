//! One run-bound view over the existing canonical todo store.

use super::run::TaskTodoCheckpoint;
use ai_agents_core::Result;
use ai_agents_core::autonomy::{TaskRunKey, TaskRunStorageError};
use ai_agents_tools::{TodoItem, TodoRunBinding, TodoStatus, TodoStore};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

/// A stale view fails after release; the canonical store remains the sole mutable authority.
pub struct RunTodoAdapter {
    store: TodoStore,
    binding: TodoRunBinding,
    released: AtomicBool,
}

/// Rejects ambiguous canonical item identities before binding checkpoint data.
pub(crate) fn validate_todo_items(items: &[TodoItem]) -> Result<()> {
    let mut ids = HashSet::new();
    if items.iter().any(|item| {
        item.id.trim().is_empty() || item.content.trim().is_empty() || !ids.insert(&item.id)
    }) {
        return Err(TaskRunStorageError::InvalidCheckpoint.into());
    }
    Ok(())
}

impl RunTodoAdapter {
    /// Starts with an empty run list so a prior session cannot prove this objective complete.
    pub fn begin(store: TodoStore, key: &TaskRunKey) -> Result<Self> {
        let binding = TodoRunBinding {
            agent_id: key.agent_id.clone(),
            run_id: key.run_id.clone(),
            token: uuid::Uuid::new_v4().to_string(),
        };
        Self::attach(store, binding, vec![])
    }

    /// Restores the exact checkpoint binding only into an unbound canonical store.
    pub fn restore(
        store: TodoStore,
        key: &TaskRunKey,
        checkpoint: TaskTodoCheckpoint,
    ) -> Result<Self> {
        if checkpoint.binding.agent_id != key.agent_id
            || checkpoint.binding.run_id != key.run_id
            || checkpoint.binding.token.is_empty()
        {
            return Err(TaskRunStorageError::InvalidCheckpoint.into());
        }
        Self::attach(store, checkpoint.binding, checkpoint.items)
    }

    // Binding and list replacement share one lock, preventing a partial restore or a second authority.
    fn attach(store: TodoStore, binding: TodoRunBinding, items: Vec<TodoItem>) -> Result<Self> {
        validate_todo_items(&items)?;
        if !store.bind_run(binding.clone(), items) {
            return Err(TaskRunStorageError::Owned.into());
        }
        Ok(Self {
            store,
            binding,
            released: AtomicBool::new(false),
        })
    }

    /// Copies exact checkpoint data only if this adapter still owns the current run list.
    pub fn checkpoint(&self) -> Result<TaskTodoCheckpoint> {
        if self.released.load(Ordering::Acquire) {
            return Err(TaskRunStorageError::Conflict.into());
        }
        let items = self
            .store
            .list_for_run(&self.binding)
            .ok_or(TaskRunStorageError::Conflict)?;
        if self.released.load(Ordering::Acquire) {
            return Err(TaskRunStorageError::Conflict.into());
        }
        validate_todo_items(&items)?;
        Ok(TaskTodoCheckpoint {
            binding: self.binding.clone(),
            items,
        })
    }

    /// Empty, cleared and entirely cancelled lists cannot establish successful progress.
    pub fn all_done(&self) -> Result<bool> {
        let items = self.checkpoint()?.items;
        Ok(items.iter().any(|i| i.status == TodoStatus::Completed)
            && items
                .iter()
                .all(|i| matches!(i.status, TodoStatus::Completed | TodoStatus::Cancelled)))
    }

    /// Explicit release is required; dropping a view must not silently abandon a paused run.
    pub fn release(&self) -> Result<()> {
        // Retire this handle before releasing the canonical binding so restoring the same run cannot revive it.
        if self.released.swap(true, Ordering::AcqRel) || !self.store.release_run(&self.binding) {
            return Err(TaskRunStorageError::Conflict.into());
        }
        Ok(())
    }
}
