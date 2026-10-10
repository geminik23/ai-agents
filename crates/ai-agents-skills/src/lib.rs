//! Skill system for AI Agents framework

pub mod definition;
pub mod executor;
pub mod loader;
pub mod router;

pub use definition::{
    SkillContext, SkillDefinition, SkillExecutionCursor, SkillRef, SkillStep, StepResult,
};
pub use executor::{SkillExecutionObserver, SkillExecutor};
pub use loader::SkillLoader;
pub use router::SkillRouter;
