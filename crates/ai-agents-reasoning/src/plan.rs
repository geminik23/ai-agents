use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub goal: String,
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Plan {
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            goal: goal.into(),
            steps: Vec::new(),
            status: PlanStatus::Pending,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn with_steps(mut self, steps: Vec<PlanStep>) -> Self {
        self.steps = steps;
        self
    }

    pub fn add_step(&mut self, step: PlanStep) {
        self.steps.push(step);
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.status, PlanStatus::Completed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.status, PlanStatus::Failed { .. })
    }

    pub fn pending_steps(&self) -> impl Iterator<Item = &PlanStep> {
        self.steps.iter().filter(|s| s.status.is_pending())
    }

    pub fn completed_steps(&self) -> impl Iterator<Item = &PlanStep> {
        self.steps.iter().filter(|s| s.status.is_completed())
    }

    pub fn next_executable_step(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|s| {
            s.status.is_pending() && s.dependencies.iter().all(|dep| self.is_step_completed(dep))
        })
    }

    fn is_step_completed(&self, step_id: &str) -> bool {
        self.steps
            .iter()
            .find(|s| s.id == step_id)
            .map(|s| s.status.is_completed())
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub description: String,
    pub action: PlanAction,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub status: StepStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

impl PlanStep {
    pub fn new(description: impl Into<String>, action: PlanAction) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            action,
            dependencies: Vec::new(),
            status: StepStatus::Pending,
            result: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    pub fn mark_running(&mut self) {
        self.status = StepStatus::Running;
    }

    pub fn mark_completed(&mut self, result: Option<serde_json::Value>) {
        self.status = StepStatus::Completed;
        self.result = result;
    }

    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = StepStatus::Failed {
            error: error.into(),
        };
    }

    pub fn mark_skipped(&mut self, reason: impl Into<String>) {
        self.status = StepStatus::Skipped {
            reason: reason.into(),
        };
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PlanAction {
    Tool {
        tool: String,
        #[serde(default)]
        args: serde_json::Value,
    },
    Skill {
        skill: String,
    },
    Think {
        prompt: String,
    },
    Respond {
        template: String,
    },
}

impl PlanAction {
    pub fn tool(name: impl Into<String>, args: serde_json::Value) -> Self {
        PlanAction::Tool {
            tool: name.into(),
            args,
        }
    }

    pub fn skill(name: impl Into<String>) -> Self {
        PlanAction::Skill { skill: name.into() }
    }

    pub fn think(prompt: impl Into<String>) -> Self {
        PlanAction::Think {
            prompt: prompt.into(),
        }
    }

    pub fn respond(template: impl Into<String>) -> Self {
        PlanAction::Respond {
            template: template.into(),
        }
    }

    pub fn is_tool(&self) -> bool {
        matches!(self, PlanAction::Tool { .. })
    }

    pub fn is_skill(&self) -> bool {
        matches!(self, PlanAction::Skill { .. })
    }

    pub fn is_think(&self) -> bool {
        matches!(self, PlanAction::Think { .. })
    }

    pub fn is_respond(&self) -> bool {
        matches!(self, PlanAction::Respond { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PlanStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Failed {
        error: String,
    },
    Replanned {
        reason: String,
        new_plan_id: String,
    },
}

impl PlanStatus {
    pub fn is_pending(&self) -> bool {
        matches!(self, PlanStatus::Pending)
    }

    pub fn is_in_progress(&self) -> bool {
        matches!(self, PlanStatus::InProgress)
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, PlanStatus::Completed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, PlanStatus::Failed { .. })
    }

    pub fn is_replanned(&self) -> bool {
        matches!(self, PlanStatus::Replanned { .. })
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            PlanStatus::Completed | PlanStatus::Failed { .. } | PlanStatus::Replanned { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StepStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed {
        error: String,
    },
    Skipped {
        reason: String,
    },
}

impl StepStatus {
    pub fn is_pending(&self) -> bool {
        matches!(self, StepStatus::Pending)
    }

    pub fn is_running(&self) -> bool {
        matches!(self, StepStatus::Running)
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, StepStatus::Completed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, StepStatus::Failed { .. })
    }

    pub fn is_skipped(&self) -> bool {
        matches!(self, StepStatus::Skipped { .. })
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            StepStatus::Completed | StepStatus::Failed { .. } | StepStatus::Skipped { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_creation() {
        let plan = Plan::new("Test goal");
        assert_eq!(plan.goal, "Test goal");
        assert!(plan.steps.is_empty());
        assert!(plan.status.is_pending());
    }

    #[test]
    fn test_plan_with_steps() {
        let step1 = PlanStep::new("Step 1", PlanAction::think("Think about it"));
        let step2 = PlanStep::new("Step 2", PlanAction::respond("Respond"));

        let plan = Plan::new("Multi-step goal").with_steps(vec![step1, step2]);
        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn test_plan_step_dependencies() {
        let step1 = PlanStep::new("Step 1", PlanAction::think("Think"))
            .with_id("step1")
            .with_dependencies(vec![]);

        let step2 = PlanStep::new("Step 2", PlanAction::respond("Respond"))
            .with_id("step2")
            .with_dependencies(vec!["step1".to_string()]);

        let mut plan = Plan::new("Goal").with_steps(vec![step1, step2]);

        // First executable should be step1
        let next = plan.next_executable_step().unwrap();
        assert_eq!(next.id, "step1");

        // Mark step1 as completed
        plan.steps[0].mark_completed(None);

        // Now step2 should be executable
        let next = plan.next_executable_step().unwrap();
        assert_eq!(next.id, "step2");
    }

    #[test]
    fn test_plan_action_types() {
        let tool = PlanAction::tool("search", serde_json::json!({"query": "test"}));
        assert!(tool.is_tool());

        let skill = PlanAction::skill("greeting");
        assert!(skill.is_skill());

        let think = PlanAction::think("Consider the options");
        assert!(think.is_think());

        let respond = PlanAction::respond("Final answer: {{ result }}");
        assert!(respond.is_respond());
    }

    #[test]
    fn test_plan_action_serde() {
        let action = PlanAction::tool("http", serde_json::json!({"url": "https://example.com"}));
        let json = serde_json::to_string(&action).unwrap();
        let parsed: PlanAction = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_tool());
    }

    #[test]
    fn test_plan_status() {
        assert!(PlanStatus::Pending.is_pending());
        assert!(!PlanStatus::Pending.is_terminal());

        assert!(PlanStatus::InProgress.is_in_progress());
        assert!(!PlanStatus::InProgress.is_terminal());

        assert!(PlanStatus::Completed.is_completed());
        assert!(PlanStatus::Completed.is_terminal());

        let failed = PlanStatus::Failed {
            error: "Error".to_string(),
        };
        assert!(failed.is_failed());
        assert!(failed.is_terminal());

        let replanned = PlanStatus::Replanned {
            reason: "Better approach".to_string(),
            new_plan_id: "plan2".to_string(),
        };
        assert!(replanned.is_replanned());
        assert!(replanned.is_terminal());
    }

    #[test]
    fn test_step_status() {
        assert!(StepStatus::Pending.is_pending());
        assert!(!StepStatus::Pending.is_terminal());

        assert!(StepStatus::Running.is_running());
        assert!(!StepStatus::Running.is_terminal());

        assert!(StepStatus::Completed.is_completed());
        assert!(StepStatus::Completed.is_terminal());

        let failed = StepStatus::Failed {
            error: "Error".to_string(),
        };
        assert!(failed.is_failed());
        assert!(failed.is_terminal());

        let skipped = StepStatus::Skipped {
            reason: "Not needed".to_string(),
        };
        assert!(skipped.is_skipped());
        assert!(skipped.is_terminal());
    }

    #[test]
    fn test_plan_step_state_transitions() {
        let mut step = PlanStep::new("Test step", PlanAction::think("Think"));

        assert!(step.status.is_pending());

        step.mark_running();
        assert!(step.status.is_running());

        step.mark_completed(Some(serde_json::json!({"answer": 42})));
        assert!(step.status.is_completed());
        assert!(step.result.is_some());
    }

    #[test]
    fn test_plan_step_failure() {
        let mut step = PlanStep::new("Test step", PlanAction::tool("http", serde_json::json!({})));

        step.mark_running();
        step.mark_failed("Connection timeout");

        assert!(step.status.is_failed());
        if let StepStatus::Failed { error } = &step.status {
            assert_eq!(error, "Connection timeout");
        }
    }
}
