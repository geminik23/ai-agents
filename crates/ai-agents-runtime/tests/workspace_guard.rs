// The workspace guard must compose with the unchanged tool policy on the real shared executor path.
use ai_agents_core::{ToolCallSource, ToolExecutionRequest, ToolInvoker};
use ai_agents_llm::mock::MockLLMProvider;
use ai_agents_runtime::{AgentBuilder, RuntimeAgent, autonomy::HostRevisionSource};
use ai_agents_tools::{
    PathPolicyConfig, ToolPolicyConfig, ToolRegistry, ToolSecurityConfig, ToolSecurityEngine,
    WorkspaceGuard, WorkspaceScope,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct Fixture {
    outer: PathBuf,
    root: PathBuf,
}

impl Fixture {
    // The root sits inside a policy-writable parent so the guard's own containment check is observable.
    fn new() -> Self {
        let dir =
            std::env::temp_dir().join(format!("ai-agents-workspace-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let outer = std::fs::canonicalize(&dir).unwrap();
        let root = outer.join("workspace");
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(root.join("tests").join("acceptance.rs"), "protected").unwrap();
        std::fs::write(outer.join("sibling.txt"), "user").unwrap();
        Self { outer, root }
    }
}

impl Drop for Fixture {
    // Cleanup is limited to this test's uniquely owned directory.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.outer);
    }
}

// Writes are allowed under the parent, the checker inputs are denied, and approval is pre-granted so only policy and the guard decide.
fn agent(fixture: &Fixture, guard: &Arc<WorkspaceGuard>) -> RuntimeAgent {
    let mut registry = ToolRegistry::new();
    guard.install(&mut registry).unwrap();
    let mut security = ToolSecurityConfig {
        enabled: true,
        fail_closed: true,
        ..Default::default()
    };
    let outer = fixture.outer.to_string_lossy().into_owned();
    security.tools.insert(
        "file_write".into(),
        ToolPolicyConfig {
            write_paths: vec![outer.clone()],
            paths: PathPolicyConfig {
                deny: vec![fixture.root.join("tests").to_string_lossy().into_owned()],
                ..Default::default()
            },
            overwrite_existing: true,
            create_parent_dirs: true,
            allow_without_confirmation: true,
            ..Default::default()
        },
    );
    security.tools.insert(
        "file_read".into(),
        ToolPolicyConfig {
            read_paths: vec![outer],
            allow_without_confirmation: true,
            ..Default::default()
        },
    );
    let mut provider = MockLLMProvider::new("workspace-guard");
    provider.set_response("done");
    AgentBuilder::new()
        .system_prompt("test")
        .llm(Arc::new(provider))
        .tools(registry)
        .tool_security(ToolSecurityEngine::try_new(security).unwrap())
        .build()
        .unwrap()
}

async fn write(
    agent: &RuntimeAgent,
    id: &str,
    path: &Path,
    overwrite: bool,
) -> ai_agents_core::ToolExecutionRecord {
    agent
        .invoke_tool(ToolExecutionRequest::new(
            id,
            "file_write",
            json!({"path":path,"content":"agent","overwrite":overwrite,"create_parent_dirs":true}),
            ToolCallSource::Manual,
        ))
        .await
        .unwrap()
}

#[tokio::test]
async fn denied_checker_inputs_and_outside_paths_never_reach_a_mutation() {
    let fixture = Fixture::new();
    let guard = WorkspaceGuard::new(fixture.root.clone(), WorkspaceScope::Root, vec![]).unwrap();
    let agent = agent(&fixture, &guard);

    // Policy deny blocks the checker input before the tool body, so the revision does not move.
    let protected = fixture.root.join("tests").join("acceptance.rs");
    let denied = write(&agent, "denied", &protected, true).await;
    assert!(!denied.executed);
    assert_eq!(std::fs::read_to_string(&protected).unwrap(), "protected");
    assert_eq!(guard.revision().as_deref(), Some("0"));

    // The policy permits the parent directory, but the guard still confines the body to the workspace root.
    let sibling = fixture.outer.join("sibling.txt");
    let outside = write(&agent, "outside", &sibling, true).await;
    assert!(!outside.success);
    assert!(
        outside.output.contains("outside the workspace root"),
        "{}",
        outside.output
    );
    assert_eq!(std::fs::read_to_string(&sibling).unwrap(), "user");
    assert_eq!(guard.revision().as_deref(), Some("0"));
    assert!(!guard.requires_reconciliation());
}

#[tokio::test]
async fn guarded_creation_advances_the_validation_target_revision() {
    let fixture = Fixture::new();
    let guard = WorkspaceGuard::new(fixture.root.clone(), WorkspaceScope::Root, vec![]).unwrap();
    let agent = agent(&fixture, &guard);
    let source: Arc<dyn HostRevisionSource> = guard.clone();
    assert_eq!(source.revision().unwrap(), "0");

    let created = fixture.root.join("src").join("feature.rs");
    let record = write(&agent, "create", &created, false).await;
    assert!(record.executed && record.success, "{}", record.output);
    assert_eq!(std::fs::read_to_string(&created).unwrap(), "agent");
    assert_eq!(source.revision().unwrap(), "2");

    // A guard-created file cannot be silently overwritten; the refusal happens before any new mutation.
    let rewrite = write(&agent, "rewrite", &created, true).await;
    assert!(!rewrite.success);
    assert!(
        rewrite.output.contains("already created"),
        "{}",
        rewrite.output
    );
    assert_eq!(source.revision().unwrap(), "2");

    // A replaced created file poisons the guard, and the validation target reports unavailable instead of a stale token.
    let replacement = fixture.root.join("replacement.rs");
    std::fs::write(&replacement, "user").unwrap();
    std::fs::rename(&replacement, &created).unwrap();
    let read = agent
        .invoke_tool(ToolExecutionRequest::new(
            "read",
            "file_read",
            json!({"path":created}),
            ToolCallSource::Manual,
        ))
        .await
        .unwrap();
    assert!(read.executed && !read.success);
    assert!(read.output.contains("identity changed"), "{}", read.output);
    assert!(source.revision().is_err());
    assert!(guard.requires_reconciliation());
}
