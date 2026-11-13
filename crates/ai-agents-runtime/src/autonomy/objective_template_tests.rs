//! Explicit and automatic profiles share accepted-input aliases, not mutable context aliases.
use crate::AgentBuilder;
use std::sync::Arc;

// Accepted invocation aliases cannot be shadowed by host context, and undefined template fields remain errors.
#[test]
fn explicit_objective_input_aliases_are_immutable_and_strict() {
    let provider = super::runner_tests::RecordingProvider::new(&[]);
    let agent = AgentBuilder::new()
        .system_prompt("Help")
        .llm(Arc::clone(&provider) as Arc<dyn ai_agents_core::LLMProvider>)
        .build()
        .unwrap();
    agent
        .set_context("user_input", serde_json::json!("unrelated"))
        .unwrap();
    assert_eq!(
        agent
            .render_autonomy_objective("{{ input }}|{{ user_input }}|{{ objective }}", "accepted")
            .unwrap(),
        "accepted|accepted|accepted"
    );
    assert!(
        agent
            .render_autonomy_objective("{{ absent }}", "accepted")
            .is_err()
    );
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
}
