use std::collections::HashMap;
use std::time::Instant;

use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_hooks::AgentHooks;
use tracing::{debug, info, warn};

use super::types::{PipelineResult, PipelineStage, StageOutput};
use crate::Agent;
use crate::spawner::AgentRegistry;
use crate::turn_context::current_turn_actor_context;

/// Chains agents sequentially, retaining acknowledged outputs, prepared stage input and actor sender scope for task resume.
/// Suspended stages never become failed outputs and completed stage hooks are not repeated by ordinary reentry.
pub async fn pipeline(
    registry: &AgentRegistry,
    input: &str,
    stages: &[PipelineStage],
    timeout_ms: Option<u64>,
    hooks: Option<&dyn AgentHooks>,
    context_values: Option<&HashMap<String, serde_json::Value>>,
) -> Result<PipelineResult> {
    let authorized = crate::autonomy::composition::enter_composition_dispatch();
    crate::autonomy::composition::scope_dispatch_authority(
        authorized,
        Box::pin(pipeline_dispatch(
            registry,
            input,
            stages,
            timeout_ms,
            hooks,
            context_values,
        )),
    )
    .await
}

/// Saved stage cursor selection belongs only to the framework parent entry, not to callbacks invoking a nested public pipeline.
async fn pipeline_dispatch(
    registry: &AgentRegistry,
    input: &str,
    stages: &[PipelineStage],
    timeout_ms: Option<u64>,
    hooks: Option<&dyn AgentHooks>,
    context_values: Option<&HashMap<String, serde_json::Value>>,
) -> Result<PipelineResult> {
    if stages.is_empty() {
        return Err(AgentError::Config("Pipeline has no stages".into()));
    }

    use crate::autonomy::composition;
    let execution = crate::autonomy::current_execution();
    let frame = execution
        .as_ref()
        .map(|execution| execution.composition_frame())
        .transpose()?
        .flatten();
    let original_input = input.to_string();
    let mut cursor = if let Some(value) = frame
        .as_ref()
        .and_then(|frame| frame.cursor.get("pipeline"))
    {
        serde_json::from_value::<TaskPipelineCursor>(value.clone())?
    } else {
        let expiry = if let Some(frame) = &frame {
            frame.expires_at
        } else {
            timeout_ms
                .map(|millis| {
                    let millis = i64::try_from(millis)
                        .map_err(|_| AgentError::Config("pipeline deadline overflow".into()))?;
                    chrono::Utc::now()
                        .checked_add_signed(chrono::Duration::milliseconds(millis))
                        .ok_or_else(|| AgentError::Config("pipeline deadline overflow".into()))
                })
                .transpose()?
        };
        TaskPipelineCursor {
            next_stage: 0,
            current_input: input.into(),
            outputs: Vec::new(),
            completed: HashMap::new(),
            actor: current_turn_actor_context(),
            prepared_input: None,
            expiry,
        }
    };
    if cursor.next_stage > stages.len() || cursor.outputs.len() != cursor.next_stage {
        return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
    }
    let mut current_input = cursor.current_input.clone();
    let mut stage_outputs = cursor.outputs.clone();
    let mut completed_stages = cursor.completed.clone();
    let pipeline_start = Instant::now();
    let mut actor_context = cursor.actor.clone();

    for (i, stage) in stages.iter().enumerate().skip(cursor.next_stage) {
        let agent_id = &stage.agent_id;

        // Check total pipeline timeout before starting the next stage.
        if frame.is_some()
            && cursor
                .expiry
                .is_some_and(|expiry| chrono::Utc::now() >= expiry)
        {
            if let Some(execution) = &execution {
                execution.stop("pipeline_timeout");
            }
            return Err(AgentError::HITLTimeout);
        }
        if let Some(timeout) = timeout_ms.filter(|_| frame.is_none()) {
            let elapsed = pipeline_start.elapsed().as_millis() as u64;
            if elapsed >= timeout {
                warn!(stage = i, "Pipeline timeout exceeded");
                break;
            }
        }

        let agent = registry.get(agent_id).ok_or_else(|| {
            AgentError::Other(format!(
                "Pipeline stage {} agent not found: {}",
                i, agent_id
            ))
        })?;

        // Build the effective input for this stage.
        let effective_input = if let Some(prepared) = &cursor.prepared_input {
            prepared.clone()
        } else if let Some(ref tmpl) = stage.input {
            render_stage_template(
                tmpl,
                &current_input,
                &original_input,
                &completed_stages,
                context_values,
            )
        } else {
            current_input.clone()
        };

        debug!(stage = i, agent = %agent_id, "Pipeline stage starting");
        let stage_start = Instant::now();

        cursor.prepared_input = Some(effective_input.clone());
        if let (Some(execution), Some(frame)) = (&execution, &frame) {
            let mut saved = frame.cursor.clone();
            saved["pipeline"] = serde_json::to_value(&cursor)?;
            Box::pin(execution.checkpoint_composition_cursor(&frame.runtime_id, saved)).await?;
        }
        let slot = frame
            .as_ref()
            .map(|frame| {
                frame
                    .children
                    .get(i)
                    .cloned()
                    .ok_or(crate::autonomy::TaskRunStorageError::InvalidCheckpoint)
            })
            .transpose()?;
        let work = composition::run_composition_child(
            composition::current_composition(),
            slot,
            composition::group_response_custody(),
            async {
                if let Some(context) = actor_context.clone() {
                    agent
                        .chat_with_actor_context(&effective_input, context)
                        .await
                } else {
                    agent.chat(&effective_input).await
                }
            },
        );
        let remaining = if frame.is_some() {
            cursor
                .expiry
                .map(|expiry| (expiry - chrono::Utc::now()).to_std().unwrap_or_default())
        } else {
            timeout_ms.map(|timeout| {
                std::time::Duration::from_millis(
                    timeout.saturating_sub(pipeline_start.elapsed().as_millis() as u64),
                )
            })
        };
        let stage_result = if let Some(remaining) = remaining {
            match tokio::time::timeout(remaining, work).await {
                Ok(result) => result,
                Err(_) => {
                    warn!(stage = i, agent = %agent_id, "Pipeline stage timed out");
                    stage_outputs.push(StageOutput {
                        agent_id: agent_id.clone(),
                        output: String::new(),
                        duration_ms: stage_start.elapsed().as_millis() as u64,
                        skipped: true,
                    });
                    break;
                }
            }
        } else {
            work.await
        };

        let duration_ms = stage_start.elapsed().as_millis() as u64;

        match stage_result {
            Ok(response) => {
                let output = response.content.clone();
                stage_outputs.push(StageOutput {
                    agent_id: agent_id.clone(),
                    output: output.clone(),
                    duration_ms,
                    skipped: false,
                });
                completed_stages.insert(agent_id.clone(), output.clone());
                current_input = output;
                if let Some(context) = actor_context.as_mut() {
                    *context = context.for_sender(agent_id.clone());
                }

                if let Some(h) = hooks {
                    h.on_pipeline_stage(i, agent_id, duration_ms).await;
                }
                cursor.next_stage = i + 1;
                cursor.current_input = current_input.clone();
                cursor.outputs = stage_outputs.clone();
                cursor.completed = completed_stages.clone();
                cursor.actor = actor_context.clone();
                cursor.prepared_input = None;
                if let (Some(execution), Some(frame)) = (&execution, &frame) {
                    let mut saved = frame.cursor.clone();
                    saved["pipeline"] = serde_json::to_value(&cursor)?;
                    Box::pin(execution.checkpoint_composition_cursor(&frame.runtime_id, saved))
                        .await?;
                }
            }
            Err(error @ AgentError::TaskSuspended(_)) => return Err(error),
            Err(e) => {
                warn!(stage = i, agent = %agent_id, error = %e, "Pipeline stage failed");
                stage_outputs.push(StageOutput {
                    agent_id: agent_id.clone(),
                    output: format!("Error: {}", e),
                    duration_ms,
                    skipped: false,
                });
                return Err(e);
            }
        }
    }

    let final_response = AgentResponse::new(current_input);

    info!(
        stages = stage_outputs.len(),
        total_ms = pipeline_start.elapsed().as_millis() as u64,
        "Pipeline completed"
    );

    if let Some(h) = hooks {
        h.on_pipeline_complete(
            stage_outputs.len(),
            pipeline_start.elapsed().as_millis() as u64,
        )
        .await;
    }

    Ok(PipelineResult {
        response: final_response,
        stage_outputs,
    })
}

/// Exact sequential cursor includes pending template output so resume cannot rerender it against changed context.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskPipelineCursor {
    next_stage: usize,
    current_input: String,
    outputs: Vec<StageOutput>,
    completed: HashMap<String, String>,
    actor: Option<crate::TurnActorContext>,
    prepared_input: Option<String>,
    expiry: Option<chrono::DateTime<chrono::Utc>>,
}

//
// Render a stage input template using minijinja.
// Falls back to simple string replacement if minijinja fails.
//
// Available variables:
//   {{ previous_output }}    - output from the immediately previous stage
//   {{ original_input }}     - the user's original input
//   {{ user_input }}         - alias for original_input (consistent with concurrent)
//   {{ stages.<agent_id> }} - output from any earlier stage by agent ID
//   {{ context.<key> }}     - values from the context manager (when provided)
//
fn render_stage_template(
    template: &str,
    previous_output: &str,
    original_input: &str,
    completed_stages: &HashMap<String, String>,
    context_values: Option<&HashMap<String, serde_json::Value>>,
) -> String {
    let mut env = minijinja::Environment::new();
    if env.add_template("stage", template).is_err() {
        return fallback_replace(template, previous_output, original_input);
    }

    let stages_value = minijinja::Value::from_serialize(completed_stages);

    let mut ctx = std::collections::BTreeMap::new();
    ctx.insert(
        "previous_output".to_string(),
        minijinja::Value::from(previous_output),
    );
    ctx.insert(
        "original_input".to_string(),
        minijinja::Value::from(original_input),
    );
    ctx.insert(
        "user_input".to_string(),
        minijinja::Value::from(original_input),
    );
    ctx.insert("stages".to_string(), stages_value);

    if let Some(cv) = context_values {
        ctx.insert("context".to_string(), minijinja::Value::from_serialize(cv));
    }

    match env.get_template("stage") {
        Ok(tmpl) => match tmpl.render(minijinja::Value::from_serialize(&ctx)) {
            Ok(rendered) => rendered,
            Err(e) => {
                debug!("minijinja render failed, falling back to replace: {}", e);
                fallback_replace(template, previous_output, original_input)
            }
        },
        Err(_) => fallback_replace(template, previous_output, original_input),
    }
}

/// Simple string replacement fallback for templates that only use the two basic variables.
fn fallback_replace(template: &str, previous_output: &str, original_input: &str) -> String {
    template
        .replace("{{ previous_output }}", previous_output)
        .replace("{{ original_input }}", original_input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_stage_from_string() {
        let stage = PipelineStage::from("writer");
        assert_eq!(stage.agent_id, "writer");
        assert!(stage.input.is_none());
    }

    #[test]
    fn test_pipeline_stage_with_input() {
        let stage = PipelineStage::id("reviewer")
            .with_input("Review this:\n\n{{ previous_output }}\n\nOriginal: {{ original_input }}");
        assert_eq!(stage.agent_id, "reviewer");
        assert!(
            stage
                .input
                .as_ref()
                .unwrap()
                .contains("{{ previous_output }}")
        );
    }

    #[test]
    fn test_template_basic_replacement() {
        let tmpl = "Review:\n{{ previous_output }}\n\nOriginal: {{ original_input }}";
        let stages = HashMap::new();
        let result = render_stage_template(tmpl, "draft content", "write a poem", &stages, None);
        assert!(result.contains("draft content"));
        assert!(result.contains("write a poem"));
        assert!(!result.contains("{{"));
    }

    #[test]
    fn test_template_stages_by_agent_id() {
        let mut stages = HashMap::new();
        stages.insert("writer".to_string(), "A great draft.".to_string());
        stages.insert(
            "reviewer".to_string(),
            "Looks good, minor typo on line 3.".to_string(),
        );

        let tmpl = "Draft:\n{{ stages.writer }}\n\nReview:\n{{ stages.reviewer }}\n\nOriginal: {{ original_input }}";
        let result = render_stage_template(
            tmpl,
            "Looks good, minor typo on line 3.",
            "write a poem",
            &stages,
            None,
        );
        assert!(result.contains("A great draft."));
        assert!(result.contains("minor typo on line 3"));
        assert!(result.contains("write a poem"));
    }

    #[test]
    fn test_template_stages_empty_before_first_stage() {
        let stages = HashMap::new();
        let tmpl = "Input: {{ original_input }}";
        let result = render_stage_template(tmpl, "", "hello", &stages, None);
        assert_eq!(result, "Input: hello");
    }

    #[test]
    fn test_template_mixed_previous_and_named() {
        let mut stages = HashMap::new();
        stages.insert("writer".to_string(), "The original draft.".to_string());

        let tmpl = "Previous: {{ previous_output }}\nWriter: {{ stages.writer }}";
        let result = render_stage_template(tmpl, "reviewer feedback", "request", &stages, None);
        assert!(result.contains("reviewer feedback"));
        assert!(result.contains("The original draft."));
    }

    #[test]
    fn test_fallback_replace() {
        let tmpl = "{{ previous_output }} and {{ original_input }}";
        let result = fallback_replace(tmpl, "prev", "orig");
        assert_eq!(result, "prev and orig");
    }

    #[test]
    fn test_template_backward_compatible_replace_syntax() {
        // Old-style templates with only previous_output and original_input still work.
        let stages = HashMap::new();
        let tmpl =
            "Review this draft:\n\n{{ previous_output }}\n\nOriginal request: {{ original_input }}";
        let result = render_stage_template(tmpl, "my draft", "write something", &stages, None);
        assert!(result.contains("my draft"));
        assert!(result.contains("write something"));
    }

    #[test]
    fn test_template_with_context_values() {
        let stages = HashMap::new();
        let mut ctx = HashMap::new();
        ctx.insert(
            "user".to_string(),
            serde_json::json!({"name": "Alice", "tier": "premium"}),
        );

        let tmpl = "Hello {{ context.user.name }}! Input: {{ user_input }}";
        let result = render_stage_template(tmpl, "", "analyze AAPL", &stages, Some(&ctx));
        assert!(result.contains("Alice"));
        assert!(result.contains("analyze AAPL"));
    }
}
