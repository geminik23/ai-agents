use ai_agents::{AgentBuilder, AgentHooks, AgentResponse, ReasoningMode, ReflectionMode, Result};
use async_trait::async_trait;
use example_support::{Repl, init_tracing};
use std::io::{self, Write};
use std::sync::Arc;

// ============================================================================
// Hooks — display reasoning/reflection metadata after each response
// ============================================================================

struct ReasoningHooks;

#[async_trait]
impl AgentHooks for ReasoningHooks {
    async fn on_response(&self, response: &AgentResponse) {
        let Some(ref metadata) = response.metadata else {
            return;
        };

        if let Some(reasoning) = metadata.get("reasoning") {
            println!("  --- Reasoning ---");
            if let Some(mode) = reasoning.get("mode_used") {
                println!("  Mode: {}", mode);
            }
            if let Some(iterations) = reasoning.get("iterations") {
                println!("  Iterations: {}", iterations);
            }
            if let Some(true) = reasoning.get("auto_detected").and_then(|v| v.as_bool()) {
                println!("  Auto-detected: yes");
            }
            if let Some(thinking) = reasoning.get("thinking").and_then(|v| v.as_str()) {
                if !thinking.is_empty() {
                    println!("  Thinking:");
                    for line in thinking.lines().take(5) {
                        println!("    {}", line);
                    }
                    if thinking.lines().count() > 5 {
                        println!("    ... (truncated)");
                    }
                }
            }
            println!();
        }

        if let Some(reflection) = metadata.get("reflection") {
            println!("  --- Reflection ---");
            if let Some(attempts) = reflection.get("attempts") {
                println!("  Attempts: {}", attempts);
            }
            if let Some(eval) = reflection.get("final_evaluation") {
                if let Some(passed) = eval.get("passed").and_then(|v| v.as_bool()) {
                    println!("  Passed: {}", if passed { "yes" } else { "no" });
                }
                if let Some(conf) = eval.get("confidence").and_then(|v| v.as_f64()) {
                    println!("  Confidence: {:.0}%", conf * 100.0);
                }
                if let Some(arr) = eval.get("criteria_results").and_then(|v| v.as_array()) {
                    for c in arr {
                        let name = c.get("criterion").and_then(|v| v.as_str()).unwrap_or("?");
                        let ok = c.get("passed").and_then(|v| v.as_bool()).unwrap_or(false);
                        println!("    {} {}", if ok { "+" } else { "-" }, name);
                    }
                }
            }
            println!();
        }
    }
}

// ============================================================================
// Startup menu — choose reasoning and reflection modes
// ============================================================================

fn ask_choice(prompt: &str, options: &[&str], default: usize) -> usize {
    println!("{}", prompt);
    for (i, opt) in options.iter().enumerate() {
        println!("  {}. {}", i + 1, opt);
    }
    print!("Choice [1-{}, default={}]: ", options.len(), default + 1);
    io::stdout().flush().ok();

    let mut buf = String::new();
    io::stdin().read_line(&mut buf).ok();
    buf.trim()
        .parse::<usize>()
        .ok()
        .filter(|&n| n >= 1 && n <= options.len())
        .map(|n| n - 1)
        .unwrap_or(default)
}

fn select_modes() -> (ReasoningMode, ReflectionMode) {
    let ri = ask_choice(
        "Reasoning mode:",
        &[
            "none   - Direct response (fastest)",
            "cot    - Chain-of-Thought (step by step)",
            "react  - Reason-Act-Observe loop",
            "plan   - Plan-and-Execute (multi-step)",
            "auto   - LLM decides when to reason",
        ],
        4,
    );
    let reasoning = match ri {
        0 => ReasoningMode::None,
        1 => ReasoningMode::CoT,
        2 => ReasoningMode::React,
        3 => ReasoningMode::PlanAndExecute,
        _ => ReasoningMode::Auto,
    };

    println!();

    let fi = ask_choice(
        "Reflection (self-correction):",
        &[
            "disabled - No self-evaluation",
            "enabled  - Always evaluate responses",
            "auto     - LLM decides when needed",
        ],
        2,
    );
    let reflection = match fi {
        0 => ReflectionMode::Disabled,
        1 => ReflectionMode::Enabled,
        _ => ReflectionMode::Auto,
    };

    println!();
    (reasoning, reflection)
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    println!("=== Reasoning Agent Demo ===\n");
    let (reasoning_mode, reflection_mode) = select_modes();

    let agent = AgentBuilder::from_template("reasoning_agent")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .hooks(Arc::new(ReasoningHooks))
        .build()?;

    // Override modes from YAML defaults with user's startup selection
    let mut reasoning_cfg = agent.reasoning_config().clone();
    reasoning_cfg.mode = reasoning_mode.clone();
    let mut reflection_cfg = agent.reflection_config().clone();
    reflection_cfg.enabled = reflection_mode.clone();
    let agent = agent
        .with_reasoning(reasoning_cfg)
        .with_reflection(reflection_cfg);

    Repl::new(agent)
        .welcome(&format!(
            "Reasoning: {:?} | Reflection: {:?}",
            reasoning_mode, reflection_mode
        ))
        .show_tool_calls()
        .hint("Try: 'What is 15 * 7 + 23?' (simple calculation)")
        .hint("Try: 'Explain how photosynthesis works' (reasoning)")
        .hint("Try: 'Compare Python and Rust for web dev' (analytical)")
        .hint("Try: 'Plan a 3-day trip to Tokyo' (multi-step planning)")
        .run()
        .await
}
