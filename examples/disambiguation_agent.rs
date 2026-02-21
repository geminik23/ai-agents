use ai_agents::{
    AgentBuilder, AgentHooks, AgentResponse, ClarificationStyle, MaxAttemptsAction, Result,
};
use async_trait::async_trait;
use example_support::{Repl, init_tracing};
use std::io::{self, Write};
use std::sync::Arc;

// ============================================================================
// Hooks — display disambiguation metadata after each response
// ============================================================================

struct DisambiguationHooks;

#[async_trait]
impl AgentHooks for DisambiguationHooks {
    async fn on_response(&self, response: &AgentResponse) {
        let Some(ref metadata) = response.metadata else {
            return;
        };
        let Some(disambiguation) = metadata.get("disambiguation") else {
            return;
        };

        println!("  --- Disambiguation ---");

        if let Some(s) = disambiguation.get("status").and_then(|v| v.as_str()) {
            println!("  Status: {}", s);
        }

        if let Some(detection) = disambiguation.get("detection") {
            if let Some(t) = detection.get("type") {
                println!("  Ambiguity type: {:?}", t);
            }
            if let Some(c) = detection.get("confidence").and_then(|v| v.as_f64()) {
                println!("  Confidence: {:.0}%", c * 100.0);
            }
            if let Some(arr) = detection.get("what_is_unclear").and_then(|v| v.as_array()) {
                if !arr.is_empty() {
                    println!("  Unclear:");
                    for item in arr.iter().filter_map(|v| v.as_str()) {
                        println!("    - {}", item);
                    }
                }
            }
        }

        if let Some(arr) = disambiguation.get("options").and_then(|v| v.as_array()) {
            if !arr.is_empty() {
                println!("  Options:");
                for (i, opt) in arr.iter().enumerate() {
                    if let Some(label) = opt.get("label").and_then(|v| v.as_str()) {
                        println!("    {}. {}", i + 1, label);
                    }
                }
            }
        }

        if let Some(arr) = disambiguation.get("clarifying").and_then(|v| v.as_array()) {
            if !arr.is_empty() {
                println!("  Clarifying aspects:");
                for a in arr.iter().filter_map(|v| v.as_str()) {
                    println!("    - {}", a);
                }
            }
        }
        println!();
    }
}

// ============================================================================
// Startup menu — choose clarification style and fallback action
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

fn select_options() -> (ClarificationStyle, MaxAttemptsAction) {
    let si = ask_choice(
        "Clarification style:",
        &[
            "auto    - LLM decides best style",
            "options - Multiple choice questions",
            "open    - Open-ended questions",
            "yes_no  - Binary yes/no questions",
            "hybrid  - Options + 'other' choice",
        ],
        0,
    );
    let style = match si {
        1 => ClarificationStyle::Options,
        2 => ClarificationStyle::Open,
        3 => ClarificationStyle::YesNo,
        4 => ClarificationStyle::Hybrid,
        _ => ClarificationStyle::Auto,
    };

    println!();

    let fi = ask_choice(
        "Fallback when max attempts reached:",
        &[
            "best_guess - Continue with best interpretation",
            "apologize  - Apologize and stop",
            "escalate   - Request human intervention",
        ],
        0,
    );
    let fallback = match fi {
        1 => MaxAttemptsAction::ApologizeAndStop,
        2 => MaxAttemptsAction::Escalate,
        _ => MaxAttemptsAction::ProceedWithBestGuess,
    };

    println!();
    (style, fallback)
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    println!("=== Disambiguation Agent Demo ===\n");
    let (style, fallback) = select_options();

    let agent = AgentBuilder::from_template("disambiguation_agent")?
        .auto_configure_llms()?
        .auto_configure_features()?
        .hooks(Arc::new(DisambiguationHooks))
        .build()?;

    // Override style/fallback from YAML defaults with user's startup selection
    let mut config = agent
        .disambiguation_manager()
        .expect("disambiguation configured in YAML")
        .config()
        .clone();
    config.clarification.style = style.clone();
    config.clarification.on_max_attempts = fallback;
    let agent = agent.with_disambiguation(config);

    Repl::new(agent)
        .welcome(&format!("Disambiguation: enabled | Style: {:?}", style))
        .show_tool_calls()
        .hint("Ambiguous: 'Send it'             (missing target)")
        .hint("Ambiguous: 'Do the thing'         (vague action)")
        .hint("Ambiguous: '그거 보내줘'           (Korean: vague)")
        .hint("Ambiguous: 'あれをお願いします'     (Japanese: vague)")
        .hint("Clear:     'What time is it?'     (no disambiguation)")
        .hint("Skipped:   'Hello!'               (social)")
        .run()
        .await
}
