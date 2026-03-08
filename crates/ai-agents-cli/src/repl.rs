use ai_agents::memory::{estimate_message_tokens, estimate_tokens};
use ai_agents::{Agent, AgentResponse, RuntimeAgent, StreamChunk};
use futures::StreamExt;
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplMode {
    Chat,
    Streaming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptStyle {
    Simple,
    WithState,
}

/// Return type for custom command handlers registered via [`CliRepl::on_command`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandResult {
    Handled,
    NotHandled,
}

type CommandHandler = Box<dyn Fn(&str, &RuntimeAgent) -> CommandResult + Send + Sync>;

#[derive(Debug, Clone)]
pub struct CliReplConfig {
    pub welcome: Option<String>,
    pub prompt: PromptStyle,
    pub mode: ReplMode,
    pub show_tool_calls: bool,
    pub show_state_transitions: bool,
    pub show_timing: bool,
    pub builtin_commands: bool,
    pub hints: Vec<String>,
}

impl Default for CliReplConfig {
    fn default() -> Self {
        Self {
            welcome: None,
            prompt: PromptStyle::Simple,
            mode: ReplMode::Chat,
            show_tool_calls: false,
            show_state_transitions: false,
            show_timing: false,
            builtin_commands: true,
            hints: Vec::new(),
        }
    }
}

enum CommandAction {
    Continue,
    Quit,
}

pub struct CliRepl {
    agent: RuntimeAgent,
    config: CliReplConfig,
    command_handler: Option<CommandHandler>,
}

// Manual Debug impl because CommandHandler (a boxed closure) doesn't implement Debug.
impl std::fmt::Debug for CliRepl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliRepl")
            .field("agent", &"RuntimeAgent { .. }")
            .field("config", &self.config)
            .field(
                "command_handler",
                &if self.command_handler.is_some() {
                    "Some(<fn>)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

impl CliRepl {
    pub fn new(agent: RuntimeAgent) -> Self {
        Self {
            agent,
            config: CliReplConfig::default(),
            command_handler: None,
        }
    }

    pub fn with_config(mut self, config: CliReplConfig) -> Self {
        self.config = config;
        self
    }

    pub fn welcome(mut self, msg: impl Into<String>) -> Self {
        self.config.welcome = Some(msg.into());
        self
    }

    pub fn hint(mut self, msg: impl Into<String>) -> Self {
        self.config.hints.push(msg.into());
        self
    }

    pub fn prompt(mut self, style: PromptStyle) -> Self {
        self.config.prompt = style;
        self
    }

    pub fn streaming(mut self) -> Self {
        self.config.mode = ReplMode::Streaming;
        self
    }

    pub fn show_tool_calls(mut self) -> Self {
        self.config.show_tool_calls = true;
        self
    }

    pub fn show_state(mut self) -> Self {
        self.config.show_state_transitions = true;
        self.config.prompt = PromptStyle::WithState;
        self
    }

    pub fn show_timing(mut self) -> Self {
        self.config.show_timing = true;
        self
    }

    pub fn no_builtin_commands(mut self) -> Self {
        self.config.builtin_commands = false;
        self
    }

    /// Register a custom command handler, called before built-in commands.
    pub fn on_command<F>(mut self, handler: F) -> Self
    where
        F: Fn(&str, &RuntimeAgent) -> CommandResult + Send + Sync + 'static,
    {
        self.command_handler = Some(Box::new(handler));
        self
    }

    pub async fn run(self) -> ai_agents::Result<()> {
        if let Some(ref welcome) = self.config.welcome {
            println!("{}", welcome);
            println!();
        }

        let info = self.agent.info();
        println!("Agent: {} v{}", info.name, info.version);
        if let Some(state) = self.agent.current_state() {
            println!("State: {}", state);
        }

        if !self.config.hints.is_empty() {
            println!();
            for hint in &self.config.hints {
                println!("  {}", hint);
            }
        }

        println!();
        if self.config.builtin_commands {
            println!("Type '/help' for commands, '/quit' to exit.");
        } else {
            println!("Type '/quit' to exit.");
        }
        println!();

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            let prompt_str = match self.config.prompt {
                PromptStyle::Simple => "You > ".to_string(),
                PromptStyle::WithState => {
                    let state = self.agent.current_state().unwrap_or_else(|| "—".into());
                    format!("[{}] You > ", state)
                }
            };

            print!("{}", prompt_str);
            stdout.flush().ok();

            let mut input = String::new();
            if stdin.read_line(&mut input).is_err() {
                break;
            }
            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            // 1. Try the user-supplied custom command handler first
            if let Some(ref handler) = self.command_handler {
                if handler(input, &self.agent) == CommandResult::Handled {
                    continue;
                }
            }

            // 2. Try built-in commands
            match self.handle_command(input).await {
                Some(CommandAction::Quit) => {
                    println!("Goodbye!");
                    break;
                }
                Some(CommandAction::Continue) => continue,
                None => {}
            }

            // 3. Normal chat / streaming
            let start = std::time::Instant::now();

            match self.config.mode {
                ReplMode::Chat => self.handle_chat(input).await,
                ReplMode::Streaming => self.handle_streaming(input).await,
            }

            if self.config.show_timing {
                let elapsed = start.elapsed();
                println!("  ({:.1}s)", elapsed.as_secs_f64());
            }
        }

        Ok(())
    }

    async fn handle_command(&self, input: &str) -> Option<CommandAction> {
        if !self.config.builtin_commands {
            return match input.to_lowercase().as_str() {
                "/quit" | "/exit" => Some(CommandAction::Quit),
                _ => None,
            };
        }

        match input.to_lowercase().as_str() {
            "/quit" | "/exit" => Some(CommandAction::Quit),
            "/help" | "?" => {
                self.print_help();
                Some(CommandAction::Continue)
            }
            "/reset" => {
                if let Err(e) = self.agent.reset().await {
                    eprintln!("[Error] Reset failed: {}", e);
                } else {
                    println!("Agent reset.");
                    if let Some(state) = self.agent.current_state() {
                        println!("State: {}", state);
                    }
                }
                Some(CommandAction::Continue)
            }
            "/state" => {
                match self.agent.current_state() {
                    Some(state) => println!("Current state: {}", state),
                    None => println!("No state machine active."),
                }
                Some(CommandAction::Continue)
            }
            "/history" => {
                let history = self.agent.state_history();
                if history.is_empty() {
                    println!("No state transitions yet.");
                } else {
                    println!("State transitions:");
                    for event in &history {
                        println!("  {} → {} ({})", event.from, event.to, event.reason);
                    }
                }
                Some(CommandAction::Continue)
            }
            "/info" => {
                let info = self.agent.info();
                println!("Agent: {} v{}", info.name, info.version);
                if let Some(ref desc) = info.description {
                    println!("Description: {}", desc);
                }
                println!("Skills: {}", self.agent.skills().len());
                if let Some(state) = self.agent.current_state() {
                    println!("State: {}", state);
                }
                Some(CommandAction::Continue)
            }
            "/memory" | "/mem" => {
                self.print_memory_status().await;
                Some(CommandAction::Continue)
            }
            _ => None,
        }
    }

    fn print_help(&self) {
        println!("Commands:");
        println!("  /help, ?      Show this help message");
        println!("  /quit, /exit  Exit the REPL");
        println!("  /reset        Clear memory and reset state");
        println!("  /state        Show current state");
        println!("  /history      Show state transition history");
        println!("  /info         Show agent information");
        println!("  /memory, /mem Show memory status and token budget");
        println!();
    }

    async fn print_memory_status(&self) {
        let snapshot = match self.agent.save_state().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[Error] Failed to read memory: {}\n", e);
                return;
            }
        };

        let msg_count = snapshot.memory.messages.len();
        let has_summary = snapshot.memory.summary.is_some();

        println!("\n--- Memory Status ---");

        // Message counts
        if has_summary {
            println!("  Messages: {} recent + summary", msg_count);
        } else {
            println!("  Messages: {}", msg_count);
        }

        // Summary info
        if let Some(ref summary) = snapshot.memory.summary {
            let tokens = estimate_tokens(summary);
            let preview = if summary.len() > 80 {
                format!("{}...", &summary[..80])
            } else {
                summary.clone()
            };
            println!("  Summary:  {} tokens", tokens);
            println!("            \"{}\"", preview);
        } else {
            println!("  Summary:  none");
        }

        // Token budget display
        if let Some(budget) = self.agent.memory_token_budget() {
            let recent_tokens: u32 = snapshot
                .memory
                .messages
                .iter()
                .map(|m| estimate_message_tokens(m))
                .sum();
            let summary_tokens = snapshot
                .memory
                .summary
                .as_ref()
                .map(|s| estimate_tokens(s))
                .unwrap_or(0);
            let used = recent_tokens + summary_tokens;
            let pct = if budget.total > 0 {
                used as f64 / budget.total as f64 * 100.0
            } else {
                0.0
            };

            println!();
            println!("  Token Budget:");
            println!(
                "    Total:           {:>5} / {:>5} ({:.1}%)",
                used, budget.total, pct
            );

            let s_pct = if budget.allocation.summary > 0 {
                summary_tokens as f64 / budget.allocation.summary as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "    Summary:         {:>5} / {:>5} ({:.1}%)",
                summary_tokens, budget.allocation.summary, s_pct
            );

            let r_pct = if budget.allocation.recent_messages > 0 {
                recent_tokens as f64 / budget.allocation.recent_messages as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "    Recent messages: {:>5} / {:>5} ({:.1}%)",
                recent_tokens, budget.allocation.recent_messages, r_pct
            );

            println!(
                "    Facts:           {:>5} / {:>5} ( 0.0%)",
                0, budget.allocation.facts
            );

            println!("    Overflow:        {:?}", budget.overflow_strategy);
            println!("    Warning at:      {}%", budget.warn_at_percent);
        }

        println!("---------------------\n");
    }

    async fn handle_chat(&self, input: &str) {
        match self.agent.chat(input).await {
            Ok(response) => self.print_response(&response),
            Err(e) => eprintln!("\n[Error] {}\n", e),
        }
    }

    fn print_response(&self, response: &AgentResponse) {
        println!("\nAgent: {}\n", response.content);

        if self.config.show_tool_calls {
            if let Some(ref calls) = response.tool_calls {
                if !calls.is_empty() {
                    let names: Vec<&str> = calls.iter().map(|t| t.name.as_str()).collect();
                    println!("  Tools used: {}", names.join(", "));
                }
            }
        }
    }

    async fn handle_streaming(&self, input: &str) {
        match self.agent.chat_stream(input).await {
            Ok(mut stream) => {
                print!("\nAgent: ");
                io::stdout().flush().ok();

                while let Some(chunk) = stream.next().await {
                    match chunk {
                        StreamChunk::Content { text } => {
                            print!("{}", text);
                            io::stdout().flush().ok();
                        }
                        StreamChunk::ToolCallStart { name, .. } => {
                            if self.config.show_tool_calls {
                                print!("\n  [Tool: {}...", name);
                                io::stdout().flush().ok();
                            }
                        }
                        StreamChunk::ToolResult {
                            output, success, ..
                        } => {
                            if self.config.show_tool_calls {
                                if success {
                                    print!(" ✓]");
                                } else {
                                    print!(" ✗ {}]", output);
                                }
                                io::stdout().flush().ok();
                            }
                        }
                        StreamChunk::ToolCallEnd { .. } => {}
                        StreamChunk::ToolCallDelta { .. } => {}
                        StreamChunk::StateTransition { from, to } => {
                            if self.config.show_state_transitions {
                                let from_str = from.as_deref().unwrap_or("—");
                                print!("\n  [State: {} → {}]", from_str, to);
                                io::stdout().flush().ok();
                            }
                        }
                        StreamChunk::Done {} => {
                            println!("\n");
                            break;
                        }
                        StreamChunk::Error { message } => {
                            eprintln!("\n[Stream Error: {}]\n", message);
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("\n[Error] {}\n", e);
            }
        }
    }
}
