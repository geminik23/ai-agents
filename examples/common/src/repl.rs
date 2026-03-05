use ai_agents::{Agent, AgentResponse, RuntimeAgent, StreamChunk};
use futures::StreamExt;
use std::io::{self, Write};
use std::sync::Arc;

/// Return value from a custom command handler.
pub enum CommandResult {
    /// The command was handled; skip further processing for this input.
    Handled,
    /// The input was not a recognised custom command; fall through to built-ins.
    NotHandled,
}

/// The interactive REPL wrapper around a `RuntimeAgent`.
pub struct Repl {
    agent: RuntimeAgent,
    config: ReplConfig,
}

/// Configuration for the REPL behaviour and display.
pub struct ReplConfig {
    /// Banner printed at startup
    pub welcome: Option<String>,
    /// How to format the input prompt.
    pub prompt: PromptStyle,
    /// Chat mode (full responses) vs streaming mode (token-by-token).
    pub mode: ReplMode,
    /// Whether to show tool calls in the output.
    pub show_tool_calls: bool,
    /// Whether to show state transitions in the output.
    pub show_state_transitions: bool,
    /// Display elapsed time per response.
    pub show_timing: bool,
    /// Whether to include built-in commands like "help" and "reset".
    pub builtin_commands: bool,
    /// Additional hints or instructions to show at startup.
    pub hints: Vec<String>,
    /// Optional callback invoked before built-in command handling.
    /// If the callback returns `CommandResult::Handled`, built-in handling is skipped.
    pub on_command:
        Option<Arc<dyn Fn(&str, &RuntimeAgent) -> CommandResult + Send + Sync>>,
    /// Optional callback invoked when the REPL is about to exit.
    pub on_quit: Option<Arc<dyn Fn(&RuntimeAgent) + Send + Sync>>,
}

/// Controls how the input prompt is rendered.
pub enum PromptStyle {
    /// `"You > "`
    Simple,
    /// `"[greeting] You > "`
    WithState,
    /// Custom prompt via closure.
    Custom(Box<dyn Fn(&RuntimeAgent) -> String + Send + Sync>),
}

/// Whether responses are returned in full or streamed token-by-token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplMode {
    Chat,
    Streaming,
}

enum CommandAction {
    Continue,
    Quit,
}

impl Repl {
    pub fn new(agent: RuntimeAgent) -> Self {
        Self {
            agent,
            config: ReplConfig {
                welcome: None,
                prompt: PromptStyle::Simple,
                mode: ReplMode::Chat,
                show_tool_calls: false,
                show_state_transitions: false,
                show_timing: false,
                builtin_commands: true,
                hints: vec![],
                on_command: None,
                on_quit: None,
            },
        }
    }

    /// Access the underlying agent (e.g. from an `on_command` callback).
    pub fn agent(&self) -> &RuntimeAgent {
        &self.agent
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

    /// Also sets `PromptStyle::WithState`.
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

    /// Register a custom command handler invoked before the built-in commands.
    ///
    /// The callback receives the raw input line and a reference to the agent.
    /// Return `CommandResult::Handled` to consume the input, or
    /// `CommandResult::NotHandled` to fall through to built-in handling.
    pub fn on_command(
        mut self,
        handler: impl Fn(&str, &RuntimeAgent) -> CommandResult + Send + Sync + 'static,
    ) -> Self {
        self.config.on_command = Some(Arc::new(handler));
        self
    }

    /// Register a callback that runs when the REPL is about to exit.
    pub fn on_quit(
        mut self,
        handler: impl Fn(&RuntimeAgent) + Send + Sync + 'static,
    ) -> Self {
        self.config.on_quit = Some(Arc::new(handler));
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
            println!("Type 'help' for commands, 'quit' to exit.");
        } else {
            println!("Type 'quit' to exit.");
        }
        println!();

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            let prompt_str = match &self.config.prompt {
                PromptStyle::Simple => "You > ".to_string(),
                PromptStyle::WithState => {
                    let state = self.agent.current_state().unwrap_or_else(|| "—".into());
                    format!("[{}] You > ", state)
                }
                PromptStyle::Custom(f) => f(&self.agent),
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

            // Try custom command handler first
            if let Some(ref handler) = self.config.on_command {
                if let CommandResult::Handled = handler(input, &self.agent) {
                    continue;
                }
            }

            match self.handle_command(input).await {
                Some(CommandAction::Quit) => {
                    if let Some(ref handler) = self.config.on_quit {
                        handler(&self.agent);
                    }
                    println!("Goodbye!");
                    break;
                }
                Some(CommandAction::Continue) => continue,
                None => {}
            }

            let start = std::time::Instant::now();

            match self.config.mode {
                ReplMode::Chat => {
                    self.handle_chat(input).await;
                }
                ReplMode::Streaming => {
                    self.handle_streaming(input).await;
                }
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
                "quit" | "exit" => Some(CommandAction::Quit),
                _ => None,
            };
        }

        match input.to_lowercase().as_str() {
            "quit" | "exit" => Some(CommandAction::Quit),

            "help" | "?" => {
                self.print_help();
                Some(CommandAction::Continue)
            }

            "reset" => {
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

            "state" => {
                match self.agent.current_state() {
                    Some(s) => println!("Current state: {}", s),
                    None => println!("No state machine active."),
                }
                Some(CommandAction::Continue)
            }

            "history" => {
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

            "info" => {
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

            _ => None,
        }
    }

    fn print_help(&self) {
        println!("Commands:");
        println!("  help, ?     Show this help message");
        println!("  quit, exit  Exit the REPL");
        println!("  reset       Clear memory and reset state");
        println!("  state       Show current state");
        println!("  history     Show state transition history");
        println!("  info        Show agent information");
        println!();
    }

    async fn handle_chat(&self, input: &str) {
        match self.agent.chat(input).await {
            Ok(response) => {
                self.print_response(&response);
            }
            Err(e) => {
                eprintln!("\n[Error] {}\n", e);
            }
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
