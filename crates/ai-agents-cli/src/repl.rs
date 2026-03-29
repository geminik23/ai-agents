use std::io::{self, Write};
use std::sync::Arc;

use ai_agents::memory::{estimate_message_tokens, estimate_tokens};
use ai_agents::spec::AgentSpec;
use ai_agents::{Agent, AgentResponse, RuntimeAgent, StreamChunk};
use futures::StreamExt;

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

        // Commands that need the original input (preserve agent ID casing).
        let lower = input.to_lowercase();
        if lower.starts_with("/save") {
            self.handle_save(input).await;
            return Some(CommandAction::Continue);
        }
        if lower.starts_with("/load") {
            self.handle_load(input).await;
            return Some(CommandAction::Continue);
        }
        if lower.starts_with("/delete") {
            self.handle_delete(input).await;
            return Some(CommandAction::Continue);
        }

        match lower.as_str() {
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
                        println!("  {} -> {} ({})", event.from, event.to, event.reason);
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
                if self.agent.has_spawner() {
                    if let Some(registry) = self.agent.spawner_registry() {
                        println!("Spawned agents: {}", registry.list().len());
                    }
                }
                Some(CommandAction::Continue)
            }
            "/memory" | "/mem" => {
                self.print_memory_status().await;
                Some(CommandAction::Continue)
            }
            "/sessions" => {
                self.handle_sessions().await;
                Some(CommandAction::Continue)
            }
            _ => None,
        }
    }

    fn print_help(&self) {
        println!("Commands:");
        println!("  /help, ?             Show this help message");
        println!("  /quit, /exit         Exit the REPL");
        println!("  /reset               Clear memory and reset state");
        println!("  /state               Show current state");
        println!("  /history             Show state transition history");
        println!("  /info                Show agent information");
        println!("  /memory, /mem        Show memory status and token budget");
        println!("  /save [name]         Save session (parent + all spawned agents)");
        println!("  /save self [name]    Save parent session only");
        println!("  /save agent <id>     Save one spawned agent's session");
        println!("  /load [name]         Load session (parent + restore spawned agents)");
        println!("  /load self [name]    Load parent session only");
        println!("  /load agent <id>     Load one spawned agent's session");
        println!("  /sessions            List saved sessions");
        println!("  /delete <name>       Delete a saved session");
        println!();
    }

    // Session persistence commands

    /// Ensure storage is initialized. Returns true if storage is available.
    async fn ensure_storage(&self) -> bool {
        if self.agent.storage().is_some() {
            return true;
        }
        if let Err(e) = self.agent.init_storage().await {
            eprintln!("[Error] Failed to init storage: {}", e);
            return false;
        }
        if self.agent.storage().is_some() {
            return true;
        }
        eprintln!("No storage configured. Add a storage: section to the YAML.");
        false
    }

    async fn handle_save(&self, input: &str) {
        if !self.ensure_storage().await {
            return;
        }
        let (scope, name) = match parse_save_load_args(input) {
            Some(parsed) => parsed,
            None => return,
        };
        match scope {
            SaveScope::All => self.save_all(&name).await,
            SaveScope::SelfOnly => self.save_self(&name).await,
            SaveScope::Agent(id) => self.save_agent(&id, &name).await,
        }
    }

    async fn save_all(&self, name: &str) {
        // Save parent with full snapshot (includes registry manifest).
        let snapshot = match self.agent.save_state_full().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[Error] Failed to build snapshot: {}", e);
                return;
            }
        };
        if let Some(storage) = self.agent.storage() {
            if let Err(e) = storage.save(name, &snapshot).await {
                eprintln!("[Error] Failed to save parent session: {}", e);
                return;
            }
        }

        // Save each child agent's session.
        let mut child_count = 0;
        if let Some(registry) = self.agent.spawner_registry() {
            for info in registry.list() {
                if let Some(agent) = registry.get(&info.id) {
                    let _ = agent.init_storage().await;
                    if let Err(e) = agent.save_session(name).await {
                        eprintln!("  [Warn] Failed to save {}: {}", info.id, e);
                        continue;
                    }
                    child_count += 1;
                }
            }
        }

        if child_count > 0 {
            println!("Saved session '{}' (parent + {} agents)", name, child_count);
        } else {
            println!("Saved session '{}'", name);
        }
    }

    async fn save_self(&self, name: &str) {
        if let Err(e) = self.agent.save_session(name).await {
            eprintln!("[Error] Failed to save session: {}", e);
            return;
        }
        println!("Saved session '{}' (parent only)", name);
    }

    async fn save_agent(&self, id: &str, name: &str) {
        let registry = match self.agent.spawner_registry() {
            Some(r) => r,
            None => {
                eprintln!("No spawner configured.");
                return;
            }
        };
        let agent = match registry.get(id) {
            Some(a) => a,
            None => {
                eprintln!("Agent not found: {}", id);
                return;
            }
        };
        let _ = agent.init_storage().await;
        if let Err(e) = agent.save_session(name).await {
            eprintln!("[Error] Failed to save {}: {}", id, e);
            return;
        }
        println!("Saved agent '{}' session '{}'", id, name);
    }

    async fn handle_load(&self, input: &str) {
        if !self.ensure_storage().await {
            return;
        }
        let (scope, name) = match parse_save_load_args(input) {
            Some(parsed) => parsed,
            None => return,
        };
        match scope {
            SaveScope::All => self.load_all(&name).await,
            SaveScope::SelfOnly => self.load_self(&name).await,
            SaveScope::Agent(id) => self.load_agent(&id, &name).await,
        }
    }

    async fn load_all(&self, name: &str) {
        // Peek at the parent snapshot to read the registry manifest before restoring.
        let manifest = {
            let storage = match self.agent.storage() {
                Some(s) => s,
                None => {
                    eprintln!("No storage available.");
                    return;
                }
            };
            match storage.load(name).await {
                Ok(Some(snapshot)) => snapshot.spawned_agents.clone(),
                Ok(None) => {
                    eprintln!("Session '{}' not found.", name);
                    return;
                }
                Err(e) => {
                    eprintln!("[Error] Failed to load session: {}", e);
                    return;
                }
            }
        };

        // Restore parent state.
        match self.agent.load_session(name).await {
            Ok(true) => {}
            Ok(false) => {
                eprintln!("Session '{}' not found.", name);
                return;
            }
            Err(e) => {
                eprintln!("[Error] Failed to load parent session: {}", e);
                return;
            }
        }

        // Recreate spawned agents from manifest.
        let mut child_count = 0;
        if let (Some(entries), Some(spawner), Some(registry)) = (
            manifest,
            self.agent.spawner(),
            self.agent.spawner_registry(),
        ) {
            for entry in &entries {
                // If agent already exists in registry, just restore its session.
                if let Some(agent) = registry.get(&entry.id) {
                    let _ = agent.init_storage().await;
                    let _ = agent.load_session(name).await;
                    child_count += 1;
                    continue;
                }

                // Recreate from saved spec YAML with the original agent ID.
                let spec: AgentSpec = match serde_yaml::from_str(&entry.spec_yaml) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("  [Warn] Failed to parse spec for {}: {}", entry.id, e);
                        continue;
                    }
                };
                match spawner.spawn_with_id(entry.id.clone(), spec).await {
                    Ok(agent) => {
                        let agent_handle = Arc::clone(&agent.agent);
                        let agent_id = agent.id.clone();
                        if let Err(e) = registry.register(agent).await {
                            eprintln!("  [Warn] Failed to register {}: {}", entry.id, e);
                            continue;
                        }
                        // Restore child's session from NamespacedStorage.
                        let _ = agent_handle.init_storage().await;
                        match agent_handle.load_session(name).await {
                            Ok(_) => child_count += 1,
                            Err(e) => {
                                eprintln!("  [Warn] Failed to restore {}: {}", agent_id, e);
                                child_count += 1;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  [Warn] Failed to recreate {}: {}", entry.id, e);
                    }
                }
            }
        }

        if child_count > 0 {
            println!(
                "Loaded session '{}' (parent + {} agents)",
                name, child_count
            );
        } else {
            println!("Loaded session '{}'", name);
        }
    }

    async fn load_self(&self, name: &str) {
        match self.agent.load_session(name).await {
            Ok(true) => println!("Loaded session '{}' (parent only)", name),
            Ok(false) => eprintln!("Session '{}' not found.", name),
            Err(e) => eprintln!("[Error] Failed to load session: {}", e),
        }
    }

    async fn load_agent(&self, id: &str, name: &str) {
        let registry = match self.agent.spawner_registry() {
            Some(r) => r,
            None => {
                eprintln!("No spawner configured.");
                return;
            }
        };
        let agent = match registry.get(id) {
            Some(a) => a,
            None => {
                eprintln!("Agent not found: {}. Must be registered first.", id);
                return;
            }
        };
        let _ = agent.init_storage().await;
        match agent.load_session(name).await {
            Ok(true) => println!("Loaded agent '{}' session '{}'", id, name),
            Ok(false) => eprintln!("No saved session '{}' for agent '{}'", name, id),
            Err(e) => eprintln!("[Error] Failed to load {}: {}", id, e),
        }
    }

    async fn handle_sessions(&self) {
        if !self.ensure_storage().await {
            return;
        }
        match self.agent.list_sessions().await {
            Ok(sessions) => {
                if sessions.is_empty() {
                    println!("No saved sessions.");
                    return;
                }
                let mut parent_keys = Vec::new();
                let mut child_keys = Vec::new();
                for s in &sessions {
                    if s.contains('/') {
                        child_keys.push(s.as_str());
                    } else {
                        parent_keys.push(s.as_str());
                    }
                }
                println!("Saved sessions:");
                for s in &parent_keys {
                    println!("  {}", s);
                }
                if !child_keys.is_empty() {
                    println!("Agent sessions:");
                    for s in &child_keys {
                        println!("  {}", s);
                    }
                }
            }
            Err(e) => eprintln!("[Error] {}", e),
        }
    }

    async fn handle_delete(&self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let name = match parts.get(1) {
            Some(n) => n.to_string(),
            None => {
                eprintln!("Usage: /delete <session_name>");
                return;
            }
        };

        if !self.ensure_storage().await {
            return;
        }

        // Read manifest from saved snapshot so we can delete child sessions
        // even if those agents are no longer in the registry.
        let manifest_ids: Vec<String> = if let Some(storage) = self.agent.storage() {
            match storage.load(&name).await {
                Ok(Some(snapshot)) => snapshot
                    .spawned_agents
                    .unwrap_or_default()
                    .iter()
                    .map(|e| e.id.clone())
                    .collect(),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };

        if let Err(e) = self.agent.delete_session(&name).await {
            eprintln!("[Error] Failed to delete session: {}", e);
            return;
        }

        // Delete child sessions using the manifest.
        let mut child_count = 0;
        if let Some(storage) = self.agent.storage() {
            for id in &manifest_ids {
                let child_key = format!("{}/{}", id, name);
                let _ = storage.delete(&child_key).await;
                child_count += 1;
            }
        }

        if child_count > 0 {
            println!(
                "Deleted session '{}' (parent + {} agents)",
                name, child_count
            );
        } else {
            println!("Deleted session '{}'", name);
        }
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

// Command parsing helpers

enum SaveScope {
    All,
    SelfOnly,
    Agent(String),
}

/// Parse /save and /load arguments into (scope, session_name).
fn parse_save_load_args(input: &str) -> Option<(SaveScope, String)> {
    let parts: Vec<&str> = input.split_whitespace().collect();
    // parts[0] = "/save" or "/load"
    match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
        None => Some((SaveScope::All, "default".to_string())),
        Some("self") => {
            let name = parts.get(2).unwrap_or(&"default").to_string();
            Some((SaveScope::SelfOnly, name))
        }
        Some("agent") => {
            let id = match parts.get(2) {
                Some(id) if !id.is_empty() => id.to_string(),
                _ => {
                    eprintln!("Usage: {} agent <id> [name]", parts[0]);
                    return None;
                }
            };
            let name = parts
                .get(3)
                .map(|s| s.to_string())
                .unwrap_or_else(|| id.clone());
            Some((SaveScope::Agent(id), name))
        }
        Some(name) => Some((SaveScope::All, name.to_string())),
    }
}
