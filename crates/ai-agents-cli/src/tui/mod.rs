//
// TUI module: ratatui-based terminal interface with panels and streaming.
//

pub mod app;
pub mod event;
pub mod log_layer;
pub mod palette;
pub mod theme;
pub mod widgets;

use std::io;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tracing_subscriber::layer::SubscriberExt;

use ai_agents::RuntimeAgent;

use crate::repl::CliReplConfig;

use self::app::{App, UpdateResult};
use self::event::AppMessage;
use self::log_layer::TuiLogLayer;
use self::palette::resolve_theme;

/// Run the ratatui TUI event loop.
pub async fn run_tui(
    agent: RuntimeAgent,
    config: CliReplConfig,
    theme_name: Option<String>,
) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AppMessage>();

    // Install a global tracing subscriber that captures logs into the TUI channel.
    // This must happen BEFORE enable_raw_mode so no fmt subscriber ever writes to the terminal.
    // Default to WARN to avoid flooding the chat with per-turn INFO lines.
    let log_level = std::env::var("RUST_LOG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(tracing::Level::WARN);
    let log_layer = TuiLogLayer::new(tx.clone(), log_level);
    let subscriber = tracing_subscriber::registry().with(log_layer);
    let _ = tracing::subscriber::set_global_default(subscriber);

    // Initialize storage eagerly so fact_store is ready before the first turn.
    let _ = agent.init_storage().await;

    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;
    terminal.clear().context("failed to clear terminal")?;

    // Resolve the initial theme. CLI flag / YAML metadata -> fallback to "dark".
    let initial_theme_name = theme_name.unwrap_or_else(|| "dark".to_string());
    let initial_theme = resolve_theme(&initial_theme_name).unwrap_or_else(|| {
        // Unknown name: fall back to dark and show a toast later.
        resolve_theme("dark").unwrap()
    });

    // Spawn async terminal event reader using crossterm event-stream.
    let tx_keys = tx.clone();
    tokio::spawn(async move {
        let mut reader = EventStream::new();
        while let Some(event_result) = reader.next().await {
            match event_result {
                Ok(Event::Key(key)) => {
                    if tx_keys.send(AppMessage::Key(key)).is_err() {
                        break;
                    }
                }
                Ok(Event::Resize(w, h)) => {
                    if tx_keys.send(AppMessage::Resize(w, h)).is_err() {
                        break;
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });

    // Spawn tick timer for spinner animation and toast expiry.
    let tx_tick = tx.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            interval.tick().await;
            if tx_tick.send(AppMessage::Tick).is_err() {
                break;
            }
        }
    });

    let agent_arc = Arc::new(agent);
    let mut app = App::new(agent_arc, config, tx, initial_theme, initial_theme_name);

    // Main event loop.
    loop {
        terminal
            .draw(|frame| app.render(frame))
            .context("failed to draw frame")?;

        if let Some(msg) = rx.recv().await {
            if app.update(msg).await == UpdateResult::Quit {
                break;
            }
        } else {
            break;
        }
    }

    // Teardown.
    disable_raw_mode().context("failed to disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("failed to leave alternate screen")?;
    terminal.show_cursor().context("failed to show cursor")?;

    Ok(())
}
