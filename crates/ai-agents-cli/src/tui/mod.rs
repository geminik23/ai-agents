//
// TUI module: ratatui-based terminal interface with panels and streaming.
//

pub mod app;
pub mod event;
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
use tracing_subscriber::fmt::writer::MakeWriterExt;

use ai_agents::RuntimeAgent;

use crate::repl::CliReplConfig;

use self::app::{App, UpdateResult};
use self::event::AppMessage;

/// Run the ratatui TUI event loop.
pub async fn run_tui(agent: RuntimeAgent, config: CliReplConfig) -> Result<()> {
    // Replace the global tracing subscriber with one that writes to stderr
    // so log output does not corrupt the alternate screen.
    let stderr_subscriber = tracing_subscriber::fmt()
        .with_writer(std::io::stderr.with_max_level(tracing::Level::WARN))
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "ai_agents=warn".to_string()))
        .finish();
    let _guard = tracing::subscriber::set_default(stderr_subscriber);

    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AppMessage>();

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
    let mut app = App::new(agent_arc, config, tx);

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
