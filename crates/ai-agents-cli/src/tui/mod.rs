//! Bounded terminal presentation and shared foreground hosting have independent lifetimes.

pub mod app;
mod commands;
pub mod event;
mod host;
pub mod log_layer;
pub mod palette;
pub mod theme;
pub mod widgets;

use self::{
    app::{App, UpdateResult},
    event::AppMessage,
    log_layer::TuiLogLayer,
    palette::resolve_theme,
};
use crate::{
    repl::CliReplConfig,
    session::{Endpoint, FrontendSession, Input, Output},
};
use ai_agents::RuntimeAgent;
use anyhow::{Context, Result};
use crossterm::{
    event::{Event, EventStream},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{io, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tracing_subscriber::layer::SubscriberExt;

/// Restores terminal state even when partial setup or drawing fails; this is not task cleanup acknowledgement.
struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}

/// Hosts an already configured runtime; ordinary approval handlers remain the embedding caller's responsibility.
/// The CLI uses checked preparation and a bounded bridge before entering this function's presentation loop.
pub async fn run_tui(
    agent: RuntimeAgent,
    config: CliReplConfig,
    theme_name: Option<String>,
) -> Result<()> {
    let (session, input, output) = FrontendSession::embedded();
    agent.set_question_handler(Some(session.bridge()));
    run_prepared(agent, config, theme_name, session, input, output).await
}

/// Polls the original shared foreground driver through shutdown rather than dropping work when a screen closes.
pub(crate) async fn run_prepared(
    agent: RuntimeAgent,
    config: CliReplConfig,
    theme_name: Option<String>,
    mut session: FrontendSession,
    input: mpsc::Sender<Input>,
    mut output: mpsc::Receiver<Output>,
) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<AppMessage>(32);
    let log_level = std::env::var("RUST_LOG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(tracing::Level::WARN);
    let subscriber = tracing_subscriber::registry().with(TuiLogLayer::new(tx.clone(), log_level));
    let _ = tracing::subscriber::set_global_default(subscriber);
    enable_raw_mode().context("failed to enable raw mode")?;
    let _terminal_guard = TerminalGuard;
    execute!(io::stdout(), EnterAlternateScreen).context("failed to enter alternate screen")?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;
    let name = theme_name.unwrap_or_else(|| "dark".into());
    let theme = resolve_theme(&name).unwrap_or_else(|| resolve_theme("dark").unwrap());
    let agent = Arc::new(agent);
    let mut app = App::new(agent.clone(), config.clone(), tx, theme, name);
    app.attach_host(input.clone());
    let (endpoint, endpoint_rx) = tokio::sync::watch::channel(Endpoint::Open);
    session.set_endpoint(endpoint_rx);
    let mut driver = Box::pin(host::run(session, agent, config));
    let mut reader = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    let mut stopping = false;
    let mut output_open = true;
    let mut input_open = true;
    let mut draw_error = None;
    loop {
        if draw_error.is_none()
            && let Err(error) = terminal.draw(|frame| app.render(frame))
        {
            draw_error = Some(error);
            stopping = true;
            // A broken renderer is confirmed endpoint loss; close delivery without pretending its work is settled.
            output.close();
            let _ = endpoint.send(Endpoint::Shutdown);
        }
        tokio::select! {
            result=&mut driver => {
                // The original driver has returned an actual result or protected error; drain its bounded suffix first.
                while let Ok(event)=output.try_recv() {app.host_output(event);}
                if let Some(error)=draw_error {return Err(error.into());}
                result?;
                return Ok(());
            }
            event=output.recv(), if output_open => {
                match event {Some(event)=>app.host_output(event),None=>output_open=false}
            }
            event=reader.next(), if input_open => {
                match event {
                    Some(Ok(Event::Key(key))) if key.kind != crossterm::event::KeyEventKind::Release && !(key.kind == crossterm::event::KeyEventKind::Repeat && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) && key.code == crossterm::event::KeyCode::Char('c')) => {
                        if app.update(AppMessage::Key(key)).await==UpdateResult::Quit {
                            if stopping { anyhow::bail!("forced exit: task cleanup is unconfirmed; inspect ownership before recovery"); }
                            stopping=true;
                            let _=endpoint.send(Endpoint::Shutdown);
                        }
                    }
                    Some(Ok(Event::Resize(w,h))) => {app.update(AppMessage::Resize(w,h)).await;}
                    Some(Err(_)) | None => { input_open=false;let _=endpoint.send(Endpoint::Closed); }
                    _=>{}
                }
            }
            Some(messages)=app.commands.next(), if !app.commands.is_empty() => {app.command_output(messages);}
            Some(message)=rx.recv() => {app.update(message).await;}
            _=tick.tick() => {app.update(AppMessage::Tick).await;}
        }
    }
}
