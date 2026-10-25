//! TUI transport for the shared foreground driver; it never invents task execution or cleanup policy.

#[cfg(test)]
#[path = "host_tests.rs"]
mod tests;

use crate::{
    repl::CliReplConfig,
    session::{FrontendSession, Output},
};
use ai_agents::{RuntimeAgent, TaskStreamOperation};
use anyhow::{Context, Result, bail};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Polls original work and its controls through the same foreground session used by the plain REPL.
/// The caller polls this future until return, including after display/input loss; dropping it is not cleanup acknowledgement.
pub(crate) async fn run(
    mut io: FrontendSession,
    agent: Arc<RuntimeAgent>,
    config: CliReplConfig,
) -> Result<()> {
    while let Some(line) = io.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "/quit" | "/exit") {
            return Ok(());
        }
        let (tx, rx) = mpsc::channel(32);
        let result = if let Some(rest) = line.strip_prefix("/task run ") {
            let parsed = if let Some(rest) = rest.strip_prefix("--profile ") {
                rest.split_once(' ')
                    .map(|(p, o)| (Some(p.to_owned()), o.to_owned()))
                    .context("expected /task run --profile ID OBJECTIVE")
            } else {
                Ok((None, rest.to_owned()))
            };
            match parsed {
                Ok((profile, objective)) => {
                    io.drive(
                        crate::session::task_work(
                            agent.clone(),
                            TaskStreamOperation::Run { objective, profile },
                            tx,
                            false,
                        ),
                        rx,
                    )
                    .await
                }
                Err(error) => {
                    io.send_ui(Output::Rejected {
                        generation: io.generation(),
                        message: format!("[input error] {error}"),
                    })
                    .await?;
                    continue;
                }
            }
        } else if line.starts_with("/task ") {
            io.send_ui(Output::Rejected {
                generation: io.generation(),
                message: "No foreground task owns this control; use /task run OBJECTIVE.".into(),
            })
            .await?;
            continue;
        } else {
            io.drive(
                crate::session::chat_work(agent.clone(), line.into(), config.clone(), tx),
                rx,
            )
            .await
        };
        match result {
            Ok(result) => {
                let quit = result.quit;
                if !io.output_suppressed() {
                    io.send_ui(Output::Result {
                        generation: io.generation(),
                        result: Box::new(result),
                    })
                    .await?;
                }
                if quit {
                    return Ok(());
                }
            }
            Err(error) => {
                if error.is::<crate::session::ProtectedSession>() || io.output_closed() {
                    return Err(error);
                }
                io.send_ui(Output::Rejected {
                    generation: io.generation(),
                    message: format!("[Error] {error}"),
                })
                .await?;
            }
        }
    }
    Ok(())
}

/// Queue admission never waits on the renderer's thread and never silently discards a user decision.
pub(crate) fn send(
    input: &mpsc::Sender<crate::session::Input>,
    value: crate::session::Input,
) -> Result<()> {
    if input.try_send(value).is_err() {
        bail!("input queue full or closed; decision was not submitted");
    }
    Ok(())
}
