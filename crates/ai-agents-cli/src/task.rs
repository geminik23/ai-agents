//! CLI task commands are adapters over configured runtime hosting and scoped store-only controls.

use crate::{
    cli::{TaskAgentArgs, TaskArgs, TaskCommand},
    run::RunOptions,
    session::{FrontendSession, OperationResult, task_work},
};
use ai_agents::{
    CancellationOutcome, TaskHostScope, TaskRunResult, TaskRunStatus, TaskStreamOperation,
};
use anyhow::{Context, Result};
use std::{path::Path, sync::Arc};
use tokio::sync::mpsc;

#[derive(Debug)]
pub struct TaskExit(pub u8);
impl std::fmt::Display for TaskExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "task exit {}", self.0)
    }
}
impl std::error::Error for TaskExit {}

pub async fn run(args: TaskArgs) -> Result<()> {
    let result = run_inner(args).await;
    match result {
        Ok(0) => Ok(()),
        Ok(code) => Err(TaskExit(code).into()),
        Err(error) if error.downcast_ref::<TaskExit>().is_some() => Err(error),
        Err(error) => {
            eprintln!("Task error: {error:#}");
            Err(TaskExit(2).into())
        }
    }
}

fn scope(host: &TaskAgentArgs) -> Result<TaskHostScope> {
    let prepared = crate::run::load_prepared_spec(&host.agent)?;
    TaskHostScope::from_prepared_spec(
        &prepared,
        host.agent.parent().unwrap_or(Path::new(".")),
        host.actor.clone(),
    )
    .map_err(Into::into)
}

// Store-only operations inspect both exact current and legacy namespaces without resuming or reinterpreting saved limits.
async fn management_stores(
    host: &TaskAgentArgs,
) -> Result<Vec<Arc<ai_agents::autonomy::ScopedTaskRunStore>>> {
    let prepared = crate::run::load_prepared_spec(&host.agent)?;
    let base = host.agent.parent().unwrap_or(Path::new("."));
    let current = TaskHostScope::from_prepared_spec(&prepared, base, host.actor.clone())?;
    let legacy = TaskHostScope::from_spec(prepared.spec(), base, host.actor.clone())?;
    let same = current.identity() == legacy.identity();
    let mut stores = vec![current.open_store().await?];
    if !same {
        stores.push(legacy.open_store().await?);
    }
    Ok(stores)
}

// Never guess a namespace from an untrusted run ID; only one authorized exact-scope candidate may be used.
async fn management_store(
    host: &TaskAgentArgs,
    id: &str,
) -> Result<Arc<ai_agents::autonomy::ScopedTaskRunStore>> {
    let mut selected = None;
    for store in management_stores(host).await? {
        if store
            .list_compatible()
            .await?
            .iter()
            .any(|row| row.key.run_id == id)
        {
            if selected.is_some() {
                anyhow::bail!("task matches multiple authorized namespaces");
            }
            selected = Some(store);
        }
    }
    selected.ok_or_else(|| {
        anyhow::anyhow!("task was not found in the current or compatible legacy namespace")
    })
}

async fn run_inner(args: TaskArgs) -> Result<u8> {
    match args.command {
        TaskCommand::Run(args) => {
            let mut io = FrontendSession::terminal(args.host.json, !args.non_interactive);
            let options = RunOptions {
                agent_path: args.host.agent.clone(),
                welcome: None,
                extra_hints: vec![],
                stream: None,
                show_tools: None,
                show_state: None,
                show_timing: None,
                no_builtins: false,
                contexts: args.contexts,
                context_file: args.context_file,
                actor: args.host.actor.clone(),
                plain: true,
                theme: None,
            };
            let prepared = crate::run::load_prepared_spec(&options.agent_path)?;
            let overrides = args
                .overrides
                .as_ref()
                .map(|path| {
                    let yaml = std::fs::read_to_string(path)?;
                    ai_agents::autonomy::parse_task_overrides(&yaml)
                        .map(Box::new)
                        .map_err(anyhow::Error::from)
                })
                .transpose()?;
            preflight_run(
                prepared.spec(),
                args.profile.as_deref(),
                &args.objective,
                overrides.as_deref(),
            )?;
            let namespace = scope(&args.host)?;
            let agent = crate::run::build_frontend_agent(
                &options.agent_path,
                prepared,
                io.bridge(),
                Some(namespace),
            )
            .await?;
            crate::run::prepare_frontend(&agent, &options).await?;
            let agent = Arc::new(agent);
            let (tx, rx) = mpsc::channel(32);
            let result = io
                .drive(
                    task_work(
                        agent,
                        match overrides {
                            Some(overrides) => TaskStreamOperation::RunWithOverrides {
                                objective: args.objective,
                                profile: args.profile,
                                overrides,
                            },
                            None => TaskStreamOperation::Run {
                                objective: args.objective,
                                profile: args.profile,
                            },
                        },
                        tx,
                        args.host.json,
                    ),
                    rx,
                )
                .await?;
            let OperationResult::Task(initial) = result.primary else {
                unreachable!()
            };
            let final_result = result
                .results
                .get(&initial.run.key.run_id)
                .unwrap_or(&initial);
            let text = result_text(final_result, args.host.json)?;
            if io.output_suppressed() {
                eprintln!(
                    "Task {} revision {}: {:?}; stdout delivery suppressed during shutdown",
                    final_result.run.key.run_id, final_result.run.revision, final_result.run.status
                );
            } else {
                io.write(text).await?;
                io.finish().await?;
            }
            Ok(result_code(
                final_result,
                result.prepared_pauses.get(&final_result.run.key.run_id)
                    == Some(&final_result.run.revision),
            ))
        }
        TaskCommand::List(host) => {
            let mut rows = Vec::new();
            for store in management_stores(&host).await? {
                rows.extend(store.list_compatible().await?);
            }
            rows.sort_by(|a, b| a.key.run_id.cmp(&b.key.run_id));
            if host.json {
                println!("{}", serde_json::to_string(&rows)?);
            } else {
                for row in rows {
                    println!("{}\t{}\t{:?}", row.key.run_id, row.revision, row.status);
                }
            }
            Ok(0)
        }
        TaskCommand::Show(args) => {
            let store = management_store(&args.host, &args.run_id).await?;
            let view = ai_agents::TaskRunDisplay::read(store.as_ref(), &args.run_id).await?;
            println!("{}", serde_json::to_string_pretty(&view)?);
            Ok(0)
        }
        TaskCommand::Cancel(args) => {
            let revision = args
                .revision
                .context("cancel requires --revision from an observed snapshot")?;
            let store = management_store(&args.target.host, &args.target.run_id).await?;
            let receipt =
                CancellationOutcome::request(store.as_ref(), &args.target.run_id, revision).await?;
            if args.target.host.json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                println!(
                    "{:?}: {} revision {} (a request is not remote cleanup acknowledgement)",
                    receipt.disposition, receipt.run.key.run_id, receipt.run.revision
                );
            }
            Ok(0)
        }
        TaskCommand::Cleanup(args) => {
            let revision = args
                .revision
                .context("cleanup requires --revision from an observed cancellation receipt")?;
            let prepared = crate::run::load_prepared_spec(&args.target.host.agent)?;
            let timeout = prepared
                .spec()
                .autonomy
                .restoration_timeout_ms
                .context("configured cleanup requires autonomy.restoration_timeout_ms")?;
            let host = Arc::new(ai_agents::autonomy::ConfiguredTaskRestoreHost::new(
                prepared,
                args.target
                    .host
                    .agent
                    .parent()
                    .unwrap_or(Path::new("."))
                    .into(),
                args.target.host.actor.clone(),
                std::time::Duration::from_millis(timeout.get()),
            )?);
            let cancelled = host.cancel_prepared(args.target.run_id, revision).await?;
            print!("{}", result_text(&cancelled, args.target.host.json)?);
            Ok(if cancelled.run.status == TaskRunStatus::Cancelled {
                0
            } else {
                1
            })
        }
        TaskCommand::Recover(args) => {
            let prepared = crate::run::load_prepared_spec(&args.target.host.agent)?;
            let timeout = prepared
                .spec()
                .autonomy
                .restoration_timeout_ms
                .context("configured task recovery requires autonomy.restoration_timeout_ms")?;
            let host = Arc::new(ai_agents::autonomy::ConfiguredTaskRestoreHost::new(
                prepared,
                args.target
                    .host
                    .agent
                    .parent()
                    .unwrap_or(Path::new("."))
                    .into(),
                args.target.host.actor.clone(),
                std::time::Duration::from_millis(timeout.get()),
            )?);
            let disposition = match args.action {
                crate::cli::TaskRecoveryAction::ResumePrepared => {
                    ai_agents::autonomy::TaskRecoveryDisposition::ResumePrepared
                }
                crate::cli::TaskRecoveryAction::CancelPrepared => {
                    ai_agents::autonomy::TaskRecoveryDisposition::CancelPrepared
                }
            };
            let recovered = host
                .recover_prepared(args.target.run_id, args.revision, disposition, args.reason)
                .await?;
            if matches!(
                disposition,
                ai_agents::autonomy::TaskRecoveryDisposition::CancelPrepared
            ) {
                let cancelled = host
                    .cancel_prepared(recovered.key.run_id, recovered.revision)
                    .await?;
                print!("{}", result_text(&cancelled, args.target.host.json)?);
                return Ok(if cancelled.run.status == TaskRunStatus::Cancelled {
                    0
                } else {
                    1
                });
            } else if args.target.host.json {
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "type":"recovery", "run_id": recovered.key.run_id, "revision": recovered.revision, "status": recovered.status,
                    }))?
                );
            } else {
                println!(
                    "Task {} revision {}: {:?}; resume still requires fresh review",
                    recovered.key.run_id, recovered.revision, recovered.status
                );
            }
            Ok(0)
        }
        TaskCommand::Resume(args) => {
            let revision = args
                .revision
                .context("resume requires --revision from an observed snapshot")?;
            let namespace = scope(&args.target.host)?;
            let prepared = crate::run::load_prepared_spec(&args.target.host.agent)?;
            let timeout = prepared
                .spec()
                .autonomy
                .restoration_timeout_ms
                .context("configured task resume requires autonomy.restoration_timeout_ms")?;
            let mut io = FrontendSession::terminal(args.target.host.json, true);
            let host = crate::run::configured_restore_host(
                &args.target.host.agent,
                prepared,
                io.bridge(),
                Some(namespace),
                timeout,
            )?;
            let (tx, rx) = mpsc::channel(32);
            let id = args.target.run_id;
            let result = io
                .drive(
                    async move {
                        let (_runner, paused) = host.restore(id, revision).await?;
                        drop(tx);
                        Ok(OperationResult::Task(Box::new(paused)))
                    },
                    rx,
                )
                .await?;
            let OperationResult::Task(initial) = result.primary else {
                unreachable!()
            };
            let final_result = result
                .results
                .get(&initial.run.key.run_id)
                .unwrap_or(&initial);
            if io.output_suppressed() {
                eprintln!(
                    "Task {} revision {}: {:?}; stdout delivery suppressed during shutdown",
                    final_result.run.key.run_id, final_result.run.revision, final_result.run.status
                );
            } else {
                io.write(result_text(final_result, args.target.host.json)?)
                    .await?;
                io.finish().await?;
            }
            Ok(result_code(
                final_result,
                result.prepared_pauses.get(&final_result.run.key.run_id)
                    == Some(&final_result.run.revision),
            ))
        }
    }
}

/// Resolves the selected profile and validates configured restoration prerequisites before provider or storage preparation.
pub(crate) fn preflight_run(
    spec: &ai_agents::spec::AgentSpec,
    selected: Option<&str>,
    objective: &str,
    overrides: Option<&ai_agents::autonomy::AutonomyProfile>,
) -> Result<()> {
    let profile = ai_agents::autonomy::resolve_profile(
        &spec.autonomy,
        ai_agents::autonomy::AutonomyScope::Task,
        selected,
        None,
        overrides,
        Some(objective),
        &Default::default(),
    )?;
    if !profile.enabled {
        anyhow::bail!("selected autonomy profile is disabled");
    }
    if profile
        .settings
        .persistence
        .as_ref()
        .is_some_and(|policy| policy.enabled != Some(false))
    {
        if spec.autonomy.restoration_timeout_ms.is_none() {
            anyhow::bail!(
                "durable task execution requires autonomy.restoration_timeout_ms for the configured host's work budget"
            );
        }
        if !matches!(spec.storage, ai_agents::spec::StorageConfig::Sqlite(_)) {
            anyhow::bail!("durable task execution requires SQLite task storage");
        }
        if spec
            .tools
            .iter()
            .flatten()
            .any(|tool| tool.to_mcp_config().is_some())
        {
            anyhow::bail!(
                "durable MCP hosting requires a Rust component state binding; the CLI cannot infer server state from registration"
            );
        }
        ai_agents::autonomy::SystemTaskClock::sample()?;
    }
    Ok(())
}

// Initial and reconstructed tasks expose the same bounded result shape, never the sensitive restoration checkpoint.
fn result_text(result: &TaskRunResult, json_mode: bool) -> Result<String> {
    if json_mode {
        Ok(format!(
            "{}\n",
            serde_json::to_string(&serde_json::json!({
                "type":"result", "run_id": result.run.key.run_id, "revision": result.run.revision,
                "status": result.run.status, "counters": result.run.counters,
            }))?
        ))
    } else {
        Ok(format!(
            "\nTask {} revision {}: {:?}\n{}\n",
            result.run.key.run_id,
            result.run.revision,
            result.run.status,
            result
                .final_response
                .as_ref()
                .map_or("", |response| response.content.as_str())
        ))
    }
}

// A Paused status alone cannot claim the reserved durable-exit code; the foreground host must confirm that exact revision.
pub(crate) fn result_code(result: &TaskRunResult, prepared_pause: bool) -> u8 {
    match result.run.status {
        TaskRunStatus::Completed => 0,
        TaskRunStatus::Paused if prepared_pause => 3,
        _ => 1,
    }
}
