use std::io::{self, Write};
use std::sync::Arc;

use ai_agents::tools::{QuestionHandler, QuestionRequest, QuestionResponse};
use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;

use crate::tui::event::{AppMessage, PendingQuestion};

/// Plain terminal question handler for the `ask_user` tool.
pub struct CliQuestionHandler;

impl CliQuestionHandler {
    /// Create a question handler that reads from stdin.
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl QuestionHandler for CliQuestionHandler {
    async fn ask_question(&self, request: QuestionRequest) -> QuestionResponse {
        tokio::task::spawn_blocking(move || prompt_question(request))
            .await
            .unwrap_or_else(|_| QuestionResponse {
                answered: false,
                selected: Vec::new(),
                other_text: None,
                timed_out: false,
                unavailable: true,
            })
    }
}

/// TUI question handler that sends modal requests to the app event loop.
#[derive(Clone)]
pub struct TuiQuestionHandler {
    tx: UnboundedSender<AppMessage>,
}

impl TuiQuestionHandler {
    /// Create a TUI handler backed by the app message channel.
    pub fn new(tx: UnboundedSender<AppMessage>) -> Arc<Self> {
        Arc::new(Self { tx })
    }
}

#[async_trait]
impl QuestionHandler for TuiQuestionHandler {
    async fn ask_question(&self, request: QuestionRequest) -> QuestionResponse {
        let (respond_to, receive) = oneshot::channel();
        if self
            .tx
            .send(AppMessage::Question(PendingQuestion {
                request,
                respond_to,
            }))
            .is_err()
        {
            return QuestionResponse {
                answered: false,
                selected: Vec::new(),
                other_text: None,
                timed_out: false,
                unavailable: true,
            };
        }
        receive.await.unwrap_or_else(|_| QuestionResponse {
            answered: false,
            selected: Vec::new(),
            other_text: None,
            timed_out: false,
            unavailable: true,
        })
    }
}

fn prompt_question(request: QuestionRequest) -> QuestionResponse {
    println!();
    println!("+-----------------------------------------+");
    println!("|              QUESTION                   |");
    println!("+-----------------------------------------+");
    println!("  {}", request.question);
    if !request.options.is_empty() {
        println!();
        for (index, option) in request.options.iter().enumerate() {
            println!("  {}. {}", index + 1, option);
        }
    }
    println!("+-----------------------------------------+");
    if request.multi_select {
        print!("  Select options by number, comma-separated");
    } else if request.options.is_empty() || request.allow_other {
        print!("  Answer");
    } else {
        print!("  Select option by number");
    }
    if request.default.is_some() {
        print!(" [default available]");
    }
    print!(": ");
    io::stdout().flush().unwrap_or_default();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or_default();
    let input = input.trim();
    if input.is_empty() {
        return response_from_default(request.default, false);
    }

    if !request.options.is_empty() {
        let selected = if request.multi_select {
            input
                .split(',')
                .filter_map(|part| option_by_index(part.trim(), &request.options))
                .collect::<Vec<_>>()
        } else {
            option_by_index(input, &request.options)
                .into_iter()
                .collect()
        };
        if !selected.is_empty() {
            return QuestionResponse {
                answered: true,
                selected,
                other_text: None,
                timed_out: false,
                unavailable: false,
            };
        }
    }

    if request.allow_other {
        return QuestionResponse {
            answered: true,
            selected: Vec::new(),
            other_text: Some(input.to_string()),
            timed_out: false,
            unavailable: false,
        };
    }

    response_from_default(request.default, false)
}

fn option_by_index(value: &str, options: &[String]) -> Option<String> {
    value
        .parse::<usize>()
        .ok()
        .and_then(|index| options.get(index.saturating_sub(1)).cloned())
}

pub(crate) fn response_from_default(
    default: Option<serde_json::Value>,
    timed_out: bool,
) -> QuestionResponse {
    match default {
        Some(serde_json::Value::String(text)) => QuestionResponse {
            answered: true,
            selected: vec![text],
            other_text: None,
            timed_out,
            unavailable: false,
        },
        Some(serde_json::Value::Array(values)) => QuestionResponse {
            answered: true,
            selected: values
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
            other_text: None,
            timed_out,
            unavailable: false,
        },
        Some(value) => QuestionResponse {
            answered: true,
            selected: Vec::new(),
            other_text: Some(value.to_string()),
            timed_out,
            unavailable: false,
        },
        None => QuestionResponse {
            answered: false,
            selected: Vec::new(),
            other_text: None,
            timed_out,
            unavailable: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_string_becomes_selected_answer() {
        let response = response_from_default(Some(serde_json::json!("sqlite")), false);
        assert!(response.answered);
        assert_eq!(response.selected, vec!["sqlite"]);
    }
}
