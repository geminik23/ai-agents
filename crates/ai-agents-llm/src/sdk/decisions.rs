//! Single-choice requests to OpenAI Decisions. The caller interprets and acts on the answer.

use super::{Error, transport};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashSet, time::Duration};

/// An exact label and optional guidance for when it applies.
#[derive(Serialize)]
pub struct ChoiceOption<'a> {
    #[serde(rename = "value")]
    pub id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<&'a str>,
}

/// One text classification question. The model is selected by the caller.
pub struct ChoiceRequest<'a> {
    pub model: &'a str,
    pub input: &'a str,
    pub question: &'a str,
    pub options: &'a [ChoiceOption<'a>],
    pub timeout: Option<Duration>,
}

/// A completed answer, including refusals that still incur token usage.
#[derive(Debug)]
pub struct ChoiceResponse {
    pub outcome: ChoiceOutcome,
    pub usage: Option<DecisionUsage>,
    pub model: String,
}

/// A refusal is a successful protocol exchange with no selected label.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum ChoiceOutcome {
    Selected { id: String, confidence: Option<f64> },
    Refused,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct DecisionUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone)]
pub struct DecisionClient {
    transport: transport::Transport,
}

impl DecisionClient {
    /// The base URL includes the API prefix, such as `https://api.openai.com/v1`.
    pub fn new(base_url: &str, api_key: String) -> Result<Self, Error> {
        if api_key.trim().is_empty() {
            return Err(Error::Config("Decisions requires a bearer token".into()));
        }
        Ok(Self {
            transport: transport::Transport::new(
                base_url,
                "decisions",
                transport::bearer_headers(Some(api_key))?,
            )?,
        })
    }

    /// Checks the endpoint's limits without dispatching a request.
    pub fn validate(&self, request: &ChoiceRequest<'_>) -> Result<(), Error> {
        if request.model.trim().is_empty() || request.question.trim().is_empty() {
            return Err(Error::Config("Decisions needs a model and question".into()));
        }
        // OpenAI defines this range for a choice question; other endpoints own their limits.
        if !(2..=255).contains(&request.options.len()) {
            return Err(Error::Config("Decisions requires 2 to 255 choices".into()));
        }
        let mut ids = HashSet::new();
        if request
            .options
            .iter()
            .any(|option| option.id.trim().is_empty() || !ids.insert(option.id))
        {
            return Err(Error::Config(
                "choice IDs must be nonempty and unique".into(),
            ));
        }
        Ok(())
    }

    /// Sends one request and accepts only an answer to its single named question.
    pub async fn choose(&self, request: &ChoiceRequest<'_>) -> Result<ChoiceResponse, Error> {
        self.validate(request)?;
        let body = json!({
            "model": request.model,
            "input": request.input,
            "questions": [{"name": QUESTION_NAME, "type": "choice",
                "instructions": request.question, "choices": request.options}]
        });
        let response: WireResponse = self
            .transport
            .send(&body, request.timeout)
            .await?
            .json()
            .await
            .map_err(|error| {
                if error.is_decode() {
                    Error::Protocol("invalid Decisions response")
                } else {
                    Error::Transport
                }
            })?;
        response.finish(request)
    }
}

const QUESTION_NAME: &str = "selection";

#[derive(Deserialize)]
struct WireResponse {
    answers: Vec<WireAnswer>,
    model: String,
    usage: Option<DecisionUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireAnswer {
    Choice {
        name: String,
        choice: String,
        confidence: Option<f64>,
    },
    Refusal {
        name: String,
    },
}

impl WireResponse {
    fn finish(self, request: &ChoiceRequest<'_>) -> Result<ChoiceResponse, Error> {
        let [answer]: [WireAnswer; 1] = self
            .answers
            .try_into()
            .map_err(|_| Error::Protocol("expected exactly one decision answer"))?;
        let outcome = match answer {
            WireAnswer::Choice {
                name,
                choice,
                confidence,
            } if name == QUESTION_NAME => {
                if !request.options.iter().any(|option| option.id == choice) {
                    return Err(Error::Protocol("decision selected an unknown choice"));
                }
                if confidence.is_some_and(|value| !(0.0..=1.0).contains(&value)) {
                    return Err(Error::Protocol("invalid decision confidence"));
                }
                ChoiceOutcome::Selected {
                    id: choice,
                    confidence,
                }
            }
            WireAnswer::Refusal { name } if name == QUESTION_NAME => ChoiceOutcome::Refused,
            _ => {
                return Err(Error::Protocol(
                    "decision answer does not match its question",
                ));
            }
        };
        if self.model.trim().is_empty()
            || self.usage.as_ref().is_some_and(|usage| {
                usage.input_tokens.checked_add(usage.output_tokens) != Some(usage.total_tokens)
            })
        {
            return Err(Error::Protocol("invalid Decisions response model or usage"));
        }
        Ok(ChoiceResponse {
            outcome,
            usage: self.usage,
            model: self.model,
        })
    }
}
