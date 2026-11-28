use std::ops::Range;

use ai_agents_core::{
    AgentError, ChatMessage, IncompleteNativeExchange, NativeHistoryInspection, Role,
    inspect_native_history, native_readable_projection,
};

/// Memory-specific view of native history that treats an entire signed user turn as one unit.
pub(crate) struct NativeRetentionInspection {
    core: NativeHistoryInspection,
    signed_turns: Vec<Range<usize>>,
}

impl NativeRetentionInspection {
    /// Inspects provider-state-bearing exchanges and expands their boundaries to whole user turns.
    pub(crate) fn inspect(messages: &[ChatMessage]) -> ai_agents_core::Result<Self> {
        let core = inspect_native_history(messages).map_err(memory_error)?;
        let mut signed_turns = Vec::<Range<usize>>::new();

        for exchange in core.exchanges() {
            let start = messages[..exchange.message_start()]
                .iter()
                .rposition(|message| message.role == Role::User)
                .ok_or_else(|| {
                    AgentError::MemoryError(
                        "signed native exchange has no preceding user boundary".to_string(),
                    )
                })?;
            let end = messages[exchange.message_end()..]
                .iter()
                .position(|message| message.role == Role::User)
                .map_or(messages.len(), |offset| exchange.message_end() + offset);

            if let Some(previous) = signed_turns.last_mut()
                && previous.start == start
            {
                previous.end = previous.end.max(end);
            } else {
                signed_turns.push(start..end);
            }
        }

        Ok(Self { core, signed_turns })
    }

    /// Returns the earliest active native boundary that compression must retain.
    pub(crate) fn protected_suffix_start(&self) -> Option<usize> {
        self.core.protected_suffix_start()
    }

    /// Finds the earliest signed user turn with an unrecorded result, including ended turns.
    pub(crate) fn incomplete_suffix_start(&self) -> Option<usize> {
        self.core
            .exchanges()
            .iter()
            .filter(|exchange| !exchange.is_complete())
            .filter_map(|exchange| {
                self.signed_turns
                    .iter()
                    .find(|range| range.contains(&exchange.message_start()))
                    .map(|range| range.start)
            })
            .min()
    }

    /// Reports whether removing this prefix preserves core exchanges and whole signed user turns.
    pub(crate) fn is_safe_prefix_len(&self, count: usize) -> bool {
        self.core.is_safe_prefix_len(count)
            && self
                .signed_turns
                .iter()
                .all(|range| count <= range.start || count >= range.end)
    }

    /// Finds the greatest whole-turn-safe prefix no larger than the requested count.
    pub(crate) fn safe_prefix_len_at_most(&self, requested: usize) -> usize {
        let core_bound = self.core.safe_prefix_len_at_most(requested);
        (0..=core_bound)
            .rev()
            .find(|count| self.is_safe_prefix_len(*count))
            .unwrap_or(0)
    }

    /// Finds the smallest whole-turn-safe prefix in the requested inclusive range.
    pub(crate) fn safe_prefix_len_between(&self, required: usize, maximum: usize) -> Option<usize> {
        let core_candidate = self.core.safe_prefix_len_at_least(required)?;
        (core_candidate..=maximum).find(|count| self.is_safe_prefix_len(*count))
    }

    /// Finds the previous safe cut before `start`, or zero when no earlier boundary exists.
    pub(crate) fn previous_safe_prefix_len(&self, start: usize) -> usize {
        self.safe_prefix_len_at_most(start.saturating_sub(1))
    }

    /// Returns signed assistant indexes and observations whose final result was not recorded.
    pub(crate) fn incomplete_exchanges(&self) -> Vec<(usize, IncompleteNativeExchange)> {
        self.core
            .exchanges()
            .iter()
            .filter(|exchange| !exchange.is_complete())
            .map(|exchange| {
                (
                    exchange.message_start(),
                    IncompleteNativeExchange {
                        exchange_id: Some(exchange.state().exchange_id().to_string()),
                        missing_result_ids: exchange
                            .missing_result_ids()
                            .into_iter()
                            .map(str::to_string)
                            .collect(),
                        unverified_legacy: false,
                    },
                )
            })
            .collect()
    }
}

/// Clones messages while replacing opaque native replay state with readable control summaries.
pub(crate) fn readable_projection(
    messages: &[ChatMessage],
) -> ai_agents_core::Result<Vec<ChatMessage>> {
    messages
        .iter()
        .cloned()
        .map(|mut message| {
            if matches!(message.role, Role::Assistant | Role::Tool | Role::Function) {
                message.content =
                    native_readable_projection(&message.content).map_err(memory_error)?;
            }
            Ok(message)
        })
        .collect()
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyIncompleteStatus {
    native_exchange_status: String,
    missing_result_ids: Vec<String>,
    description: String,
}

// Old summaries carry only ambiguous text, so recovered observations must remain explicitly unverified.
pub(crate) fn extract_legacy_incomplete(
    summary: Option<String>,
) -> (Option<String>, Vec<IncompleteNativeExchange>) {
    let Some(text) = summary else {
        return (None, Vec::new());
    };
    let mut remaining = text.as_str();
    let mut observations = Vec::new();
    loop {
        let (prefix, line) = remaining.rsplit_once('\n').unwrap_or(("", remaining));
        let Ok(status) = serde_json::from_str::<LegacyIncompleteStatus>(line) else {
            break;
        };
        if status.native_exchange_status != "incomplete"
            || status.description != "final execution result was not recorded"
            || status.missing_result_ids.is_empty()
        {
            break;
        }
        observations.push(IncompleteNativeExchange {
            exchange_id: None,
            missing_result_ids: status.missing_result_ids,
            unverified_legacy: true,
        });
        remaining = prefix;
        if remaining.is_empty() {
            break;
        }
    }
    if observations.is_empty() {
        return (Some(text), observations);
    }
    observations.reverse();
    (
        (!remaining.is_empty()).then(|| remaining.to_string()),
        observations,
    )
}

/// Reports whether a legacy summary ends with an unverified incomplete-exchange marker.
/// Such text may originate from a model or user and cannot prove execution or authorize a retry.
pub fn has_legacy_incomplete_summary(summary: Option<&str>) -> bool {
    !extract_legacy_incomplete(summary.map(str::to_string))
        .1
        .is_empty()
}

fn memory_error(error: impl std::fmt::Display) -> AgentError {
    AgentError::MemoryError(error.to_string())
}
