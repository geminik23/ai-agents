use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_hooks::AgentHooks;
use ai_agents_llm::{ChatMessage, LLMProvider};
use ai_agents_state::{
    ChatStyle, GroupChatStateConfig, MaxIterationsAction, TerminationMethod, TurnMethod,
};

use super::types::{ChatTurn, GroupChatResult};
use crate::Agent;
use crate::spawner::AgentRegistry;

/// Run a multi-turn multi-agent conversation.
pub async fn group_chat(
    registry: &AgentRegistry,
    topic: &str,
    config: &GroupChatStateConfig,
    llm: Option<&dyn LLMProvider>,
    hooks: Option<&dyn AgentHooks>,
) -> Result<GroupChatResult> {
    if config.participants.is_empty() {
        return Err(AgentError::Config("No participants in group chat".into()));
    }

    let start = std::time::Instant::now();
    let mut transcript: Vec<ChatTurn> = Vec::new();
    let mut accumulated_context = format!("Topic: {}\n", topic);
    let mut rounds_completed = 0u32;
    let mut stall_count = 0u32;
    let mut last_content_hash = String::new();

    if matches!(config.style, ChatStyle::MakerChecker) {
        return run_maker_checker(registry, topic, config, llm, hooks).await;
    }

    if matches!(config.style, ChatStyle::Debate) {
        return run_debate(registry, topic, config, llm, hooks).await;
    }

    let method = config
        .manager
        .as_ref()
        .and_then(|m| m.method.as_ref())
        .cloned()
        .unwrap_or(TurnMethod::RoundRobin);

    for round in 0..config.max_rounds {
        if let Some(timeout) = config.timeout_ms {
            if start.elapsed().as_millis() as u64 >= timeout {
                return Ok(build_result(&transcript, round, "timeout"));
            }
        }

        let speakers_count = if matches!(method, TurnMethod::LlmDirected) {
            let llm_ref = llm.ok_or_else(|| {
                AgentError::Config("LlmDirected turn method requires an LLM provider".into())
            })?;

            run_llm_directed_round(
                registry,
                llm_ref,
                config,
                topic,
                round,
                &mut transcript,
                &mut accumulated_context,
                hooks,
            )
            .await? as usize
        } else {
            let participant_order = get_turn_order(config, round, &transcript);
            let count = participant_order.len();

            for participant_id in &participant_order {
                let agent = registry.get(participant_id).ok_or_else(|| {
                    AgentError::Other(format!(
                        "Group chat participant not found: {}",
                        participant_id
                    ))
                })?;

                let role_line = match find_role(&config.participants, participant_id) {
                    Some(role) => format!("\nYour role: {}\n", role),
                    None => String::new(),
                };

                let prompt = format!(
                    "{}\n\nConversation so far:\n{}{}\nIt is your turn to contribute.",
                    accumulated_context,
                    format_transcript(&transcript),
                    role_line,
                );

                let response = agent.chat(&prompt).await?;
                let content = response.content.clone();

                transcript.push(ChatTurn {
                    speaker: participant_id.clone(),
                    round,
                    content: content.clone(),
                });

                if let Some(h) = hooks {
                    h.on_group_chat_round(round, participant_id, &content).await;
                }

                accumulated_context.push_str(&format!("\n{}: {}", participant_id, content));
            }

            count
        };

        rounds_completed = round + 1;

        // Detect stalling by comparing recent content across rounds.
        let current_hash = transcript
            .iter()
            .rev()
            .take(speakers_count)
            .map(|t| t.content.as_str())
            .collect::<Vec<_>>()
            .join("|");

        if current_hash == last_content_hash {
            stall_count += 1;
        } else {
            stall_count = 0;
        }
        last_content_hash = current_hash;

        // MaxRounds disables stall detection - run the full max_rounds count.
        if !matches!(config.termination.method, TerminationMethod::MaxRounds) {
            if matches!(config.termination.method, TerminationMethod::ManagerDecides) {
                if let Some(ref manager_id) = config.manager.as_ref().and_then(|m| m.agent.as_ref())
                {
                    let manager_agent = registry.get(manager_id).ok_or_else(|| {
                        AgentError::Config(format!(
                            "Manager agent not found in registry: {}",
                            manager_id
                        ))
                    })?;
                    let decision =
                        ask_manager_continue(manager_agent.as_ref(), topic, &transcript).await?;
                    if decision.action == "end" {
                        return Ok(build_result(&transcript, rounds_completed, "manager_ended"));
                    }
                } else if stall_count >= config.termination.max_stall_rounds {
                    return Ok(build_result(
                        &transcript,
                        rounds_completed,
                        "stall_detected",
                    ));
                }
            } else if stall_count >= config.termination.max_stall_rounds {
                return Ok(build_result(
                    &transcript,
                    rounds_completed,
                    "stall_detected",
                ));
            }
        }

        if matches!(config.style, ChatStyle::Consensus)
            || matches!(
                config.termination.method,
                TerminationMethod::ConsensusReached
            )
        {
            if let Some(llm) = llm {
                if check_consensus(llm, &transcript).await? {
                    return Ok(build_result(
                        &transcript,
                        rounds_completed,
                        "consensus_reached",
                    ));
                }
            }
        }
    }

    Ok(build_result(
        &transcript,
        rounds_completed,
        "max_rounds_reached",
    ))
}

/// Determine turn order for a round based on the configured method.
fn get_turn_order(
    config: &GroupChatStateConfig,
    _round: u32,
    _transcript: &[ChatTurn],
) -> Vec<String> {
    let method = config
        .manager
        .as_ref()
        .and_then(|m| m.method.as_ref())
        .cloned()
        .unwrap_or(TurnMethod::RoundRobin);

    match method {
        TurnMethod::RoundRobin => config.participants.iter().map(|p| p.id.clone()).collect(),
        TurnMethod::Random => {
            use rand::seq::SliceRandom;
            let mut order: Vec<String> = config.participants.iter().map(|p| p.id.clone()).collect();
            let mut rng = rand::thread_rng();
            order.shuffle(&mut rng);
            order
        }
        TurnMethod::LlmDirected => {
            unreachable!("LlmDirected is handled in the main loop, not via get_turn_order")
        }
    }
}

/// Ask the LLM to pick the next speaker based on the conversation so far.
async fn select_next_speaker(
    llm: &dyn LLMProvider,
    participants: &[ai_agents_state::ChatParticipant],
    transcript: &[ChatTurn],
    topic: &str,
) -> Result<String> {
    let participant_list = participants
        .iter()
        .map(|p| match &p.role {
            Some(role) => format!("- {} ({})", p.id, role),
            None => format!("- {}", p.id),
        })
        .collect::<Vec<_>>()
        .join("\n");

    let transcript_text = if transcript.is_empty() {
        "No messages yet. This is the start of the conversation.".to_string()
    } else {
        format_transcript(transcript)
    };

    let messages = vec![
        ChatMessage::system(&format!(
            "You are a conversation manager.\n\
             Pick which participant should speak next.\n\
             Respond with only the participant ID, nothing else.\n\n\
             Participants:\n{}",
            participant_list
        )),
        ChatMessage::user(&format!(
            "Topic: {}\n\nConversation so far:\n{}\n\nWho should speak next?",
            topic, transcript_text
        )),
    ];

    let response = llm
        .complete(&messages, None)
        .await
        .map_err(|e| AgentError::LLM(format!("Speaker selection failed: {}", e)))?;

    let raw = response.content.trim().to_lowercase();

    // Fuzzy match against participant IDs.
    for p in participants {
        if raw.contains(&p.id.to_lowercase()) {
            return Ok(p.id.clone());
        }
    }

    // Fallback: first participant.
    tracing::warn!(
        llm_output = raw,
        "LLM speaker selection did not match any participant ID, falling back to first"
    );
    Ok(participants[0].id.clone())
}

/// Run one round where the LLM picks each speaker one at a time.
async fn run_llm_directed_round(
    registry: &AgentRegistry,
    llm: &dyn LLMProvider,
    config: &GroupChatStateConfig,
    topic: &str,
    round: u32,
    transcript: &mut Vec<ChatTurn>,
    accumulated_context: &mut String,
    hooks: Option<&dyn AgentHooks>,
) -> Result<u32> {
    let max_speakers = config.participants.len() as u32;
    let mut speakers_this_round = 0u32;

    while speakers_this_round < max_speakers {
        let next_id =
            if let Some(ref manager_id) = config.manager.as_ref().and_then(|m| m.agent.as_ref()) {
                let manager_agent = registry.get(manager_id).ok_or_else(|| {
                    AgentError::Config(format!(
                        "Manager agent not found in registry: {}",
                        manager_id
                    ))
                })?;
                manager_select_speaker(
                    manager_agent.as_ref(),
                    &config.participants,
                    transcript,
                    topic,
                )
                .await?
            } else {
                select_next_speaker(llm, &config.participants, transcript, topic).await?
            };

        let agent = registry.get(&next_id).ok_or_else(|| {
            AgentError::Other(format!("Group chat participant not found: {}", next_id))
        })?;

        let role_line = match find_role(&config.participants, &next_id) {
            Some(role) => format!("\nYour role: {}\n", role),
            None => String::new(),
        };

        let prompt = format!(
            "{}\n\nConversation so far:\n{}{}\nIt is your turn to contribute.",
            accumulated_context,
            format_transcript(transcript),
            role_line,
        );

        let response = agent.chat(&prompt).await?;
        let content = response.content.clone();

        transcript.push(ChatTurn {
            speaker: next_id.clone(),
            round,
            content: content.clone(),
        });

        if let Some(h) = hooks {
            h.on_group_chat_round(round, &next_id, &content).await;
        }

        accumulated_context.push_str(&format!("\n{}: {}", next_id, content));
        speakers_this_round += 1;
    }

    Ok(speakers_this_round)
}

fn format_transcript(transcript: &[ChatTurn]) -> String {
    transcript
        .iter()
        .map(|t| format!("[Round {}] {}: {}", t.round, t.speaker, t.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Look up a participant's role by ID.
fn find_role<'a>(
    participants: &'a [ai_agents_state::ChatParticipant],
    id: &str,
) -> Option<&'a str> {
    participants
        .iter()
        .find(|p| p.id == id)
        .and_then(|p| p.role.as_deref())
}

/// Ask an LLM whether participants have reached consensus.
async fn check_consensus(llm: &dyn LLMProvider, transcript: &[ChatTurn]) -> Result<bool> {
    let transcript_text = format_transcript(transcript);
    let messages = vec![
        ChatMessage::system(
            "Analyze this conversation transcript. \
             Have all participants reached agreement? \
             Respond with only 'yes' or 'no'.",
        ),
        ChatMessage::user(&transcript_text),
    ];

    let response = llm
        .complete(&messages, None)
        .await
        .map_err(|e| AgentError::LLM(format!("Consensus check failed: {}", e)))?;

    Ok(response.content.trim().to_lowercase().starts_with("yes"))
}

/// Manager decision for continuing or ending a group chat.
#[allow(dead_code)]
struct ManagerDecision {
    action: String,
    reason: String,
}

/// Ask the manager agent whether the conversation should continue.
async fn ask_manager_continue(
    manager: &dyn Agent,
    topic: &str,
    transcript: &[ChatTurn],
) -> Result<ManagerDecision> {
    let transcript_text = format_transcript(transcript);
    let prompt = format!(
        "You are managing a group conversation.\n\n\
         Topic: {}\n\n\
         Conversation so far:\n{}\n\n\
         Should the conversation continue for another round?\n\
         Respond in JSON: {{\"action\": \"continue\" or \"end\", \"reason\": \"brief explanation\"}}",
        topic, transcript_text
    );

    let response = manager.chat(&prompt).await?;
    let raw = response.content.trim().to_string();

    // Try JSON extraction.
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw) {
        let action = parsed
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("continue")
            .to_string();
        let reason = parsed
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return Ok(ManagerDecision { action, reason });
    }

    // Try extracting JSON from mixed text.
    if let Some(start) = raw.find('{') {
        if let Some(end) = raw.rfind('}') {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw[start..=end]) {
                let action = parsed
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("continue")
                    .to_string();
                let reason = parsed
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                return Ok(ManagerDecision { action, reason });
            }
        }
    }

    // Fuzzy text fallback.
    let lower = raw.to_lowercase();
    let action = if lower.contains("end") || lower.contains("stop") || lower.contains("conclude") {
        "end".to_string()
    } else {
        "continue".to_string()
    };
    Ok(ManagerDecision {
        action,
        reason: raw,
    })
}

/// Ask the manager agent to select the next speaker.
async fn manager_select_speaker(
    manager: &dyn Agent,
    participants: &[ai_agents_state::ChatParticipant],
    transcript: &[ChatTurn],
    topic: &str,
) -> Result<String> {
    let participant_list = participants
        .iter()
        .map(|p| match &p.role {
            Some(role) => format!("- {} ({})", p.id, role),
            None => format!("- {}", p.id),
        })
        .collect::<Vec<_>>()
        .join("\n");

    let transcript_text = if transcript.is_empty() {
        "No messages yet. This is the start of the conversation.".to_string()
    } else {
        format_transcript(transcript)
    };

    let prompt = format!(
        "You are managing a group conversation.\n\n\
         Participants:\n{}\n\n\
         Topic: {}\n\n\
         Conversation so far:\n{}\n\n\
         Pick which participant should speak next.\n\
         Respond with only the participant ID, nothing else.",
        participant_list, topic, transcript_text
    );

    let response = manager.chat(&prompt).await?;
    let raw = response.content.trim().to_lowercase();

    // Fuzzy match against participant IDs.
    for p in participants {
        if raw.contains(&p.id.to_lowercase()) {
            return Ok(p.id.clone());
        }
    }

    // Fallback: first participant.
    tracing::warn!(
        llm_output = raw,
        "Manager speaker selection did not match any participant, falling back to first"
    );
    Ok(participants[0].id.clone())
}

fn build_result(transcript: &[ChatTurn], rounds: u32, reason: &str) -> GroupChatResult {
    let content = format_transcript(transcript);

    GroupChatResult {
        response: AgentResponse::new(content),
        transcript: transcript.to_vec(),
        rounds_completed: rounds,
        termination_reason: reason.to_string(),
    }
}

/// One agent creates content, another reviews it, looping until accepted.
async fn run_maker_checker(
    registry: &AgentRegistry,
    topic: &str,
    config: &GroupChatStateConfig,
    _llm: Option<&dyn LLMProvider>,
    hooks: Option<&dyn AgentHooks>,
) -> Result<GroupChatResult> {
    if config.participants.len() < 2 {
        return Err(AgentError::Config(
            "Maker-checker requires at least 2 participants".into(),
        ));
    }

    let maker_id = &config.participants[0].id;
    let checker_id = &config.participants[1].id;
    let max_iter = config
        .maker_checker
        .as_ref()
        .map(|mc| mc.max_iterations)
        .unwrap_or(3);
    let criteria = config
        .maker_checker
        .as_ref()
        .map(|mc| mc.acceptance_criteria.as_str())
        .unwrap_or("The content is accurate and complete");

    let maker = registry
        .get(maker_id)
        .ok_or_else(|| AgentError::Other(format!("Maker agent not found: {}", maker_id)))?;
    let checker = registry
        .get(checker_id)
        .ok_or_else(|| AgentError::Other(format!("Checker agent not found: {}", checker_id)))?;

    let mut transcript = Vec::new();
    let mut current_draft = String::new();

    for iteration in 0..max_iter {
        let maker_prompt = if iteration == 0 {
            format!("Create content for: {}", topic)
        } else {
            format!(
                "Revise your previous draft based on this feedback:\n\nDraft:\n{}\n\n\
                 Feedback from reviewer will follow.",
                current_draft
            )
        };

        let maker_response = maker.chat(&maker_prompt).await?;
        current_draft = maker_response.content.clone();
        transcript.push(ChatTurn {
            speaker: maker_id.clone(),
            round: iteration,
            content: current_draft.clone(),
        });

        if let Some(h) = hooks {
            h.on_group_chat_round(iteration, maker_id, &current_draft)
                .await;
        }

        let checker_prompt = format!(
            "Review this content against these criteria: {}\n\nContent:\n{}\n\n\
             If it meets the criteria, respond with 'APPROVED'. \
             Otherwise, provide specific feedback for improvement.",
            criteria, current_draft
        );

        let checker_response = checker.chat(&checker_prompt).await?;
        let feedback = checker_response.content.clone();
        transcript.push(ChatTurn {
            speaker: checker_id.clone(),
            round: iteration,
            content: feedback.clone(),
        });

        if let Some(h) = hooks {
            h.on_group_chat_round(iteration, checker_id, &feedback)
                .await;
        }

        if feedback.to_uppercase().contains("APPROVED") {
            return Ok(GroupChatResult {
                response: AgentResponse::new(current_draft),
                transcript,
                rounds_completed: iteration + 1,
                termination_reason: "accepted".into(),
            });
        }
    }

    let on_max = config
        .maker_checker
        .as_ref()
        .map(|mc| mc.on_max_iterations.clone())
        .unwrap_or_default();

    match on_max {
        MaxIterationsAction::AcceptLast => Ok(GroupChatResult {
            response: AgentResponse::new(current_draft),
            transcript,
            rounds_completed: max_iter,
            termination_reason: "max_iterations".into(),
        }),
        MaxIterationsAction::Fail => Err(AgentError::Other(format!(
            "Maker-checker failed to reach acceptance after {} iterations",
            max_iter
        ))),
        MaxIterationsAction::Escalate => Ok(GroupChatResult {
            response: AgentResponse::new(current_draft),
            transcript,
            rounds_completed: max_iter,
            termination_reason: "escalated".into(),
        }),
    }
}

/// Structured rounds with pro/con arguments and a synthesizer agent.
async fn run_debate(
    registry: &AgentRegistry,
    topic: &str,
    config: &GroupChatStateConfig,
    _llm: Option<&dyn LLMProvider>,
    hooks: Option<&dyn AgentHooks>,
) -> Result<GroupChatResult> {
    let rounds = config.debate.as_ref().map(|d| d.rounds).unwrap_or(3);

    let mut transcript = Vec::new();
    let mut debate_text = String::new();

    for round in 0..rounds {
        for participant in &config.participants {
            let agent = registry.get(&participant.id).ok_or_else(|| {
                AgentError::Other(format!("Debate participant not found: {}", participant.id))
            })?;

            let role_hint = participant.role.as_deref().unwrap_or("participant");

            let prompt = format!(
                "Topic: {}\nYour role: {}\nRound {} of {}.\n\n{}\n\nProvide your argument.",
                topic,
                role_hint,
                round + 1,
                rounds,
                if debate_text.is_empty() {
                    "This is the opening round.".to_string()
                } else {
                    format!("Previous arguments:\n{}", debate_text)
                }
            );

            let response = agent.chat(&prompt).await?;
            let content = response.content.clone();

            debate_text.push_str(&format!(
                "\n[{} - Round {}]: {}",
                participant.id,
                round + 1,
                content
            ));
            transcript.push(ChatTurn {
                speaker: participant.id.clone(),
                round,
                content: content.clone(),
            });

            if let Some(h) = hooks {
                h.on_group_chat_round(round, &participant.id, &content)
                    .await;
            }
        }
    }

    // Synthesize the debate if a synthesizer agent is configured.
    let conclusion = if let Some(ref debate_config) = config.debate {
        if let Some(synth_agent) = registry.get(&debate_config.synthesizer) {
            let synth_prompt = format!(
                "Synthesize this debate into a balanced conclusion:\n\n{}",
                debate_text
            );
            let synth_response = synth_agent.chat(&synth_prompt).await?;
            synth_response.content
        } else {
            debate_text
        }
    } else {
        debate_text
    };

    Ok(GroupChatResult {
        response: AgentResponse::new(conclusion),
        transcript,
        rounds_completed: rounds,
        termination_reason: "debate_complete".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_result_formats_full_transcript() {
        let transcript = vec![
            ChatTurn {
                speaker: "merchant".into(),
                round: 0,
                content: "Good morning!".into(),
            },
            ChatTurn {
                speaker: "guard".into(),
                round: 0,
                content: "Morning.".into(),
            },
            ChatTurn {
                speaker: "merchant".into(),
                round: 1,
                content: "Heard about wolves.".into(),
            },
        ];
        let result = build_result(&transcript, 2, "max_rounds_reached");
        assert!(
            result
                .response
                .content
                .contains("[Round 0] merchant: Good morning!")
        );
        assert!(
            result
                .response
                .content
                .contains("[Round 0] guard: Morning.")
        );
        assert!(
            result
                .response
                .content
                .contains("[Round 1] merchant: Heard about wolves.")
        );
        assert_eq!(result.rounds_completed, 2);
        assert_eq!(result.termination_reason, "max_rounds_reached");
        assert_eq!(result.transcript.len(), 3);
    }

    #[test]
    fn test_build_result_empty_transcript() {
        let result = build_result(&[], 0, "no_participants");
        assert!(result.response.content.is_empty());
        assert!(result.transcript.is_empty());
    }

    #[test]
    fn test_get_turn_order_round_robin() {
        let config = GroupChatStateConfig {
            participants: vec![
                ai_agents_state::ChatParticipant {
                    id: "alice".into(),
                    role: None,
                },
                ai_agents_state::ChatParticipant {
                    id: "bob".into(),
                    role: Some("reviewer".into()),
                },
            ],
            manager: None,
            max_rounds: 3,
            style: ChatStyle::Brainstorm,
            timeout_ms: None,
            termination: ai_agents_state::TerminationConfig {
                method: TerminationMethod::MaxRounds,
                max_stall_rounds: 2,
            },
            maker_checker: None,
            debate: None,
            input: None,
            context_mode: None,
        };

        let order = get_turn_order(&config, 0, &[]);
        assert_eq!(order, vec!["alice".to_string(), "bob".to_string()]);
    }

    #[test]
    fn test_get_turn_order_random_returns_all_participants() {
        let config = GroupChatStateConfig {
            participants: vec![
                ai_agents_state::ChatParticipant {
                    id: "alice".into(),
                    role: None,
                },
                ai_agents_state::ChatParticipant {
                    id: "bob".into(),
                    role: None,
                },
                ai_agents_state::ChatParticipant {
                    id: "carol".into(),
                    role: None,
                },
            ],
            manager: Some(ai_agents_state::ChatManagerConfig {
                agent: None,
                method: Some(TurnMethod::Random),
            }),
            max_rounds: 3,
            style: ChatStyle::Brainstorm,
            timeout_ms: None,
            termination: ai_agents_state::TerminationConfig {
                method: TerminationMethod::MaxRounds,
                max_stall_rounds: 2,
            },
            maker_checker: None,
            debate: None,
            input: None,
            context_mode: None,
        };

        let order = get_turn_order(&config, 0, &[]);
        assert_eq!(order.len(), 3);
        assert!(order.contains(&"alice".to_string()));
        assert!(order.contains(&"bob".to_string()));
        assert!(order.contains(&"carol".to_string()));
    }

    #[test]
    #[should_panic(expected = "LlmDirected is handled in the main loop")]
    fn test_get_turn_order_llm_directed_is_unreachable() {
        let config = GroupChatStateConfig {
            participants: vec![ai_agents_state::ChatParticipant {
                id: "alice".into(),
                role: None,
            }],
            manager: Some(ai_agents_state::ChatManagerConfig {
                agent: None,
                method: Some(TurnMethod::LlmDirected),
            }),
            max_rounds: 3,
            style: ChatStyle::Brainstorm,
            timeout_ms: None,
            termination: ai_agents_state::TerminationConfig {
                method: TerminationMethod::MaxRounds,
                max_stall_rounds: 2,
            },
            maker_checker: None,
            debate: None,
            input: None,
            context_mode: None,
        };

        // This should panic because LlmDirected is not handled in get_turn_order.
        let _ = get_turn_order(&config, 0, &[]);
    }

    #[test]
    fn test_find_role_found() {
        let participants = vec![
            ai_agents_state::ChatParticipant {
                id: "architect".into(),
                role: Some("system architect".into()),
            },
            ai_agents_state::ChatParticipant {
                id: "security".into(),
                role: Some("security reviewer".into()),
            },
        ];
        assert_eq!(
            find_role(&participants, "security"),
            Some("security reviewer")
        );
        assert_eq!(
            find_role(&participants, "architect"),
            Some("system architect")
        );
    }

    #[test]
    fn test_find_role_none() {
        let participants = vec![ai_agents_state::ChatParticipant {
            id: "bob".into(),
            role: None,
        }];
        assert_eq!(find_role(&participants, "bob"), None);
    }

    #[test]
    fn test_find_role_missing_id() {
        let participants = vec![ai_agents_state::ChatParticipant {
            id: "alice".into(),
            role: Some("lead".into()),
        }];
        assert_eq!(find_role(&participants, "unknown"), None);
    }
}
