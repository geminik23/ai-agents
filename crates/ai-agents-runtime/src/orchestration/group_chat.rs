use ai_agents_core::{AgentError, AgentResponse, Result};
use ai_agents_hooks::AgentHooks;
use ai_agents_llm::{ChatMessage, LLMProvider};
use ai_agents_observability::{ObservationPurpose, with_observation_purpose};
use ai_agents_state::{
    ChatStyle, GroupChatStateConfig, MaxIterationsAction, TerminationMethod, TurnMethod,
};

use super::types::{ChatTurn, GroupChatResult};
use crate::spawner::AgentRegistry;
use crate::turn_context::current_turn_actor_context;
use crate::{Agent, RuntimeAgent, TurnActorContext};

/// Each parent-controlled call binds a stable conversation coordinate before a participant can suspend.
async fn chat_agent(
    agent: &RuntimeAgent,
    input: &str,
    actor_context: Option<TurnActorContext>,
    coordinate: &str,
) -> Result<AgentResponse> {
    use crate::autonomy::composition;
    let execution = crate::autonomy::current_execution();
    let frame = execution
        .as_ref()
        .map(|execution| execution.composition_frame())
        .transpose()?
        .flatten();
    let slot = if let (Some(execution), Some(frame)) = (&execution, &frame) {
        Some(
            Box::pin(execution.bind_composition_call(
                &frame.runtime_id,
                coordinate,
                &agent.info().id,
            ))
            .await?,
        )
    } else {
        None
    };
    composition::run_composition_child(
        composition::current_composition(),
        slot,
        composition::group_response_custody(),
        Box::pin(async {
            if let Some(context) = actor_context {
                agent.chat_with_actor_context(input, context).await
            } else {
                agent.chat(input).await
            }
        }),
    )
    .await
}

/// Run a multi-turn multi-agent conversation.
pub async fn group_chat(
    registry: &AgentRegistry,
    topic: &str,
    config: &GroupChatStateConfig,
    llm: Option<&dyn LLMProvider>,
    hooks: Option<&dyn AgentHooks>,
) -> Result<GroupChatResult> {
    group_chat_with_llms(
        registry,
        topic,
        config,
        super::GroupLLMs::shared(llm),
        hooks,
    )
    .await
}

/// Separates speaker and consensus decisions while retaining a task cursor for transcript, chosen speaker and termination phase.
/// Random orders and completed turns are acknowledged before another child can park, so resume does not select or emit them again.
pub async fn group_chat_with_llms(
    registry: &AgentRegistry,
    topic: &str,
    config: &GroupChatStateConfig,
    llms: super::GroupLLMs<'_>,
    hooks: Option<&dyn AgentHooks>,
) -> Result<GroupChatResult> {
    let authorized = crate::autonomy::composition::enter_composition_dispatch();
    crate::autonomy::composition::scope_dispatch_authority(
        authorized,
        Box::pin(group_chat_dispatch(registry, topic, config, llms, hooks)),
    )
    .await
}

/// The original conversation algorithms share one cursor only with the owning dispatch; provider callbacks cannot borrow it.
async fn group_chat_dispatch(
    registry: &AgentRegistry,
    topic: &str,
    config: &GroupChatStateConfig,
    llms: super::GroupLLMs<'_>,
    hooks: Option<&dyn AgentHooks>,
) -> Result<GroupChatResult> {
    if config.participants.is_empty() {
        return Err(AgentError::Config("No participants in group chat".into()));
    }

    let start = std::time::Instant::now();
    let mut cursor =
        read_chat_cursor::<GroupChatCursor>("group_chat")?.unwrap_or_else(|| GroupChatCursor {
            round: 0,
            speaker_index: 0,
            order: Vec::new(),
            pending_speaker: None,
            phase: 0,
            transcript: Vec::new(),
            context: format!("Topic: {topic}\n"),
            rounds_completed: 0,
            stall_count: 0,
            last_content_hash: String::new(),
        });
    if cursor.round > config.max_rounds
        || cursor.phase > 2
        || cursor.speaker_index > config.participants.len()
    {
        return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
    }
    let mut transcript = cursor.transcript.clone();
    let mut accumulated_context = cursor.context.clone();
    let mut rounds_completed = cursor.rounds_completed;
    let mut stall_count = cursor.stall_count;
    let mut last_content_hash = cursor.last_content_hash.clone();

    if matches!(config.style, ChatStyle::MakerChecker) {
        return Box::pin(run_maker_checker(
            registry,
            topic,
            config,
            llms.speaker,
            hooks,
        ))
        .await;
    }

    if matches!(config.style, ChatStyle::Debate) {
        return Box::pin(run_debate(registry, topic, config, llms.speaker, hooks)).await;
    }

    let method = config
        .manager
        .as_ref()
        .and_then(|m| m.method.as_ref())
        .cloned()
        .unwrap_or(TurnMethod::RoundRobin);

    for round in cursor.round..config.max_rounds {
        if let Some(timeout) = config.timeout_ms
            && start.elapsed().as_millis() as u64 >= timeout
        {
            return Ok(build_result(&transcript, round, "timeout"));
        }

        let speakers_count = if cursor.phase != 0 {
            cursor.speaker_index
        } else if matches!(method, TurnMethod::LlmDirected) {
            let llm_ref = llms.speaker.ok_or_else(|| {
                AgentError::Config("LlmDirected turn method requires an LLM provider".into())
            })?;

            Box::pin(run_llm_directed_round(
                registry,
                llm_ref,
                config,
                topic,
                round,
                &mut transcript,
                &mut accumulated_context,
                hooks,
            ))
            .await? as usize
        } else {
            if cursor.order.is_empty() {
                cursor.order = get_turn_order(config, round, &transcript);
            }
            Box::pin(checkpoint_chat_cursor("group_chat", &cursor)).await?;
            let participant_order = cursor.order.clone();
            let count = participant_order.len();

            for (index, participant_id) in participant_order
                .iter()
                .enumerate()
                .skip(cursor.speaker_index)
            {
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

                let response = chat_agent(
                    agent.as_ref(),
                    &prompt,
                    current_turn_actor_context(),
                    &format!("round:{round}:speaker:{index}"),
                )
                .await?;
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
                cursor.speaker_index = index + 1;
                cursor.transcript = transcript.clone();
                cursor.context = accumulated_context.clone();
                Box::pin(checkpoint_chat_cursor("group_chat", &cursor)).await?;
            }

            count
        };

        if let Some(saved) = read_chat_cursor::<GroupChatCursor>("group_chat")? {
            cursor = saved;
        }
        if cursor.phase == 0 {
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
            cursor.rounds_completed = rounds_completed;
            cursor.stall_count = stall_count;
            cursor.last_content_hash = last_content_hash.clone();
            cursor.phase = 1;
            cursor.speaker_index = speakers_count;
            cursor.transcript = transcript.clone();
            cursor.context = accumulated_context.clone();
            Box::pin(checkpoint_chat_cursor("group_chat", &cursor)).await?;
        }

        // MaxRounds disables stall detection - run the full max_rounds count.
        if cursor.phase == 1 && !matches!(config.termination.method, TerminationMethod::MaxRounds) {
            if matches!(config.termination.method, TerminationMethod::ManagerDecides) {
                if let Some(ref manager_id) = config.manager.as_ref().and_then(|m| m.agent.as_ref())
                {
                    let manager_agent = registry.get(manager_id).ok_or_else(|| {
                        AgentError::Config(format!(
                            "Manager agent not found in registry: {}",
                            manager_id
                        ))
                    })?;
                    let decision = ask_manager_continue(
                        manager_agent.as_ref(),
                        topic,
                        &transcript,
                        current_turn_actor_context(),
                        &format!("round:{round}:termination"),
                    )
                    .await?;
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

        cursor.phase = 2;
        Box::pin(checkpoint_chat_cursor("group_chat", &cursor)).await?;
        if (matches!(config.style, ChatStyle::Consensus)
            || matches!(
                config.termination.method,
                TerminationMethod::ConsensusReached
            ))
            && let Some(llm) = llms.consensus
            && check_consensus(llm, &transcript).await?
        {
            return Ok(build_result(
                &transcript,
                rounds_completed,
                "consensus_reached",
            ));
        }
        cursor.round = round + 1;
        cursor.phase = 0;
        cursor.speaker_index = 0;
        cursor.order.clear();
        cursor.pending_speaker = None;
        Box::pin(checkpoint_chat_cursor("group_chat", &cursor)).await?;
    }

    Ok(build_result(
        &transcript,
        rounds_completed,
        "max_rounds_reached",
    ))
}

/// Safe parent cursor separates speaking from termination so a parked manager cannot restart a completed round.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupChatCursor {
    round: u32,
    speaker_index: usize,
    order: Vec<String>,
    pending_speaker: Option<String>,
    phase: u8,
    transcript: Vec<ChatTurn>,
    context: String,
    rounds_completed: u32,
    stall_count: u32,
    last_content_hash: String,
}

/// Reads only the live owning dispatch's acknowledged cursor; nested public orchestration has no ancestor cursor authority.
fn read_chat_cursor<T: serde::de::DeserializeOwned>(key: &str) -> Result<Option<T>> {
    let Some(execution) = crate::autonomy::current_execution() else {
        return Ok(None);
    };
    let Some(frame) = execution.composition_frame()? else {
        return Ok(None);
    };
    frame
        .cursor
        .get(key)
        .map(|value| serde_json::from_value(value.clone()).map_err(AgentError::from))
        .transpose()
}

/// Each completed conversation step is stored before any later child can suspend; failed acknowledgement retains ownership.
async fn checkpoint_chat_cursor<T: serde::Serialize>(key: &str, cursor: &T) -> Result<()> {
    let Some(execution) = crate::autonomy::current_execution() else {
        return Ok(());
    };
    let Some(frame) = execution.composition_frame()? else {
        return Ok(());
    };
    let mut saved = frame.cursor;
    saved[key] = serde_json::to_value(cursor)?;
    Box::pin(execution.checkpoint_composition_cursor(&frame.runtime_id, saved)).await?;
    Ok(())
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
        ChatMessage::system(format!(
            "You are a conversation manager.\n\
             Pick which participant should speak next.\n\
             Respond with only the participant ID, nothing else.\n\n\
             Participants:\n{}",
            participant_list
        )),
        ChatMessage::user(format!(
            "Topic: {}\n\nConversation so far:\n{}\n\nWho should speak next?",
            topic, transcript_text
        )),
    ];

    let response = with_observation_purpose(
        ObservationPurpose::OrchestrationConversation,
        ai_agents_llm::managed_completion(llm, &messages, None),
    )
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
/// The separate mutable transcript and context arguments preserve coupled round update ordering.
#[allow(clippy::too_many_arguments)]
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
    let mut cursor =
        read_chat_cursor::<GroupChatCursor>("group_chat")?.unwrap_or_else(|| GroupChatCursor {
            round,
            speaker_index: 0,
            order: Vec::new(),
            pending_speaker: None,
            phase: 0,
            transcript: transcript.clone(),
            context: accumulated_context.clone(),
            rounds_completed: round,
            stall_count: 0,
            last_content_hash: String::new(),
        });
    let mut speakers_this_round = cursor.speaker_index as u32;

    while speakers_this_round < max_speakers {
        let next_id = if let Some(speaker) = &cursor.pending_speaker {
            speaker.clone()
        } else {
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
                    current_turn_actor_context(),
                    &format!("round:{round}:select:{speakers_this_round}"),
                )
                .await?
            } else {
                select_next_speaker(llm, &config.participants, transcript, topic).await?
            }
        };
        cursor.pending_speaker = Some(next_id.clone());
        Box::pin(checkpoint_chat_cursor("group_chat", &cursor)).await?;

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

        let response = chat_agent(
            agent.as_ref(),
            &prompt,
            current_turn_actor_context(),
            &format!("round:{round}:speaker:{speakers_this_round}"),
        )
        .await?;
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
        cursor.speaker_index = speakers_this_round as usize;
        cursor.pending_speaker = None;
        cursor.transcript = transcript.clone();
        cursor.context = accumulated_context.clone();
        Box::pin(checkpoint_chat_cursor("group_chat", &cursor)).await?;
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

    let response = with_observation_purpose(
        ObservationPurpose::OrchestrationConversation,
        ai_agents_llm::managed_completion(llm, &messages, None),
    )
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
    manager: &RuntimeAgent,
    topic: &str,
    transcript: &[ChatTurn],
    actor_context: Option<TurnActorContext>,
    coordinate: &str,
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

    let response = chat_agent(manager, &prompt, actor_context, coordinate).await?;
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
    if let Some(start) = raw.find('{')
        && let Some(end) = raw.rfind('}')
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw[start..=end])
    {
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
    manager: &RuntimeAgent,
    participants: &[ai_agents_state::ChatParticipant],
    transcript: &[ChatTurn],
    topic: &str,
    actor_context: Option<TurnActorContext>,
    coordinate: &str,
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

    let response = chat_agent(manager, &prompt, actor_context, coordinate).await?;
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

/// One agent creates content and another reviews it; the task cursor preserves the draft and exact maker/checker position.
/// A parked review never repeats the already acknowledged maker turn or its hook.
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

    let mut cursor =
        read_chat_cursor::<MakerCheckerCursor>("maker_checker")?.unwrap_or_else(|| {
            MakerCheckerCursor {
                iteration: 0,
                reviewing: false,
                draft: String::new(),
                transcript: Vec::new(),
                accepted: false,
            }
        });
    if cursor.iteration > max_iter {
        return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
    }
    if cursor.accepted {
        return Ok(GroupChatResult {
            response: AgentResponse::new(cursor.draft),
            transcript: cursor.transcript,
            rounds_completed: cursor.iteration,
            termination_reason: "accepted".into(),
        });
    }
    let mut transcript = cursor.transcript.clone();
    let mut current_draft = cursor.draft.clone();
    let actor_context = current_turn_actor_context();

    for iteration in cursor.iteration..max_iter {
        if !cursor.reviewing {
            let maker_prompt = if iteration == 0 {
                format!("Create content for: {}", topic)
            } else {
                format!(
                    "Revise your previous draft based on this feedback:\n\nDraft:\n{}\n\n\
                 Feedback from reviewer will follow.",
                    current_draft
                )
            };

            let maker_response = chat_agent(
                maker.as_ref(),
                &maker_prompt,
                actor_context.clone(),
                &format!("maker:{iteration}"),
            )
            .await?;
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
            cursor.reviewing = true;
            cursor.draft = current_draft.clone();
            cursor.transcript = transcript.clone();
            Box::pin(checkpoint_chat_cursor("maker_checker", &cursor)).await?;
        }

        let checker_prompt = format!(
            "Review this content against these criteria: {}\n\nContent:\n{}\n\n\
             If it meets the criteria, respond with 'APPROVED'. \
             Otherwise, provide specific feedback for improvement.",
            criteria, current_draft
        );

        let checker_response = chat_agent(
            checker.as_ref(),
            &checker_prompt,
            actor_context.clone(),
            &format!("checker:{iteration}"),
        )
        .await?;
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

        cursor.accepted = feedback.to_uppercase().contains("APPROVED");
        cursor.iteration = iteration + 1;
        cursor.reviewing = false;
        cursor.transcript = transcript.clone();
        Box::pin(checkpoint_chat_cursor("maker_checker", &cursor)).await?;
        if cursor.accepted {
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

/// Exact maker/checker position and draft are independent from the general conversation cursor.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MakerCheckerCursor {
    iteration: u32,
    reviewing: bool,
    draft: String,
    transcript: Vec<ChatTurn>,
    accepted: bool,
}

/// Debate coordinates preserve acknowledged arguments and a separately completed synthesis.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DebateCursor {
    round: u32,
    participant_index: usize,
    transcript: Vec<ChatTurn>,
    text: String,
    conclusion: Option<String>,
}

/// Structured rounds preserve the participant cursor and do not repeat synthesis after its result is acknowledged.
async fn run_debate(
    registry: &AgentRegistry,
    topic: &str,
    config: &GroupChatStateConfig,
    _llm: Option<&dyn LLMProvider>,
    hooks: Option<&dyn AgentHooks>,
) -> Result<GroupChatResult> {
    let rounds = config.debate.as_ref().map(|d| d.rounds).unwrap_or(3);

    let mut cursor = read_chat_cursor::<DebateCursor>("debate")?.unwrap_or_else(|| DebateCursor {
        round: 0,
        participant_index: 0,
        transcript: Vec::new(),
        text: String::new(),
        conclusion: None,
    });
    if cursor.round > rounds || cursor.participant_index > config.participants.len() {
        return Err(crate::autonomy::TaskRunStorageError::InvalidCheckpoint.into());
    }
    let mut transcript = cursor.transcript.clone();
    let mut debate_text = cursor.text.clone();
    let actor_context = current_turn_actor_context();

    for round in cursor.round..rounds {
        for (index, participant) in config
            .participants
            .iter()
            .enumerate()
            .skip(cursor.participant_index)
        {
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

            let response = chat_agent(
                agent.as_ref(),
                &prompt,
                actor_context.clone(),
                &format!("debate:{round}:{index}"),
            )
            .await?;
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
            cursor.participant_index = index + 1;
            cursor.text = debate_text.clone();
            cursor.transcript = transcript.clone();
            Box::pin(checkpoint_chat_cursor("debate", &cursor)).await?;
        }
        cursor.round = round + 1;
        cursor.participant_index = 0;
        Box::pin(checkpoint_chat_cursor("debate", &cursor)).await?;
    }

    // Synthesize the debate if a synthesizer agent is configured.
    let conclusion = if let Some(conclusion) = &cursor.conclusion {
        conclusion.clone()
    } else if let Some(ref debate_config) = config.debate {
        if let Some(synth_agent) = registry.get(&debate_config.synthesizer) {
            let synth_prompt = format!(
                "Synthesize this debate into a balanced conclusion:\n\n{}",
                debate_text
            );
            let synth_response = chat_agent(
                synth_agent.as_ref(),
                &synth_prompt,
                actor_context.clone(),
                "debate:synthesis",
            )
            .await?;
            synth_response.content
        } else {
            debate_text
        }
    } else {
        debate_text
    };

    cursor.conclusion = Some(conclusion.clone());
    Box::pin(checkpoint_chat_cursor("debate", &cursor)).await?;
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
