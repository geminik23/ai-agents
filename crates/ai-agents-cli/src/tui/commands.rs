//! Session commands run outside the input/render reducer while preserving the existing runtime APIs.
use crate::repl::parse_relationship_perspective;
use ai_agents::RuntimeAgent;
use std::sync::Arc;

pub(super) struct CommandContext {
    pub(super) agent: Arc<RuntimeAgent>,
    pub(super) messages: Vec<String>,
}
impl CommandContext {
    pub(super) async fn run(&mut self, input: &str) {
        let mut words = input.split_whitespace();
        let command = words.next().unwrap_or("").to_ascii_lowercase();
        let argument = words.next();
        let name = argument.unwrap_or("default");
        if command == "/delete" && argument.is_none() {
            self.add_system_message("Usage: /delete <session_name>");
            return;
        }
        let result: anyhow::Result<String> = match command.as_str() {
            "/reset" => self
                .agent
                .reset()
                .await
                .map(|_| "Agent reset.".into())
                .map_err(Into::into),
            "/save" => self
                .agent
                .save_session(name)
                .await
                .map(|_| format!("Saved '{name}'"))
                .map_err(Into::into),
            "/load" => self
                .agent
                .load_session(name)
                .await
                .map(|found| format!("Session '{name}' loaded: {found}"))
                .map_err(Into::into),
            "/delete" => self
                .agent
                .delete_session(name)
                .await
                .map(|_| format!("Deleted '{name}'"))
                .map_err(Into::into),
            "/cleanup" => self
                .agent
                .cleanup_expired_sessions()
                .await
                .map(|n| format!("Cleaned up {n} expired sessions"))
                .map_err(Into::into),
            "/actor" => {
                self.handle_tui_actor_command(input).await;
                return;
            }
            "/facts" => {
                self.handle_tui_facts_command(input).await;
                return;
            }
            "/relationship" | "/rel" => {
                self.handle_tui_relationship_command(input).await;
                return;
            }
            "/sessions" => {
                self.handle_tui_sessions_command(input).await;
                return;
            }
            _ => Ok("Unknown session command".into()),
        };
        match result {
            Ok(message) => self.add_system_message(&message),
            Err(e) => self.add_system_message(&format!("[Error] {e}")),
        }
    }
    fn add_system_message(&mut self, text: &str) {
        self.messages.push(text.to_owned());
    }
    fn add_toast(&mut self, text: &str) {
        self.add_system_message(text);
    }
    async fn handle_tui_actor_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("") => match self.agent.actor_id() {
                Some(id) => self.add_system_message(&format!("Actor: {}", id)),
                None => self.add_system_message("No actor ID set. Use /actor set <id>"),
            },
            Some("set") => {
                if let Some(id) = parts.get(2) {
                    match self.agent.set_actor_id(id) {
                        Ok(()) => {
                            if let Err(e) = self.agent.load_actor_memory().await {
                                self.add_system_message(&format!(
                                    "[Warning] Actor set but failed to load memory: {}",
                                    e
                                ));
                            } else {
                                let count = self.agent.actor_facts().len();
                                self.add_toast(&format!(
                                    "Actor set to: {}. {} fact(s) loaded.",
                                    id, count
                                ));
                            }
                        }
                        Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                    }
                } else {
                    self.add_system_message("Usage: /actor set <id>");
                }
            }
            Some("facts") => {
                let actor_id = match self.agent.actor_id() {
                    Some(id) => id,
                    None => {
                        self.add_system_message("No actor ID set. Use /actor set <id>");
                        return;
                    }
                };
                let facts = self.agent.actor_facts();
                let cat_filter = parts.get(2).map(|s| s.to_lowercase());
                let filtered: Vec<_> = if let Some(ref cat) = cat_filter {
                    facts
                        .iter()
                        .filter(|f| f.category.to_string().to_lowercase() == *cat)
                        .collect()
                } else {
                    facts.iter().collect()
                };
                if filtered.is_empty() {
                    self.add_system_message(&format!("No facts for actor: {}", actor_id));
                } else {
                    let mut msg =
                        format!("Facts for actor {} ({} shown):", actor_id, filtered.len());
                    for fact in &filtered {
                        msg.push_str(&format!(
                            "\n  [{}] {} ({:.2})",
                            fact.category, fact.content, fact.confidence
                        ));
                    }
                    self.add_system_message(&msg);
                }
            }
            Some("delete") => {
                let actor_id = match self.agent.actor_id() {
                    Some(id) => id,
                    None => {
                        self.add_system_message("No actor ID set. Use /actor set <id>");
                        return;
                    }
                };
                match self.agent.delete_actor_data(&actor_id).await {
                    Ok(()) => self.add_toast(&format!("All data deleted for actor: {}", actor_id)),
                    Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                }
            }
            _ => {
                self.add_system_message("Usage: /actor [set <id> | facts [category] | delete]");
            }
        }
    }

    async fn handle_tui_facts_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
            Some("extract") => {
                let n: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
                match self.agent.extract_facts(n).await {
                    Ok(facts) if facts.is_empty() => {
                        self.add_system_message("No new facts extracted.");
                    }
                    Ok(facts) => {
                        let mut msg = format!("Extracted {} fact(s):", facts.len());
                        for fact in &facts {
                            msg.push_str(&format!(
                                "\n  [{}] {} ({:.2})",
                                fact.category, fact.content, fact.confidence
                            ));
                        }
                        self.add_system_message(&msg);
                    }
                    Err(e) => {
                        self.add_system_message(&format!("[Error] Extraction failed: {}", e));
                    }
                }
            }
            _ => {
                let facts = self.agent.actor_facts();
                if facts.is_empty() {
                    self.add_system_message("No facts for current actor.");
                } else {
                    let mut msg = format!("Facts ({}):", facts.len());
                    for fact in &facts {
                        msg.push_str(&format!(
                            "\n  [{}] {} ({:.2})",
                            fact.category, fact.content, fact.confidence
                        ));
                    }
                    self.add_system_message(&msg);
                }
            }
        }
    }

    async fn handle_tui_relationship_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let Some(manager) = self.agent.relationship_manager() else {
            self.add_system_message("Relationship memory is not configured.");
            return;
        };
        let actor_id = match self.agent.actor_id() {
            Some(id) => id,
            None => {
                self.add_system_message("No actor ID set. Use /actor set <id>");
                return;
            }
        };
        let _ = self.agent.load_actor_relationship().await;

        match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
            None | Some("") => {},
            Some("events") => {
                let relationship = manager.get_or_create(&actor_id, None);
                if relationship.notable_events.is_empty() {
                    self.add_system_message(&format!(
                        "No notable relationship events for actor: {}",
                        actor_id
                    ));
                    return;
                }
                let mut msg = format!("Relationship events for actor {}:", actor_id);
                for event in &relationship.notable_events {
                    msg.push_str(&format!(
                        "\n  [{}] ({:.2}) {}",
                        event.timestamp.to_rfc3339(),
                        event.significance,
                        event.description
                    ));
                }
                self.add_system_message(&msg);
            }
            Some("set") => {
                let dimension = match parts.get(2) {
                    Some(d) => *d,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship set <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let delta = match parts.get(3).and_then(|s| s.parse::<f64>().ok()) {
                    Some(v) => v,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship set <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let reason = if parts.len() > 4 {
                    Some(parts[4..].join(" "))
                } else {
                    None
                };
                match self
                    .agent
                    .update_relationship_dimension(dimension, delta, reason.as_deref())
                    .await
                {
                    Ok(change) => self.add_toast(&format!(
                        "{} {} {:+.2} -> {:.2}",
                        change.perspective, change.dimension, change.delta, change.current
                    )),
                    Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                }
            }
            Some("setp") => {
                let perspective = match parts.get(2) {
                    Some(value) => match parse_relationship_perspective(value) {
                        Ok(p) => p,
                        Err(e) => {
                            self.add_system_message(&format!(
                                "{}. Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                                e
                            ));
                            return;
                        }
                    },
                    None => {
                        self.add_system_message(
                            "Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let dimension = match parts.get(3) {
                    Some(d) => *d,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let delta = match parts.get(4).and_then(|s| s.parse::<f64>().ok()) {
                    Some(v) => v,
                    None => {
                        self.add_system_message(
                            "Usage: /relationship setp <perspective> <dimension> <delta> [reason]",
                        );
                        return;
                    }
                };
                let reason = if parts.len() > 5 {
                    Some(parts[5..].join(" "))
                } else {
                    None
                };
                match self
                    .agent
                    .update_relationship_dimension_for_perspective(
                        perspective,
                        dimension,
                        delta,
                        reason.as_deref(),
                    )
                    .await
                {
                    Ok(change) => self.add_toast(&format!(
                        "{} {} {:+.2} -> {:.2}",
                        change.perspective, change.dimension, change.delta, change.current
                    )),
                    Err(e) => self.add_system_message(&format!("[Error] {}", e)),
                }
            }
            _ => self.add_system_message(
                "Usage: /relationship, /relationship events, /relationship set <dimension> <delta> [reason], /relationship setp <perspective> <dimension> <delta> [reason]",
            ),
        }
    }

    async fn handle_tui_sessions_command(&mut self, input: &str) {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let mut actor_filter: Option<String> = None;
        let mut tag_filter: Option<String> = None;
        let mut i = 1;
        while i < parts.len() {
            match parts[i] {
                "--actor" => {
                    if let Some(v) = parts.get(i + 1) {
                        actor_filter = Some(v.to_string());
                        i += 2;
                        continue;
                    }
                }
                "--tag" => {
                    if let Some(v) = parts.get(i + 1) {
                        tag_filter = Some(v.to_string());
                        i += 2;
                        continue;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        if actor_filter.is_some() || tag_filter.is_some() {
            let filter = ai_agents::facts::SessionFilter {
                actor_id: actor_filter,
                tags: tag_filter.map(|t| vec![t]),
                agent_id: None,
                created_after: None,
                created_before: None,
                limit: None,
            };
            match self.agent.list_sessions_filtered(&filter).await {
                Ok(summaries) if summaries.is_empty() => {
                    self.add_system_message("No sessions match the filter.");
                }
                Ok(summaries) => {
                    let mut msg = format!("Sessions ({} matched):", summaries.len());
                    for s in &summaries {
                        let actor = s.actor_id.as_deref().unwrap_or("-");
                        let tags = if s.tags.is_empty() {
                            String::new()
                        } else {
                            format!(" [{}]", s.tags.join(","))
                        };
                        msg.push_str(&format!(
                            "\n  {}  actor={}{}  msgs={}",
                            s.session_id, actor, tags, s.message_count
                        ));
                    }
                    self.add_system_message(&msg);
                }
                Err(e) => self.add_system_message(&format!("[Error] {}", e)),
            }
            return;
        }

        match self.agent.list_sessions().await {
            Ok(sessions) if sessions.is_empty() => {
                self.add_system_message("No saved sessions.");
            }
            Ok(sessions) => {
                let mut msg = format!("Sessions ({}):", sessions.len());
                for s in &sessions {
                    msg.push_str(&format!("\n  {}", s));
                }
                self.add_system_message(&msg);
            }
            Err(e) => self.add_system_message(&format!("[Error] {}", e)),
        }
    }
}
