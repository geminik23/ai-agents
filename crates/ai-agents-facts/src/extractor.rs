//! Fact extraction from conversation messages using an LLM.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tracing::{debug, warn};
use uuid::Uuid;

use ai_agents_core::types::{FactCategory, KeyFact};
use ai_agents_core::{ChatMessage, LLMConfig, LLMProvider, Result, Role};

use crate::config::{CategoryDefinition, FactsConfig};
use crate::dedup::deduplicate_exact;

/// Extracts facts from conversation messages using an LLM.
#[async_trait]
pub trait FactExtractor: Send + Sync {
    /// Extract facts from recent messages, given existing facts for deduplication.
    async fn extract(
        &self,
        messages: &[ChatMessage],
        existing_facts: &[KeyFact],
        actor_id: Option<&str>,
        categories: &[CategoryDefinition],
    ) -> Result<Vec<KeyFact>>;
}

/// LLM-based fact extractor.
pub struct LLMFactExtractor {
    llm: Arc<dyn LLMProvider>,
    config: FactsConfig,
}

impl LLMFactExtractor {
    pub fn new(llm: Arc<dyn LLMProvider>, config: FactsConfig) -> Self {
        Self { llm, config }
    }

    /// Build the extraction prompt with category descriptions and existing facts.
    fn build_prompt(
        &self,
        messages: &[ChatMessage],
        existing_facts: &[KeyFact],
        categories: &[CategoryDefinition],
    ) -> String {
        if let Some(ref custom) = self.config.extraction_prompt {
            return custom.clone();
        }

        let mut prompt = String::from(
            "You are a fact extraction assistant.\n\
             Extract key facts from the conversation below.\n\n\
             Categories to look for:\n",
        );

        // Built-in category descriptions.
        let builtins = [
            (
                "user_preference",
                "Preferences, likes, dislikes, and choices the actor expressed",
            ),
            (
                "user_context",
                "Background info about the user: job title, location, home, family, personal situation. Never about the assistant.",
            ),
            (
                "decision",
                "Decisions the user made during the conversation",
            ),
            ("agreement", "Things the user agreed to or confirmed"),
        ];

        for cat_name in &self.config.categories {
            if let Some((_, desc)) = builtins.iter().find(|(n, _)| n == cat_name) {
                prompt.push_str(&format!("- {}: {}\n", cat_name, desc));
            }
        }

        for cat in categories {
            prompt.push_str(&format!("- {}: {}\n", cat.name, cat.description));
        }

        // If no categories configured, include all builtins.
        if self.config.categories.is_empty() && categories.is_empty() {
            for (name, desc) in &builtins {
                prompt.push_str(&format!("- {}: {}\n", name, desc));
            }
        }

        prompt.push_str(
            "\nFor each fact, output a JSON array. Each element:\n\
             {\n  \"category\": \"<category_name>\",\n  \"content\": \"<fact in English>\",\n  \
             \"confidence\": <0.0 to 1.0>,\n  \"actor_id\": \"<who this fact is about, or null>\"\n}\n\n\
             Content style rules (very important):\n\
             - Write each fact as a short, direct statement about the user.\n\
             - Start the content with \"User \" or use the user's name when known. Examples: \"User name is Jay\", \"User works as an AI engineer\", \"Jay prefers Python\".\n\
             - Do NOT write facts in third person about \"the actor\", \"this person\", \"the user's identity\", or similar abstract phrasing.\n\
             - Keep each fact under 15 words. One fact per item, no compound sentences.\n\n\
             Selection rules:\n\
             - Output facts in English regardless of conversation language.\n\
             - \"actor_id\" is the person the fact describes. null if general.\n\
             - Only extract facts with confidence >= 0.5.\n\
             - Do not extract greetings, filler, or obvious conversation mechanics.\n\
             - If a fact updates or contradicts an existing fact, extract the new version.\n\
             - Do NOT extract statements about what the assistant knows, can do, or cannot access.\n\
             - Do NOT extract the assistant's capabilities, limitations, or tool access (e.g. do not extract 'the assistant cannot access account details').\n\
             - Do NOT extract meta-facts about the conversation itself (e.g. what was asked, what was said, or the act of asking).\n\
             - Only extract concrete facts about the user: name, job, location, background, preferences, decisions, and agreements.\n\
             - Pick the most specific category. A name, job, or location is user_context, NOT user_preference.\n",
        );

        // Both dedup methods include the existing fact list so the LLM can avoid
        // extracting paraphrases or semantic duplicates. The difference is what
        // happens after parsing: llm trusts the LLM output and skips the local
        // Levenshtein filter; exact runs the Levenshtein filter as an additional
        // safety net on top of the LLM-level guidance.
        if !existing_facts.is_empty() {
            match self.config.dedup.method {
                crate::config::DedupMethod::Llm => {
                    prompt.push_str(
                        "\nExisting facts (do not duplicate or paraphrase, but update if contradicted):\n",
                    );
                    for fact in existing_facts {
                        prompt.push_str(&format!("- [{}] {}\n", fact.category, fact.content));
                    }
                }
                crate::config::DedupMethod::Exact => {
                    // Include the full list so the LLM knows what is already stored
                    // and avoids generating paraphrases. The Levenshtein filter still
                    // runs post-parse as a second layer of protection.
                    prompt.push_str(
                        "\nExisting facts (do not extract duplicates or paraphrases of these):\n",
                    );
                    for fact in existing_facts {
                        prompt.push_str(&format!("- [{}] {}\n", fact.category, fact.content));
                    }
                }
            }
        }

        prompt.push_str("\nConversation:\n");
        for msg in messages {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
                _ => "Other",
            };
            prompt.push_str(&format!("{}: {}\n", role, msg.content));
        }

        prompt.push_str("\nOutput JSON array only. No explanation.\n");

        prompt
    }

    /// Parse the LLM response into KeyFact structs.
    fn parse_response(&self, response: &str, actor_id: Option<&str>) -> Vec<KeyFact> {
        // Try to find a JSON array in the response.
        let json_str = extract_json_array(response);
        let json_str = match json_str {
            Some(s) => s,
            None => {
                warn!("fact extractor: no JSON array found in LLM response");
                return vec![];
            }
        };

        let parsed: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => {
                warn!("fact extractor: failed to parse JSON: {}", e);
                return vec![];
            }
        };

        let now = Utc::now();
        let mut facts = Vec::new();

        for item in parsed {
            let category_str = item
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let content = item
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let confidence = item
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let fact_actor = item
                .get("actor_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| actor_id.map(|s| s.to_string()));

            if content.is_empty() || confidence < 0.5 {
                continue;
            }

            let category = parse_category(&category_str);

            facts.push(KeyFact {
                id: Uuid::new_v4().to_string(),
                actor_id: fact_actor,
                category,
                content,
                confidence,
                salience: 1.0,
                extracted_at: now,
                last_accessed: None,
                source_message_id: None,
                source_language: None,
            });
        }

        facts
    }
}

#[async_trait]
impl FactExtractor for LLMFactExtractor {
    async fn extract(
        &self,
        messages: &[ChatMessage],
        existing_facts: &[KeyFact],
        actor_id: Option<&str>,
        categories: &[CategoryDefinition],
    ) -> Result<Vec<KeyFact>> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let prompt = self.build_prompt(messages, existing_facts, categories);

        let extraction_messages = vec![ChatMessage {
            role: Role::User,
            content: prompt,
            name: None,
            timestamp: None,
        }];

        let config = LLMConfig {
            temperature: Some(0.1),
            max_tokens: Some(2048),
            ..Default::default()
        };

        let response = self.llm.complete(&extraction_messages, Some(&config)).await;

        match response {
            Ok(llm_response) => {
                let mut facts = self.parse_response(&llm_response.content, actor_id);

                // Apply local exact dedup only when method = exact.
                // With method = llm, the prompt already instructs the LLM to skip duplicates,
                // so we keep its output intact to allow semantic updates to surface.
                if self.config.dedup.enabled
                    && self.config.dedup.method == crate::config::DedupMethod::Exact
                {
                    facts = deduplicate_exact(&facts, existing_facts);
                }

                debug!(
                    "fact extractor: extracted {} facts from {} messages",
                    facts.len(),
                    messages.len()
                );

                Ok(facts)
            }
            Err(e) => {
                // Propagate as Err so the caller (auto_extract_facts) can log a
                // visible warning under the runtime module filter.
                Err(ai_agents_core::AgentError::LLM(format!(
                    "fact extractor LLM call failed: {}",
                    e
                )))
            }
        }
    }
}

/// Parse a category string into FactCategory enum.
fn parse_category(s: &str) -> FactCategory {
    match s {
        "user_preference" | "preference" => FactCategory::UserPreference,
        "user_context" | "context" => FactCategory::UserContext,
        "decision" => FactCategory::Decision,
        "agreement" => FactCategory::Agreement,
        _ => FactCategory::Custom(s.to_string()),
    }
}

/// Extract a JSON array substring from potentially noisy LLM output.
fn extract_json_array(text: &str) -> Option<String> {
    // Try direct parse first.
    if serde_json::from_str::<Vec<serde_json::Value>>(text).is_ok() {
        return Some(text.to_string());
    }

    // Look for ```json ... ``` fenced blocks.
    if let Some(start) = text.find("```json") {
        let after_fence = &text[start + 7..];
        if let Some(end) = after_fence.find("```") {
            let inner = after_fence[..end].trim();
            if serde_json::from_str::<Vec<serde_json::Value>>(inner).is_ok() {
                return Some(inner.to_string());
            }
        }
    }
    if let Some(start) = text.find("```") {
        let after_fence = &text[start + 3..];
        if let Some(end) = after_fence.find("```") {
            let inner = after_fence[..end].trim();
            if serde_json::from_str::<Vec<serde_json::Value>>(inner).is_ok() {
                return Some(inner.to_string());
            }
        }
    }

    // Look for the first '[' and last ']'.
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                let candidate = &text[start..=end];
                if serde_json::from_str::<Vec<serde_json::Value>>(candidate).is_ok() {
                    return Some(candidate.to_string());
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_category() {
        assert_eq!(
            parse_category("user_preference"),
            FactCategory::UserPreference
        );
        assert_eq!(parse_category("preference"), FactCategory::UserPreference);
        assert_eq!(parse_category("user_context"), FactCategory::UserContext);
        assert_eq!(parse_category("decision"), FactCategory::Decision);
        assert_eq!(parse_category("agreement"), FactCategory::Agreement);
        assert_eq!(
            parse_category("suspicion"),
            FactCategory::Custom("suspicion".to_string())
        );
    }

    #[test]
    fn test_extract_json_array_direct() {
        let input = r#"[{"category":"decision","content":"Chose plan A","confidence":0.9,"actor_id":null}]"#;
        let result = extract_json_array(input);
        assert!(result.is_some());
    }

    #[test]
    fn test_extract_json_array_fenced() {
        let input = "Here are the facts:\n```json\n[{\"category\":\"decision\",\"content\":\"test\",\"confidence\":0.8}]\n```\n";
        let result = extract_json_array(input);
        assert!(result.is_some());
    }

    #[test]
    fn test_extract_json_array_embedded() {
        let input = "The extracted facts are: [{\"category\":\"decision\",\"content\":\"test\",\"confidence\":0.8}] and that is all.";
        let result = extract_json_array(input);
        assert!(result.is_some());
    }

    #[test]
    fn test_extract_json_array_empty() {
        let input = "[]";
        let result = extract_json_array(input);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "[]");
    }

    #[test]
    fn test_extract_json_array_no_array() {
        let input = "No facts found in this conversation.";
        let result = extract_json_array(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_basic() {
        let extractor = LLMFactExtractor::new(Arc::new(MockLLM), FactsConfig::default());

        let response = r#"[
            {"category": "user_preference", "content": "Prefers vegetarian food", "confidence": 0.95, "actor_id": null},
            {"category": "decision", "content": "Chose the premium plan", "confidence": 0.88, "actor_id": null}
        ]"#;

        let facts = extractor.parse_response(response, Some("user_1"));
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].category, FactCategory::UserPreference);
        assert_eq!(facts[0].content, "Prefers vegetarian food");
        assert_eq!(facts[0].confidence, 0.95);
        assert_eq!(facts[0].actor_id.as_deref(), Some("user_1"));
        assert_eq!(facts[1].category, FactCategory::Decision);
    }

    #[test]
    fn test_parse_response_filters_low_confidence() {
        let extractor = LLMFactExtractor::new(Arc::new(MockLLM), FactsConfig::default());

        let response = r#"[
            {"category": "decision", "content": "Maybe likes coffee", "confidence": 0.3},
            {"category": "decision", "content": "Definitely likes tea", "confidence": 0.9}
        ]"#;

        let facts = extractor.parse_response(response, None);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "Definitely likes tea");
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let extractor = LLMFactExtractor::new(Arc::new(MockLLM), FactsConfig::default());

        let facts = extractor.parse_response("not json at all", None);
        assert!(facts.is_empty());
    }

    // Verify that the extraction prompt contains the anti-noise rules that
    // prevent the LLM from extracting statements about the assistant's
    // capabilities or limitations (e.g. "the assistant cannot access X").
    #[test]
    fn test_build_prompt_contains_anti_noise_rules() {
        let extractor = LLMFactExtractor::new(Arc::new(MockLLM), FactsConfig::default());
        let prompt = extractor.build_prompt(&[], &[], &[]);

        assert!(
            prompt.contains("Do NOT extract statements about what the assistant knows"),
            "prompt must forbid extracting assistant-capability statements"
        );
        assert!(
            prompt.contains("Do NOT extract the assistant's capabilities"),
            "prompt must forbid extracting assistant limitations"
        );
        assert!(
            prompt.contains("Only extract concrete facts about the user"),
            "prompt must restrict extraction to user facts"
        );
    }

    // Verify that with method = exact and existing facts, the prompt includes
    // the full existing fact list (not just a count) so the LLM can avoid
    // generating paraphrases like "The user is named Jay" when "The user's
    // name is Jay" already exists.
    #[test]
    fn test_build_prompt_exact_dedup_includes_existing_facts() {
        use ai_agents_core::types::FactCategory;
        use chrono::Utc;

        let config = FactsConfig::default(); // dedup.method defaults to Exact
        let extractor = LLMFactExtractor::new(Arc::new(MockLLM), config);

        let existing = vec![KeyFact {
            id: "id-1".to_string(),
            actor_id: Some("user_1".to_string()),
            category: FactCategory::UserContext,
            content: "The user's name is Jay".to_string(),
            confidence: 0.90,
            salience: 1.0,
            extracted_at: Utc::now(),
            last_accessed: None,
            source_message_id: None,
            source_language: None,
        }];

        let prompt = extractor.build_prompt(&[], &existing, &[]);

        assert!(
            prompt.contains("The user's name is Jay"),
            "exact dedup prompt must list existing fact content so LLM avoids paraphrases"
        );
        assert!(
            prompt.contains("do not extract duplicates or paraphrases"),
            "exact dedup prompt must instruct the LLM to skip paraphrases"
        );
    }

    // Verify that with method = llm, the prompt also includes existing facts
    // with the instruction to avoid duplicating or paraphrasing them.
    #[test]
    fn test_build_prompt_llm_dedup_includes_existing_facts() {
        use crate::config::{DedupConfig, DedupMethod};
        use ai_agents_core::types::FactCategory;
        use chrono::Utc;

        let mut config = FactsConfig::default();
        config.dedup = DedupConfig {
            enabled: true,
            method: DedupMethod::Llm,
        };
        let extractor = LLMFactExtractor::new(Arc::new(MockLLM), config);

        let existing = vec![KeyFact {
            id: "id-2".to_string(),
            actor_id: Some("user_1".to_string()),
            category: FactCategory::UserContext,
            content: "Jay works as an AI engineer".to_string(),
            confidence: 0.90,
            salience: 1.0,
            extracted_at: Utc::now(),
            last_accessed: None,
            source_message_id: None,
            source_language: None,
        }];

        let prompt = extractor.build_prompt(&[], &existing, &[]);

        assert!(
            prompt.contains("Jay works as an AI engineer"),
            "llm dedup prompt must list existing fact content"
        );
        assert!(
            prompt.contains("do not duplicate or paraphrase"),
            "llm dedup prompt must instruct the LLM to skip paraphrases"
        );
    }

    // Minimal mock LLM for testing parse_response (no actual LLM calls).
    struct MockLLM;

    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<ai_agents_core::types::LLMResponse, ai_agents_core::LLMError>
        {
            Ok(ai_agents_core::types::LLMResponse::new(
                "[]",
                ai_agents_core::types::FinishReason::Stop,
            ))
        }

        async fn complete_stream(
            &self,
            _messages: &[ChatMessage],
            _config: Option<&LLMConfig>,
        ) -> std::result::Result<
            Box<
                dyn futures::Stream<
                        Item = std::result::Result<
                            ai_agents_core::LLMChunk,
                            ai_agents_core::LLMError,
                        >,
                    > + Unpin
                    + Send,
            >,
            ai_agents_core::LLMError,
        > {
            Err(ai_agents_core::LLMError::Other(
                "streaming not supported".into(),
            ))
        }

        fn provider_name(&self) -> &str {
            "mock"
        }

        fn supports(&self, _capability: ai_agents_core::LLMFeature) -> bool {
            false
        }
    }
}
