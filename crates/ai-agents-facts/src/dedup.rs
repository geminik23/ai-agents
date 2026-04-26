//! Deduplication logic for extracted facts.

use ai_agents_core::types::KeyFact;

/// Remove facts that duplicate existing ones using normalized string comparison.
///
/// A new fact is considered a duplicate when:
/// - Same category and similarity >= 0.85 (catches near-paraphrases like
///   "User name is Jay" vs "User is named Jay"), OR
/// - Different category but content is essentially identical (similarity >= 0.95).
///   This guards against the LLM re-classifying the same content under a
///   different category (e.g. "User name is Jay" stored once as user_context
///   and again as user_preference).
///
/// Cross-category dedup intentionally only fires for high-similarity strings.
/// Two semantically-related but distinct facts (e.g. user_context "User lives
/// in Berlin" vs user_preference "User likes Berlin") have different content
/// and stay separate.
pub fn deduplicate_exact(new_facts: &[KeyFact], existing_facts: &[KeyFact]) -> Vec<KeyFact> {
    new_facts
        .iter()
        .filter(|new| {
            let dominated = existing_facts.iter().any(|existing| {
                let sim = normalized_similarity(&new.content, &existing.content);
                if existing.category == new.category {
                    sim >= 0.85
                } else {
                    sim >= 0.95
                }
            });
            !dominated
        })
        .cloned()
        .collect()
}

/// Normalize a string for comparison: lowercase, trim, collapse whitespace.
fn normalize(s: &str) -> String {
    let trimmed = s.trim().to_lowercase();
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_space = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result
}

/// Compute similarity between two strings using Levenshtein ratio.
///
/// Returns a value between 0.0 (completely different) and 1.0 (identical).
fn normalized_similarity(a: &str, b: &str) -> f64 {
    let a = normalize(a);
    let b = normalize(b);

    if a == b {
        return 1.0;
    }

    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }

    let dist = levenshtein(&a, &b);
    1.0 - (dist as f64 / max_len as f64)
}

/// Basic Levenshtein distance implementation.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_agents_core::types::FactCategory;
    use chrono::Utc;

    fn make_fact(category: FactCategory, content: &str) -> KeyFact {
        KeyFact {
            id: uuid::Uuid::new_v4().to_string(),
            actor_id: Some("test".to_string()),
            category,
            content: content.to_string(),
            confidence: 0.9,
            salience: 1.0,
            extracted_at: Utc::now(),
            last_accessed: None,
            source_message_id: None,
            source_language: None,
        }
    }

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("  Hello   World  "), "hello world");
        assert_eq!(normalize("ABC"), "abc");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_different() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_normalized_similarity_identical() {
        let sim = normalized_similarity("Actor is vegetarian", "actor is vegetarian");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalized_similarity_very_similar() {
        let sim = normalized_similarity("Actor is vegetarian", "Actor is a vegetarian");
        assert!(sim > 0.85);
    }

    #[test]
    fn test_normalized_similarity_different() {
        let sim = normalized_similarity("Actor is vegetarian", "Actor likes spicy food");
        assert!(sim < 0.7);
    }

    #[test]
    fn test_deduplicate_exact_removes_duplicates() {
        let existing = vec![make_fact(
            FactCategory::UserPreference,
            "Actor is vegetarian",
        )];

        let new_facts = vec![
            make_fact(FactCategory::UserPreference, "Actor is vegetarian"),
            make_fact(FactCategory::Decision, "Chose premium plan"),
        ];

        let result = deduplicate_exact(&new_facts, &existing);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Chose premium plan");
    }

    #[test]
    fn test_deduplicate_exact_keeps_different_category_when_content_differs() {
        // Different content in different category is kept.
        let existing = vec![make_fact(
            FactCategory::UserPreference,
            "User likes vegetarian food",
        )];

        let new_facts = vec![make_fact(FactCategory::Decision, "User chose plan A")];

        let result = deduplicate_exact(&new_facts, &existing);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_deduplicate_exact_drops_cross_category_duplicate() {
        // Same content under a different category is rejected.
        // Without this guard, the LLM can store the same fact twice by
        // assigning it to user_context once and user_preference next time.
        let existing = vec![make_fact(FactCategory::UserContext, "User name is Jay")];

        let new_facts = vec![make_fact(FactCategory::UserPreference, "User name is Jay")];

        let result = deduplicate_exact(&new_facts, &existing);
        assert!(
            result.is_empty(),
            "identical content across categories must be deduped"
        );
    }

    #[test]
    fn test_deduplicate_exact_drops_same_category_near_duplicate() {
        // Same-category near-duplicates with minor edits (added article, plural,
        // small word change) must be deduped under the 0.85 threshold.
        // Levenshtein cannot reliably catch reordered paraphrases like
        // "User name is Jay" vs "User is named Jay" - those are handled by
        // showing existing facts to the LLM in build_prompt(), not here.
        let existing = vec![make_fact(
            FactCategory::UserContext,
            "User works as an AI engineer",
        )];

        let new_facts = vec![make_fact(
            FactCategory::UserContext,
            "User works as a AI engineer",
        )];

        let result = deduplicate_exact(&new_facts, &existing);
        assert!(
            result.is_empty(),
            "same-category near-duplicates must be deduped"
        );
    }

    #[test]
    fn test_deduplicate_exact_no_existing() {
        let new_facts = vec![
            make_fact(FactCategory::UserPreference, "Likes coffee"),
            make_fact(FactCategory::Decision, "Chose plan B"),
        ];

        let result = deduplicate_exact(&new_facts, &[]);
        assert_eq!(result.len(), 2);
    }
}
