//! Secret reveal condition evaluation against context values.

use std::collections::HashMap;

use serde_json::Value;

use crate::config::SecretRevealCondition;
use ai_agents_state::evaluate_context_matchers;

/// Evaluate a secret reveal condition against context values.
/// Returns true if the condition is satisfied.
pub fn evaluate_reveal_condition(
    condition: &SecretRevealCondition,
    context: &HashMap<String, Value>,
) -> bool {
    match condition {
        SecretRevealCondition::Context(matchers) => evaluate_context_matchers(matchers, context),
        SecretRevealCondition::All(conditions) => conditions
            .iter()
            .all(|c| evaluate_reveal_condition(c, context)),
        SecretRevealCondition::Any(conditions) => conditions
            .iter()
            .any(|c| evaluate_reveal_condition(c, context)),
    }
}

/// Check which secrets should be revealed given the current context.
/// Returns indices of secrets whose conditions are satisfied.
pub fn evaluate_secrets(
    secrets: &[crate::config::PersonaSecret],
    context: &HashMap<String, Value>,
) -> Vec<usize> {
    secrets
        .iter()
        .enumerate()
        .filter(|(_, secret)| {
            secret
                .reveal_conditions
                .as_ref()
                .map(|cond| evaluate_reveal_condition(cond, context))
                .unwrap_or(false)
        })
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PersonaSecret, SecretRevealCondition};
    use ai_agents_state::{CompareOp, ContextMatcher};
    use serde_json::json;

    fn make_context(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    fn gte_matcher(threshold: f64) -> ContextMatcher {
        ContextMatcher::Compare(CompareOp::Gte(threshold))
    }

    fn eq_matcher(val: Value) -> ContextMatcher {
        ContextMatcher::Exact(val)
    }

    #[test]
    fn test_single_context_condition_pass() {
        let mut matchers = HashMap::new();
        matchers.insert("trust".to_string(), gte_matcher(0.8));
        let cond = SecretRevealCondition::Context(matchers);

        let ctx = make_context(vec![("trust", json!(0.9))]);
        assert!(evaluate_reveal_condition(&cond, &ctx));
    }

    #[test]
    fn test_single_context_condition_fail() {
        let mut matchers = HashMap::new();
        matchers.insert("trust".to_string(), gte_matcher(0.8));
        let cond = SecretRevealCondition::Context(matchers);

        let ctx = make_context(vec![("trust", json!(0.5))]);
        assert!(!evaluate_reveal_condition(&cond, &ctx));
    }

    #[test]
    fn test_all_conditions() {
        let mut m1 = HashMap::new();
        m1.insert("trust".to_string(), gte_matcher(0.5));
        let mut m2 = HashMap::new();
        m2.insert("is_member".to_string(), eq_matcher(json!(true)));

        let cond = SecretRevealCondition::All(vec![
            SecretRevealCondition::Context(m1),
            SecretRevealCondition::Context(m2),
        ]);

        // Both pass.
        let ctx = make_context(vec![("trust", json!(0.8)), ("is_member", json!(true))]);
        assert!(evaluate_reveal_condition(&cond, &ctx));

        // One fails.
        let ctx2 = make_context(vec![("trust", json!(0.8)), ("is_member", json!(false))]);
        assert!(!evaluate_reveal_condition(&cond, &ctx2));
    }

    #[test]
    fn test_any_conditions() {
        let mut m1 = HashMap::new();
        m1.insert("trust".to_string(), gte_matcher(0.9));
        let mut m2 = HashMap::new();
        m2.insert("is_admin".to_string(), eq_matcher(json!(true)));

        let cond = SecretRevealCondition::Any(vec![
            SecretRevealCondition::Context(m1),
            SecretRevealCondition::Context(m2),
        ]);

        // Only second passes.
        let ctx = make_context(vec![("trust", json!(0.1)), ("is_admin", json!(true))]);
        assert!(evaluate_reveal_condition(&cond, &ctx));

        // Neither passes.
        let ctx2 = make_context(vec![("trust", json!(0.1)), ("is_admin", json!(false))]);
        assert!(!evaluate_reveal_condition(&cond, &ctx2));
    }

    #[test]
    fn test_nested_conditions() {
        let mut m1 = HashMap::new();
        m1.insert("trust".to_string(), gte_matcher(0.5));
        let mut m2 = HashMap::new();
        m2.insert("rank".to_string(), eq_matcher(json!("captain")));
        let mut m3 = HashMap::new();
        m3.insert("is_admin".to_string(), eq_matcher(json!(true)));

        let cond = SecretRevealCondition::All(vec![
            SecretRevealCondition::Context(m1),
            SecretRevealCondition::Any(vec![
                SecretRevealCondition::Context(m2),
                SecretRevealCondition::Context(m3),
            ]),
        ]);

        // trust passes, rank matches.
        let ctx = make_context(vec![
            ("trust", json!(0.7)),
            ("rank", json!("captain")),
            ("is_admin", json!(false)),
        ]);
        assert!(evaluate_reveal_condition(&cond, &ctx));

        // trust passes, is_admin matches.
        let ctx2 = make_context(vec![
            ("trust", json!(0.7)),
            ("rank", json!("private")),
            ("is_admin", json!(true)),
        ]);
        assert!(evaluate_reveal_condition(&cond, &ctx2));

        // trust fails.
        let ctx3 = make_context(vec![
            ("trust", json!(0.2)),
            ("rank", json!("captain")),
            ("is_admin", json!(true)),
        ]);
        assert!(!evaluate_reveal_condition(&cond, &ctx3));
    }

    #[test]
    fn test_missing_context_value_fails() {
        let mut matchers = HashMap::new();
        matchers.insert("trust".to_string(), gte_matcher(0.5));
        let cond = SecretRevealCondition::Context(matchers);

        let ctx = HashMap::new();
        assert!(!evaluate_reveal_condition(&cond, &ctx));
    }

    #[test]
    fn test_evaluate_secrets_selects_matching() {
        let secrets = vec![
            PersonaSecret {
                content: "secret A".into(),
                reveal_conditions: Some(SecretRevealCondition::Context({
                    let mut m = HashMap::new();
                    m.insert("trust".to_string(), gte_matcher(0.8));
                    m
                })),
            },
            PersonaSecret {
                content: "secret B".into(),
                reveal_conditions: Some(SecretRevealCondition::Context({
                    let mut m = HashMap::new();
                    m.insert("trust".to_string(), gte_matcher(0.3));
                    m
                })),
            },
            PersonaSecret {
                content: "manual only".into(),
                reveal_conditions: None,
            },
        ];

        let ctx = make_context(vec![("trust", json!(0.5))]);
        let revealed = evaluate_secrets(&secrets, &ctx);
        // Only secret B (index 1) passes - trust 0.5 >= 0.3 but not >= 0.8.
        // Secret C has no conditions so never auto-reveals.
        assert_eq!(revealed, vec![1]);
    }

    #[test]
    fn test_evaluate_secrets_none_match() {
        let secrets = vec![PersonaSecret {
            content: "hidden".into(),
            reveal_conditions: Some(SecretRevealCondition::Context({
                let mut m = HashMap::new();
                m.insert("trust".to_string(), gte_matcher(0.99));
                m
            })),
        }];

        let ctx = make_context(vec![("trust", json!(0.1))]);
        let revealed = evaluate_secrets(&secrets, &ctx);
        assert!(revealed.is_empty());
    }

    #[test]
    fn test_evaluate_secrets_all_match() {
        let secrets = vec![
            PersonaSecret {
                content: "A".into(),
                reveal_conditions: Some(SecretRevealCondition::Context({
                    let mut m = HashMap::new();
                    m.insert("trust".to_string(), gte_matcher(0.1));
                    m
                })),
            },
            PersonaSecret {
                content: "B".into(),
                reveal_conditions: Some(SecretRevealCondition::Context({
                    let mut m = HashMap::new();
                    m.insert("trust".to_string(), gte_matcher(0.2));
                    m
                })),
            },
        ];

        let ctx = make_context(vec![("trust", json!(0.9))]);
        let revealed = evaluate_secrets(&secrets, &ctx);
        assert_eq!(revealed, vec![0, 1]);
    }

    #[test]
    fn test_nested_context_path() {
        let mut matchers = HashMap::new();
        matchers.insert("relationships.guard.trust".to_string(), gte_matcher(0.7));
        let cond = SecretRevealCondition::Context(matchers);

        let ctx = make_context(vec![("relationships", json!({"guard": {"trust": 0.85}}))]);
        assert!(evaluate_reveal_condition(&cond, &ctx));
    }

    #[test]
    fn test_exists_matcher() {
        let mut matchers = HashMap::new();
        matchers.insert("badge".to_string(), ContextMatcher::Exists { exists: true });
        let cond = SecretRevealCondition::Context(matchers);

        let ctx = make_context(vec![("badge", json!("gold"))]);
        assert!(evaluate_reveal_condition(&cond, &ctx));

        let empty_ctx = HashMap::new();
        assert!(!evaluate_reveal_condition(&cond, &empty_ctx));
    }

    #[test]
    fn test_condition_yaml_parse_and_evaluate() {
        let yaml = r#"
all:
  - context:
      trust:
        gte: 0.8
  - context:
      is_member:
        eq: true
"#;
        let cond: SecretRevealCondition = serde_yaml::from_str(yaml).unwrap();

        // Both conditions pass.
        let ctx = make_context(vec![("trust", json!(0.9)), ("is_member", json!(true))]);
        assert!(evaluate_reveal_condition(&cond, &ctx));

        // One condition fails.
        let ctx2 = make_context(vec![("trust", json!(0.9)), ("is_member", json!(false))]);
        assert!(!evaluate_reveal_condition(&cond, &ctx2));

        // Verify JSON roundtrip works (used by snapshot persistence).
        let json_val = serde_json::to_value(&cond).unwrap();
        let from_json: SecretRevealCondition = serde_json::from_value(json_val).unwrap();
        assert!(evaluate_reveal_condition(&from_json, &ctx));
        assert!(!evaluate_reveal_condition(&from_json, &ctx2));
    }
}
