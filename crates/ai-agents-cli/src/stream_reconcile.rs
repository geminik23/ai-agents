use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinalContent<'a> {
    Unchanged,
    Suffix(&'a str),
    Authoritative(&'a str),
}

pub(crate) fn reconcile_final_content<'a>(
    streamed: &str,
    final_content: &'a str,
) -> FinalContent<'a> {
    if final_content == streamed {
        FinalContent::Unchanged
    } else if let Some(suffix) = final_content.strip_prefix(streamed) {
        FinalContent::Suffix(suffix)
    } else {
        FinalContent::Authoritative(final_content)
    }
}

pub(crate) fn unique_tool_names(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}

pub(crate) fn unseen_unique_tool_names(
    names: impl IntoIterator<Item = String>,
    observed: &[String],
) -> Vec<String> {
    let mut seen: HashSet<String> = observed.iter().cloned().collect();
    names
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_final_content_is_unchanged() {
        assert_eq!(
            reconcile_final_content("hello", "hello"),
            FinalContent::Unchanged
        );
    }

    #[test]
    fn extending_final_content_returns_only_suffix() {
        assert_eq!(
            reconcile_final_content("hello", "hello world"),
            FinalContent::Suffix(" world")
        );
    }

    #[test]
    fn divergent_final_content_is_authoritative() {
        assert_eq!(
            reconcile_final_content("draft", "final answer"),
            FinalContent::Authoritative("final answer")
        );
    }

    #[test]
    fn empty_stream_treats_final_content_as_suffix() {
        assert_eq!(
            reconcile_final_content("", "complete answer"),
            FinalContent::Suffix("complete answer")
        );
    }

    #[test]
    fn unique_tool_names_preserve_first_seen_order() {
        assert_eq!(
            unique_tool_names(vec![
                "search".to_string(),
                "read".to_string(),
                "search".to_string(),
            ]),
            vec!["search".to_string(), "read".to_string()]
        );
    }

    #[test]
    fn unseen_tool_names_omit_chunk_observations_and_duplicates() {
        assert_eq!(
            unseen_unique_tool_names(
                vec![
                    "search".to_string(),
                    "read".to_string(),
                    "read".to_string(),
                    "write".to_string(),
                ],
                &["search".to_string(), "write".to_string()],
            ),
            vec!["read".to_string()]
        );
    }
}
