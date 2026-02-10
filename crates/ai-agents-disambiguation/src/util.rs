/// Extract JSON object from LLM response that may contain markdown or extra text
pub(crate) fn extract_json(content: &str) -> &str {
    let trimmed = content.trim();

    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return trimmed;
    }

    // Try markdown code block with json tag
    if let Some(start) = trimmed.find("```json") {
        if let Some(end) = trimmed[start..]
            .find("```\n")
            .or_else(|| trimmed[start..].rfind("```"))
        {
            let json_start = start + 7;
            let json_end = start + end;
            if json_end > json_start {
                return trimmed[json_start..json_end].trim();
            }
        }
    }

    // Try generic code block
    if let Some(start) = trimmed.find("```") {
        let after_ticks = &trimmed[start + 3..];
        if let Some(end) = after_ticks.find("```") {
            let content = &after_ticks[..end];
            if let Some(newline) = content.find('\n') {
                return content[newline..].trim();
            }
            return content.trim();
        }
    }

    // Try bare JSON object
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return &trimmed[start..=end];
        }
    }

    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_raw() {
        let input = r#"{"is_ambiguous": true}"#;
        assert_eq!(extract_json(input), r#"{"is_ambiguous": true}"#);
    }

    #[test]
    fn test_extract_json_with_markdown() {
        let input = "```json\n{\"is_ambiguous\": true}\n```";
        assert_eq!(extract_json(input), r#"{"is_ambiguous": true}"#);
    }

    #[test]
    fn test_extract_json_with_surrounding_text() {
        let input = "Here is the result:\n{\"is_ambiguous\": false, \"confidence\": 0.9}\nThat's my analysis.";
        let result = extract_json(input);
        assert!(result.starts_with('{'));
        assert!(result.contains("is_ambiguous"));
    }

    #[test]
    fn test_extract_json_with_whitespace() {
        let input = "  \n  {\"key\": \"value\"}  \n  ";
        assert_eq!(extract_json(input), r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_generic_code_block() {
        let input = "```\n{\"key\": \"value\"}\n```";
        assert_eq!(extract_json(input), r#"{"key": "value"}"#);
    }
}
