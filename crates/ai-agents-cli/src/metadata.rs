use ai_agents::spec::{CliHitlMetadata, CliMetadata, CliPromptStyle};
use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct ResolvedCliMetadata {
    pub welcome: Option<String>,
    pub hints: Vec<String>,
    pub show_tools: Option<bool>,
    pub show_state: Option<bool>,
    pub show_timing: Option<bool>,
    pub streaming: Option<bool>,
    pub prompt_style: Option<CliPromptStyle>,
    pub disable_builtin_commands: Option<bool>,
    pub hitl: Option<CliHitlMetadata>,
    pub theme: Option<String>,
}

impl ResolvedCliMetadata {
    pub fn from_metadata_value(metadata: Option<&Value>) -> Self {
        let Some(root) = metadata else {
            return Self::default();
        };

        let Some(cli_value) = root.get("cli") else {
            return Self::default();
        };

        serde_json::from_value::<CliMetadata>(cli_value.clone())
            .map(Self::from)
            .unwrap_or_default()
    }

    pub fn merge_overrides(self, overrides: CliOverrides) -> Self {
        Self {
            welcome: overrides.welcome.or(self.welcome),
            hints: overrides.hints.unwrap_or(self.hints),
            show_tools: overrides.show_tools.or(self.show_tools),
            show_state: overrides.show_state.or(self.show_state),
            show_timing: overrides.show_timing.or(self.show_timing),
            streaming: overrides.streaming.or(self.streaming),
            prompt_style: overrides.prompt_style.or(self.prompt_style),
            disable_builtin_commands: overrides
                .disable_builtin_commands
                .or(self.disable_builtin_commands),
            hitl: overrides.hitl.or(self.hitl),
            theme: overrides.theme.or(self.theme),
        }
    }
}

impl From<CliMetadata> for ResolvedCliMetadata {
    fn from(value: CliMetadata) -> Self {
        Self {
            welcome: value.welcome,
            hints: value.hints,
            show_tools: value.show_tools,
            show_state: value.show_state,
            show_timing: value.show_timing,
            streaming: value.streaming,
            prompt_style: value.prompt_style,
            disable_builtin_commands: value.disable_builtin_commands,
            hitl: value.hitl,
            theme: value.theme,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    pub welcome: Option<String>,
    pub hints: Option<Vec<String>>,
    pub show_tools: Option<bool>,
    pub show_state: Option<bool>,
    pub show_timing: Option<bool>,
    pub streaming: Option<bool>,
    pub prompt_style: Option<CliPromptStyle>,
    pub disable_builtin_commands: Option<bool>,
    pub hitl: Option<CliHitlMetadata>,
    pub theme: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn returns_default_when_metadata_missing() {
        let metadata = ResolvedCliMetadata::from_metadata_value(None);
        assert!(metadata.welcome.is_none());
        assert!(metadata.hints.is_empty());
        assert!(metadata.show_tools.is_none());
    }

    #[test]
    fn returns_default_when_cli_section_missing() {
        let root = json!({
            "template": "demo"
        });

        let metadata = ResolvedCliMetadata::from_metadata_value(Some(&root));
        assert!(metadata.welcome.is_none());
        assert!(metadata.hints.is_empty());
    }

    #[test]
    fn extracts_cli_metadata_from_json_value() {
        let root = json!({
            "template": "demo",
            "cli": {
                "welcome": "=== Demo ===",
                "hints": ["Try: hello", "Try: help"],
                "show_tools": true,
                "show_state": false,
                "show_timing": true,
                "streaming": true,
                "prompt_style": "with_state",
                "disable_builtin_commands": false
            }
        });

        let metadata = ResolvedCliMetadata::from_metadata_value(Some(&root));
        assert_eq!(metadata.welcome.as_deref(), Some("=== Demo ==="));
        assert_eq!(metadata.hints, vec!["Try: hello", "Try: help"]);
        assert_eq!(metadata.show_tools, Some(true));
        assert_eq!(metadata.show_state, Some(false));
        assert_eq!(metadata.show_timing, Some(true));
        assert_eq!(metadata.streaming, Some(true));
        assert_eq!(metadata.prompt_style, Some(CliPromptStyle::WithState));
        assert_eq!(metadata.disable_builtin_commands, Some(false));
    }

    #[test]
    fn ignores_invalid_cli_metadata_shape() {
        let root = json!({
            "cli": "not-an-object"
        });

        let metadata = ResolvedCliMetadata::from_metadata_value(Some(&root));
        assert!(metadata.welcome.is_none());
        assert!(metadata.hints.is_empty());
    }

    #[test]
    fn merge_overrides_prefers_explicit_override_values() {
        let base = ResolvedCliMetadata {
            welcome: Some("Base".into()),
            hints: vec!["base".into()],
            show_tools: Some(false),
            show_state: Some(false),
            show_timing: Some(false),
            streaming: Some(false),
            prompt_style: Some(CliPromptStyle::Simple),
            disable_builtin_commands: Some(false),
            hitl: None,
            theme: None,
        };

        let overrides = CliOverrides {
            welcome: Some("Override".into()),
            hints: Some(vec!["override-1".into(), "override-2".into()]),
            show_tools: Some(true),
            show_state: Some(true),
            show_timing: Some(true),
            streaming: Some(true),
            prompt_style: Some(CliPromptStyle::WithState),
            disable_builtin_commands: Some(true),
            hitl: None,
            theme: None,
        };

        let merged = base.merge_overrides(overrides);
        assert_eq!(merged.welcome.as_deref(), Some("Override"));
        assert_eq!(merged.hints, vec!["override-1", "override-2"]);
        assert_eq!(merged.show_tools, Some(true));
        assert_eq!(merged.show_state, Some(true));
        assert_eq!(merged.show_timing, Some(true));
        assert_eq!(merged.streaming, Some(true));
        assert_eq!(merged.prompt_style, Some(CliPromptStyle::WithState));
        assert_eq!(merged.disable_builtin_commands, Some(true));
    }

    #[test]
    fn merge_overrides_keeps_base_values_when_override_missing() {
        let base = ResolvedCliMetadata {
            welcome: Some("Base".into()),
            hints: vec!["base".into()],
            show_tools: Some(true),
            show_state: Some(true),
            show_timing: Some(false),
            streaming: Some(false),
            prompt_style: Some(CliPromptStyle::WithState),
            disable_builtin_commands: Some(false),
            hitl: None,
            theme: None,
        };

        let merged = base.clone().merge_overrides(CliOverrides::default());
        assert_eq!(merged.welcome, base.welcome);
        assert_eq!(merged.hints, base.hints);
        assert_eq!(merged.show_tools, base.show_tools);
        assert_eq!(merged.show_state, base.show_state);
        assert_eq!(merged.show_timing, base.show_timing);
        assert_eq!(merged.streaming, base.streaming);
        assert_eq!(merged.prompt_style, base.prompt_style);
        assert_eq!(
            merged.disable_builtin_commands,
            base.disable_builtin_commands
        );
    }
}
