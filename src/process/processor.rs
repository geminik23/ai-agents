//! Process processor for executing input/output transformations

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::error::{AgentError, Result};
use crate::llm::{ChatMessage, LLMRegistry};
use crate::process::config::*;

#[derive(Debug, Clone, Default)]
pub struct ProcessData {
    pub content: String,
    pub original: String,
    pub context: HashMap<String, serde_json::Value>,
    pub metadata: ProcessMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct ProcessMetadata {
    pub stages_executed: Vec<String>,
    pub timing: HashMap<String, u64>,
    pub warnings: Vec<String>,
    pub rejected: bool,
    pub rejection_reason: Option<String>,
}

impl ProcessData {
    pub fn new(content: impl Into<String>) -> Self {
        let content = content.into();
        Self {
            original: content.clone(),
            content,
            context: HashMap::new(),
            metadata: ProcessMetadata::default(),
        }
    }

    pub fn with_context(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.context.insert(key.into(), value);
        self
    }
}

#[derive(Debug)]
pub struct ProcessProcessor {
    config: ProcessConfig,
    llm_registry: Option<Arc<LLMRegistry>>,
}

impl Default for ProcessProcessor {
    fn default() -> Self {
        Self::new(ProcessConfig::default())
    }
}

impl ProcessProcessor {
    pub fn new(config: ProcessConfig) -> Self {
        Self {
            config,
            llm_registry: None,
        }
    }

    pub fn with_llm_registry(mut self, registry: Arc<LLMRegistry>) -> Self {
        self.llm_registry = Some(registry);
        self
    }

    pub async fn process_input(&self, input: &str) -> Result<ProcessData> {
        let mut data = ProcessData::new(input);

        for stage in &self.config.input {
            data = self.execute_stage(stage, data).await?;
            if data.metadata.rejected {
                break;
            }
        }

        Ok(data)
    }

    pub async fn process_output(
        &self,
        output: &str,
        input_context: &HashMap<String, serde_json::Value>,
    ) -> Result<ProcessData> {
        let mut data = ProcessData::new(output);
        data.context = input_context.clone();

        for stage in &self.config.output {
            data = self.execute_stage(stage, data).await?;
            if data.metadata.rejected {
                break;
            }
        }

        Ok(data)
    }

    fn execute_stage<'a>(
        &'a self,
        stage: &'a ProcessStage,
        data: ProcessData,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ProcessData>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();
            let stage_name = stage
                .id()
                .map(String::from)
                .unwrap_or_else(|| self.get_stage_type_name(stage));

            // Check condition before executing stage
            if let Some(condition) = stage.condition() {
                if !self.evaluate_condition_expr(condition, &data) {
                    if self.config.settings.debug.log_stages {
                        tracing::debug!(
                            "[Process] Stage skipped (condition not met): {}",
                            stage_name
                        );
                    }
                    return Ok(data);
                }
            }

            if self.config.settings.debug.log_stages {
                tracing::debug!("[Process] Executing stage: {}", stage_name);
            }

            let data_clone = data.clone();
            let result = match stage {
                ProcessStage::Normalize(s) => self.execute_normalize(&s.config, data).await,
                ProcessStage::Detect(s) => self.execute_detect(&s.config, data).await,
                ProcessStage::Extract(s) => self.execute_extract(&s.config, data).await,
                ProcessStage::Sanitize(s) => self.execute_sanitize(&s.config, data).await,
                ProcessStage::Transform(s) => self.execute_transform(&s.config, data).await,
                ProcessStage::Validate(s) => self.execute_validate(&s.config, data).await,
                ProcessStage::Format(s) => self.execute_format(&s.config, data).await,
                ProcessStage::Enrich(s) => self.execute_enrich(&s.config, data).await,
                ProcessStage::Conditional(s) => self.execute_conditional(&s.config, data).await,
            };

            match result {
                Ok(mut d) => {
                    d.metadata.stages_executed.push(stage_name.clone());
                    if self.config.settings.debug.include_timing {
                        d.metadata
                            .timing
                            .insert(stage_name, start.elapsed().as_millis() as u64);
                    }
                    Ok(d)
                }
                Err(e) => {
                    let mut fallback_data = data_clone;
                    match self.config.settings.on_stage_error.default {
                        StageErrorAction::Stop => Err(e),
                        StageErrorAction::Continue => {
                            fallback_data
                                .metadata
                                .warnings
                                .push(format!("Stage {} failed: {}", stage_name, e));
                            Ok(fallback_data)
                        }
                        StageErrorAction::Retry => {
                            if let Some(retry_config) = &self.config.settings.on_stage_error.retry {
                                for _ in 0..retry_config.max_retries {
                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        retry_config.backoff_ms,
                                    ))
                                    .await;
                                }
                            }
                            fallback_data
                                .metadata
                                .warnings
                                .push(format!("Stage {} failed after retries: {}", stage_name, e));
                            Ok(fallback_data)
                        }
                    }
                }
            }
        })
    }

    fn get_stage_type_name(&self, stage: &ProcessStage) -> String {
        match stage {
            ProcessStage::Normalize(_) => "normalize".to_string(),
            ProcessStage::Detect(_) => "detect".to_string(),
            ProcessStage::Extract(_) => "extract".to_string(),
            ProcessStage::Sanitize(_) => "sanitize".to_string(),
            ProcessStage::Transform(_) => "transform".to_string(),
            ProcessStage::Validate(_) => "validate".to_string(),
            ProcessStage::Format(_) => "format".to_string(),
            ProcessStage::Enrich(_) => "enrich".to_string(),
            ProcessStage::Conditional(_) => "conditional".to_string(),
        }
    }

    async fn execute_normalize(
        &self,
        config: &NormalizeConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        let mut content = data.content.clone();

        if config.trim {
            content = content.trim().to_string();
        }

        if config.collapse_whitespace {
            content = content.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        if config.lowercase {
            content = content.to_lowercase();
        }

        // Unicode normalization would require unicode-normalization crate
        // For now, we skip it as it's optional

        data.content = content;
        Ok(data)
    }

    async fn execute_detect(
        &self,
        config: &DetectConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        let llm = self.get_llm(config.llm.as_deref())?;

        let detection_types: Vec<&str> = config
            .detect
            .iter()
            .map(|d| match d {
                DetectionType::Language => "language (ISO 639-1 code)",
                DetectionType::Sentiment => "sentiment (positive, negative, neutral)",
                DetectionType::Intent => "intent",
                DetectionType::Topic => "topic",
                DetectionType::Formality => "formality (formal, informal)",
                DetectionType::Urgency => "urgency (low, medium, high, critical)",
            })
            .collect();

        let intents_desc = if !config.intents.is_empty() {
            let intents: Vec<String> = config
                .intents
                .iter()
                .map(|i| format!("- {}: {}", i.id, i.description))
                .collect();
            format!("\n\nAvailable intents:\n{}", intents.join("\n"))
        } else {
            String::new()
        };

        let prompt = format!(
            "Analyze the following text and detect: {}\n{}\n\n\
             Respond with JSON only: {{\"language\": \"...\", \"sentiment\": \"...\", \"intent\": \"...\", ...}}\n\n\
             Text: {}",
            detection_types.join(", "),
            intents_desc,
            data.content
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm
            .complete(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        if let Ok(result) =
            serde_json::from_str::<serde_json::Value>(&extract_json(&response.content))
        {
            for (key, context_path) in &config.store_in_context {
                if let Some(value) = result.get(key) {
                    data.context.insert(context_path.clone(), value.clone());
                }
            }
            data.context.insert("detection".to_string(), result);
        }

        Ok(data)
    }

    async fn execute_extract(
        &self,
        config: &ExtractConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        let llm = self.get_llm(config.llm.as_deref())?;

        let schema_desc: Vec<String> = config
            .schema
            .iter()
            .map(|(name, schema)| {
                let type_str = format!("{:?}", schema.field_type).to_lowercase();
                let desc = schema.description.as_deref().unwrap_or("");
                let values = if !schema.values.is_empty() {
                    format!(" (values: {})", schema.values.join(", "))
                } else {
                    String::new()
                };
                let required = if schema.required { " [required]" } else { "" };
                format!("- {}: {} - {}{}{}", name, type_str, desc, values, required)
            })
            .collect();

        let prompt = format!(
            "Extract the following fields from the text:\n{}\n\n\
             Respond with JSON only. Use null for fields not found.\n\n\
             Text: {}",
            schema_desc.join("\n"),
            data.content
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm
            .complete(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        if let Ok(result) =
            serde_json::from_str::<serde_json::Value>(&extract_json(&response.content))
        {
            if let Some(context_path) = &config.store_in_context {
                data.context.insert(context_path.clone(), result.clone());
            }
            data.context.insert("extracted".to_string(), result);
        }

        Ok(data)
    }

    async fn execute_sanitize(
        &self,
        config: &SanitizeConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        let llm = self.get_llm(config.llm.as_deref())?;

        let mut instructions = Vec::new();

        if let Some(pii_config) = &config.pii {
            if !pii_config.types.is_empty() {
                let pii_types: Vec<String> = pii_config
                    .types
                    .iter()
                    .map(|t| format!("{:?}", t).to_lowercase())
                    .collect();
                let action = match pii_config.action {
                    PIIAction::Mask => format!("replace with '{}'", pii_config.mask_char.repeat(4)),
                    PIIAction::Remove => "remove completely".to_string(),
                    PIIAction::Flag => "wrap with [PII: type]".to_string(),
                };
                instructions.push(format!("PII types to {}: {}", action, pii_types.join(", ")));
            }
        }

        if let Some(harmful_config) = &config.harmful {
            if !harmful_config.detect.is_empty() {
                let types: Vec<String> = harmful_config
                    .detect
                    .iter()
                    .map(|t| format!("{:?}", t).to_lowercase())
                    .collect();
                instructions.push(format!("Detect harmful content: {}", types.join(", ")));
            }
        }

        if !config.remove.is_empty() {
            instructions.push(format!(
                "Remove any mentions of: {}",
                config.remove.join(", ")
            ));
        }

        if instructions.is_empty() {
            return Ok(data);
        }

        let prompt = format!(
            "Sanitize the following text according to these rules:\n{}\n\n\
             Return only the sanitized text, nothing else.\n\n\
             Text: {}",
            instructions.join("\n"),
            data.content
        );

        let messages = vec![ChatMessage::user(&prompt)];
        let response = llm
            .complete(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        data.content = response.content.trim().to_string();
        Ok(data)
    }

    async fn execute_transform(
        &self,
        config: &TransformConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        let prompt = match &config.prompt {
            Some(p) => p.clone(),
            None => return Ok(data),
        };

        let llm = self.get_llm(config.llm.as_deref())?;

        let full_prompt = format!("{}\n\nOriginal text:\n{}", prompt, data.content);

        let messages = vec![ChatMessage::user(&full_prompt)];
        let response = llm
            .complete(&messages, None)
            .await
            .map_err(|e| AgentError::LLM(e.to_string()))?;

        data.content = response.content.trim().to_string();
        Ok(data)
    }

    async fn execute_validate(
        &self,
        config: &ValidateConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        // Rule-based validation
        for rule in &config.rules {
            match rule {
                ValidationRule::MinLength {
                    min_length,
                    on_fail,
                } => {
                    if data.content.len() < *min_length {
                        match on_fail.action {
                            ValidationActionType::Reject => {
                                data.metadata.rejected = true;
                                data.metadata.rejection_reason = Some(format!(
                                    "Content too short: {} < {} characters",
                                    data.content.len(),
                                    min_length
                                ));
                                return Ok(data);
                            }
                            ValidationActionType::Warn => {
                                data.metadata.warnings.push(format!(
                                    "Content shorter than {} characters",
                                    min_length
                                ));
                            }
                            ValidationActionType::Truncate => {} // N/A for min_length
                        }
                    }
                }
                ValidationRule::MaxLength {
                    max_length,
                    on_fail,
                } => {
                    if data.content.len() > *max_length {
                        match on_fail.action {
                            ValidationActionType::Truncate => {
                                data.content = data.content.chars().take(*max_length).collect();
                            }
                            ValidationActionType::Reject => {
                                data.metadata.rejected = true;
                                data.metadata.rejection_reason = Some(format!(
                                    "Content too long: {} > {} characters",
                                    data.content.len(),
                                    max_length
                                ));
                                return Ok(data);
                            }
                            ValidationActionType::Warn => {
                                data.metadata
                                    .warnings
                                    .push(format!("Content longer than {} characters", max_length));
                            }
                        }
                    }
                }
                ValidationRule::Pattern { pattern, on_fail } => {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if !re.is_match(&data.content) {
                            match on_fail.action {
                                ValidationActionType::Reject => {
                                    data.metadata.rejected = true;
                                    data.metadata.rejection_reason =
                                        Some("Content does not match required pattern".to_string());
                                    return Ok(data);
                                }
                                ValidationActionType::Warn => {
                                    data.metadata.warnings.push(
                                        "Content does not match expected pattern".to_string(),
                                    );
                                }
                                ValidationActionType::Truncate => {} // N/A for pattern
                            }
                        }
                    }
                }
            }
        }

        // LLM-based validation
        if !config.criteria.is_empty() {
            let llm = self.get_llm(config.llm.as_deref())?;

            let criteria_list = config
                .criteria
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}. {}", i + 1, c))
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                "Evaluate if the following content meets these criteria:\n{}\n\n\
                 Respond with JSON: {{\"passes\": true/false, \"score\": 0.0-1.0, \"issues\": [\"...\"]}}\n\n\
                 Content: {}",
                criteria_list, data.content
            );

            let messages = vec![ChatMessage::user(&prompt)];
            let response = llm
                .complete(&messages, None)
                .await
                .map_err(|e| AgentError::LLM(e.to_string()))?;

            if let Ok(result) =
                serde_json::from_str::<serde_json::Value>(&extract_json(&response.content))
            {
                let score = result.get("score").and_then(|s| s.as_f64()).unwrap_or(1.0) as f32;
                let passes = result
                    .get("passes")
                    .and_then(|p| p.as_bool())
                    .unwrap_or(true);

                if !passes || score < config.threshold {
                    match config.on_fail.action {
                        ValidationFailType::Reject => {
                            data.metadata.rejected = true;
                            let issues = result
                                .get("issues")
                                .and_then(|i| i.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                })
                                .unwrap_or_else(|| "Validation failed".to_string());
                            data.metadata.rejection_reason = Some(issues);
                            return Ok(data);
                        }
                        ValidationFailType::Regenerate => {
                            data.metadata
                                .warnings
                                .push("Content may need regeneration".to_string());
                        }
                        ValidationFailType::Warn => {
                            if let Some(issues) = result.get("issues").and_then(|i| i.as_array()) {
                                for issue in issues {
                                    if let Some(s) = issue.as_str() {
                                        data.metadata.warnings.push(s.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(data)
    }

    async fn execute_format(
        &self,
        config: &FormatConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        let template = if let Some(channel) = &config.channel {
            config
                .channels
                .get(channel)
                .and_then(|c| c.template.as_ref())
                .or(config.template.as_ref())
        } else {
            config.template.as_ref()
        };

        if let Some(tmpl) = template {
            // Simple template substitution
            let mut result = tmpl.clone();
            result = result.replace("{{ response }}", &data.content);
            result = result.replace("{{response}}", &data.content);

            // Replace context variables
            for (key, value) in &data.context {
                let placeholder = format!("{{{{ context.{} }}}}", key);
                let placeholder_no_space = format!("{{{{context.{}}}}}", key);
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    _ => value.to_string(),
                };
                result = result.replace(&placeholder, &value_str);
                result = result.replace(&placeholder_no_space, &value_str);
            }

            data.content = result;
        }

        // Apply channel-specific max_length
        if let Some(channel) = &config.channel {
            if let Some(channel_config) = config.channels.get(channel) {
                if let Some(max_len) = channel_config.max_length {
                    if data.content.len() > max_len {
                        data.content = data.content.chars().take(max_len).collect();
                    }
                }
            }
        }

        Ok(data)
    }

    async fn execute_enrich(
        &self,
        config: &EnrichConfig,
        mut data: ProcessData,
    ) -> Result<ProcessData> {
        let result = match &config.source {
            EnrichSource::None => return Ok(data),
            EnrichSource::Api {
                url,
                method: _,
                headers: _,
                body: _,
                extract: _,
            } => {
                // API enrichment would require HTTP client
                // For now, add a warning
                data.metadata
                    .warnings
                    .push(format!("API enrichment not yet implemented: {}", url));
                return Ok(data);
            }
            EnrichSource::File { path, format } => {
                // File enrichment
                match std::fs::read_to_string(path) {
                    Ok(content) => match format.as_deref() {
                        Some("json") => serde_json::from_str(&content).ok(),
                        Some("yaml") => serde_yaml::from_str(&content).ok(),
                        _ => Some(serde_json::Value::String(content)),
                    },
                    Err(e) => match config.on_error {
                        EnrichErrorAction::Stop => return Err(AgentError::IoError(e)),
                        EnrichErrorAction::Continue | EnrichErrorAction::Warn => {
                            data.metadata
                                .warnings
                                .push(format!("File read failed: {}", e));
                            return Ok(data);
                        }
                    },
                }
            }
            EnrichSource::Tool { tool, args: _ } => {
                // Tool execution would need tool registry access
                data.metadata
                    .warnings
                    .push(format!("Tool enrichment not yet implemented: {}", tool));
                return Ok(data);
            }
        };

        if let Some(value) = result {
            if let Some(context_path) = &config.store_in_context {
                data.context.insert(context_path.clone(), value);
            }
        }

        Ok(data)
    }

    async fn execute_conditional(
        &self,
        config: &ConditionalConfig,
        data: ProcessData,
    ) -> Result<ProcessData> {
        let condition_met = self.evaluate_condition(&config.condition, &data);

        let stages = if condition_met {
            &config.then_stages
        } else {
            &config.else_stages
        };

        let mut result = data;
        for stage in stages {
            result = self.execute_stage(stage, result).await?;
            if result.metadata.rejected {
                break;
            }
        }

        Ok(result)
    }

    fn evaluate_condition(&self, condition: &Option<ConditionExpr>, data: &ProcessData) -> bool {
        match condition {
            None => true,
            Some(expr) => self.evaluate_condition_expr(expr, data),
        }
    }

    fn evaluate_condition_expr(&self, condition: &ConditionExpr, data: &ProcessData) -> bool {
        match condition {
            ConditionExpr::All { all } => all.iter().all(|c| self.evaluate_condition_expr(c, data)),
            ConditionExpr::Any { any } => any.iter().any(|c| self.evaluate_condition_expr(c, data)),
            ConditionExpr::Simple(map) => self.evaluate_simple_condition(map, data),
        }
    }

    fn evaluate_simple_condition(
        &self,
        map: &std::collections::HashMap<String, serde_json::Value>,
        data: &ProcessData,
    ) -> bool {
        for (path, expected) in map {
            let actual = self.get_nested_value(&data.context, path);

            // Handle { exists: true/false }
            if let Some(obj) = expected.as_object() {
                if let Some(exists_val) = obj.get("exists") {
                    let should_exist = exists_val.as_bool().unwrap_or(true);
                    let does_exist =
                        actual.is_some() && !matches!(actual, Some(serde_json::Value::Null));
                    if does_exist != should_exist {
                        return false;
                    }
                    continue;
                }
            }

            // Direct value comparison
            match (actual, expected) {
                (Some(a), e) if a == e => continue,
                (None, serde_json::Value::Null) => continue,
                _ => return false,
            }
        }
        true
    }

    fn get_nested_value<'a>(
        &self,
        context: &'a std::collections::HashMap<String, serde_json::Value>,
        path: &str,
    ) -> Option<&'a serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return None;
        }

        let mut current: Option<&serde_json::Value> = context.get(parts[0]);

        for part in &parts[1..] {
            current = current.and_then(|v| {
                if let serde_json::Value::Object(obj) = v {
                    obj.get(*part)
                } else {
                    None
                }
            });
        }

        current
    }

    fn get_llm(&self, alias: Option<&str>) -> Result<Arc<dyn crate::llm::LLMProvider>> {
        let registry = self
            .llm_registry
            .as_ref()
            .ok_or_else(|| AgentError::Config("LLM registry not configured for process".into()))?;

        match alias {
            Some(name) => registry
                .get(name)
                .map_err(|e| AgentError::LLM(e.to_string())),
            None => registry
                .router()
                .or_else(|_| registry.default())
                .map_err(|e| AgentError::LLM(e.to_string())),
        }
    }
}

fn extract_json(response: &str) -> String {
    let trimmed = response.trim();

    if trimmed.starts_with("```json") {
        if let Some(end) = trimmed[7..].find("```") {
            return trimmed[7..7 + end].trim().to_string();
        }
    }

    if trimmed.starts_with("```") {
        if let Some(end) = trimmed[3..].find("```") {
            return trimmed[3..3 + end].trim().to_string();
        }
    }

    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return trimmed[start..=end].to_string();
        }
    }

    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_data_new() {
        let data = ProcessData::new("test content");
        assert_eq!(data.content, "test content");
        assert_eq!(data.original, "test content");
        assert!(data.context.is_empty());
    }

    #[test]
    fn test_process_data_with_context() {
        let data = ProcessData::new("test").with_context("key", serde_json::json!("value"));
        assert!(data.context.contains_key("key"));
    }

    #[tokio::test]
    async fn test_normalize_trim() {
        let processor = ProcessProcessor::default();
        let config = NormalizeConfig {
            trim: true,
            ..Default::default()
        };
        let data = ProcessData::new("  hello world  ");
        let result = processor.execute_normalize(&config, data).await.unwrap();
        assert_eq!(result.content, "hello world");
    }

    #[tokio::test]
    async fn test_normalize_collapse_whitespace() {
        let processor = ProcessProcessor::default();
        let config = NormalizeConfig {
            trim: true,
            collapse_whitespace: true,
            ..Default::default()
        };
        let data = ProcessData::new("hello    world\n\ntest");
        let result = processor.execute_normalize(&config, data).await.unwrap();
        assert_eq!(result.content, "hello world test");
    }

    #[tokio::test]
    async fn test_normalize_lowercase() {
        let processor = ProcessProcessor::default();
        let config = NormalizeConfig {
            lowercase: true,
            ..Default::default()
        };
        let data = ProcessData::new("Hello World");
        let result = processor.execute_normalize(&config, data).await.unwrap();
        assert_eq!(result.content, "hello world");
    }

    #[tokio::test]
    async fn test_validate_min_length_reject() {
        let processor = ProcessProcessor::default();
        let config = ValidateConfig {
            rules: vec![ValidationRule::MinLength {
                min_length: 10,
                on_fail: ValidationAction {
                    action: ValidationActionType::Reject,
                    message: None,
                },
            }],
            ..Default::default()
        };
        let data = ProcessData::new("short");
        let result = processor.execute_validate(&config, data).await.unwrap();
        assert!(result.metadata.rejected);
    }

    #[tokio::test]
    async fn test_validate_max_length_truncate() {
        let processor = ProcessProcessor::default();
        let config = ValidateConfig {
            rules: vec![ValidationRule::MaxLength {
                max_length: 5,
                on_fail: ValidationAction {
                    action: ValidationActionType::Truncate,
                    message: None,
                },
            }],
            ..Default::default()
        };
        let data = ProcessData::new("hello world");
        let result = processor.execute_validate(&config, data).await.unwrap();
        assert_eq!(result.content, "hello");
        assert!(!result.metadata.rejected);
    }

    #[tokio::test]
    async fn test_format_simple_template() {
        let processor = ProcessProcessor::default();
        let config = FormatConfig {
            template: Some("Response: {{ response }}".to_string()),
            ..Default::default()
        };
        let data = ProcessData::new("Hello!");
        let result = processor.execute_format(&config, data).await.unwrap();
        assert_eq!(result.content, "Response: Hello!");
    }

    #[test]
    fn test_extract_json() {
        assert_eq!(extract_json(r#"{"key": 1}"#), r#"{"key": 1}"#);
        assert_eq!(extract_json("```json\n{\"key\": 1}\n```"), r#"{"key": 1}"#);
        assert_eq!(extract_json("Some text {\"key\": 1} more"), r#"{"key": 1}"#);
    }

    #[test]
    fn test_evaluate_condition_empty() {
        let processor = ProcessProcessor::default();
        let data = ProcessData::new("test");
        assert!(processor.evaluate_condition(&None, &data));
    }

    #[test]
    fn test_evaluate_condition_simple_exists_true() {
        let processor = ProcessProcessor::default();
        let mut data = ProcessData::new("test");
        data.context.insert(
            "session".to_string(),
            serde_json::json!({ "user_name": "Alice" }),
        );

        let mut map = std::collections::HashMap::new();
        map.insert(
            "session.user_name".to_string(),
            serde_json::json!({ "exists": true }),
        );
        let condition = ConditionExpr::Simple(map);

        assert!(processor.evaluate_condition_expr(&condition, &data));
    }

    #[test]
    fn test_evaluate_condition_simple_exists_false() {
        let processor = ProcessProcessor::default();
        let data = ProcessData::new("test");

        let mut map = std::collections::HashMap::new();
        map.insert(
            "session.user_name".to_string(),
            serde_json::json!({ "exists": false }),
        );
        let condition = ConditionExpr::Simple(map);

        assert!(processor.evaluate_condition_expr(&condition, &data));
    }

    #[test]
    fn test_evaluate_condition_all() {
        let processor = ProcessProcessor::default();
        let mut data = ProcessData::new("test");
        data.context.insert(
            "session".to_string(),
            serde_json::json!({ "user_name": "Alice", "language": "en" }),
        );

        let mut map1 = std::collections::HashMap::new();
        map1.insert(
            "session.user_name".to_string(),
            serde_json::json!({ "exists": true }),
        );
        let mut map2 = std::collections::HashMap::new();
        map2.insert(
            "session.language".to_string(),
            serde_json::json!({ "exists": true }),
        );

        let condition = ConditionExpr::All {
            all: vec![ConditionExpr::Simple(map1), ConditionExpr::Simple(map2)],
        };

        assert!(processor.evaluate_condition_expr(&condition, &data));
    }

    #[test]
    fn test_evaluate_condition_any() {
        let processor = ProcessProcessor::default();
        let mut data = ProcessData::new("test");
        data.context.insert(
            "session".to_string(),
            serde_json::json!({ "tier": "premium" }),
        );

        let mut map1 = std::collections::HashMap::new();
        map1.insert("session.tier".to_string(), serde_json::json!("premium"));
        let mut map2 = std::collections::HashMap::new();
        map2.insert("session.tier".to_string(), serde_json::json!("enterprise"));

        let condition = ConditionExpr::Any {
            any: vec![ConditionExpr::Simple(map1), ConditionExpr::Simple(map2)],
        };

        assert!(processor.evaluate_condition_expr(&condition, &data));
    }

    #[test]
    fn test_evaluate_condition_value_match() {
        let processor = ProcessProcessor::default();
        let mut data = ProcessData::new("test");
        data.context.insert(
            "input".to_string(),
            serde_json::json!({ "sentiment": "negative" }),
        );

        let mut map = std::collections::HashMap::new();
        map.insert("input.sentiment".to_string(), serde_json::json!("negative"));
        let condition = ConditionExpr::Simple(map);

        assert!(processor.evaluate_condition_expr(&condition, &data));
    }

    #[test]
    fn test_get_nested_value() {
        let processor = ProcessProcessor::default();
        let mut context = std::collections::HashMap::new();
        context.insert(
            "session".to_string(),
            serde_json::json!({ "user": { "name": "Alice" } }),
        );

        let result = processor.get_nested_value(&context, "session.user.name");
        assert_eq!(result, Some(&serde_json::json!("Alice")));

        let result = processor.get_nested_value(&context, "session.nonexistent");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_stage_skipped_when_condition_false() {
        let config = ProcessConfig {
            input: vec![ProcessStage::Extract(ExtractStage {
                id: Some("skip_me".to_string()),
                condition: Some(ConditionExpr::Simple({
                    let mut map = std::collections::HashMap::new();
                    map.insert(
                        "session.user".to_string(),
                        serde_json::json!({ "exists": false }),
                    );
                    map
                })),
                config: ExtractConfig::default(),
            })],
            settings: ProcessSettings {
                debug: ProcessDebugConfig {
                    log_stages: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let processor = ProcessProcessor::new(config);

        let mut data = ProcessData::new("test");
        data.context.insert(
            "session".to_string(),
            serde_json::json!({ "user": "Alice" }),
        );

        let result = processor.process_input("test").await.unwrap();
        assert!(
            !result
                .metadata
                .stages_executed
                .contains(&"skip_me".to_string())
        );
    }
}
