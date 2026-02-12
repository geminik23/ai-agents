use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;

use crate::generate_schema;
use ai_agents_core::{Tool, ToolResult};

pub struct FileTool;

impl FileTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FileInput {
    /// Operation: read, write, append, exists, delete, list, mkdir, info
    operation: String,
    /// File or directory path
    path: String,
    /// Content to write (for write/append)
    #[serde(default)]
    content: Option<String>,
    /// Glob pattern for list operation (e.g., '*.json')
    #[serde(default)]
    pattern: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReadOutput {
    content: String,
    path: String,
    size: usize,
}

#[derive(Debug, Serialize)]
struct WriteOutput {
    success: bool,
    path: String,
    bytes_written: usize,
}

#[derive(Debug, Serialize)]
struct ExistsOutput {
    exists: bool,
    path: String,
    is_file: bool,
    is_dir: bool,
}

#[derive(Debug, Serialize)]
struct DeleteOutput {
    success: bool,
    path: String,
}

#[derive(Debug, Serialize)]
struct ListOutput {
    entries: Vec<ListEntry>,
    path: String,
    count: usize,
}

#[derive(Debug, Serialize)]
struct ListEntry {
    name: String,
    path: String,
    is_file: bool,
    is_dir: bool,
    size: Option<u64>,
}

#[derive(Debug, Serialize)]
struct MkdirOutput {
    success: bool,
    path: String,
}

#[derive(Debug, Serialize)]
struct InfoOutput {
    path: String,
    exists: bool,
    is_file: bool,
    is_dir: bool,
    size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modified: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created: Option<String>,
}

#[async_trait]
impl Tool for FileTool {
    fn id(&self) -> &str {
        "file"
    }

    fn name(&self) -> &str {
        "File Operations"
    }

    fn description(&self) -> &str {
        "Read, write, and manage files. Operations: read (read file content), write (write content to file), append (append to file), exists (check if path exists), delete (delete file/directory), list (list directory contents), mkdir (create directory), info (get file metadata)."
    }

    fn input_schema(&self) -> Value {
        generate_schema::<FileInput>()
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let input: FileInput = match serde_json::from_value(args) {
            Ok(input) => input,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match input.operation.to_lowercase().as_str() {
            "read" => self.handle_read(&input),
            "write" => self.handle_write(&input),
            "append" => self.handle_append(&input),
            "exists" => self.handle_exists(&input),
            "delete" => self.handle_delete(&input),
            "list" => self.handle_list(&input),
            "mkdir" => self.handle_mkdir(&input),
            "info" => self.handle_info(&input),
            _ => ToolResult::error(format!(
                "Unknown operation: {}. Valid: read, write, append, exists, delete, list, mkdir, info",
                input.operation
            )),
        }
    }
}

impl FileTool {
    fn handle_read(&self, input: &FileInput) -> ToolResult {
        match fs::read_to_string(&input.path) {
            Ok(content) => {
                let output = ReadOutput {
                    size: content.len(),
                    content,
                    path: input.path.clone(),
                };
                self.to_result(&output)
            }
            Err(e) => ToolResult::error(format!("Read error: {}", e)),
        }
    }

    fn handle_write(&self, input: &FileInput) -> ToolResult {
        let content = input.content.as_deref().unwrap_or("");
        match fs::write(&input.path, content) {
            Ok(_) => {
                let output = WriteOutput {
                    success: true,
                    path: input.path.clone(),
                    bytes_written: content.len(),
                };
                self.to_result(&output)
            }
            Err(e) => ToolResult::error(format!("Write error: {}", e)),
        }
    }

    fn handle_append(&self, input: &FileInput) -> ToolResult {
        use std::fs::OpenOptions;
        use std::io::Write;

        let content = input.content.as_deref().unwrap_or("");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&input.path);

        match file {
            Ok(mut f) => match f.write_all(content.as_bytes()) {
                Ok(_) => {
                    let output = WriteOutput {
                        success: true,
                        path: input.path.clone(),
                        bytes_written: content.len(),
                    };
                    self.to_result(&output)
                }
                Err(e) => ToolResult::error(format!("Append error: {}", e)),
            },
            Err(e) => ToolResult::error(format!("File open error: {}", e)),
        }
    }

    fn handle_exists(&self, input: &FileInput) -> ToolResult {
        let path = Path::new(&input.path);
        let output = ExistsOutput {
            exists: path.exists(),
            path: input.path.clone(),
            is_file: path.is_file(),
            is_dir: path.is_dir(),
        };
        self.to_result(&output)
    }

    fn handle_delete(&self, input: &FileInput) -> ToolResult {
        let path = Path::new(&input.path);
        let result = if path.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        };

        match result {
            Ok(_) => {
                let output = DeleteOutput {
                    success: true,
                    path: input.path.clone(),
                };
                self.to_result(&output)
            }
            Err(e) => ToolResult::error(format!("Delete error: {}", e)),
        }
    }

    fn handle_list(&self, input: &FileInput) -> ToolResult {
        let path = Path::new(&input.path);
        if !path.is_dir() {
            return ToolResult::error(format!("Not a directory: {}", input.path));
        }

        let pattern = input.pattern.as_deref();

        match fs::read_dir(path) {
            Ok(entries) => {
                let mut list_entries = Vec::new();

                for entry in entries.flatten() {
                    let file_name = entry.file_name().to_string_lossy().to_string();

                    if let Some(pat) = pattern {
                        if !self.matches_pattern(&file_name, pat) {
                            continue;
                        }
                    }

                    let metadata = entry.metadata().ok();
                    let entry_path = entry.path();

                    list_entries.push(ListEntry {
                        name: file_name,
                        path: entry_path.to_string_lossy().to_string(),
                        is_file: entry_path.is_file(),
                        is_dir: entry_path.is_dir(),
                        size: metadata.map(|m| m.len()),
                    });
                }

                let output = ListOutput {
                    count: list_entries.len(),
                    entries: list_entries,
                    path: input.path.clone(),
                };
                self.to_result(&output)
            }
            Err(e) => ToolResult::error(format!("List error: {}", e)),
        }
    }

    fn handle_mkdir(&self, input: &FileInput) -> ToolResult {
        match fs::create_dir_all(&input.path) {
            Ok(_) => {
                let output = MkdirOutput {
                    success: true,
                    path: input.path.clone(),
                };
                self.to_result(&output)
            }
            Err(e) => ToolResult::error(format!("Mkdir error: {}", e)),
        }
    }

    fn handle_info(&self, input: &FileInput) -> ToolResult {
        let path = Path::new(&input.path);

        if !path.exists() {
            let output = InfoOutput {
                path: input.path.clone(),
                exists: false,
                is_file: false,
                is_dir: false,
                size: None,
                modified: None,
                created: None,
            };
            return self.to_result(&output);
        }

        let metadata = match fs::metadata(path) {
            Ok(m) => m,
            Err(e) => return ToolResult::error(format!("Metadata error: {}", e)),
        };

        let modified = metadata.modified().ok().map(|t| {
            let datetime: chrono::DateTime<chrono::Utc> = t.into();
            datetime.to_rfc3339()
        });

        let created = metadata.created().ok().map(|t| {
            let datetime: chrono::DateTime<chrono::Utc> = t.into();
            datetime.to_rfc3339()
        });

        let output = InfoOutput {
            path: input.path.clone(),
            exists: true,
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            size: Some(metadata.len()),
            modified,
            created,
        };
        self.to_result(&output)
    }

    fn matches_pattern(&self, name: &str, pattern: &str) -> bool {
        let pattern = pattern.trim();
        if pattern.is_empty() || pattern == "*" {
            return true;
        }

        if pattern.starts_with("*.") {
            let ext = &pattern[2..];
            return name.ends_with(&format!(".{}", ext));
        }

        if pattern.ends_with(".*") {
            let prefix = &pattern[..pattern.len() - 2];
            return name.starts_with(prefix);
        }

        if pattern.starts_with('*') && pattern.ends_with('*') {
            let middle = &pattern[1..pattern.len() - 1];
            return name.contains(middle);
        }

        name == pattern
    }

    fn to_result<T: Serialize>(&self, output: &T) -> ToolResult {
        match serde_json::to_string(output) {
            Ok(json) => ToolResult::ok(json),
            Err(e) => ToolResult::error(format!("Serialization error: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_write_and_read() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let path_str = file_path.to_str().unwrap();
        let tool = FileTool::new();

        let result = tool
            .execute(serde_json::json!({
                "operation": "write",
                "path": path_str,
                "content": "hello world"
            }))
            .await;
        assert!(result.success);

        let result = tool
            .execute(serde_json::json!({
                "operation": "read",
                "path": path_str
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("hello world"));
    }

    #[tokio::test]
    async fn test_append() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("append.txt");
        let path_str = file_path.to_str().unwrap();
        let tool = FileTool::new();

        tool.execute(serde_json::json!({
            "operation": "write",
            "path": path_str,
            "content": "line1\n"
        }))
        .await;

        tool.execute(serde_json::json!({
            "operation": "append",
            "path": path_str,
            "content": "line2\n"
        }))
        .await;

        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("line1"));
        assert!(content.contains("line2"));
    }

    #[tokio::test]
    async fn test_exists() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("exists.txt");
        let path_str = file_path.to_str().unwrap();
        let tool = FileTool::new();

        let result = tool
            .execute(serde_json::json!({
                "operation": "exists",
                "path": path_str
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("\"exists\":false"));

        fs::write(&file_path, "test").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "exists",
                "path": path_str
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("\"exists\":true"));
    }

    #[tokio::test]
    async fn test_delete() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("delete.txt");
        let path_str = file_path.to_str().unwrap();
        let tool = FileTool::new();

        fs::write(&file_path, "test").unwrap();
        assert!(file_path.exists());

        let result = tool
            .execute(serde_json::json!({
                "operation": "delete",
                "path": path_str
            }))
            .await;
        assert!(result.success);
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_list() {
        let dir = tempdir().unwrap();
        let tool = FileTool::new();

        fs::write(dir.path().join("a.txt"), "a").unwrap();
        fs::write(dir.path().join("b.json"), "b").unwrap();
        fs::write(dir.path().join("c.txt"), "c").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "list",
                "path": dir.path().to_str().unwrap()
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("\"count\":3"));

        let result = tool
            .execute(serde_json::json!({
                "operation": "list",
                "path": dir.path().to_str().unwrap(),
                "pattern": "*.txt"
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("\"count\":2"));
    }

    #[tokio::test]
    async fn test_mkdir() {
        let dir = tempdir().unwrap();
        let new_dir = dir.path().join("new/nested/dir");
        let tool = FileTool::new();

        let result = tool
            .execute(serde_json::json!({
                "operation": "mkdir",
                "path": new_dir.to_str().unwrap()
            }))
            .await;
        assert!(result.success);
        assert!(new_dir.exists());
    }

    #[tokio::test]
    async fn test_info() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("info.txt");
        let tool = FileTool::new();

        fs::write(&file_path, "test content").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "info",
                "path": file_path.to_str().unwrap()
            }))
            .await;
        assert!(result.success);
        assert!(result.output.contains("\"is_file\":true"));
        assert!(result.output.contains("\"size\":12"));
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = FileTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "invalid",
                "path": "/tmp/test"
            }))
            .await;
        assert!(!result.success);
    }
}
