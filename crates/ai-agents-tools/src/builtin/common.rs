use chrono::{DateTime, Utc};
use regex::RegexBuilder;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use ai_agents_core::{ToolOperationKind, ToolResult, ToolSafetyMetadata, ToolSideEffectLevel};

pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 20_000;
pub const DEFAULT_MAX
