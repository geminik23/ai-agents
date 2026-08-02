use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const RELEASE_VERSION: &str = "1.0.0";
const RELEASE_LICENSE: &str = "Apache-2.0";
const RELEASE_RUST_VERSION: &str = "1.88";
const INTERNAL_REQUIREMENT: &str = "^1.0.0";
const RELEASE_RECORD_SCHEMA_VERSION: u32 = 1;

const RELEASE_ORDER: [&str; 23] = [
    "ai-agents-core",
    "ai-agents-reasoning",
    "ai-agents-llm",
    "ai-agents-memory",
    "ai-agents-context",
    "ai-agents-storage",
    "ai-agents-disambiguation",
    "ai-agents-hitl",
    "ai-agents-recovery",
    "ai-agents-template",
    "ai-agents-process",
    "ai-agents-state",
    "ai-agents-tools",
    "ai-agents-relationships",
    "ai-agents-hooks",
    "ai-agents-observability",
    "ai-agents-skills",
    "ai-agents-persona",
    "ai-agents-facts",
    "ai-agents-runtime",
    "ai-agents-eval",
    "ai-agents",
    "ai-agents-cli",
];

const GENERATED_PACKAGE_FILES: [&str; 4] = [
    ".cargo_vcs_info.json",
    "Cargo.lock",
    "Cargo.toml",
    "Cargo.toml.orig",
];

/// Reports whether the release checkout passed or was rejected by a release condition.
#[derive(Debug)]
pub enum ReleasePreflightOutcome {
    Passed {
        artifact_dir: PathBuf,
    },
    Rejected {
        artifact_dir: PathBuf,
        reasons: Vec<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
    workspace_members: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoPackage {
    name: String,
    version: String,
    id: String,
    license: Option<String>,
    rust_version: Option<String>,
    publish: Option<Vec<String>>,
    dependencies: Vec<CargoDependency>,
}

#[derive(Debug, Clone, Deserialize)]
struct CargoDependency {
    name: String,
    req: String,
    kind: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ToolVersions {
    rustc_1_88_0: String,
    cargo_1_88_0: String,
    rustc_1_97_0: String,
    cargo_1_97_0: String,
}

#[derive(Debug, Serialize)]
struct ReleaseRecord {
    schema_version: u32,
    version: String,
    git_sha: String,
    cargo_lock_sha256: String,
    tool_versions: ToolVersions,
    cargo_metadata: String,
    packages: Vec<PackageRecord>,
}

#[derive(Debug, Serialize)]
struct PackageRecord {
    name: String,
    file_list: String,
}

struct IdentityInput<'a> {
    git_status: &'a str,
    lock_before: &'a str,
    lock_after: &'a str,
    tool_versions: &'a ToolVersions,
    metadata: &'a CargoMetadata,
    release_order: &'a [&'a str],
}

/// Runs a non-publishing release identity and package-content preflight.
pub fn run_release_preflight(root: &Path) -> Result<ReleasePreflightOutcome> {
    let root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve repository root '{}'", root.display()))?;
    let lock_path = root.join("Cargo.lock");
    if !lock_path.is_file() {
        return Err(anyhow!("required release input is missing: Cargo.lock"));
    }

    let git_sha = command_line(&root, "git", &["rev-parse", "HEAD"])?;
    validate_git_sha(&git_sha)?;
    let git_status = command_text(
        &root,
        "git",
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    let tool_versions = collect_tool_versions(&root)?;
    let lock_before = sha256_file(&lock_path)?;

    let artifact_dir = root.join("target/release-preflight").join(&git_sha);
    fs::create_dir_all(&artifact_dir)
        .with_context(|| format!("failed to create '{}'", artifact_dir.display()))?;
    remove_stale_release_record(&artifact_dir)?;

    let metadata_output = run_command(
        &root,
        "cargo",
        &["+1.97.0", "metadata", "--format-version", "1", "--locked"],
    )?;
    let metadata_path = artifact_dir.join("cargo-metadata.json");
    fs::write(&metadata_path, &metadata_output.stdout)
        .with_context(|| format!("failed to write '{}'", metadata_path.display()))?;
    require_success("cargo metadata", &metadata_output)?;
    let metadata: CargoMetadata = serde_json::from_slice(&metadata_output.stdout)
        .context("failed to parse cargo metadata JSON")?;
    let lock_after_metadata = sha256_file(&lock_path)?;

    let identity = IdentityInput {
        git_status: &git_status,
        lock_before: &lock_before,
        lock_after: &lock_after_metadata,
        tool_versions: &tool_versions,
        metadata: &metadata,
        release_order: &RELEASE_ORDER,
    };
    let identity_rejections = validate_identity(&identity);
    if !identity_rejections.is_empty() {
        return Ok(ReleasePreflightOutcome::Rejected {
            artifact_dir: relative_artifact_dir(&root, &artifact_dir),
            reasons: identity_rejections,
        });
    }

    let packages_dir = artifact_dir.join("packages");
    fs::create_dir_all(&packages_dir)
        .with_context(|| format!("failed to create '{}'", packages_dir.display()))?;
    let mut package_rejections = Vec::new();
    let mut package_records = Vec::with_capacity(RELEASE_ORDER.len());

    for crate_name in RELEASE_ORDER {
        let current_lock = sha256_file(&lock_path)?;
        if current_lock != lock_before {
            package_rejections.push(format!(
                "Cargo.lock changed before package inspection for '{crate_name}'"
            ));
            break;
        }

        let output = run_command(
            &root,
            "cargo",
            &["+1.97.0", "package", "--list", "--locked", "-p", crate_name],
        )?;
        let list_path = packages_dir.join(format!("{crate_name}.list"));
        fs::write(&list_path, &output.stdout)
            .with_context(|| format!("failed to write '{}'", list_path.display()))?;
        require_success(&format!("cargo package --list for {crate_name}"), &output)?;
        let list = std::str::from_utf8(&output.stdout)
            .with_context(|| format!("package list for '{crate_name}' is not UTF-8"))?;
        package_rejections.extend(validate_package_list(crate_name, list));
        package_records.push(PackageRecord {
            name: crate_name.to_string(),
            file_list: format!("packages/{crate_name}.list"),
        });
    }

    let lock_after_packages = sha256_file(&lock_path)?;
    if lock_after_packages != lock_before {
        package_rejections.push("Cargo.lock changed during package inspection".to_string());
    }
    let final_sha = command_line(&root, "git", &["rev-parse", "HEAD"])?;
    if final_sha != git_sha {
        package_rejections.push(format!(
            "Git HEAD changed during preflight from '{git_sha}' to '{final_sha}'"
        ));
    }
    let final_status = command_text(
        &root,
        "git",
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !final_status.is_empty() {
        package_rejections.push("worktree became dirty during package inspection".to_string());
    }

    if !package_rejections.is_empty() {
        return Ok(ReleasePreflightOutcome::Rejected {
            artifact_dir: relative_artifact_dir(&root, &artifact_dir),
            reasons: package_rejections,
        });
    }

    let record = ReleaseRecord {
        schema_version: RELEASE_RECORD_SCHEMA_VERSION,
        version: RELEASE_VERSION.to_string(),
        git_sha,
        cargo_lock_sha256: lock_before,
        tool_versions,
        cargo_metadata: "cargo-metadata.json".to_string(),
        packages: package_records,
    };
    write_release_record(&artifact_dir, &record)?;

    Ok(ReleasePreflightOutcome::Passed {
        artifact_dir: relative_artifact_dir(&root, &artifact_dir),
    })
}

fn collect_tool_versions(root: &Path) -> Result<ToolVersions> {
    Ok(ToolVersions {
        rustc_1_88_0: command_line(root, "rustc", &["+1.88.0", "--version"])?,
        cargo_1_88_0: command_line(root, "cargo", &["+1.88.0", "--version"])?,
        rustc_1_97_0: command_line(root, "rustc", &["+1.97.0", "--version"])?,
        cargo_1_97_0: command_line(root, "cargo", &["+1.97.0", "--version"])?,
    })
}

fn validate_identity(input: &IdentityInput<'_>) -> Vec<String> {
    let mut rejections = Vec::new();
    if !input.git_status.is_empty() {
        rejections.push(format!(
            "worktree is dirty ({} tracked or untracked status entries)",
            input.git_status.lines().count()
        ));
    }
    if input.lock_before != input.lock_after {
        rejections.push("Cargo.lock changed while collecting release identity".to_string());
    }
    validate_tool_version(
        "rustc +1.88.0",
        &input.tool_versions.rustc_1_88_0,
        "rustc",
        "1.88.0",
        &mut rejections,
    );
    validate_tool_version(
        "cargo +1.88.0",
        &input.tool_versions.cargo_1_88_0,
        "cargo",
        "1.88.0",
        &mut rejections,
    );
    validate_tool_version(
        "rustc +1.97.0",
        &input.tool_versions.rustc_1_97_0,
        "rustc",
        "1.97.0",
        &mut rejections,
    );
    validate_tool_version(
        "cargo +1.97.0",
        &input.tool_versions.cargo_1_97_0,
        "cargo",
        "1.97.0",
        &mut rejections,
    );
    validate_release_order(input.release_order, &mut rejections);
    validate_metadata(input.metadata, input.release_order, &mut rejections);
    rejections
}

fn validate_tool_version(
    command: &str,
    output: &str,
    tool: &str,
    expected_version: &str,
    rejections: &mut Vec<String>,
) {
    let mut lines = output.lines();
    let fields: Vec<_> = lines
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .collect();
    let exact = fields.first() == Some(&tool)
        && fields.get(1) == Some(&expected_version)
        && fields.len() >= 3
        && lines.next().is_none();
    if !exact {
        rejections.push(format!(
            "{command} reported '{output}', expected exact {tool} version {expected_version}"
        ));
    }
}

fn validate_release_order(release_order: &[&str], rejections: &mut Vec<String>) {
    if release_order != RELEASE_ORDER {
        rejections.push(format!(
            "release package order differs from the required order: {}",
            release_order.join(", ")
        ));
    }
}

fn validate_metadata(
    metadata: &CargoMetadata,
    release_order: &[&str],
    rejections: &mut Vec<String>,
) {
    let workspace_ids: BTreeSet<_> = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect();
    let mut workspace_packages = BTreeMap::new();
    for package in metadata
        .packages
        .iter()
        .filter(|package| workspace_ids.contains(package.id.as_str()))
    {
        if workspace_packages
            .insert(package.name.as_str(), package)
            .is_some()
        {
            rejections.push(format!(
                "workspace metadata contains duplicate package name '{}'",
                package.name
            ));
        }
    }

    let expected: BTreeSet<_> = RELEASE_ORDER.into_iter().collect();
    let publishable: BTreeSet<_> = workspace_packages
        .values()
        .filter(|package| package_is_publishable(package))
        .map(|package| package.name.as_str())
        .collect();
    let missing: Vec<_> = expected.difference(&publishable).copied().collect();
    let extra: Vec<_> = publishable.difference(&expected).copied().collect();
    if !missing.is_empty() {
        rejections.push(format!(
            "publishable package set is missing: {}",
            missing.join(", ")
        ));
    }
    if !extra.is_empty() {
        rejections.push(format!(
            "publishable package set has unexpected entries: {}",
            extra.join(", ")
        ));
    }

    match workspace_packages.get("xtask") {
        Some(package) if package.publish.as_ref().is_some_and(Vec::is_empty) => {}
        Some(_) => rejections.push("xtask must have publish = false".to_string()),
        None => rejections.push("workspace metadata is missing xtask".to_string()),
    }

    for name in RELEASE_ORDER {
        let Some(package) = workspace_packages.get(name) else {
            continue;
        };
        if package.version != RELEASE_VERSION {
            rejections.push(format!(
                "package '{name}' has version '{}', expected '{RELEASE_VERSION}'",
                package.version
            ));
        }
        if package.license.as_deref() != Some(RELEASE_LICENSE) {
            rejections.push(format!(
                "package '{name}' has license '{}', expected '{RELEASE_LICENSE}'",
                package.license.as_deref().unwrap_or("<missing>")
            ));
        }
        if package.rust_version.as_deref() != Some(RELEASE_RUST_VERSION) {
            rejections.push(format!(
                "package '{name}' has rust-version '{}', expected '{RELEASE_RUST_VERSION}'",
                package.rust_version.as_deref().unwrap_or("<missing>")
            ));
        }
        for dependency in package
            .dependencies
            .iter()
            .filter(|dependency| expected.contains(dependency.name.as_str()))
        {
            if dependency.req != INTERNAL_REQUIREMENT {
                rejections.push(format!(
                    "package '{name}' {} dependency '{}' has requirement '{}', expected '{INTERNAL_REQUIREMENT}'",
                    dependency_kind(dependency),
                    dependency.name,
                    dependency.req
                ));
            }
        }
    }

    validate_topology(&workspace_packages, release_order, &expected, rejections);
}

fn validate_topology(
    packages: &BTreeMap<&str, &CargoPackage>,
    release_order: &[&str],
    expected: &BTreeSet<&str>,
    rejections: &mut Vec<String>,
) {
    let positions: BTreeMap<_, _> = release_order
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, index))
        .collect();
    for name in release_order {
        let Some(package) = packages.get(name) else {
            continue;
        };
        let Some(package_position) = positions.get(name) else {
            continue;
        };
        for dependency in package
            .dependencies
            .iter()
            .filter(|dependency| expected.contains(dependency.name.as_str()))
        {
            let Some(dependency_position) = positions.get(dependency.name.as_str()) else {
                continue;
            };
            if dependency_position >= package_position {
                rejections.push(format!(
                    "release order is not topological: package '{name}' appears before its {} dependency '{}'",
                    dependency_kind(dependency),
                    dependency.name
                ));
            }
        }
    }
}

fn package_is_publishable(package: &CargoPackage) -> bool {
    match &package.publish {
        None => true,
        Some(registries) => !registries.is_empty(),
    }
}

fn dependency_kind(dependency: &CargoDependency) -> &str {
    dependency.kind.as_deref().unwrap_or("normal")
}

fn validate_package_list(crate_name: &str, raw: &str) -> Vec<String> {
    let files: BTreeSet<_> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(normalize_package_path)
        .collect();
    let mut rejections = Vec::new();

    if !files.iter().any(|path| {
        Path::new(path).file_name().and_then(|name| name.to_str()) == Some("LICENSE-APACHE")
    }) {
        rejections.push(format!(
            "package '{crate_name}' is missing LICENSE-APACHE material"
        ));
    }
    for expected in GENERATED_PACKAGE_FILES {
        if !files.contains(expected) {
            rejections.push(format!(
                "package '{crate_name}' is missing Cargo-generated file '{expected}'"
            ));
        }
    }
    for path in &files {
        if let Some(reason) = forbidden_package_path(path) {
            rejections.push(format!(
                "package '{crate_name}' contains forbidden file '{path}' ({reason})"
            ));
        }
    }
    rejections
}

fn normalize_package_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

fn forbidden_package_path(path: &str) -> Option<&'static str> {
    let normalized = normalize_package_path(path);
    let components: Vec<_> = normalized
        .split('/')
        .filter(|component| !component.is_empty())
        .collect();
    for component in &components {
        let lower = component.to_ascii_lowercase();
        if lower == "temp" || lower == "tmp" || lower == "target" {
            return Some("temporary or build output");
        }
        if lower == ".env" || lower.starts_with(".env.") {
            return Some("environment file");
        }
        if matches!(
            lower.as_str(),
            "eval-output"
                | "eval_output"
                | "eval-results"
                | "eval_results"
                | "eval-reports"
                | "eval_reports"
        ) {
            return Some("evaluation output");
        }
    }
    let lower = normalized.to_ascii_lowercase();
    if [".db", ".sqlite", ".sqlite3", ".mdb"]
        .iter()
        .any(|extension| lower.ends_with(extension))
    {
        return Some("database file");
    }
    None
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open '{}' for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read '{}' for hashing", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn command_line(root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let text = command_text(root, program, args)?;
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.lines().count() != 1 {
        return Err(anyhow!(
            "{} produced an invalid single-line response",
            command_display(program, args)
        ));
    }
    Ok(trimmed.to_string())
}

fn command_text(root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = run_command(root, program, args)?;
    require_success(&command_display(program, args), &output)?;
    String::from_utf8(output.stdout)
        .with_context(|| format!("{} output is not UTF-8", command_display(program, args)))
}

fn run_command(root: &Path, program: &str, args: &[&str]) -> Result<Output> {
    Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to run {}", command_display(program, args)))
}

fn require_success(description: &str, output: &Output) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(anyhow!(
        "{description} failed with status {}: {}",
        output.status,
        stderr.trim()
    ))
}

fn command_display(program: &str, args: &[&str]) -> String {
    std::iter::once(program)
        .chain(args.iter().copied())
        .collect::<Vec<_>>()
        .join(" ")
}

fn validate_git_sha(sha: &str) -> Result<()> {
    if sha.len() >= 40 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(anyhow!("git rev-parse returned invalid SHA '{sha}'"))
    }
}

fn remove_stale_release_record(artifact_dir: &Path) -> Result<()> {
    let record = artifact_dir.join("release-record.json");
    if record.exists() {
        fs::remove_file(&record)
            .with_context(|| format!("failed to remove stale '{}'", record.display()))?;
    }
    Ok(())
}

fn write_release_record(artifact_dir: &Path, record: &ReleaseRecord) -> Result<()> {
    let destination = artifact_dir.join("release-record.json");
    let temporary = artifact_dir.join(format!(".release-record.json.tmp-{}", std::process::id()));
    if temporary.exists() {
        fs::remove_file(&temporary)
            .with_context(|| format!("failed to remove stale '{}'", temporary.display()))?;
    }

    let mut json = serde_json::to_vec_pretty(record)?;
    json.push(b'\n');
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("failed to create '{}'", temporary.display()))?;
        file.write_all(&json)
            .with_context(|| format!("failed to write '{}'", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync '{}'", temporary.display()))?;
        fs::rename(&temporary, &destination).with_context(|| {
            format!(
                "failed to atomically rename '{}' to '{}'",
                temporary.display(),
                destination.display()
            )
        })?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

fn relative_artifact_dir(root: &Path, artifact_dir: &Path) -> PathBuf {
    artifact_dir
        .strip_prefix(root)
        .unwrap_or(artifact_dir)
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_tool_versions() -> ToolVersions {
        ToolVersions {
            rustc_1_88_0: "rustc 1.88.0 (test 2025-01-01)".to_string(),
            cargo_1_88_0: "cargo 1.88.0 (test 2025-01-01)".to_string(),
            rustc_1_97_0: "rustc 1.97.0 (test 2026-01-01)".to_string(),
            cargo_1_97_0: "cargo 1.97.0 (test 2026-01-01)".to_string(),
        }
    }

    fn package(name: &str, dependencies: Vec<CargoDependency>) -> CargoPackage {
        CargoPackage {
            name: name.to_string(),
            version: RELEASE_VERSION.to_string(),
            id: format!("path+file:///workspace/{name}#{RELEASE_VERSION}"),
            license: Some(RELEASE_LICENSE.to_string()),
            rust_version: Some(RELEASE_RUST_VERSION.to_string()),
            publish: None,
            dependencies,
        }
    }

    fn dependency(name: &str) -> CargoDependency {
        CargoDependency {
            name: name.to_string(),
            req: INTERNAL_REQUIREMENT.to_string(),
            kind: None,
        }
    }

    fn valid_metadata() -> CargoMetadata {
        let mut packages: Vec<_> = RELEASE_ORDER
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let dependencies = index
                    .checked_sub(1)
                    .map(|previous| vec![dependency(RELEASE_ORDER[previous])])
                    .unwrap_or_default();
                package(name, dependencies)
            })
            .collect();
        packages.push(CargoPackage {
            name: "xtask".to_string(),
            version: "0.0.0".to_string(),
            id: "path+file:///workspace/xtask#0.0.0".to_string(),
            license: None,
            rust_version: Some(RELEASE_RUST_VERSION.to_string()),
            publish: Some(Vec::new()),
            dependencies: Vec::new(),
        });
        let workspace_members = packages.iter().map(|package| package.id.clone()).collect();
        CargoMetadata {
            packages,
            workspace_members,
        }
    }

    fn identity_rejections(
        metadata: &CargoMetadata,
        status: &str,
        lock_after: &str,
        order: &[&str],
    ) -> Vec<String> {
        let versions = valid_tool_versions();
        validate_identity(&IdentityInput {
            git_status: status,
            lock_before: "same-lock",
            lock_after,
            tool_versions: &versions,
            metadata,
            release_order: order,
        })
    }

    fn valid_package_list() -> &'static str {
        ".cargo_vcs_info.json\nCargo.lock\nCargo.toml\nCargo.toml.orig\nLICENSE-APACHE\nsrc/lib.rs\n"
    }

    #[test]
    fn dirty_worktree_is_rejected() {
        let rejections = identity_rejections(
            &valid_metadata(),
            " M README.md\n?? local.txt\n",
            "same-lock",
            &RELEASE_ORDER,
        );
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("worktree is dirty"))
        );
    }

    #[test]
    fn wrong_package_version_is_rejected() {
        let mut metadata = valid_metadata();
        metadata.packages[0].version = "1.0.1".to_string();
        let rejections = identity_rejections(&metadata, "", "same-lock", &RELEASE_ORDER);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("has version '1.0.1'"))
        );
    }

    #[test]
    fn missing_package_is_rejected() {
        let mut metadata = valid_metadata();
        let removed = metadata.packages.remove(0);
        metadata.workspace_members.retain(|id| id != &removed.id);
        let rejections = identity_rejections(&metadata, "", "same-lock", &RELEASE_ORDER);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("set is missing"))
        );
    }

    #[test]
    fn extra_package_is_rejected() {
        let mut metadata = valid_metadata();
        let extra = package("ai-agents-extra", Vec::new());
        metadata.workspace_members.push(extra.id.clone());
        metadata.packages.push(extra);
        let rejections = identity_rejections(&metadata, "", "same-lock", &RELEASE_ORDER);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("unexpected entries"))
        );
    }

    #[test]
    fn reordered_release_package_is_rejected() {
        let metadata = valid_metadata();
        let mut order = RELEASE_ORDER;
        order.swap(0, 1);
        let rejections = identity_rejections(&metadata, "", "same-lock", &order);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("required order"))
        );
    }

    #[test]
    fn publishable_xtask_is_rejected() {
        let mut metadata = valid_metadata();
        metadata
            .packages
            .iter_mut()
            .find(|package| package.name == "xtask")
            .unwrap()
            .publish = None;
        let rejections = identity_rejections(&metadata, "", "same-lock", &RELEASE_ORDER);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("publish = false"))
        );
    }

    #[test]
    fn lock_drift_is_rejected() {
        let rejections = identity_rejections(&valid_metadata(), "", "changed-lock", &RELEASE_ORDER);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("Cargo.lock changed"))
        );
    }

    #[test]
    fn invalid_topological_order_is_rejected() {
        let mut metadata = valid_metadata();
        metadata.packages[0]
            .dependencies
            .push(dependency("ai-agents-cli"));
        let rejections = identity_rejections(&metadata, "", "same-lock", &RELEASE_ORDER);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("not topological"))
        );
    }

    #[test]
    fn missing_license_material_is_rejected() {
        let list = valid_package_list().replace("LICENSE-APACHE\n", "");
        let rejections = validate_package_list("ai-agents-core", &list);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("missing LICENSE-APACHE"))
        );
    }

    #[test]
    fn forbidden_package_file_is_rejected() {
        let list = format!(
            "{}target/eval-output/results.sqlite\n",
            valid_package_list()
        );
        let rejections = validate_package_list("ai-agents-core", &list);
        assert!(
            rejections
                .iter()
                .any(|reason| reason.contains("forbidden file"))
        );
    }

    #[test]
    fn synthetic_valid_identity_and_package_list_pass() {
        assert!(identity_rejections(&valid_metadata(), "", "same-lock", &RELEASE_ORDER).is_empty());
        assert!(validate_package_list("ai-agents-core", valid_package_list()).is_empty());
    }
}
