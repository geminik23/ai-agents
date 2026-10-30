//! Shared structural key preflight for agent and external-skill YAML.

/// Reports merge and non-string keys before framework or adapter-specific decoding.
pub fn unsupported_yaml_keys(value: &serde_yaml::Value) -> Vec<String> {
    fn collect(value: &serde_yaml::Value, path: &str, unsupported: &mut Vec<String>) {
        match value {
            serde_yaml::Value::Mapping(mapping) => {
                for (key, child) in mapping {
                    let Some(key) = key.as_str() else {
                        unsupported.push(if path.is_empty() {
                            "<non-string-key>".into()
                        } else {
                            format!("{path}.<non-string-key>")
                        });
                        continue;
                    };
                    let child_path = if path.is_empty() {
                        key.to_string()
                    } else {
                        format!("{path}.{key}")
                    };
                    if key == "<<" {
                        unsupported.push(child_path);
                    } else {
                        collect(child, &child_path, unsupported);
                    }
                }
            }
            serde_yaml::Value::Sequence(values) => {
                for (index, child) in values.iter().enumerate() {
                    collect(child, &format!("{path}[{index}]"), unsupported);
                }
            }
            _ => {}
        }
    }
    let mut unsupported = Vec::new();
    collect(value, "", &mut unsupported);
    unsupported
}
