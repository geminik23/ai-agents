use chrono::{Datelike, Local, Timelike, Utc};
use serde_json::{Value, json};

use super::source::BuiltinSource;

pub fn get_builtin_value(source: &BuiltinSource, agent_name: &str, agent_version: &str) -> Value {
    match source {
        BuiltinSource::Datetime => get_datetime_context(),
        BuiltinSource::Session => get_session_context(),
        BuiltinSource::Agent => get_agent_context(agent_name, agent_version),
    }
}

fn get_datetime_context() -> Value {
    let utc = Utc::now();
    let local = Local::now();

    json!({
        "utc": utc.to_rfc3339(),
        "local": local.to_rfc3339(),
        "date": local.format("%Y-%m-%d").to_string(),
        "time": local.format("%H:%M:%S").to_string(),
        "year": local.year(),
        "month": local.month(),
        "day": local.day(),
        "hour": local.hour(),
        "minute": local.minute(),
        "day_of_week": local.weekday().to_string(),
        "timestamp": utc.timestamp(),
    })
}

fn get_session_context() -> Value {
    json!({
        "id": uuid::Uuid::new_v4().to_string(),
        "started_at": Utc::now().to_rfc3339(),
    })
}

fn get_agent_context(name: &str, version: &str) -> Value {
    json!({
        "name": name,
        "version": version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datetime_context() {
        let ctx = get_builtin_value(&BuiltinSource::Datetime, "", "");
        assert!(ctx.get("utc").is_some());
        assert!(ctx.get("local").is_some());
        assert!(ctx.get("date").is_some());
        assert!(ctx.get("time").is_some());
        assert!(ctx.get("year").is_some());
        assert!(ctx.get("day_of_week").is_some());
    }

    #[test]
    fn test_session_context() {
        let ctx = get_builtin_value(&BuiltinSource::Session, "", "");
        assert!(ctx.get("id").is_some());
        assert!(ctx.get("started_at").is_some());
    }

    #[test]
    fn test_agent_context() {
        let ctx = get_builtin_value(&BuiltinSource::Agent, "TestAgent", "1.0.0");
        assert_eq!(ctx.get("name").unwrap(), "TestAgent");
        assert_eq!(ctx.get("version").unwrap(), "1.0.0");
    }
}
