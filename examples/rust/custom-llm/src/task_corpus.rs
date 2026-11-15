// A tiny synthetic corpus shows how an application supplies verifiable research evidence.
// Flow: catalog -> actual reads -> structured facts -> managed semantic comparison -> completion.
// Replace these documents and the domain predicate for your application; keep execution in the shared runtime.
use ai_agents::autonomy::*;
use ai_agents::tools::{Tool, ToolExecutionContext, ToolResult};
use ai_agents::{AgentError, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

pub const VERSION: &str = "synthetic-alerts-v1";

/// Immutable, invented service facts; these are not current product claims or public network sources.
pub fn documents() -> Value {
    json!([
        {"id":"alerts-a","title":"Aster alerts","monthly_usd":20,"events_per_minute":100,"p95_ms":800,"body":"Aster costs 20 USD/month for 100 events/minute. Measured p95 delivery is 800 ms in a synthetic single-region test. Email only; no regional failover. Real outage delivery was not measured."},
        {"id":"alerts-b","title":"Birch alerts","monthly_usd":40,"events_per_minute":400,"p95_ms":250,"body":"Birch costs 40 USD/month for 400 events/minute. Measured p95 delivery is 250 ms in a synthetic two-region test. Email and webhook delivery; failover is manual. Real outage delivery was not measured."},
        {"id":"alerts-c","title":"Cedar alerts","monthly_usd":70,"events_per_minute":1000,"p95_ms":120,"body":"Cedar costs 70 USD/month for 1000 events/minute. Measured p95 delivery is 120 ms in a synthetic two-region test. Email, webhook and SMS; SMS has an unspecified extra fee. Real outage delivery was not measured."}
    ])
}

#[derive(Default)]
pub struct Corpus {
    read: Mutex<BTreeMap<String, String>>,
    revision: AtomicU64,
}
impl HostRevisionSource for Corpus {
    fn revision(&self) -> Result<String> {
        Ok(self.revision.load(Ordering::SeqCst).to_string())
    }
}
pub struct Catalog(pub Arc<Corpus>);
#[async_trait]
impl Tool for Catalog {
    fn id(&self) -> &str {
        "research_catalog"
    }
    fn name(&self) -> &str {
        "Synthetic alert service catalog"
    }
    fn description(&self) -> &str {
        "List synthetic source IDs or read one complete source. Only successful reads count as research evidence."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","additionalProperties":false,"properties":{"op":{"type":"string","enum":["list","read"]},"id":{"type":"string"}},"required":["op"]})
    }
    async fn execute(&self, args: Value, ctx: ToolExecutionContext) -> ToolResult {
        if args
            .as_object()
            .is_none_or(|a| a.keys().any(|k| k != "op" && k != "id"))
        {
            return ToolResult::error("unknown catalog argument");
        }
        let docs = documents();
        match args["op"].as_str() {
            Some("list") if args.get("id").is_none() => ToolResult::ok(json!({"version":VERSION,"sources":docs.as_array().unwrap().iter().map(|d|json!({"id":d["id"],"title":d["title"]})).collect::<Vec<_>>()} ).to_string()),
            Some("read") => {
                let Some(doc) = docs.as_array().unwrap().iter().find(|d|d["id"].is_string() && d["id"] == args["id"]) else { return ToolResult::error("unknown synthetic source ID"); };
                let id=doc["id"].as_str().unwrap();
                let mut receipts=self.0.read.lock().unwrap();
                if let Some(previous)=receipts.get(&ctx.call_id) {
                    if previous!=id {return ToolResult::error("catalog call identity conflict");}
                } else {
                    if receipts.len()>=4096 {return ToolResult::error("catalog receipt capacity reached; start a fresh host");}
                    receipts.insert(ctx.call_id,id.to_owned());
                    self.0.revision.fetch_add(1,Ordering::SeqCst);
                }
                ToolResult::ok(json!({"version":VERSION,"source":doc}).to_string())
            }
            _ => ToolResult::error("expected list or read with a registered ID"),
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusConfig {
    llm: String,
    pass_threshold: f64,
    question: String,
}

pub struct CorpusValidator {
    descriptor: AdapterDescriptor,
    corpus: Arc<Corpus>,
}
impl CorpusValidator {
    pub fn new(corpus: Arc<Corpus>) -> Self {
        Self {
            descriptor: AdapterDescriptor {
                id: "host.corpus".into(),
                contract_version: 1,
                needs_judge: true,
                requires_host_revision: true,
                ..Default::default()
            },
            corpus,
        }
    }
}
fn decision(outcome: GateOutcome, unique: usize, score: f64, reason: &str) -> ValidationDecision {
    ValidationDecision::Complete {
        outcome,
        reason: reason.into(),
        metrics: json!({"unique":unique,"score":score}),
        evidence_refs: vec![],
    }
}

/// Checks only structural facts and actual eligible reads; natural-language reasoning belongs to the managed judge.
fn factual_answer(answer: &Value, read: &BTreeSet<String>) -> bool {
    let Some(rows) = answer["sources"].as_array() else {
        return false;
    };
    let mut ids = BTreeSet::new();
    let docs = documents();
    for row in rows {
        let Some(id) = row["id"].as_str() else {
            return false;
        };
        if !ids.insert(id.to_string()) || !read.contains(id) {
            return false;
        }
        let Some(doc) = docs.as_array().unwrap().iter().find(|d| d["id"] == id) else {
            return false;
        };
        if ["monthly_usd", "events_per_minute", "p95_ms"]
            .iter()
            .any(|k| row[k] != doc[k])
        {
            return false;
        }
    }
    ids.len() >= 2
        && &ids == read
        && ["comparison", "uncertainty"]
            .iter()
            .all(|k| answer[k].as_str().is_some_and(|s| !s.trim().is_empty()))
}
impl AutonomyValidator for CorpusValidator {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }
    fn validate_config(&self, config: &Value) -> Result<()> {
        let config: CorpusConfig = serde_json::from_value(config.clone()).map_err(|_| {
            AgentError::Config("host.corpus requires llm and pass_threshold".into())
        })?;
        if config.question.trim().is_empty()
            || config.llm.is_empty()
            || !(0.0..=1.0).contains(&config.pass_threshold)
        {
            return Err(AgentError::Config("invalid corpus judge settings".into()));
        }
        Ok(())
    }
    // A model-supplied source name cannot establish a read or replace the immutable document sent to the judge.
    fn evaluate(&self, input: &ValidationInput<'_>) -> Result<ValidationDecision> {
        let receipts = self.corpus.read.lock().unwrap().clone();
        let Some(current) = input
            .evidence
            .current
            .as_ref()
            .filter(|c| c.complete && c.identity.eligible(input.scope, true, true))
        else {
            return Ok(decision(
                GateOutcome::Unknown,
                0,
                0.0,
                "current answer unavailable",
            ));
        };
        let answer = serde_json::from_str::<Value>(&current.value.response).unwrap_or(Value::Null);
        let observed: BTreeSet<String> = input
            .evidence
            .tools
            .iter()
            .filter_map(|entry| {
                if !entry.complete || !entry.identity.eligible(input.scope, false, false) {
                    return None;
                }
                let r = &entry.value;
                if !tool_record_eligible(r)
                    || !r.success
                    || r.output_truncated
                    || r.canonical_id != "research_catalog"
                    || r.executed_arguments["op"] != "read"
                {
                    return None;
                }
                let value: Value = serde_json::from_str(&r.output).ok()?;
                let id = value["source"]["id"].as_str()?;
                (value["version"] == VERSION
                    && r.executed_arguments["id"] == id
                    && receipts.get(&r.call_id).is_some_and(|stored| stored == id))
                .then(|| id.to_owned())
            })
            .collect();
        // Only receipts matched to this run and objective can contribute; earlier REPL tasks never supply credit.
        let read = observed;
        if !factual_answer(&answer, &read) {
            return Ok(decision(
                GateOutcome::Fail,
                read.len(),
                0.0,
                "return sources as distinct read IDs with exact numeric monthly_usd/events_per_minute/p95_ms; comparison and uncertainty must be nonempty JSON strings",
            ));
        }
        let config: CorpusConfig = serde_json::from_value(input.config.clone())
            .map_err(|e| AgentError::Config(e.to_string()))?;
        if let Some(observation) = input.observations.get("comparison") {
            let ObservationResult::Judge { score } = observation.result else {
                return Ok(decision(
                    GateOutcome::Unknown,
                    read.len(),
                    0.0,
                    "judge unavailable",
                ));
            };
            return Ok(decision(
                if score >= config.pass_threshold {
                    GateOutcome::Pass
                } else {
                    GateOutcome::Fail
                },
                read.len(),
                score,
                "grounded comparison",
            ));
        }
        let docs: Vec<_> = documents()
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| read.contains(d["id"].as_str().unwrap()))
            .cloned()
            .collect();
        Ok(ValidationDecision::NeedObservations { requests:vec![ValidationObservationRequest::Judge {
            id:"comparison".into(),
            config:serde_json::from_value(json!({"llm":config.llm,"pass_threshold":config.pass_threshold,"criteria":["The answer compares the supplied source facts and relevant trade-offs, addresses the requested budget/capacity/latency constraints, and acknowledges that synthetic measurements do not establish real outage reliability."]})).map_err(|e|AgentError::Config(e.to_string()))?,
            response:json!({"candidate":answer,"retrieved_sources":docs,"version":VERSION}).to_string(),
            objective:config.question,
        }], checkpoint:Value::Null })
    }
}

pub fn register(extensions: &mut AutonomyExtensions) -> Result<Arc<dyn Tool>> {
    let corpus = Arc::new(Corpus::default());
    extensions.register_validator(Arc::new(CorpusValidator::new(corpus.clone())))?;
    extensions.register_validation_target("corpus", "research-corpus", corpus.clone())?;
    Ok(Arc::new(Catalog(corpus)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn catalog_rejects_unknown_ids_and_counts_distinct_actual_reads() {
        let state = Arc::new(Corpus::default());
        let tool = Catalog(state.clone());
        let call = |args| tool.execute(args, ToolExecutionContext::test("research_catalog"));
        assert!(!call(json!({"op":"read","id":"fabricated"})).await.success);
        assert!(
            !call(json!({"op":"read","id":"alerts-a","path":"/tmp"}))
                .await
                .success
        );
        assert!(call(json!({"op":"list"})).await.success);
        assert_eq!(state.revision().unwrap(), "0");
        for _ in 0..2 {
            assert!(call(json!({"op":"read","id":"alerts-a"})).await.success);
        }
        assert_eq!(state.revision().unwrap(), "1");
        assert_eq!(state.read.lock().unwrap().len(), 1);
        assert!(Corpus::default().read.lock().unwrap().is_empty());
    }
    #[test]
    fn structural_answer_rejects_duplicates_unknowns_and_wrong_facts() {
        let mut answer = json!({"sources":[{"id":"alerts-a","monthly_usd":20,"events_per_minute":100,"p95_ms":800},{"id":"alerts-b","monthly_usd":40,"events_per_minute":400,"p95_ms":250}],"comparison":"trade-offs","uncertainty":"synthetic measurements"});
        let read = BTreeSet::from(["alerts-a".into(), "alerts-b".into()]);
        assert!(factual_answer(&answer, &read));
        let mut wrong_type = answer.clone();
        wrong_type["comparison"] = json!(["a comparison"]);
        assert!(!factual_answer(&wrong_type, &read));
        wrong_type = answer.clone();
        wrong_type["uncertainty"] = json!({"note":"synthetic"});
        assert!(!factual_answer(&wrong_type, &read));
        assert!(!factual_answer(
            &answer,
            &BTreeSet::from(["alerts-a".into()])
        ));
        answer["sources"][1]["monthly_usd"] = json!(0);
        assert!(!factual_answer(&answer, &read));
        answer["sources"][1] = answer["sources"][0].clone();
        assert!(!factual_answer(&answer, &read));
    }
}
