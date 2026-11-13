use crate::*;
use ai_agents_core::autonomy::ToolWriteFootprint;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[test]
fn rs_footprint_all_builtin_ids_have_explicit_read_preview_or_host_contracts() {
    let mut seen = Vec::new();
    for tool in builtin::all_builtin_tools() {
        let args = match tool.id() {
            "file" => json!({"operation":"info","path":"."}),
            "file_write" => json!({"path":"fixture.txt","content":"x","dry_run":true}),
            "file_edit" => {
                json!({"path":"fixture.txt","old_text":"a","new_text":"b","dry_run":true})
            }
            "patch" => json!({"patch":"fixture","dry_run":true}),
            "copy_path" | "move_path" => {
                json!({"source_path":"source","destination_path":"dest","dry_run":true})
            }
            "delete_path" => json!({"path":"fixture","dry_run":true}),
            "command" => json!({"argv":["fixture"]}),
            "web_search" => json!({"query":"fixture"}),
            "ask_user" => json!({"question":"fixture?"}),
            "http" => json!({"method":"GET","url":"https://example.invalid/"}),
            _ => json!({}),
        };
        let footprint = tool
            .declared_write_footprint(&args, &ToolExecutionContext::test(tool.id()), 0)
            .unwrap()
            .unwrap_or_else(|| panic!("missing {}", tool.id()));
        assert!(footprint.targets.is_empty(), "{}", tool.id());
        seen.push(tool.id().to_string());
    }
    seen.sort();
    let mut expected = builtin::BUILTIN_TOOL_IDS.map(str::to_string).to_vec();
    expected.sort();
    assert_eq!(seen, expected);
    let http = HttpTool::new();
    assert!(
        http.declared_write_footprint(
            &json!({"method":"POST","url":"https://example.invalid/"}),
            &ToolExecutionContext::test("http"),
            5
        )
        .unwrap()
        .is_none()
    );
    assert!(
        ProcessCommandRunner
            .declared_write_footprint(
                &CommandRequest::default(),
                &ToolExecutionContext::test("command"),
                5
            )
            .unwrap()
            .is_none()
    );
}

async fn check_swap(tool: &dyn Tool, args: Value, replace: impl FnOnce()) {
    let mut ctx = ToolExecutionContext::test(tool.id());
    let prepared = tool
        .prepare_write_footprint(&args, &ctx, 0)
        .unwrap()
        .unwrap();
    let guard = prepared.binding.unwrap();
    guard.validate().unwrap();
    ctx.admitted_write_footprint = Some(prepared.footprint);
    assert!(ctx.admitted_write_footprint.is_some());
    assert!(tool.execute(args.clone(), ctx.clone()).await.success);
    replace();
    assert!(guard.validate().is_err());
    let denied = tool.execute(args.clone(), ctx).await;
    assert!(!denied.success);
    assert!(
        denied
            .output
            .contains("host footprint or actual implementation changed")
    );
    assert!(
        tool.execute(args, ToolExecutionContext::test(tool.id()))
            .await
            .success,
        "ordinary calls retain existing host selection"
    );
}

struct Question(Arc<AtomicUsize>);
#[async_trait]
impl QuestionHandler for Question {
    fn declared_write_footprint(
        &self,
        _: &QuestionRequest,
        _: &ToolExecutionContext,
        _: usize,
    ) -> ai_agents_core::Result<Option<ToolWriteFootprint>> {
        Ok(Some(ToolWriteFootprint::empty("fixture.question")))
    }
    async fn ask_question(&self, _: QuestionRequest) -> QuestionResponse {
        self.0.fetch_add(1, Ordering::SeqCst);
        QuestionResponse {
            answered: true,
            selected: vec!["yes".into()],
            ..Default::default()
        }
    }
}

#[tokio::test]
async fn rs_footprint_host_slot_swaps_cannot_reuse_identical_capability_labels() {
    let slots: CommandRunnerSlot = Arc::new(RwLock::new(Arc::new(StaticCommandRunner::new(
        HashMap::from([(
            vec!["fixture".into()],
            CommandResponse {
                success: true,
                termination: "exited".into(),
                ..Default::default()
            },
        )]),
    ))));
    let tool = CommandTool::new(slots.clone());
    check_swap(&tool, json!({"argv":["fixture"]}), || {
        *slots.write() = Arc::new(StaticCommandRunner::new(HashMap::from([(
            vec!["fixture".into()],
            CommandResponse {
                success: true,
                ..Default::default()
            },
        )])))
    })
    .await;

    let slots: DiagnosticsProviderSlot = Arc::new(RwLock::new(Arc::new(
        StaticDiagnosticsProvider::new(vec![]),
    )));
    let tool = DiagnosticsTool::new(slots.clone());
    check_swap(&tool, json!({}), || {
        *slots.write() = Arc::new(StaticDiagnosticsProvider::new(vec![]))
    })
    .await;

    let slots: WebSearchProviderSlot = Arc::new(RwLock::new(Arc::new(
        StaticWebSearchProvider::new(HashMap::new()),
    )));
    let tool = WebSearchTool::with_provider_slot(slots.clone());
    check_swap(&tool, json!({"query":"fixture"}), || {
        *slots.write() = Arc::new(StaticWebSearchProvider::new(HashMap::new()))
    })
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let slots: QuestionHandlerSlot = Arc::new(RwLock::new(Some(Arc::new(Question(calls.clone())))));
    let tool = AskUserTool::new(slots.clone());
    check_swap(&tool, json!({"question":"fixture?"}), || {
        *slots.write() = Some(Arc::new(Question(calls.clone())))
    })
    .await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "rejected host never executes"
    );
}

#[test]
fn rs_footprint_guard_retains_actual_arc_until_admission_retires() {
    let slots: CommandRunnerSlot = Arc::new(RwLock::new(Arc::new(StaticCommandRunner::default())));
    let old = Arc::downgrade(&slots.read());
    let guard = CommandTool::new(slots.clone())
        .prepare_write_footprint(
            &json!({"argv":["fixture"]}),
            &ToolExecutionContext::test("command"),
            0,
        )
        .unwrap()
        .unwrap()
        .binding
        .unwrap();
    *slots.write() = Arc::new(StaticCommandRunner::default());
    assert!(
        old.upgrade().is_some(),
        "allocation cannot be reused during admitted waits"
    );
    assert!(guard.validate().is_err());
    drop(guard);
    assert!(old.upgrade().is_none());
}

struct Unbounded;
#[async_trait]
impl DiagnosticsProvider for Unbounded {
    async fn diagnostics(&self, _: DiagnosticsRequest) -> DiagnosticsResponse {
        panic!("unbounded host must not run");
    }
}

#[test]
fn rs_footprint_readonly_metadata_never_certifies_an_opaque_host() {
    let slot: DiagnosticsProviderSlot = Arc::new(RwLock::new(Arc::new(Unbounded)));
    let tool = DiagnosticsTool::new(slot);
    assert!(tool.safety_metadata().read_only);
    assert!(
        tool.declared_write_footprint(&json!({}), &ToolExecutionContext::test("diagnostics"), 1)
            .unwrap()
            .is_none()
    );
}
