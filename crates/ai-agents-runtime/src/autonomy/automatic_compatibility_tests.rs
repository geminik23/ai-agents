//! Hosted ordinary fallback preserves the existing distinction between legacy error chunks and authoritative event responses.

use super::*;

// Merely installing an inactive scoped declaration cannot erase an ordinary approval rejection or duplicate committed stream content.
#[tokio::test]
async fn auto_ordinary_legacy_rejection_keeps_error_while_event_api_keeps_final() {
    let yaml = r#"
name: HostedOrdinaryStream
system_prompt: Use the requested tool.
tools: [echo]
streaming: {enabled: true}
hitl:
  tools:
    echo:
      require_approval: true
      approval_message: Approve echo?
states:
  initial: ordinary
  states:
    ordinary: {}
    scoped:
      autonomy: {enabled: true, mode: state_until_exit, completion: {response_contains: done}}
"#;
    for legacy in [false, true] {
        let mut mock = ai_agents_llm::mock::MockLLMProvider::new("isolated-rejection");
        mock.add_response(ai_agents_core::LLMResponse::new(
            r#"{"tool":"echo","arguments":{"message":"hello"}}"#,
            ai_agents_core::FinishReason::Stop,
        ));
        let runtime = AgentBuilder::from_yaml(yaml)
            .unwrap()
            .llm(Arc::new(mock))
            .auto_configure_features()
            .unwrap()
            .build()
            .unwrap();
        if legacy {
            let mut stream = runtime.chat_stream("echo hello").await.unwrap();
            let mut errors = 0;
            let mut done = 0;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    crate::StreamChunk::Error { message } => {
                        assert!(message.contains("Operation cancelled"));
                        errors += 1;
                    }
                    crate::StreamChunk::Done {} => done += 1,
                    _ => {}
                }
            }
            assert_eq!(errors, 1);
            assert_eq!(done, 1);
        } else {
            let mut stream = runtime.chat_stream_events("echo hello").await.unwrap();
            let mut finals = 0;
            while let Some(event) = stream.next().await {
                match event {
                    AgentStreamEvent::Final(response) => {
                        assert!(response.content.contains("Operation cancelled"));
                        finals += 1;
                    }
                    AgentStreamEvent::Chunk(crate::StreamChunk::Error { message }) => {
                        panic!("ordinary rejection should finalize the event API: {message}")
                    }
                    AgentStreamEvent::Task(_) => {
                        panic!("inactive state unexpectedly started task work")
                    }
                    _ => {}
                }
            }
            assert_eq!(finals, 1);
        }
        assert!(runtime.automatic_task_result().is_none());
    }
}
