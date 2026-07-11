# Replay Evals

This directory is reserved for replay-oriented eval notes and shared cassette policy.

Record/replay is useful for local iteration over live-provider behavior without repeated provider calls:

```sh
cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search_record \
  --record target/eval/cassettes/tools_code_search.jsonl

cargo run -p ai-agents-cli -- eval \
  --scenarios examples/eval/live/examples/tools/code_search_live.yaml \
  --output target/eval/live/examples/tools/code_search_replay \
  --replay target/eval/cassettes/tools_code_search.jsonl
```

Treat cassettes as test artifacts. They may contain user-visible provider responses, so do not commit unreviewed or private cassette data.
