# kcastle-agent

Native Rust agent harness used by [K in Castle](https://github.com/shenxiangzhuang/kcastle).

It provides a concrete OpenAI Responses run loop, append-only state and JSONL sessions, context
compaction, cancellation-safe tool execution, and a local shell capability.
