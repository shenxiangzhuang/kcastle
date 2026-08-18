# kcastle-agent

Native Rust agent harness used by [Kcastle](https://github.com/shenxiangzhuang/kcastle).

It provides an OpenAI Responses run loop, append-only state and JSONL sessions, context compaction,
cancellation-safe tool execution, and a local shell capability. Applications can construct the
default session-backed agent or inject their own `StateCommit` and `AgentTool` implementations with
`Agent::from_parts`.
