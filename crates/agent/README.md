# kcastle-agent

Native Rust agent harness used by [Kcastle](https://github.com/shenxiangzhuang/kcastle).

It provides an OpenAI Responses run loop, append-only state and JSONL sessions, context compaction,
cancellation-safe tool execution, and a local shell capability. Applications can construct the
default session-backed agent or inject their own `StateCommit` and `AgentTool` implementations with
`Agent::from_parts`.

Sessions use the versioned v1 JSONL event format. The first record is a strict format header;
pre-v1 logs are rejected and are not migrated. Every following event has a contiguous sequence,
wall time, process-local monotonic time, optional source-event links, and an optional append/replace
surface operation. Loading and writing both validate turn/step nesting, tool and compaction
lifecycles, and source references before rebuilding `State` from the event stream. Writers advance
an incremental validator for every event and batch adjacent streaming chunks for up to 16 ms or
64 KiB; structural events flush the batch before they are acknowledged. This avoids both replaying
the full log and issuing a synchronous write for every token.

Session catalogs cache each file's validated header and search projection by file size and modified
time. Unchanged logs are not reparsed on sidebar or search refreshes; when a file changes, its full
event lifecycle is validated again before it can re-enter the catalog.

Tool calls record four independent milestones: call observed, execution dispatched, execution
finished, and result committed. This preserves real parallel completion order while committing
tool results in model order. Assistant streaming records typed text, reasoning, and tool-argument
deltas so consumers can derive lifecycle duration, TTFT, generation time, and throughput without
using UI clocks. Assistant duration and TTFT start at the durable step boundary, matching DSH even
when automatic compaction happens before the provider request.
