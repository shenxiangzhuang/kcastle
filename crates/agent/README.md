# kcastle-agent

Native Rust agent harness used by [Kcastle](https://github.com/shenxiangzhuang/kcastle).

It provides an OpenAI Responses run loop, transactional append-only sessions, context compaction,
cancellation-safe tool execution, and a local shell capability.

Session v2 uses one project-local SQLite WAL database as its only runtime source of truth. JSONL is
an explicit export format. Version 1 JSONL files are intentionally ignored and are not migrated or
dual-written. Each logical transition is a revision-checked, idempotent transaction. The
`SessionMachine` validates the complete candidate batch without mutation; only a matching SQLite
commit receipt may advance the live machine or be published to a UI. Connection ambiguity is
resolved by transaction ID rather than by blindly retrying.

Inputs, request snapshots, tool dispatch intent, execution milestones, results, compaction, and
terminal recovery all have explicit typed correlation IDs. Model and tool effects start only after
their intent commits. A tool interrupted after dispatch is recorded with unknown side effects and
is never retried automatically. Streaming observations are committed in short batches, while every
structural boundary remains atomic.

Tool calls record call observation, authorization, durable dispatch intent, observed execution
start and completion, and result attachment independently. This preserves real parallel completion
order while attaching tool results in model order. Assistant streaming records typed text,
reasoning, and tool-call deltas so consumers derive duration, TTFT, generation time, and throughput
without UI clocks. Assistant duration and TTFT start at `ModelRequestStarted`, so automatic
compaction after the step boundary is not double-counted as model time.
