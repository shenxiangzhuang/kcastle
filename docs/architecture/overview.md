# Workspace architecture

Status: accepted

Kcastle is a Rust workspace with an agent core and one desktop application:

| Package | Crate | Responsibility |
| --- | --- | --- |
| `kcastle-agent` | `kcastle_agent` library | UI-independent agent runtime, transactional `SessionMachine`, SQLite `SessionStore`, canonical session facts, compaction, `Env`, and tools |
| `kcastle-desktop` | desktop binary | GPUI rendering, desktop input, approvals, session orchestration, and desktop dependency composition |

Dependencies flow only from `kcastle-desktop` to `kcastle-agent`; the agent core must never import
desktop infrastructure. The core owns harness runtime and persistence semantics. Product
configuration, provider catalogs, session presentation, search formatting, and title policy belong
to the desktop crate.

The core uses Tokio, the async-openai Responses API, Serde, and rusqlite/SQLite WAL. The desktop
uses GPUI. Use `async-openai` directly rather than adding a provider abstraction.

See [Session v2](session-v2.md) for session transactions and replay semantics, and
[Desktop app storage](app-storage.md) for product-level persistence.
