# kcastle — Architecture

A general-purpose agent application built on `kagent`. kcastle is a long-running
daemon (local or cloud) that manages agent sessions and exposes them through
multiple channels — CLI, Telegram, and future adapters.

For quick start and overview, see the [README](../README.md).

## Core Design Principles

1. **Session-centric** — the session is the atomic unit. Everything (trace,
   metadata) lives inside a session directory.
2. **Single-writer** — a session is used by one channel at a time. Any channel
   can *resume* any session by ID, but concurrent access to the same session
   is not supported. This keeps the design simple and avoids fan-out complexity.
3. **Lossless rebuild** — sessions can be suspended and rebuilt from their
   persisted trace. The kagent `Trace` is the source of truth; the `Agent`
   is ephemeral and reconstructed on resume.
4. **kagent-native** — kcastle adds no extra event types or message wrappers.
   It uses `kagent.Agent`, `AgentEvent`, `Trace`, and `TraceStore` directly.
5. **Thematic naming** — the top-level orchestrator is `Castle` (the castle
   that houses agent "k"). Communication adapters are `Channel`s — each
   channel is a pathway to reach the agent (CLI, Telegram, etc.).

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│  kcastle                                                 │
│                                                          │
│  ┌─── Channels ────────────────────────────────────┐     │
│  │  CLIChannel           TelegramChannel           │     │
│  │  (prompt_toolkit)     (aiogram / python-tg-bot) │     │
│  └───────┬───────────────────────┬─────────────────┘     │
│          │                       │                        │
│          └───────────┬───────────┘                        │
│                      │                                    │
│          ┌───────────▼───────────┐                        │
│          │    SessionManager     │                        │
│          │  create / resume /    │                        │
│          │  list / drop          │                        │
│          └───────────┬───────────┘                        │
│                      │                                    │
│          ┌───────────▼───────────┐                        │
│          │      Session          │                        │
│          │  agent: kagent.Agent  │                        │
│          │  trace: kagent.Trace  │                        │
│          └───────────┬───────────┘                        │
│                      │                                    │
│          ┌───────────▼───────────┐                        │
│          │   Session Directory   │ (per-session folder)   │
│          │   trace.jsonl         │                        │
│          │   meta.json           │                        │
│          └───────────────────────┘                        │
│                                                          │
│          ┌───────────────────────┐                        │
│          │      Castle           │                        │
│          │  config + lifecycle   │                        │
│          └───────────────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## Data Storage Layout

Each session is a self-contained directory. No global index file — the
filesystem *is* the registry. `SessionManager` scans `sessions/` to
discover all sessions.

```
~/.kcastle/
├── config.yaml                      # Global configuration (YAML)
├── skills/                          # User-level skills
│   ├── <skill_id_1>/
│   │   ├── skill.yaml
│   │   ├── prompt.md
│   │   └── tools.py
│   └── ...
└── sessions/
    ├── <session_id_1>/
    │   ├── meta.json                # Session metadata
    │   └── trace.jsonl              # kagent Trace (append-only)
    ├── <session_id_2>/
    │   ├── meta.json
    │   └── trace.jsonl
    └── ...
```

### `meta.json` — Per-Session Metadata

```json
{
  "id": "a1b2c3d4e5f6",
  "name": "refactor api layer",
  "created_at": 1740000000000,
  "created_at_iso": "2025-02-20T00:00:00+08:00",
  "last_active_at": 1740003600000,
  "last_active_at_iso": "2025-02-20T01:00:00+08:00"
}
```

- **Timestamps**: millisecond-precision integers (`created_at`, `last_active_at`)
  plus ISO 8601 strings with local timezone (`*_iso`) for human readability.
  The integer is the source of truth; the ISO string is informational.
- **Identity**: `id` (immutable, = directory name) + `name` (mutable, user-assigned label).
- **No model or prompt fields** — the model is chosen per interaction (not
  per session), and the system prompt is assembled dynamically (see below).
  Session metadata tracks *identity and timing* only.

### `trace.jsonl` — Conversation History

Standard kagent `JsonlTraceStore` format: first line is a header, subsequent
lines are serialized `TraceEntry` objects. This is the single source of truth
for reconstructing an `Agent` via `Trace.from_records()`.

### Session Identity

Each session has a unique **session ID** that doubles as its directory name.
The ID format depends on who creates the session:

| Creator | Session ID format | Example |
|---------|------------------|---------|
| CLI (default) | UUID hex (short) | `a1b2c3d4` |
| CLI (`k -S <id>`) | user-specified | `my-project` |
| Telegram private | `tg-u{user_id}` | `tg-u123456` |
| Telegram group | `tg-g{chat_id}` | `tg-g-987654` |

This follows bub's approach: the channel context **determines** the session ID
directly — no lookup table needed. Telegram private chats and groups each get
their own session automatically.

**Cross-channel resume**: any channel can resume any session by its ID.
`k -S tg-u123456` opens a Telegram user's session from the CLI. This works
naturally because the session directory is the same regardless of which
channel accesses it.

**Session resolution heuristics** (no persisted mapping):
- **CLI** `k -C`: resume the most recently active session (by `last_active_at`).
- **CLI** `k -S <id>`: explicit session selection.
- **Telegram**: deterministic `tg-u{user_id}` / `tg-g{chat_id}` — auto-created
  on first message, resumed on subsequent messages.

## Module Layout

```
kcastle/
├── __init__.py           # Public API re-exports
├── castle.py             # Castle — top-level lifecycle orchestrator
├── cli.py                # `k` command entry point
├── config.py             # CastleConfig, YAML + env var loading, built-in registry
├── daemon.py             # Daemon process management (start/stop/status/restart)
├── setup.py              # First-run env var detection + config generation
├── skills/
│   ├── __init__.py       # Re-exports: SkillManager, Skill
│   ├── manager.py        # Skill discovery/search across builtin/user/project
│   ├── skill.py          # SKILL.md loading/parsing + hint rendering
│   ├── skill-creator/    # Builtin skill package
│   └── skill-installer/  # Builtin skill package
├── session/
│   ├── __init__.py       # Re-exports: Session, SessionManager, SessionTraceStore
│   ├── session.py        # Session — wraps kagent.Agent + metadata
│   ├── manager.py        # SessionManager — session lifecycle CRUD
│   └── store.py          # SessionTraceStore — per-session trace persistence
└── channels/
    ├── __init__.py       # Channel protocol definition
    ├── cli.py            # CLI channel (prompt_toolkit)
    └── telegram.py       # Telegram channel
```

This mirrors kagent's `trace/` sub-package pattern — related types are grouped
into a sub-package with `__init__.py` re-exports. The `Channel` protocol lives
in `channels/__init__.py`, eliminating the awkward `channel.py` + `channels/`
parallel naming.

## Provider Configuration

Providers are configured as explicit provider profiles in `config.yaml`.
Each profile maps directly to a concrete kai provider class.

### Built-in provider registry

kcastle maintains a built-in registry of supported providers:

| Provider name | Class | Base URL | Env var |
|--------------|-------|----------|---------|
| `deepseek-openai` | `DeepseekOpenAI` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `deepseek-anthropic` | `DeepseekAnthropic` | `https://api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` |
| `minimax-openai` | `MinimaxOpenAI` | `https://api.minimaxi.com/v1` | `MINIMAX_API_KEY` |
| `minimax-anthropic` | `MinimaxAnthropic` | `https://api.minimaxi.com/anthropic` | `MINIMAX_API_KEY` |

Each built-in provider includes a default model catalogue, so the user config
does **not** need to list models.

### Minimal config

Because all provider details are built-in, a working `config.yaml` only needs
to select the default provider and model:

```yaml
default:
  provider: deepseek-openai
  model: deepseek-chat
```

This is the config written by the first-run setup (see below). Users can
extend it with additional sections as needed:

```yaml
default:
  provider: deepseek-openai
  model: deepseek-chat

agent:
  system_prompt: "You are a helpful AI assistant."
  max_turns: 100

channels:
  cli:
    enabled: true
  telegram:
    enabled: false
    token: ${KCASTLE_TG_TOKEN}
```

### Merge behaviour

At load time, `load_config` merges built-in providers with user config:

1. **Builtins form the base** — all 6 built-in providers are available by
   default, with `${VAR}` env var references for API keys.
2. **User overrides** — if the user defines a provider with the same name as
   a built-in (e.g. `deepseek-openai`), user-provided fields override the
   built-in values (shallow merge).
3. **Custom providers** — the user can define entirely new providers that
   don't match any built-in name.
4. **Env var expansion** — `${VAR}` references in both builtins and user
   config are expanded after the merge.

### First-run setup

On first launch, `k` detects that `config.yaml` is missing and runs a
non-interactive setup:

1. Scan for known API key env vars (`DEEPSEEK_API_KEY`, `MINIMAX_API_KEY`).
2. Print detection results (which keys are set / missing).
3. Pick the first detected vendor as the default.
4. Ask the user to confirm writing the config (`[Y/n]`).
5. Write a minimal `config.yaml` with only the `default:` section.

No interactive wizard, no prompts for API keys — the user is expected to
have already exported the relevant env var.

### Key design decisions

| Aspect | Decision |
|--------|----------|
| Provider identity | `provider` is the canonical config key and maps directly to a factory constructor. |
| Compatibility families | DeepSeek and MiniMax each expose two explicit provider profiles (OpenAI-compatible and Anthropic-compatible). |
| Built-in models | Builtins include model catalogues; user config only overrides when needed. |
| Env var interpolation | `${VAR}` in any YAML string value is expanded at load time. |
| Default selection | `default.provider` + `default.model` selects the active combination. Overridable via `KCASTLE_PROVIDER` / `KCASTLE_MODEL` env vars. |
| Model-specific options | Extra keys in a model entry (e.g. `max_tokens`, `reasoning`) are forwarded to the kai constructor. |

## Skills Architecture (bub-aligned)

kcastle supports Claude-style skills as a first-class runtime capability:
discover, search, use, create, and update. This is implemented at the kcastle
layer (application concern), while kagent remains a runtime that executes
flattened `Tool` lists.

### Skill Sources and Priority

Skills are discovered from three layers, with deterministic override priority:

1. **Builtin** (read-only, shipped with kcastle)
2. **User** (`~/.kcastle/skills`)
3. **Project** (`<project_root>/.skills`)

Conflict rule: `project > user > builtin` for the same `skill_id`.

### Project Root Discovery

Project root is resolved from the current workspace using nearest-parent rules:

1. Directory containing `.git/`
2. Else directory containing `pyproject.toml`
3. Else current working directory

If `<project_root>/.skills` exists, it is included as the highest-priority
skill layer.

### Skill Unit Format

Each skill is a directory containing `SKILL.md`:

```
<skills_dir>/<skill_id>/
└── SKILL.md            # required: YAML frontmatter + Markdown instructions
```

`SKILL.md` format:

```markdown
---
name: skill-name
description: What this skill helps with
tags: [optional, tags]
---

# Instructions
...
```

Invalid skill directories are skipped with warnings (non-fatal).

### Runtime Resolution and Loading

Current PoC behavior is split into two phases:

1. **Startup-time discovery**
  - `SkillManager.discover()` scans builtin/user/project layers
  - `Skill.render_compact(all_skills)` injects compact metadata into system prompt
2. **Per-turn explicit expansion**
  - User input is parsed for `$skill-name` hints via `Skill.extract_hints(...)`
  - Matched skills are expanded into the current turn input via `Skill.render_expanded(...)`

This keeps default prompts compact while allowing full instructions only when explicitly requested.

### Create/Update Lifecycle

kcastle exposes management operations for skill evolution:

- `create` — scaffold new skill directory with `skill.yaml`
- `update` — patch metadata/prompt/tools in place
- `search` — inspect currently discoverable skills

Builtin skills are read-only. User/project skills are writable. PoC uses direct
filesystem updates with atomic writes; richer `draft -> active` publishing can
be added later.

## Core Components

### Session

A `Session` wraps a `kagent.Agent` instance plus its metadata. It is the unit
of interaction — one channel talks to one session at a time.

```
Session
├── id: str                          # Session ID (= directory name)
├── name: str | None                 # Human-readable label
├── meta: SessionMeta                # Created/active timestamps
├── agent: kagent.Agent              # The live agent (None when suspended)
├── trace_store: SessionTraceStore   # Writes trace.jsonl in session dir
│
├── run(user_input) → AsyncIterator[AgentEvent]
├── steer(msg) / abort()
├── suspend() → None                 # Drops agent, trace already persisted
└── is_running: bool
```

**Lifecycle:**

```
                create()              run(input)
  ────────────► [IDLE] ─────────────► [RUNNING] ──┐
                  ▲                       │        │ (loop ends)
                  │                       ▼        │
                  │                   [IDLE] ◄─────┘
                  │
             resume(id)               suspend()
  (from disk) ──►[IDLE] ────────────► [SUSPENDED]
                                      (agent=None, trace on disk)
```

- **IDLE**: agent is alive in memory, ready for `run()`.
- **RUNNING**: `agent.run()` is active, events streaming.
- **SUSPENDED**: agent discarded, only `meta.json` + `trace.jsonl` remain.
  `resume()` rebuilds the agent from the trace.

### SessionTraceStore

A thin `TraceStore` implementation that writes to the session's own directory
instead of a shared `traces/` folder:

```python
class SessionTraceStore(TraceStore):
    """Writes trace.jsonl inside the session directory."""
    def __init__(self, session_dir: Path): ...
```

This keeps each session fully self-contained — the session directory can be
moved, backed up, or deleted as a unit.

### SessionManager

Manages session lifecycle. Uses the filesystem as the source of truth —
no separate registry file.

```
SessionManager
├── sessions_dir: Path               # ~/.kcastle/sessions/
├── sessions: dict[str, Session]     # In-memory cache of live sessions
├── agent_factory: AgentFactory      # Creates Agent from config + trace
│
├── create(session_id?, name?) → Session   # New session (auto-ID if omitted)
├── get_or_create(session_id) → Session    # Resume if exists, create if not
├── resume(session_id) → Session           # Load from disk
├── get(session_id) → Session | None       # From memory only
├── suspend(session_id)                    # Drop from memory
├── list() → list[SessionInfo]             # Scan sessions/ directory
├── drop(session_id)                       # Delete session directory
└── latest() → Session | None              # Most recently active
```

`list()` scans the `sessions/` directory and reads each `meta.json` — the
filesystem is the registry. No separate index file to keep in sync.

`get_or_create(session_id)` is the primary API for channels: Telegram calls
`get_or_create("tg-u123")` and gets back a session regardless of whether it
already exists.

**Agent factory**: A callable that knows how to create a `kagent.Agent` given a
`Trace`. Encapsulates provider, system prompt, tools, context builder, hooks:

```python
type AgentFactory = Callable[[Trace], Agent]
```

This separates "how to build an agent" from "which session to attach it to".

### SessionRegistry

**Removed** — the filesystem is the registry. `SessionManager.list()` scans
`sessions/` and reads `meta.json` files directly. No `registry.json`, no
`registry.py`.

### Channel Protocol

```python
class Channel(Protocol):
    @property
    def name(self) -> str: ...
    async def start(self, castle: Castle) -> None: ...
    async def stop(self) -> None: ...
```

Each channel:
1. Receives input from its platform.
2. Resolves which session to use (via `SessionManager` — create, resume, or
   get the current in-memory session).
3. Calls `session.run(input)` and iterates over `AgentEvent`s.
4. Renders events for its platform.

Session management commands (`/session new`, `/session list`, `/session switch <id>`,
etc.) are handled uniformly — each channel parses its platform's command format
and delegates to `SessionManager`.

### Castle

Top-level orchestrator that wires everything together. Named after the project —
the castle that houses agent "k".

```
Castle
├── config: CastleConfig
├── session_manager: SessionManager
├── channels: list[Channel]
│
├── create(config) → Castle          # Factory: build manager, channels
├── run()                            # Start all channels (asyncio.gather)
└── shutdown()                       # Stop channels, suspend sessions
```

**Startup flow:**

```
1. Detect missing config → run first-run setup if needed
2. Load config (YAML + built-in providers + env vars)
3. Create SessionManager (with AgentFactory from config)
3. Create channel instances
4. Start channels concurrently
5. Wait (long-running) or handle signals
```

**Shutdown flow (SIGINT/SIGTERM):**

```
1. Stop all channels (close connections, cancel listeners)
2. Abort any running agent sessions
3. Suspend all in-memory sessions (trace already persisted)
```

## Channel Designs

### CLI Channel

Interactive terminal using `prompt_toolkit`.

```
$ k                    # New session (auto-generated ID)
$ k -C                 # Continue most recently active session
$ k -S <id>            # Resume specific session by ID
$ k -d                 # Daemon mode (foreground, no interactive CLI)

$ k start              # Start daemon in background
$ k stop               # Stop the background daemon
$ k status             # Show daemon status
$ k restart            # Restart the daemon

k> hello               # User input → session.run("hello")
                       # Stream AgentEvent → render to terminal

k> /session list
  a1b2c3d4  refactor api layer    2h ago
  tg-u123   (unnamed)             1d ago

k> /session switch tg-u123
Switched to session "tg-u123"

k> /session new my-project
Created session "my-project" (id: my-project)
```

**Rendering**: `AgentEvent` → terminal output:
- `StreamChunk` → incremental text (token by token)
- `ToolExecStart/End` → tool call status line
- `AgentError` → highlighted error message
- `TurnEnd` → separator / stats (tokens, duration)

### Telegram Channel

Bot that auto-creates a session per chat. Works in both private chats
and group chats (responds when mentioned or replied to).

**Session mapping** (bub-style, no lookup table):
- Private chat → `session_id = "tg-u{user_id}"`
- Group chat → `session_id = "tg-g{chat_id}"`

The channel calls `SessionManager.get_or_create(session_id)` on each message.
If the session exists, it’s resumed; if not, it’s created. No binding table,
no registry — the session ID is derived deterministically from the chat context.

```
User (private): hello
Bot:  [agent response, streamed via message edits]

User (@bot in group): summarize this thread
Bot:  [agent response]

User: /session new
Bot:  ✓ Created new session, previous session archived

User: /session list
Bot:  tg-u123456 — (current, 2h ago)
      a1b2c3d4 — refactor api layer (1d ago)
```

**Group chat behavior**:
- Bot responds when **mentioned** (`@botname`) or **replied to**
- The entire group shares one session (one `chat_id` = one session)
- Messages include sender info in the user input for context

**Message batching**: Short-interval messages (within ~1s) in the same chat are
batched into a single `session.run()` call — avoids triggering multiple agent
runs for rapid-fire messages.

**Streaming strategy**: Send initial message on first text delta, then
`editMessageText` on subsequent deltas (throttled to avoid Telegram rate limits).

## Session ↔ Channel Interaction

```
Channel                     SessionManager              Session
   │                              │                        │
   │  resume(session_id)          │                        │
   │─────────────────────────────►│                        │
   │                              │  (load trace from disk)│
   │                              │  (create Agent)        │
   │           session            │                        │
   │◄─────────────────────────────│                        │
   │                              │                        │
   │  session.run("hello")        │                        │
   │──────────────────────────────┼───────────────────────►│
   │                              │                        │  agent.run()
   │          AgentEvent stream   │                        │
   │◄─────────────────────────────┼────────────────────────│
   │                              │                        │
   │  (render for platform)       │                        │
   │                              │                        │
```

## What kcastle Owns

- **Channel adapters** — CLI, Telegram (and future: Discord, Slack, Web)
- **Session management** — create, resume, suspend, drop, list
- **Persistence orchestration** — session directories, trace store wiring
- **Skill management** — layered discovery, override, search, create/update
- **Configuration** — model selection, system prompt, tool registration, channel config
- **Rendering** — platform-specific `AgentEvent` → output formatting

## What kcastle Does NOT Own

These live in lower layers:

- **Agent runtime** — `kagent` (agent loop, state, context builders)
- **LLM abstraction** — `kai` (providers, streaming, tool schemas)
- **Trace format** — `kagent` (TraceEntry serialization, Trace reconstruction)
- **Context management** — `kagent` (compaction, sliding window, adaptive)
- **Skill execution loop** — still `kagent` tool-calling runtime once skills are
  flattened into `list[Tool]`

## System Prompt Design

The system prompt is **assembled dynamically** at agent creation time (not stored
per session). This follows the pattern used by bub and kimi-cli — both build the
prompt from composable blocks rather than persisting a static string.

### Composition

The prompt is built by concatenating ordered blocks:

```
┌────────────────────────────────────────────────┐
│  Base identity / persona               (config) │
├────────────────────────────────────────────────┤
│  Runtime context (datetime, cwd, etc.)  (auto)   │
├────────────────────────────────────────────────┤
│  Workspace rules (AGENTS.md)           (file)    │
├────────────────────────────────────────────────┤
│  Tool descriptions                     (auto)    │
└────────────────────────────────────────────────┘
```

| Block | Source | When |
|-------|--------|------|
| **Base identity** | `config.toml` or prompt file | Static per deployment |
| **Runtime context** | Auto-injected (current time, platform, etc.) | Every agent creation |
| **Workspace rules** | `AGENTS.md` in working directory (if present) | Every agent creation |
| **Tool descriptions** | From registered `kai.Tool` list | Every agent creation |

### Implementation

The system prompt is built by the `AgentFactory` (or a dedicated prompt builder
callable it uses) each time an `Agent` is created — including on session resume.
This ensures the prompt always reflects the latest config, workspace rules, and
tool set. The prompt is passed to `kagent.Agent(system=...)` and never persisted
separately.

For the PoC phase, a simple block-concatenation approach (like bub) is sufficient.
Template rendering (like kimi-cli's Jinja2 agent specs) can be added later if
customization needs grow.
