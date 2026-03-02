# kcastle

Agent application with multi-endpoint support for the K agent framework.

kcastle is a general-purpose agent application built on `kagent`. It provides a unified
agent runtime accessible from multiple endpoints — each endpoint connects to the same
agent instance and shares conversation history via `kagent.Trace`.

## Architecture

```
┌─────────────────────────────────────────────┐
│  kcastle                                    │
│                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ CLI      │ │ Telegram │ │ Discord  │    │
│  │ Endpoint │ │ Endpoint │ │ Endpoint │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       │             │            │          │
│       └──────┬──────┴────────────┘          │
│              │                              │
│       ┌──────▼──────┐                       │
│       │ kagent.Agent│ ◄── EventBus (SPMC)   │
│       │ + Trace     │                       │
│       └──────┬──────┘                       │
│              │                              │
│       ┌──────▼──────┐                       │
│       │  TraceStore │ (persistence)         │
│       └─────────────┘                       │
└─────────────────────────────────────────────┘
```

### Endpoints

Each endpoint is an adapter that:
1. Receives user input from its platform (terminal, Telegram, Discord, etc.)
2. Sends commands to the shared `Agent` instance (`run()`, `steer()`, `abort()`)
3. Subscribes to the `Agent`'s event stream and renders output for its platform

### Shared State

All endpoints share the same `Agent` instance and its `Trace`. This means:
- Conversation history is unified across all endpoints
- A message sent via Telegram is visible when resuming from the CLI
- Pause/resume works by persisting the `Trace` via `TraceStore` and reconstructing the `Agent`

### What kcastle owns

- **Endpoint adapters** — CLI, Telegram, Discord (and future endpoints)
- **Provider factory/registry** — protocol-to-provider construction policy (`kcastle.provider_factory`)
- **Persistence** — `TraceStore` configuration and session management
- **Tool registration** — Domain-specific tools for the agent
- **Configuration** — Agent setup, system prompts, model selection
- **UI/rendering** — Platform-specific output formatting

### What kcastle does NOT own

These live in lower layers:
- **Agent runtime** — `kagent` (agent loop, state, events, context builders)
- **LLM abstraction** — `kai` (providers, streaming, tool schemas)
