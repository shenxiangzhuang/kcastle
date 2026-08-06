# kcastle-agent

A minimal OpenAI Responses Agent core with optional harness infrastructure.

```python
from openai import AsyncOpenAI

from pathlib import Path

from kagent import Agent, Session

session = Session.create(Path.home() / ".kcastle" / "sessions")

agent = Agent(
    client=AsyncOpenAI(),
    model="gpt-5.6-sol",
    instructions="You are a helpful agent.",
    state=session.state,
    commit=session.commit,
)

async for event in agent.run("Hello"):
    print(event)
```

`Agent` is the brain: it owns causal State, the loop, steering, follow-ups, cancellation, events,
and compaction. `kagent.harness` surrounds it with Session persistence, an explicit Env, and
executable ToolRuntime adapters. User interfaces render events and provide approvals only.
