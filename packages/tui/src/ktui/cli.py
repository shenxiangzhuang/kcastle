"""TUI composition root."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from kagent import Agent, CompactionConfig, Env, Session, ToolRuntime
from kagent.harness.tools import shell_tool
from openai import AsyncOpenAI

from ktui.app import AgentTUI, Backend

_DEFAULT_INSTRUCTIONS = """You are K, a capable agent working in the current directory.
Use the available tools to inspect reality, act, verify results, and continue until the user's
request is genuinely complete. Keep the user informed through concise final answers.
"""


def backends_from_env(env: Mapping[str, str]) -> tuple[Backend, ...]:
    """Return all configured Responses API backends in preference order."""

    backends: list[Backend] = []
    deepseek_key = env.get("DEEPSEEK_API_KEY")
    if deepseek_key and deepseek_key.strip():
        backends.append(
            Backend(
                name="DeepSeek",
                client=AsyncOpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"),
                model="deepseek-v4-flash",
                context_window=1_000_000,
            )
        )

    openai_key = env.get("OPENAI_API_KEY")
    if openai_key and openai_key.strip():
        backends.append(
            Backend(
                name="OpenAI",
                client=AsyncOpenAI(api_key=openai_key),
                model="gpt-5.5",
                context_window=1_050_000,
            )
        )

    if not backends:
        raise ValueError("set DEEPSEEK_API_KEY or OPENAI_API_KEY")
    return tuple(backends)


def main() -> None:
    parser = argparse.ArgumentParser(description="K agent TUI")
    parser.add_argument("--model")
    parser.add_argument("--context-window", type=int)
    parser.add_argument("--session", type=Path, help="resume a session JSONL file")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path.home() / ".kcastle" / "sessions",
    )
    args = parser.parse_args()

    try:
        backends = backends_from_env(os.environ)
    except ValueError as error:
        parser.error(str(error))
    backend = backends[0]
    if args.model is not None:
        backend = replace(backend, model=args.model)
    if args.context_window is not None:
        backend = replace(backend, context_window=args.context_window)
    backends = (backend, *backends[1:])

    session = (
        Session.open(args.session) if args.session is not None else Session.create(args.session_dir)
    )
    tools = ToolRuntime(Env(Path.cwd()), [shell_tool])
    agent = Agent(
        client=backend.client,
        model=backend.model,
        instructions=_DEFAULT_INSTRUCTIONS,
        tools=tools.schemas,
        state=session.state,
        commit=session.commit,
        compaction=CompactionConfig(context_window=backend.context_window),
    )
    AgentTUI(agent=agent, tools=tools, backends=backends, session=session).run()
