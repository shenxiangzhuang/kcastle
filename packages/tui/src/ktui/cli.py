"""TUI composition root."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

from kagent import Agent, CompactionConfig, Env, Session, ToolRuntime
from kagent.harness.tools import shell_tool
from openai import AsyncOpenAI

from ktui.app import AgentTUI, Backend

_DEFAULT_INSTRUCTIONS = """You are K, a capable agent working in the current directory.
Use the available tools to inspect reality, act, verify results, and continue until the user's
request is genuinely complete. Keep the user informed through concise final answers.
"""

_HELP = """A minimal agent.

Usage: {prog} [OPTIONS] [COMMAND]

Commands:
  self  Manage the kcastle executable

Options:
  -h, --help     Display the concise help for this command
  -V, --version  Display the kcastle version
"""

_SELF_HELP = """Manage the kcastle executable

Usage: {prog} [OPTIONS] <COMMAND>

Commands:
  update  Update kcastle

Options:
  -h, --help  Display the concise help for this command
"""


class _HelpParser(argparse.ArgumentParser):
    help_text: str | None = None

    def format_help(self) -> str:
        if self.help_text is None:
            return super().format_help()
        return self.help_text.format(prog=self.prog)


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
    parser = _HelpParser(
        usage="%(prog)s [OPTIONS] [COMMAND]",
        add_help=False,
    )
    parser.help_text = _HELP
    parser.add_argument("-h", "--help", action="help")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {version('kcastle')}",
    )
    commands = parser.add_subparsers(dest="command")
    self_parser = commands.add_parser("self", add_help=False)
    self_parser.prog = "kcastle self"
    self_parser.help_text = _SELF_HELP
    self_parser.add_argument("-h", "--help", action="help")
    self_commands = self_parser.add_subparsers(required=True)
    self_commands.add_parser("update")
    args = parser.parse_args()

    if args.command == "self":
        subprocess.run(["uv", "tool", "upgrade", "kcastle", "--prerelease", "allow"], check=True)
        return

    try:
        backends = backends_from_env(os.environ)
    except ValueError as error:
        parser.error(str(error))
    backend = backends[0]
    session = Session.create(Path.home() / ".kcastle" / "sessions")
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
