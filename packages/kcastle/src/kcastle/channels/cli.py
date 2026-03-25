"""CLI channel — interactive terminal using prompt_toolkit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kagent import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentStart,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)
from kai import TextDelta
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from kcastle.session.session import Session

if TYPE_CHECKING:
    from kcastle.castle import Castle


def _render_event(event: AgentEvent) -> None:
    """Render a single AgentEvent to the terminal."""
    match event:
        case AgentStart():
            pass  # silent
        case TurnStart():
            pass  # silent
        case StreamChunk(event=stream_event):
            if isinstance(stream_event, TextDelta):
                sys.stdout.write(stream_event.delta)
                sys.stdout.flush()
        case ToolExecStart(tool_name=name, arguments=args):
            _args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"\n⚙ {name}({_args_str})", flush=True)
        case ToolExecEnd(tool_name=name, is_error=is_err, duration_ms=dur):
            status = "✗" if is_err else "✓"
            print(f"  {status} {name} ({dur:.0f}ms)", flush=True)
        case TurnEnd():
            print(flush=True)  # newline after streamed text
        case AgentError(error=err):
            print(f"\n✗ Error: {err}", file=sys.stderr, flush=True)
        case AgentEnd():
            pass  # silent
        case _:
            pass


_COMMANDS = {
    "/session list": "List all sessions",
    "/session new": "Create a new session (optional: --id <id> [name])",
    "/session switch": "Switch to a session by ID",
    "/help": "Show available commands",
    "/quit": "Exit the CLI",
}


def parse_session_new_args(args: list[str]) -> tuple[str | None, str]:
    """Parse args after ``/session new``.

    Supports:
    - ``/session new``
    - ``/session new <name...>``
    - ``/session new --id <id> [name...]``
    """
    if args[:1] == ["--id"]:
        if len(args) < 2:
            raise ValueError("Usage: /session new [--id <id>] [name]")
        sid = args[1]
        name = " ".join(args[2:]) if len(args) > 2 else ""
        return sid, name

    name = " ".join(args) if args else ""
    return None, name


async def _handle_command(line: str, castle: Castle, session: Session) -> Session | None:
    """Handle a slash command.  Returns new session if switched, None otherwise."""
    parts = line.strip().split()
    cmd = " ".join(parts[:2]) if len(parts) >= 2 else parts[0]

    if cmd == "/help":
        print("\nAvailable commands:")
        for c, desc in _COMMANDS.items():
            print(f"  {c:20s}  {desc}")
        print()
        return None

    if cmd == "/quit":
        return None  # Caller checks line == "/quit"

    if parts[0] == "/session":
        if len(parts) < 2:
            print("Usage: /session <list|new|switch>")
            return None

        sub = parts[1]
        manager = castle.session_manager

        if sub == "list":
            sessions = manager.list()
            if not sessions:
                print("  (no sessions)")
            else:
                for s in sessions:
                    marker = " *" if s.id == session.id else ""
                    name_str = f"  {s.name}" if s.name else ""
                    print(f"  {s.id}{name_str}{marker}")
            print()
            return None

        if sub == "new":
            try:
                sid, name = parse_session_new_args(parts[2:])
            except ValueError as e:
                print(str(e))
                return None

            new_session = manager.create(session_id=sid, name=name)
            print(f'Created session "{new_session.id}"')
            return new_session

        if sub == "switch":
            if len(parts) < 3:
                print("Usage: /session switch <id>")
                return None
            target_id = parts[2]
            try:
                new_session = manager.get_or_create(target_id)
                print(f'Switched to session "{target_id}"')
                return new_session
            except (KeyError, ValueError) as e:
                print(f"Error: {e}")
                return None

    print(f"Unknown command: {line}")
    return None


class CLIChannel:
    """Interactive CLI channel using stdin/stdout with prompt_toolkit.

    Uses ``prompt_toolkit.PromptSession`` for proper line editing, arrow-key
    navigation, CJK character support, persistent history, and slash-command
    completion.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        continue_latest: bool = False,
    ) -> None:
        self._session_id = session_id
        self._continue_latest = continue_latest
        self._running = False

    @property
    def name(self) -> str:
        return "cli"

    @staticmethod
    def _build_prompt(home: Path) -> PromptSession[str]:
        """Build a ``PromptSession`` with history and slash-command completion."""
        history_dir = home / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(history_dir / "cli.history"))

        commands = list(_COMMANDS.keys())
        completer = WordCompleter(commands, ignore_case=True)

        return PromptSession(
            history=history,
            completer=completer,
            complete_while_typing=False,
        )

    async def start(self, castle: Castle) -> None:
        """Start the interactive CLI loop."""
        self._running = True
        manager = castle.session_manager

        if self._session_id:
            session = manager.get_or_create(self._session_id)
            print(f'Resumed session "{session.id}"')
        elif self._continue_latest:
            session = manager.latest()
            if session is None:
                session = manager.create()
                print(f'No previous session. Created "{session.id}"')
            else:
                print(f'Continuing session "{session.id}"')
        else:
            session = manager.create()
            print(f'New session "{session.id}"')

        print("Type /help for commands, /quit to exit.\n")

        prompt_session = self._build_prompt(castle.config.home)

        while self._running:
            try:
                with patch_stdout():
                    line = await prompt_session.prompt_async("k> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            line = line.strip()
            if not line:
                continue

            if line == "/quit":
                break

            if line.startswith("/"):
                new_session = await _handle_command(line, castle, session)
                if new_session is not None:
                    session = new_session
                continue

            try:
                user_input = castle.prepare_user_input(line)
                async for event in session.run(user_input):
                    _render_event(event)
            except (RuntimeError, ValueError, KeyError) as e:
                print(f"\n✗ Error: {e}", file=sys.stderr, flush=True)

        manager.suspend(session.id)

    async def stop(self) -> None:
        """Stop the CLI channel."""
        self._running = False
