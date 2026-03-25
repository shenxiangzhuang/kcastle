"""CLI channel — interactive terminal using prompt_toolkit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kagent import (
    AgentError,
    AgentEvent,
    AgentStart,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)
from kai import StreamEvent, TextDelta, ThinkDelta, ToolCallBegin
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from kcastle.session.session import Session

if TYPE_CHECKING:
    from kcastle.castle import Castle


class _StatusLine:
    """An ephemeral status line that overwrites itself in-place."""

    def __init__(self) -> None:
        self._visible = False

    def show(self, text: str) -> None:
        """Display a dim status message on the current line."""
        sys.stdout.write(f"\r\033[2K\033[2m{text}\033[0m")
        sys.stdout.flush()
        self._visible = True

    def clear(self) -> None:
        """Erase the status line (no-op if already clear)."""
        if not self._visible:
            return
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        self._visible = False


class _EventRenderer:
    """Renders agent events with ephemeral status indicators."""

    def __init__(self) -> None:
        self._status = _StatusLine()
        self._phase = "idle"

    def render(self, event: AgentEvent) -> None:
        match event:
            case AgentStart():
                pass
            case TurnStart():
                self._status.show("› thinking...")
                self._phase = "thinking"
            case StreamChunk(event=stream_event):
                self._render_stream(stream_event)
            case ToolExecStart(tool_name=name, arguments=args):
                _args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                self._status.show(f"› {name}({_args_str})")
            case ToolExecEnd(tool_name=name, is_error=is_err, duration_ms=dur):
                marker = "✗" if is_err else "›"
                self._status.show(f"{marker} {name} done ({dur:.0f}ms)")
            case TurnEnd():
                self._status.clear()
                if self._phase == "streaming":
                    print(flush=True)  # newline after streamed text
                self._phase = "idle"
            case AgentError(error=err):
                self._status.clear()
                print(f"\n✗ Error: {err}", file=sys.stderr, flush=True)
                self._phase = "idle"
            case _:
                self._status.clear()
                self._phase = "idle"

    def _render_stream(self, event: StreamEvent) -> None:
        match event:
            case ThinkDelta():
                if self._phase != "reasoning":
                    self._status.show("› reasoning...")
                    self._phase = "reasoning"
            case TextDelta(delta=delta):
                if self._phase != "streaming":
                    self._status.clear()
                    self._phase = "streaming"
                sys.stdout.write(delta)
                sys.stdout.flush()
            case ToolCallBegin(name=name):
                self._status.show(f"› calling {name}...")
                self._phase = "tool_call"
            case _:
                pass


_COMMANDS = {
    "/session list": "List all sessions",
    "/session new": "Create a new session (optional: --id <id> [name])",
    "/session switch": "Switch to a session by ID",
    "/help": "Show available commands",
    "/quit": "Exit the CLI",
}


class _SlashCompleter(Completer):
    """Auto-complete slash commands when the input starts with ``/``."""

    def get_completions(self, document, complete_event):  # type: ignore[override]
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _COMMANDS.items():
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


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

        return PromptSession(
            history=history,
            completer=_SlashCompleter(),
            complete_while_typing=True,
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
        renderer = _EventRenderer()

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
                    renderer.render(event)
            except (RuntimeError, ValueError, KeyError) as e:
                print(f"\n✗ Error: {e}", file=sys.stderr, flush=True)

        manager.suspend(session.id)

    async def stop(self) -> None:
        """Stop the CLI channel."""
        self._running = False
