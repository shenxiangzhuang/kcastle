"""Textual projection of a kagent event stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path

from kagent import (
    Agent,
    CompactionConfig,
    CompactionFinished,
    CompactionStarted,
    ModelStarted,
    RunFinished,
    Session,
    SessionError,
    SessionInfo,
    State,
    TextDelta,
    ToolFinished,
    ToolResult,
    ToolRuntime,
    ToolStarted,
)
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall, ResponseUsage
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.command import CommandPalette
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widgets import Button, Collapsible, Input, Label, Markdown, OptionList, Static
from textual.widgets.option_list import Option

_STREAM_FLUSH_INTERVAL = 0.05


def _tool_summary(arguments: str) -> str:
    """Return a compact, readable argument preview for a tool row."""

    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        value = arguments
    if isinstance(value, dict):
        if len(value) == 1:
            preview = str(next(iter(value.values())))
        else:
            preview = "  ".join(f"{key}={item}" for key, item in value.items())
    else:
        preview = str(value)
    preview = " ".join(preview.split())
    return f"  {preview[:117]}{'…' if len(preview) > 117 else ''}" if preview else ""


@dataclass(frozen=True, slots=True)
class Backend:
    """One detected Responses API backend available to the TUI."""

    name: str
    client: AsyncOpenAI
    model: str
    context_window: int


class PermissionMode(StrEnum):
    """TUI policy for tools that request approval."""

    ASK = "ask"
    ALLOW_ALL = "allow all"


class Transcript(VerticalScroll):
    """Append-only transcript with one mutable assistant streaming block."""

    def __init__(self) -> None:
        super().__init__(id="transcript")
        self._assistant: Markdown | None = None
        self._assistant_buffer: list[str] = []
        self._assistant_flush_timer: Timer | None = None
        self._assistant_started = False
        self._tools: dict[str, tuple[Collapsible, Static, str, str]] = {}

    async def write(self, label: str, value: str, *, style: str = "") -> None:
        await self.mount(Static(Text.assemble((f"{label}\n", style), value), classes="entry"))
        self.scroll_end(animate=False)

    async def start_assistant(self) -> None:
        await self.flush_assistant()
        self._assistant = Markdown(classes="assistant-body")
        self._assistant_started = False
        await self.mount(
            Vertical(
                Static(Text("K", style="bold #c4b5fd"), classes="assistant-label"),
                self._assistant,
                classes="entry",
            )
        )
        self.scroll_end(animate=False)

    async def append_assistant(self, delta: str) -> None:
        if self._assistant is None:
            return
        if not self._assistant_started:
            self._assistant_started = True
            await self._assistant.append(delta)
            self.scroll_end(animate=False)
        else:
            self._assistant_buffer.append(delta)
            if self._assistant_flush_timer is None:
                self._assistant_flush_timer = self.set_timer(
                    _STREAM_FLUSH_INTERVAL, self.flush_assistant
                )

    async def flush_assistant(self) -> None:
        """Render buffered streaming text immediately."""

        if self._assistant_flush_timer is not None:
            self._assistant_flush_timer.stop()
            self._assistant_flush_timer = None
        if self._assistant is None or not self._assistant_buffer:
            return
        delta = "".join(self._assistant_buffer)
        self._assistant_buffer.clear()
        await self._assistant.append(delta)
        self.scroll_end(animate=False)

    async def start_tool(self, call_id: str, name: str, arguments: str) -> None:
        """Add one compact, expandable tool call."""

        detail = Static(arguments, markup=False, classes="tool-detail")
        tool = Collapsible(
            detail,
            title=f"◌  {name}{_tool_summary(arguments)}",
            collapsed=True,
            classes="tool-call",
        )
        self._tools[call_id] = (tool, detail, arguments, name)
        await self.mount(tool)
        self.scroll_end(animate=False)

    async def finish_tool(self, call_id: str, name: str, output: str, *, is_error: bool) -> None:
        """Complete a tool call without expanding its potentially large output."""

        current = self._tools.pop(call_id, None)
        if current is None:
            return
        tool, detail, arguments, _ = current
        tool.title = f"{'!' if is_error else '✓'}  {name}{_tool_summary(arguments)}"
        tool.set_class(is_error, "tool-error")
        detail.update(f"Arguments\n{arguments}\n\nOutput\n{output}")
        self.scroll_end(animate=False)

    async def load(self, state: State) -> None:
        """Replace the projection with the readable messages from a session."""

        await self.flush_assistant()
        self.anchor()
        await self.query(".entry, .tool-call, .session-history").remove()
        self._assistant = None
        self._tools.clear()
        transcript: list[str] = []

        async def flush_transcript() -> None:
            if not transcript:
                return
            await self.mount(Markdown("\n\n---\n\n".join(transcript), classes="session-history"))
            transcript.clear()

        latest = state.latest_compaction
        if latest is not None:
            transcript.append(f"**Earlier context (compacted)**\n\n{latest.summary}")
        for item in state.active_items():
            role = item.get("role")
            content = item.get("content")
            if role == "user" and isinstance(content, str):
                transcript.append(f"**You**\n\n{content}")
            elif role == "assistant" and isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "output_text":
                        continue
                    value = part.get("text")
                    if isinstance(value, str):
                        parts.append(value)
                text = "".join(parts)
                if text:
                    transcript.append(f"**K**\n\n{text}")
            elif item.get("type") == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                arguments = item.get("arguments")
                if (
                    isinstance(call_id, str)
                    and isinstance(name, str)
                    and isinstance(arguments, str)
                ):
                    await flush_transcript()
                    await self.start_tool(call_id, name, arguments)
            elif item.get("type") == "function_call_output":
                call_id = item.get("call_id")
                output = item.get("output")
                current = self._tools.get(call_id) if isinstance(call_id, str) else None
                if isinstance(call_id, str) and current is not None and isinstance(output, str):
                    await self.finish_tool(call_id, current[3], output, is_error=False)
        await flush_transcript()
        if self.children:
            self.scroll_end(animate=False)


class ApprovalScreen(ModalScreen[bool]):
    """Confirmation boundary for tools with external effects."""

    BINDINGS = [("escape", "deny", "Deny")]

    def __init__(self, call: ResponseFunctionToolCall) -> None:
        super().__init__()
        self.call = call

    def compose(self) -> ComposeResult:
        with Container(id="approval"):
            yield Label(f"Allow {self.call.name}?\n\n{self.call.arguments}")
            yield Button("Allow", id="allow", variant="success")
            yield Button("Deny", id="deny", variant="error")

    @on(Button.Pressed, "#allow")
    def allow(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def deny(self) -> None:
        self.dismiss(False)


class PickerScreen[T](ModalScreen[T | None]):
    """Small modal picker backed by Textual's native option list."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, heading: str, choices: tuple[tuple[str, T], ...]) -> None:
        super().__init__()
        self.heading = heading
        self.choices = choices

    def compose(self) -> ComposeResult:
        with Container(id="picker"):
            yield Label(self.heading)
            yield OptionList(
                *(Option(label, id=str(index)) for index, (label, _) in enumerate(self.choices))
            )

    @on(OptionList.OptionSelected)
    def select(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.choices[event.option_index][1])

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentTUI(App[None]):
    """A thin Textual adapter over one continuing Agent instance."""

    CSS = """
    Screen {
        layout: vertical;
        color: #d8dee9;
        background: #0d1117;
    }
    .banner {
        height: 3;
        padding: 1 3 0 3;
        color: #8b949e;
        background: #161b22;
        border-bottom: solid #30363d;
    }
    #transcript { height: 1fr; padding: 1 3; background: #0d1117; }
    .entry {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        padding-left: 1;
        border-left: solid #30363d;
    }
    .assistant-label { height: 1; color: #c4b5fd; }
    .assistant-body { height: auto; }
    .assistant-body, .session-history {
        link-color: #c4b5fd;
        link-color-hover: #ede9fe;
        link-style: underline;
    }
    .assistant-body MarkdownHeader, .session-history MarkdownHeader { color: #e5e7eb; }
    .assistant-body MarkdownTableContent > .header,
    .session-history MarkdownTableContent > .header { color: #c4b5fd; }
    .tool-call {
        height: auto;
        margin: 0 0 0 1;
        padding: 0 1;
        color: #8b949e;
        background: #161b22;
        border-left: solid #30363d;
        border-top: none;
    }
    .tool-call:focus-within { background: #1c2128; }
    .tool-call.tool-error { color: #ff7b72; border-left: solid #ff7b72; }
    .tool-call > CollapsibleTitle { width: 100%; }
    .tool-detail {
        height: auto;
        max-height: 14;
        padding: 1;
        color: #b1bac4;
        background: #0d1117;
        overflow-y: auto;
    }
    #status {
        height: 2;
        padding: 0 3 1 3;
        color: #6e7681;
        background: #0d1117;
    }
    #composer {
        height: 3;
        margin: 0 3 1 3;
        padding: 0 1;
        color: #f0f3f6;
        background: #161b22;
        border: round #30363d;
    }
    #composer:focus { border: round #a78bfa; }
    ApprovalScreen { align: center middle; }
    #approval { width: 70%; height: auto; padding: 2; border: round #a78bfa; }
    #approval Button { margin-top: 1; margin-right: 1; }
    PickerScreen { align: center middle; }
    #picker { width: 70%; height: auto; max-height: 70%; padding: 2; border: round #a78bfa; }
    #picker OptionList { height: auto; max-height: 20; margin-top: 1; }
    """

    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("/", "slash", show=False, priority=True),
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+q", "ignore", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        agent: Agent,
        tools: ToolRuntime | None = None,
        backends: Iterable[Backend] = (),
        session: Session | None = None,
        permission_mode: PermissionMode = PermissionMode.ASK,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.tools = tools
        self.backends = tuple(backends)
        self.session = session
        self.permission_mode = permission_mode
        self._approval_lock = asyncio.Lock()
        self.activity = "idle"
        self.cached_tokens = 0
        self.used_tokens = 0
        response = agent.state.latest_response
        self._set_usage(None if response is None else response.usage)
        self.working_directory = tools.env.cwd if tools is not None else Path.cwd()

    def compose(self) -> ComposeResult:
        yield Transcript()
        yield Static(self.status_text, id="status")
        yield Input(placeholder="Message K…  / for commands", id="composer")

    async def on_mount(self) -> None:
        transcript = self.query_one(Transcript)
        if self.session is not None:
            await transcript.load(self.session.state)
        banner = Static(self.banner, classes="banner")
        first_entry = transcript.children[0] if transcript.children else None
        await transcript.mount(banner, before=first_entry)
        self.query_one("#composer", Input).focus()

    @property
    def banner(self) -> Text:
        return Text.assemble(
            ("K", "bold #c4b5fd"),
            (f" v{version('kcastle')}", "dim"),
            ("  ·  ", "#484f58"),
            str(self.working_directory),
        )

    @property
    def status_text(self) -> str:
        config = self.agent.compaction_config
        window = "?" if config is None else f"{config.context_window:,}"
        context = f"{self.cached_tokens:,}/{self.used_tokens:,}/{window}"
        return (
            f"{self.activity} · permissions: {self.permission_mode.value}"
            f" · {self.agent.model} · context {context}"
        )

    def show_status(self, activity: str) -> None:
        self.activity = activity
        self.query_one("#status", Static).update(self.status_text)

    def _set_usage(self, usage: ResponseUsage | None) -> None:
        self.cached_tokens = 0 if usage is None else usage.input_tokens_details.cached_tokens
        self.used_tokens = 0 if usage is None else usage.total_tokens

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        del screen
        yield SystemCommand("/resume", "Switch to a persisted session", self.action_resume)
        yield SystemCommand("/model", "Switch model backend", self.action_model)
        yield SystemCommand("/compact", "Compact the current context", self.action_compact)
        if self.agent.is_running:
            yield SystemCommand(
                "/queue", "Queue a message after the current run", self.action_queue
            )
        yield SystemCommand(
            "/permissions",
            f"Switch approval policy (currently {self.permission_mode.value})",
            self.action_permissions,
        )
        yield SystemCommand("/exit", "Exit K", self.action_exit)

    @on(Input.Submitted, "#composer")
    async def submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        transcript = self.query_one(Transcript)

        if text.startswith("/"):
            match text:
                case "/resume":
                    self.action_resume()
                case "/model":
                    self.action_model()
                case "/compact":
                    self.action_compact()
                case "/permissions":
                    self.action_permissions()
                case "/exit":
                    self.action_exit()
                case "/queue":
                    self.show_status("/queue requires a message")
                case _ if text.startswith("/queue ") and self.agent.is_running:
                    queued = text.removeprefix("/queue ").strip()
                    if queued:
                        self.agent.queue(queued)
                        await transcript.write("Queued", queued, style="bold yellow")
                    else:
                        self.show_status("/queue requires a message")
                case _:
                    self.show_status(f"unknown command: {text}")
            return

        if self.agent.is_running:
            self.agent.steer(text)
            await transcript.write("Steer", text, style="bold magenta")
            return

        await transcript.write("You", text, style="bold green")
        self.run_agent(text)

    @work(exclusive=True, exit_on_error=False)
    async def run_agent(self, text: str) -> None:
        transcript = self.query_one(Transcript)
        assistant_started = False
        try:
            executor = None if self.tools is None else self.execute_tool
            async for event in self.agent.run(text, execute_tool=executor):
                if not isinstance(event, TextDelta):
                    await transcript.flush_assistant()
                match event:
                    case ModelStarted(turn=turn):
                        self.show_status(f"thinking · turn {turn}")
                        assistant_started = False
                    case TextDelta(text=delta):
                        if not assistant_started:
                            await transcript.start_assistant()
                            assistant_started = True
                        await transcript.append_assistant(delta)
                    case ToolStarted(call=call):
                        self.show_status(f"running {call.name}")
                        await transcript.start_tool(call.call_id, call.name, call.arguments)
                    case ToolFinished(call=call, output=output, is_error=is_error):
                        await transcript.finish_tool(
                            call.call_id, call.name, output, is_error=is_error
                        )
                    case CompactionStarted(tokens_before=tokens):
                        self.show_status(f"compacting · ~{tokens:,} tokens")
                    case CompactionFinished():
                        await transcript.write("Context", "Compaction complete", style="dim")
                    case RunFinished(usage=usage):
                        self._set_usage(usage)
                        self.show_status("idle")
        except Exception as error:
            self.show_status("error")
            await transcript.write("Error", f"{type(error).__name__}: {error}", style="bold red")
        finally:
            await transcript.flush_assistant()

    async def approve(self, call: ResponseFunctionToolCall) -> bool:
        if self.permission_mode is PermissionMode.ALLOW_ALL:
            return True
        async with self._approval_lock:
            if self.permission_mode is PermissionMode.ALLOW_ALL:
                return True
            return await self.push_screen_wait(ApprovalScreen(call))

    async def execute_tool(self, call: ResponseFunctionToolCall) -> ToolResult:
        if self.tools is None:
            raise RuntimeError("tool runtime is not configured")
        return await self.tools.execute(call, approve=self.approve)

    def action_slash(self) -> None:
        focused = self.focused
        if isinstance(focused, Input) and focused.id == "composer" and not focused.value:
            self.push_screen(CommandPalette(id="--command-palette"))
        elif isinstance(focused, Input):
            focused.insert_text_at_cursor("/")

    def action_cancel(self) -> None:
        if self.agent.is_running:
            self.agent.abort()
            self.show_status("cancelled")

    def action_ignore(self) -> None:
        """Shadow Textual's default Ctrl+Q binding."""

    def action_queue(self) -> None:
        composer = self.query_one("#composer", Input)
        composer.value = "/queue "
        composer.cursor_position = len(composer.value)
        composer.focus()

    def action_permissions(self) -> None:
        self.permission_mode = (
            PermissionMode.ALLOW_ALL
            if self.permission_mode is PermissionMode.ASK
            else PermissionMode.ASK
        )
        self.show_status("idle" if not self.agent.is_running else self.activity)

    def action_exit(self) -> None:
        self.exit()

    @work(group="control", exclusive=True, exit_on_error=False)
    async def action_resume(self) -> None:
        if self.agent.is_running:
            self.show_status("cancel the active run before resuming")
            return
        if self.session is None:
            self.show_status("session persistence is not configured")
            return
        try:
            sessions = Session.list(self.session.info.path.parent)
        except SessionError as error:
            self.show_status("session error")
            await self.query_one(Transcript).write("Error", str(error), style="bold red")
            return
        if not sessions:
            self.show_status("no persisted sessions")
            return
        info = await self.push_screen_wait(
            PickerScreen(
                "Resume session",
                tuple((self._session_label(info), info) for info in sessions),
            )
        )
        if info is not None:
            try:
                await self.resume(Session.open(info.path))
            except SessionError as error:
                self.show_status("session error")
                await self.query_one(Transcript).write("Error", str(error), style="bold red")

    async def resume(self, session: Session) -> None:
        """Switch the continuing Agent to a persisted session."""

        self.agent.state = session.state
        self.agent.commit = session.commit
        self.session = session
        self.permission_mode = PermissionMode.ASK
        response = session.state.latest_response
        self._set_usage(None if response is None else response.usage)
        await self.query_one(Transcript).load(session.state)
        self.show_status(f"resumed {session.info.title}")

    @staticmethod
    def _session_label(info: SessionInfo) -> str:
        return f"{info.title} · {info.created_at.astimezone():%Y-%m-%d %H:%M}"

    @work(group="control", exclusive=True, exit_on_error=False)
    async def action_model(self) -> None:
        if self.agent.is_running:
            self.show_status("cancel the active run before switching model")
            return
        if not self.backends:
            self.show_status("no alternative models detected")
            return
        backend = await self.push_screen_wait(
            PickerScreen(
                "Select model",
                tuple((f"{item.name} · {item.model}", item) for item in self.backends),
            )
        )
        if backend is None:
            return
        self.agent.client = backend.client
        self.agent.model = backend.model
        self.agent.compaction_config = CompactionConfig(context_window=backend.context_window)
        self._set_usage(None)
        self.show_status(f"model: {backend.model}")

    @work(group="control", exclusive=True, exit_on_error=False)
    async def action_compact(self) -> None:
        if self.agent.is_running:
            self.show_status("cancel the active run before compacting")
            return
        self.show_status("compacting")
        try:
            await self.agent.compact()
        except Exception as error:
            self.show_status("compaction error")
            await self.query_one(Transcript).write(
                "Error", f"{type(error).__name__}: {error}", style="bold red"
            )
            return
        await self.query_one(Transcript).write("Context", "Compaction complete", style="dim")
        self.show_status("idle")
