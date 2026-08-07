"""Textual projection of a kagent event stream."""

from __future__ import annotations

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
from textual.widgets import Button, Input, Label, Markdown, OptionList, Static
from textual.widgets.option_list import Option

_STREAM_FLUSH_INTERVAL = 0.05


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

    async def write(self, label: str, value: str, *, style: str = "") -> None:
        await self.mount(Static(Text.assemble((f"{label}\n", style), value), classes="entry"))
        self.scroll_end(animate=False)

    async def start_assistant(self) -> None:
        await self.flush_assistant()
        self._assistant = Markdown(classes="assistant-body")
        self._assistant_started = False
        await self.mount(
            Vertical(
                Static(Text("K", style="bold cyan"), classes="assistant-label"),
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

    async def load(self, state: State) -> None:
        """Replace the projection with the readable messages from a session."""

        await self.flush_assistant()
        await self.query(".entry, .session-history").remove()
        self._assistant = None
        transcript: list[str] = []
        for item in state.items():
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
        if transcript:
            await self.mount(Markdown("\n\n---\n\n".join(transcript), classes="session-history"))
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
    Screen { layout: vertical; }
    .banner {
        height: auto;
        padding: 0 2;
        color: $text;
        background: $panel;
        border-bottom: solid $accent;
    }
    #transcript { height: 1fr; padding: 1 2; }
    .entry { width: 100%; height: auto; margin-bottom: 1; }
    .assistant-label { height: 1; }
    .assistant-body { height: auto; }
    #status { height: 1; padding: 0 2; color: $text-muted; }
    #composer {
        height: 3;
        margin: 0 2 1 2;
        padding: 0 1;
        color: $text;
        background: $surface;
        border: round $accent;
    }
    ApprovalScreen { align: center middle; }
    #approval { width: 70%; height: auto; padding: 2; border: round $accent; }
    #approval Button { margin-top: 1; margin-right: 1; }
    PickerScreen { align: center middle; }
    #picker { width: 70%; height: auto; max-height: 70%; padding: 2; border: round $accent; }
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
            ("K CASTLE", "bold cyan"),
            (f" v{version('kcastle')}", "dim"),
            ("  minimal agent harness", "dim"),
            "\n",
            ("cwd      ", "bold"),
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
        try:
            executor = None if self.tools is None else self.execute_tool
            async for event in self.agent.run(text, execute_tool=executor):
                if not isinstance(event, TextDelta):
                    await transcript.flush_assistant()
                match event:
                    case ModelStarted(turn=turn):
                        self.show_status(f"thinking · turn {turn}")
                        await transcript.start_assistant()
                    case TextDelta(text=delta):
                        await transcript.append_assistant(delta)
                    case ToolStarted(call=call):
                        self.show_status(f"running {call.name}")
                        await transcript.write(
                            "Tool",
                            f"{call.name}({call.arguments})",
                            style="bold blue",
                        )
                    case ToolFinished(call=call, output=output, is_error=is_error):
                        label = f"{call.name} {'error' if is_error else 'done'}"
                        style = "bold red" if is_error else "dim"
                        await transcript.write(label, output, style=style)
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
