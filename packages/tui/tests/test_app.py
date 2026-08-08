import asyncio
from importlib.metadata import version
from pathlib import Path

import pytest
from kagent import Agent, CompactionConfig, Env, ResponseMetadata, Session, State, ToolRuntime
from ktui.app import AgentTUI, ApprovalScreen, PermissionMode, Transcript
from openai.types.responses import ResponseFunctionToolCall, ResponseUsage
from tests_fakes import fake_client
from textual import events
from textual.command import CommandPalette
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Input, Label, Markdown, OptionList, Static


async def test_app_starts_and_accepts_input(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    path = session.info.path
    agent = Agent(
        client=fake_client("hello"),
        model="model-x",
        instructions="test",
        state=session.state,
        commit=session.commit,
        compaction=CompactionConfig(context_window=50_000),
    )
    app = AgentTUI(agent=agent, tools=ToolRuntime(Env(tmp_path)))

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", Input)
        transcript = app.query_one(Transcript)
        banner_widget = app.query_one(".banner", Static)
        banner = str(banner_widget.render())
        assert app.theme == "kcastle"
        assert app.focused is composer
        assert composer.outer_size.height == 3
        assert banner_widget.parent is transcript
        assert f"v{version('kcastle')}" in banner
        assert str(tmp_path) in banner
        assert "model-x" not in banner

        await pilot.press("h", "i", "enter")
        await pilot.pause()

        status = str(app.query_one("#status").render())
        user = str(transcript.query_one(".user-entry", Static).render())
        assert "›  hi" in user
        assert not transcript.query(".assistant-label")
        assert "idle" in status
        assert "model-x · context 80/150/50,000" in status
        assert not transcript.query(".banner")
        assert path.exists()


async def test_slash_opens_only_k_commands_and_quit_shortcuts_are_disabled() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)

    async with app.run_test() as pilot:
        commands = [command.title for command in app.get_system_commands(app.screen)]
        assert commands == ["/resume", "/model", "/compact", "/permissions", "/exit"]

        await pilot.click("#composer")
        await pilot.press("/")
        await pilot.pause()
        assert isinstance(app.screen, CommandPalette)

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, CommandPalette)

        await pilot.press("ctrl+p", "ctrl+q")
        await pilot.pause()
        assert not isinstance(app.screen, CommandPalette)
        assert app.query_one("#composer") is not None

        app.query_one(Transcript).focus()
        await pilot.press("/")
        await pilot.pause()
        assert isinstance(app.screen, CommandPalette)


async def test_typing_from_transcript_returns_to_composer_and_bottom() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)

    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        composer = app.query_one("#composer", Input)
        await transcript.mount(Static("\n".join(f"line {index}" for index in range(50))))
        await pilot.pause()
        transcript.scroll_home(animate=False)
        transcript.focus()
        await pilot.pause()

        await pilot.press("x")
        await pilot.pause()

        assert app.focused is composer
        assert composer.value == "x"
        assert transcript.scroll_y == transcript.max_scroll_y

        transcript.focus()
        transcript.post_message(events.Paste(" pasted"))
        await pilot.pause()

        assert app.focused is composer
        assert composer.value == "x pasted"


async def test_assistant_messages_render_as_markdown() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)

    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        await transcript.start_assistant()
        source = "| name | value |\n| --- | --- |\n| K | 1 |\n\n[OpenAI](https://openai.com)"
        for delta in source:
            await transcript.append_assistant(delta)
        await transcript.flush_assistant()
        await pilot.pause()

        markdown = transcript.query_one(Markdown)
        assert markdown.source == source
        assert len(markdown.query("MarkdownTable")) == 1
        assert "https://openai.com" in markdown.source


async def test_tool_output_stays_collapsed_until_requested() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)
    call = ResponseFunctionToolCall(
        type="function_call",
        call_id="call-1",
        name="shell",
        arguments='{"command":"find ."}',
    )

    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        await transcript.start_tool(call.call_id, call.name, call.arguments)
        output = "\n".join(["with [t,e] markup", *(f"line {index}" for index in range(30))])
        await transcript.finish_tool(call.call_id, call.name, output, is_error=False)
        await pilot.pause()

        tool = transcript.query_one(Collapsible)
        assert tool.collapsed
        assert tool.title == "✓  shell  find ."
        assert "with [t,e] markup" in str(tool.query_one(".tool-detail-content", Static).render())
        await pilot.click(".tool-call", offset=(30, 0))
        await pilot.pause()
        assert not tool.collapsed
        detail = tool.query_one(".tool-detail", VerticalScroll)
        detail.focus()
        await pilot.press("end")
        await pilot.pause()
        assert detail.max_scroll_y > 0
        assert detail.scroll_y == detail.max_scroll_y


async def test_compacted_history_renders_summary_and_active_suffix() -> None:
    state = State()
    state.append_user("old message")
    kept = state.append_user("kept message")
    state.append_compaction(summary="earlier summary", first_kept_id=kept.id, tokens_before=42)
    agent = Agent(client=fake_client("hello"), model="test", instructions="test", state=state)
    app = AgentTUI(agent=agent)

    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        await transcript.load(state)
        await pilot.pause()

        source = transcript.query_one(".session-history", Markdown).source
        assert "earlier summary" in source
        assert "kept message" in source
        assert "old message" not in source
        assert "old message" in str(list(state.items()))


async def test_permission_mode_defaults_to_ask_and_can_allow_all() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)

    async with app.run_test() as pilot:
        assert app.permission_mode is PermissionMode.ASK
        assert "permissions: ask" in str(app.query_one("#status").render())

        await pilot.click("#composer")
        await pilot.press("/", *"permissions", "enter")
        await pilot.pause()

        assert app.permission_mode is PermissionMode.ALLOW_ALL
        assert "permissions: allow all" in str(app.query_one("#status").render())
        assert await app.approve(
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call",
                name="shell",
                arguments='{"command":"pwd"}',
            )
        )


async def test_approval_can_allow_all() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)
    call = ResponseFunctionToolCall(
        type="function_call",
        call_id="call",
        name="shell",
        arguments='{"command":"pwd"}',
    )

    async with app.run_test() as pilot:
        approval = app.run_worker(app.approve(call))
        await pilot.pause()

        await pilot.click("#allow-all")
        assert await approval.wait()
        assert app.permission_mode is PermissionMode.ALLOW_ALL
        assert "permissions: allow all" in str(app.query_one("#status").render())
        assert await app.approve(call)


async def test_approval_prompts_are_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)
    active = 0
    peak = 0

    async def prompt(_: object) -> bool:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return True

    monkeypatch.setattr(app, "push_screen_wait", prompt)
    calls = [
        ResponseFunctionToolCall(
            type="function_call",
            call_id=f"call-{index}",
            name="shell",
            arguments='{"command":"pwd"}',
        )
        for index in (1, 2)
    ]

    assert all(await asyncio.gather(*(app.approve(call) for call in calls)))
    assert peak == 1


async def test_approval_panel_formats_tool_arguments() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)
    call = ResponseFunctionToolCall(
        type="function_call",
        call_id="call-1",
        name="shell",
        arguments='{"command":"pwd"}',
    )

    async with app.run_test() as pilot:
        await app.push_screen(ApprovalScreen(call))
        await pilot.pause()

        assert str(app.screen.query_one("#approval-title", Label).render()) == "Permission required"
        assert str(app.screen.query_one("#approval-tool", Label).render()) == "shell"
        assert '"command": "pwd"' in str(
            app.screen.query_one("#approval-details", VerticalScroll).query_one(Static).render()
        )
        assert [button.id for button in app.screen.query("#approval-actions Button")] == [
            "deny",
            "allow",
            "allow-all",
        ]


async def test_resume_switches_state_and_commit_target(tmp_path: Path) -> None:
    current = Session.create(tmp_path)
    resumed = Session.create(tmp_path)
    await resumed.commit(resumed.state.append_user("earlier message"))
    resumed_path = resumed.info.path
    current_path = current.info.path
    resumed_created_at = resumed.info.created_at
    agent = Agent(
        client=fake_client("hello"),
        model="test",
        instructions="test",
        state=current.state,
        commit=current.commit,
    )
    app = AgentTUI(agent=agent, session=current)

    async with app.run_test() as pilot:
        app.action_resume()
        await pilot.pause()
        options = app.screen.query_one(OptionList)
        label = str(options.get_option_at_index(0).prompt)
        assert resumed.info.title in label
        assert str(resumed_created_at.year) in label
        assert resumed_path.name not in label
        await pilot.press("escape")

        await app.resume(resumed)
        transcript = app.query_one(Transcript)
        assert not transcript.query(".banner")
        assert "earlier message" in transcript.query_one(".session-history", Markdown).source
        entry = agent.state.append_user("new message")
        assert agent.commit is not None
        await agent.commit(entry)

        assert app.session is resumed
        assert str(app.query_one("#status", Static).render()).startswith("idle ·")
        assert "earlier message" in str(agent.state.context())
        assert "new message" in resumed_path.read_text()
        assert not current_path.exists()


async def test_resume_reports_an_unreadable_session(tmp_path: Path) -> None:
    current = Session.create(tmp_path)
    damaged = Session.create(tmp_path)
    await damaged.commit(damaged.state.append_user("damaged"))
    with damaged.info.path.open("a") as file:
        file.write('{"type":"items","id":2}\n')
    agent = Agent(
        client=fake_client("hello"),
        model="test",
        instructions="test",
        state=current.state,
        commit=current.commit,
    )
    app = AgentTUI(agent=agent, session=current)

    async with app.run_test() as pilot:
        app.action_resume()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert "session error" in str(app.query_one("#status").render())
        error = app.query_one(Transcript).query(".entry").last()
        assert "Cannot open session" in str(error.render())


async def test_resumed_session_is_rendered_on_start(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("earlier message"))
    await session.commit(
        session.state.append_items(
            [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "shell",
                    "arguments": '{"command":"pwd"}',
                }
            ]
        )
    )
    await session.commit(
        session.state.append_items(
            [{"type": "function_call_output", "call_id": "call-1", "output": "/tmp"}]
        )
    )
    await session.commit(
        session.state.append_items(
            [
                {
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "shell",
                    "arguments": '{"command":"git status"}',
                }
            ]
        )
    )
    await session.commit(
        session.state.append_items(
            [{"type": "function_call_output", "call_id": "call-2", "output": "clean"}]
        )
    )
    usage = ResponseUsage(
        input_tokens=120,
        input_tokens_details={"cached_tokens": 80, "cache_write_tokens": 0},
        output_tokens=30,
        output_tokens_details={"reasoning_tokens": 10},
        total_tokens=150,
    )
    await session.commit(
        session.state.append_items(
            [{"role": "assistant", "content": []}],
            response=ResponseMetadata(id="response", model="test", usage=usage),
        )
    )
    session = Session.open(session.info.path)
    agent = Agent(
        client=fake_client("hello"),
        model="test",
        instructions="test",
        state=session.state,
        commit=session.commit,
    )

    app = AgentTUI(agent=agent, session=session)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert any(
            "earlier message" in str(widget.render())
            for widget in app.query_one(Transcript).query(Static)
        )
        transcript = app.query_one(Transcript)
        tools = list(transcript.query(Collapsible))
        assert [tool.title for tool in tools] == ["✓  shell  pwd", "✓  shell  git status"]
        assert all(tool.collapsed for tool in tools)
        assert tools[1].region.y == tools[0].region.bottom
        assert "/tmp" in str(tools[0].query_one(".tool-detail-content", Static).render())
        assert not transcript.query(".assistant-label")
        assert "context 80/150/?" in str(app.query_one("#status", Static).render())


async def test_resume_stays_at_bottom_while_history_finishes_layout(tmp_path: Path) -> None:
    current = Session.create(tmp_path)
    resumed = Session.create(tmp_path)
    for index in range(30):
        await resumed.commit(resumed.state.append_user(f"message {index}"))
    agent = Agent(
        client=fake_client("hello"),
        model="test",
        instructions="test",
        state=current.state,
        commit=current.commit,
    )
    app = AgentTUI(agent=agent, session=current)

    async with app.run_test() as pilot:
        await app.resume(resumed)
        await pilot.pause()
        transcript = app.query_one(Transcript)
        previous_max = transcript.max_scroll_y

        await transcript.mount(Static("\n".join(f"late line {index}" for index in range(30))))
        await pilot.pause()

        assert transcript.max_scroll_y > previous_max
        assert transcript.scroll_y == transcript.max_scroll_y


async def test_escape_cancels_the_active_agent_run() -> None:
    gate = asyncio.Event()
    agent = Agent(client=fake_client("hello", gate), model="test", instructions="test")
    app = AgentTUI(agent=agent)

    async with app.run_test() as pilot:
        await pilot.click("#composer")
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        assert agent.is_running

        app.action_resume()
        await pilot.pause()
        assert agent.is_running

        await pilot.press("escape")
        await pilot.pause()

        assert not agent.is_running
        assert "cancelled" in str(app.query_one("#status").render())
