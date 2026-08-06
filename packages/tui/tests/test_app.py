import asyncio
from pathlib import Path

from kagent import Agent, CompactionConfig, Env, ResponseMetadata, Session, ToolRuntime
from ktui.app import AgentTUI, PermissionMode, Transcript
from openai.types.responses import ResponseFunctionToolCall, ResponseUsage
from tests_fakes import fake_client
from textual.command import CommandPalette
from textual.widgets import Input, Markdown, OptionList, Static


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
        assert app.focused is composer
        assert composer.outer_size.height == 3
        assert banner_widget.parent is transcript
        assert str(tmp_path) in banner
        assert "model-x" not in banner

        await pilot.press("h", "i", "enter")
        await pilot.pause()

        status = str(app.query_one("#status").render())
        assert "idle" in status
        assert "model-x · context 80/150/50,000" in status
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


async def test_assistant_messages_render_as_markdown() -> None:
    agent = Agent(client=fake_client("hello"), model="test", instructions="test")
    app = AgentTUI(agent=agent)

    async with app.run_test() as pilot:
        transcript = app.query_one(Transcript)
        await transcript.start_assistant()
        await transcript.append_assistant(
            "| name | value |\n| --- | --- |\n| K | 1 |\n\n[OpenAI](https://openai.com)"
        )
        await pilot.pause()

        markdown = transcript.query_one(Markdown)
        assert len(markdown.query("MarkdownTable")) == 1
        assert "https://openai.com" in markdown.source


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
        assert transcript.query_one(".banner", Static) is not None
        assert "earlier message" in transcript.query_one(".session-history", Markdown).source
        entry = agent.state.append_user("new message")
        assert agent.commit is not None
        await agent.commit(entry)

        assert app.session is resumed
        assert "earlier message" in str(agent.state.context())
        assert "new message" in resumed_path.read_text()
        assert not current_path.exists()


async def test_resumed_session_is_rendered_on_start(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("earlier message"))
    usage = ResponseUsage(
        input_tokens=120,
        input_tokens_details={"cached_tokens": 80},
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
        assert "context 80/150/?" in str(app.query_one("#status", Static).render())


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
