import asyncio
from pathlib import Path

import pytest
from kagent import Env, Tool, ToolRuntime
from kagent.harness.tools import shell_tool
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict


class EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


async def echo(params: EchoParams, _: Env) -> str:
    return params.text


async def test_tool_validates_and_executes() -> None:
    tool = Tool(name="echo", description="Echo text", params=EchoParams, run=echo)
    runtime = ToolRuntime(Env(Path.cwd()), [tool])

    valid = ResponseFunctionToolCall(
        type="function_call",
        call_id="1",
        name="echo",
        arguments='{"text":"hello"}',
    )
    invalid = ResponseFunctionToolCall(
        type="function_call",
        call_id="2",
        name="echo",
        arguments="not json",
    )
    extra = ResponseFunctionToolCall(
        type="function_call",
        call_id="3",
        name="echo",
        arguments='{"text":"hello","extra":true}',
    )

    assert (await runtime.execute(valid)).output == "hello"
    assert (await runtime.execute(invalid)).is_error
    assert (await runtime.execute(extra)).is_error
    assert tool.schema()["type"] == "function"


async def test_shell_timeout_kills_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    class HangingProcess:
        returncode: int | None = None
        killed = False

        async def communicate(self) -> tuple[bytes, None]:
            await asyncio.Event().wait()
            return b"", None

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return -9

    process = HangingProcess()

    async def create_process(*_: object, **__: object) -> HangingProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", create_process)
    call = ResponseFunctionToolCall(
        type="function_call",
        call_id="timeout",
        name="shell",
        arguments='{"command":"sleep", "timeout":0.001}',
    )

    result = await shell_tool.execute(call, Env(Path.cwd()))

    assert result.is_error
    assert process.killed
