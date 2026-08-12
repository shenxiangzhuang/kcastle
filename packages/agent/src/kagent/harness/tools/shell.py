"""Local shell tool."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from pydantic import BaseModel, Field

from kagent.harness.env import Env
from kagent.harness.tools.tool import Tool


class ShellParams(BaseModel):
    command: str = Field(description="Shell command to run in the working directory")
    timeout: float = Field(default=120, gt=0, le=600)


async def run_shell(params: ShellParams, env: Env) -> str:
    process = await asyncio.create_subprocess_shell(
        params.command,
        cwd=env.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=params.timeout)
    except TimeoutError:
        raise TimeoutError(f"command timed out after {params.timeout:g}s") from None
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
    output = stdout.decode(errors="replace")
    prefix = f"exit_code={process.returncode}\n"
    return prefix + output[-(100_000 - len(prefix)) :]


shell_tool = Tool(
    name="shell",
    description="Run a shell command in the current working directory and return combined output.",
    params=ShellParams,
    run=run_shell,
    requires_approval=True,
)
