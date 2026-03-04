"""Simple agent loop — multi-round tool calling until the model gives a final answer.

Demonstrates a minimal agent loop: ask the model a question with tools available,
execute any tool calls, feed results back, and repeat until the model responds
with text only (no more tool calls). Uses the `rich` library for terminal rendering.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/agent_loop.py
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from kai import (
    Context,
    Done,
    Error,
    Message,
    OpenAIChatCompletions,
    TextDelta,
    ThinkDelta,
    Tool,
    ToolCallBegin,
    ToolCallEnd,
    ToolResult,
    stream,
)

console = Console()

# ---------------------------------------------------------------------------
# Tool definitions — locally executable, no network required
# ---------------------------------------------------------------------------


class GetSystemInfo(Tool):
    """Get current operating system information."""

    name: str = "get_system_info"
    description: str = "Get current operating system information (OS name, version, architecture)."

    async def execute(self, params: object) -> ToolResult:
        return ToolResult(
            output=json.dumps(
                {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                }
            )
        )


class GetPythonVersion(Tool):
    """Get the Python interpreter version and executable path."""

    name: str = "get_python_version"
    description: str = "Get the Python interpreter version and executable path."

    async def execute(self, params: object) -> ToolResult:
        return ToolResult(
            output=json.dumps(
                {
                    "version": sys.version,
                    "executable": sys.executable,
                    "implementation": platform.python_implementation(),
                }
            )
        )


class GetEnvVariable(Tool):
    """Read the value of an environment variable."""

    name: str = "get_env_variable"
    description: str = "Read the value of an environment variable."

    class Params(BaseModel):
        name: str = Field(description="Environment variable name")

    async def execute(self, params: GetEnvVariable.Params) -> ToolResult:
        value = os.environ.get(params.name)
        if value is None:
            return ToolResult.error(f"Variable '{params.name}' is not set")
        return ToolResult(output=json.dumps({"name": params.name, "value": value}))


TOOLS: list[Tool] = [GetSystemInfo(), GetPythonVersion(), GetEnvVariable()]


TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}


async def execute_tool_call(name: str, arguments: str) -> str:
    """Parse arguments and execute a tool."""
    tool = TOOL_MAP.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    args: dict[str, object] = json.loads(arguments) if arguments.strip() else {}
    result = await tool.execute(args)
    return result.output


# ---------------------------------------------------------------------------
# Stream one round with rich Live display
# ---------------------------------------------------------------------------


async def stream_round(
    provider: OpenAIChatCompletions,
    messages: list[Message],
    round_num: int,
) -> Message | None:
    """Stream one LLM round, render output with rich, return the done message."""
    console.rule(f"[bold blue]Round {round_num}[/bold blue]")

    done_msg: Message | None = None
    tool_results: list[Message] = []
    text_buffer = ""
    think_buffer = ""

    async for event in stream(provider, Context(messages=messages, tools=TOOLS)):
        match event:
            # --- Thinking ---
            case ThinkDelta(delta=text):
                think_buffer += text

            # --- Text ---
            case TextDelta(delta=text):
                # Flush think buffer when text starts
                if think_buffer and not text_buffer:
                    think_preview = think_buffer[:200] + ("..." if len(think_buffer) > 200 else "")
                    console.print(
                        Panel(
                            Text(think_preview, style="dim italic"),
                            title="Thinking",
                            subtitle=f"{len(think_buffer)} chars",
                            border_style="bright_black",
                        )
                    )
                    think_buffer = ""
                text_buffer += text

            # --- Tool calls ---
            case ToolCallBegin(name=name):
                console.print(
                    f"  [bold yellow]> calling[/bold yellow] [cyan]{name}[/cyan]",
                    end="",
                )
            case ToolCallEnd():
                pass

            # --- Done / Error ---
            case Done(message=msg):
                done_msg = msg
                # Execute tool calls from the done message
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        result = await execute_tool_call(tc.name, tc.arguments)
                        args_short = tc.arguments[:80] + ("..." if len(tc.arguments) > 80 else "")
                        console.print(f"({args_short})")
                        result_short = result[:120] + ("..." if len(result) > 120 else "")
                        console.print(f"    [dim]→ {result_short}[/dim]")
                        tool_results.append(Message.tool_result(tc.id, result))
            case Error(error=err):
                console.print(f"[bold red]Error:[/bold red] {err}")

    # Render assistant text
    if text_buffer:
        console.print(Panel(Markdown(text_buffer), title="Assistant", border_style="green"))

    # Show usage
    if done_msg and done_msg.usage:
        u = done_msg.usage
        usage_text = Text(
            f"tokens: {u.input_tokens} in + {u.output_tokens} out = {u.total_tokens} total",
            style="dim",
        )
        console.print(usage_text)

    # Append assistant message + tool results for next round
    if done_msg is not None and tool_results:
        messages.append(done_msg)
        messages.extend(tool_results)

    return done_msg


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def agent_loop(question: str) -> None:
    """Run the agent loop: ask → tool calls → respond, repeat until done."""
    # provider = OpenAIChatCompletions(model="gpt-4o")
    provider = OpenAIChatCompletions(
        model="deepseek-reasoner",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    console.print(Panel(question, title="User", border_style="bright_blue"))

    messages: list[Message] = [Message(role="user", content=question)]

    round_num = 0
    while True:
        round_num += 1
        done_msg = await stream_round(provider, messages, round_num)

        if done_msg is None:
            console.print("[bold red]Stream ended without a done message.[/bold red]")
            break

        # If model made tool calls, results are already appended — loop again
        if done_msg.tool_calls:
            continue

        # No more tool calls — we're done
        break

    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    question = (
        "Tell me about my current system environment: "
        "what OS am I running, what Python version, and what is my HOME path?"
    )
    await agent_loop(question)


if __name__ == "__main__":
    asyncio.run(main())
