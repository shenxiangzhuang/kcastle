"""Simple agent loop — multi-round tool calling until the model gives a final answer.

Demonstrates a minimal agent loop: ask the model a question with tools available,
execute any tool calls, feed results back, and repeat until the model responds
with text only (no more tool calls). Uses the `rich` library for terminal rendering.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/agent_loop.py
"""

import asyncio
import json
import os
import platform
import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from kai import (
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    OpenAICompletions,
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkDeltaEvent,
    ThinkEndEvent,
    ThinkStartEvent,
    Tool,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    stream,
)

console = Console()

# ---------------------------------------------------------------------------
# Tool definitions — locally executable, no network required
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="get_system_info",
        description="Get current operating system information (OS name, version, architecture).",
        parameters={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_python_version",
        description="Get the Python interpreter version and executable path.",
        parameters={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_env_variable",
        description="Read the value of an environment variable.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Environment variable name"},
            },
            "required": ["name"],
        },
    ),
]


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool locally and return a result string."""
    args: dict[str, Any] = json.loads(arguments) if arguments.strip() else {}
    match name:
        case "get_system_info":
            return json.dumps(
                {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                }
            )
        case "get_python_version":
            return json.dumps(
                {
                    "version": sys.version,
                    "executable": sys.executable,
                    "implementation": platform.python_implementation(),
                }
            )
        case "get_env_variable":
            var_name = args.get("name", "")
            value = os.environ.get(var_name)
            if value is None:
                return json.dumps({"error": f"Variable '{var_name}' is not set"})
            return json.dumps({"name": var_name, "value": value})
        case _:
            return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Stream one round with rich Live display
# ---------------------------------------------------------------------------


async def stream_round(
    provider: OpenAICompletions,
    messages: list[Message],
    round_num: int,
) -> Message | None:
    """Stream one LLM round, render output with rich, return the done message."""
    console.rule(f"[bold blue]Round {round_num}[/bold blue]")

    done_msg: Message | None = None
    tool_results: list[Message] = []
    text_buffer = ""
    think_buffer = ""

    async for event in await stream(provider, Context(messages=messages, tools=TOOLS)):
        match event:
            case StartEvent():
                pass

            # --- Thinking ---
            case ThinkStartEvent():
                think_buffer = ""
            case ThinkDeltaEvent(delta=text):
                think_buffer += text
            case ThinkEndEvent():
                if think_buffer:
                    think_preview = think_buffer[:200] + ("..." if len(think_buffer) > 200 else "")
                    console.print(
                        Panel(
                            Text(think_preview, style="dim italic"),
                            title="Thinking",
                            subtitle=f"{len(think_buffer)} chars",
                            border_style="bright_black",
                        )
                    )

            # --- Text ---
            case TextStartEvent():
                text_buffer = ""
            case TextDeltaEvent(delta=text):
                text_buffer += text
            case TextEndEvent():
                pass

            # --- Tool calls ---
            case ToolCallStartEvent(name=name):
                console.print(
                    f"  [bold yellow]> calling[/bold yellow] [cyan]{name}[/cyan]",
                    end="",
                )
            case ToolCallDeltaEvent():
                pass
            case ToolCallEndEvent(tool_call=tc):
                result = execute_tool(tc.name, tc.arguments)
                args_short = tc.arguments[:80] + ("..." if len(tc.arguments) > 80 else "")
                console.print(f"({args_short})")
                result_short = result[:120] + ("..." if len(result) > 120 else "")
                console.print(f"    [dim]→ {result_short}[/dim]")
                tool_results.append(Message.tool_result(tc.id, result))

            # --- Done / Error ---
            case DoneEvent(message=msg):
                done_msg = msg
            case ErrorEvent(error=err):
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
    # provider = OpenAICompletions(model="gpt-4o")
    provider = OpenAICompletions(
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
