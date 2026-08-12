from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import ClassVar, cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseUsage


class Item:
    type = "message"

    def model_dump(self, **_: object) -> dict[str, object]:
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }


class Response:
    id = "response"
    model = "test"
    output_text = "hello"
    output: ClassVar[list[Item]] = [Item()]
    usage = ResponseUsage(
        input_tokens=120,
        input_tokens_details={"cached_tokens": 80, "cache_write_tokens": 0},
        output_tokens=30,
        output_tokens_details={"reasoning_tokens": 10},
        total_tokens=150,
    )


class Event:
    def __init__(self, event_type: str) -> None:
        self.type = event_type
        self.delta = "hello"
        self.response = Response()


class Stream:
    async def __aiter__(self) -> AsyncIterator[Event]:
        yield Event("response.output_text.delta")
        yield Event("response.completed")


class Responses:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate

    async def create(self, **_: object) -> Stream:
        if self.gate is not None:
            await self.gate.wait()
        return Stream()


class Client:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.responses = Responses(gate)


def fake_client(_: str, gate: asyncio.Event | None = None) -> AsyncOpenAI:
    return cast(AsyncOpenAI, cast(object, Client(gate)))
