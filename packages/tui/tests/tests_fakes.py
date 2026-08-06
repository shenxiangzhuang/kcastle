from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI


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
    output_text = "hello"
    output = [Item()]
    usage = None


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
