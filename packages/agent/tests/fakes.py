"""Small OpenAI Responses fakes for harness tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall


class FakeItem:
    def __init__(
        self,
        item_type: str,
        *,
        text: str = "",
        call_id: str = "",
        name: str = "",
        arguments: str = "{}",
    ) -> None:
        self.type = item_type
        self.text = text
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

    def model_dump(self, **_: object) -> dict[str, object]:
        if self.type == "function_call":
            return {
                "type": self.type,
                "call_id": self.call_id,
                "name": self.name,
                "arguments": self.arguments,
            }
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": self.text}],
        }


class FakeResponse:
    def __init__(
        self,
        response_id: str,
        text: str,
        items: list[FakeItem | ResponseFunctionToolCall] | None = None,
    ) -> None:
        self.id = response_id
        self.output_text = text
        self.output = items if items is not None else [FakeItem("message", text=text)]
        self.usage = None


@dataclass(slots=True)
class FakeEvent:
    type: str
    response: FakeResponse | None = None
    delta: str = ""


class FakeStream:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    async def __aiter__(self) -> AsyncIterator[FakeEvent]:
        if self.response.output_text:
            yield FakeEvent("response.output_text.delta", delta=self.response.output_text)
        yield FakeEvent("response.completed", response=self.response)


class FakeResponses:
    def __init__(self, responses: list[FakeResponse], *, summary: str = "summary") -> None:
        self.responses = list(responses)
        self.summary = summary
        self.requests: list[dict[str, object]] = []

    async def create(self, **request: object) -> object:
        self.requests.append(request)
        if request.get("stream") is True:
            if not self.responses:
                raise AssertionError("no fake response remaining")
            return FakeStream(self.responses.pop(0))
        return FakeResponse("compaction", self.summary)


class FakeClient:
    def __init__(self, responses: list[FakeResponse], *, summary: str = "summary") -> None:
        self.responses = FakeResponses(responses, summary=summary)

    def as_openai(self) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cast(object, self))
