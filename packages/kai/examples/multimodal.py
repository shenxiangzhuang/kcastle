"""Multimodal input — sending images to the model.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/multimodal.py
"""

import asyncio
import os

from kai import Context, ImagePart, Message, OpenAIChatCompletions, TextPart, complete


async def main() -> None:
    # provider = OpenAIChatCompletions(model="gpt-4o")
    provider = OpenAIChatCompletions(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # From file (replace with a real image path)
    # image_data = base64.b64encode(Path("photo.png").read_bytes()).decode()

    # For demo: a tiny 1x1 red PNG
    image_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "2mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )

    message = Message(
        role="user",
        content=[
            TextPart(text="What do you see in this image?"),
            ImagePart(data=image_data, mime_type="image/png"),
        ],
    )

    context = Context(messages=[message])
    response = await complete(provider, context)
    print(response.extract_text())


if __name__ == "__main__":
    asyncio.run(main())
