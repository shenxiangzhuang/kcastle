"""Emoji Reactor Tool - Uses LLM to intelligently select emoji reactions."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from kai import Context, Message, complete

if TYPE_CHECKING:
    from kcastle.castle import Castle

# Telegram Bot API only allows these emojis for bot reactions.
# Using any emoji outside this set causes set_message_reaction to fail.
TELEGRAM_REACTION_EMOJIS: frozenset[str] = frozenset(
    {
        "👍",
        "👎",
        "❤",
        "🔥",
        "🥰",
        "👏",
        "😁",
        "🤔",
        "🤯",
        "😱",
        "🤬",
        "😢",
        "🎉",
        "🤩",
        "🤮",
        "💩",
        "🙏",
        "👌",
        "🕊",
        "🤡",
        "🥱",
        "🥴",
        "😍",
        "🐳",
        "❤\u200d🔥",
        "🌚",
        "🌭",
        "💯",
        "🤣",
        "⚡",
        "🍌",
        "🏆",
        "💔",
        "🤨",
        "😐",
        "🍓",
        "🍾",
        "💋",
        "🖕",
        "😈",
        "😴",
        "😭",
        "🤓",
        "👻",
        "👨\u200d💻",
        "👀",
        "🎃",
        "🙈",
        "😇",
        "😨",
        "🤝",
        "✍",
        "🤗",
        "🫡",
        "🎅",
        "🎄",
        "☃",
        "💅",
        "🤪",
        "🗿",
        "🆒",
        "💘",
        "🙉",
        "🦄",
        "😘",
        "💊",
        "🙊",
        "😎",
        "👾",
        "🤷\u200d♂",
        "🤷",
        "🤷\u200d♀",
        "😡",
    }
)

DEFAULT_REACTION = "👀"


class EmojiReactor:
    """Tool that uses LLM to select appropriate emoji reactions for messages."""

    EMOJI_PROMPT = textwrap.dedent("""
        You are an emoji reaction selector. Given a message, select the most
        appropriate single emoji to react with.

        Consider:
        - The emotion and tone of the message
        - Cultural context (support both English and Chinese)
        - The topic being discussed
        - Whether it's a question, statement, greeting, etc.

        You MUST choose from ONLY these emojis (Telegram allowed reactions):
        - Questions/Thinking: 🤔, 🤨, 👀
        - Happiness/Agreement: 😁, 👍, 💯, 😎, 🆒
        - Excitement: 🎉, 🔥, ⚡, 🤩, ✨ is NOT allowed
        - Gratitude/Warmth: 🙏, 🤗, ❤, 😘, 🥰
        - Technical/Code: 👨‍💻, 🤓, 👾
        - Surprise: 🤯, 😱, 😨
        - Sadness/Concern: 😢, 😭, 💔
        - Humor: 🤣, 🤡, 🌚
        - Greeting/Acknowledgment: 🤗, 👋 is NOT allowed, use 🤗
        - General: 👀, 👏, 🫡

        Respond with ONLY a single emoji from the list above, nothing else.

        Message: {message}
    """).strip()

    def __init__(self, castle: Castle | None = None):
        self._castle = castle

    def set_castle(self, castle: Castle) -> None:
        """Set the castle instance for model access."""
        self._castle = castle

    async def get_reaction(self, message: str, session_id: str | None = None) -> str:
        """Get an appropriate emoji reaction for the given message.

        Args:
            message: The message to react to
            session_id: Optional session ID to use the same model as the conversation

        Returns:
            A single emoji character from the Telegram allowed set
        """
        if not self._castle:
            return self._simple_fallback(message)

        try:
            # Get the provider to use
            if session_id:
                provider_name, model_id = self._castle.get_active_model(session_id)
            else:
                models = self._castle.available_models()
                if not models:
                    raise ValueError("No models available")
                provider_name, model_id = models[0]

            provider = self._castle.model_manager.build_provider(provider_name, model_id)

            prompt = self.EMOJI_PROMPT.format(message=message)
            context = Context(messages=[Message(role="user", content=prompt)])

            response = await complete(
                provider,
                context,
                max_tokens=10,
                temperature=0.7,
            )

            emoji = response.extract_text().strip()

            # Validate against Telegram allowed set
            if emoji in TELEGRAM_REACTION_EMOJIS:
                return emoji

            # LLM returned an invalid emoji — fall back
            return self._simple_fallback(message)

        except Exception as e:
            if self._castle:
                from kcastle.log import logger

                logger.debug("Failed to get emoji reaction: %s", e)
            return self._simple_fallback(message)

    def _simple_fallback(self, message: str) -> str:
        """Simple rule-based fallback using only Telegram-allowed emojis."""
        message_lower = message.lower()

        if "?" in message or "？" in message:
            return "🤔"
        elif any(word in message_lower for word in ["thank", "谢谢", "感谢"]):
            return "🙏"
        elif any(word in message_lower for word in ["hi", "hello", "你好", "嗨"]):
            return "🤗"
        elif "!" in message or "！" in message:
            return "🔥"
        else:
            return DEFAULT_REACTION


# Global instance for convenience
emoji_reactor = EmojiReactor()
