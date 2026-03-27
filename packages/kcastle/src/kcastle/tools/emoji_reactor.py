"""Emoji Reactor Tool - Uses LLM to intelligently select emoji reactions."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from kai import Context, Message, complete

if TYPE_CHECKING:
    from kcastle.castle import Castle


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

        Available emoji categories:
        - Greetings: 👋, 🙋‍♂️, 🙋‍♀️
        - Questions/Curiosity: 🤔, 🧐, ❓, 💭
        - Happiness/Agreement: 😊, 😄, 👍, ✅, 💯
        - Excitement: 🎉, 🎊, ✨, 🌟, 🔥
        - Gratitude: 🙏, 😊, 💝, 🤗
        - Technical/Code: 💻, ⌨️, 🖥️, 🐛, 🔧, 🚀
        - Thinking/Analysis: 🧠, 📊, 📈, 🔍
        - Help/Support: 🤝, 💪, 🆘, 📋
        - Surprise: 😮, 😲, 🤯, 👀
        - Sadness/Concern: 😔, 😟, 💔, 😢
        - General conversation: 💬, 📝, 💡, 🗨️

        Respond with ONLY a single emoji, nothing else.

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
            A single emoji character
        """
        if not self._castle:
            # Simple fallback logic when no LLM available
            return self._simple_fallback(message)

        try:
            # Get the provider to use
            if session_id:
                provider_name, model_id = self._castle.get_active_model(session_id)
            else:
                # Get the first available provider
                models = self._castle.available_models()
                if not models:
                    raise ValueError("No models available")
                provider_name, model_id = models[0]

            # Build the provider instance
            provider = self._castle.model_manager.build_provider(provider_name, model_id)

            # Create the prompt
            prompt = self.EMOJI_PROMPT.format(message=message)

            # Get emoji from LLM using kai's complete function
            context = Context(messages=[Message(role="user", content=prompt)])

            response = await complete(
                provider,
                context,
                max_tokens=10,  # We only need one emoji
                temperature=0.7,  # Some creativity but not too random
            )

            # Extract and validate the emoji
            emoji = response.extract_text().strip()

            # More robust validation
            # Check if it's a reasonable length for an emoji (1-2 grapheme clusters)
            # Most emojis are 1-4 bytes, but some can be longer with modifiers
            if not emoji or len(emoji) > 8 or len(emoji.encode("utf-8")) > 16:
                return "💬"  # Default fallback

            # Additional check: ensure it's not plain text
            if emoji.isalpha() or emoji.isdigit():
                return "💬"  # Default fallback

            return emoji

        except Exception as e:
            # Log error and return default
            if self._castle:
                from kcastle.log import logger

                logger.debug("Failed to get emoji reaction: %s", e)
            return "💬"  # Safe default

    def _simple_fallback(self, message: str) -> str:
        """Simple rule-based fallback when LLM is not available."""
        message_lower = message.lower()

        # Very basic pattern matching
        if "?" in message or "？" in message:
            return "🤔"
        elif any(word in message_lower for word in ["thank", "谢谢"]):
            return "😊"
        elif any(word in message_lower for word in ["hi", "hello", "你好"]):
            return "👋"
        elif "!" in message or "！" in message:
            return "✨"
        else:
            return "💬"


# Global instance for convenience
emoji_reactor = EmojiReactor()
