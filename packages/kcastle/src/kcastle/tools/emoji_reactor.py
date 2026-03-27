"""Emoji Reactor Tool - Uses LLM to intelligently select emoji reactions."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from kai import Kai
from kai.models import ChatMessage

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
        self._kai: Kai | None = None

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
            # Try to use the same model as the session if provided
            if session_id:
                provider_name, model_id = self._castle.get_active_model(session_id)
                provider = self._castle.get_provider(provider_name)
                kai_instance = Kai(provider=provider) if provider else self._get_default_kai()
            else:
                kai_instance = self._get_default_kai()

            # Create the prompt
            prompt = self.EMOJI_PROMPT.format(message=message)

            # Get emoji from LLM
            response = await kai_instance.completion(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=10,  # We only need one emoji
                temperature=0.7,  # Some creativity but not too random
            )

            # Extract and validate the emoji
            emoji = response.text.strip()

            # Basic validation - should be a single emoji
            if len(emoji) > 4:  # Most emojis are 1-4 bytes
                return "💬"  # Default fallback

            return emoji

        except Exception as e:
            # Log error and return default
            if self._castle:
                from kcastle.log import logger

                logger.debug("Failed to get emoji reaction: %s", e)
            return "💬"  # Safe default

    def _get_default_kai(self) -> Kai:
        """Get a default Kai instance using the castle's primary provider."""
        if not self._castle:
            raise ValueError("Castle not set")

        # Get the first available provider
        models = self._castle.available_models()
        if not models:
            raise ValueError("No models available")

        provider_name, _ = models[0]
        provider = self._castle.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")

        return Kai(provider=provider)

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
