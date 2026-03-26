"""Telegram channel — bot that auto-creates sessions per chat.

Works in both private chats and group chats.  Session IDs are derived
deterministically from the Telegram chat context:

- Private: ``tg-u{user_id}``
- Group:   ``tg-g{chat_id}``

Responds in groups only when mentioned or replied to.

Requires the ``python-telegram-bot`` package (``pip install python-telegram-bot``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from kagent import (
    AgentError,
    AgentEvent,
    StreamChunk,
)
from kai import TextDelta
from kai.errors import KaiError
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegramify_markdown import markdownify  # pyright: ignore[reportMissingTypeStubs]

from kcastle.log import logger

if TYPE_CHECKING:
    from kcastle.castle import Castle


def _session_id_for_chat(chat_type: str, chat_id: int, user_id: int | None) -> str:
    """Derive a deterministic session ID from Telegram chat context."""
    if chat_type == "private":
        return f"tg-u{user_id}" if user_id else f"tg-u{chat_id}"
    return f"tg-g{chat_id}"


def _render_events_to_text(events: list[AgentEvent]) -> str:
    """Render collected agent events to a single Markdown text.

    Only the final assistant text is included; tool-execution
    indicators are omitted so that the Telegram user sees only
    the final reply.
    """
    parts: list[str] = []
    for event in events:
        match event:
            case StreamChunk(event=stream_event):
                if isinstance(stream_event, TextDelta):
                    parts.append(stream_event.delta)
            case AgentError(error=err):
                error_msg = str(err)
                # Check for content moderation errors
                if "Content Exists Risk" in error_msg:
                    parts.append(
                        "\n⚠️ The AI provider blocked this request due to content moderation. "
                        "This may happen with certain topics or conversation contexts. "
                        "Try rephrasing your message or starting a new conversation with /new."
                    )
                else:
                    parts.append(f"\n❌ Error: {err}")
            case _:
                pass
    return "".join(parts).strip()


class TelegramChannel:
    """Telegram bot channel using ``python-telegram-bot``."""

    def __init__(self, *, token: str, bot_username: str = "") -> None:
        self._token = token
        self._bot_username = bot_username
        self._castle: Castle | None = None
        self._app: Any = None  # telegram.ext.Application
        self._active_sessions: dict[str, str] = {}  # Maps base_sid -> active_sid

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self, castle: Castle) -> None:
        """Start the Telegram bot (long-polling)."""
        self._castle = castle

        self._app = Application.builder().token(self._token).build()

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("new", self._cmd_new))
        self._app.add_handler(CommandHandler("switch", self._cmd_switch))
        self._app.add_handler(CommandHandler("model", self._cmd_model))
        self._app.add_handler(CommandHandler("sessions", self._cmd_sessions))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CallbackQueryHandler(self._on_model_selected, pattern=r"^model:"))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))

        logger.info("Starting Telegram bot")
        await self._app.initialize()
        await self._app.start()

        await self._app.bot.set_my_commands(
            [
                BotCommand("new", "Start a new session"),
                BotCommand("switch", "Switch to a different session"),
                BotCommand("model", "Switch model"),
                BotCommand("sessions", "List all sessions"),
                BotCommand("help", "Show available commands"),
            ]
        )

        await self._app.updater.start_polling()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app is not None:
            logger.info("Stopping Telegram bot")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def _cmd_start(self, update: Any, context: Any) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "👋 Hi! I'm k, your AI assistant. Send me a message to start chatting.\n\n"
            "Commands:\n"
            "/new — Start a new session\n"
            "/sessions — List all sessions\n"
            "/switch <id> — Switch to a session\n"
            "/model — Switch model\n"
            "/help — Show available commands"
        )

    async def _cmd_help(self, update: Any, context: Any) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "Available commands:\n\n"
            "/new [name] — Start a new session (clears context)\n"
            "/sessions — List all your sessions\n"
            "/switch <id> — Switch to a different session\n"
            "/model — Switch model for current session\n"
            "/help — Show this help message\n\n"
            "Just send any message to chat with me."
        )

    async def _cmd_new(self, update: Any, context: Any) -> None:
        """Handle /new — create a fresh session."""
        if self._castle is None:
            return
        chat = update.effective_chat
        user = update.effective_user
        base_sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)

        # Find the current session and suspend it
        manager = self._castle.session_manager
        current_sessions = [s for s in manager.list() if s.id.startswith(base_sid)]
        for session in current_sessions:
            manager.suspend(session.id)

        # Generate a new unique session ID with timestamp
        import time
        new_sid = f"{base_sid}-{int(time.time())}"

        name = " ".join(context.args) if context.args else ""
        manager.create(session_id=new_sid, name=name)

        # Store the active session mapping
        self._active_sessions[base_sid] = new_sid

        await update.message.reply_text(f"✓ New session started: {new_sid}")

    async def _cmd_switch(self, update: Any, context: Any) -> None:
        """Handle /switch — switch to a different session."""
        if self._castle is None:
            return

        if not context.args:
            await update.message.reply_text(
                "Usage: `/switch <session-id>`\n"
                "Use `/sessions` to see available sessions.",
                parse_mode="Markdown"
            )
            return

        target_sid = context.args[0]
        chat = update.effective_chat
        user = update.effective_user
        base_sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)

        # Check if the session exists and belongs to this user
        manager = self._castle.session_manager
        sessions = manager.list()
        user_sessions = [s for s in sessions if s.id.startswith(base_sid)]

        if not any(s.id == target_sid for s in user_sessions):
            await update.message.reply_text(
                f"❌ Session `{target_sid}` not found or doesn't belong to you.\n"
                f"Use `/sessions` to see your available sessions.",
                parse_mode="Markdown"
            )
            return

        # Update the active session mapping
        self._active_sessions[base_sid] = target_sid
        await update.message.reply_text(f"✓ Switched to session: `{target_sid}`", parse_mode="Markdown")

    async def _cmd_sessions(self, update: Any, context: Any) -> None:
        """Handle /sessions — list sessions."""
        if self._castle is None:
            return
        chat = update.effective_chat
        user = update.effective_user
        base_sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)
        current_sid = self._active_sessions.get(base_sid, base_sid)

        manager = self._castle.session_manager
        sessions = manager.list()

        # Filter sessions for this user/chat
        user_sessions = [s for s in sessions if s.id.startswith(base_sid)]

        if not user_sessions:
            await update.message.reply_text("No sessions for this chat.")
            return

        lines: list[str] = []
        for s in user_sessions[:10]:
            marker = " ⬅️ (active)" if s.id == current_sid else ""
            name_str = f" — {s.name}" if s.name else ""
            # Extract timestamp from session ID if present
            if "-" in s.id and s.id != base_sid:
                timestamp = s.id.split("-")[-1]
                from datetime import datetime
                try:
                    dt = datetime.fromtimestamp(int(timestamp))
                    time_str = dt.strftime(" [%Y-%m-%d %H:%M]")
                except:
                    time_str = ""
            else:
                time_str = " [default]"

            lines.append(f"• `{s.id}`{name_str}{time_str}{marker}")

        await update.message.reply_text(
            f"**Your sessions:**\n\n" + "\n".join(lines) +
            f"\n\nUse `/switch <session-id>` to switch sessions\n"
            f"Use `/new [name]` to create a new session",
            parse_mode="Markdown"
        )

    async def _cmd_model(self, update: Any, context: Any) -> None:
        """Handle /model — show model selection keyboard."""
        if self._castle is None:
            return

        chat = update.effective_chat
        user = update.effective_user
        base_sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)
        sid = self._active_sessions.get(base_sid, base_sid)

        self._castle.session_manager.get_or_create(sid)

        models = self._castle.available_models()
        if not models:
            await update.message.reply_text("No models available (check API keys).")
            return

        current_provider, current_model = self._castle.get_active_model(sid)

        # Build inline keyboard — one button per model
        buttons: list[list[Any]] = []
        for provider_name, model_id in models:
            is_current = provider_name == current_provider and model_id == current_model
            label = f"{'✓ ' if is_current else ''}{model_id} ({provider_name})"
            callback_data = f"model:{provider_name}:{model_id}"
            buttons.append([InlineKeyboardButton(label, callback_data=callback_data)])

        header = f"Current: *{current_model}* ({current_provider})\n\nSelect a model:"
        await update.message.reply_text(
            header,
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode="Markdown",
        )

    async def _on_model_selected(self, update: Any, context: Any) -> None:
        """Handle inline keyboard callback for model selection."""
        if self._castle is None:
            return

        query = update.callback_query
        await query.answer()

        # Parse callback data: "model:<provider>:<model_id>"
        data: str = query.data or ""
        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        _, provider_name, model_id = parts

        chat = update.effective_chat
        user = update.effective_user
        base_sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)
        sid = self._active_sessions.get(base_sid, base_sid)

        self._castle.session_manager.get_or_create(sid)

        try:
            self._castle.switch_model(provider_name, model_id, session_id=sid)
            text = f"✓ Session model switched to *{model_id}* ({provider_name})"
            await query.edit_message_text(text, parse_mode="Markdown")
        except ValueError as e:
            await query.edit_message_text(f"✗ {e}")
        except KeyError as e:
            await query.edit_message_text(f"✗ {e}")

    async def _on_message(self, update: Any, context: Any) -> None:
        """Handle user messages."""
        if self._castle is None or update.message is None:
            return

        chat = update.effective_chat
        user = update.effective_user
        message_text: str = update.message.text or ""

        if not message_text.strip():
            return

        # In groups, only respond when mentioned or replied to
        if chat.type in ("group", "supergroup"):
            is_reply = (
                update.message.reply_to_message is not None
                and update.message.reply_to_message.from_user is not None
                and update.message.reply_to_message.from_user.is_bot
            )
            is_mentioned = self._bot_username and f"@{self._bot_username}" in message_text
            if not is_reply and not is_mentioned:
                return

            if self._bot_username:
                message_text = message_text.replace(f"@{self._bot_username}", "").strip()

            sender_name = user.full_name if user else "Unknown"
            message_text = f"[{sender_name}]: {message_text}"

        # Get the active session for this user/chat
        base_sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)

        # If no active session is tracked, find the most recent one
        if base_sid not in self._active_sessions:
            manager = self._castle.session_manager
            sessions = manager.list()
            user_sessions = [s for s in sessions if s.id.startswith(base_sid)]
            if user_sessions:
                # Use the most recently active session
                latest = max(user_sessions, key=lambda s: s.last_active_at or 0)
                self._active_sessions[base_sid] = latest.id

        sid = self._active_sessions.get(base_sid, base_sid)

        manager = self._castle.session_manager
        session = manager.get_or_create(sid)

        typing_task = asyncio.create_task(self._send_typing(chat.id))

        collected_events: list[Any] = []
        try:
            prepared_input = self._castle.prepare_user_input(message_text)
            async for event in session.run(prepared_input):
                collected_events.append(event)
        except (KaiError, RuntimeError, ValueError, KeyError, OSError) as e:
            logger.exception("Error in session %s", sid)
            error_msg = str(e)

            # Check for content moderation errors
            if "Content Exists Risk" in error_msg:
                await update.message.reply_text(
                    "\u26a0\ufe0f The AI provider blocked this request due to content moderation. "
                    "This may happen with certain topics or conversation contexts. "
                    "Try rephrasing your message or starting a new conversation with /new."
                )
            else:
                await update.message.reply_text(f"\u274c Error: {e}")
            return
        finally:
            typing_task.cancel()

        response = _render_events_to_text(collected_events)
        if response:
            await self._send_markdown(update.message, response)

    async def _send_markdown(self, message: Any, text: str) -> None:
        """Send *text* as MarkdownV2 via ``telegramify-markdown``.

        Falls back to plain text if the library is unavailable or
        the conversion / send fails.
        """
        converted: str | None = None
        try:
            converted = str(markdownify(text))
        except (TypeError, ValueError, RuntimeError):
            logger.debug("Markdown conversion failed — sending plain text", exc_info=True)

        if converted:
            try:
                for i in range(0, len(converted), 4096):
                    chunk = converted[i : i + 4096]
                    await message.reply_text(chunk, parse_mode="MarkdownV2")
                return
            except (TelegramError, OSError, RuntimeError, ValueError):
                logger.debug("MarkdownV2 send failed — falling back to plain text", exc_info=True)

        for i in range(0, len(text), 4096):
            chunk = text[i : i + 4096]
            await message.reply_text(chunk)

    async def _send_typing(self, chat_id: int) -> None:
        """Continuously send 'typing' action until cancelled.

        Telegram typing indicator expires after ~5 seconds, so we
        re-send it every 4 seconds to keep it alive.
        """
        try:
            while True:
                await self._app.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                )
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except (TelegramError, OSError, RuntimeError):
            logger.debug("Typing indicator error (non-fatal)", exc_info=True)
