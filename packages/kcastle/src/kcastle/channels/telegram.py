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
import logging
from typing import TYPE_CHECKING, Any

from kagent import (
    AgentError,
    AgentEvent,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
)
from kai import TextDeltaEvent

if TYPE_CHECKING:
    from kcastle.castle import Castle

_log = logging.getLogger("kcastle.channels.telegram")


def _session_id_for_chat(chat_type: str, chat_id: int, user_id: int | None) -> str:
    """Derive a deterministic session ID from Telegram chat context."""
    if chat_type == "private":
        return f"tg-u{user_id}" if user_id else f"tg-u{chat_id}"
    # group / supergroup
    return f"tg-g{chat_id}"


def _render_events_to_text(events: list[AgentEvent]) -> str:
    """Render collected agent events to a single Markdown text."""
    parts: list[str] = []
    for event in events:
        match event:
            case StreamChunk(event=stream_event):
                if isinstance(stream_event, TextDeltaEvent):
                    parts.append(stream_event.delta)
            case ToolExecStart(tool_name=name):
                parts.append(f"\n⚙ _{name}_")
            case ToolExecEnd(tool_name=name, is_error=is_err, duration_ms=dur):
                status = "✗" if is_err else "✓"
                parts.append(f" {status} ({dur:.0f}ms)\n")
            case AgentError(error=err):
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

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self, castle: Castle) -> None:
        """Start the Telegram bot (long-polling)."""
        try:
            from telegram.ext import (  # type: ignore[import-untyped]
                Application,  # pyright: ignore[reportUnknownVariableType]
                CallbackQueryHandler,  # pyright: ignore[reportUnknownVariableType]
                CommandHandler,  # pyright: ignore[reportUnknownVariableType]
                MessageHandler,  # pyright: ignore[reportUnknownVariableType]
                filters,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError:
            _log.error(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )
            return

        self._castle = castle

        self._app = Application.builder().token(self._token).build()  # type: ignore[reportUnknownMemberType]

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))  # type: ignore[reportUnknownMemberType]
        self._app.add_handler(CommandHandler("new", self._cmd_new))  # type: ignore[reportUnknownMemberType]
        self._app.add_handler(CommandHandler("model", self._cmd_model))  # type: ignore[reportUnknownMemberType]
        self._app.add_handler(CommandHandler("sessions", self._cmd_sessions))  # type: ignore[reportUnknownMemberType]
        self._app.add_handler(CommandHandler("help", self._cmd_help))  # type: ignore[reportUnknownMemberType]
        self._app.add_handler(CallbackQueryHandler(self._on_model_selected, pattern=r"^model:"))  # type: ignore[reportUnknownMemberType]
        self._app.add_handler(  # type: ignore[reportUnknownMemberType]
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)  # type: ignore[reportUnknownMemberType]
        )

        _log.info("Starting Telegram bot")
        await self._app.initialize()  # type: ignore[reportUnknownMemberType]
        await self._app.start()  # type: ignore[reportUnknownMemberType]

        # Register command menu so Telegram shows suggestions on "/"
        from telegram import BotCommand  # type: ignore[import-untyped]

        await self._app.bot.set_my_commands(  # type: ignore[reportUnknownMemberType]
            [
                BotCommand("new", "Start a new session"),
                BotCommand("model", "Switch model"),
                BotCommand("sessions", "List all sessions"),
                BotCommand("help", "Show available commands"),
            ]
        )

        await self._app.updater.start_polling()  # type: ignore[union-attr]

        # Keep running until stopped
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app is not None:
            _log.info("Stopping Telegram bot")
            await self._app.updater.stop()  # type: ignore[union-attr]
            await self._app.stop()
            await self._app.shutdown()

    # --- Handlers ---

    async def _cmd_start(self, update: Any, context: Any) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "👋 Hi! I'm k, your AI assistant. Send me a message to start chatting.\n\n"
            "Commands:\n"
            "/new — Start a new session\n"
            "/model — Switch model\n"
            "/sessions — List all sessions\n"
            "/help — Show available commands"
        )

    async def _cmd_help(self, update: Any, context: Any) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "Available commands:\n\n"
            "/new — Start a new session (clears context)\n"
            "/model — Switch model\n"
            "/sessions — List all sessions\n"
            "/help — Show this help message\n\n"
            "Just send any message to chat with me."
        )

    async def _cmd_new(self, update: Any, context: Any) -> None:
        """Handle /new — create a fresh session."""
        if self._castle is None:
            return
        chat = update.effective_chat
        user = update.effective_user
        sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)
        manager = self._castle.session_manager
        manager.suspend(sid)
        name = " ".join(context.args) if context.args else ""
        manager.create(session_id=sid, name=name)
        await update.message.reply_text("✓ New session started.")

    async def _cmd_sessions(self, update: Any, context: Any) -> None:
        """Handle /sessions — list sessions."""
        if self._castle is None:
            return
        chat = update.effective_chat
        user = update.effective_user
        current_sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)
        manager = self._castle.session_manager
        sessions = manager.list()
        if not sessions:
            await update.message.reply_text("No sessions.")
            return
        lines: list[str] = []
        for s in sessions[:10]:
            marker = " (current)" if s.id == current_sid else ""
            name_str = f" — {s.name}" if s.name else ""
            lines.append(f"• `{s.id}`{name_str}{marker}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_model(self, update: Any, context: Any) -> None:
        """Handle /model — show model selection keyboard."""
        if self._castle is None:
            return

        from telegram import (  # type: ignore[import-untyped]
            InlineKeyboardButton,
            InlineKeyboardMarkup,
        )

        models = self._castle.available_models()
        if not models:
            await update.message.reply_text("No models available (check API keys).")
            return

        current_provider = self._castle.active_provider_name
        current_model = self._castle.active_model

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

        try:
            self._castle.switch_model(provider_name, model_id)
            text = f"✓ Switched to *{model_id}* ({provider_name})"
            await query.edit_message_text(text, parse_mode="Markdown")
        except ValueError as e:
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

            # Strip bot mention from text
            if self._bot_username:
                message_text = message_text.replace(f"@{self._bot_username}", "").strip()

            # Add sender info for group context
            sender_name = user.full_name if user else "Unknown"
            message_text = f"[{sender_name}]: {message_text}"

        # Resolve session
        sid = _session_id_for_chat(chat.type, chat.id, user.id if user else None)
        manager = self._castle.session_manager
        session = manager.get_or_create(sid)

        # Show "typing..." indicator while the agent is working
        typing_task = asyncio.create_task(self._send_typing(chat.id))

        # Run agent and collect events
        collected_events: list[Any] = []
        try:
            async for event in session.run(message_text):
                collected_events.append(event)
        except Exception as e:
            _log.exception("Error in session %s", sid)
            await update.message.reply_text(f"\u274c Error: {e}")
            return
        finally:
            typing_task.cancel()

        # Render and send response
        response = _render_events_to_text(collected_events)  # pyright: ignore[reportUnknownArgumentType]
        if response:
            await self._send_markdown(update.message, response)

    async def _send_markdown(self, message: Any, text: str) -> None:
        """Send *text* as MarkdownV2 via ``telegramify-markdown``.

        Falls back to plain text if the library is unavailable or
        the conversion / send fails.
        """
        converted: str | None = None
        try:
            from telegramify_markdown import markdownify  # type: ignore[import-untyped]

            converted = str(markdownify(text))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        except ImportError:
            _log.debug("telegramify-markdown not installed — sending plain text")
        except Exception:
            _log.debug("Markdown conversion failed — sending plain text", exc_info=True)

        # Try MarkdownV2 first, fall back to plain text
        if converted:
            try:
                for i in range(0, len(converted), 4096):
                    chunk = converted[i : i + 4096]
                    await message.reply_text(chunk, parse_mode="MarkdownV2")
                return
            except Exception:
                _log.debug("MarkdownV2 send failed — falling back to plain text", exc_info=True)

        # Fallback: plain text
        for i in range(0, len(text), 4096):
            chunk = text[i : i + 4096]
            await message.reply_text(chunk)

    async def _send_typing(self, chat_id: int) -> None:
        """Continuously send 'typing' action until cancelled.

        Telegram typing indicator expires after ~5 seconds, so we
        re-send it every 4 seconds to keep it alive.
        """
        try:
            from telegram.constants import ChatAction  # type: ignore[import-untyped]

            while True:
                await self._app.bot.send_chat_action(  # type: ignore[union-attr]
                    chat_id=chat_id,
                    action=ChatAction.TYPING,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                )
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception:
            _log.debug("Typing indicator error (non-fatal)", exc_info=True)
