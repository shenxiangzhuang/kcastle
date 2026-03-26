"""Tests for Telegram multi-session support."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kcastle.channels.telegram import TelegramChannel, _session_id_for_chat
from kcastle.session import SessionInfo


class TestTelegramSessions:
    """Test multi-session support in Telegram channel."""

    def test_session_id_generation(self) -> None:
        """Test that session IDs are generated correctly for different chat types."""
        # Private chat
        sid = _session_id_for_chat("private", chat_id=123, user_id=456)
        assert sid == "tg-u456"

        # Private chat without user_id
        sid = _session_id_for_chat("private", chat_id=123, user_id=None)
        assert sid == "tg-u123"

        # Group chat
        sid = _session_id_for_chat("group", chat_id=789, user_id=456)
        assert sid == "tg-g789"  # Group chats use chat ID, not user ID

    @pytest.mark.asyncio
    async def test_new_command_creates_unique_sessions(self, tmp_path: Path) -> None:
        """Test that /new creates unique sessions with timestamps."""
        # Mock the castle and session manager
        mock_castle = MagicMock()
        mock_manager = MagicMock()
        mock_castle.session_manager = mock_manager
        mock_manager.list.return_value = []

        # Create channel
        channel = TelegramChannel(token="fake-token", bot_username="testbot")
        channel._castle = mock_castle
        channel._active_sessions = {}
        # Mock update and context
        update = MagicMock()
        update.effective_chat.type = "private"
        update.effective_chat.id = 123
        update.effective_user.id = 456
        update.message.reply_text = AsyncMock()

        context = MagicMock()
        context.args = ["test session"]

        # Call _cmd_new
        await channel._cmd_new(update, context)
        # Check that create was called with a unique session ID
        mock_manager.create.assert_called_once()
        call_args = mock_manager.create.call_args
        session_id = call_args.kwargs["session_id"]
        name = call_args.kwargs["name"]

        # Session ID should have timestamp suffix
        assert session_id.startswith("tg-u456-")
        assert len(session_id.split("-")) == 3  # tg-u456-timestamp
        assert name == "test session"

        # Active session should be updated
        assert channel._active_sessions["tg-u456"] == session_id

    @pytest.mark.asyncio
    async def test_switch_command(self, tmp_path: Path) -> None:
        """Test that /switch changes the active session."""
        # Mock castle and manager
        mock_castle = MagicMock()
        mock_manager = MagicMock()
        mock_castle.session_manager = mock_manager

        # Create some mock sessions
        mock_sessions = [
            SessionInfo(
                id="tg-u456",
                name="default",
                created_at=1000,
                last_active_at=1000,
            ),
            SessionInfo(
                id="tg-u456-12345",
                name="session 1",
                created_at=2000,
                last_active_at=3000,
            ),
            SessionInfo(
                id="tg-u456-67890",
                name="session 2",
                created_at=4000,
                last_active_at=5000,
            ),
        ]
        mock_manager.list.return_value = mock_sessions

        # Create channel
        channel = TelegramChannel(token="fake-token", bot_username="testbot")
        channel._castle = mock_castle
        channel._active_sessions = {"tg-u456": "tg-u456"}
        # Mock update and context
        update = MagicMock()
        update.effective_chat.type = "private"
        update.effective_chat.id = 123
        update.effective_user.id = 456
        update.message.reply_text = AsyncMock()

        context = MagicMock()
        context.args = ["tg-u456-12345"]

        # Call _cmd_switch
        await channel._cmd_switch(update, context)
        # Check that active session was updated
        assert channel._active_sessions["tg-u456"] == "tg-u456-12345"
        # Check success message
        update.message.reply_text.assert_called_with(
            "✓ Switched to session: `tg-u456-12345`", parse_mode="Markdown"
        )

    @pytest.mark.asyncio
    async def test_auto_detect_recent_session(self, tmp_path: Path) -> None:
        """Test that the most recent session is auto-detected after restart."""
        # Mock castle and manager
        mock_castle = MagicMock()
        mock_manager = MagicMock()
        mock_castle.session_manager = mock_manager
        mock_castle.prepare_user_input.return_value = "test input"

        # Create mock sessions with different last_active_at
        mock_sessions = [
            SessionInfo(
                id="tg-u456",
                name="old default",
                created_at=1000,
                last_active_at=1000,
            ),
            SessionInfo(
                id="tg-u456-12345",
                name="older session",
                created_at=2000,
                last_active_at=3000,
            ),
            SessionInfo(
                id="tg-u456-67890",
                name="newest session",
                created_at=4000,
                last_active_at=9000,  # Most recent
            ),
        ]
        mock_manager.list.return_value = mock_sessions

        # Mock session with async iterator
        async def mock_run(input_text: str) -> AsyncIterator[object]:
            """Mock run that returns an async iterator."""
            # Return empty iterator - no events
            return
            yield  # Make it a generator

        mock_session = MagicMock()
        mock_session.run = mock_run
        mock_manager.get_or_create.return_value = mock_session

        # Create channel with empty active sessions (simulating restart)
        channel = TelegramChannel(token="fake-token", bot_username="testbot")
        channel._castle = mock_castle
        channel._active_sessions = {}
        # Mock update
        update = MagicMock()
        update.effective_chat.type = "private"
        update.effective_chat.id = 123
        update.effective_user.id = 456
        update.message.text = "Hello"
        update.message.reply_text = AsyncMock()

        context = MagicMock()

        # Mock typing task
        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_task.cancel = MagicMock()
            mock_create_task.return_value = mock_task

            # Call _on_message
            await channel._on_message(update, context)
        # Should use the most recent session
        assert channel._active_sessions["tg-u456"] == "tg-u456-67890"
        mock_manager.get_or_create.assert_called_with("tg-u456-67890")
