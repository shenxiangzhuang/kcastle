"""kcastle — Agent application with multi-channel support.

A general-purpose agent application built on ``kagent``.  kcastle manages
agent sessions and exposes them through multiple channels (CLI, Telegram).

Quick start::

    from kcastle import Castle

    castle = Castle.create()
    await castle.run()

Or from the command line::

    $ k                    # New session
    $ k -C                 # Continue latest session
    $ k -S <id>            # Resume specific session
"""

from kcastle.castle import Castle
from kcastle.channels import Channel
from kcastle.config import CastleConfig, ChannelConfig, ModelConfig, ProviderConfig, load_config
from kcastle.session import Session, SessionInfo, SessionManager, SessionMeta, SessionTraceStore
from kcastle.skills import Skill, SkillManager

__all__ = [
    "Castle",
    "CastleConfig",
    "ChannelConfig",
    "ModelConfig",
    "ProviderConfig",
    "load_config",
    "Session",
    "SessionInfo",
    "SessionManager",
    "SessionMeta",
    "SessionTraceStore",
    "Channel",
    "Skill",
    "SkillManager",
]
