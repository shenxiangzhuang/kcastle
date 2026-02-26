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

import logging as _logging

from kcastle.castle import Castle
from kcastle.channels import Channel
from kcastle.config import CastleConfig, ChannelConfig, ModelConfig, ProviderConfig, load_config
from kcastle.session import Session, SessionInfo, SessionManager, SessionMeta, SessionTraceStore
from kcastle.skills import LoadedSkill, SkillManager, SkillMeta, SkillResolver

__all__ = [
    # Core
    "Castle",
    "CastleConfig",
    "ChannelConfig",
    "ModelConfig",
    "ProviderConfig",
    "load_config",
    # Session
    "Session",
    "SessionInfo",
    "SessionManager",
    "SessionMeta",
    "SessionTraceStore",
    # Channel
    "Channel",
    # Skills
    "SkillManager",
    "SkillMeta",
    "SkillResolver",
    "LoadedSkill",
]

_logging.getLogger("kcastle").addHandler(_logging.NullHandler())
