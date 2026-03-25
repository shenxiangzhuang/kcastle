"""kcastle — Agent application with multi-channel support.

A general-purpose agent application built on ``kagent``.  kcastle manages
agent sessions and exposes them through multiple channels (CLI, Telegram).

Quick start::

    from kcastle import Castle

    castle = Castle.create()
    await castle.run()

Or from the command line (``k`` is a shortcut for ``kcastle``)::

    $ k                          # New session
    $ k -C                       # Continue latest session
    $ k -S <id>                  # Resume specific session
"""

from kcastle.castle import Castle
from kcastle.config import CastleConfig, load_config

__all__ = [
    "Castle",
    "CastleConfig",
    "load_config",
]
