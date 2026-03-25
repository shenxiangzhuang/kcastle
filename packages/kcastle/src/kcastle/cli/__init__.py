"""CLI entry point for kcastle.

Usage::

    $ kcastle                    # New session (auto-generated ID)
    $ k                         # Alias for kcastle
    $ kcastle -C                 # Continue most recently active session
    $ k -C                       # Continue most recently active session
    $ kcastle -S <id>            # Resume specific session by ID
    $ k -S <id>                  # Resume specific session by ID
    $ kcastle -d                 # Daemon mode (no interactive CLI, foreground)
    $ k -d                       # Daemon mode (no interactive CLI, foreground)
    $ kcastle start              # Start daemon in background
    $ k start                    # Start daemon in background
    $ kcastle stop               # Stop the background daemon
    $ k stop                     # Stop the background daemon
    $ kcastle status             # Show daemon status
    $ k status                   # Show daemon status
    $ kcastle restart            # Restart the daemon
    $ k restart                  # Restart the daemon
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def main() -> None:
    """Main entry point for the ``kcastle`` and ``k`` commands."""
    parser = argparse.ArgumentParser(
        prog="k",
        description="k / kcastle — AI agent with session management",
    )
    parser.add_argument(
        "-S",
        "--session",
        metavar="ID",
        help="Resume a specific session by ID",
    )
    parser.add_argument(
        "-C",
        "--continue",
        dest="continue_latest",
        action="store_true",
        help="Continue the most recently active session",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Run in daemon mode (no interactive CLI, only background channels)",
    )
    parser.add_argument(
        "--home",
        metavar="DIR",
        help="Override kcastle home directory (default: ~/.kcastle)",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="Start daemon in background")
    sub.add_parser("stop", help="Stop the background daemon")
    sub.add_parser("status", help="Show daemon status")
    sub.add_parser("restart", help="Restart the daemon")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    from pathlib import Path

    home = Path(args.home) if args.home else None

    if args.command in ("start", "stop", "status", "restart"):
        from kcastle.config import _DEFAULT_HOME  # pyright: ignore[reportPrivateUsage]

        from .daemon import (
            daemon_restart,
            daemon_start,
            daemon_status,
            daemon_stop,
        )

        resolved_home = home or _DEFAULT_HOME
        if args.command == "start":
            daemon_start(resolved_home, verbose=args.verbose)
        elif args.command == "stop":
            daemon_stop(resolved_home)
        elif args.command == "status":
            daemon_status(resolved_home)
        elif args.command == "restart":
            daemon_restart(resolved_home, verbose=args.verbose)
        return

    from kcastle.config import load_config

    from .setup import needs_setup, run_setup

    if needs_setup(home):
        run_setup(home)

    config = load_config(home=home)

    from kcastle.castle import Castle

    castle = Castle.create(
        config,
        session_id=args.session,
        continue_latest=args.continue_latest,
        daemon=args.daemon,
    )

    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(castle.run())


if __name__ == "__main__":
    main()
