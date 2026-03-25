"""CLI entry point for the ``kcastle`` command (``k`` is a shortcut alias).

Usage::

    $ k                          # New session (auto-generated ID)
    $ k -C                       # Continue most recently active session
    $ k -S <id>                  # Resume specific session by ID
    $ k -d                       # Daemon mode (foreground, no interactive CLI)
    $ k --verbose                # Show app lifecycle logs
    $ k --debug                  # Show detailed debug logs
    $ k start|stop|status|restart  # Manage background daemon
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    """Configure CLI logging for normal, verbose, and debug modes."""
    level = logging.DEBUG if debug else logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if verbose:
        for name in ("kcastle", "kagent", "kai"):
            logging.getLogger(name).setLevel(logging.INFO)

    if not debug:
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)


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
        help="Show informative app logs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed debug logs, including transport internals",
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

    _configure_logging(verbose=args.verbose, debug=args.debug)

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
            daemon_start(resolved_home, verbose=args.verbose, debug=args.debug)
        elif args.command == "stop":
            daemon_stop(resolved_home)
        elif args.command == "status":
            daemon_status(resolved_home)
        elif args.command == "restart":
            daemon_restart(resolved_home, verbose=args.verbose, debug=args.debug)
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
