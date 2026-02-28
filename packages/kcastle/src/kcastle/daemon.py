"""Daemon process management for kcastle.

Provides start/stop/status/restart operations for the background ``k``
process.  The daemon is a detached ``k -d`` subprocess whose PID is
tracked in ``~/.kcastle/k.pid`` and whose output is logged to
``~/.kcastle/k.log``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _pid_file(home: Path) -> Path:
    return home / "k.pid"


def _log_file(home: Path) -> Path:
    return home / "k.log"


def _is_alive(pid: int) -> bool:
    """Check whether a process with the given PID is running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it — treat as alive.
        return True
    return True


def _read_pid(home: Path) -> int | None:
    """Read PID from the pid file.  Returns ``None`` if missing or stale."""
    pf = _pid_file(home)
    if not pf.is_file():
        return None
    try:
        pid = int(pf.read_text().strip())
    except (ValueError, OSError):
        return None
    if not _is_alive(pid):
        pf.unlink(missing_ok=True)
        return None
    return pid


def _print_log_tail(log: Path, lines: int = 5) -> None:
    """Print the last *lines* of the log file."""
    if not log.is_file():
        return
    with contextlib.suppress(OSError):
        tail = log.read_text(encoding="utf-8").rstrip().splitlines()[-lines:]
        if tail:
            print(f"  Log ({_DIM}{log}{_RESET}):")
            for line in tail:
                print(f"    {_DIM}{line}{_RESET}")


def daemon_status(home: Path) -> None:
    """Print whether the daemon is running."""
    pid = _read_pid(home)
    if pid is not None:
        print(f"  {_GREEN}●{_RESET} Daemon is {_BOLD}running{_RESET} (PID {pid})")
        log = _log_file(home)
        if log.is_file():
            print(f"  Log: {_DIM}{log}{_RESET}")
    else:
        print(f"  {_DIM}●{_RESET} Daemon is {_BOLD}not running{_RESET}")


def daemon_start(home: Path, *, verbose: bool = False) -> None:
    """Start the daemon as a detached background process."""
    pid = _read_pid(home)
    if pid is not None:
        print(f"  {_YELLOW}Daemon already running{_RESET} (PID {pid})")
        return

    home.mkdir(parents=True, exist_ok=True)

    log = _log_file(home)
    cmd = [sys.executable, "-m", "kcastle.cli", "-d"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--home", str(home)])

    with open(log, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    pid_path = _pid_file(home)
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    # Poll briefly to catch immediate startup failures
    # (e.g. no channels configured, bad config, missing provider).
    for _ in range(20):
        time.sleep(0.15)
        if proc.poll() is not None:
            pid_path.unlink(missing_ok=True)
            print(f"  {_RED}●{_RESET} Daemon failed to start")
            _print_log_tail(log)
            return

    print(f"  {_GREEN}●{_RESET} Daemon started (PID {proc.pid})")
    print(f"  Log: {_DIM}{log}{_RESET}")


def daemon_stop(home: Path) -> None:
    """Stop the running daemon (SIGTERM)."""
    pid = _read_pid(home)
    if pid is None:
        print(f"  {_DIM}Daemon is not running{_RESET}")
        return

    os.kill(pid, signal.SIGTERM)

    # Wait up to 5 seconds for the process to exit.
    for _ in range(50):
        if not _is_alive(pid):
            break
        time.sleep(0.1)
    else:
        # Force kill if it didn't exit gracefully.
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)

    _pid_file(home).unlink(missing_ok=True)
    print(f"  {_GREEN}✓{_RESET} Daemon stopped (PID {pid})")


def daemon_restart(home: Path, *, verbose: bool = False) -> None:
    """Restart the daemon (stop + start)."""
    pid = _read_pid(home)
    if pid is not None:
        daemon_stop(home)
    daemon_start(home, verbose=verbose)
