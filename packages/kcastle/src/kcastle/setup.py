"""First-run setup for kcastle.

Detects API keys from environment variables and writes a minimal
``config.yaml`` so the user can start immediately.  No interactive
wizard — just detection, confirmation, and a two-line config.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from kcastle.config import config_file_path

_VENDORS: list[tuple[str, str, str, str]] = [
    ("DeepSeek", "DEEPSEEK_API_KEY", "deepseek-openai", "deepseek-chat"),
    ("MiniMax", "MINIMAX_API_KEY", "minimax-openai", "MiniMax-Text-01"),
]

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def needs_setup(home: Path | None = None) -> bool:
    """Check whether first-run setup is needed."""
    return not config_file_path(home).is_file()


def run_setup(home: Path | None = None) -> Path:
    """Detect API keys and write a minimal ``config.yaml``.

    Returns the path to the generated config file.
    """
    path = config_file_path(home)

    print(f"\n  {_BOLD}🏰 kcastle — first-run setup{_RESET}\n")

    print("  Detecting API keys…")
    detected: list[tuple[str, str, str, str]] = []
    for display, env_var, provider, model in _VENDORS:
        if os.environ.get(env_var):
            print(f"    {_GREEN}✓{_RESET} {env_var}")
            detected.append((display, env_var, provider, model))
        else:
            print(f"    {_DIM}✗ {env_var}{_RESET}")

    if not detected:
        print(f"\n  {_YELLOW}No API keys found.{_RESET}")
        print("  Set one of the above environment variables, e.g.:\n")
        print(f'    {_DIM}export DEEPSEEK_API_KEY="sk-..."{_RESET}\n')
        print("  Then run `k` again.\n")
        sys.exit(1)

    _display, _env, provider, model = detected[0]
    print(f"\n  Default: {_BOLD}{provider}{_RESET} ({model})")

    try:
        answer = input(f"\n  Write {_DIM}{path}{_RESET}? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if answer and answer != "y":
        print("  Aborted.\n")
        sys.exit(0)

    config: dict[str, object] = {
        "default": {"provider": provider, "model": model},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")

    print(f"\n  {_GREEN}✓{_RESET} Done. Run {_BOLD}k{_RESET} to start.\n")
    return path
