"""First-run setup for kcastle.

Detects API keys from environment variables and writes a minimal
``config.yaml`` so the user can start immediately.  No interactive
wizard — just detection, confirmation, and a two-line config.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from kcastle.config import config_file_path


@dataclass(frozen=True, slots=True)
class VendorPreset:
    """First-run provider preset driven by environment variable detection."""

    display_name: str
    env_var: str
    provider: str
    model: str


_VENDOR_PRESETS: list[VendorPreset] = [
    VendorPreset(
        display_name="DeepSeek",
        env_var="DEEPSEEK_API_KEY",
        provider="deepseek-openai",
        model="deepseek-v4-flash",
    ),
    VendorPreset(
        display_name="MiniMax",
        env_var="MINIMAX_API_KEY",
        provider="minimax-openai",
        model="MiniMax-Text-01",
    ),
]

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def needs_setup(home: Path | None = None) -> bool:
    """Check whether first-run setup is needed."""
    return not config_file_path(home).is_file()


def _detect_presets() -> list[VendorPreset]:
    """Detect configured vendor presets from environment variables."""
    print("  Detecting API keys…")
    detected: list[VendorPreset] = []
    for preset in _VENDOR_PRESETS:
        if os.environ.get(preset.env_var):
            print(f"    {_GREEN}✓{_RESET} {preset.env_var}")
            detected.append(preset)
        else:
            print(f"    {_DIM}✗ {preset.env_var}{_RESET}")
    return detected


def _print_missing_keys_hint() -> None:
    """Print user guidance when no vendor API key is detected."""
    print(f"\n  {_YELLOW}No API keys found.{_RESET}")
    print("  Set one of the above environment variables, e.g.:\n")
    print(f'    {_DIM}export DEEPSEEK_API_KEY="sk-..."{_RESET}\n')
    print("  Then run `kcastle` again.\n")


def _confirm_write(path: Path) -> bool:
    """Ask user whether to write config file."""
    try:
        answer = input(f"\n  Write {_DIM}{path}{_RESET}? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return not answer or answer == "y"


def _write_minimal_config(path: Path, preset: VendorPreset) -> None:
    """Write minimal default config file for selected preset."""
    config: dict[str, object] = {
        "default": {
            "provider": preset.provider,
            "model": preset.model,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def run_setup(home: Path | None = None) -> Path:
    """Detect API keys and write a minimal ``config.yaml``.

    Returns the path to the generated config file.
    """
    path = config_file_path(home)

    print(f"\n  {_BOLD}🏰 kcastle — first-run setup{_RESET}\n")

    detected = _detect_presets()

    if not detected:
        _print_missing_keys_hint()
        sys.exit(1)

    chosen = detected[0]
    print(f"\n  Default: {_BOLD}{chosen.provider}{_RESET} ({chosen.model})")

    if not _confirm_write(path):
        print("  Aborted.\n")
        sys.exit(0)

    _write_minimal_config(path, chosen)

    print(f"\n  {_GREEN}✓{_RESET} Done. Run {_BOLD}kcastle{_RESET} to start.\n")
    return path
