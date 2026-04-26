from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kcastle.cli import main


def test_daemon_start_runs_first_run_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_calls: list[Path] = []
    daemon_calls: list[Path] = []

    def fake_run_setup(home: Path | None = None) -> Path:
        assert home is not None
        setup_calls.append(home)
        return home / "config.yaml"

    def fake_daemon_start(home: Path, *, verbose: bool = False, debug: bool = False) -> None:
        assert not verbose
        assert not debug
        daemon_calls.append(home)

    monkeypatch.setattr(sys, "argv", ["k", "--home", str(tmp_path), "start"])
    monkeypatch.setattr("kcastle.cli.setup.run_setup", fake_run_setup)
    monkeypatch.setattr("kcastle.cli.daemon.daemon_start", fake_daemon_start)

    main()

    assert setup_calls == [tmp_path]
    assert daemon_calls == [tmp_path]


def test_daemon_stop_does_not_run_first_run_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_calls: list[Path | None] = []
    stop_calls: list[Path] = []

    def fake_run_setup(home: Path | None = None) -> Path:
        setup_calls.append(home)
        return tmp_path / "config.yaml"

    def fake_daemon_stop(home: Path) -> None:
        stop_calls.append(home)

    monkeypatch.setattr(sys, "argv", ["k", "--home", str(tmp_path), "stop"])
    monkeypatch.setattr("kcastle.cli.setup.run_setup", fake_run_setup)
    monkeypatch.setattr("kcastle.cli.daemon.daemon_stop", fake_daemon_stop)

    main()

    assert setup_calls == []
    assert stop_calls == [tmp_path]
