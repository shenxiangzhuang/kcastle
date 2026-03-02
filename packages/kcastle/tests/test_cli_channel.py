from __future__ import annotations

import pytest

from kcastle.channels.cli import parse_session_new_args


def test_parse_session_new_args_without_id_uses_name_only() -> None:
    sid, name = parse_session_new_args(["project", "alpha"])

    assert sid is None
    assert name == "project alpha"


def test_parse_session_new_args_with_explicit_id() -> None:
    sid, name = parse_session_new_args(["--id", "demo-id", "session", "name"])

    assert sid == "demo-id"
    assert name == "session name"


def test_parse_session_new_args_with_id_flag_requires_value() -> None:
    with pytest.raises(ValueError, match="Usage: /session new"):
        parse_session_new_args(["--id"])
