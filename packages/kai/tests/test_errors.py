"""Tests for kai.errors module."""

from kai.errors import ErrorKind, KaiError


def test_error_kinds() -> None:
    assert ErrorKind.CONNECTION == "connection"
    assert ErrorKind.TIMEOUT == "timeout"
    assert ErrorKind.STATUS == "status"
    assert ErrorKind.EMPTY_RESPONSE == "empty_response"
    assert ErrorKind.PROVIDER == "provider"


def test_status_error() -> None:
    err = KaiError(ErrorKind.STATUS, "HTTP 429: Rate limited")
    assert err.kind == ErrorKind.STATUS
    assert "429" in str(err)
    assert "Rate limited" in str(err)


def test_provider_error_message() -> None:
    err = KaiError(ErrorKind.PROVIDER, "something broke")
    assert "something broke" in str(err)
    assert err.kind == ErrorKind.PROVIDER


def test_with_cause() -> None:
    original = ValueError("bad value")
    err = KaiError(ErrorKind.PROVIDER, "wrapped").with_cause(original)
    assert err.cause is original
    assert err.message == "wrapped"


def test_frozen() -> None:
    import contextlib

    err = KaiError(ErrorKind.TIMEOUT, "timed out")
    with contextlib.suppress(AttributeError):
        err.kind = ErrorKind.PROVIDER  # type: ignore[misc]
