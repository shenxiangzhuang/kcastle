"""Tests for kai.errors module."""

from kai.errors import (
    ConnectionError,
    EmptyResponseError,
    KaiError,
    ProviderError,
    StatusError,
    TimeoutError,
)


def test_error_hierarchy() -> None:
    assert issubclass(ProviderError, KaiError)
    assert issubclass(ConnectionError, ProviderError)
    assert issubclass(TimeoutError, ProviderError)
    assert issubclass(StatusError, ProviderError)
    assert issubclass(EmptyResponseError, ProviderError)


def test_status_error_code() -> None:
    err = StatusError(429, "Rate limited")
    assert err.status_code == 429
    assert "Rate limited" in str(err)


def test_provider_error_message() -> None:
    err = ProviderError("something broke")
    assert str(err) == "something broke"
