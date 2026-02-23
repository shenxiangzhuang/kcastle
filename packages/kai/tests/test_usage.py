"""Tests for kai.usage module."""

from kai.usage import TokenUsage


def test_token_usage_total_input_tokens() -> None:
    usage = TokenUsage(
        input_tokens=100, output_tokens=50, cache_read_tokens=20, cache_write_tokens=5
    )
    assert usage.total_input_tokens == 125


def test_token_usage_total_tokens() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    assert usage.total_tokens == 150


def test_token_usage_defaults() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=20)
    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0


def test_token_usage_add() -> None:
    a = TokenUsage(input_tokens=10, output_tokens=20, cache_read_tokens=5)
    b = TokenUsage(input_tokens=30, output_tokens=40, cache_write_tokens=10)
    result = a + b
    assert result.input_tokens == 40
    assert result.output_tokens == 60
    assert result.cache_read_tokens == 5
    assert result.cache_write_tokens == 10


def test_token_usage_frozen() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=20)
    try:
        usage.input_tokens = 99  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except AttributeError:
        pass
