"""Token usage statistics."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage for a single LLM call."""

    input_tokens: int
    """Number of input tokens (excluding cache)."""

    output_tokens: int
    """Number of output tokens."""

    cache_read_tokens: int = 0
    """Number of tokens read from cache."""

    cache_write_tokens: int = 0
    """Number of tokens written to cache."""

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )
