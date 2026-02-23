"""Provider-agnostic error hierarchy."""


class KaiError(Exception):
    """Base error for all kai errors."""


class ProviderError(KaiError):
    """Base error for all provider-related errors."""


class ConnectionError(ProviderError):
    """The API connection failed."""


class TimeoutError(ProviderError):
    """The API request timed out."""


class StatusError(ProviderError):
    """The API returned an HTTP error status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmptyResponseError(ProviderError):
    """The API returned an empty response."""
