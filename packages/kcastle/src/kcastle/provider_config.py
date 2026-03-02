"""Provider configuration type owned by kcastle application layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Provider construction config.

    A config captures vendor identity, protocol, model, and endpoint/auth
    options needed to construct a concrete provider instance.
    """

    vendor: str
    protocol: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    extra_body: dict[str, object] | None = None
    options: dict[str, object] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    @property
    def name(self) -> str:
        """Canonical profile name, e.g. ``deepseek-openai-completions``."""
        return f"{self.vendor}-{self.protocol.lower()}"
