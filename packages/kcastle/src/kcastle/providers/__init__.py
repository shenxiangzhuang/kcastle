"""Provider configuration, factory, and built-in definitions."""

from kcastle.providers.builtins import builtin_provider_dicts, merge_builtin_providers
from kcastle.providers.config import ModelConfig, ProviderConfig, ProviderEntry
from kcastle.providers.factory import (
    ProviderRegistry,
    build_provider_entry,
    create_provider,
    parse_models,
    parse_providers,
)
from kcastle.providers.model_manager import ModelManager

__all__ = [
    "ModelConfig",
    "ModelManager",
    "ProviderConfig",
    "ProviderEntry",
    "ProviderRegistry",
    "build_provider_entry",
    "builtin_provider_dicts",
    "create_provider",
    "merge_builtin_providers",
    "parse_models",
    "parse_providers",
]
