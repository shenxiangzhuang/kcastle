"""Built-in provider definitions and merge logic.

Contains the default provider profiles that ship with kcastle and the
logic to merge user overrides on top of them.
"""

from __future__ import annotations

from typing import Any


def _to_str_dict(d: object) -> dict[str, Any]:
    """Coerce an untyped dict from YAML into ``dict[str, Any]``."""
    if not isinstance(d, dict):
        return {}
    return {
        str(k): v  # pyright: ignore[reportUnknownArgumentType]
        for k, v in d.items()  # pyright: ignore[reportUnknownVariableType]
    }


def builtin_provider_dicts() -> dict[str, dict[str, Any]]:
    """Built-in provider definitions.

    Builtins are explicit provider IDs.  User config can override any
    field; missing fields fall back to these defaults.
    """
    ds_models: dict[str, object] = {
        "deepseek-v4-flash": {"active": True},
        "deepseek-v4-pro": {"active": True},
    }
    mm_models: dict[str, object] = {
        "MiniMax-M2.5": {"active": True},
        "MiniMax-M2.5-highspeed": {"active": True},
        "MiniMax-M2": {"active": True},
    }
    return {
        "deepseek-openai": {
            "api_key": "${DEEPSEEK_API_KEY}",
            "base_url": "https://api.deepseek.com",
            "models": dict(ds_models),
        },
        "deepseek-anthropic": {
            "api_key": "${DEEPSEEK_API_KEY}",
            "base_url": "https://api.deepseek.com/anthropic",
            "models": dict(ds_models),
        },
        "minimax-openai": {
            "api_key": "${MINIMAX_API_KEY}",
            "base_url": "https://api.minimaxi.com/v1",
            "models": dict(mm_models),
            # MiniMax OpenAI endpoint embeds thinking as <think> tags in
            # content by default. Setting reasoning_split=True instructs the API
            # to separate thinking into the reasoning_details field.
            "extra_body": {"reasoning_split": True},
        },
        "minimax-anthropic": {
            "api_key": "${MINIMAX_API_KEY}",
            "base_url": "https://api.minimaxi.com/anthropic",
            "models": dict(mm_models),
        },
    }


def merge_builtin_providers(data: dict[str, Any]) -> None:
    """Merge built-in provider definitions into raw config data (in-place).

    Builtins form the base; user-provided providers override individual
    fields.  New user-defined providers are added as-is.
    """
    merged: dict[str, dict[str, Any]] = dict(builtin_provider_dicts())
    user_providers: object = data.get("providers")
    if isinstance(user_providers, dict):
        for provider, cfg in user_providers.items():  # pyright: ignore[reportUnknownVariableType]
            provider_name = str(provider).lower()  # pyright: ignore[reportUnknownArgumentType]
            if not isinstance(cfg, dict):
                continue

            user_cfg = _to_str_dict(cfg)  # pyright: ignore[reportUnknownArgumentType]
            if provider_name not in merged:
                merged[provider_name] = user_cfg
                continue

            base_cfg = merged[provider_name]
            merged_models = _to_str_dict(base_cfg.get("models"))
            user_models_raw = user_cfg.get("models")
            if isinstance(user_models_raw, dict):
                merged_models = {
                    **merged_models,
                    **_to_str_dict(user_models_raw),  # pyright: ignore[reportUnknownArgumentType]
                }

            merged[provider_name] = {
                **base_cfg,
                **{k: v for k, v in user_cfg.items() if k != "models"},
                "models": merged_models,
            }
    data["providers"] = merged
