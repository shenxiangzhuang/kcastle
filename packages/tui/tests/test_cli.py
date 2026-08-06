import pytest
from ktui.cli import backends_from_env


def test_deepseek_backend_is_selected() -> None:
    backend = backends_from_env({"DEEPSEEK_API_KEY": "deepseek-key"})[0]

    assert backend.client.api_key == "deepseek-key"
    assert str(backend.client.base_url) == "https://api.deepseek.com"
    assert backend.model == "deepseek-v4-flash"
    assert backend.context_window == 1_000_000


def test_openai_backend_is_selected() -> None:
    backend = backends_from_env({"OPENAI_API_KEY": "openai-key"})[0]

    assert backend.client.api_key == "openai-key"
    assert str(backend.client.base_url) == "https://api.openai.com/v1/"
    assert backend.model == "gpt-5.5"
    assert backend.context_window == 1_050_000


def test_all_detected_backends_are_available_in_preference_order() -> None:
    backends = backends_from_env(
        {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "OPENAI_API_KEY": "openai-key",
        }
    )

    assert [backend.name for backend in backends] == ["DeepSeek", "OpenAI"]


def test_backend_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY or OPENAI_API_KEY"):
        backends_from_env({"DEEPSEEK_API_KEY": " "})
