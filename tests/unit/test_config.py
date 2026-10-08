from __future__ import annotations

from pathlib import Path

from agentlab.config import DEFAULT_MODEL, Secret, Settings


def test_defaults_without_environment() -> None:
    settings = Settings.from_env({})
    assert settings.model == DEFAULT_MODEL
    assert settings.skills_dir == Path("skills")
    assert settings.anthropic_api_key is None


def test_reads_environment() -> None:
    settings = Settings.from_env(
        {
            "ANTHROPIC_API_KEY": "sk-test",
            "AGENTLAB_MODEL": "claude-haiku-5-5",
            "AGENTLAB_RUNS_DIR": "/tmp/runs",
        }
    )
    assert settings.model == "claude-haiku-5-5"
    assert settings.runs_dir == Path("/tmp/runs")
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.reveal() == "sk-test"


def test_blank_key_is_treated_as_missing() -> None:
    assert Settings.from_env({"ANTHROPIC_API_KEY": "  "}).anthropic_api_key is None


def test_secret_never_appears_in_repr() -> None:
    settings = Settings(anthropic_api_key=Secret("sk-very-secret"))
    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings)
