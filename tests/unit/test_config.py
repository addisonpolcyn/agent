from __future__ import annotations

from pathlib import Path

import pytest

from agentlab.config import DEFAULT_MODEL, ConfigError, Secret, Settings, read_env_file


def test_defaults_without_environment() -> None:
    settings = Settings.from_env({})
    assert settings.model == DEFAULT_MODEL
    assert settings.tools_dir == Path("tools")
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


def test_read_env_file_parses_common_forms(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "ANTHROPIC_API_KEY=sk-plain\n"
        "export AGENTLAB_MODEL=claude-haiku-5-5\n"
        'AGENTLAB_RUNS_DIR="/tmp/my runs"\n'
        "AGENTLAB_CASES_DIR='cases'\n"
        "EMPTY=\n"
        "WITH_EQUALS=a=b\n"
    )
    assert read_env_file(env_file) == {
        "ANTHROPIC_API_KEY": "sk-plain",
        "AGENTLAB_MODEL": "claude-haiku-5-5",
        "AGENTLAB_RUNS_DIR": "/tmp/my runs",
        "AGENTLAB_CASES_DIR": "cases",
        "EMPTY": "",
        "WITH_EQUALS": "a=b",
    }


def test_missing_env_file_is_empty(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / ".env") == {}


@pytest.mark.parametrize("line", ["no equals sign", "=value", "BAD KEY=1"])
def test_malformed_env_line_is_an_error(tmp_path: Path, line: str) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"OK=1\n{line}\n")
    with pytest.raises(ConfigError, match=r"\.env:2: expected KEY=VALUE"):
        read_env_file(env_file)


def test_environment_overrides_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-from-file\nAGENTLAB_MODEL=claude-haiku-5-5\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-environment")
    monkeypatch.delenv("AGENTLAB_MODEL", raising=False)

    settings = Settings.load(env_file)

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.reveal() == "sk-from-environment"
    assert settings.model == "claude-haiku-5-5"
