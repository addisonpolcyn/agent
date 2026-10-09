from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from agentlab.agent.loop import Agent
from agentlab.cli import EXIT_ERROR, EXIT_OK, main
from agentlab.llm.client import LLMError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agentlab.agent.loop import AgentRun
    from agentlab.learning.approval import Approver
    from agentlab.models import Message


@pytest.fixture(autouse=True)
def repo_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo_root: Path) -> None:
    # Run from an empty directory so a developer's real .env can never leak into tests.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTLAB_TOOLS_DIR", str(repo_root / "tools"))
    monkeypatch.setenv("AGENTLAB_CASES_DIR", str(repo_root / "evals" / "cases"))
    monkeypatch.setenv("AGENTLAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AGENTLAB_LEARNED_DIR", str(tmp_path / "learned"))
    # File-reading eval cases name paths relative to the repo root.
    (tmp_path / "evals").symlink_to(repo_root / "evals")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_ask_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "--offline", "What is 123 * 456?"]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("The result is 56088.")
    assert 'tool calculator({"expression": "123 * 456"}) -> {"result": 56088}' in out


def test_ask_without_key_fails_clearly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "What is 1 + 1?"]) == EXIT_ERROR
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_eval_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--offline"]) == EXIT_OK
    assert "cases passed (100%)" in capsys.readouterr().out


def test_flywheel_offline_records_iterations(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["flywheel", "--offline"]) == EXIT_OK
    assert main(["flywheel", "--offline"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "baseline" in out
    assert "Newly failing: none" in out
    assert len(list((tmp_path / "runs").iterdir())) == 2


def test_tools_lists_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tools"]) == EXIT_OK
    assert "calculator  [arithmetic]" in capsys.readouterr().out


def test_bad_tools_dir_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENTLAB_TOOLS_DIR", str(tmp_path / "missing"))
    assert main(["tools"]) == EXIT_ERROR
    assert "tools directory not found" in capsys.readouterr().err


def test_chat_answers_each_question_until_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("What is 2 + 3?\n\nWhat is 6 * 7?\nexit\n"))
    assert main(["chat", "--offline"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "The result is 5." in out
    assert "The result is 42." in out


def test_chat_ends_cleanly_at_end_of_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("What is 1 + 1?\n"))
    assert main(["chat", "--offline"]) == EXIT_OK
    assert "The result is 2." in capsys.readouterr().out


def test_chat_keeps_going_after_an_llm_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def flaky_run(
        _self: Agent,
        task: str,
        *,
        approver: Approver | None = None,
        history: Sequence[Message] = (),
    ) -> AgentRun:
        calls.append(task)
        raise LLMError("service unavailable")

    monkeypatch.setattr(Agent, "run", flaky_run)
    monkeypatch.setattr("sys.stdin", io.StringIO("first\nsecond\n"))
    assert main(["chat", "--offline"]) == EXIT_OK
    assert calls == ["first", "second"]
    assert capsys.readouterr().err.count("error: service unavailable") == 2


def test_reads_settings_from_dotenv_in_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    repo_root: Path,
) -> None:
    monkeypatch.delenv("AGENTLAB_TOOLS_DIR")
    (tmp_path / ".env").write_text(f"AGENTLAB_TOOLS_DIR={repo_root / 'tools'}\n")
    assert main(["tools"]) == EXIT_OK
    assert "calculator" in capsys.readouterr().out


def test_malformed_dotenv_is_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("this is not a setting\n")
    assert main(["tools"]) == EXIT_ERROR
    assert ".env:1: expected KEY=VALUE" in capsys.readouterr().err


def test_chat_carries_the_conversation_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real_run = Agent.run

    def recording_run(
        self: Agent,
        task: str,
        *,
        approver: Approver | None = None,
        history: Sequence[Message] = (),
    ) -> AgentRun:
        seen.append(len(history))
        return real_run(self, task, approver=approver, history=history)

    monkeypatch.setattr(Agent, "run", recording_run)
    monkeypatch.setattr("sys.stdin", io.StringIO("What is 2 + 3?\nWhat is 6 * 7?\nreset\nhi\n"))
    assert main(["chat", "--offline"]) == EXIT_OK
    first, second, after_reset = seen
    assert first == 0
    assert second > 0, "the second question sees the first exchange"
    assert after_reset == 0
