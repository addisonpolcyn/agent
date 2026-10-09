"""The ``run_command`` starter tool, installed and run exactly as a learned tool is."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from agentlab.learning.learner import (
    CODE_FILE,
    STARTER_DIR,
    TOOL_FILE,
    install_starter_tools,
    load_learned,
)
from agentlab.tools.catalog import ToolCatalog

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def catalog(tmp_path: Path) -> ToolCatalog:
    install_starter_tools(tmp_path / "learned")
    return ToolCatalog({}).with_tools(load_learned(tmp_path / "learned"))


def ok(stdout: str = "", stderr: str = "", exit_code: int = 0) -> dict[str, Any]:
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "timed_out": False}


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("echo hello", ok("hello\n")),
        ("exit 3", ok(exit_code=3)),
        ("echo oops 1>&2", ok(stderr="oops\n")),
        ("printf 'a\\nb'; echo x 1>&2; exit 7", ok("a\nb", "x\n", 7)),
        ("true", ok()),
    ],
)
def test_returns_exit_code_and_both_streams(
    catalog: ToolCatalog, command: str, expected: dict[str, Any]
) -> None:
    assert catalog.execute("run_command", {"command": command}).output == expected


def test_runs_in_the_given_directory(catalog: ToolCatalog, tmp_path: Path) -> None:
    result = catalog.execute("run_command", {"command": "pwd", "cwd": str(tmp_path)})
    assert result.output == ok(f"{tmp_path}\n")


def test_timeout_kills_the_whole_process_group_and_keeps_partial_output(
    catalog: ToolCatalog,
) -> None:
    started = time.monotonic()
    arguments = {"command": "echo partial; sleep 5 & wait", "timeout_seconds": 0.5}
    result = catalog.execute("run_command", arguments)
    assert result.output == {
        "exit_code": None,
        "stdout": "partial\n",
        "stderr": "",
        "timed_out": True,
    }
    assert time.monotonic() - started < 4, "the backgrounded sleep must be killed too"


def test_stdin_is_empty_so_commands_cannot_block_on_input(catalog: ToolCatalog) -> None:
    assert catalog.execute("run_command", {"command": "cat", "timeout_seconds": 5}).output == ok()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"command": ""}, "'command'"),
        ({}, "'command'"),
        ({"command": "echo hi", "timeout_seconds": 0}, "'timeout_seconds'"),
        ({"command": "echo hi", "timeout_seconds": True}, "'timeout_seconds'"),
        ({"command": "echo hi", "cwd": "/definitely/not/a/real/dir"}, "no such file"),
    ],
)
def test_bad_input_is_a_tool_error_not_a_crash(
    catalog: ToolCatalog, arguments: dict[str, Any], message: str
) -> None:
    result = catalog.execute("run_command", arguments)
    assert result.error is not None
    assert message in result.error
    assert "crashed" not in result.error


def test_install_copies_starters_and_never_overwrites(tmp_path: Path) -> None:
    store = tmp_path / "learned"
    assert install_starter_tools(store) == ["run_command"]
    shipped = STARTER_DIR / "run_command"
    for name in (TOOL_FILE, CODE_FILE):
        assert (store / "run_command" / name).read_text() == (shipped / name).read_text()

    edited = store / "run_command" / CODE_FILE
    edited.write_text(edited.read_text() + "\n# local edit\n")
    assert install_starter_tools(store) == []
    assert edited.read_text().endswith("# local edit\n")
