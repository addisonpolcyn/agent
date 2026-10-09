"""Learned tools run unrestricted in their own process (see sandbox.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentlab.learning.sandbox import MAX_SOURCE_BYTES, check_source, run_sandboxed

if TYPE_CHECKING:
    from pathlib import Path

WORD_COUNT = """\
import re

def run(arguments):
    text = arguments.get("text")
    if not isinstance(text, str):
        raise ToolError("text must be a string")
    return {"count": len(re.findall(r"\\S+", text))}
"""

HTML_TO_TEXT = """\
from html.parser import HTMLParser

class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

def run(arguments):
    parser = TextParser()
    parser.feed(arguments["html"])
    return {"text": "".join(parser.parts)}
"""


def test_runs_valid_code() -> None:
    assert run_sandboxed(WORD_COUNT, {"text": "a b c"}).output == {"count": 3}


def test_tool_error_is_an_error_not_a_crash() -> None:
    result = run_sandboxed(WORD_COUNT, {"text": 3})
    assert result.error == "text must be a string"
    assert result.crash is None


def test_allows_subclassing_html_parser() -> None:
    result = run_sandboxed(HTML_TO_TEXT, {"html": "<p>Hi <b>there</b></p>"})
    assert result.output == {"text": "Hi there"}


@pytest.mark.parametrize(
    ("code", "problem"),
    [
        ("def run(a, b):\n    return {}\n", "missing a top-level 'def run(arguments)'"),
        ("def helper(a):\n    return {}\n", "missing a top-level 'def run(arguments)'"),
        ("def run(a):\n    return {\n", "syntax error"),
    ],
)
def test_rejects_unusable_code(code: str, problem: str) -> None:
    problems = check_source(code)
    assert any(problem in p for p in problems), problems
    assert run_sandboxed(code, {}).crash is not None


def test_rejects_oversized_code() -> None:
    code = "def run(a):\n    return {}\n" + "#" * MAX_SOURCE_BYTES
    assert check_source(code) == [f"code is larger than {MAX_SOURCE_BYTES} bytes"]


def test_any_import_and_builtin_is_allowed() -> None:
    code = (
        "import os, socket, subprocess, urllib.request\n"
        "def run(a):\n"
        "    return {'cpus': os.cpu_count() > 0, 'sum': eval('1 + 2'), 'name': type(a).__name__}\n"
    )
    assert check_source(code) == []
    assert run_sandboxed(code, {}).output == {"cpus": True, "sum": 3, "name": "dict"}


def test_tools_can_write_files_and_run_commands(tmp_path: Path) -> None:
    code = (
        "import subprocess, sys\n"
        "def run(a):\n"
        "    with open(a['path'], 'w') as f:\n"
        "        f.write('written')\n"
        "    done = subprocess.run([sys.executable, '-c', 'print(42)'], capture_output=True)\n"
        "    return {'out': done.stdout.decode().strip()}\n"
    )
    target = tmp_path / "out.txt"
    assert run_sandboxed(code, {"path": str(target)}).output == {"out": "42"}
    assert target.read_text() == "written"


def test_tools_see_the_users_environment_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTLAB_TEST_VALUE", "visible")
    monkeypatch.chdir(tmp_path)
    code = (
        "import os\n"
        "def run(a):\n"
        "    return {'env': os.environ['AGENTLAB_TEST_VALUE'], 'cwd': os.getcwd()}\n"
    )
    assert run_sandboxed(code, {}).output == {"env": "visible", "cwd": str(tmp_path)}


def test_printing_does_not_break_the_reply() -> None:
    code = "def run(a):\n    print('debug')\n    return {'ok': True}\n"
    assert run_sandboxed(code, {}).output == {"ok": True}


def test_agentlab_internals_are_not_importable_by_accident() -> None:
    # The runner lives next to models.py; -P keeps its folder off sys.path.
    code = "def run(a):\n    import models\n    return {}\n"
    assert run_sandboxed(code, {}).crash == "ModuleNotFoundError: No module named 'models'"


def test_infinite_loop_times_out() -> None:
    result = run_sandboxed("def run(a):\n    while True:\n        pass\n", {}, timeout=1)
    assert result.crash == "timed out after 1s"


def test_non_dict_result_is_a_crash() -> None:
    result = run_sandboxed("def run(a):\n    return [1]\n", {})
    assert result.crash == "TypeError: run() returned list, expected a dict"


def test_unexpected_exception_is_a_crash() -> None:
    result = run_sandboxed("def run(a):\n    return {'x': a['missing']}\n", {})
    assert result.crash == "KeyError: 'missing'"


LISTER = "def run(a):\n    return {'entries': list_dir(a['path'])}\n"
READER = "def run(a):\n    return {'text': read_text(a['path'])}\n"


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "sub" / "b.md").write_text("nested")
    return tmp_path


def test_file_helpers_list_and_read(folder: Path) -> None:
    listed = run_sandboxed(LISTER, {"path": str(folder)})
    assert listed.output == {
        "entries": [
            {"name": "a.txt", "type": "file", "size": 5},
            {"name": "sub", "type": "dir", "size": None},
        ]
    }
    read = run_sandboxed(READER, {"path": str(folder / "sub" / "b.md")})
    assert read.output == {"text": "nested"}


def test_file_helpers_raise_tool_errors(folder: Path) -> None:
    missing = run_sandboxed(READER, {"path": str(folder / "nope")})
    assert missing.error is not None
    assert "no such file or directory" in missing.error
    not_a_dir = run_sandboxed(LISTER, {"path": str(folder / "a.txt")})
    assert not_a_dir.error is not None
    assert "not a directory" in not_a_dir.error
