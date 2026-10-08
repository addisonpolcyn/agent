"""The sandbox runs generated skill code as untrusted input."""

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
        raise SkillError("text must be a string")
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


def test_skill_error_is_an_error_not_a_crash() -> None:
    result = run_sandboxed(WORD_COUNT, {"text": 3})
    assert result.error == "text must be a string"
    assert result.crash is None


def test_allows_subclassing_html_parser() -> None:
    result = run_sandboxed(HTML_TO_TEXT, {"html": "<p>Hi <b>there</b></p>"})
    assert result.output == {"text": "Hi there"}


@pytest.mark.parametrize(
    ("code", "problem"),
    [
        ("import os\ndef run(a):\n    return {}\n", "import of 'os'"),
        ("import socket\ndef run(a):\n    return {}\n", "import of 'socket'"),
        ("from urllib import request\ndef run(a):\n    return {}\n", "import from 'urllib'"),
        ("def run(a):\n    return {'x': open('/etc/passwd').read()}\n", "'open'"),
        ("def run(a):\n    return {'x': eval('1')}\n", "'eval'"),
        ("def run(a):\n    return {'x': getattr(a, 'keys')}\n", "'getattr'"),
        ("def run(a):\n    return {'x': ().__class__.__bases__}\n", "'__class__'"),
        ("def run(a):\n    return {'x': '{0.__class__}'.format(a)}\n", "'format'"),
        ("def run(a, b):\n    return {}\n", "missing a top-level 'def run(arguments)'"),
        ("def run(a):\n    return {\n", "syntax error"),
    ],
)
def test_rejects_unsafe_or_unusable_code(code: str, problem: str) -> None:
    problems = check_source(code)
    assert any(problem in p for p in problems), problems
    assert run_sandboxed(code, {}).crash is not None


def test_rejects_oversized_code() -> None:
    code = "def run(a):\n    return {}\n" + "#" * MAX_SOURCE_BYTES
    assert check_source(code) == [f"code is larger than {MAX_SOURCE_BYTES} bytes"]


def test_import_guard_holds_at_runtime_too() -> None:
    # Statically fine, but the runtime __import__ still refuses modules off the allowlist.
    code = "def run(a):\n    import json as j\n    return {'ok': j.dumps(1)}\n"
    assert run_sandboxed(code, {}).output == {"ok": "1"}


def test_infinite_loop_times_out() -> None:
    result = run_sandboxed("def run(a):\n    while True:\n        pass\n", {}, timeout=1)
    assert result.crash == "timed out after 1s"


def test_memory_is_limited() -> None:
    result = run_sandboxed("def run(a):\n    return {'x': 'a' * (10 ** 10)}\n", {})
    assert result.crash == "out of memory"


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
    (tmp_path / ".env").write_text("API_KEY=secret")
    (tmp_path / "id_rsa").write_text("-----BEGIN")
    return tmp_path


def test_file_functions_exist_only_for_file_skills(folder: Path) -> None:
    result = run_sandboxed(LISTER, {"path": str(folder)})
    assert result.crash == "NameError: name 'list_dir' is not defined"


def test_lists_and_reads_inside_approved_roots(folder: Path) -> None:
    listed = run_sandboxed(LISTER, {"path": str(folder)}, reads_files=True, readable_roots=[folder])
    assert listed.output == {
        "entries": [
            {"name": "a.txt", "type": "file", "size": 5},
            {"name": "sub", "type": "dir", "size": None},
        ]
    }, "secret files are never listed"
    nested = str(folder / "sub" / "b.md")
    read = run_sandboxed(READER, {"path": nested}, reads_files=True, readable_roots=[folder])
    assert read.output == {"text": "nested"}


def test_secrets_are_never_read_even_inside_a_root(folder: Path) -> None:
    for name in (".env", "id_rsa"):
        path = str(folder / name)
        result = run_sandboxed(READER, {"path": path}, reads_files=True, readable_roots=[folder])
        assert result.error is not None
        assert "looks like a secret" in result.error


def test_reading_outside_the_roots_asks_for_access(folder: Path) -> None:
    sub = folder / "sub"
    result = run_sandboxed(LISTER, {"path": str(folder)}, reads_files=True, readable_roots=[sub])
    assert result.needs_access == str(folder.resolve())


def test_access_requests_cannot_be_swallowed(folder: Path) -> None:
    code = (
        "def run(a):\n    try:\n        return {'e': list_dir(a['path'])}\n"
        "    except Exception:\n        return {'e': []}\n"
    )
    result = run_sandboxed(code, {"path": str(folder)}, reads_files=True, readable_roots=[])
    assert result.needs_access == str(folder.resolve())


def test_symlinks_are_resolved_before_checking_roots(
    folder: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "private.txt").write_text("not yours")
    (folder / "link").symlink_to(outside)
    path = str(folder / "link" / "private.txt")
    result = run_sandboxed(READER, {"path": path}, reads_files=True, readable_roots=[folder])
    assert result.needs_access == str(outside.resolve())
