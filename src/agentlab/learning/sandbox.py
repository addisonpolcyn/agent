"""Run generated skill code in a separate process. Not a security boundary.

agentlab is a single-user toy for now, so learned skills run unrestricted: any import, full
builtins, the user's environment and working directory, the network, files and commands.
The protections are the user's approval of the plan (see ``learner.py``) and the runtime
harness. Restrictions come back before anyone else uses this.

What the separate process still buys: a crash, hang or runaway skill can't take the agent
down, and a wall-clock timeout stops it.

``check_source`` only checks that the code is usable: valid Python, not huge, with a
top-level ``run(arguments)``. Every skill also gets two convenience functions, ``list_dir``
and ``read_text`` (see ``_runner.py``).
"""

from __future__ import annotations

import ast
import json
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from agentlab.models import JSONObject

MAX_SOURCE_BYTES = 20_000
MAX_OUTPUT_BYTES = 1_000_000
DEFAULT_TIMEOUT_SECONDS = 30.0
RUNNER = Path(__file__).with_name("_runner.py")

MAX_DIR_ENTRIES = 500
MAX_READ_BYTES = 1_000_000


@dataclass(frozen=True)
class SandboxResult:
    """One execution. ``error`` is a ``SkillError`` the code raised on purpose; ``crash`` is
    anything else (an exception, a timeout, malformed output)."""

    output: JSONObject | None = None
    error: str | None = None
    crash: str | None = None


def check_source(code: str) -> list[str]:
    """Problems that make ``code`` unusable; empty when it may run."""
    if len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
        return [f"code is larger than {MAX_SOURCE_BYTES} bytes"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    if not any(_is_run_function(node) for node in tree.body):
        return ["missing a top-level 'def run(arguments)' with exactly one parameter"]
    return []


def run_sandboxed(
    code: str, arguments: JSONObject, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> SandboxResult:
    """Execute ``run(arguments)`` from ``code`` in a separate process, with a timeout."""
    problems = check_source(code)
    if problems:
        return SandboxResult(crash="rejected: " + "; ".join(problems))
    payload = json.dumps(
        {
            "code": code,
            "arguments": arguments,
            "max_entries": MAX_DIR_ENTRIES,
            "max_read_bytes": MAX_READ_BYTES,
        }
    )
    try:
        completed = subprocess.run(
            # -P: the runner's own folder is not importable, so `import models` can't pick up
            # agentlab internals.
            [sys.executable, "-P", "-B", str(RUNNER)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            start_new_session=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(crash=f"timed out after {timeout:g}s")
    return _parse_reply(completed)


def _is_run_function(node: ast.stmt) -> bool:
    if not isinstance(node, ast.FunctionDef) or node.name != "run":
        return False
    args = node.args
    return (
        len(args.posonlyargs) + len(args.args) == 1
        and not args.kwonlyargs
        and args.vararg is None
        and args.kwarg is None
    )


def _parse_reply(completed: subprocess.CompletedProcess[str]) -> SandboxResult:
    if completed.returncode < 0:
        return SandboxResult(crash=f"killed by {signal.Signals(-completed.returncode).name}")
    if len(completed.stdout) > MAX_OUTPUT_BYTES:
        return SandboxResult(crash=f"output larger than {MAX_OUTPUT_BYTES} bytes")
    try:
        reply = json.loads(completed.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError, IndexError:
        detail = completed.stderr.strip().splitlines()[-1:] or [f"exit {completed.returncode}"]
        return SandboxResult(crash=f"no result from the skill process: {detail[0]}")
    return _from_reply(reply)


def _from_reply(reply: object) -> SandboxResult:
    match reply:
        case {"output": dict()}:
            return SandboxResult(output=cast("JSONObject", cast("JSONObject", reply)["output"]))
        case {"error": str() as error}:
            return SandboxResult(error=error)
        case {"crash": str() as crash}:
            return SandboxResult(crash=crash)
        case _:
            return SandboxResult(crash="malformed reply from the skill process")
