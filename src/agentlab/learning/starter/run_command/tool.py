"""Run one shell command on the user's machine. A starter tool: learned-tool code that ships
with agentlab and is copied into the learned directory at launch (``install_starter_tools``).

It runs in the tool process like any learned tool, which predefines ``ToolError``; the import
below is for type checking only. Side-effecting and unrestricted: the command runs with the
user's environment, files and network. Acceptable while agentlab is a single-user toy (see
docs/roadmap.md).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentlab.models import JSONObject

    # Typing only: at runtime the tool process supplies ToolError.
    from agentlab.tools.models import ToolError  # noqa: TC004

DEFAULT_TIMEOUT_SECONDS = 60
SHELL = "/bin/sh"
# How long to wait for output after killing a timed-out command.
DRAIN_SECONDS = 5


def run(arguments: Mapping[str, Any]) -> JSONObject:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ToolError("'command' must be a non-empty string")
    timeout = _timeout(arguments.get("timeout_seconds"))
    cwd = _cwd(arguments.get("cwd"))
    try:
        # A new session puts the command in its own process group, so a timeout can kill
        # everything it started, not just the shell.
        proc = subprocess.Popen(
            [SHELL, "-c", command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolError(f"failed to start command: {exc}") from exc
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
        return _result(None, out, err, timed_out=True)
    return _result(proc.returncode, out, err, timed_out=False)


def _timeout(value: object) -> float:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ToolError("'timeout_seconds' must be a number greater than 0")
    return float(value)


def _cwd(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ToolError("'cwd' must be a non-empty string")
    path = os.path.expanduser(value)
    if not os.path.exists(path):
        raise ToolError(f"cwd: no such file or directory: {path}")
    if not os.path.isdir(path):
        raise ToolError(f"cwd is not a directory: {path}")
    return path


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    # The group may already be gone if the command exited just as the timeout hit.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


def _result(exit_code: int | None, out: bytes, err: bytes, *, timed_out: bool) -> JSONObject:
    return {
        "exit_code": exit_code,
        "stdout": out.decode("utf-8", errors="replace"),
        "stderr": err.decode("utf-8", errors="replace"),
        "timed_out": timed_out,
    }
