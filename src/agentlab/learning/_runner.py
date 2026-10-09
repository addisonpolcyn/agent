"""Tool child process. Executed as a script by ``sandbox.run_sandboxed``; never imported.

Reads ``{"code", "arguments", "max_entries", "max_read_bytes"}`` from stdin, runs
``run(arguments)`` with full Python, and writes exactly one JSON line: ``{"output": {...}}``,
``{"error": ...}`` (the code raised ``ToolError``) or ``{"crash": ...}`` (anything else).

Besides ``ToolError``, the code gets two conveniences: ``list_dir(path)`` and
``read_text(path)``. They're shortcuts with friendly errors, not limits: ``open``, ``os`` and
everything else are available too.
"""

import json
import stat
import sys
from pathlib import Path
from typing import Any


class ToolError(Exception):
    """Raised by generated code for expected failures (bad input). Reported, not a crash."""


def _file_api(rules: dict[str, Any]) -> dict[str, Any]:
    def resolve(raw: object) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError("path must be a non-empty string")
        try:
            return Path(raw).expanduser().resolve(strict=True)
        except FileNotFoundError:
            raise ToolError(f"no such file or directory: {raw}") from None
        except (OSError, RuntimeError) as exc:
            raise ToolError(f"cannot resolve {raw}: {exc}") from None

    def list_dir(path: object) -> list[dict[str, Any]]:
        directory = resolve(path)
        if not directory.is_dir():
            raise ToolError(f"not a directory: {path}")
        children = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        return [_entry(child) for child in children[: rules["max_entries"]]]

    def read_text(path: object) -> str:
        file = resolve(path)
        if not file.is_file():
            raise ToolError(f"not a file: {path}")
        with file.open("rb") as handle:
            data = handle.read(rules["max_read_bytes"])
        if b"\x00" in data:
            raise ToolError(f"not a text file: {path}")
        return data.decode("utf-8", errors="replace")

    return {"list_dir": list_dir, "read_text": read_text}


def _entry(child: Path) -> dict[str, Any]:
    info = child.lstat()
    if stat.S_ISLNK(info.st_mode):
        kind = "symlink"
    elif stat.S_ISDIR(info.st_mode):
        kind = "dir"
    else:
        kind = "file" if stat.S_ISREG(info.st_mode) else "other"
    return {"name": child.name, "type": kind, "size": info.st_size if kind == "file" else None}


def main() -> None:
    request = json.loads(sys.stdin.read())
    namespace = {"__name__": "learned_tool", "ToolError": ToolError, **_file_api(request)}
    # Anything the tool prints goes to stderr, so stdout carries only the reply.
    stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        reply = _execute(request, namespace)
    finally:
        sys.stdout = stdout
    sys.stdout.write(json.dumps(reply, allow_nan=False) + "\n")


def _execute(request: dict[str, Any], namespace: dict[str, Any]) -> dict[str, Any]:
    try:
        exec(compile(request["code"], "<learned tool>", "exec"), namespace)
        output = namespace["run"](request["arguments"])
        if not isinstance(output, dict):
            raise TypeError(f"run() returned {type(output).__name__}, expected a dict")
        json.dumps(output, allow_nan=False)
    except ToolError as exc:
        return {"error": str(exc) or "ToolError"}
    except RecursionError:
        return {"crash": "recursion too deep"}
    except MemoryError:
        return {"crash": "out of memory"}
    except Exception as exc:
        return {"crash": f"{type(exc).__name__}: {exc}"}
    return {"output": output}


main()
