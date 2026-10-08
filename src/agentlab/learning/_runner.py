"""Sandbox child process. Executed as a script by ``sandbox.run_sandboxed``; never imported.

Reads ``{"code", "arguments", "allowed_modules", "file_access"}`` from stdin, runs
``run(arguments)`` with restricted builtins, and writes exactly one JSON line:
``{"output": {...}}``, ``{"error": ...}`` (the code raised ``SkillError``), ``{"crash": ...}``
(anything else) or ``{"needs_access": "<dir>"}`` (it tried to read an unapproved folder).

File access, when granted, is two read-only functions, never ``open``: ``list_dir(path)`` and
``read_text(path)``. They see only ``readable_roots`` and never return secret-looking files.
"""

import builtins
import json
import stat
import sys
from pathlib import Path
from typing import Any

SAFE_BUILTINS = [
    "abs",
    "all",
    "any",
    "bool",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "hash",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "ord",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "zip",
    "ArithmeticError",
    "AttributeError",
    "Exception",
    "IndexError",
    "KeyError",
    "LookupError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "True",
    "False",
    "None",
]


class SkillError(Exception):
    """Raised by generated code for expected failures (bad input). Reported, not a crash."""


class NeedsAccess(BaseException):
    """Not an ``Exception``, so generated code's ``except Exception`` can't swallow it."""


def _file_api(rules: dict[str, Any], state: dict[str, str]) -> dict[str, Any]:
    roots = [Path(root).resolve() for root in rules["readable_roots"]]
    base = Path(rules["base_dir"])
    names = frozenset(rules["secret_names"])
    prefixes = tuple(rules["secret_prefixes"])
    suffixes = tuple(rules["secret_suffixes"])

    def secret(name: str) -> bool:
        lowered = name.lower()
        return lowered in names or lowered.startswith(prefixes) or lowered.endswith(suffixes)

    def resolve(raw: object) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise SkillError("path must be a non-empty string")
        try:
            # strict: symlinks are followed, so the root and secret checks see the real path.
            path = (base / Path(raw).expanduser()).resolve(strict=True)
        except FileNotFoundError:
            raise SkillError(f"no such file or directory: {raw}") from None
        except (OSError, RuntimeError) as exc:
            raise SkillError(f"cannot resolve {raw}: {exc}") from None
        if any(secret(part) for part in path.parts):
            raise SkillError(f"{raw} looks like a secret (keys, credentials, .env); never read")
        return path

    def require(directory: Path) -> None:
        if not any(directory.is_relative_to(root) for root in roots):
            state["needs_access"] = str(directory)
            raise NeedsAccess(str(directory))

    def list_dir(path: object) -> list[dict[str, Any]]:
        directory = resolve(path)
        if not directory.is_dir():
            raise SkillError(f"not a directory: {path}")
        require(directory)
        entries: list[dict[str, Any]] = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if secret(child.name):
                continue
            entries.append(_entry(child))
            if len(entries) >= rules["max_entries"]:
                break
        return entries

    def read_text(path: object) -> str:
        file = resolve(path)
        if not file.is_file():
            raise SkillError(f"not a file: {path}")
        require(file.parent)
        with file.open("rb") as handle:
            data = handle.read(rules["max_read_bytes"])
        if b"\x00" in data:
            raise SkillError(f"not a text file: {path}")
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


def _namespace(allowed_modules: frozenset[str], extra: dict[str, Any]) -> dict[str, Any]:
    def guarded_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if level or name not in allowed_modules:
            raise ImportError(f"import of {name!r} is not allowed")
        return __import__(name, globals, locals, fromlist, level)

    safe: dict[str, Any] = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    safe["__import__"] = guarded_import
    safe["__build_class__"] = builtins.__build_class__
    return {
        "__builtins__": safe,
        "__name__": "learned_skill",
        "SkillError": SkillError,
        **extra,
    }


def main() -> None:
    request = json.loads(sys.stdin.read())
    state: dict[str, str] = {}
    rules = request.get("file_access")
    extra = _file_api(rules, state) if rules else {}
    namespace = _namespace(frozenset(request["allowed_modules"]), extra)
    try:
        reply = _execute(request, namespace)
    except NeedsAccess:
        reply = {}
    if "needs_access" in state:
        # Whatever the code did after a refused read, the answer is "ask the user first".
        reply = {"needs_access": state["needs_access"]}
    sys.stdout.write(json.dumps(reply, allow_nan=False) + "\n")


def _execute(request: dict[str, Any], namespace: dict[str, Any]) -> dict[str, Any]:
    try:
        exec(compile(request["code"], "<learned skill>", "exec"), namespace)
        output = namespace["run"](request["arguments"])
        if not isinstance(output, dict):
            raise TypeError(f"run() returned {type(output).__name__}, expected a dict")
        json.dumps(output, allow_nan=False)
    except SkillError as exc:
        return {"error": str(exc) or "SkillError"}
    except RecursionError:
        return {"crash": "recursion too deep"}
    except MemoryError:
        return {"crash": "out of memory"}
    except Exception as exc:
        return {"crash": f"{type(exc).__name__}: {exc}"}
    return {"output": output}


main()
