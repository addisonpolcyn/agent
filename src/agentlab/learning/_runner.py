"""Sandbox child process. Executed as a script by ``sandbox.run_sandboxed``; never imported.

Reads ``{"code", "arguments", "allowed_modules"}`` from stdin, runs ``run(arguments)`` with
restricted builtins, and writes exactly one JSON line: ``{"output": {...}}``, ``{"error": ...}``
(the code raised ``SkillError``) or ``{"crash": ...}`` (anything else).
"""

import builtins
import json
import sys
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


def _reply(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, allow_nan=False) + "\n")


def _namespace(allowed_modules: frozenset[str]) -> dict[str, Any]:
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
    return {"__builtins__": safe, "__name__": "learned_skill", "SkillError": SkillError}


def main() -> None:
    request = json.loads(sys.stdin.read())
    namespace = _namespace(frozenset(request["allowed_modules"]))
    try:
        exec(compile(request["code"], "<learned skill>", "exec"), namespace)
        output = namespace["run"](request["arguments"])
        if not isinstance(output, dict):
            _reply({"crash": f"run() returned {type(output).__name__}, expected a dict"})
            return
        _reply({"output": output})
    except SkillError as exc:
        _reply({"error": str(exc) or "SkillError"})
    except RecursionError:
        _reply({"crash": "recursion too deep"})
    except MemoryError:
        _reply({"crash": "out of memory"})
    except Exception as exc:
        _reply({"crash": f"{type(exc).__name__}: {exc}"})


main()
