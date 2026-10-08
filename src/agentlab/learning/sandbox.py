"""Run generated skill code as untrusted input.

Defense in depth, not a perfect sandbox:

1. ``check_source`` rejects code outside a small AST allowlist: only pure-data stdlib
   imports, no file/process/reflection builtins, no ``_private`` or dunder access (except
   ``__init__``), no ``str.format`` (which can reach attributes through its template).
2. ``run_sandboxed`` executes it in a separate ``python -I -B`` process with an empty
   environment, an empty working directory, restricted builtins (see ``_runner.py``), a
   wall-clock timeout and CPU/memory/file/process rlimits (Linux/macOS).

Skills that read files get two read-only functions, never ``open``: ``list_dir`` and
``read_text``. They see only the ``readable_roots`` the user approved, and never secret-looking
files. Reading anywhere else stops the skill with ``needs_access`` so the caller can ask.

There is no OS-level network block; it rests on the import allowlist. Humans review the code
before first use (see ``learner.py``).
"""

from __future__ import annotations

import ast
import json
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentlab.models import JSONObject

MAX_SOURCE_BYTES = 8_000
MAX_OUTPUT_BYTES = 1_000_000
DEFAULT_TIMEOUT_SECONDS = 5.0
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
RUNNER = Path(__file__).with_name("_runner.py")

ALLOWED_MODULES = frozenset(
    {
        "collections",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "html",
        "html.parser",
        "itertools",
        "json",
        "math",
        "re",
        "statistics",
        "string",
        "textwrap",
        "unicodedata",
    }
)
BANNED_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "format",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "type",
        "vars",
    }
)
# ``"{0.__class__}".format(x)`` reaches attributes the AST check never sees.
BANNED_ATTRIBUTES = frozenset({"format", "format_map", "Formatter"})
ALLOWED_DUNDERS = frozenset({"__init__"})

MAX_DIR_ENTRIES = 500
MAX_READ_BYTES = 1_000_000
SECRET_NAMES = frozenset(
    {".aws", ".docker", ".git-credentials", ".gnupg", ".kube", ".netrc", ".pypirc", ".ssh"}
)
SECRET_PREFIXES = (".env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials")
SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx")


@dataclass(frozen=True)
class SandboxResult:
    """One execution. ``error`` is a ``SkillError`` the code raised on purpose; ``crash`` is
    anything else (an exception, a timeout, a limit, malformed output)."""

    output: JSONObject | None = None
    error: str | None = None
    crash: str | None = None
    # A folder the code tried to read outside its readable roots: ask the user, then rerun.
    needs_access: str | None = None


def check_source(code: str) -> list[str]:
    """Problems that make ``code`` unsafe or unusable; empty when it may run."""
    if len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
        return [f"code is larger than {MAX_SOURCE_BYTES} bytes"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    problems = [problem for node in ast.walk(tree) for problem in _node_problems(node)]
    if not any(_is_run_function(node) for node in tree.body):
        problems.append("missing a top-level 'def run(arguments)' with exactly one parameter")
    return problems


def run_sandboxed(
    code: str,
    arguments: JSONObject,
    *,
    reads_files: bool = False,
    readable_roots: Sequence[Path] = (),
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SandboxResult:
    """Execute ``run(arguments)`` from ``code`` in an isolated, resource-limited process.

    With ``reads_files``, the code gets ``list_dir`` and ``read_text`` limited to
    ``readable_roots``.
    """
    problems = check_source(code)
    if problems:
        return SandboxResult(crash="rejected by the sandbox: " + "; ".join(problems))
    file_access = (
        {
            "readable_roots": [str(root) for root in readable_roots],
            # Relative paths mean the user's working directory, not the sandbox's empty one.
            "base_dir": str(Path.cwd()),
            "secret_names": sorted(SECRET_NAMES),
            "secret_prefixes": list(SECRET_PREFIXES),
            "secret_suffixes": list(SECRET_SUFFIXES),
            "max_entries": MAX_DIR_ENTRIES,
            "max_read_bytes": MAX_READ_BYTES,
        }
        if reads_files
        else None
    )
    payload = json.dumps(
        {
            "code": code,
            "arguments": arguments,
            "allowed_modules": sorted(ALLOWED_MODULES),
            "file_access": file_access,
        }
    )
    # The wall-clock timeout normally fires first; the CPU limit is the backstop.
    cpu_seconds = int(timeout) + 1
    with tempfile.TemporaryDirectory(prefix="agentlab-sandbox-") as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(RUNNER)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env={},
                start_new_session=True,
                preexec_fn=lambda: _limit_resources(cpu_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(crash=f"timed out after {timeout:g}s")
    return _parse_reply(completed)


def _node_problems(node: ast.AST) -> list[str]:
    match node:
        case ast.Import(names=names):
            problems = [
                f"import of {a.name!r} is not allowed"
                for a in names
                if a.name not in ALLOWED_MODULES
            ]
        case ast.ImportFrom(module=module, level=level) if (
            level or module is None or module not in ALLOWED_MODULES
        ):
            problems = [f"import from {module!r} is not allowed"]
        case ast.Name(id=name) if name in BANNED_NAMES or _private(name):
            problems = [f"use of {name!r} is not allowed"]
        case ast.Attribute(attr=attr) if attr in BANNED_ATTRIBUTES or _private(attr):
            problems = [f"attribute {attr!r} is not allowed"]
        case ast.FunctionDef(name=name) | ast.ClassDef(name=name) if _private(name):
            problems = [f"definition of {name!r} is not allowed"]
        case ast.AsyncFunctionDef() | ast.Await() | ast.Global() | ast.Nonlocal():
            problems = [f"{type(node).__name__} is not allowed"]
        case _:
            problems = []
    return problems


def _private(name: str) -> bool:
    return name.startswith("_") and name not in ALLOWED_DUNDERS


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


def _limit_resources(cpu_seconds: int) -> None:
    """Runs in the child before exec, so the limits cover the whole interpreter."""
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _parse_reply(completed: subprocess.CompletedProcess[str]) -> SandboxResult:
    if completed.returncode < 0:
        return SandboxResult(crash=f"killed by {signal.Signals(-completed.returncode).name}")
    if len(completed.stdout) > MAX_OUTPUT_BYTES:
        return SandboxResult(crash=f"output larger than {MAX_OUTPUT_BYTES} bytes")
    try:
        reply = json.loads(completed.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError, IndexError:
        detail = completed.stderr.strip().splitlines()[-1:] or [f"exit {completed.returncode}"]
        return SandboxResult(crash=f"no result from the sandbox: {detail[0]}")
    return _from_reply(reply)


def _from_reply(reply: object) -> SandboxResult:
    match reply:
        case {"output": dict()}:
            return SandboxResult(output=cast("JSONObject", cast("JSONObject", reply)["output"]))
        case {"error": str() as error}:
            return SandboxResult(error=error)
        case {"crash": str() as crash}:
            return SandboxResult(crash=crash)
        case {"needs_access": str() as directory}:
            return SandboxResult(needs_access=directory)
        case _:
            return SandboxResult(crash="malformed reply from the sandbox")
