"""The runtime eval harness: build a tool on the fly and decide whether it is junk.

    spec -> contract + tests (blind to code) -> [code -> checks]{1..MAX_ATTEMPTS} -> verdict

A candidate is *ready* only if it passes every check:

- ``static``         the code is valid Python with a top-level ``run(arguments)``
- ``visible_tests``  the tests the code writer saw pass
- ``holdout_tests``  tests it never saw pass too (catches code fitted to the examples)
- ``no_crashes``     invalid input raises ``ToolError``; nothing else blows up
- ``output_schema``  every output matches the declared schema
- ``non_constant``   different inputs don't all produce the same output
- ``deterministic``  the same input gives the same output twice
- ``hardcoded``      test inputs don't appear as literals in the code

Tools that read files are tested against fixture files: each test gets a fresh folder with
its files, and "{root}" in its arguments names it.

The tests are fixed for every attempt, so the specification can't drift toward whatever the
code happens to do. Retries get the failed checks as feedback, but never the held-out inputs.
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, cast

from agentlab.evals.models import CheckResult
from agentlab.evals.runner import tool_checks
from agentlab.learning.author import AuthorError
from agentlab.learning.models import CandidateReport, HarnessReport, ToolContract
from agentlab.learning.sandbox import SandboxResult, check_source, run_sandboxed
from agentlab.tools.models import ToolResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agentlab.evals.models import EvalCase
    from agentlab.learning.author import ToolAuthor
    from agentlab.learning.models import NewToolSpec, ToolCandidate
    from agentlab.models import JSONObject


class Runner(Protocol):
    """How code gets executed: ``run_sandboxed``, or a fake in tests."""

    def __call__(self, code: str, arguments: JSONObject) -> SandboxResult: ...


MAX_ATTEMPTS = 3
MAX_CONTRACT_ATTEMPTS = 2
MIN_TESTS = 6
HOLDOUT_EVERY = 3
DETERMINISM_SAMPLES = 3
MIN_HARDCODED_LENGTH = 4
ROOT_PLACEHOLDER = "{root}"

# bool is a subclass of int in Python, but not a number in JSON Schema.
_TYPE_CHECKS: dict[str, Callable[[object], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def build_tool(
    spec: NewToolSpec,
    author: ToolAuthor,
    *,
    manifest_path: Path,
    run: Runner = run_sandboxed,
    max_attempts: int = MAX_ATTEMPTS,
) -> HarnessReport:
    """Write tests, then up to ``max_attempts`` implementations, and report the verdict."""
    contract, problem = _write_contract(spec, author)
    if contract is None:
        return HarnessReport(spec, "failed", f"could not write a usable test suite: {problem}")
    visible, holdout = split_tests(contract.tests)
    holdout_ids = frozenset(case.id for case in holdout)
    reports: list[CandidateReport] = []
    feedback: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            candidate = author.write_code(spec, contract, visible, feedback, manifest_path)
        except AuthorError as exc:
            report = CandidateReport(attempt, (CheckResult("author", False, str(exc)),))
        else:
            report = evaluate_candidate(candidate, contract, holdout_ids, run=run, attempt=attempt)
            if report.passed:
                summary = _pass_summary(contract, holdout_ids)
                return HarnessReport(
                    spec, "ready", summary, (*reports, report), contract, candidate
                )
        reports.append(report)
        if len(reports) >= 2 and reports[-2].failure_signature == report.failure_signature:
            return _failed(spec, reports, contract, "the same failures repeated, so it is stuck")
        feedback = [f"{check.name}: {check.detail}" for check in report.checks if not check.passed]
    return _failed(spec, reports, contract, f"still failing after {max_attempts} attempts")


def evaluate_candidate(
    candidate: ToolCandidate,
    contract: ToolContract,
    holdout_ids: frozenset[str],
    *,
    run: Runner = run_sandboxed,
    attempt: int = 1,
) -> CandidateReport:
    code = candidate.code
    problems = check_source(code)
    if problems:
        return CandidateReport(attempt, (CheckResult("static", False, "; ".join(problems)),))
    with tempfile.TemporaryDirectory(prefix="agentlab-fixtures-") as tmp:
        roots = {
            case.id: _make_fixtures(Path(tmp) / str(i), case.files)
            for i, case in enumerate(contract.tests)
        }

        def run_case(case: EvalCase) -> SandboxResult:
            return run(code, cast("JSONObject", _with_root(case.arguments, str(roots[case.id]))))

        results = {case.id: run_case(case) for case in contract.tests}
        checks = [
            CheckResult("static", True),
            _tests_check(contract.tests, results, holdout_ids),
            _holdout_check(contract.tests, results, holdout_ids),
            _crash_check(contract.tests, results, holdout_ids),
            _output_schema_check(results, contract.output_schema),
            _non_constant_check(contract.tests, results),
            _determinism_check(contract.tests, results, run_case),
            _hardcoded_check(code, contract.tests),
        ]
    return CandidateReport(attempt, tuple(checks))


def contract_problems(contract: ToolContract, *, reads_files: bool = False) -> list[str]:
    """Reasons a generated test suite is too weak or inconsistent to judge code with."""
    problems: list[str] = []
    if reads_files and not any(case.files for case in contract.tests):
        problems.append("a tool that reads files needs tests with fixture files")
    problems.extend(
        f"{case.id}: fixture path {rel!r} must be relative, without '..'"
        for case in contract.tests
        for rel in case.files
        if not _safe_relative(rel)
    )
    for key, schema in (("input", contract.input_schema), ("output", contract.output_schema)):
        if schema.get("type") != "object":
            problems.append(f"the {key} schema must have type 'object'")
    tests = contract.tests
    if len(tests) < MIN_TESTS:
        problems.append(f"needs at least {MIN_TESTS} tests, got {len(tests)}")
    if len({case.id for case in tests}) != len(tests):
        problems.append("test names must be unique")
    if len({_canonical([case.arguments, case.files]) for case in tests}) != len(tests):
        problems.append("tests must differ in their arguments or files")
    for kind in ("edge", "error"):
        if not any(kind in case.tags for case in tests):
            problems.append(f"needs at least one {kind!r} test")
    expected = [case for case in tests if case.expect.output_equals is not None]
    if len({_canonical(case.expect.output_equals) for case in expected}) < 2:
        problems.append("needs at least two tests with different expected outputs")
    for case in tests:
        if "error" in case.tags and case.expect.error_contains is None:
            problems.append(f"{case.id}: an 'error' test must expect an error")
        if case.expect.output_equals is None:
            continue
        problems.extend(
            f"{case.id}: arguments {error}"
            for error in schema_errors(case.arguments, contract.input_schema)
        )
        problems.extend(
            f"{case.id}: expected output {error}"
            for error in schema_errors(case.expect.output_equals, contract.output_schema)
        )
    return problems


def split_tests(tests: Sequence[EvalCase]) -> tuple[tuple[EvalCase, ...], tuple[EvalCase, ...]]:
    """Every third normal or edge test is held out from the code writer. Deterministic, so
    reruns compare.

    Error tests are always visible: they expect a message the code writer can't infer, so
    holding one out tests guessing, not generalization, and retries can't fix it.
    """
    answers = [c for c in tests if "error" not in c.tags]
    held = {c.id for i, c in enumerate(answers) if i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1}
    visible = tuple(c for c in tests if c.id not in held)
    holdout = tuple(c for c in tests if c.id in held)
    return visible, holdout


def schema_errors(value: object, schema: JSONObject, where: str = "$") -> list[str]:
    """A small JSON Schema subset: type, enum, required, properties, additionalProperties
    (false), items. Enough for tool I/O, without a dependency."""
    expected = schema.get("type")
    if expected is not None:
        names = cast("list[str]", expected) if isinstance(expected, list) else [str(expected)]
        if not any(_is_type(value, name) for name in names):
            return [f"at {where}: expected {'/'.join(names)}, got {_type_name(value)}"]
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in cast("list[object]", enum):
        return [f"at {where}: {value!r} is not one of {enum}"]
    if isinstance(value, dict):
        return _object_errors(cast("dict[str, Any]", value), schema, where)
    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        item_schema = cast("JSONObject", items)
        return [
            error
            for i, item in enumerate(cast("list[object]", value))
            for error in schema_errors(item, item_schema, f"{where}[{i}]")
        ]
    return []


def _write_contract(spec: NewToolSpec, author: ToolAuthor) -> tuple[ToolContract | None, str]:
    problem = ""
    feedback: list[str] = []
    for _ in range(MAX_CONTRACT_ATTEMPTS):
        try:
            contract = author.write_contract(spec, feedback)
        except AuthorError as exc:
            problem, feedback = str(exc), [str(exc)]
            continue
        problems = contract_problems(contract, reads_files=spec.reads_files)
        if not problems:
            return contract, ""
        problem, feedback = "; ".join(problems), problems
    return None, problem


def _as_tool_result(result: SandboxResult) -> ToolResult:
    if result.output is not None:
        return ToolResult(output=result.output)
    if result.error is not None:
        return ToolResult(error=result.error)
    return ToolResult(error=f"crashed: {result.crash}")


def _failing_tests(
    tests: Sequence[EvalCase], results: dict[str, SandboxResult]
) -> list[tuple[EvalCase, list[CheckResult]]]:
    failing: list[tuple[EvalCase, list[CheckResult]]] = []
    for case in tests:
        failed = [
            check
            for check in tool_checks(case.expect, _as_tool_result(results[case.id]))
            if not check.passed
        ]
        if failed:
            failing.append((case, failed))
    return failing


def _tests_check(
    tests: Sequence[EvalCase], results: dict[str, SandboxResult], holdout_ids: frozenset[str]
) -> CheckResult:
    visible = [case for case in tests if case.id not in holdout_ids]
    failing = _failing_tests(visible, results)
    detail = "; ".join(f"{case.id}: {checks[0].detail}" for case, checks in failing)
    return CheckResult("visible_tests", not failing, detail)


def _holdout_check(
    tests: Sequence[EvalCase], results: dict[str, SandboxResult], holdout_ids: frozenset[str]
) -> CheckResult:
    # Only counts leave the harness: the held-out inputs must stay unseen by the code writer.
    holdout = [case for case in tests if case.id in holdout_ids]
    failing = _failing_tests(holdout, results)
    visible_ok = not _failing_tests([c for c in tests if c.id not in holdout_ids], results)
    detail = f"{len(failing)} of {len(holdout)} held-out tests failed"
    if failing and visible_ok:
        detail += "; the visible tests pass, so the code may be fitted to the examples"
    return CheckResult("holdout_tests", not failing, detail if failing else "")


def _crash_check(
    tests: Sequence[EvalCase], results: dict[str, SandboxResult], holdout_ids: frozenset[str]
) -> CheckResult:
    crashes = [
        f"{case.id}: {results[case.id].crash}" if case.id not in holdout_ids else "a held-out test"
        for case in tests
        if results[case.id].crash is not None
    ]
    detail = "crashed instead of returning or raising ToolError: " + "; ".join(crashes)
    return CheckResult("no_crashes", not crashes, detail if crashes else "")


def _output_schema_check(results: dict[str, SandboxResult], schema: JSONObject) -> CheckResult:
    errors = [
        error
        for result in results.values()
        if result.output is not None
        for error in schema_errors(result.output, schema)
    ]
    return CheckResult("output_schema", not errors, "; ".join(sorted(set(errors))[:5]))


def _non_constant_check(
    tests: Sequence[EvalCase], results: dict[str, SandboxResult]
) -> CheckResult:
    expected = {
        _canonical(case.expect.output_equals)
        for case in tests
        if case.expect.output_equals is not None
    }
    actual = {_canonical(r.output) for r in results.values() if r.output is not None}
    constant = len(expected) > 1 and len(actual) == 1
    detail = f"every input produced the same output {next(iter(actual), '')}"
    return CheckResult("non_constant", not constant, detail if constant else "")


def _determinism_check(
    tests: Sequence[EvalCase],
    results: dict[str, SandboxResult],
    run_case: Callable[[EvalCase], SandboxResult],
) -> CheckResult:
    unstable = [
        case.id for case in tests[:DETERMINISM_SAMPLES] if run_case(case) != results[case.id]
    ]
    detail = f"different results on a second run: {', '.join(unstable)}"
    return CheckResult("deterministic", not unstable, detail if unstable else "")


def _make_fixtures(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    for rel, text in files.items():
        if not _safe_relative(rel):
            continue  # reported by contract_problems
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _safe_relative(rel: str) -> bool:
    path = PurePosixPath(rel)
    return bool(rel) and not path.is_absolute() and ".." not in path.parts and "\\" not in rel


def _with_root(value: object, root: str) -> object:
    if isinstance(value, str):
        return value.replace(ROOT_PLACEHOLDER, root)
    if isinstance(value, dict):
        return {k: _with_root(v, root) for k, v in cast("dict[str, object]", value).items()}
    if isinstance(value, list):
        return [_with_root(v, root) for v in cast("list[object]", value)]
    return value


def _hardcoded_check(code: str, tests: Sequence[EvalCase]) -> CheckResult:
    literals = {
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    inputs = {
        s
        for case in tests
        for s in _strings(case.arguments)
        if len(s.strip()) >= MIN_HARDCODED_LENGTH and ROOT_PLACEHOLDER not in s
    }
    found = sorted(literals & inputs)
    detail = "test inputs appear as literals in the code: " + ", ".join(
        repr(s[:30]) for s in found[:3]
    )
    return CheckResult("hardcoded", not found, detail if found else "")


def _pass_summary(contract: ToolContract, holdout_ids: frozenset[str]) -> str:
    total = len(contract.tests)
    return (
        f"{total}/{total} tests passed ({len(holdout_ids)} held out) · output schema ok · "
        "deterministic · non-constant · no hard-coded inputs"
    )


def _failed(
    spec: NewToolSpec, reports: list[CandidateReport], contract: ToolContract, why: str
) -> HarnessReport:
    last = reports[-1]
    failing = ", ".join(sorted(check.name for check in last.checks if not check.passed))
    return HarnessReport(spec, "failed", f"{why} (failing: {failing})", tuple(reports), contract)


def _object_errors(value: dict[str, Any], schema: JSONObject, where: str) -> list[str]:
    errors = [
        f"at {where}: missing required {key!r}"
        for key in cast("list[str]", schema.get("required", []))
        if key not in value
    ]
    properties = cast("dict[str, JSONObject]", schema.get("properties", {}))
    for key, item in value.items():
        if key in properties:
            errors.extend(schema_errors(item, properties[key], f"{where}.{key}"))
        elif schema.get("additionalProperties") is False:
            errors.append(f"at {where}: unexpected property {key!r}")
    return errors


def _is_type(value: object, name: str) -> bool:
    check = _TYPE_CHECKS.get(name)
    return check is not None and check(value)


def _type_name(value: object) -> str:
    for name in ("boolean", "integer", "number", "string", "object", "array", "null"):
        if _is_type(value, name):
            return name
    return type(value).__name__


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in cast("dict[str, object]", value).values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in cast("list[object]", value) for s in _strings(item)]
    return []


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True)
