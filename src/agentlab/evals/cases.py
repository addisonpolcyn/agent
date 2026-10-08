"""Load evaluation cases from ``<cases_dir>/*.toml``."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any, cast

from agentlab.evals.models import EvalCase, Expectations

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.evals.models import CaseKind


class EvalCaseError(Exception):
    """An eval case file is malformed."""


_AGENT_CHECKS = frozenset(
    {
        "answer_contains",
        "answer_excludes_patterns",
        "skills_used",
        "no_skills_used",
        "capability_gap",
        "stop_reason",
    }
)
_SKILL_CHECKS = frozenset({"output_equals", "error_contains"})


def load_cases(cases_dir: Path) -> list[EvalCase]:
    if not cases_dir.is_dir():
        raise EvalCaseError(f"eval cases directory not found: {cases_dir}")
    cases = [load_case(path) for path in sorted(cases_dir.glob("*.toml"))]
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise EvalCaseError(f"duplicate eval case ids: {', '.join(duplicates)}")
    return cases


def load_case(path: Path) -> EvalCase:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise EvalCaseError(f"{path}: invalid TOML: {exc}") from exc

    kind = data.get("kind")
    if kind not in ("agent", "skill"):
        raise EvalCaseError(f"{path}: 'kind' must be 'agent' or 'skill'")
    expect = cast("dict[str, Any]", data.get("expect", {}))
    _check_expectation_keys(path, kind, expect)

    case = EvalCase(
        id=_require_str(data, "id", path),
        kind=kind,
        description=_require_str(data, "description", path),
        tags=tuple(cast("list[str]", data.get("tags", []))),
        task=data.get("task"),
        skill=data.get("skill"),
        arguments=cast("dict[str, Any]", data.get("arguments", {})),
        expect=_expectations(expect),
        path=path,
    )
    if kind == "agent" and not case.task:
        raise EvalCaseError(f"{path}: agent cases need a 'task'")
    if kind == "skill" and not case.skill:
        raise EvalCaseError(f"{path}: skill cases need a 'skill'")
    return case


def _check_expectation_keys(path: Path, kind: CaseKind, expect: dict[str, Any]) -> None:
    allowed = _AGENT_CHECKS if kind == "agent" else _SKILL_CHECKS
    if not expect:
        raise EvalCaseError(f"{path}: '[expect]' must declare at least one check")
    unknown = sorted(set(expect) - allowed)
    if unknown:
        raise EvalCaseError(f"{path}: unsupported checks for {kind} cases: {', '.join(unknown)}")


def _expectations(expect: dict[str, Any]) -> Expectations:
    return Expectations(
        answer_contains=tuple(expect.get("answer_contains", ())),
        answer_excludes_patterns=tuple(expect.get("answer_excludes_patterns", ())),
        skills_used=tuple(expect.get("skills_used", ())),
        no_skills_used=bool(expect.get("no_skills_used", False)),
        capability_gap=expect.get("capability_gap"),
        stop_reason=expect.get("stop_reason"),
        output_equals=expect.get("output_equals"),
        error_contains=expect.get("error_contains"),
    )


def _require_str(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalCaseError(f"{path}: '{key}' must be a non-empty string")
    return value
