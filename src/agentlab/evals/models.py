"""Evaluation data: what a case expects, and what happened when it ran."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.models import JSONObject

type CaseKind = Literal["agent", "skill"]
type Approval = Literal["approve", "deny_plan", "deny_skill"]
APPROVALS: tuple[Approval, ...] = ("approve", "deny_plan", "deny_skill")


@dataclass(frozen=True)
class Expectations:
    """Checks a case can declare. Unset fields are not checked.

    Agent cases: ``answer_contains`` (case-insensitive), ``answer_excludes_patterns`` (regex),
    ``skills_used`` (each must have been invoked), ``no_skills_used``, ``capability_gap``,
    ``stop_reason``, ``skills_learned`` (each must have been learned), ``learning_outcome``
    (one ``propose_skill_plan`` ended this way; ``"none"`` means learning was never tried).
    Skill cases: ``output_equals``, ``error_contains``.
    """

    answer_contains: tuple[str, ...] = ()
    answer_excludes_patterns: tuple[str, ...] = ()
    skills_used: tuple[str, ...] = ()
    no_skills_used: bool = False
    capability_gap: str | None = None
    stop_reason: str | None = None
    skills_learned: tuple[str, ...] = ()
    learning_outcome: str | None = None
    output_equals: JSONObject | None = None
    error_contains: str | None = None


@dataclass(frozen=True)
class EvalCase:
    id: str
    kind: CaseKind
    description: str
    expect: Expectations
    tags: tuple[str, ...] = ()
    task: str | None = None
    skill: str | None = None
    arguments: JSONObject = field(default_factory=dict[str, object])
    # Skill cases for file-reading skills: relative path -> text, created in a fresh folder
    # whose path replaces "{root}" in the arguments.
    files: dict[str, str] = field(default_factory=dict[str, str])
    approval: Approval = "deny_plan"
    # Turns sent first, in the same conversation, so the task can refer back to them.
    earlier_turns: tuple[str, ...] = ()
    # Needs a real model's judgment: skipped (never passed) by offline runs.
    model_only: bool = False
    path: Path | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    kind: CaseKind
    tags: tuple[str, ...]
    checks: tuple[CheckResult, ...]
    error: str | None = None
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        if self.skipped is not None or self.error is not None:
            return False
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        if self.error is not None or not self.checks:
            return 0.0
        return sum(c.passed for c in self.checks) / len(self.checks)

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_json(self) -> JSONObject:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "tags": list(self.tags),
            "passed": self.passed,
            "score": self.score,
            "error": self.error,
            "skipped": self.skipped,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


@dataclass(frozen=True)
class EvalSummary:
    results: tuple[EvalResult, ...]

    @property
    def scored(self) -> tuple[EvalResult, ...]:
        """Results that count: everything except skipped cases."""
        return tuple(r for r in self.results if r.skipped is None)

    @property
    def skipped_count(self) -> int:
        return len(self.results) - len(self.scored)

    @property
    def passed_count(self) -> int:
        return sum(r.passed for r in self.scored)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / len(self.scored) if self.scored else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.scored) / len(self.scored) if self.scored else 0.0

    @property
    def all_passed(self) -> bool:
        return self.passed_count == len(self.scored)

    def failures_by_mode(self) -> dict[str, list[tuple[str, str]]]:
        """Group failures by check name (the failure mode) -> [(case_id, detail)]."""
        modes: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for result in self.results:
            if result.error is not None:
                modes["error"].append((result.case_id, result.error))
            for check in result.failed_checks:
                modes[check.name].append((result.case_id, check.detail))
        return dict(modes)

    def to_json(self) -> JSONObject:
        return {
            "total": len(self.scored),
            "skipped": self.skipped_count,
            "passed": self.passed_count,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "results": [r.to_json() for r in self.results],
        }
