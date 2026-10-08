"""Data for learning a skill: what to build, its contract, candidates and harness reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agentlab.evals.models import CheckResult, EvalCase
    from agentlab.models import JSONObject
    from agentlab.skills.models import SkillManifest

type Verdict = Literal["ready", "failed"]


@dataclass(frozen=True)
class SkillSpec:
    """A generic skill the agent proposed to build, in its own words."""

    name: str
    capability: str
    purpose: str
    inputs: str
    outputs: str
    reads_files: bool = False

    def to_json(self) -> JSONObject:
        return {
            "name": self.name,
            "capability": self.capability,
            "purpose": self.purpose,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "reads_files": self.reads_files,
        }


@dataclass(frozen=True)
class SkillContract:
    """Schemas and tests, written from the spec alone, before any code exists.

    Tests are skill eval cases (``output_equals`` or ``error_contains``) tagged ``normal``,
    ``edge`` or ``error``.
    """

    input_schema: JSONObject
    output_schema: JSONObject
    tests: tuple[EvalCase, ...]


@dataclass(frozen=True)
class SkillCandidate:
    """One attempt at an implementation."""

    manifest: SkillManifest
    code: str


@dataclass(frozen=True)
class CandidateReport:
    attempt: int
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failure_signature(self) -> frozenset[tuple[str, str]]:
        """What failed and how; two attempts with the same signature are stuck."""
        return frozenset((check.name, check.detail) for check in self.checks if not check.passed)

    def to_json(self) -> JSONObject:
        return {
            "attempt": self.attempt,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


@dataclass(frozen=True)
class HarnessReport:
    """The outcome of building one skill: ready to offer to the user, or why not."""

    spec: SkillSpec
    verdict: Verdict
    reason: str
    reports: tuple[CandidateReport, ...] = ()
    contract: SkillContract | None = None
    candidate: SkillCandidate | None = None

    @property
    def attempts(self) -> int:
        return len(self.reports)

    def summary(self) -> str:
        """One line for the user, e.g. at the approval gate."""
        if self.verdict == "failed":
            return f"{self.spec.name}: not ready after {self.attempts} attempt(s): {self.reason}"
        return f"{self.spec.name}: ready after {self.attempts} attempt(s): {self.reason}"

    def to_json(self) -> JSONObject:
        return {
            "spec": self.spec.to_json(),
            "verdict": self.verdict,
            "reason": self.reason,
            "attempts": [report.to_json() for report in self.reports],
            "tests": [] if self.contract is None else [_test_json(c) for c in self.contract.tests],
        }


def _test_json(case: EvalCase) -> JSONObject:
    return {
        "name": case.id,
        "kind": case.tags[0] if case.tags else "",
        "arguments": case.arguments,
        "files": case.files,
        "expect_output": case.expect.output_equals,
        "expect_error": case.expect.error_contains,
    }
