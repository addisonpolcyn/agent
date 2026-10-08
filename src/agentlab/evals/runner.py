"""Run eval cases and score them.

Agent cases run the full loop and are checked against the trace (answer, skills used,
capability gaps, stop reason). Skill cases call one skill directly: component evaluation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentlab.evals.models import CheckResult, EvalCase, EvalResult, EvalSummary
from agentlab.llm.client import LLMError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agentlab.agent.loop import Agent, AgentRun
    from agentlab.evals.models import Expectations
    from agentlab.skills.catalog import SkillCatalog
    from agentlab.skills.models import SkillResult


def run_suite(cases: Iterable[EvalCase], agent: Agent, catalog: SkillCatalog) -> EvalSummary:
    return EvalSummary(tuple(run_case(case, agent, catalog) for case in cases))


def run_case(case: EvalCase, agent: Agent, catalog: SkillCatalog) -> EvalResult:
    match case:
        case EvalCase(kind="skill", skill=str() as skill):
            checks = skill_checks(case.expect, catalog.execute(skill, case.arguments))
        case EvalCase(kind="agent", task=str() as task):
            try:
                checks = agent_checks(case.expect, agent.run(task))
            except LLMError as exc:
                return EvalResult(case.id, case.kind, case.tags, (), error=f"LLM error: {exc}")
        case _:
            raise ValueError(f"eval case {case.id!r} has no input for kind {case.kind!r}")
    return EvalResult(case.id, case.kind, case.tags, tuple(checks))


def agent_checks(expect: Expectations, run: AgentRun) -> list[CheckResult]:
    answer = run.answer or ""
    checks: list[CheckResult] = []
    for needle in expect.answer_contains:
        found = _comparable(needle) in _comparable(answer)
        checks.append(_check("answer_contains", found, f"{needle!r} not in answer {answer!r}"))
    for pattern in expect.answer_excludes_patterns:
        match = re.search(pattern, answer)
        detail = f"answer matched /{pattern}/: {match.group(0)!r}" if match else ""
        checks.append(_check("answer_excludes_patterns", match is None, detail))
    for skill in expect.skills_used:
        used = skill in run.skills_used
        checks.append(_check("skills_used", used, f"{skill!r} not in {list(run.skills_used)}"))
    if expect.no_skills_used:
        detail = f"skills were used: {list(run.skills_used)}"
        checks.append(_check("no_skills_used", not run.skills_used, detail))
    if expect.capability_gap is not None:
        reported = [gap.capability for gap in run.capability_gaps]
        detail = f"expected gap {expect.capability_gap!r}, reported {reported}"
        checks.append(_check("capability_gap", expect.capability_gap in reported, detail))
    if expect.stop_reason is not None:
        detail = f"expected {expect.stop_reason!r}, got {run.stop_reason!r}"
        checks.append(_check("stop_reason", run.stop_reason == expect.stop_reason, detail))
    return checks


def skill_checks(expect: Expectations, result: SkillResult) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if expect.output_equals is not None:
        detail = f"expected {expect.output_equals}, got {result.as_content()}"
        checks.append(_check("output_equals", result.output == expect.output_equals, detail))
    if expect.error_contains is not None:
        error = result.error or ""
        detail = f"expected error containing {expect.error_contains!r}, got {result.error!r}"
        checks.append(_check("error_contains", expect.error_contains in error, detail))
    return checks


def format_summary(summary: EvalSummary) -> str:
    lines: list[str] = []
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"{status}  {result.case_id}  ({result.kind}, score {result.score:.2f})")
        if result.error is not None:
            lines.append(f"      error: {result.error}")
        lines.extend(f"      {c.name}: {c.detail}" for c in result.failed_checks)
    lines.append(
        f"\n{summary.passed_count}/{len(summary.results)} cases passed "
        f"({summary.pass_rate:.0%}), mean score {summary.mean_score:.2f}"
    )
    return "\n".join(lines)


_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _comparable(text: str) -> str:
    """Case-fold and drop thousands separators: '56,088' and '56088' state the same number."""
    return _THOUSANDS_SEPARATOR.sub("", text).casefold()


def _check(name: str, passed: bool, failure_detail: str) -> CheckResult:
    return CheckResult(name, passed, "" if passed else failure_detail)
