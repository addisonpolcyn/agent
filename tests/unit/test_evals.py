from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentlab.agent.loop import Agent, AgentRun, CapabilityGap, ToolInvocation
from agentlab.evals.cases import EvalCaseError, load_case, load_cases
from agentlab.evals.models import CheckResult, EvalCase, EvalResult, EvalSummary, Expectations
from agentlab.evals.runner import agent_checks, format_summary, run_case, skill_checks
from agentlab.llm.fake import ScriptedLLM
from agentlab.skills.models import SkillResult

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.skills.catalog import SkillCatalog


def agent_run(
    answer: str = "",
    skills: tuple[str, ...] = (),
    gaps: tuple[str, ...] = (),
) -> AgentRun:
    return AgentRun(
        task="t",
        answer=answer,
        stop_reason="answered",
        steps=1,
        invocations=tuple(ToolInvocation(s, {}, SkillResult(output={})) for s in skills),
        capability_gaps=tuple(CapabilityGap(g, "") for g in gaps),
    )


def names_failed(checks: list[CheckResult]) -> list[str]:
    return [c.name for c in checks if not c.passed]


def test_agent_checks_pass_on_matching_trace() -> None:
    expect = Expectations(
        answer_contains=("56088",),
        skills_used=("calculator",),
        stop_reason="answered",
    )
    checks = agent_checks(expect, agent_run("The result is 56088.", skills=("calculator",)))
    assert len(checks) == 3
    assert names_failed(checks) == []


def test_answer_contains_is_case_insensitive() -> None:
    checks = agent_checks(Expectations(answer_contains=("TOKYO",)), agent_run("tokyo"))
    assert names_failed(checks) == []


@pytest.mark.parametrize(
    ("needle", "answer"),
    [("56088", "123 \u00d7 456 = **56,088**"), ("1234567", "1,234,567"), ("56,088", "56088")],
)
def test_answer_contains_ignores_thousands_separators(needle: str, answer: str) -> None:
    checks = agent_checks(Expectations(answer_contains=(needle,)), agent_run(answer))
    assert names_failed(checks) == []


def test_answer_contains_keeps_other_commas() -> None:
    checks = agent_checks(Expectations(answer_contains=("12",)), agent_run("1,2"))
    assert names_failed(checks) == ["answer_contains"]


def test_agent_checks_report_each_failure_mode() -> None:
    expect = Expectations(
        answer_contains=("56088",),
        answer_excludes_patterns=(r"\$\d",),
        no_skills_used=True,
        capability_gap="current_information",
        stop_reason="max_steps",
    )
    checks = agent_checks(expect, agent_run("Flights from $450", skills=("calculator",)))
    assert names_failed(checks) == [
        "answer_contains",
        "answer_excludes_patterns",
        "no_skills_used",
        "capability_gap",
        "stop_reason",
    ]
    excluded = next(c for c in checks if c.name == "answer_excludes_patterns")
    assert "'$4'" in excluded.detail


def test_skill_checks() -> None:
    ok = skill_checks(Expectations(output_equals={"result": 1}), SkillResult(output={"result": 1}))
    bad = skill_checks(Expectations(error_contains="zero"), SkillResult(output={"result": 1}))
    assert names_failed(ok) == []
    assert names_failed(bad) == ["error_contains"]


def test_result_scoring() -> None:
    result = EvalResult("c", "agent", (), (CheckResult("a", True), CheckResult("b", False, "x")))
    assert not result.passed
    assert result.score == 0.5
    assert EvalResult("c", "agent", (), ()).score == 0.0
    errored = EvalResult("c", "agent", (), (CheckResult("a", True),), error="boom")
    assert not errored.passed
    assert errored.score == 0.0


def test_summary_groups_failures_by_mode() -> None:
    summary = EvalSummary(
        (
            EvalResult("one", "agent", (), (CheckResult("skills_used", False, "missing"),)),
            EvalResult("two", "agent", (), (CheckResult("skills_used", False, "missing"),)),
            EvalResult("three", "agent", (), (CheckResult("stop_reason", True),)),
            EvalResult("four", "agent", (), (), error="LLM error: down"),
        )
    )
    assert summary.passed_count == 1
    assert summary.pass_rate == 0.25
    assert summary.failures_by_mode() == {
        "skills_used": [("one", "missing"), ("two", "missing")],
        "error": [("four", "LLM error: down")],
    }
    assert "1/4 cases passed (25%)" in format_summary(summary)


def test_run_case_records_llm_errors(catalog: SkillCatalog) -> None:
    case = EvalCase("c", "agent", "d", Expectations(stop_reason="answered"), task="hi")
    result = run_case(case, Agent(ScriptedLLM([]), catalog), catalog)
    assert result.error is not None
    assert result.error.startswith("LLM error")


def test_loads_repository_cases(cases_dir: Path) -> None:
    cases = {case.id: case for case in load_cases(cases_dir)}
    assert {"ask_arithmetic", "calculator_component", "flight_sfo_tokyo"} <= set(cases)
    flight = cases["flight_sfo_tokyo"]
    assert flight.kind == "agent"
    assert "north-star" in flight.tags
    assert flight.expect.capability_gap == "current_information"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('id = "x"\nkind = "other"', "'kind' must be"),
        ('id = "x"\nkind = "agent"\ndescription = "d"\ntask = "t"', "at least one check"),
        (
            'id = "x"\nkind = "skill"\ndescription = "d"\nskill = "s"\n'
            "[expect]\nno_skills_used = true",
            "unsupported checks for skill cases: no_skills_used",
        ),
        (
            'id = "x"\nkind = "agent"\ndescription = "d"\n[expect]\nstop_reason = "answered"',
            "need a 'task'",
        ),
        (
            'kind = "agent"\ndescription = "d"\ntask = "t"\n[expect]\nstop_reason = "answered"',
            "'id'",
        ),
        ("id = ", "invalid TOML"),
    ],
)
def test_rejects_malformed_cases(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "case.toml"
    path.write_text(text)
    with pytest.raises(EvalCaseError, match=message):
        load_case(path)


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    case = (
        'id = "same"\nkind = "agent"\ndescription = "d"\ntask = "t"\n'
        '[expect]\nstop_reason = "answered"'
    )
    (tmp_path / "a.toml").write_text(case)
    (tmp_path / "b.toml").write_text(case)
    with pytest.raises(EvalCaseError, match="duplicate eval case ids: same"):
        load_cases(tmp_path)
